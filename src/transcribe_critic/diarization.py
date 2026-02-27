"""
Diarization module for the speech transcription pipeline.

Handles speaker diarization using pyannote.audio, assigning speaker labels
to transcript segments based on word-level timestamps from Whisper.
"""

import json
import os
from pathlib import Path

import numpy as np

from transcribe_critic.shared import (
    tprint as print,
    SpeechConfig, SpeechData,
    DIARIZATION_JSON, DIARIZED_TXT,
    create_llm_client, llm_call_with_retry,
    is_up_to_date, _save_json, _should_skip,
)


def diarize_audio(config: SpeechConfig, data: SpeechData) -> None:
    """Run speaker diarization on the audio and produce a labeled transcript.

    Stage 2b in the pipeline: runs after transcription, before merge.
    """
    if not config.diarize:
        return

    print()
    print("[diarize] Diarizing audio...")

    diarization_json = config.output_dir / DIARIZATION_JSON
    diarized_txt = config.output_dir / DIARIZED_TXT

    if _should_skip(config, diarized_txt, "diarize audio", data.audio_path):
        if diarized_txt.exists():
            data.diarization_path = diarized_txt
        return

    if not data.audio_path or not data.audio_path.exists():
        print("  Warning: No audio file available for diarization")
        return

    if not data.transcript_segments:
        print("  Warning: No transcript segments available for diarization")
        return

    # Check pyannote availability
    try:
        from pyannote.audio import Pipeline  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "pyannote.audio is required for diarization. Install it with:\n"
            "  pip install pyannote.audio\n"
            "You also need a HuggingFace token (HF_TOKEN) with access to "
            "pyannote/speaker-diarization-3.1"
        )

    # Run pyannote diarization
    speaker_segments = _run_pyannote(config, data, diarization_json)

    # Assign speakers to transcript words/segments
    _assign_speakers_to_words(data, speaker_segments)

    # Identify speaker names
    _identify_speakers(config, data)

    # Format and save diarized transcript
    diarized_text = _format_diarized_transcript(data)
    with open(diarized_txt, 'w') as f:
        f.write(diarized_text)

    data.diarization_path = diarized_txt
    print(f"  Diarized transcript saved: {diarized_txt.name}")


def _make_progress_hook():
    """Return a pyannote hook callback that prints step progress."""
    last_step = [None]
    last_pct = [None]

    def hook(step_name, step_artifact, **kwargs):
        completed = kwargs.get("completed", None)
        total = kwargs.get("total", None)
        if completed is not None and total and total > 0:
            pct = int(100 * completed / total)
            pct_bucket = pct // 10 * 10  # round down to nearest 10%
            if completed >= total:
                print(f"  {step_name}: done")
                last_pct[0] = None
            elif pct_bucket != last_pct[0]:
                print(f"  {step_name}: {pct}% ({completed}/{total})")
                last_pct[0] = pct_bucket
        elif step_name != last_step[0]:
            last_step[0] = step_name

    return hook


def _run_pyannote(config: SpeechConfig, data: SpeechData,
                  output_path: Path) -> list:
    """Run pyannote speaker diarization and return speaker segments.

    Runs pyannote's pipeline as individual steps with checkpointing.
    The two expensive steps (segmentation and embeddings) are cached as
    .npy files in the output directory. On resume, cached steps are skipped.

    Returns a list of {start, end, speaker} dicts.
    """
    # Final result cache — if this exists, everything is done
    if output_path.exists():
        print("  Loading cached diarization segments...")
        with open(output_path, 'r') as f:
            return json.load(f)

    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        # Fall back to cached token from huggingface-cli login
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            hf_token = token_path.read_text().strip()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable required for pyannote. "
            "Get a token at https://huggingface.co/settings/tokens "
            "and accept the pyannote/speaker-diarization-3.1 model terms."
        )

    print("  Loading pyannote pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # Use GPU if available (MPS on Apple Silicon, CUDA on NVIDIA)
    import torch
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("  Using CUDA GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        print("  Using Apple Metal GPU")
    else:
        print("  Using CPU")

    diarization = _run_pyannote_steps(config, data, pipeline)

    # Convert to list of dicts
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    # Save final result
    _save_json(output_path, speaker_segments)
    print(f"  Found {len(speaker_segments)} speaker segments")

    return speaker_segments


def _get_embeddings_checkpointed(pipeline, file, binary_segmentations,
                                  output_dir, checkpoint_every=10):
    """Extract embeddings with periodic checkpointing.

    Replicates pyannote's get_embeddings loop but saves partial results
    every `checkpoint_every` batches. On resume, completed batches are
    skipped by loading from the partial checkpoint.

    Returns embeddings array of shape (num_chunks, num_speakers, dimension).
    """
    import itertools
    import math
    import torch
    from einops import rearrange

    def _batchify(iterable, batch_size=32, fillvalue=None):
        """Group iterable into fixed-size batches (mirrors pyannote's batchify)."""
        args = [iter(iterable)] * batch_size
        return itertools.zip_longest(*args, fillvalue=fillvalue)

    partial_npy = output_dir / "diarization_embeddings_partial.npy"
    partial_meta = output_dir / "diarization_embeddings_partial.json"

    exclude_overlap = pipeline.embedding_exclude_overlap
    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

    # Prepare clean segmentations (same logic as pyannote's get_embeddings)
    if exclude_overlap:
        min_num_samples = pipeline._embedding.min_num_samples
        num_samples = duration * pipeline._embedding.sample_rate
        min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

        clean_frames = 1.0 * (
            np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
        )
        from pyannote.core import SlidingWindowFeature
        clean_segmentations = SlidingWindowFeature(
            binary_segmentations.data * clean_frames,
            binary_segmentations.sliding_window,
        )
    else:
        min_num_frames = -1
        from pyannote.core import SlidingWindowFeature
        clean_segmentations = SlidingWindowFeature(
            binary_segmentations.data, binary_segmentations.sliding_window
        )

    def iter_waveform_and_mask():
        for (chunk, masks), (_, clean_masks) in zip(
            binary_segmentations, clean_segmentations
        ):
            waveform, _ = pipeline._audio.crop(file, chunk, mode="pad")
            masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
            clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)
            for mask, clean_mask in zip(masks.T, clean_masks.T):
                if np.sum(clean_mask) > min_num_frames:
                    used_mask = clean_mask
                else:
                    used_mask = mask
                yield waveform[None], torch.from_numpy(used_mask)[None]

    batch_size = pipeline.embedding_batch_size
    batches = _batchify(
        iter_waveform_and_mask(),
        batch_size=batch_size,
        fillvalue=(None, None),
    )
    batch_count = math.ceil(num_chunks * num_speakers / batch_size)

    # Check for partial checkpoint
    completed_batches = 0
    embedding_batches = []
    if partial_npy.exists() and partial_meta.exists():
        with open(partial_meta, 'r') as f:
            meta = json.load(f)
        completed_batches = meta.get("completed_batches", 0)
        if completed_batches > 0:
            partial_data = np.load(partial_npy)
            embedding_batches = list(partial_data)
            print(f"  Resuming embeddings from batch {completed_batches}/{batch_count}")

    # Skip already-completed batches (advances the generator past audio crops
    # for those batches — cheap compared to embedding model inference)
    for _ in range(completed_batches):
        next(batches, None)

    print(f"  Embeddings: {batch_count} batches, checkpointing every {checkpoint_every}")
    last_pct_bucket = None

    for i, batch in enumerate(batches, completed_batches + 1):
        waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))
        waveform_batch = torch.vstack(waveforms)
        mask_batch = torch.vstack(masks)

        embedding_batch = pipeline._embedding(waveform_batch, masks=mask_batch)
        embedding_batches.append(embedding_batch)

        pct = int(100 * i / batch_count)
        pct_bucket = pct // 10 * 10
        if i >= batch_count:
            print(f"  embeddings: done")
        elif pct_bucket != last_pct_bucket:
            print(f"  embeddings: {pct}% ({i}/{batch_count})")
            last_pct_bucket = pct_bucket

        # Periodic checkpoint
        if i % checkpoint_every == 0 and i < batch_count:
            stacked = np.vstack(embedding_batches)
            np.save(partial_npy, stacked)
            _save_json(partial_meta, {"completed_batches": i, "total_batches": batch_count})

    all_embeddings = np.vstack(embedding_batches)
    embeddings = rearrange(all_embeddings, "(c s) d -> c s d", c=num_chunks)
    return embeddings


def _run_pyannote_steps(config, data, pipeline):
    """Run pyannote diarization as individual steps with checkpointing.

    Each expensive step caches its output as a .npy file. On resume,
    cached steps are skipped if the checkpoint is newer than the audio
    (DAG-style staleness check).

    Returns a pyannote Annotation with speaker labels.
    """
    from pyannote.core import SlidingWindow, SlidingWindowFeature
    from pyannote.audio.utils.signal import binarize

    # Convert MP3 to WAV for pyannote (MP3 decoding has sample-count rounding issues)
    # DAG: MP3 → WAV → segmentation → embeddings → clustering → reconstruction
    audio_path = data.audio_path
    wav_path = audio_path.with_suffix(".wav")
    if audio_path.suffix.lower() == ".mp3":
        if is_up_to_date(wav_path, audio_path):
            print("  Reusing: audio.wav")
        else:
            print("  Converting MP3 to WAV for diarization (avoids sample-count rounding)...")
            import subprocess
            subprocess.run(
                ["ffmpeg", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(wav_path)],
                check=True, capture_output=True,
            )
        audio_path = wav_path
    seg_npy = config.output_dir / "diarization_segmentation.npy"
    seg_meta = config.output_dir / "diarization_segmentation_meta.json"
    emb_npy = config.output_dir / "diarization_embeddings.npy"

    file = {"uri": "audio", "audio": str(audio_path)}
    hook = _make_progress_hook()

    # --- Step 1: Segmentation (expensive, ~50% of runtime) ---
    if is_up_to_date(seg_npy, audio_path) and seg_meta.exists():
        print("  Step 1/6: Reusing cached segmentation")
        seg_data = np.load(seg_npy)
        with open(seg_meta, 'r') as f:
            meta = json.load(f)
        sw = SlidingWindow(
            start=meta["start"], duration=meta["duration"], step=meta["step"])
        segmentations = SlidingWindowFeature(seg_data, sw)
    else:
        print("  Step 1/6: Segmentation (neural model, this is slow)...")
        segmentations = pipeline.get_segmentations(file, hook=hook)
        # Cache for resumption
        np.save(seg_npy, segmentations.data)
        sw = segmentations.sliding_window
        _save_json(seg_meta, {
            "start": sw.start, "duration": sw.duration, "step": sw.step,
        })
        print(f"  Segmentation cached ({seg_npy.name})")

    # --- Step 2: Binarization (cheap) ---
    print("  Step 2/6: Binarization...")
    if pipeline._segmentation.model.specifications.powerset:
        binarized = segmentations
    else:
        binarized = binarize(
            segmentations,
            onset=pipeline.segmentation.threshold,
            initial_state=False,
        )

    # --- Step 3: Speaker counting (cheap) ---
    print("  Step 3/6: Speaker counting...")
    count = pipeline.speaker_count(
        binarized,
        pipeline._segmentation.model.receptive_field,
        warm_up=(0.0, 0.0),
    )

    # Early exit if no speakers detected
    if np.nanmax(count.data) == 0.0:
        from pyannote.core import Annotation
        print("  No speakers detected in audio")
        return Annotation(uri=file["uri"])

    # --- Step 4: Embeddings (expensive, ~40% of runtime) ---
    if is_up_to_date(emb_npy, seg_npy) and is_up_to_date(emb_npy, audio_path):
        print("  Step 4/6: Reusing cached embeddings")
        embeddings = np.load(emb_npy)
    else:
        print("  Step 4/6: Extracting embeddings...")
        embeddings = _get_embeddings_checkpointed(
            pipeline, file, binarized, config.output_dir,
            checkpoint_every=10,
        )
        np.save(emb_npy, embeddings)
        # Clean up partial checkpoint now that final is saved
        partial_npy = config.output_dir / "diarization_embeddings_partial.npy"
        partial_meta = config.output_dir / "diarization_embeddings_partial.json"
        for p in (partial_npy, partial_meta):
            if p.exists():
                p.unlink()
        print(f"  Embeddings cached ({emb_npy.name})")

    # --- Step 5: Clustering (cheap) ---
    print("  Step 5/6: Clustering...")
    num_speakers = config.num_speakers
    min_speakers = None
    max_speakers = None
    if num_speakers:
        min_speakers = num_speakers
        max_speakers = num_speakers

    hard_clusters, _, centroids = pipeline.clustering(
        embeddings=embeddings,
        segmentations=binarized,
        num_clusters=num_speakers,
        min_clusters=min_speakers,
        max_clusters=max_speakers,
        file=file,
        frames=pipeline._segmentation.model.receptive_field,
    )

    # Cap instantaneous speaker count
    max_spk = np.max(hard_clusters) + 1
    if max_speakers:
        max_spk = max(max_spk, max_speakers)
    count.data = np.minimum(count.data, max_spk).astype(np.int8)

    # --- Step 6: Reconstruction (cheap) ---
    print("  Step 6/6: Reconstruction...")

    # Mark inactive speakers
    inactive = np.sum(binarized.data, axis=1) == 0
    hard_clusters[inactive] = -2

    # Build main diarization (with overlap)
    discrete_diarization = pipeline.reconstruct(
        segmentations, hard_clusters, count)
    diarization = pipeline.to_annotation(
        discrete_diarization,
        min_duration_on=0.0,
        min_duration_off=pipeline.segmentation.min_duration_off,
    )
    diarization.uri = file["uri"]

    # Rename integer labels to SPEAKER_00, SPEAKER_01, ...
    mapping = {
        label: expected
        for label, expected in zip(diarization.labels(), pipeline.classes())
    }
    diarization = diarization.rename_labels(mapping=mapping)

    return diarization


def _assign_speakers_to_words(data: SpeechData, speaker_segments: list) -> None:
    """Assign speaker labels to transcript segments based on word timestamps.

    Uses word midpoints to find the overlapping speaker segment.
    Then assigns segment-level speaker by majority vote of its words.
    """
    if not speaker_segments:
        return

    for seg in data.transcript_segments:
        if seg.get("words"):
            # Word-level assignment: find speaker for each word's midpoint
            word_speakers = []
            for word in seg["words"]:
                mid = (word["start"] + word["end"]) / 2
                speaker = _find_speaker_at_time(mid, speaker_segments)
                word["speaker"] = speaker
                if speaker:
                    word_speakers.append(speaker)

            # Segment speaker = majority of word speakers
            if word_speakers:
                seg["speaker"] = max(set(word_speakers),
                                     key=word_speakers.count)
        else:
            # Fallback: use segment midpoint
            mid = (seg["start"] + seg["end"]) / 2
            seg["speaker"] = _find_speaker_at_time(mid, speaker_segments)


def _find_speaker_at_time(time_point: float, speaker_segments: list) -> str:
    """Find which speaker is active at a given time point."""
    for seg in speaker_segments:
        if seg["start"] <= time_point <= seg["end"]:
            return seg["speaker"]
    # If no exact match, find nearest segment
    if speaker_segments:
        nearest = min(speaker_segments,
                      key=lambda s: min(abs(s["start"] - time_point),
                                        abs(s["end"] - time_point)))
        return nearest["speaker"]
    return "UNKNOWN"


def _identify_speakers(config: SpeechConfig, data: SpeechData) -> None:
    """Map generic speaker labels (SPEAKER_00) to real names.

    Uses manual --speaker-names if provided, otherwise asks the LLM
    to identify speakers from introductions in the transcript.
    """
    # Collect unique speakers in order of first appearance
    seen = []
    for seg in data.transcript_segments:
        speaker = seg.get("speaker")
        if speaker and speaker not in seen:
            seen.append(speaker)

    if not seen:
        return

    if config.speaker_names:
        # Manual mapping: names ordered by first appearance
        name_map = {}
        for i, speaker_id in enumerate(seen):
            if i < len(config.speaker_names):
                name_map[speaker_id] = config.speaker_names[i]
        _apply_speaker_names(data, name_map)
        print(f"  Speaker mapping (manual): {name_map}")
        return

    if config.no_llm:
        print(f"  Speakers (no LLM): {', '.join(seen)}")
        return

    # Use LLM to identify speakers from the transcript intro
    intro_text = _get_intro_text(data)
    if not intro_text:
        print(f"  Could not extract intro for speaker identification")
        return

    name_map = _llm_identify_speakers(config, seen, intro_text, data.metadata)
    if name_map:
        _apply_speaker_names(data, name_map)
        print(f"  Speaker mapping (LLM): {name_map}")
    else:
        print(f"  Could not identify speakers, keeping generic labels")


def _get_intro_text(data: SpeechData) -> str:
    """Get the first ~500 words of diarized transcript for speaker identification."""
    lines = []
    word_count = 0
    for seg in data.transcript_segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")
        lines.append(f"{speaker}: {text}")
        word_count += len(text.split())
        if word_count >= 500:
            break
    return "\n".join(lines)


def _llm_identify_speakers(config: SpeechConfig, speakers: list,
                            intro_text: str, metadata: dict = None) -> dict:
    """Use LLM to identify speaker names from transcript introductions."""
    speaker_list = ", ".join(speakers)

    metadata_section = ""
    if metadata:
        parts = []
        if metadata.get("title"):
            parts.append(f"Title: {metadata['title']}")
        if metadata.get("description"):
            parts.append(f"Description: {metadata['description']}")
        if metadata.get("channel"):
            parts.append(f"Channel: {metadata['channel']}")
        if parts:
            metadata_section = (
                "\n\nMETADATA (from the video source — use for correct name spellings):\n"
                + "\n".join(parts)
            )

    prompt = f"""Below is the beginning of a transcript with speaker labels.
Identify who each speaker is based on introductions, context clues, or how they refer to each other.
{metadata_section}

SPEAKERS: {speaker_list}

TRANSCRIPT:
{intro_text}

For each speaker label, provide their real name. Use the metadata above for correct spellings when available. If you cannot determine a speaker's name, keep the original label.
Reply ONLY with a JSON object mapping speaker labels to names, like:
{{"SPEAKER_00": "John Smith", "SPEAKER_01": "Jane Doe"}}"""

    client = create_llm_client(config)

    try:
        message = llm_call_with_retry(
            client, config,
            model=config.claude_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text.strip()

        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"  Warning: LLM speaker identification failed: {e}")

    return {}


def _apply_speaker_names(data: SpeechData, name_map: dict) -> None:
    """Apply speaker name mapping to all transcript segments."""
    for seg in data.transcript_segments:
        speaker = seg.get("speaker")
        if speaker and speaker in name_map:
            seg["speaker"] = name_map[speaker]
        # Also update word-level speakers if present
        for word in seg.get("words", []):
            ws = word.get("speaker")
            if ws and ws in name_map:
                word["speaker"] = name_map[ws]


def _format_diarized_transcript(data: SpeechData) -> str:
    """Format diarized transcript in bracketed format.

    Output: [H:MM:SS] Speaker Name: text
    This format is auto-detected by merge.py's BRACKETED_PATTERN.
    """
    lines = []
    current_speaker = None
    current_text = []
    current_start = 0

    for seg in data.transcript_segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if not text:
            continue

        if speaker != current_speaker:
            # Flush previous speaker's text
            if current_speaker is not None and current_text:
                timestamp = _format_timestamp(current_start)
                lines.append(f"[{timestamp}] {current_speaker}: {' '.join(current_text)}")
            current_speaker = speaker
            current_text = [text]
            current_start = seg.get("start", 0)
        else:
            current_text.append(text)

    # Flush last speaker
    if current_speaker is not None and current_text:
        timestamp = _format_timestamp(current_start)
        lines.append(f"[{timestamp}] {current_speaker}: {' '.join(current_text)}")

    return "\n\n".join(lines)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"
