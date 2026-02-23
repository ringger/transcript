"""
Diarization module for the speech transcription pipeline.

Handles speaker diarization using pyannote.audio, assigning speaker labels
to transcript segments based on word-level timestamps from Whisper.
"""

import functools
import json
import os
from pathlib import Path

print = functools.partial(print, flush=True)

from shared import (
    SpeechConfig, SpeechData,
    create_llm_client, llm_call_with_retry,
    _save_json, _should_skip,
)


def diarize_audio(config: SpeechConfig, data: SpeechData) -> None:
    """Run speaker diarization on the audio and produce a labeled transcript.

    Stage 2b in the pipeline: runs after transcription, before merge.
    """
    if not config.diarize:
        return

    print("\n[2b] Diarizing audio...")

    diarization_json = config.output_dir / "diarization.json"
    diarized_txt = config.output_dir / "diarized.txt"

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


def _run_pyannote(config: SpeechConfig, data: SpeechData,
                  output_path: Path) -> list:
    """Run pyannote speaker diarization and return speaker segments.

    Returns a list of {start, end, speaker} dicts.
    """
    # Check for cached diarization
    if output_path.exists():
        print("  Loading cached diarization segments...")
        with open(output_path, 'r') as f:
            return json.load(f)

    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable required for pyannote. "
            "Get a token at https://huggingface.co/settings/tokens "
            "and accept the pyannote/speaker-diarization-3.1 model terms."
        )

    print("  Loading pyannote pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    print("  Running speaker diarization...")
    kwargs = {}
    if config.num_speakers:
        kwargs["num_speakers"] = config.num_speakers
    diarization = pipeline(str(data.audio_path), **kwargs)

    # Convert to list of dicts
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    # Save for caching
    _save_json(output_path, speaker_segments)
    print(f"  Found {len(speaker_segments)} speaker segments")

    return speaker_segments


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

    name_map = _llm_identify_speakers(config, seen, intro_text)
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
                            intro_text: str) -> dict:
    """Use LLM to identify speaker names from transcript introductions."""
    speaker_list = ", ".join(speakers)

    prompt = f"""Below is the beginning of a transcript with speaker labels.
Identify who each speaker is based on introductions, context clues, or how they refer to each other.

SPEAKERS: {speaker_list}

TRANSCRIPT:
{intro_text}

For each speaker label, provide their real name. If you cannot determine a speaker's name, keep the original label.
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
