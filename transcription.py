"""
Transcription module for the speech transcription pipeline.

Handles speech-to-text transcription with Whisper models, including
multi-model ensembling via wdiff analysis and LLM adjudication.
"""

import functools
import json
from pathlib import Path

print = functools.partial(print, flush=True)

from shared import (
    SpeechConfig, SpeechData, is_up_to_date,
    create_llm_client, llm_call_with_retry,
    run_command, _save_json, _print_reusing, _dry_run_skip,
    check_dependencies, MODEL_SIZES,
)
from merge import _analyze_differences_wdiff, _filter_meaningful_diffs


def transcribe_audio(config: SpeechConfig, data: SpeechData) -> None:
    """Transcribe audio using Whisper, supporting multiple models for ensembling."""
    print("\n[2/5] Transcribing audio...")

    if not data.audio_path or not data.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {data.audio_path}")

    deps = check_dependencies()
    if not deps["mlx_whisper"] and not deps["whisper"]:
        raise RuntimeError("No Whisper implementation found. Install mlx-whisper or openai-whisper.")

    models = config.whisper_models
    print(f"  Models to run: {', '.join(models)}")

    # Run each model
    for model in models:
        _run_whisper_model(config, data, model, deps)

    # If multiple models, ensemble them
    if len(models) > 1:
        _ensemble_whisper_transcripts(config, data)
    else:
        # Single model - use it directly
        model = models[0]
        if model in data.whisper_transcripts:
            data.transcript_path = data.whisper_transcripts[model]["txt"]
            data.transcript_json_path = data.whisper_transcripts[model].get("json")

    if not data.transcript_path:
        raise FileNotFoundError("Transcript file not found after transcription")

    # Load segments from JSON (use largest model's JSON for timestamps)
    _load_transcript_segments(data)


def _run_whisper_model(config: SpeechConfig, data: SpeechData, model: str, deps: dict) -> None:
    """Run a single Whisper model and save output."""
    # Create model-specific output names
    txt_path = config.output_dir / f"{model}.txt"
    json_path = config.output_dir / f"{model}.json"

    # Check if up to date (output newer than audio input)
    if config.skip_existing and is_up_to_date(txt_path, data.audio_path):
        _print_reusing(txt_path.name)
        data.whisper_transcripts[model] = {"txt": txt_path, "json": json_path if json_path.exists() else None}
        return
    if _dry_run_skip(config, f"transcribe with Whisper {model}", f"{model}.txt"):
        return
    print(f"  Running Whisper {model}...")

    if deps["mlx_whisper"]:
        model_name = f"mlx-community/whisper-{model}-mlx"

        # mlx_whisper outputs to input filename, so we need to work around this
        # Run transcription
        for fmt in ["txt", "json"]:
            run_command(
                ["mlx_whisper", str(data.audio_path),
                 "--model", model_name,
                 "--output-format", fmt,
                 "--output-dir", str(config.output_dir)],
                f"transcribing with mlx-whisper {model} ({fmt})",
                config.verbose
            )

        # mlx_whisper names output after input file, so rename to model name
        default_txt = config.output_dir / f"{data.audio_path.stem}.txt"
        default_json = config.output_dir / f"{data.audio_path.stem}.json"

        if default_txt.exists() and default_txt != txt_path:
            # If this is not the first model, rename to avoid overwriting
            if txt_path.exists():
                txt_path.unlink()
            default_txt.rename(txt_path)

        if default_json.exists() and default_json != json_path:
            if json_path.exists():
                json_path.unlink()
            default_json.rename(json_path)

        # Handle case where file stayed as default name
        if not txt_path.exists() and default_txt.exists():
            txt_path = default_txt
        if not json_path.exists() and default_json.exists():
            json_path = default_json

    elif deps["whisper"]:
        print(f"  (Using openai-whisper - may be slow on CPU)")
        import whisper
        whisper_model = whisper.load_model(model)
        result = whisper_model.transcribe(str(data.audio_path))

        with open(txt_path, 'w') as f:
            f.write(result["text"])

        _save_json(json_path, {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language", "en")
            })

    data.whisper_transcripts[model] = {
        "txt": txt_path if txt_path.exists() else None,
        "json": json_path if json_path.exists() else None
    }
    print(f"  {model} transcript saved: {txt_path.name}")


def _ensemble_whisper_transcripts(config: SpeechConfig, data: SpeechData) -> None:
    """Ensemble multiple Whisper transcripts using wdiff."""
    # Check for existing ensembled file
    if data.audio_path:
        ensembled_path = config.output_dir / "ensembled.txt"
        # Inputs are the individual whisper transcripts
        whisper_inputs = [paths["txt"] for paths in data.whisper_transcripts.values()
                         if paths.get("txt")]
        if config.skip_existing and is_up_to_date(ensembled_path, *whisper_inputs):
            _print_reusing(ensembled_path.name)
            data.transcript_path = ensembled_path
            # Use the largest model's JSON for timestamps
            for size in MODEL_SIZES:
                if size in data.whisper_transcripts:
                    data.transcript_json_path = data.whisper_transcripts[size].get("json")
                    break
            return
        if _dry_run_skip(config, "ensemble Whisper transcripts", "ensembled.txt"):
            return
    print(f"  Ensembling {len(data.whisper_transcripts)} Whisper transcripts...")

    models = list(data.whisper_transcripts.keys())
    if len(models) < 2:
        return

    # Read all transcripts
    transcripts = {}
    for model, paths in data.whisper_transcripts.items():
        if paths["txt"] and paths["txt"].exists():
            with open(paths["txt"], 'r') as f:
                transcripts[model] = f.read()

    if len(transcripts) < 2:
        print("  Not enough transcripts to ensemble")
        return

    # Start with the largest model as base (usually most accurate)
    base_model = None
    for size in MODEL_SIZES:
        if size in transcripts:
            base_model = size
            break

    if not base_model:
        base_model = models[0]

    base_text = transcripts[base_model]
    print(f"  Using {base_model} as base transcript")

    # Compare with other models using wdiff
    other_models = [m for m in transcripts.keys() if m != base_model]

    # Analyze differences between models
    all_differences = []
    for other_model in other_models:
        other_text = transcripts[other_model]
        diffs = _analyze_differences_wdiff(other_text, base_text, config,
                                            label_a=other_model, label_b=base_model)
        if diffs:
            print(f"  Found {len(diffs)} differences between {other_model} and {base_model}")
            all_differences.extend(diffs)

    # If we have differences and API key, use Claude to resolve
    if all_differences:
        if config.no_llm:
            print("  --no-llm flag set - using base model without resolving differences")
            ensembled_text = base_text
        else:
            ensembled_text = _resolve_whisper_differences(
                base_text, transcripts, all_differences, config
            )
    else:
        print("  No significant differences found between models")
        ensembled_text = base_text

    # Save ensembled transcript
    ensembled_path = config.output_dir / "ensembled.txt"
    with open(ensembled_path, 'w') as f:
        f.write(ensembled_text)

    data.transcript_path = ensembled_path
    print(f"  Ensembled transcript saved: {ensembled_path.name}")

    # Use the largest model's JSON for timestamps
    if base_model in data.whisper_transcripts:
        data.transcript_json_path = data.whisper_transcripts[base_model].get("json")


def _resolve_whisper_differences(base_text: str, all_transcripts: dict,
                                  differences: list, config: SpeechConfig) -> str:
    """Use LLM to resolve differences between Whisper model outputs."""

    # Format differences for Claude
    meaningful_diffs = _filter_meaningful_diffs(differences)
    diff_summary = ""
    if meaningful_diffs:
        diff_summary = "\nDIFFERENCES BETWEEN WHISPER MODELS:\n"
        for i, d in enumerate(meaningful_diffs[:50], 1):
            if d["type"] == "changed":
                diff_summary += f"{i}. {d['a_source']} has \"{d['a_text']}\" vs {d['b_source']} has \"{d['b_text']}\"\n"
            elif d["type"] == "a_only":
                diff_summary += f"{i}. {d['source']} has: \"{d['text']}\" (missing in other)\n"
            elif d["type"] == "b_only":
                diff_summary += f"{i}. {d['source']} has: \"{d['text']}\" (missing in other)\n"

    # Split into chunks if transcript is long
    base_words = base_text.split()

    if len(base_words) <= config.merge_chunk_words:
        return _resolve_whisper_chunk(base_text, all_transcripts,
                                      diff_summary, config)

    # Process in chunks
    num_chunks = (len(base_words) // config.merge_chunk_words) + 1
    print(f"  Splitting ensembling into {num_chunks} chunks...")
    merged_chunks = []

    for i in range(num_chunks):
        start_idx = i * config.merge_chunk_words
        end_idx = min((i + 1) * config.merge_chunk_words, len(base_words))
        base_chunk = " ".join(base_words[start_idx:end_idx])

        # Proportionally slice other transcripts
        other_chunks = {}
        for model, text in all_transcripts.items():
            if model != "ensembled":
                words = text.split()
                ot_start = int(start_idx * len(words) / len(base_words))
                ot_end = int(end_idx * len(words) / len(base_words))
                other_chunks[model] = " ".join(words[ot_start:ot_end])

        print(f"  Ensembling chunk {i+1}/{num_chunks}...")
        chunk_result = _resolve_whisper_chunk(base_chunk, other_chunks,
                                              diff_summary, config)
        merged_chunks.append(chunk_result)

    return "\n\n".join(merged_chunks)


def _resolve_whisper_chunk(base_text: str, all_transcripts: dict,
                            diff_summary: str, config: SpeechConfig) -> str:
    """Use LLM to resolve differences in a chunk of Whisper model outputs."""
    # Format other transcripts for reference
    other_transcripts_text = ""
    for model, text in all_transcripts.items():
        if model != "ensembled":
            other_transcripts_text += f"\n--- {model.upper()} MODEL ---\n{text}\n"

    client = create_llm_client(config)

    prompt = f"""You are ensembling multiple Whisper transcripts of the same speech to create the most accurate version.

BASE TRANSCRIPT (largest model, use as primary source):
---
{base_text}
---

OTHER MODEL OUTPUTS (for reference):
{other_transcripts_text}
{diff_summary}
INSTRUCTIONS:
1. Start with the base transcript
2. Review each difference - if a smaller model captured something the larger model missed
   (especially proper nouns, technical terms, or additional content), incorporate it
3. When models disagree on a word, prefer the version that:
   - Makes more grammatical sense
   - Is a real word/name (not a transcription error like "progerium" vs "progeria")
   - Fits the context better
4. Output the COMPLETE ensembled transcript
5. Do NOT add commentary - output ONLY the transcript text

Output the ensembled transcript:"""

    message = llm_call_with_retry(
        client, config,
        model=config.claude_model,
        max_tokens=16384,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text.strip()


def _load_transcript_segments(data: SpeechData) -> None:
    """Load transcript segments with timestamps from JSON file."""
    if not data.transcript_json_path or not data.transcript_json_path.exists():
        return

    try:
        with open(data.transcript_json_path, 'r') as f:
            transcript_data = json.load(f)

        segments = transcript_data.get("segments", [])
        data.transcript_segments = [
            {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip()
            }
            for seg in segments
            if seg.get("text", "").strip()
        ]
        print(f"  Loaded {len(data.transcript_segments)} transcript segments with timestamps")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Warning: Could not parse transcript JSON: {e}")
