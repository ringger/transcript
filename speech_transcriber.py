#!/usr/bin/env python3
"""
Speech Transcriber
==================
Automates transcription of speeches from video URLs.

Pipeline:
1. Download audio, video, and captions (yt-dlp)
2. Transcribe audio (mlx-whisper or openai-whisper)
3. Extract slides via scene detection (ffmpeg)
4. Optionally analyze slides with vision API (Claude)
4b. Optionally merge YouTube captions + Whisper into "critical text" (wdiff + Claude)
5. Generate markdown with slides interleaved at correct timestamps

Usage:
    python speech_transcriber.py <url> [options]

Examples:
    # Basic usage - Whisper transcript + slides
    python speech_transcriber.py "https://youtube.com/watch?v=..."

    # Full pipeline with slide analysis
    python speech_transcriber.py "https://youtube.com/watch?v=..." --analyze-slides

    # Skip merging YouTube captions (merge is on by default)
    python speech_transcriber.py "https://youtube.com/watch?v=..." --no-merge

    # Full pipeline with slide analysis (merging is automatic)
    python speech_transcriber.py "https://youtube.com/watch?v=..." --analyze-slides

    # Ensemble multiple Whisper models for better accuracy
    python speech_transcriber.py "https://youtube.com/watch?v=..." --whisper-models small,medium

    # Run without any API calls (free, local only)
    python speech_transcriber.py "https://youtube.com/watch?v=..." --no-api

    # Custom output directory and model
    python speech_transcriber.py "https://youtube.com/watch?v=..." -o my_speech --whisper-models small
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import shutil

# Ensure print output is flushed immediately (important when redirecting to log files)
import functools
print = functools.partial(print, flush=True)

@dataclass
class SpeechConfig:
    """Configuration for speech transcription pipeline."""
    url: str
    output_dir: Path
    whisper_models: list = field(default_factory=lambda: ["small", "medium"])  # Can be multiple models
    scene_threshold: float = 0.1
    analyze_slides: bool = False
    merge_sources: bool = True  # Merge YouTube captions with Whisper (default: on)
    no_api: bool = False  # Skip all API-dependent features
    api_key: Optional[str] = None
    claude_model: str = "claude-sonnet-4-20250514"
    skip_existing: bool = True
    no_slides: bool = False  # Skip slide extraction entirely
    external_transcript: Optional[str] = None  # External transcript file path or URL to include in merge
    dry_run: bool = False  # Show what would be done without doing it
    verbose: bool = False
    # Merge tuning
    merge_chunk_words: int = 500  # Words per chunk for merge API calls
    api_max_retries: int = 5
    api_initial_backoff: int = 5  # seconds
    api_timeout: float = 120.0  # seconds per API attempt


@dataclass
class SpeechData:
    """Data collected during pipeline execution."""
    title: str = ""
    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None
    captions_path: Optional[Path] = None
    transcript_path: Optional[Path] = None  # Primary transcript (or ensembled)
    transcript_json_path: Optional[Path] = None  # JSON with timestamps
    whisper_transcripts: dict = field(default_factory=dict)  # {model: path} for each model
    merged_transcript_path: Optional[Path] = None  # Merged from multiple sources
    slides_dir: Optional[Path] = None
    slides_json_path: Optional[Path] = None
    markdown_path: Optional[Path] = None
    slide_images: list = field(default_factory=list)
    slide_metadata: list = field(default_factory=list)
    slide_timestamps: list = field(default_factory=list)  # When each slide appears
    transcript_segments: list = field(default_factory=list)  # Segments with timestamps


def is_up_to_date(output: Path, *inputs: Path) -> bool:
    """Check if output file is newer than all input files (make-style)."""
    if not output.exists():
        return False
    output_mtime = output.stat().st_mtime
    for inp in inputs:
        if inp and inp.exists() and inp.stat().st_mtime > output_mtime:
            return False
    return True


def run_command(cmd: list[str], description: str, verbose: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command with error handling."""
    if verbose:
        print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"  Error during {description}:")
        print(f"  {e.stderr}")
        raise




def api_call_with_retry(client, config: SpeechConfig, **kwargs) -> object:
    """Call client.messages.create with exponential backoff on transient errors.

    Retries on 429 (rate limit), 529 (overloaded), 500 (internal server error),
    and timeouts (APITimeoutError).
    """
    import anthropic
    if "timeout" not in kwargs:
        kwargs["timeout"] = config.api_timeout
    delay = config.api_initial_backoff
    for attempt in range(1, config.api_max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.APITimeoutError:
            if attempt < config.api_max_retries:
                print(f"    API timeout, retrying in {delay}s (attempt {attempt}/{config.api_max_retries})...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529, 500) and attempt < config.api_max_retries:
                print(f"  API {e.status_code} error, retrying in {delay}s (attempt {attempt}/{config.api_max_retries})...")
                time.sleep(delay)
                delay *= 2
            else:
                raise


def check_dependencies() -> dict[str, bool]:
    """Check for required external tools."""
    deps = {
        "yt-dlp": False,
        "ffmpeg": False,
        "mlx_whisper": False,
        "whisper": False,
    }

    # Check command-line tools
    for tool in ["yt-dlp", "ffmpeg"]:
        deps[tool] = shutil.which(tool) is not None

    # Check Python packages
    try:
        import mlx_whisper
        deps["mlx_whisper"] = True
    except ImportError:
        pass

    try:
        import whisper
        deps["whisper"] = True
    except ImportError:
        pass

    return deps


def estimate_api_cost(config: SpeechConfig, num_slides: int = 45, transcript_words: int = 6000) -> dict:
    """Estimate API costs for the configured options.

    Pricing estimates (Claude Sonnet 4):
    - Input: $3 per 1M tokens (~$0.003 per 1K tokens)
    - Output: $15 per 1M tokens (~$0.015 per 1K tokens)
    - Vision: ~$0.01-0.02 per image (varies by size)

    Returns dict with cost breakdown and total.
    """
    costs = {
        "analyze_slides": 0.0,
        "merge_sources": 0.0,
        "ensemble_whisper": 0.0,
        "total": 0.0,
        "details": []
    }

    # Estimate tokens (rough: 1 word ≈ 1.3 tokens)
    transcript_tokens = int(transcript_words * 1.3)

    if config.analyze_slides and not config.no_api:
        # Vision API: ~$0.02 per image for medium-sized slides
        slide_cost = num_slides * 0.02
        costs["analyze_slides"] = slide_cost
        costs["details"].append(f"Slide analysis: {num_slides} slides × $0.02 = ${slide_cost:.2f}")

    if config.merge_sources and not config.no_api:
        # Count merge sources: Whisper + YouTube captions + optional external
        num_sources = 2  # Whisper + YouTube
        if config.external_transcript:
            num_sources += 1
        # Each chunk sends all sources as input, output ≈ 1x base transcript
        num_chunks = max(1, transcript_words // config.merge_chunk_words + 1)
        # Input per chunk: all sources (~transcript_words * num_sources / num_chunks)
        #   + wdiff differences summary (~500 tokens overhead)
        # Output per chunk: ~transcript_words / num_chunks
        chunk_input_words = transcript_words * num_sources // num_chunks + 500
        chunk_output_words = transcript_words // num_chunks
        total_input_tokens = int(chunk_input_words * num_chunks * 1.3)
        total_output_tokens = int(chunk_output_words * num_chunks * 1.3)
        input_cost = total_input_tokens * 0.003 / 1000
        output_cost = total_output_tokens * 0.015 / 1000
        merge_cost = input_cost + output_cost
        costs["merge_sources"] = merge_cost
        costs["details"].append(
            f"Source merging: {num_sources} sources × {num_chunks} chunks = ${merge_cost:.2f}")

    if len(config.whisper_models) > 1 and not config.no_api:
        # Ensemble: chunked processing, each chunk sends base + other models + diffs
        num_models = len(config.whisper_models)
        num_chunks = max(1, transcript_words // config.merge_chunk_words + 1)
        chunk_input_words = transcript_words * num_models // num_chunks + 500
        chunk_output_words = transcript_words // num_chunks
        total_input_tokens = int(chunk_input_words * num_chunks * 1.3)
        total_output_tokens = int(chunk_output_words * num_chunks * 1.3)
        input_cost = total_input_tokens * 0.003 / 1000
        output_cost = total_output_tokens * 0.015 / 1000
        ensemble_cost = input_cost + output_cost
        costs["ensemble_whisper"] = ensemble_cost
        costs["details"].append(
            f"Whisper ensemble: {num_models} models × {num_chunks} chunks = ${ensemble_cost:.2f}")

    costs["total"] = costs["analyze_slides"] + costs["merge_sources"] + costs["ensemble_whisper"]

    return costs


def _dry_run_skip(config: SpeechConfig, action: str, output: str) -> bool:
    """In dry-run mode, print what would happen and return True to skip execution."""
    if not config.dry_run:
        return False
    print(f"  [dry-run] Would {action} → {output}")
    return True

    print("\n" + "="*50)


def print_cost_estimate(config: SpeechConfig, num_slides: int = 45, transcript_words: int = 6000) -> None:
    """Print estimated API costs before running."""
    costs = estimate_api_cost(config, num_slides, transcript_words)

    if costs["total"] == 0:
        return

    print("\n" + "="*50)
    print("ESTIMATED API COSTS")
    print("="*50)

    for detail in costs["details"]:
        print(f"  {detail}")

    print(f"\n  TOTAL: ${costs['total']:.2f} (estimate)")
    print("  Note: Actual costs may vary based on transcript length")
    print("="*50 + "\n")


def download_media(config: SpeechConfig, data: SpeechData) -> None:
    """Download audio, video, and captions using yt-dlp."""
    print("\n[1/5] Downloading media...")

    output_template = str(config.output_dir / "%(title)s.%(ext)s")

    # Get video info first to extract title
    print("  Fetching video info...")
    result = run_command(
        ["yt-dlp", "--dump-json", config.url],
        "fetching video info",
        config.verbose
    )
    info = json.loads(result.stdout)
    data.title = info.get("title", "speech")

    print(f"  Title: {data.title}")

    # Save source metadata
    metadata_path = config.output_dir / "metadata.json"
    if not metadata_path.exists() or not config.skip_existing:
        metadata = {
            "url": config.url,
            "video_id": info.get("id"),
            "title": data.title,
            "channel": info.get("channel") or info.get("uploader"),
            "upload_date": info.get("upload_date"),
            "duration_seconds": info.get("duration"),
            "description": info.get("description", "")[:500],
        }
        if config.external_transcript:
            metadata["external_transcript"] = config.external_transcript
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved: {metadata_path.name}")

    # Download audio
    audio_path = config.output_dir / "audio.mp3"
    if config.skip_existing and audio_path.exists():
        print(f"  Reusing: {audio_path.name}")
    elif not _dry_run_skip(config, "download audio", "audio.mp3"):
        print("  Downloading audio...")
        run_command(
            ["yt-dlp", "-x", "--audio-format", "mp3",
             "-o", str(config.output_dir / "audio.%(ext)s"), config.url],
            "downloading audio",
            config.verbose
        )
    data.audio_path = audio_path

    # Download video (only needed for slide extraction)
    if config.no_slides:
        print("  Skipping video download (--no-slides)")
    else:
        video_path = config.output_dir / "video.mp4"
        if config.skip_existing and video_path.exists():
            print(f"  Reusing: {video_path.name}")
        elif not _dry_run_skip(config, "download video", "video.mp4"):
            print("  Downloading video...")
            run_command(
                ["yt-dlp", "-f", "mp4",
                 "-o", str(config.output_dir / "video.%(ext)s"), config.url],
                "downloading video",
                config.verbose
            )
        data.video_path = video_path

    # Download captions if available
    captions_path = config.output_dir / "captions.en.vtt"
    if config.skip_existing and captions_path.exists():
        print(f"  Reusing: {captions_path.name}")
    elif not _dry_run_skip(config, "download captions", "captions.en.vtt"):
        print("  Downloading captions (if available)...")
        try:
            run_command(
                ["yt-dlp", "--write-auto-sub", "--sub-lang", "en", "--skip-download",
                 "-o", str(config.output_dir / "captions.%(ext)s"), config.url],
                "downloading captions",
                config.verbose
            )
        except subprocess.CalledProcessError:
            print("  No captions available")

    if captions_path.exists():
        data.captions_path = captions_path
        print(f"  Captions saved: {captions_path.name}")


def clean_vtt_captions(vtt_path: Path) -> str:
    """Convert VTT captions to clean text."""
    with open(vtt_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    text_lines = []
    seen = set()

    for line in lines:
        # Skip VTT headers
        if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
            continue
        # Skip timestamps
        if re.match(r'^\d{2}:\d{2}:\d{2}', line):
            continue
        if '-->' in line:
            continue
        if not line.strip():
            continue
        # Remove HTML tags and clean
        clean = re.sub(r'<[^>]+>', '', line).strip()
        if clean and clean not in seen:
            seen.add(clean)
            text_lines.append(clean)

    return ' '.join(text_lines)


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
        print(f"  Reusing: {txt_path.name}")
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

        with open(json_path, 'w') as f:
            json.dump({
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language", "en")
            }, f, indent=2)

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
            print(f"  Reusing: {ensembled_path.name}")
            data.transcript_path = ensembled_path
            # Use the largest model's JSON for timestamps
            model_sizes = ["large", "medium", "small", "base", "tiny"]
            for size in model_sizes:
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
    model_sizes = ["large", "medium", "small", "base", "tiny"]
    base_model = None
    for size in model_sizes:
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
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if config.no_api:
            print("  --no-api flag set - using base model without resolving differences")
            ensembled_text = base_text
        elif api_key:
            ensembled_text = _resolve_whisper_differences(
                api_key, base_text, transcripts, all_differences, config
            )
        else:
            print("  No API key - using base model without resolving differences")
            ensembled_text = base_text
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


def _resolve_whisper_differences(api_key: str, base_text: str, all_transcripts: dict,
                                  differences: list, config: SpeechConfig) -> str:
    """Use Claude to resolve differences between Whisper model outputs."""

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
        return _resolve_whisper_chunk(api_key, base_text, all_transcripts,
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
        chunk_result = _resolve_whisper_chunk(api_key, base_chunk, other_chunks,
                                              diff_summary, config)
        merged_chunks.append(chunk_result)

    return "\n\n".join(merged_chunks)


def _resolve_whisper_chunk(api_key: str, base_text: str, all_transcripts: dict,
                            diff_summary: str, config: SpeechConfig) -> str:
    """Use Claude to resolve differences in a chunk of Whisper model outputs."""
    import anthropic

    # Format other transcripts for reference
    other_transcripts_text = ""
    for model, text in all_transcripts.items():
        if model != "ensembled":
            other_transcripts_text += f"\n--- {model.upper()} MODEL ---\n{text}\n"

    client = anthropic.Anthropic(api_key=api_key)

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

    message = api_call_with_retry(
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


def extract_slides(config: SpeechConfig, data: SpeechData) -> None:
    """Extract slides from video using ffmpeg scene detection, capturing timestamps."""
    print("\n[3/5] Extracting slides...")

    if not data.video_path or not data.video_path.exists():
        print("  No video file available, skipping slide extraction")
        return

    slides_dir = config.output_dir / "slides"
    slides_dir.mkdir(exist_ok=True)
    data.slides_dir = slides_dir

    timestamps_file = config.output_dir / "slide_timestamps.json"

    existing_slides = list(slides_dir.glob("slide_*.png"))
    if config.skip_existing and existing_slides and is_up_to_date(timestamps_file, data.video_path):
        print(f"  Reusing: {len(existing_slides)} slides")
        data.slide_images = sorted(existing_slides)
        # Load existing timestamps
        _load_slide_timestamps(data, timestamps_file)
        return
    if _dry_run_skip(config, "extract slides from video", "slides/*.png"):
        return
    print(f"  Scene detection threshold: {config.scene_threshold}")

    # Run ffmpeg and capture stderr for timestamp info
    cmd = [
        "ffmpeg", "-i", str(data.video_path),
        "-vf", f"select='gt(scene,{config.scene_threshold})',showinfo",
        "-vsync", "vfr",
        str(slides_dir / "slide_%04d.png")
    ]

    if config.verbose:
        print(f"  Running: {' '.join(cmd)}")

    # ffmpeg outputs showinfo to stderr
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse timestamps from showinfo output
    # Format: [Parsed_showinfo_1 @ ...] n:   0 pts:      0 pts_time:0       ...
    timestamps = []
    for line in result.stderr.split('\n'):
        if 'pts_time:' in line:
            match = re.search(r'pts_time:(\d+\.?\d*)', line)
            if match:
                timestamps.append(float(match.group(1)))

    data.slide_images = sorted(slides_dir.glob("slide_*.png"))

    # Match timestamps to slides
    data.slide_timestamps = []
    for i, slide_path in enumerate(data.slide_images):
        timestamp = timestamps[i] if i < len(timestamps) else 0.0
        data.slide_timestamps.append({
            "slide_number": i + 1,
            "filename": slide_path.name,
            "timestamp": timestamp
        })

    # Save timestamps to JSON for reuse
    with open(timestamps_file, 'w') as f:
        json.dump(data.slide_timestamps, f, indent=2)

    print(f"  Extracted {len(data.slide_images)} slides with timestamps")
    if data.slide_timestamps:
        print(f"  Time range: {data.slide_timestamps[0]['timestamp']:.1f}s - {data.slide_timestamps[-1]['timestamp']:.1f}s")


def _load_slide_timestamps(data: SpeechData, timestamps_file: Path) -> None:
    """Load slide timestamps from JSON file."""
    try:
        with open(timestamps_file, 'r') as f:
            data.slide_timestamps = json.load(f)
        print(f"  Loaded timestamps for {len(data.slide_timestamps)} slides")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  Warning: Could not load slide timestamps: {e}")
        # Create basic timestamps (evenly distributed)
        data.slide_timestamps = [
            {"slide_number": i + 1, "filename": p.name, "timestamp": 0.0}
            for i, p in enumerate(data.slide_images)
        ]


def analyze_slides_with_vision(config: SpeechConfig, data: SpeechData) -> None:
    """Analyze slides using Claude vision API."""
    print("\n[4/5] Analyzing slides with vision API...")

    if not config.analyze_slides:
        print("  Skipped (use --analyze-slides to enable)")
        return

    if config.no_api:
        print("  Skipped (--no-api flag set)")
        return

    if not data.slide_images:
        print("  No slides to analyze")
        return

    slides_json_path = config.output_dir / "slides_transcript.json"
    if config.skip_existing and is_up_to_date(slides_json_path, *data.slide_images):
        print(f"  Reusing: {slides_json_path.name}")
        with open(slides_json_path, 'r') as f:
            slides_data = json.load(f)
        data.slide_metadata = slides_data.get("slides", [])
        data.slides_json_path = slides_json_path
        return
    if _dry_run_skip(config, "analyze slides with Claude Vision", "slides_transcript.json"):
        return

    # Get API key
    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key required for slide analysis. "
                        "Set ANTHROPIC_API_KEY environment variable or use --api-key")

    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required for slide analysis. "
                         "Install with: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)

    slides_metadata = []

    for i, slide_path in enumerate(data.slide_images):
        print(f"  Analyzing slide {i+1}/{len(data.slide_images)}: {slide_path.name}")

        # Read and encode image
        with open(slide_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Call Claude vision API
        message = api_call_with_retry(
            client, config,
            model=config.claude_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Analyze this presentation slide and return a JSON object with these fields:
- "type": one of "title", "content", "speaker", "transition", "venue", "performance", "end"
- "title": the slide title if visible (null if none)
- "subtitle": subtitle if present (null if none)
- "bullet_points": array of bullet points if present (null if none)
- "images": brief description of any images/graphics (null if none)
- "description": one-sentence description of what this slide shows

Return ONLY the JSON object, no other text."""
                        }
                    ],
                }
            ],
        )

        # Parse response
        try:
            response_text = message.content[0].text
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                slide_info = json.loads(json_match.group())
            else:
                slide_info = {"description": response_text, "type": "content"}
        except (json.JSONDecodeError, IndexError):
            slide_info = {"description": "Could not parse slide", "type": "unknown"}

        slide_info["slide_number"] = i + 1
        slide_info["filename"] = slide_path.name
        slides_metadata.append(slide_info)

    data.slide_metadata = slides_metadata

    # Save slides JSON
    slides_json_path = config.output_dir / "slides_transcript.json"
    with open(slides_json_path, 'w') as f:
        json.dump({
            "title": data.title,
            "slide_count": len(slides_metadata),
            "slides": slides_metadata
        }, f, indent=2)

    data.slides_json_path = slides_json_path
    print(f"  Slides JSON saved: {slides_json_path.name}")


def create_basic_slides_json(config: SpeechConfig, data: SpeechData) -> None:
    """Create a basic slides JSON without vision analysis."""
    slides_json_path = config.output_dir / "slides_basic.json"
    if config.skip_existing and is_up_to_date(slides_json_path, *data.slide_images):
        data.slides_json_path = slides_json_path
        return
    if _dry_run_skip(config, "create basic slides JSON", "slides_basic.json"):
        return

    slides_metadata = []
    for i, slide_path in enumerate(data.slide_images):
        slides_metadata.append({
            "slide_number": i + 1,
            "filename": slide_path.name,
            "type": "unknown",
            "description": f"Slide {i + 1} - requires manual analysis"
        })

    data.slide_metadata = slides_metadata

    slides_json_path = config.output_dir / "slides_basic.json"
    with open(slides_json_path, 'w') as f:
        json.dump({
            "title": data.title,
            "slide_count": len(slides_metadata),
            "note": "Basic metadata only - run with --analyze-slides for full analysis",
            "slides": slides_metadata
        }, f, indent=2)

    data.slides_json_path = slides_json_path


def merge_transcript_sources(config: SpeechConfig, data: SpeechData) -> None:
    """Merge YouTube captions with Whisper transcript using wdiff analysis and Claude API."""
    print("\n[4b/5] Merging transcript sources...")

    if not config.merge_sources:
        print("  Skipped (--no-merge flag set)")
        return

    if config.no_api:
        print("  Skipped (--no-api flag set)")
        return

    # Load external transcript if provided
    external_text = None
    if config.external_transcript:
        source_label = config.external_transcript
        if config.external_transcript.startswith(("http://", "https://")):
            print(f"  Fetching external transcript from URL...")
            import urllib.request
            try:
                with urllib.request.urlopen(config.external_transcript) as response:
                    raw = response.read().decode('utf-8').strip()
                # Convert HTML to text if needed
                if '<html' in raw[:500].lower() or '<body' in raw[:1000].lower():
                    external_text = _extract_text_from_html(raw)
                else:
                    external_text = raw
                source_label = config.external_transcript.split('/')[-1] or config.external_transcript
            except Exception as e:
                print(f"  Warning: Could not fetch external transcript URL: {e}")
        else:
            ext_path = Path(config.external_transcript)
            if ext_path.exists():
                with open(ext_path, 'r') as f:
                    external_text = f.read().strip()
                source_label = ext_path.name
        if external_text:
            print(f"  External transcript: {len(external_text.split())} words ({source_label})")

    # Check if we have enough sources to merge
    has_captions = data.captions_path and data.captions_path.exists()
    has_whisper = data.transcript_path and data.transcript_path.exists()

    if not has_whisper and not external_text:
        print("  No Whisper transcript or external transcript available, skipping merge")
        return

    if not has_captions and not external_text:
        print("  No YouTube captions or external transcript available, skipping merge")
        return

    merged_path = config.output_dir / "transcript_merged.txt"

    # Inputs: ensembled/whisper transcript, captions, external transcript
    merge_inputs = [p for p in [data.transcript_path, data.captions_path] if p]
    # External transcript from file (not URL) is also an input
    if config.external_transcript and not config.external_transcript.startswith(("http://", "https://")):
        ext_path = Path(config.external_transcript)
        if ext_path.exists():
            merge_inputs.append(ext_path)
    if config.skip_existing and is_up_to_date(merged_path, *merge_inputs):
        print(f"  Reusing: {merged_path.name}")
        data.merged_transcript_path = merged_path
        return
    if _dry_run_skip(config, "merge transcript sources", "transcript_merged.txt"):
        return

    # Get API key
    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key required for transcript merging. "
                        "Set ANTHROPIC_API_KEY environment variable or use --api-key")

    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required for transcript merging. "
                         "Install with: pip install anthropic")

    # Load available transcripts
    youtube_text = None
    whisper_text = None

    if has_captions:
        youtube_text = clean_vtt_captions(data.captions_path)
        print(f"  YouTube captions: {len(youtube_text.split())} words")

    if has_whisper:
        with open(data.transcript_path, 'r') as f:
            whisper_text = f.read()
        print(f"  Whisper transcript: {len(whisper_text.split())} words")

    # Build list of sources for merging
    sources = []
    if whisper_text:
        sources.append(("Whisper AI Transcript", "better punctuation, sentence structure, formatting", whisper_text))
    if youtube_text:
        sources.append(("YouTube Captions", "ASR with larger vocabulary, better for proper nouns, names, technical terms", youtube_text))
    if external_text:
        sources.append(("External Transcript", "additional reference source provided by user", external_text))

    print(f"  Merging {len(sources)} sources: {', '.join(s[0] for s in sources)}")

    if len(sources) < 2:
        print("  Need at least 2 sources to merge, skipping")
        return

    # Check if external transcript has structure (speaker labels, timestamps)
    structure = None
    if external_text:
        structure = _detect_transcript_structure(external_text)
        if structure["has_speakers"]:
            print(f"  Detected structured external transcript (format: {structure['format']}, "
                  f"speakers: {structure['has_speakers']}, timestamps: {structure['has_timestamps']})")

    # Route to structured merge if external has speaker labels
    if structure and structure["has_speakers"]:
        skeleton_segments = _parse_structured_transcript(external_text, structure["format"])
        print(f"  Parsed {len(skeleton_segments)} segments from external transcript")

        # Pass all sources — text is treated equally, structure comes from skeleton
        corrected_segments = _merge_structured(api_key, skeleton_segments, sources,
                                               config, merge_inputs)
        merged_text = _format_structured_segments(corrected_segments)
    else:
        # Flat merge: wdiff alignment and anonymous presentation
        merged_text = _merge_multi_source(api_key, sources, config, merge_inputs)

    # Save merged transcript
    with open(merged_path, 'w') as f:
        f.write(merged_text)

    data.merged_transcript_path = merged_path
    print(f"  Merged transcript saved: {merged_path.name}")
    print(f"  Merged word count: {len(merged_text.split())} words")


def _extract_text_from_html(html: str) -> str:
    """Extract transcript text from an HTML page, preserving speaker structure.

    Handles structured transcript pages (e.g., Lex Fridman format with
    ts-segment/ts-name/ts-timestamp/ts-text classes) and falls back to
    generic text extraction for other HTML.
    """
    import html as html_module
    from html.parser import HTMLParser

    # Check for structured transcript HTML (ts-segment pattern)
    if 'class="ts-segment"' in html:
        segments = re.findall(
            r'<div class="ts-segment">\s*'
            r'<span class="ts-name">([^<]+)</span>\s*'
            r'<span class="ts-timestamp"><a[^>]*>\(([^)]+)\)</a>\s*</span>\s*'
            r'<span class="ts-text">(.*?)</span>',
            html, re.DOTALL
        )
        if segments:
            lines = []
            for name, timestamp, text in segments:
                clean_text = re.sub(r'<[^>]+>', '', text)
                clean_text = html_module.unescape(clean_text).strip()
                name = html_module.unescape(name).strip()
                lines.append(f"{name} ({timestamp})")
                lines.append(clean_text)
                lines.append("")
            return "\n".join(lines)

    # Fallback: generic HTML to text
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
            self.skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ('script', 'style', 'nav', 'header', 'footer'):
                self.skip = True
            elif tag in ('p', 'br', 'div', 'h1', 'h2', 'h3', 'h4', 'li'):
                self.parts.append('\n')

        def handle_endtag(self, tag):
            if tag in ('script', 'style', 'nav', 'header', 'footer'):
                self.skip = False
            elif tag in ('p', 'div'):
                self.parts.append('\n')

        def handle_data(self, data):
            if not self.skip:
                self.parts.append(data)

    extractor = TextExtractor()
    extractor.feed(html)
    text = ''.join(extractor.parts)
    text = html_module.unescape(text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation per-word.

    Preserves word count: punctuation-only tokens (em-dashes, ellipses, etc.)
    are replaced with a placeholder so alignment maps stay in sync with
    original word positions.
    """
    words = text.split()
    normalized = []
    for w in words:
        cleaned = re.sub(r'[^\w]', '', w).lower()
        normalized.append(cleaned if cleaned else '_')
    return ' '.join(normalized)


def _analyze_differences_wdiff(text_a: str, text_b: str, config: SpeechConfig,
                               label_a: str = "source_a", label_b: str = "source_b") -> list:
    """Use wdiff to identify content differences between two texts.

    Returns a list of differences with types: 'a_only', 'b_only', 'changed'.
    """

    # Check if wdiff is available
    if not shutil.which("wdiff"):
        if config.verbose:
            print("  wdiff not found, skipping diff analysis")
        return []

    import tempfile

    # Normalize both texts for comparison
    a_normalized = _normalize_for_comparison(text_a)
    b_normalized = _normalize_for_comparison(text_b)

    # Write normalized texts to temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
        f1.write(a_normalized)
        a_file = f1.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
        f2.write(b_normalized)
        b_file = f2.name

    try:
        # Run wdiff
        result = subprocess.run(
            ["wdiff", "-s", a_file, b_file],
            capture_output=True,
            text=True
        )

        differences = []

        # Also run wdiff without -s to get actual differences
        diff_result = subprocess.run(
            ["wdiff", a_file, b_file],
            capture_output=True,
            text=True
        )

        # Parse differences - look for [-deleted-] and {+inserted+}
        diff_output = diff_result.stdout

        # [-word-] means in file1 (a) but not file2 (b)
        # {+word+} means in file2 (b) but not file1 (a)
        deleted = re.findall(r'\[-([^\]]+)-\]', diff_output)
        inserted = re.findall(r'\{\+([^\}]+)\+\}', diff_output)

        for d in deleted:
            if len(d.split()) <= 5:
                differences.append({"type": "a_only", "text": d, "source": label_a})

        for i in inserted:
            if len(i.split()) <= 5:
                differences.append({"type": "b_only", "text": i, "source": label_b})

        # Find changed pairs (adjacent delete/insert)
        changes = re.findall(r'\[-([^\]]+)-\]\s*\{\+([^\}]+)\+\}', diff_output)
        for a_ver, b_ver in changes:
            differences.append({
                "type": "changed",
                "a_text": a_ver,
                "b_text": b_ver,
                "a_source": label_a,
                "b_source": label_b,
            })

        if config.verbose:
            print(f"  wdiff stats ({label_a} vs {label_b}): {result.stderr.strip()}")

        return differences

    finally:
        # Clean up temp files
        os.unlink(a_file)
        os.unlink(b_file)


def _filter_meaningful_diffs(differences: list) -> list:
    """Filter wdiff differences to only meaningful ones (skip common words)."""
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'it'}
    meaningful_diffs = []
    for d in differences:
        if d["type"] == "changed":
            a_words = set(d["a_text"].lower().split())
            b_words = set(d["b_text"].lower().split())
            if not (a_words <= common_words and b_words <= common_words):
                meaningful_diffs.append(d)
        else:
            text = d.get("text", "").lower()
            if text and text not in common_words:
                meaningful_diffs.append(d)
    return meaningful_diffs


_WDIFF_TOKEN_PATTERN = re.compile(
    r'\[-(?P<deleted>.*?)-\]'
    r'|\{\+(?P<inserted>.*?)\+\}'
    r'|(?P<common>(?:(?!\[-|\{\+).)+)',
    re.DOTALL
)


def _parse_wdiff_tokens(wdiff_output: str) -> list[tuple[str, str]]:
    """Tokenize raw wdiff output into (type, text) tuples.

    Types:
      - "common": text present in both files
      - "deleted": text in file A only (from [-...-])
      - "inserted": text in file B only (from {+...+})
    """
    tokens = []
    for m in _WDIFF_TOKEN_PATTERN.finditer(wdiff_output):
        if m.group("deleted") is not None:
            tokens.append(("deleted", m.group("deleted")))
        elif m.group("inserted") is not None:
            tokens.append(("inserted", m.group("inserted")))
        elif m.group("common") is not None:
            text = m.group("common").strip()
            if text:
                tokens.append(("common", text))
    return tokens


def _build_wdiff_alignment(text_a: str, text_b: str,
                           config: SpeechConfig) -> list[int]:
    """Build a word-level alignment map from text_a to text_b using wdiff.

    Both texts are normalized before alignment, but the returned indices
    correspond to word positions in the original (space-split) texts since
    normalization preserves word count.

    Returns ext_to_other: list of length len(text_a.split()) + 1.
    ext_to_other[i] = the word index in text_b that corresponds to word i
    in text_a. The extra entry at the end provides the boundary for the
    last segment.
    """
    import tempfile

    norm_a = _normalize_for_comparison(text_a)
    norm_b = _normalize_for_comparison(text_b)

    a_path = None
    b_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(norm_a)
            a_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(norm_b)
            b_path = f.name

        result = subprocess.run(
            ["wdiff", a_path, b_path],
            capture_output=True, text=True
        )

        tokens = _parse_wdiff_tokens(result.stdout)

        a_pos = 0
        b_pos = 0
        a_word_count = len(norm_a.split())
        ext_to_other = [0] * (a_word_count + 1)

        for token_type, text in tokens:
            n = len(text.split())
            if token_type == "common":
                for _ in range(n):
                    if a_pos < a_word_count:
                        ext_to_other[a_pos] = b_pos
                    a_pos += 1
                    b_pos += 1
            elif token_type == "deleted":  # in A only
                for _ in range(n):
                    if a_pos < a_word_count:
                        ext_to_other[a_pos] = b_pos
                    a_pos += 1
            elif token_type == "inserted":  # in B only
                b_pos += n

        # Sentinel: end position for the last segment
        ext_to_other[a_word_count] = b_pos

        return ext_to_other

    finally:
        if a_path:
            os.unlink(a_path)
        if b_path:
            os.unlink(b_path)


def _format_differences(differences: list) -> str:
    """Format wdiff differences for inclusion in a Claude prompt."""
    diff_text = "IDENTIFIED DIFFERENCES (normalized, no punctuation):\n"
    for i, d in enumerate(differences, 1):
        if d["type"] == "changed":
            diff_text += f"{i}. {d['a_source']}: \"{d['a_text']}\" vs {d['b_source']}: \"{d['b_text']}\"\n"
        elif d["type"] == "a_only":
            diff_text += f"{i}. {d['source']} has: \"{d['text']}\" (missing in other)\n"
        elif d["type"] == "b_only":
            diff_text += f"{i}. {d['source']} has: \"{d['text']}\" (missing in other)\n"
    return diff_text


def _detect_transcript_structure(text: str) -> dict:
    """Detect if a transcript has speaker labels and/or timestamps.

    Returns dict with keys: has_speakers, has_timestamps, format.
    Supported formats:
      - "lex": "Speaker Name [(HH:MM:SS)](url) text" (Lex Fridman style)
      - "bracketed": "[HH:MM:SS] Speaker: text"
      - "speaker_only": "Speaker: text" (no timestamps)
      - None: unstructured text
    """
    lines = text.strip().split('\n')
    # Skip header lines (title, blank lines at start)
    content_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]

    # Sample first 20 content lines for pattern detection
    sample = content_lines[:20]

    # Lex Fridman format: "Speaker Name [(HH:MM:SS)](url)" or extracted "Speaker Name (HH:MM:SS)"
    lex_pattern = re.compile(r'^(\w[\w\s]+?)\s*\[?\((\d{1,2}:\d{2}:\d{2})\)\]?')
    lex_matches = sum(1 for line in sample if lex_pattern.match(line))
    if lex_matches >= 2:
        return {"has_speakers": True, "has_timestamps": True, "format": "lex"}

    # Bracketed format: "[HH:MM:SS] Speaker:"
    bracketed_pattern = re.compile(r'^\[(\d{1,2}:\d{2}:\d{2})\]\s+(\w[\w\s]+?):')
    bracketed_matches = sum(1 for line in sample if bracketed_pattern.match(line))
    if bracketed_matches >= 2:
        return {"has_speakers": True, "has_timestamps": True, "format": "bracketed"}

    # Speaker only: "Name Name:" at start of line, followed by text
    speaker_pattern = re.compile(r'^([A-Z][\w]+(?:\s+[A-Z][\w]+)*)\s*:\s+\S')
    speaker_matches = sum(1 for line in sample if speaker_pattern.match(line))
    if speaker_matches >= 2:
        return {"has_speakers": True, "has_timestamps": False, "format": "speaker_only"}

    return {"has_speakers": False, "has_timestamps": False, "format": None}


def _parse_structured_transcript(text: str, fmt: str) -> list:
    """Parse a structured transcript into segments.

    Returns list of dicts: [{"speaker": str, "timestamp": str|None, "text": str}, ...]
    """
    segments = []

    if fmt == "lex":
        # Lex Fridman: "Speaker Name [(HH:MM:SS)](url)\ntext" or "Speaker Name (HH:MM:SS)\ntext"
        pattern = re.compile(r'^(\w[\w\s]+?)\s*\[?\((\d{1,2}:\d{2}:\d{2})\)\]?')
        current = None

        for line in text.split('\n'):
            m = pattern.match(line)
            if m:
                if current:
                    segments.append(current)
                speaker = m.group(1).strip()
                timestamp = m.group(2)
                # Text may continue after the [(HH:MM:SS)](url) on the same line
                rest = pattern.sub('', line).strip()
                # Strip the markdown link portion if present
                rest = re.sub(r'^\([^)]*\)\s*', '', rest)
                current = {"speaker": speaker, "timestamp": timestamp, "text": rest}
            elif current is not None:
                if line.strip():
                    current["text"] += (" " if current["text"] else "") + line.strip()
                elif current["text"]:
                    current["text"] += "\n"
        if current:
            segments.append(current)

    elif fmt == "bracketed":
        pattern = re.compile(r'^\[(\d{1,2}:\d{2}:\d{2})\]\s+(\w[\w\s]+?):\s*(.*)')
        current = None

        for line in text.split('\n'):
            m = pattern.match(line)
            if m:
                if current:
                    segments.append(current)
                current = {"speaker": m.group(2).strip(), "timestamp": m.group(1), "text": m.group(3).strip()}
            elif current is not None:
                if line.strip():
                    current["text"] += " " + line.strip()
        if current:
            segments.append(current)

    elif fmt == "speaker_only":
        pattern = re.compile(r'^([A-Z][\w]+(?:\s+[A-Z][\w]+)*)\s*:\s+(.*)')
        current = None

        for line in text.split('\n'):
            m = pattern.match(line)
            if m:
                if current:
                    segments.append(current)
                current = {"speaker": m.group(1).strip(), "timestamp": None, "text": m.group(2).strip()}
            elif current is not None:
                if line.strip():
                    current["text"] += " " + line.strip()
        if current:
            segments.append(current)

    # Clean up trailing whitespace in text
    for seg in segments:
        seg["text"] = seg["text"].strip()

    return segments


MERGE_CHECKPOINT_VERSION = "5"


def _init_merge_chunks_dir(config: SpeechConfig) -> Path:
    """Initialize the merge_chunks directory with version tracking.

    Creates the directory, checks the version file, and clears stale
    checkpoints from previous algorithm versions.

    Returns the chunks_dir path.
    """
    chunks_dir = config.output_dir / "merge_chunks"
    chunks_dir.mkdir(exist_ok=True)
    version_file = chunks_dir / ".version"
    if not version_file.exists() or version_file.read_text().strip() != MERGE_CHECKPOINT_VERSION:
        for old_chunk in chunks_dir.glob("chunk_*.json"):
            old_chunk.unlink()
        version_file.write_text(MERGE_CHECKPOINT_VERSION)
    return chunks_dir


def _build_alignments(anchor_text: str, other_sources: list,
                      config: SpeechConfig) -> tuple[list, list]:
    """Build wdiff alignment maps from anchor text to each other source.

    Returns (alignments, other_words_lists) where:
      alignments[i] = word-position map from anchor to other_sources[i]
      other_words_lists[i] = other_sources[i] text split into words
    """
    print(f"  Building wdiff alignment for {len(other_sources)} source(s)...")
    alignments = []
    other_words_lists = []
    for name, desc, text in other_sources:
        alignments.append(_build_wdiff_alignment(anchor_text, text, config))
        other_words_lists.append(text.split())
    return alignments, other_words_lists


def _extract_aligned_chunk(anchor_words: list, start: int, end: int,
                           alignments: list, other_words_lists: list) -> list[str]:
    """Extract aligned text for a word range from all sources.

    Returns list of texts: [anchor_chunk_text, other1_chunk_text, ...].
    """
    texts = [" ".join(anchor_words[start:end])]
    for src_idx, alignment in enumerate(alignments):
        other_start = alignment[start]
        other_end = alignment[min(end, len(alignment) - 1)]
        other_text = " ".join(other_words_lists[src_idx][other_start:other_end])
        texts.append(other_text if other_text.strip() else "(no corresponding text)")
    return texts


def _count_fresh_chunks(num_chunks: int, chunks_dir: Path,
                        source_paths: list) -> int:
    """Count contiguous fresh checkpoint files from the start.

    Returns the number of fresh (up-to-date) chunks.
    """
    chunks_reused = 0
    for i in range(num_chunks):
        checkpoint_path = chunks_dir / f"chunk_{i:03d}.json"
        if source_paths and is_up_to_date(checkpoint_path, *source_paths):
            chunks_reused += 1
        else:
            break
    return chunks_reused


def _load_chunk_checkpoint(chunks_dir: Path, chunk_idx: int):
    """Load a checkpoint file and return the parsed JSON data."""
    checkpoint_path = chunks_dir / f"chunk_{chunk_idx:03d}.json"
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def _save_chunk_checkpoint(chunks_dir: Path, chunk_idx: int, data):
    """Save data as a checkpoint JSON file."""
    checkpoint_path = chunks_dir / f"chunk_{chunk_idx:03d}.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f)


def _compute_chunk_diffs(chunk_texts: list, config: SpeechConfig) -> str:
    """Compute pairwise wdiff differences for a set of source texts.

    Uses anonymous Source N labels. Returns formatted diff section string.
    """
    all_diffs = []
    for i in range(len(chunk_texts)):
        for j in range(i + 1, len(chunk_texts)):
            diffs = _analyze_differences_wdiff(
                chunk_texts[i], chunk_texts[j], config,
                label_a=f"Source {i+1}", label_b=f"Source {j+1}")
            all_diffs.extend(diffs)
    if all_diffs:
        meaningful = _filter_meaningful_diffs(all_diffs)
        if meaningful:
            return _format_differences(meaningful) + "\n"
    return ""


def _merge_structured(api_key: str, skeleton_segments: list, all_sources: list,
                      config: SpeechConfig, source_paths: list = None) -> list:
    """Merge transcript sources using blind, label-free presentation.

    The skeleton_segments provide structure (speaker labels, timestamps, segment
    boundaries) from the external transcript. Text is presented to Claude
    anonymously — no source names, no speaker labels, no timestamps — so that
    no source receives preferential treatment.

    Alignment between sources uses wdiff rather than proportional word-fraction
    slicing, with cursor-based tracking to prevent segment overlap.

    Chunks are first-class DAG artefacts in merge_chunks/. Each chunk is checked
    for staleness against source_paths and reused when fresh.

    skeleton_segments: parsed segments from _parse_structured_transcript
    all_sources: list of (name, description, text) tuples (ALL sources including external)
    source_paths: list of Path objects for staleness checks

    Returns list of merged segments (same structure as skeleton_segments).
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    chunks_dir = _init_merge_chunks_dir(config)

    # Step 1: Separate external from other sources
    external_text = None
    other_sources = []
    for name, desc, text in all_sources:
        if name == "External Transcript":
            external_text = text
        else:
            other_sources.append((name, desc, text))

    if external_text is None:
        raise ValueError("External Transcript not found in all_sources")

    num_sources = 1 + len(other_sources)  # external + others

    # Step 2: Build wdiff alignment maps (once, up front)
    ext_full_text = " ".join(seg["text"] for seg in skeleton_segments)
    ext_words = ext_full_text.split()
    alignments, other_words_lists = _build_alignments(ext_full_text, other_sources, config)

    # Step 3: Compute per-segment word ranges and extract aligned text
    seg_source_texts = []  # seg_source_texts[i] = [ext_text, other1_text, ...]
    pos = 0
    for seg in skeleton_segments:
        n = len(seg["text"].split())
        start, end = pos, pos + n
        texts = _extract_aligned_chunk(ext_words, start, end, alignments, other_words_lists)
        seg_source_texts.append(texts)
        pos = end

    # Step 4: Group segments into chunks
    chunks = []  # each chunk is a list of segment indices
    current_chunk = []
    current_word_count = 0
    for seg_idx, seg in enumerate(skeleton_segments):
        seg_words = len(seg["text"].split())
        if current_word_count + seg_words > config.merge_chunk_words and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_word_count = 0
        current_chunk.append(seg_idx)
        current_word_count += seg_words
    if current_chunk:
        chunks.append(current_chunk)

    print(f"  Merging in {len(chunks)} chunks (~{config.merge_chunk_words} words each)...")

    # Step 5: Check for reusable checkpoints
    corrected_segments = [None] * len(skeleton_segments)
    chunks_reused = _count_fresh_chunks(len(chunks), chunks_dir, source_paths)

    if chunks_reused == len(chunks):
        print(f"  Reusing all {len(chunks)} chunks from checkpoint")
        for chunk_idx, seg_indices in enumerate(chunks):
            chunk_data = _load_chunk_checkpoint(chunks_dir, chunk_idx)
            for i, seg_idx in enumerate(seg_indices):
                if i < len(chunk_data):
                    corrected_segments[seg_idx] = chunk_data[i]
        return [s for s in corrected_segments if s is not None]

    if chunks_reused > 0:
        print(f"  Reusing {chunks_reused}/{len(chunks)} chunks from checkpoint, processing {len(chunks) - chunks_reused} via API...")
        for chunk_idx in range(chunks_reused):
            chunk_data = _load_chunk_checkpoint(chunks_dir, chunk_idx)
            for i, seg_idx in enumerate(chunks[chunk_idx]):
                if i < len(chunk_data):
                    corrected_segments[seg_idx] = chunk_data[i]

    # Step 6: Process remaining chunks via API
    for chunk_idx, seg_indices in enumerate(chunks):
        if chunk_idx < chunks_reused:
            continue

        chunk_words = sum(len(seg["text"].split()) for seg in (skeleton_segments[i] for i in seg_indices))
        print(f"  Merging chunk {chunk_idx + 1}/{len(chunks)} via API ({len(seg_indices)} passages, ~{chunk_words} words/source)...")

        # Build anonymous passage text
        passage_texts = ""
        for p_idx, seg_idx in enumerate(seg_indices, 1):
            passage_texts += f"PASSAGE {p_idx}:\n"
            for src_num, src_text in enumerate(seg_source_texts[seg_idx], 1):
                passage_texts += f"  Source {src_num}: {src_text}\n"
            passage_texts += "\n"

        # Per-passage differences (anonymous labels)
        diff_section = ""
        for p_idx, seg_idx in enumerate(seg_indices, 1):
            chunk_diffs = _compute_chunk_diffs(seg_source_texts[seg_idx], config)
            if chunk_diffs:
                diff_section += f"PASSAGE {p_idx} DIFFERENCES:\n"
                diff_section += chunk_diffs

        prompt = f"""You are creating an accurate transcript by merging {num_sources} independent transcriptions of the same speech.
No source is more reliable than any other — judge each difference on its merits.

{passage_texts}{diff_section}INSTRUCTIONS:
1. For each passage, produce the most accurate version by choosing the best words from any source.
2. When sources disagree:
   - Prefer proper nouns, names, or technical terms over common/generic words
   - Prefer the version that makes more grammatical and contextual sense
   - If one source includes words that others omit, include them if they fit the context
3. Output each passage on its own line in this format:
   PASSAGE 1: [merged text]
   PASSAGE 2: [merged text]
4. Do NOT add commentary or notes — output ONLY the merged passages.
5. Output exactly {len(seg_indices)} passages.

Output the merged passages:"""

        prompt_words = len(prompt.split())
        print(f"    Prompt: ~{prompt_words} words ({len(prompt)} chars)")

        message = api_call_with_retry(
            client, config,
            model=config.claude_model,
            max_tokens=16384,
            messages=[{"role": "user", "content": prompt}]
        )

        response = message.content[0].text.strip()
        print(f"    Response: {len(response)} chars, usage: {message.usage.input_tokens} in / {message.usage.output_tokens} out")

        # Parse response: PASSAGE N: text
        passage_pattern = re.compile(
            r'PASSAGE\s+(\d+)\s*:\s*(.*?)(?=PASSAGE\s+\d+\s*:|\Z)', re.DOTALL)
        matches = passage_pattern.findall(response)
        merged_texts = {int(num): text.strip() for num, text in matches}

        # Re-attach structure from skeleton segments
        chunk_result = []
        for p_idx, seg_idx in enumerate(seg_indices, 1):
            merged = dict(skeleton_segments[seg_idx])
            if p_idx in merged_texts:
                merged["text"] = merged_texts[p_idx]
            chunk_result.append(merged)
            corrected_segments[seg_idx] = merged

        _save_chunk_checkpoint(chunks_dir, chunk_idx, chunk_result)

    return [s for s in corrected_segments if s is not None]


def _format_structured_segments(segments: list) -> str:
    """Format corrected structured segments back into readable text."""
    lines = []
    for seg in segments:
        header = f"**{seg['speaker']}**"
        if seg.get("timestamp"):
            header += f" ({seg['timestamp']})"
        lines.append(header)
        lines.append("")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _merge_multi_source(api_key: str, sources: list,
                        config: SpeechConfig, source_paths: list = None) -> str:
    """Merge multiple transcript sources using Claude API with blind presentation.

    Uses wdiff alignment (not proportional slicing) to split sources into
    aligned chunks. Sources are presented anonymously as Source 1, Source 2, etc.

    Chunks are first-class DAG artefacts in merge_chunks/. Each chunk is checked
    for staleness against source_paths and reused when fresh.

    sources: list of (name, description, text) tuples
    source_paths: list of Path objects for staleness checks
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    chunks_dir = _init_merge_chunks_dir(config)

    # Use first source as anchor for alignment and chunking
    anchor_text = sources[0][2]
    anchor_words = anchor_text.split()
    num_sources = len(sources)

    # Build wdiff alignment maps (once, up front)
    other_sources = sources[1:]
    alignments, other_words_lists = _build_alignments(anchor_text, other_sources, config)

    # Chunk by anchor word boundaries
    chunks = []  # list of (start, end) word indices in anchor
    pos = 0
    while pos < len(anchor_words):
        end = min(pos + config.merge_chunk_words, len(anchor_words))
        chunks.append((pos, end))
        pos = end

    num_chunks = len(chunks)
    print(f"  Merging in {num_chunks} chunks (~{config.merge_chunk_words} words each)...")

    # Check for reusable checkpoints
    merged_chunks = []
    chunks_reused = _count_fresh_chunks(num_chunks, chunks_dir, source_paths)

    if chunks_reused == num_chunks:
        print(f"  Reusing all {num_chunks} chunks from checkpoint")
        for i in range(num_chunks):
            merged_chunks.append(_load_chunk_checkpoint(chunks_dir, i))
        return "\n\n".join(merged_chunks)

    if chunks_reused > 0:
        print(f"  Reusing {chunks_reused}/{num_chunks} chunks from checkpoint, processing {num_chunks - chunks_reused} via API...")
        for i in range(chunks_reused):
            merged_chunks.append(_load_chunk_checkpoint(chunks_dir, i))

    for chunk_idx, (start, end) in enumerate(chunks):
        if chunk_idx < chunks_reused:
            continue

        chunk_words = end - start
        print(f"  Merging chunk {chunk_idx + 1}/{num_chunks} via API (~{chunk_words} words/source)...")

        # Extract aligned text for this chunk from all sources
        chunk_texts = _extract_aligned_chunk(anchor_words, start, end,
                                             alignments, other_words_lists)

        # Build anonymous source text
        sources_text = ""
        for src_num, src_text in enumerate(chunk_texts, 1):
            sources_text += f"Source {src_num}:\n{src_text}\n\n"

        # Per-chunk differences with anonymous labels
        diff_section = _compute_chunk_diffs(chunk_texts, config)
        if diff_section:
            diff_section = "\n" + diff_section

        prompt = f"""You are creating an accurate transcript by merging {num_sources} independent transcriptions of the same speech.
No source is more reliable than any other — judge each difference on its merits.

{sources_text}{diff_section}INSTRUCTIONS:
1. Produce the most accurate version by choosing the best words from any source.
2. When sources disagree:
   - Prefer proper nouns, names, or technical terms over common/generic words
   - Prefer the version that makes more grammatical and contextual sense
   - If one source includes words that others omit, include them if they fit the context
3. Maintain natural paragraph breaks for readability.
4. Do NOT add commentary or notes — output ONLY the merged transcript text.

Output the merged transcript:"""

        prompt_words = len(prompt.split())
        print(f"    Prompt: ~{prompt_words} words ({len(prompt)} chars)")

        message = api_call_with_retry(
            client, config,
            model=config.claude_model,
            max_tokens=16384,
            messages=[{"role": "user", "content": prompt}]
        )

        chunk_merged = message.content[0].text.strip()
        print(f"    Response: {len(chunk_merged)} chars, usage: {message.usage.input_tokens} in / {message.usage.output_tokens} out")
        _save_chunk_checkpoint(chunks_dir, chunk_idx, chunk_merged)
        merged_chunks.append(chunk_merged)

    return "\n\n".join(merged_chunks)


def _wdiff_stats(text_a: str, text_b: str) -> dict:
    """Run wdiff -s on two texts and parse the statistics.

    Returns dict with keys for each file:
      'a': {'words': int, 'common': int, 'common_pct': float}
      'b': {'words': int, 'common': int, 'common_pct': float}
    """
    norm_a = _normalize_for_comparison(text_a)
    norm_b = _normalize_for_comparison(text_b)

    import tempfile
    a_path = None
    b_path = None
    try:
        a_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        a_path = a_file.name
        b_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        b_path = b_file.name
        a_file.write(norm_a)
        b_file.write(norm_b)
        a_file.close()
        b_file.close()

        result = subprocess.run(
            ["wdiff", "-s", a_path, b_path],
            capture_output=True, text=True
        )
        # wdiff puts stats on stdout (after the diff output)
        output = result.stdout + result.stderr

        # Parse stats lines: "/path/to/file: N words  M PP% common  ..."
        stats = {}
        for line in output.strip().split('\n'):
            if a_path in line:
                m = re.search(r'(\d+)\s+words\s+(\d+)\s+(\d+)%\s+common', line)
                if m:
                    stats['a'] = {
                        'words': int(m.group(1)),
                        'common': int(m.group(2)),
                        'common_pct': int(m.group(3))
                    }
            elif b_path in line:
                m = re.search(r'(\d+)\s+words\s+(\d+)\s+(\d+)%\s+common', line)
                if m:
                    stats['b'] = {
                        'words': int(m.group(1)),
                        'common': int(m.group(2)),
                        'common_pct': int(m.group(3))
                    }
        return stats
    finally:
        if a_path:
            os.unlink(a_path)
        if b_path:
            os.unlink(b_path)


def _strip_structured_headers(text: str) -> str:
    """Strip **Speaker** (timestamp) headers from structured merged text for comparison."""
    # Remove lines like "**Speaker Name** (HH:MM:SS)" or "**Speaker Name**"
    lines = text.split('\n')
    stripped = []
    for line in lines:
        if re.match(r'^\*\*[^*]+\*\*(\s*\(\d{1,2}:\d{2}:\d{2}\))?\s*$', line):
            continue
        stripped.append(line)
    return '\n'.join(stripped)


def analyze_source_survival(config: SpeechConfig, data: SpeechData) -> None:
    """Analyze how much of each source transcript survived into the merged output."""
    print("\n[6/6] Analyzing source survival...")

    merged_path = config.output_dir / "transcript_merged.txt"
    analysis_path = config.output_dir / "analysis.md"

    if not merged_path.exists():
        if config.dry_run:
            print(f"  [dry-run] Would analyze source survival → {analysis_path.name}")
        else:
            print("  No merged transcript found, skipping analysis")
        return

    # Gather source files for DAG check
    analysis_inputs = [merged_path]
    if data.transcript_path and data.transcript_path.exists():
        analysis_inputs.append(data.transcript_path)
    if data.captions_path and data.captions_path.exists():
        analysis_inputs.append(data.captions_path)
    if config.external_transcript and not config.external_transcript.startswith(("http://", "https://")):
        ext_path = Path(config.external_transcript)
        if ext_path.exists():
            analysis_inputs.append(ext_path)

    if config.skip_existing and is_up_to_date(analysis_path, *analysis_inputs):
        print(f"  Skipping: analysis up to date")
        return
    if _dry_run_skip(config, "analyze source survival", "analysis.md"):
        return

    # Read merged transcript
    with open(merged_path, 'r') as f:
        merged_text = f.read()

    # Strip structural headers for fair comparison
    merged_clean = _strip_structured_headers(merged_text)

    # Collect sources
    sources = []

    # Whisper (ensembled or single model)
    if data.transcript_path and data.transcript_path.exists():
        with open(data.transcript_path, 'r') as f:
            whisper_text = f.read()
        label = "ensembled" if "ensembled" in data.transcript_path.name else data.transcript_path.stem
        sources.append((f"Whisper ({label})", whisper_text))

    # YouTube captions
    if data.captions_path and data.captions_path.exists():
        youtube_text = clean_vtt_captions(data.captions_path)
        sources.append(("YouTube captions", youtube_text))

    # External transcript
    if config.external_transcript:
        external_text = None
        if config.external_transcript.startswith(("http://", "https://")):
            import urllib.request
            try:
                with urllib.request.urlopen(config.external_transcript) as response:
                    raw = response.read().decode('utf-8').strip()
                if '<html' in raw[:500].lower() or '<body' in raw[:1000].lower():
                    external_text = _extract_text_from_html(raw)
                else:
                    external_text = raw
            except Exception as e:
                print(f"  Warning: Could not fetch external transcript: {e}")
        else:
            ext_path = Path(config.external_transcript)
            if ext_path.exists():
                with open(ext_path, 'r') as f:
                    external_text = f.read().strip()
        if external_text:
            sources.append(("External transcript", external_text))

    if not sources:
        print("  No source transcripts found for comparison")
        return

    # Run wdiff comparison of merged vs each source
    merged_words = len(_normalize_for_comparison(merged_clean).split())
    results = []

    for name, source_text in sources:
        stats = _wdiff_stats(merged_clean, source_text)
        if 'a' in stats and 'b' in stats:
            results.append({
                'name': name,
                'source_words': stats['b']['words'],
                'common_from_merged': stats['a']['common'],
                'common_pct_of_merged': stats['a']['common_pct'],
                'common_from_source': stats['b']['common'],
                'common_pct_of_source': stats['b']['common_pct'],
            })

    if not results:
        print("  Could not compute wdiff statistics")
        return

    # Print summary
    print(f"\n  {'Source':<25} {'Words':>8} {'Common':>8} {'% of Merged':>12} {'% of Source':>12}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*12} {'-'*12}")
    for r in results:
        print(f"  {r['name']:<25} {r['source_words']:>8,} {r['common_from_merged']:>8,} "
              f"{r['common_pct_of_merged']:>11}% {r['common_pct_of_source']:>11}%")
    print(f"  {'Merged output':<25} {merged_words:>8,}")

    # Find most similar source
    best = max(results, key=lambda r: r['common_pct_of_merged'])
    print(f"\n  Merged output most closely resembles: {best['name']} "
          f"({best['common_pct_of_merged']}% overlap)")

    # Write analysis.md
    report = [f"# Source Survival Analysis", ""]
    report.append(f"Merged transcript: `{merged_path.name}` ({merged_words:,} words)")
    report.append("")
    report.append("| Source | Words | Common | % of Merged | % of Source |")
    report.append("|--------|------:|-------:|------------:|------------:|")
    for r in results:
        report.append(f"| {r['name']} | {r['source_words']:,} | {r['common_from_merged']:,} "
                      f"| {r['common_pct_of_merged']}% | {r['common_pct_of_source']}% |")
    report.append(f"| **Merged output** | **{merged_words:,}** | | | |")
    report.append("")
    report.append(f"**Most similar source:** {best['name']} ({best['common_pct_of_merged']}% of merged words in common)")
    report.append("")
    report.append("## Column definitions")
    report.append("")
    report.append("- **Words**: word count of the source (after normalization)")
    report.append("- **Common**: words shared between merged output and this source")
    report.append("- **% of Merged**: what percentage of the merged output's words appear in this source")
    report.append("- **% of Source**: what percentage of this source's words appear in the merged output")
    report.append("")

    with open(analysis_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"\n  Analysis saved: {analysis_path.name}")


def generate_markdown(config: SpeechConfig, data: SpeechData) -> None:
    """Generate markdown document with slides interleaved at correct timestamps."""
    print("\n[5/5] Generating markdown...")

    markdown_path = config.output_dir / "transcript.md"

    # Inputs: merged or whisper transcript, slide timestamps, slides JSON
    md_inputs = [p for p in [data.merged_transcript_path, data.transcript_path,
                             data.slides_json_path] if p]
    timestamps_file = config.output_dir / "slide_timestamps.json"
    if timestamps_file.exists():
        md_inputs.append(timestamps_file)
    if config.skip_existing and is_up_to_date(markdown_path, *md_inputs):
        print(f"  Reusing: {markdown_path.name}")
        data.markdown_path = markdown_path
        return
    if _dry_run_skip(config, "generate markdown", "transcript.md"):
        return

    # Determine if we can do timestamp-based placement
    has_timestamps = (
        data.transcript_segments and
        data.slide_timestamps and
        len(data.slide_timestamps) > 0
    )

    if has_timestamps:
        print("  Using timestamp-based slide placement")
        content = _generate_interleaved_markdown(data)
    else:
        print("  Using sequential layout (no timestamps available)")
        content = _generate_sequential_markdown(data)

    # Write markdown
    with open(markdown_path, 'w') as f:
        f.write(content)

    data.markdown_path = markdown_path
    print(f"  Markdown saved: {markdown_path.name}")


def _generate_interleaved_markdown(data: SpeechData) -> str:
    """Generate markdown with slides placed at their timestamp positions.

    Note: Interleaved mode uses Whisper segments for timestamps.
    For merged transcript content, use sequential mode.
    """
    lines = [
        f"# {data.title}",
        "",
        "---",
        "",
    ]

    # Note about source
    source_note = "Whisper transcript"
    if data.merged_transcript_path and data.merged_transcript_path.exists():
        source_note = "Whisper transcript (merged version available in transcript_merged.txt)"

    # Create a list of all events (segments and slides) sorted by timestamp
    events = []

    # Add transcript segments
    for seg in data.transcript_segments:
        events.append({
            "type": "text",
            "timestamp": seg["start"],
            "content": seg["text"]
        })

    # Add slides
    for slide_info in data.slide_timestamps:
        slide_idx = slide_info["slide_number"] - 1
        if slide_idx < len(data.slide_images):
            slide_path = data.slide_images[slide_idx]

            # Get slide metadata if available
            alt_text = f"Slide {slide_info['slide_number']}"
            if data.slide_metadata and slide_idx < len(data.slide_metadata):
                meta = data.slide_metadata[slide_idx]
                if meta.get("title"):
                    alt_text = meta["title"]

            events.append({
                "type": "slide",
                "timestamp": slide_info["timestamp"],
                "filename": slide_path.name,
                "alt_text": alt_text,
                "slide_number": slide_info["slide_number"]
            })

    # Sort all events by timestamp
    events.sort(key=lambda x: x["timestamp"])

    # Group consecutive text segments into paragraphs
    current_paragraph = []
    last_slide_time = -1

    for event in events:
        if event["type"] == "text":
            current_paragraph.append(event["content"])
        elif event["type"] == "slide":
            # Flush current paragraph before slide
            if current_paragraph:
                paragraph_text = " ".join(current_paragraph)
                # Break into sentences for readability
                lines.append(_format_paragraph(paragraph_text))
                lines.append("")
                current_paragraph = []

            # Add slide
            lines.append(f"![{event['alt_text']}](slides/{event['filename']})")
            lines.append("")
            last_slide_time = event["timestamp"]

    # Flush remaining paragraph
    if current_paragraph:
        paragraph_text = " ".join(current_paragraph)
        lines.append(_format_paragraph(paragraph_text))
        lines.append("")

    # Add footer
    lines.extend([
        "---",
        "",
        f"*Generated by speech_transcriber.py (source: {source_note})*",
    ])

    return '\n'.join(lines)


def _get_best_transcript_text(data: SpeechData) -> str:
    """Get the best available transcript text (merged > whisper)."""
    # Prefer merged transcript if available
    if data.merged_transcript_path and data.merged_transcript_path.exists():
        with open(data.merged_transcript_path, 'r') as f:
            return f.read()

    # Fall back to Whisper transcript
    if data.transcript_path and data.transcript_path.exists():
        with open(data.transcript_path, 'r') as f:
            return f.read()

    return ""


def _generate_sequential_markdown(data: SpeechData) -> str:
    """Generate markdown with slides in a gallery (fallback when no timestamps)."""
    lines = [
        f"# {data.title}",
        "",
        "---",
        "",
    ]

    # Add title slide if we have slides
    if data.slide_images:
        first_slide = data.slide_images[0]
        lines.extend([
            f"[![Title Slide](slides/{first_slide.name})](slides/{first_slide.name})",
            "",
            "---",
            "",
        ])

    # Add transcript
    lines.append("## Transcript")
    lines.append("")

    transcript_text = _get_best_transcript_text(data)
    if transcript_text:
        lines.append(transcript_text)
    lines.append("")

    # Add slides gallery if we have them
    if data.slide_images:
        lines.extend([
            "---",
            "",
            "## Slides",
            "",
        ])

        for i, slide_path in enumerate(data.slide_images):
            slide_info = ""
            if data.slide_metadata and i < len(data.slide_metadata):
                meta = data.slide_metadata[i]
                if meta.get("title"):
                    slide_info = f" - {meta['title']}"
                elif meta.get("description"):
                    slide_info = f" - {meta['description'][:50]}"

            lines.append(f"### Slide {i+1}{slide_info}")
            lines.append("")
            lines.append(f"[![Slide {i+1}](slides/{slide_path.name})](slides/{slide_path.name})")
            lines.append("")

    # Add footer
    source_note = "merged transcript" if data.merged_transcript_path else "Whisper transcript"
    lines.extend([
        "---",
        "",
        f"*Generated by speech_transcriber.py (source: {source_note})*",
    ])

    return '\n'.join(lines)


def _format_paragraph(text: str) -> str:
    """Format a paragraph of text, adding line breaks at sentence boundaries for readability."""
    # Simple sentence splitting - break on . ! ? followed by space and capital
    # This keeps the text readable in markdown
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Group into ~3 sentence chunks for readability
    chunks = []
    current_chunk = []
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= 3:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return "\n\n".join(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe speeches from video URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://youtube.com/watch?v=..."
  %(prog)s "https://youtube.com/watch?v=..." --analyze-slides
  %(prog)s "https://youtube.com/watch?v=..." --no-merge
  %(prog)s "https://youtube.com/watch?v=..." --whisper-models small,medium
  %(prog)s "https://youtube.com/watch?v=..." --no-slides --external-transcript URL
  %(prog)s "https://youtube.com/watch?v=..." -o my_speech --whisper-models small
        """
    )

    # Input
    input_group = parser.add_argument_group("input")
    input_group.add_argument("url", help="URL of the speech video")
    input_group.add_argument("--external-transcript",
                        help="Path or URL to an external transcript to include in the merge process")

    # Output
    output_group = parser.add_argument_group("output")
    output_group.add_argument("-o", "--output-dir",
                        help="Output directory (default: ./transcripts/<title>)")

    # Whisper
    whisper_group = parser.add_argument_group("whisper")
    whisper_group.add_argument("--whisper-models", default="small,medium",
                        help="Whisper model(s) to use, comma-separated (default: small,medium). "
                             "Options: tiny, base, small, medium, large. "
                             "Multiple models enables ensembling for better accuracy")

    # Slides
    slides_group = parser.add_argument_group("slides")
    slides_group.add_argument("--no-slides", action="store_true",
                        help="Skip slide extraction entirely (audio/transcript only)")
    slides_group.add_argument("--scene-threshold", type=float, default=0.1,
                        help="Scene detection threshold 0-1 (default: 0.1)")
    slides_group.add_argument("--analyze-slides", action="store_true",
                        help="Use Claude vision API to analyze slides (requires API key)")

    # API
    api_group = parser.add_argument_group("API")
    api_group.add_argument("--api-key",
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    api_group.add_argument("--claude-model", default="claude-sonnet-4-20250514",
                        help="Claude model for API calls (default: claude-sonnet-4-20250514)")
    api_group.add_argument("--no-api", action="store_true",
                        help="Skip all API-dependent features (slide analysis, merging, ensembling)")
    api_group.add_argument("--no-merge", action="store_true",
                        help="Skip merging YouTube captions with Whisper (merge is on by default)")

    # Pipeline control
    pipeline_group = parser.add_argument_group("pipeline")
    pipeline_group.add_argument("--force", action="store_true",
                        help="Re-download/process even if files already exist")
    pipeline_group.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without actually doing it")
    pipeline_group.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed command output")

    args = parser.parse_args()

    # Check dependencies
    print("Checking dependencies...")
    deps = check_dependencies()

    missing = []
    if not deps["yt-dlp"]:
        missing.append("yt-dlp (install with: pip install yt-dlp)")
    if not deps["ffmpeg"]:
        missing.append("ffmpeg (install with: brew install ffmpeg)")
    if not deps["mlx_whisper"] and not deps["whisper"]:
        missing.append("mlx-whisper or openai-whisper (install with: pip install mlx-whisper)")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        sys.exit(1)

    whisper_impl = "mlx-whisper" if deps["mlx_whisper"] else "openai-whisper"
    print(f"  yt-dlp: OK")
    print(f"  ffmpeg: OK")
    print(f"  whisper: OK ({whisper_impl})")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("./transcripts/speech")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse whisper models (comma-separated)
    whisper_models = [m.strip() for m in args.whisper_models.split(",")]
    valid_models = ["tiny", "base", "small", "medium", "large"]
    for m in whisper_models:
        if m not in valid_models:
            print(f"Invalid Whisper model: {m}")
            print(f"Valid options: {', '.join(valid_models)}")
            sys.exit(1)

    # Create config
    config = SpeechConfig(
        url=args.url,
        output_dir=output_dir,
        whisper_models=whisper_models,
        scene_threshold=args.scene_threshold,
        no_slides=args.no_slides,
        analyze_slides=args.analyze_slides,
        merge_sources=not args.no_merge,
        external_transcript=args.external_transcript,
        no_api=args.no_api,
        api_key=args.api_key,
        claude_model=args.claude_model,
        skip_existing=not args.force,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    data = SpeechData()

    # Validate external transcript source
    if config.external_transcript:
        if config.external_transcript.startswith(("http://", "https://")):
            import urllib.request
            try:
                req = urllib.request.Request(config.external_transcript, method='HEAD')
                urllib.request.urlopen(req)
                print(f"External transcript URL validated: {config.external_transcript}")
            except Exception as e:
                print(f"\nError: Cannot reach external transcript URL: {config.external_transcript}")
                print(f"  {e}")
                sys.exit(1)
        elif not Path(config.external_transcript).exists():
            print(f"\nError: External transcript file not found: {config.external_transcript}")
            sys.exit(1)

    # If external transcript is provided, ensure merge is enabled
    if config.external_transcript and not config.merge_sources:
        print("\nNote: --external-transcript implies merging; enabling merge.")
        config.merge_sources = True

    # Early API key check - fail fast if API features requested without key
    api_features_requested = config.analyze_slides or config.merge_sources or len(config.whisper_models) > 1
    if api_features_requested and not config.no_api:
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("\nError: API features requested but no API key found.")
            print("\nYou requested:")
            if config.analyze_slides:
                print("  --analyze-slides (requires API)")
            if config.merge_sources:
                print("  --merge-sources [on by default] (requires API)")
            if len(config.whisper_models) > 1:
                print(f"  --whisper-models {','.join(config.whisper_models)} (ensembling requires API)")
            print("\nOptions:")
            print("  1. Set ANTHROPIC_API_KEY environment variable")
            print("  2. Use --api-key YOUR_KEY")
            print("  3. Add --no-api to skip API-dependent features")
            sys.exit(1)

    print(f"\nProcessing: {args.url}")
    print(f"Output directory: {output_dir}")

    # Fetch video duration for cost estimation
    estimated_words = 6000  # fallback
    try:
        info_result = run_command(
            ["yt-dlp", "--dump-json", config.url],
            "fetching video info for cost estimate",
            config.verbose
        )
        video_info = json.loads(info_result.stdout)
        duration_secs = video_info.get("duration", 0)
        if duration_secs:
            # ~2.5 words/second is typical conversational speech
            estimated_words = int(duration_secs * 2.5)
            duration_min = duration_secs / 60
            print(f"  Duration: {duration_min:.0f} min → ~{estimated_words:,} words estimated")
    except Exception:
        pass  # Fall back to default estimate

    # Print cost estimate if API features are enabled
    if api_features_requested and not config.no_api:
        print_cost_estimate(config, transcript_words=estimated_words)

    if config.dry_run:
        print("\n" + "="*50)
        print("DRY RUN - No actions will be taken")
        print("="*50)

    try:
        # Run pipeline — each stage skips if output already exists
        download_media(config, data)

        # Update output dir with actual title
        if data.title and not args.output_dir:
            safe_title = re.sub(r'[^\w\s-]', '', data.title)[:50].strip()
            safe_title = re.sub(r'\s+', '-', safe_title).lower()
            new_output_dir = Path(f"./transcripts/{safe_title}")
            if output_dir != new_output_dir and not new_output_dir.exists():
                output_dir.rename(new_output_dir)
                config.output_dir = new_output_dir
                # Update paths in data
                if data.audio_path:
                    data.audio_path = new_output_dir / data.audio_path.name
                if data.video_path:
                    data.video_path = new_output_dir / data.video_path.name
                if data.captions_path:
                    data.captions_path = new_output_dir / data.captions_path.name

        transcribe_audio(config, data)

        if not config.no_slides:
            extract_slides(config, data)

            if config.analyze_slides:
                analyze_slides_with_vision(config, data)
            else:
                create_basic_slides_json(config, data)

        # Merge transcript sources if requested
        merge_transcript_sources(config, data)

        generate_markdown(config, data)
        analyze_source_survival(config, data)

        # Summary
        print("\n" + "="*50)
        print("COMPLETE!")
        print("="*50)
        print(f"\nOutput directory: {config.output_dir}")
        print(f"\nGenerated files:")
        if data.audio_path and data.audio_path.exists():
            print(f"  - {data.audio_path.name} (audio)")
        if data.video_path and data.video_path.exists():
            print(f"  - {data.video_path.name} (video)")
        if data.captions_path and data.captions_path.exists():
            print(f"  - {data.captions_path.name} (captions)")
        # Show individual model transcripts if multiple
        if len(data.whisper_transcripts) > 1:
            for model, paths in data.whisper_transcripts.items():
                if paths.get("txt") and paths["txt"].exists():
                    print(f"  - {paths['txt'].name} (Whisper {model})")
        if data.transcript_path and data.transcript_path.exists():
            label = "ensembled transcript" if len(data.whisper_transcripts) > 1 else "transcript"
            print(f"  - {data.transcript_path.name} ({label})")
        if data.transcript_json_path and data.transcript_json_path.exists():
            print(f"  - {data.transcript_json_path.name} (transcript with timestamps)")
        if data.merged_transcript_path and data.merged_transcript_path.exists():
            print(f"  - {data.merged_transcript_path.name} (merged from YouTube + Whisper)")
        if data.slides_dir and data.slides_dir.exists():
            print(f"  - slides/ ({len(data.slide_images)} images)")
        timestamps_file = config.output_dir / "slide_timestamps.json"
        if timestamps_file.exists():
            print(f"  - slide_timestamps.json")
        if data.slides_json_path and data.slides_json_path.exists():
            print(f"  - {data.slides_json_path.name}")
        if data.markdown_path and data.markdown_path.exists():
            print(f"  - {data.markdown_path.name}")
        analysis_file = config.output_dir / "analysis.md"
        if analysis_file.exists():
            print(f"  - {analysis_file.name} (source survival analysis)")

        if not config.no_slides and not config.analyze_slides and data.slide_images:
            print("\nTip: Run with --analyze-slides to get detailed slide descriptions")

        if (not config.merge_sources or config.no_api) and data.captions_path and data.captions_path.exists() and not data.merged_transcript_path:
            print("Tip: YouTube captions available - run without --no-merge/--no-api to create a 'critical text'")

    except Exception as e:
        print(f"\nError: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
