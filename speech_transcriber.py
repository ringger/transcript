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
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import shutil


@dataclass
class SpeechConfig:
    """Configuration for speech transcription pipeline."""
    url: str
    output_dir: Path
    whisper_models: list = field(default_factory=lambda: ["medium"])  # Can be multiple models
    scene_threshold: float = 0.1
    analyze_slides: bool = False
    merge_sources: bool = True  # Merge YouTube captions with Whisper (default: on)
    no_api: bool = False  # Skip all API-dependent features
    api_key: Optional[str] = None
    skip_existing: bool = True
    reextract_slides: bool = False  # Force re-extraction of slides only
    dry_run: bool = False  # Show what would be done without doing it
    verbose: bool = False


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

    Pricing estimates (as of 2024, Claude Sonnet):
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
        # Merge requires sending both transcripts + getting merged output
        # Input: ~2x transcript, Output: ~1x transcript
        input_cost = (transcript_tokens * 2) * 0.003 / 1000
        output_cost = transcript_tokens * 0.015 / 1000
        merge_cost = input_cost + output_cost
        costs["merge_sources"] = merge_cost
        costs["details"].append(f"Source merging: ~{transcript_words*2} input words + {transcript_words} output = ${merge_cost:.2f}")

    if len(config.whisper_models) > 1 and not config.no_api:
        # Ensemble requires comparing transcripts
        input_cost = (transcript_tokens * len(config.whisper_models)) * 0.003 / 1000
        output_cost = transcript_tokens * 0.015 / 1000
        ensemble_cost = input_cost + output_cost
        costs["ensemble_whisper"] = ensemble_cost
        costs["details"].append(f"Whisper ensemble: {len(config.whisper_models)} models = ${ensemble_cost:.2f}")

    costs["total"] = costs["analyze_slides"] + costs["merge_sources"] + costs["ensemble_whisper"]

    return costs


def _print_dry_run(config: SpeechConfig) -> None:
    """Print what would be done without actually doing it."""
    print("\n" + "="*50)
    print("DRY RUN - No actions will be taken")
    print("="*50)

    if config.reextract_slides:
        print("\n[Reextract mode]")
        print("  1. Load existing video and transcript files")
        print(f"  2. Delete existing slides in {config.output_dir}/slides/")
        print(f"  3. Extract new slides with threshold={config.scene_threshold}")
        print("  4. Regenerate transcript.md with new slides")
    else:
        print("\n[Full pipeline]")
        print("  1. Download audio, video, captions from URL")
        print(f"  2. Transcribe with Whisper model(s): {', '.join(config.whisper_models)}")
        if len(config.whisper_models) > 1:
            if config.no_api:
                print("     - Ensemble: will use largest model (no API)")
            else:
                print("     - Ensemble: will use Claude to resolve differences")
        print(f"  3. Extract slides with scene threshold={config.scene_threshold}")

        if config.analyze_slides:
            if config.no_api:
                print("  4. Analyze slides: SKIPPED (--no-api)")
            else:
                print("  4. Analyze slides with Claude Vision API")
        else:
            print("  4. Create basic slides JSON (no analysis)")

        if config.merge_sources:
            if config.no_api:
                print("  4b. Merge sources: SKIPPED (--no-api)")
            else:
                print("  4b. Merge YouTube + Whisper transcripts with Claude (default: on)")
        else:
            print("  4b. Merge sources: SKIPPED (--no-merge)")

        print("  5. Generate markdown with slides at timestamps")

    print(f"\nOutput directory: {config.output_dir}")

    # Show what files would be created
    print("\nFiles that would be created:")
    if not config.reextract_slides:
        print("  - <title>.mp3 (audio)")
        print("  - <title>.mp4 (video)")
        print("  - <title>.en.vtt (captions, if available)")
        for model in config.whisper_models:
            print(f"  - <title>_{model}.txt (Whisper transcript)")
            print(f"  - <title>_{model}.json (with timestamps)")
        if len(config.whisper_models) > 1:
            print("  - <title>_ensembled.txt (combined transcript)")
        if config.merge_sources and not config.no_api:
            print("  - transcript_merged.txt (critical text)")

    print("  - slides/*.png (extracted frames)")
    print("  - slide_timestamps.json")
    if config.analyze_slides and not config.no_api:
        print("  - slides_transcript.json (with descriptions)")
    else:
        print("  - slides_basic.json")
    print("  - transcript.md (final output)")

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
    safe_title = re.sub(r'[^\w\s-]', '', data.title)[:50].strip()

    print(f"  Title: {data.title}")

    # Download audio
    audio_path = config.output_dir / f"{safe_title}.mp3"
    if config.skip_existing and audio_path.exists():
        print(f"  Audio exists, skipping: {audio_path.name}")
    else:
        print("  Downloading audio...")
        run_command(
            ["yt-dlp", "-x", "--audio-format", "mp3",
             "-o", str(config.output_dir / f"{safe_title}.%(ext)s"), config.url],
            "downloading audio",
            config.verbose
        )
    data.audio_path = audio_path

    # Download video
    video_path = config.output_dir / f"{safe_title}.mp4"
    if config.skip_existing and video_path.exists():
        print(f"  Video exists, skipping: {video_path.name}")
    else:
        print("  Downloading video...")
        run_command(
            ["yt-dlp", "-f", "mp4",
             "-o", str(config.output_dir / f"{safe_title}.%(ext)s"), config.url],
            "downloading video",
            config.verbose
        )
    data.video_path = video_path

    # Download captions if available
    captions_path = config.output_dir / f"{safe_title}.en.vtt"
    if config.skip_existing and captions_path.exists():
        print(f"  Captions exist, skipping: {captions_path.name}")
    else:
        print("  Downloading captions (if available)...")
        try:
            run_command(
                ["yt-dlp", "--write-auto-sub", "--sub-lang", "en", "--skip-download",
                 "-o", str(config.output_dir / f"{safe_title}.%(ext)s"), config.url],
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
    base_name = data.audio_path.stem
    txt_path = config.output_dir / f"{base_name}_{model}.txt"
    json_path = config.output_dir / f"{base_name}_{model}.json"

    # Check if already exists
    if config.skip_existing and txt_path.exists():
        print(f"  {model} transcript exists, skipping: {txt_path.name}")
        data.whisper_transcripts[model] = {"txt": txt_path, "json": json_path if json_path.exists() else None}
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

        # mlx_whisper names output after input file, so rename to include model
        default_txt = config.output_dir / f"{base_name}.txt"
        default_json = config.output_dir / f"{base_name}.json"

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
        diffs = _analyze_differences_wdiff(other_text, base_text, config)
        if diffs:
            print(f"  Found {len(diffs)} differences between {other_model} and {base_model}")
            for d in diffs:
                d["source_model"] = other_model
            all_differences.extend(diffs)

    # If we have differences and API key, use Claude to resolve
    if all_differences:
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if config.no_api:
            print("  --no-api flag set - using base model without resolving differences")
            ensembled_text = base_text
        elif api_key:
            ensembled_text = _resolve_whisper_differences(
                api_key, base_text, transcripts, all_differences, config.verbose
            )
        else:
            print("  No API key - using base model without resolving differences")
            ensembled_text = base_text
    else:
        print("  No significant differences found between models")
        ensembled_text = base_text

    # Save ensembled transcript
    base_name = data.audio_path.stem
    ensembled_path = config.output_dir / f"{base_name}_ensembled.txt"
    with open(ensembled_path, 'w') as f:
        f.write(ensembled_text)

    data.transcript_path = ensembled_path
    print(f"  Ensembled transcript saved: {ensembled_path.name}")

    # Use the largest model's JSON for timestamps
    if base_model in data.whisper_transcripts:
        data.transcript_json_path = data.whisper_transcripts[base_model].get("json")


def _resolve_whisper_differences(api_key: str, base_text: str, all_transcripts: dict,
                                  differences: list, verbose: bool = False) -> str:
    """Use Claude to resolve differences between Whisper model outputs."""
    import anthropic

    # Format differences for Claude
    diff_summary = "DIFFERENCES BETWEEN WHISPER MODELS:\n"
    for i, d in enumerate(differences[:30], 1):  # Limit to 30
        source = d.get("source_model", "other")
        if d["type"] == "changed":
            diff_summary += f"{i}. {source} has \"{d.get('youtube', d.get('whisper', ''))}\" vs base has \"{d.get('whisper', d.get('youtube', ''))}\"\n"
        elif d["type"] == "youtube_only":
            diff_summary += f"{i}. {source} has: \"{d['text']}\" (missing in base)\n"
        elif d["type"] == "whisper_only":
            diff_summary += f"{i}. Base has: \"{d['text']}\" (missing in {source})\n"

    # Format other transcripts for reference
    other_transcripts_text = ""
    for model, text in all_transcripts.items():
        if model != "ensembled":
            other_transcripts_text += f"\n--- {model.upper()} MODEL ---\n{text[:3000]}...\n"

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

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
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
        raise FileNotFoundError(f"Video file not found: {data.video_path}")

    slides_dir = config.output_dir / "slides"
    slides_dir.mkdir(exist_ok=True)
    data.slides_dir = slides_dir

    timestamps_file = config.output_dir / "slide_timestamps.json"

    # If reextract mode, clear existing slides first
    if config.reextract_slides:
        existing_slides = list(slides_dir.glob("slide_*.png"))
        if existing_slides:
            print(f"  Clearing {len(existing_slides)} existing slides for re-extraction...")
            for slide in existing_slides:
                slide.unlink()

    existing_slides = list(slides_dir.glob("slide_*.png"))
    if config.skip_existing and existing_slides and timestamps_file.exists() and not config.reextract_slides:
        print(f"  Slides exist ({len(existing_slides)} files), skipping extraction")
        data.slide_images = sorted(existing_slides)
        # Load existing timestamps
        _load_slide_timestamps(data, timestamps_file)
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


def _load_existing_data(config: SpeechConfig, data: SpeechData) -> None:
    """Load existing media and transcript files for reextract mode."""
    # Find video file
    video_files = list(config.output_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video file found in {config.output_dir}. "
                               "Run without --reextract-slides first to download media.")
    data.video_path = video_files[0]

    # Find audio file
    audio_files = list(config.output_dir.glob("*.mp3"))
    if audio_files:
        data.audio_path = audio_files[0]

    # Find captions
    caption_files = list(config.output_dir.glob("*.vtt"))
    if caption_files:
        data.captions_path = caption_files[0]

    # Extract title from filename
    data.title = data.video_path.stem

    # Find transcript files
    # Prefer ensembled, then medium, then small, then any
    for pattern in ["*_ensembled.txt", "*_medium.txt", "*_small.txt", "*.txt"]:
        txt_files = list(config.output_dir.glob(pattern))
        # Filter out merged transcript
        txt_files = [f for f in txt_files if "merged" not in f.name]
        if txt_files:
            data.transcript_path = txt_files[0]
            break

    # Find JSON with timestamps
    for pattern in ["*_medium.json", "*_small.json", "*.json"]:
        json_files = list(config.output_dir.glob(pattern))
        # Filter out slide-related JSON files
        json_files = [f for f in json_files if "slide" not in f.name and "basic" not in f.name]
        if json_files:
            data.transcript_json_path = json_files[0]
            break

    # Load transcript segments if we have JSON
    if data.transcript_json_path and data.transcript_json_path.exists():
        _load_transcript_segments(data)

    print(f"  Found video: {data.video_path.name}")
    if data.transcript_path:
        print(f"  Found transcript: {data.transcript_path.name}")
    if data.transcript_json_path:
        print(f"  Found timestamps: {data.transcript_json_path.name}")


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
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
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
    if data.slides_json_path and data.slides_json_path.exists():
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

    # Check if we have both sources
    if not data.captions_path or not data.captions_path.exists():
        print("  No YouTube captions available, skipping merge")
        return

    if not data.transcript_path or not data.transcript_path.exists():
        print("  No Whisper transcript available, skipping merge")
        return

    merged_path = config.output_dir / "transcript_merged.txt"

    if config.skip_existing and merged_path.exists():
        print(f"  Merged transcript exists, skipping: {merged_path.name}")
        data.merged_transcript_path = merged_path
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

    # Load both transcripts
    youtube_text = clean_vtt_captions(data.captions_path)
    with open(data.transcript_path, 'r') as f:
        whisper_text = f.read()

    print(f"  YouTube captions: {len(youtube_text.split())} words")
    print(f"  Whisper transcript: {len(whisper_text.split())} words")

    # First, use wdiff to identify differences (if available)
    differences = _analyze_differences_wdiff(youtube_text, whisper_text, config)

    if differences:
        print(f"  Found {len(differences)} content differences via wdiff")
        # Use focused merge that only resolves the differences
        merged_text = _merge_with_differences(api_key, whisper_text, youtube_text, differences, config.verbose)
    else:
        print("  wdiff not available, using full comparison")
        # Fall back to full comparison
        merged_text = _merge_transcript_full(api_key, youtube_text, whisper_text, config.verbose)

    # Save merged transcript
    with open(merged_path, 'w') as f:
        f.write(merged_text)

    data.merged_transcript_path = merged_path
    print(f"  Merged transcript saved: {merged_path.name}")
    print(f"  Merged word count: {len(merged_text.split())} words")


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation."""
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def _analyze_differences_wdiff(youtube_text: str, whisper_text: str, config: SpeechConfig) -> list:
    """Use wdiff to identify content differences between transcripts."""

    # Check if wdiff is available
    if not shutil.which("wdiff"):
        if config.verbose:
            print("  wdiff not found, skipping diff analysis")
        return []

    import tempfile

    # Normalize both texts for comparison
    yt_normalized = _normalize_for_comparison(youtube_text)
    wh_normalized = _normalize_for_comparison(whisper_text)

    # Write normalized texts to temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
        f1.write(yt_normalized)
        yt_file = f1.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
        f2.write(wh_normalized)
        wh_file = f2.name

    try:
        # Run wdiff
        result = subprocess.run(
            ["wdiff", "-s", yt_file, wh_file],
            capture_output=True,
            text=True
        )

        # Parse wdiff statistics output
        # Format: "file1: X words  Y common  Z deleted  W changed"
        differences = []

        # Also run wdiff without -s to get actual differences
        diff_result = subprocess.run(
            ["wdiff", yt_file, wh_file],
            capture_output=True,
            text=True
        )

        # Parse differences - look for [-deleted-] and {+inserted+}
        diff_output = diff_result.stdout

        # Find all changes
        # [-word-] means deleted from file1 (youtube)
        # {+word+} means added in file2 (whisper)
        deleted = re.findall(r'\[-([^\]]+)-\]', diff_output)
        inserted = re.findall(r'\{\+([^\}]+)\+\}', diff_output)

        # Create a list of meaningful differences
        for d in deleted:
            if len(d.split()) <= 5:  # Single words or short phrases
                differences.append({"type": "youtube_only", "text": d})

        for i in inserted:
            if len(i.split()) <= 5:
                differences.append({"type": "whisper_only", "text": i})

        # Find changed pairs (adjacent delete/insert)
        changes = re.findall(r'\[-([^\]]+)-\]\s*\{\+([^\}]+)\+\}', diff_output)
        for yt_ver, wh_ver in changes:
            differences.append({
                "type": "changed",
                "youtube": yt_ver,
                "whisper": wh_ver
            })

        if config.verbose:
            print(f"  wdiff stats: {result.stderr.strip()}")

        return differences

    finally:
        # Clean up temp files
        os.unlink(yt_file)
        os.unlink(wh_file)


def _merge_with_differences(api_key: str, whisper_text: str, youtube_text: str,
                            differences: list, verbose: bool = False) -> str:
    """Merge transcripts by resolving specific differences identified by wdiff."""
    import anthropic

    # Filter to meaningful differences (skip very common words)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'it'}
    meaningful_diffs = []
    for d in differences:
        if d["type"] == "changed":
            yt_words = set(d["youtube"].lower().split())
            wh_words = set(d["whisper"].lower().split())
            # Keep if not just common words
            if not (yt_words <= common_words and wh_words <= common_words):
                meaningful_diffs.append(d)
        else:
            text = d.get("text", "").lower()
            if text and text not in common_words:
                meaningful_diffs.append(d)

    if not meaningful_diffs:
        print("  No meaningful differences found, using Whisper transcript")
        return whisper_text

    # Format differences for Claude
    diff_text = "IDENTIFIED DIFFERENCES (normalized, no punctuation):\n"
    for i, d in enumerate(meaningful_diffs[:50], 1):  # Limit to 50 most important
        if d["type"] == "changed":
            diff_text += f"{i}. YouTube: \"{d['youtube']}\" vs Whisper: \"{d['whisper']}\"\n"
        elif d["type"] == "youtube_only":
            diff_text += f"{i}. YouTube has: \"{d['text']}\" (missing in Whisper)\n"
        elif d["type"] == "whisper_only":
            diff_text += f"{i}. Whisper has: \"{d['text']}\" (missing in YouTube)\n"

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are creating a "critical text" by merging two transcripts of the same speech.

BASE TRANSCRIPT (Whisper - use this for structure and punctuation):
---
{whisper_text}
---

REFERENCE TRANSCRIPT (YouTube - often has better proper nouns):
---
{youtube_text}
---

{diff_text}

INSTRUCTIONS:
1. Start with the Whisper transcript as your base (it has better punctuation)
2. Review each difference listed above and apply fixes where YouTube is more accurate:
   - Prefer YouTube for proper nouns, names, technical terms, scientific vocabulary
   - Prefer YouTube if it has content that Whisper missed entirely
   - Keep Whisper's version if the difference is just formatting/punctuation
3. Output the COMPLETE merged transcript with corrections applied
4. Do NOT add commentary or notes - output ONLY the transcript text

Output the merged transcript:"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16384,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text.strip()


def _merge_transcript_full(api_key: str, youtube_text: str, whisper_text: str, verbose: bool = False) -> str:
    """Full transcript merge when wdiff is not available."""
    # Split into chunks if too long
    max_chunk_words = 3000

    whisper_words = whisper_text.split()
    youtube_words = youtube_text.split()

    if len(whisper_words) <= max_chunk_words:
        return _merge_transcript_chunk(api_key, youtube_text, whisper_text, verbose)

    # Process in chunks
    merged_chunks = []
    num_chunks = (len(whisper_words) // max_chunk_words) + 1

    for i in range(num_chunks):
        start_idx = i * max_chunk_words
        end_idx = min((i + 1) * max_chunk_words, len(whisper_words))

        whisper_chunk = " ".join(whisper_words[start_idx:end_idx])

        yt_start = int(start_idx * len(youtube_words) / len(whisper_words))
        yt_end = int(end_idx * len(youtube_words) / len(whisper_words))
        youtube_chunk = " ".join(youtube_words[yt_start:yt_end])

        chunk_merged = _merge_transcript_chunk(api_key, youtube_chunk, whisper_chunk, verbose)
        merged_chunks.append(chunk_merged)

    return "\n\n".join(merged_chunks)


def _merge_transcript_chunk(api_key: str, youtube_text: str, whisper_text: str, verbose: bool = False) -> str:
    """Use Claude to merge a chunk of YouTube captions with Whisper transcript."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are creating a "critical text" by merging two transcripts of the same speech.
Your goal is to produce the most accurate transcript by combining the best elements of each source.

SOURCE 1 - YouTube Captions (often human-reviewed, better for proper nouns, names, technical terms):
---
{youtube_text}
---

SOURCE 2 - Whisper AI Transcript (better punctuation, sentence structure, formatting):
---
{whisper_text}
---

INSTRUCTIONS:
1. Use Whisper as the base for structure and punctuation
2. Prefer YouTube's version for:
   - Proper nouns and names (people, places, institutions)
   - Technical terms and scientific vocabulary
   - Specific numbers and dates
   - Any content that appears in YouTube but is missing from Whisper
3. When the sources disagree on a word:
   - If YouTube has a proper noun and Whisper has a common word, prefer YouTube
   - If one version makes more grammatical/contextual sense, prefer that one
4. Do NOT add commentary, headers, or notes - output ONLY the merged transcript text
5. Maintain natural paragraph breaks for readability

Output the merged transcript:"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text.strip()


def generate_markdown(config: SpeechConfig, data: SpeechData) -> None:
    """Generate markdown document with slides interleaved at correct timestamps."""
    print("\n[5/5] Generating markdown...")

    markdown_path = config.output_dir / "transcript.md"

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
  %(prog)s "https://youtube.com/watch?v=..." --no-api
  %(prog)s "https://youtube.com/watch?v=..." -o my_speech --whisper-models small
        """
    )

    parser.add_argument("url", help="URL of the speech video")
    parser.add_argument("-o", "--output-dir",
                        help="Output directory (default: ./transcripts/<title>)")
    parser.add_argument("--whisper-models", default="medium",
                        help="Whisper model(s) to use, comma-separated (default: medium). "
                             "Options: tiny, base, small, medium, large. "
                             "Example: --whisper-models small,medium for ensembling")
    parser.add_argument("--scene-threshold", type=float, default=0.1,
                        help="Scene detection threshold 0-1 (default: 0.1)")
    parser.add_argument("--analyze-slides", action="store_true",
                        help="Use Claude vision API to analyze slides (requires API key)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merging YouTube captions with Whisper (merge is on by default)")
    parser.add_argument("--api-key",
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--no-api", action="store_true",
                        help="Skip all API-dependent features (slide analysis, merging, ensembling)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-download/process even if files exist")
    parser.add_argument("--reextract-slides", action="store_true",
                        help="Force re-extraction of slides with current threshold (skips download/transcription)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without actually doing it")
    parser.add_argument("-v", "--verbose", action="store_true",
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
        analyze_slides=args.analyze_slides,
        merge_sources=not args.no_merge,
        no_api=args.no_api,
        api_key=args.api_key,
        skip_existing=not args.no_skip,
        reextract_slides=args.reextract_slides,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    data = SpeechData()

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

    # Print cost estimate if API features are enabled
    if api_features_requested and not config.no_api:
        print_cost_estimate(config)

    # Dry run mode - show what would be done
    if config.dry_run:
        _print_dry_run(config)
        return

    try:
        # Run pipeline
        if config.reextract_slides:
            # Reextract mode: skip download and transcription, load existing data
            print("\n[Reextract mode] Skipping download and transcription...")
            _load_existing_data(config, data)
        else:
            download_media(config, data)

            # Update output dir with actual title
            if data.title and not args.output_dir:
                safe_title = re.sub(r'[^\w\s-]', '', data.title)[:50].strip()
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

        extract_slides(config, data)

        if config.analyze_slides:
            analyze_slides_with_vision(config, data)
        else:
            create_basic_slides_json(config, data)

        # Merge transcript sources if requested
        merge_transcript_sources(config, data)

        generate_markdown(config, data)

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

        if not config.analyze_slides and data.slide_images:
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
