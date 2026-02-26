"""
Slides module for the speech transcription pipeline.

Handles slide extraction from video via ffmpeg scene detection,
vision-based slide analysis, and basic slide metadata generation.
"""

import base64
import json
import re
import subprocess
from pathlib import Path

from transcribe_critic.shared import (
    tprint as print,
    SpeechConfig, SpeechData, is_up_to_date,
    SLIDE_TIMESTAMPS_JSON, SLIDES_TRANSCRIPT_JSON,
    create_llm_client, llm_call_with_retry,
    _save_json, _print_reusing, _dry_run_skip, _should_skip,
)


def extract_slides(config: SpeechConfig, data: SpeechData) -> None:
    """Extract slides from video using ffmpeg scene detection, capturing timestamps."""
    print()
    print("[slides] Extracting slides...")

    if not data.video_path or not data.video_path.exists():
        print("  No video file available, skipping slide extraction")
        return

    slides_dir = config.output_dir / "slides"
    slides_dir.mkdir(exist_ok=True)
    data.slides_dir = slides_dir

    timestamps_file = config.output_dir / SLIDE_TIMESTAMPS_JSON

    existing_slides = list(slides_dir.glob("slide_*.png"))
    if existing_slides and _should_skip(config, timestamps_file, "extract slides from video", data.video_path):
        data.slide_images = sorted(existing_slides)
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
    _save_json(timestamps_file, data.slide_timestamps)

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
    print()
    print("[slides] Analyzing slides with vision API...")

    if not config.analyze_slides:
        print("  Skipped (use --analyze-slides to enable)")
        return

    if config.no_llm:
        print("  Skipped (--no-llm flag set)")
        return

    if not data.slide_images:
        print("  No slides to analyze")
        return

    slides_json_path = config.output_dir / SLIDES_TRANSCRIPT_JSON
    if _should_skip(config, slides_json_path, "analyze slides with vision LLM", *data.slide_images):
        if slides_json_path.exists():
            with open(slides_json_path, 'r') as f:
                slides_data = json.load(f)
            data.slide_metadata = slides_data.get("slides", [])
            data.slides_json_path = slides_json_path
        return

    client = create_llm_client(config)

    slides_metadata = []

    for i, slide_path in enumerate(data.slide_images):
        print(f"  Analyzing slide {i+1}/{len(data.slide_images)}: {slide_path.name}")

        # Read and encode image
        with open(slide_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Call vision LLM
        message = llm_call_with_retry(
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
    slides_json_path = config.output_dir / SLIDES_TRANSCRIPT_JSON
    _save_json(slides_json_path, {
        "title": data.title,
        "slide_count": len(slides_metadata),
        "slides": slides_metadata
    })

    data.slides_json_path = slides_json_path
    print(f"  Slides JSON saved: {slides_json_path.name}")


def create_basic_slides_json(config: SpeechConfig, data: SpeechData) -> None:
    """Create a basic slides JSON without vision analysis."""
    slides_json_path = config.output_dir / "slides_basic.json"
    if _should_skip(config, slides_json_path, "create basic slides JSON", *data.slide_images):
        if slides_json_path.exists():
            data.slides_json_path = slides_json_path
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
    _save_json(slides_json_path, {
        "title": data.title,
        "slide_count": len(slides_metadata),
        "note": "Basic metadata only - run with --analyze-slides for full analysis",
        "slides": slides_metadata
    })

    data.slides_json_path = slides_json_path
