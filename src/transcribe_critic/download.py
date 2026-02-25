"""
Download module for the speech transcription pipeline.

Handles downloading audio, video, and captions from video URLs using yt-dlp.
"""

import json
import re
import subprocess
from pathlib import Path

from transcribe_critic.shared import (
    tprint as print,
    SpeechConfig, SpeechData,
    AUDIO_MP3, METADATA_JSON, CAPTIONS_VTT,
    run_command, _save_json, _print_reusing, _dry_run_skip,
)


def download_media(config: SpeechConfig, data: SpeechData, info: dict = None) -> None:
    """Download audio, video, and captions using yt-dlp."""
    print()
    print("[1] Downloading media...")

    output_template = str(config.output_dir / "%(title)s.%(ext)s")

    # Get video info first to extract title (skip if pre-fetched or cached)
    metadata_path = config.output_dir / METADATA_JSON
    if info is None and config.skip_existing and metadata_path.exists():
        # Reuse cached metadata (avoids yt-dlp call on re-runs and eval)
        with open(metadata_path) as f:
            info = json.load(f)
        print("  Reusing cached metadata")
    elif info is None:
        print("  Fetching video info...")
        result = run_command(
            ["yt-dlp", "--dump-json", config.url],
            "fetching video info",
            config.verbose
        )
        info = json.loads(result.stdout)
    data.title = info.get("title", "speech")

    print(f"  Title: {data.title}")

    # Save source metadata (metadata_path already set above)
    if config.dry_run:
        print(f"  [dry-run] Would save metadata → {metadata_path.name}")
    elif not metadata_path.exists() or not config.skip_existing:
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
        _save_json(metadata_path, metadata)
        data.metadata = metadata
        print(f"  Metadata saved: {metadata_path.name}")
    elif metadata_path.exists():
        with open(metadata_path) as f:
            data.metadata = json.load(f)

    # Download audio
    audio_path = config.output_dir / AUDIO_MP3
    if config.skip_existing and audio_path.exists():
        _print_reusing(audio_path.name)
    elif not _dry_run_skip(config, "download audio", AUDIO_MP3):
        print("  Downloading audio...")
        run_command(
            ["yt-dlp", "-x", "--audio-format", "mp3",
             "-o", str(config.output_dir / "audio.%(ext)s"), config.url],
            "downloading audio",
            config.verbose
        )
    data.audio_path = audio_path

    # Download video (only needed for slide extraction)
    if config.podcast or config.no_slides:
        print("  Skipping video download (--podcast)" if config.podcast
              else "  Skipping video download (--no-slides)")
    else:
        video_path = config.output_dir / "video.mp4"
        if config.skip_existing and video_path.exists():
            _print_reusing(video_path.name)
        elif not _dry_run_skip(config, "download video", "video.mp4"):
            print("  Downloading video...")
            run_command(
                ["yt-dlp", "-f", "mp4",
                 "-o", str(config.output_dir / "video.%(ext)s"), config.url],
                "downloading video",
                config.verbose
            )
        data.video_path = video_path

    # Download captions if available (skip for podcasts — no captions)
    captions_path = config.output_dir / CAPTIONS_VTT
    if config.podcast:
        print("  Skipping captions download (--podcast)")
    elif config.skip_existing and captions_path.exists():
        _print_reusing(captions_path.name)
    elif not _dry_run_skip(config, "download captions", CAPTIONS_VTT):
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
