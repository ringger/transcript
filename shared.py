"""
Shared types and utilities for the speech transcription pipeline.

Contains SpeechConfig, SpeechData, and utility functions used by both
transcriber.py and merge.py.
"""

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


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
