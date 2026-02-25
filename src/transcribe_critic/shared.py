"""
Shared types and utilities for the speech transcription pipeline.

Contains SpeechConfig, SpeechData, and utility functions used by
transcriber.py, merge.py, and all pipeline stage modules.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import builtins


def tprint(*args, **kwargs):
    """Print with [HH:MM:SS] timestamp prefix.

    Skips the timestamp for carriage-return progress lines (end != newline)
    so that in-place progress updates remain clean.
    """
    if kwargs.get("end", "\n") != "\n":
        builtins.print(*args, flush=True, **kwargs)
        return
    stamp = time.strftime("[%H:%M:%S]")
    builtins.print(stamp, *args, flush=True, **kwargs)


print = tprint

# Whisper model sizes in descending quality order (used for base-model selection)
MODEL_SIZES = ["large", "medium", "small", "base", "tiny"]


@dataclass
class SpeechConfig:
    """Configuration for speech transcription pipeline."""
    url: str
    output_dir: Path
    whisper_models: list = field(default_factory=lambda: ["small", "medium"])  # Can be multiple models
    scene_threshold: float = 0.1
    analyze_slides: bool = False
    merge_sources: bool = True  # Merge YouTube captions with Whisper (default: on)
    no_llm: bool = False  # Skip all LLM-dependent features (merging, ensembling, slide analysis)
    api_key: Optional[str] = None
    claude_model: str = "claude-sonnet-4-20250514"  # Anthropic API model; ignored when local=True (uses local_model)
    skip_existing: bool = True
    no_slides: bool = False  # Skip slide extraction entirely
    podcast: bool = False  # Podcast mode: audio-only, skip video + captions
    external_transcript: Optional[str] = None  # External transcript file path or URL to include in merge
    diarize: bool = False  # Enable speaker diarization via pyannote
    num_speakers: Optional[int] = None  # Exact speaker count hint for diarization
    speaker_names: Optional[list] = None  # Manual speaker name mapping (ordered by first appearance)
    steps: Optional[list] = None  # Run only these pipeline steps (None = all)
    dry_run: bool = False  # Show what would be done without doing it
    verbose: bool = False
    # Merge tuning
    merge_chunk_words: int = 500  # Words per chunk for merge API calls (multi-source merge)
    merge_diff_context_words: int = 30  # Words of context around each diff (Whisper ensembling)
    merge_max_diffs_per_call: int = 50  # Max diffs per LLM call (Whisper ensembling)
    api_max_retries: int = 5
    api_initial_backoff: int = 5  # seconds
    api_timeout: float = 120.0  # seconds per API attempt
    # Local LLM (default) vs cloud API
    local: bool = True  # Use local Ollama by default
    local_model: str = "qwen2.5"  # Default Ollama model for text
    local_vision_model: str = "llava"  # Default Ollama model for vision
    ollama_base_url: str = "http://localhost:11434/v1/"


# Standard output filenames — single source of truth
AUDIO_MP3 = "audio.mp3"
AUDIO_WAV = "audio.wav"
METADATA_JSON = "metadata.json"
CAPTIONS_VTT = "captions.en.vtt"
WHISPER_MERGED_TXT = "whisper_merged.txt"
DIARIZATION_JSON = "diarization.json"
DIARIZED_TXT = "diarized.txt"
TRANSCRIPT_MERGED_TXT = "transcript_merged.txt"
ANALYSIS_MD = "analysis.md"
TRANSCRIPT_MD = "transcript.md"
SLIDE_TIMESTAMPS_JSON = "slide_timestamps.json"
SLIDES_TRANSCRIPT_JSON = "slides_transcript.json"


@dataclass
class SpeechData:
    """Data collected during pipeline execution."""
    title: str = ""
    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None
    captions_path: Optional[Path] = None
    transcript_path: Optional[Path] = None  # Primary transcript (or whisper_merged)
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
    diarization_path: Optional[Path] = None  # Diarized structured transcript
    metadata: dict = field(default_factory=dict)  # Source metadata (title, description, channel, etc.)


def is_up_to_date(output: Path, *inputs: Path) -> bool:
    """Check if output file is newer than all input files (make-style)."""
    if not output.exists():
        return False
    output_mtime = output.stat().st_mtime
    for inp in inputs:
        if inp and inp.exists() and inp.stat().st_mtime > output_mtime:
            return False
    return True


class _NormalizedResponse:
    """Wraps an OpenAI-compatible response to match the Anthropic response shape.

    Downstream code accesses message.content[0].text and message.usage.input_tokens,
    so this adapter translates the OpenAI format to match.
    """

    def __init__(self, openai_response):
        text = openai_response.choices[0].message.content or ""
        self.content = [type('Block', (), {'text': text})()]
        usage = openai_response.usage
        self.usage = type('Usage', (), {
            'input_tokens': usage.prompt_tokens or 0 if usage else 0,
            'output_tokens': usage.completion_tokens or 0 if usage else 0,
        })()


def _convert_messages_to_openai(messages: list) -> list:
    """Convert Anthropic-style messages to OpenAI-compatible format.

    Handles vision messages with base64 images (Anthropic uses type="image"
    with source.type="base64"; OpenAI uses type="image_url" with a data URI).
    """
    converted = []
    for msg in messages:
        if isinstance(msg.get("content"), str):
            converted.append(msg)
        elif isinstance(msg.get("content"), list):
            parts = []
            for block in msg["content"]:
                if block.get("type") == "text":
                    parts.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image":
                    source = block.get("source", {})
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{data}"
                        }
                    })
                else:
                    parts.append(block)
            converted.append({"role": msg["role"], "content": parts})
        else:
            converted.append(msg)
    return converted


def _has_vision_content(messages: list) -> bool:
    """Check if any message contains image content blocks."""
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if block.get("type") == "image":
                    return True
    return False


def create_llm_client(config: SpeechConfig):
    """Create either an Anthropic or OpenAI-compatible (Ollama) client."""
    if config.local:
        from openai import OpenAI
        return OpenAI(base_url=config.ollama_base_url, api_key="ollama")
    else:
        import anthropic
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=api_key)


def llm_call_with_retry(client, config: SpeechConfig, **kwargs) -> object:
    """Call the LLM with exponential backoff on transient errors.

    Supports both Anthropic and OpenAI-compatible (Ollama) clients.
    Returns a response with .content[0].text and .usage attributes.
    """
    def _retry_with_backoff(call_fn, timeout_exc, status_exc, retryable_codes, label):
        delay = config.api_initial_backoff
        for attempt in range(1, config.api_max_retries + 1):
            try:
                return call_fn()
            except timeout_exc:
                if attempt < config.api_max_retries:
                    print(f"    {label} timeout, retrying in {delay}s (attempt {attempt}/{config.api_max_retries})...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
            except status_exc as e:
                if e.status_code in retryable_codes and attempt < config.api_max_retries:
                    print(f"    {label} {e.status_code} error, retrying in {delay}s (attempt {attempt}/{config.api_max_retries})...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

    if config.local:
        # OpenAI-compatible path (Ollama)
        from openai import APITimeoutError, APIStatusError
        model = config.local_model
        if _has_vision_content(kwargs.get("messages", [])):
            model = config.local_vision_model
        openai_kwargs = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": _convert_messages_to_openai(kwargs["messages"]),
        }

        def _call_openai():
            return _NormalizedResponse(client.chat.completions.create(**openai_kwargs))

        return _retry_with_backoff(
            _call_openai, APITimeoutError, APIStatusError,
            (429, 500, 502, 503), "LLM")
    else:
        # Anthropic path
        import anthropic
        if "timeout" not in kwargs:
            kwargs["timeout"] = config.api_timeout

        def _call_anthropic():
            return client.messages.create(**kwargs)

        return _retry_with_backoff(
            _call_anthropic, anthropic.APITimeoutError, anthropic.APIStatusError,
            (429, 529, 500), "API")


# Backward compatibility: tests may still import this old name
api_call_with_retry = llm_call_with_retry


# ---------------------------------------------------------------------------
# Pipeline utilities (used across download, transcription, slides, output)
# ---------------------------------------------------------------------------

def run_command(cmd: list[str], description: str, verbose: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command with error handling."""
    if verbose:
        print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"  Error: {description}")
        print(f"  {e.stderr}")
        raise


def _save_json(path: Path, data) -> None:
    """Write data to a JSON file with standard formatting."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _print_reusing(label: str) -> None:
    """Print a 'Reusing' message for a cached artifact."""
    print(f"  Reusing: {label}")


def _dry_run_skip(config: SpeechConfig, action: str, output: str) -> bool:
    """In dry-run mode, print what would happen and return True to skip execution."""
    if not config.dry_run:
        return False
    print(f"  [dry-run] Would {action} → {output}")
    return True


def _should_skip(config: SpeechConfig, output: Path, action: str,
                  *inputs: Path) -> bool:
    """Check if a pipeline stage should skip: output is fresh or dry-run mode.

    Combines the is_up_to_date + _print_reusing + _dry_run_skip pattern
    that repeats across all pipeline stages.

    Returns True if the stage should skip (and prints the reason).
    """
    if config.skip_existing and is_up_to_date(output, *inputs):
        _print_reusing(output.name)
        return True
    if _dry_run_skip(config, action, output.name):
        return True
    return False


def _collect_source_paths(config: SpeechConfig, data: SpeechData,
                          extra: list = None) -> list:
    """Collect source file paths for DAG staleness checks.

    Includes transcript, captions, and external transcript (if it's a local file).
    """
    paths = list(extra or [])
    if data.transcript_path and data.transcript_path.exists():
        paths.append(data.transcript_path)
    if data.captions_path and data.captions_path.exists():
        paths.append(data.captions_path)
    if config.external_transcript and not config.external_transcript.startswith(("http://", "https://")):
        ext_path = Path(config.external_transcript)
        if ext_path.exists():
            paths.append(ext_path)
    if data.diarization_path and data.diarization_path.exists():
        paths.append(data.diarization_path)
    return paths


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
