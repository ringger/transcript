"""
Shared types and utilities for the speech transcription pipeline.

Contains SpeechConfig, SpeechData, and utility functions used by both
transcriber.py and merge.py.
"""

import os
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
    no_llm: bool = False  # Skip all LLM-dependent features (merging, ensembling, slide analysis)
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
    # Local LLM (default) vs cloud API
    local: bool = True  # Use local Ollama by default
    local_model: str = "qwen2.5"  # Default Ollama model for text
    local_vision_model: str = "llava"  # Default Ollama model for vision
    ollama_base_url: str = "http://localhost:11434/v1/"


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
    delay = config.api_initial_backoff

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

        for attempt in range(1, config.api_max_retries + 1):
            try:
                response = client.chat.completions.create(**openai_kwargs)
                return _NormalizedResponse(response)
            except APITimeoutError:
                if attempt < config.api_max_retries:
                    print(f"    LLM timeout, retrying in {delay}s (attempt {attempt}/{config.api_max_retries})...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
            except APIStatusError as e:
                if e.status_code in (429, 500, 502, 503) and attempt < config.api_max_retries:
                    print(f"    LLM {e.status_code} error, retrying in {delay}s (attempt {attempt}/{config.api_max_retries})...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
    else:
        # Anthropic path
        import anthropic
        if "timeout" not in kwargs:
            kwargs["timeout"] = config.api_timeout
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
                    print(f"    API {e.status_code} error, retrying in {delay}s (attempt {attempt}/{config.api_max_retries})...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise


# Keep old name as alias for backward compatibility in tests
api_call_with_retry = llm_call_with_retry
