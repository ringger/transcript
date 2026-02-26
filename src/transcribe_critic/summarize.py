"""Transcript summarization using a configurable LLM backend."""

from __future__ import annotations

import os
from dataclasses import replace

from transcribe_critic.shared import (
    SUMMARY_MD,
    SpeechConfig,
    SpeechData,
    _should_skip,
    create_llm_client,
    llm_call_with_retry,
)

SUMMARY_SYSTEM_PROMPT = (
    "You are an expert summarizer. Given a transcript, produce a concise, "
    "well-structured Markdown summary. Include:\n"
    "- A brief overview (1-2 paragraphs)\n"
    "- Key points or arguments as a bulleted list\n"
    "- Speakers mentioned (if any)\n"
    "- Notable quotes or claims (if any)\n\n"
    "Be faithful to the content. Do not editorialize or add information "
    "not present in the transcript."
)


def _resolve_summary_config(config: SpeechConfig) -> SpeechConfig:
    """Return a SpeechConfig with LLM fields overridden for summarization.

    If summary-specific flags were provided, they override the main LLM
    backend.  Otherwise the main backend is inherited.
    """
    overrides = {}

    if config.summary_local is not None:
        overrides["local"] = config.summary_local

    effective_local = overrides.get("local", config.local)

    if config.summary_model is not None:
        if effective_local:
            overrides["local_model"] = config.summary_model
        else:
            overrides["claude_model"] = config.summary_model

    if config.summary_api_key is not None:
        overrides["api_key"] = config.summary_api_key

    if not overrides:
        return config
    return replace(config, **overrides)


def _get_best_transcript(data: SpeechData) -> str | None:
    """Return the best available transcript text.

    Prefers diarized (has speaker labels), then merged, whisper, captions.
    """
    for path in [data.diarization_path, data.merged_transcript_path,
                 data.transcript_path]:
        if path and path.exists():
            text = path.read_text().strip()
            if text:
                return text

    if data.captions_path and data.captions_path.exists():
        text = data.captions_path.read_text().strip()
        if text:
            return text

    return None


def summarize_transcript(config: SpeechConfig, data: SpeechData) -> None:
    """Generate a summary of the transcript using the configured summary LLM."""
    print()
    print("[summarize] Generating summary...")

    if not config.summarize:
        print("  Skipped (summarization disabled)")
        return

    summary_path = config.output_dir / SUMMARY_MD

    # Collect inputs for DAG staleness check
    summary_inputs = [
        p for p in [data.diarization_path, data.merged_transcript_path,
                     data.transcript_path]
        if p and p.exists()
    ]

    if _should_skip(config, summary_path, "summarize transcript", *summary_inputs):
        if summary_path.exists():
            data.summary_path = summary_path
        return

    if config.no_llm:
        print("  Skipped (--no-llm)")
        return

    transcript_text = _get_best_transcript(data)
    if not transcript_text:
        print("  No transcript available to summarize, skipping")
        return

    # Resolve summary-specific LLM backend
    summary_cfg = _resolve_summary_config(config)
    client = create_llm_client(summary_cfg)

    model = summary_cfg.claude_model if not summary_cfg.local else summary_cfg.local_model
    print(f"  Using model: {model}")

    word_count = len(transcript_text.split())
    print(f"  Transcript: {word_count:,} words")

    message = llm_call_with_retry(
        client, summary_cfg,
        model=model,
        max_tokens=4096,
        system=SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Summarize this transcript:\n\n{transcript_text}"}],
    )

    summary = message.content[0].text.strip()
    print(f"  Response: {len(summary)} chars, usage: "
          f"{message.usage.input_tokens} in / {message.usage.output_tokens} out")

    with open(summary_path, "w") as f:
        f.write(summary)
        f.write("\n")

    data.summary_path = summary_path
    print(f"  Summary saved: {summary_path.name}")
