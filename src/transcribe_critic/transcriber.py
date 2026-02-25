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
    transcribe-critic <url> [options]

Examples:
    # Basic usage - Whisper transcript + slides
    transcribe-critic "https://youtube.com/watch?v=..."

    # Full pipeline with slide analysis
    transcribe-critic "https://youtube.com/watch?v=..." --analyze-slides

    # Skip merging YouTube captions (merge is on by default)
    transcribe-critic "https://youtube.com/watch?v=..." --no-merge

    # Full pipeline with slide analysis (merging is automatic)
    transcribe-critic "https://youtube.com/watch?v=..." --analyze-slides

    # Ensemble multiple Whisper models for better accuracy
    transcribe-critic "https://youtube.com/watch?v=..." --whisper-models small,medium

    # Run without any LLM (Whisper only, free)
    transcribe-critic "https://youtube.com/watch?v=..." --no-llm

    # Use Anthropic Claude API instead of local Ollama
    transcribe-critic "https://youtube.com/watch?v=..." --api

    # Custom output directory and model
    transcribe-critic "https://youtube.com/watch?v=..." -o my_speech --whisper-models small
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from transcribe_critic import __version__
from transcribe_critic.shared import (
    tprint as print,
    SpeechConfig, SpeechData, is_up_to_date,
    AUDIO_MP3, AUDIO_WAV, CAPTIONS_VTT, WHISPER_MERGED_TXT,
    DIARIZATION_JSON, DIARIZED_TXT, TRANSCRIPT_MERGED_TXT,
    ANALYSIS_MD, SLIDE_TIMESTAMPS_JSON,
    run_command, _print_reusing, _dry_run_skip, _should_skip,
    _collect_source_paths, check_dependencies,
)

SECTION_SEPARATOR = "=" * 50

# Valid pipeline step names
VALID_STEPS = {"download", "transcribe", "ensemble", "diarize", "slides", "merge", "markdown", "analysis"}


def _should_run_step(step_name: str, config: SpeechConfig) -> bool:
    """Check if a pipeline step should run based on --steps filter."""
    if config.steps is None:
        return True
    return step_name in config.steps


def _hydrate_data(config: SpeechConfig, data: SpeechData) -> None:
    """Populate SpeechData from existing files on disk.

    Called when --steps is used so that later steps can find outputs
    from earlier steps that were skipped in this run.
    """
    d = config.output_dir

    # Audio
    for name in (AUDIO_MP3, AUDIO_WAV):
        p = d / name
        if p.exists():
            data.audio_path = p
            break

    # Captions
    cap = d / CAPTIONS_VTT
    if cap.exists():
        data.captions_path = cap

    # Whisper model transcripts
    for txt in sorted(d.glob("whisper_*.txt")):
        name = txt.stem  # e.g. "whisper_medium"
        if "merged" in name:
            continue
        model = name.removeprefix("whisper_")
        json_path = d / f"whisper_{model}.json"
        data.whisper_transcripts[model] = {
            "txt": txt,
            "json": json_path if json_path.exists() else None,
        }

    # Whisper merged (primary transcript when ensembling)
    merged = d / WHISPER_MERGED_TXT
    if merged.exists():
        data.transcript_path = merged
    elif data.whisper_transcripts:
        # Fall back to largest single model
        from transcribe_critic.shared import MODEL_SIZES
        for size in MODEL_SIZES:
            if size in data.whisper_transcripts:
                data.transcript_path = data.whisper_transcripts[size]["txt"]
                break

    # Transcript JSON (use largest model's JSON for timestamps)
    if data.whisper_transcripts:
        from transcribe_critic.shared import MODEL_SIZES
        for size in MODEL_SIZES:
            if size in data.whisper_transcripts:
                data.transcript_json_path = data.whisper_transcripts[size].get("json")
                break

    # Diarization
    diar_txt = d / DIARIZED_TXT
    if diar_txt.exists():
        data.diarization_path = diar_txt

    # Merged transcript
    tm = d / TRANSCRIPT_MERGED_TXT
    if tm.exists():
        data.merged_transcript_path = tm


# API pricing per 1K tokens, keyed by model family prefix (as of 2025-05)
MODEL_PRICING = {
    "claude-opus-4":   {"input": 0.015, "output": 0.075},
    "claude-sonnet-4": {"input": 0.003, "output": 0.015},
    "claude-sonnet-3": {"input": 0.003, "output": 0.015},
    "claude-haiku-4":  {"input": 0.001, "output": 0.005},
    "claude-haiku-3":  {"input": 0.0008, "output": 0.004},
}
DEFAULT_PRICING = MODEL_PRICING["claude-sonnet-4"]  # fallback
VISION_COST_PER_IMAGE = 0.02               # ~$0.01-0.02 per medium slide
TOKENS_PER_WORD = 1.3                      # rough word-to-token ratio


def _get_model_pricing(model_name: str) -> dict:
    """Return {input, output} cost per 1K tokens for the given model name."""
    for prefix, pricing in MODEL_PRICING.items():
        if model_name.startswith(prefix):
            return pricing
    return DEFAULT_PRICING

# Pipeline stage modules
from transcribe_critic.download import download_media, clean_vtt_captions
from transcribe_critic.transcription import transcribe_audio
from transcribe_critic.slides import extract_slides, analyze_slides_with_vision, create_basic_slides_json
from transcribe_critic.output import generate_markdown
from transcribe_critic.diarization import diarize_audio

# Merge logic
from transcribe_critic.merge import (
    _extract_text_from_html,
    _normalize_for_comparison,
    _detect_transcript_structure,
    _parse_structured_transcript,
    _format_structured_segments,
    _merge_structured,
    _merge_multi_source,
    _wdiff_stats,
)


def _slugify_title(title: str, max_len: int = 50) -> str:
    """Convert a title to a filesystem-safe slug."""
    safe = re.sub(r'[^\w\s-]', '', title)[:max_len].strip()
    return re.sub(r'\s+', '-', safe).lower()


def _fetch_metadata(url: str, verbose: bool = False) -> dict:
    """Fetch video/audio metadata via yt-dlp without downloading."""
    result = run_command(
        ["yt-dlp", "--dump-json", url],
        "fetching media info",
        verbose,
    )
    return json.loads(result.stdout)


def _load_external_transcript(config: SpeechConfig) -> tuple:
    """Load an external transcript from a URL or file path.

    Returns (text, source_label) or (None, source_label) on failure.
    """
    source = config.external_transcript
    source_label = source
    if source.startswith(("http://", "https://")):
        print(f"  Fetching external transcript from URL...")
        import urllib.request
        try:
            with urllib.request.urlopen(source) as response:
                raw = response.read().decode('utf-8').strip()
            if '<html' in raw[:500].lower() or '<body' in raw[:1000].lower():
                text = _extract_text_from_html(raw)
            else:
                text = raw
            source_label = source.split('/')[-1] or source
            return text, source_label
        except Exception as e:
            print(f"  Warning: Could not fetch external transcript URL: {e}")
            return None, source_label
    else:
        ext_path = Path(source)
        source_label = ext_path.name
        if ext_path.exists():
            with open(ext_path, 'r') as f:
                return f.read().strip(), source_label
        return None, source_label


def estimate_api_cost(config: SpeechConfig, num_slides: int = 45, transcript_words: int = 6000) -> dict:
    """Estimate API costs for the configured options.

    Pricing is looked up from MODEL_PRICING based on config.claude_model.
    Returns dict with cost breakdown and total.
    """
    costs = {
        "analyze_slides": 0.0,
        "merge_sources": 0.0,
        "ensemble_whisper": 0.0,
        "total": 0.0,
        "details": []
    }

    if config.no_llm:
        return costs

    pricing = _get_model_pricing(config.claude_model)

    if config.analyze_slides:
        slide_cost = num_slides * VISION_COST_PER_IMAGE
        costs["analyze_slides"] = slide_cost
        costs["details"].append(f"Slide analysis: {num_slides} slides × ${VISION_COST_PER_IMAGE} = ${slide_cost:.2f}")

    if config.merge_sources:
        num_sources = 2  # Whisper + YouTube
        if config.external_transcript:
            num_sources += 1
        num_chunks = max(1, transcript_words // config.merge_chunk_words + 1)
        chunk_input_words = transcript_words * num_sources // num_chunks + 500
        chunk_output_words = transcript_words // num_chunks
        total_input_tokens = int(chunk_input_words * num_chunks * TOKENS_PER_WORD)
        total_output_tokens = int(chunk_output_words * num_chunks * TOKENS_PER_WORD)
        input_cost = total_input_tokens * pricing["input"] / 1000
        output_cost = total_output_tokens * pricing["output"] / 1000
        merge_cost = input_cost + output_cost
        costs["merge_sources"] = merge_cost
        costs["details"].append(
            f"Source merging: {num_sources} sources × {num_chunks} chunks = ${merge_cost:.2f}")

    if len(config.whisper_models) > 1:
        num_models = len(config.whisper_models)
        num_chunks = max(1, transcript_words // config.merge_chunk_words + 1)
        chunk_input_words = transcript_words * num_models // num_chunks + 500
        chunk_output_words = transcript_words // num_chunks
        total_input_tokens = int(chunk_input_words * num_chunks * TOKENS_PER_WORD)
        total_output_tokens = int(chunk_output_words * num_chunks * TOKENS_PER_WORD)
        input_cost = total_input_tokens * pricing["input"] / 1000
        output_cost = total_output_tokens * pricing["output"] / 1000
        ensemble_cost = input_cost + output_cost
        costs["ensemble_whisper"] = ensemble_cost
        costs["details"].append(
            f"Whisper ensemble: {num_models} models × {num_chunks} clusters = ${ensemble_cost:.2f}")

    costs["total"] = costs["analyze_slides"] + costs["merge_sources"] + costs["ensemble_whisper"]

    return costs


def print_cost_estimate(config: SpeechConfig, num_slides: int = 45, transcript_words: int = 6000) -> None:
    """Print estimated API costs before running."""
    costs = estimate_api_cost(config, num_slides, transcript_words)

    if costs["total"] == 0:
        return

    print()
    print(SECTION_SEPARATOR)
    print("ESTIMATED API COSTS")
    print(SECTION_SEPARATOR)

    for detail in costs["details"]:
        print(f"  {detail}")

    print()
    print(f"  TOTAL: ${costs['total']:.2f} (estimate, using {config.claude_model})")
    print("  Note: Actual costs may vary based on transcript length")
    print(SECTION_SEPARATOR)
    print()


def merge_transcript_sources(config: SpeechConfig, data: SpeechData) -> None:
    """Merge transcript sources (Whisper, captions, external) using wdiff alignment and LLM adjudication."""
    print()
    print("[4b] Merging transcript sources...")

    if not config.merge_sources:
        print("  Skipped (--no-merge flag set)")
        return

    if config.no_llm:
        print("  Skipped (--no-llm flag set)")
        return

    # Load external transcript if provided
    external_text = None
    if config.external_transcript:
        external_text, source_label = _load_external_transcript(config)
        if external_text:
            print(f"  External transcript: {len(external_text.split())} words ({source_label})")

    # Check if we have enough sources to merge
    has_captions = data.captions_path and data.captions_path.exists()
    has_whisper = data.transcript_path and data.transcript_path.exists()

    if not has_whisper and not external_text:
        print("  No Whisper transcript or external transcript available, skipping merge")
        return

    # Use diarized transcript as a structured source when available
    has_diarized = data.diarization_path and data.diarization_path.exists()

    if not has_captions and not external_text and not has_diarized:
        print("  No YouTube captions, external transcript, or diarization available, skipping merge")
        return

    merged_path = config.output_dir / TRANSCRIPT_MERGED_TXT

    merge_inputs = _collect_source_paths(config, data)
    if _should_skip(config, merged_path, "merge transcript sources", *merge_inputs):
        if merged_path.exists():
            data.merged_transcript_path = merged_path
        return

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

    # Load diarized transcript if available — provides structural skeleton
    # (not added as a text source since its text duplicates the Whisper transcript)
    diarized_text = None
    if has_diarized:
        with open(data.diarization_path, 'r') as f:
            diarized_text = f.read()
        if external_text:
            print(f"  Diarized transcript available but external transcript provides structure")
        else:
            print(f"  Diarized transcript: {len(diarized_text.split())} words (structural skeleton)")

    print(f"  Merging {len(sources)} sources: {', '.join(s[0] for s in sources)}")

    if len(sources) < 2 and not diarized_text:
        print("  Need at least 2 sources to merge, skipping")
        return

    # Check if external or diarized transcript has structure (speaker labels).
    # External takes priority as skeleton (authoritative speaker names);
    # diarized is used as skeleton only when no external is provided.
    structure = None
    structured_text = external_text or diarized_text
    if structured_text:
        structure = _detect_transcript_structure(structured_text)
        struct_label = "external" if external_text else "diarized"
        if structure["has_speakers"]:
            print(f"  Detected structured {struct_label} transcript (format: {structure['format']}, "
                  f"speakers: {structure['has_speakers']}, timestamps: {structure['has_timestamps']})")

    # Route to structured merge if we have speaker labels
    if structure and structure["has_speakers"]:
        skeleton_segments = _parse_structured_transcript(structured_text, structure["format"])
        print(f"  Parsed {len(skeleton_segments)} segments from {struct_label} transcript")

        if len(sources) < 2 and not external_text:
            # Diarized skeleton with only Whisper — no LLM merge needed.
            # The diarized transcript already has structure + Whisper text.
            print("  Single source with diarized skeleton — using diarized text directly")
            merged_text = diarized_text
        else:
            # Add the skeleton's text as a source so _merge_structured can
            # use it as the alignment anchor.
            skeleton_source_name = "External Transcript"
            if not external_text and diarized_text:
                skeleton_source_name = "Diarized Transcript"
                plain_diarized = " ".join(
                    seg["text"] for seg in skeleton_segments)
                sources.append((
                    skeleton_source_name,
                    "diarized transcript with speaker labels",
                    plain_diarized,
                ))
            corrected_segments = _merge_structured(
                skeleton_segments, sources, config, merge_inputs,
                skeleton_source_name=skeleton_source_name)
            merged_text = _format_structured_segments(corrected_segments)
    else:
        # Flat merge: wdiff alignment and anonymous presentation
        merged_text = _merge_multi_source(sources, config, merge_inputs)

    # Save merged transcript
    with open(merged_path, 'w') as f:
        f.write(merged_text)

    data.merged_transcript_path = merged_path
    print(f"  Merged transcript saved: {merged_path.name}")
    print(f"  Merged word count: {len(merged_text.split())} words")

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
    print()
    print("[6] Analyzing source survival...")

    merged_path = config.output_dir / TRANSCRIPT_MERGED_TXT
    analysis_path = config.output_dir / ANALYSIS_MD

    if not merged_path.exists():
        if config.dry_run:
            print(f"  [dry-run] Would analyze source survival → {analysis_path.name}")
        else:
            print("  No merged transcript found, skipping analysis")
        return

    # Gather source files for DAG check
    analysis_inputs = _collect_source_paths(config, data, extra=[merged_path])

    if _should_skip(config, analysis_path, "analyze source survival", *analysis_inputs):
        return

    # Read merged transcript
    with open(merged_path, 'r') as f:
        merged_text = f.read()

    # Strip structural headers for fair comparison
    merged_clean = _strip_structured_headers(merged_text)

    # Collect sources
    sources = []

    # Whisper (merged from multiple models, or single model)
    if data.transcript_path and data.transcript_path.exists():
        with open(data.transcript_path, 'r') as f:
            whisper_text = f.read()
        label = "whisper_merged" if WHISPER_MERGED_TXT in data.transcript_path.name else data.transcript_path.stem
        sources.append((f"Whisper ({label})", whisper_text))

    # YouTube captions
    if data.captions_path and data.captions_path.exists():
        youtube_text = clean_vtt_captions(data.captions_path)
        sources.append(("YouTube captions", youtube_text))

    # External transcript
    if config.external_transcript:
        external_text, _ = _load_external_transcript(config)
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
    print()
    print(f"  {'Source':<25} {'Words':>8} {'Common':>8} {'Output Coverage':>16} {'Retention':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*16} {'-'*10}")
    for r in results:
        print(f"  {r['name']:<25} {r['source_words']:>8,} {r['common_from_merged']:>8,} "
              f"{r['common_pct_of_merged']:>15}% {r['common_pct_of_source']:>9}%")
    print(f"  {'Merged output':<25} {merged_words:>8,}")

    # Find most similar source
    best = max(results, key=lambda r: r['common_pct_of_merged'])
    print()
    print(f"  Merged output most closely resembles: {best['name']} "
          f"({best['common_pct_of_merged']}% overlap)")

    # Write analysis.md
    report = [f"# Source Survival Analysis", ""]
    report.append(f"Merged transcript: `{merged_path.name}` ({merged_words:,} words)")
    report.append("")
    report.append("| Source | Words | Common | Output Coverage | Retention |")
    report.append("|--------|------:|-------:|----------------:|----------:|")
    for r in results:
        report.append(f"| {r['name']} | {r['source_words']:,} | {r['common_from_merged']:,} "
                      f"| {r['common_pct_of_merged']}% | {r['common_pct_of_source']}% |")
    report.append(f"| **Merged output** | **{merged_words:,}** | | | |")
    report.append("")
    report.append(f"**Most similar source:** {best['name']} ({best['common_pct_of_merged']}% output coverage)")
    report.append("")
    report.append("## Column definitions")
    report.append("")
    report.append("- **Words**: word count of the source (after normalization)")
    report.append("- **Common**: words shared between merged output and this source")
    report.append("- **Output Coverage**: what percentage of the merged output's words appear in this source")
    report.append("- **Retention**: what percentage of this source's words survived into the merged output")
    report.append("")

    with open(analysis_path, 'w') as f:
        f.write('\n'.join(report))
    print()
    print(f"  Analysis saved: {analysis_path.name}")


def main():
    parser = argparse.ArgumentParser(
        prog="transcribe-critic",
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
  %(prog)s --podcast "https://www.iheart.com/podcast/.../episode/..."
        """
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")

    # Input
    input_group = parser.add_argument_group("input")
    input_group.add_argument("url", help="URL of the speech video or podcast episode")
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
    input_group.add_argument("--podcast", action="store_true",
                        help="Podcast mode: audio-only, skip video and captions download (implies --no-slides)")
    slides_group.add_argument("--scene-threshold", type=float, default=0.1,
                        help="Scene detection threshold 0-1 (default: 0.1)")
    slides_group.add_argument("--analyze-slides", action="store_true",
                        help="Use Claude vision API to analyze slides (requires API key)")

    # LLM backend
    llm_group = parser.add_argument_group("LLM backend")
    llm_group.add_argument("--api", action="store_true",
                        help="Use Anthropic Claude API instead of local Ollama (requires API key)")
    llm_group.add_argument("--api-key",
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var; implies --api)")
    llm_group.add_argument("--claude-model", default="claude-sonnet-4-20250514",
                        help="Claude model for API calls (default: claude-sonnet-4-20250514)")
    llm_group.add_argument("--local-model", default="qwen2.5:14b",
                        help="Ollama model for text tasks (default: qwen2.5:14b)")
    llm_group.add_argument("--ollama-url", default="http://localhost:11434/v1/",
                        help="Ollama server URL (default: http://localhost:11434/v1/)")
    llm_group.add_argument("--no-llm", action="store_true",
                        help="Skip all LLM-dependent features (merging, ensembling, slide analysis)")
    llm_group.add_argument("--no-merge", action="store_true",
                        help="Skip merging YouTube captions with Whisper (merge is on by default)")

    # Diarization
    diarize_group = parser.add_argument_group("diarization")
    diarize_group.add_argument("--diarize", action="store_true",
                        help="Enable speaker diarization via pyannote (requires pyannote.audio and HF_TOKEN)")
    diarize_group.add_argument("--num-speakers", type=int, default=None,
                        help="Exact number of speakers (improves diarization accuracy)")
    diarize_group.add_argument("--speaker-names",
                        help="Comma-separated speaker names in order of first appearance "
                             "(e.g., 'Ross Douthat,Dario Amodei')")

    # Pipeline control
    pipeline_group = parser.add_argument_group("pipeline")
    pipeline_group.add_argument("--steps",
                        help="Run only these pipeline steps (comma-separated). "
                             "Steps: download, transcribe, ensemble, diarize, slides, merge, markdown, analysis. "
                             "Implies --force for listed steps. Existing outputs are used for skipped steps.")
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

    # Fetch metadata early to resolve output directory from title
    media_info = None
    if not args.output_dir:
        try:
            print()
            print("Fetching media info...")
            media_info = _fetch_metadata(args.url, args.verbose)
            title = media_info.get("title", "speech")
            slug = _slugify_title(title)
            output_dir = Path(f"./transcripts/{slug}")
        except Exception:
            output_dir = Path("./transcripts/speech")
    else:
        output_dir = Path(args.output_dir)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Parse whisper models (comma-separated)
    whisper_models = [m.strip() for m in args.whisper_models.split(",")]
    valid_models = ["tiny", "base", "small", "medium", "large"]
    for m in whisper_models:
        if m not in valid_models:
            print(f"Invalid Whisper model: {m}")
            print(f"Valid options: {', '.join(valid_models)}")
            sys.exit(1)

    # Determine LLM backend: --api or --api-key switches to cloud API
    use_api = args.api or bool(args.api_key)

    # Parse --steps
    steps = None
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
        invalid = set(steps) - VALID_STEPS
        if invalid:
            print(f"Invalid step(s): {', '.join(sorted(invalid))}")
            print(f"Valid steps: {', '.join(sorted(VALID_STEPS))}")
            sys.exit(1)

    # Podcast mode implies --no-slides
    no_slides = args.no_slides or args.podcast

    # Create config
    config = SpeechConfig(
        url=args.url,
        output_dir=output_dir,
        whisper_models=whisper_models,
        scene_threshold=args.scene_threshold,
        no_slides=no_slides,
        podcast=args.podcast,
        analyze_slides=args.analyze_slides,
        merge_sources=not args.no_merge,
        external_transcript=args.external_transcript,
        diarize=args.diarize,
        num_speakers=args.num_speakers,
        speaker_names=[n.strip() for n in args.speaker_names.split(",")] if args.speaker_names else None,
        no_llm=args.no_llm,
        local=not use_api,
        local_model=args.local_model,
        ollama_base_url=args.ollama_url,
        api_key=args.api_key,
        claude_model=args.claude_model,
        steps=steps,
        skip_existing=not args.force if not steps else False,
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
                print()
                print(f"Error: Cannot reach external transcript URL: {config.external_transcript}")
                print(f"  {e}")
                sys.exit(1)
        elif not Path(config.external_transcript).exists():
            print()
            print(f"Error: External transcript file not found: {config.external_transcript}")
            sys.exit(1)

    # If external transcript is provided, ensure merge is enabled
    if config.external_transcript and not config.merge_sources:
        print()
        print("Note: --external-transcript implies merging; enabling merge.")
        config.merge_sources = True

    # Early validation: if using cloud API, check for API key
    llm_features_requested = config.analyze_slides or config.merge_sources or len(config.whisper_models) > 1
    if llm_features_requested and not config.no_llm and not config.local:
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print()
            print("Error: --api mode requested but no API key found.")
            print()
            print("Options:")
            print("  1. Set ANTHROPIC_API_KEY environment variable")
            print("  2. Use --api-key YOUR_KEY")
            print("  3. Remove --api to use local Ollama instead (free)")
            print("  4. Add --no-llm to skip LLM-dependent features")
            sys.exit(1)

    # Show LLM backend info
    if config.no_llm:
        print(f"  LLM: disabled (--no-llm)")
    elif config.local:
        print(f"  LLM: local Ollama ({config.local_model})")
    else:
        print(f"  LLM: Anthropic API ({config.claude_model})")

    print()
    print(f"Processing: {args.url}")
    print(f"Output directory: {output_dir}")

    # Use pre-fetched metadata for cost estimation
    estimated_words = 6000  # fallback
    if media_info:
        duration_secs = media_info.get("duration", 0)
        if duration_secs:
            estimated_words = int(duration_secs * 2.5)
            duration_min = duration_secs / 60
            print(f"  Duration: {duration_min:.0f} min → ~{estimated_words:,} words estimated")

    # Print cost estimate if using cloud API
    if llm_features_requested and not config.no_llm and not config.local:
        print_cost_estimate(config, transcript_words=estimated_words)

    if config.dry_run:
        print()
        print(SECTION_SEPARATOR)
        print("DRY RUN - No actions will be taken")
        print(SECTION_SEPARATOR)

    try:
        # Hydrate data from existing files when running selective steps
        if config.steps:
            _hydrate_data(config, data)
            print(f"  Running steps: {', '.join(config.steps)}")
            print()

        # Run pipeline — each stage skips if output already exists
        if _should_run_step("download", config):
            download_media(config, data, info=media_info)

        if _should_run_step("transcribe", config):
            transcribe_audio(config, data)

        # Ensemble can be re-run independently of transcription
        if _should_run_step("ensemble", config) and not _should_run_step("transcribe", config):
            from transcribe_critic.transcription import _ensemble_whisper_transcripts
            _ensemble_whisper_transcripts(config, data)

        if config.diarize and _should_run_step("diarize", config):
            diarize_audio(config, data)

        if not config.no_slides and _should_run_step("slides", config):
            extract_slides(config, data)

            if config.analyze_slides:
                analyze_slides_with_vision(config, data)
            else:
                create_basic_slides_json(config, data)

        # Merge transcript sources if requested
        if _should_run_step("merge", config):
            merge_transcript_sources(config, data)

        if _should_run_step("markdown", config):
            generate_markdown(config, data)

        if _should_run_step("analysis", config):
            analyze_source_survival(config, data)

        # Summary
        print()
        print(SECTION_SEPARATOR)
        print("COMPLETE!")
        print(SECTION_SEPARATOR)
        print()
        print(f"Output directory: {config.output_dir}")
        print()
        print(f"Generated files:")
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
            label = "whisper-merged transcript" if len(data.whisper_transcripts) > 1 else "transcript"
            print(f"  - {data.transcript_path.name} ({label})")
        if data.transcript_json_path and data.transcript_json_path.exists():
            print(f"  - {data.transcript_json_path.name} (transcript with timestamps)")
        if data.diarization_path and data.diarization_path.exists():
            print(f"  - {data.diarization_path.name} (diarized transcript)")
        if data.merged_transcript_path and data.merged_transcript_path.exists():
            print(f"  - {data.merged_transcript_path.name} (merged from YouTube + Whisper)")
        if data.slides_dir and data.slides_dir.exists():
            print(f"  - slides/ ({len(data.slide_images)} images)")
        timestamps_file = config.output_dir / SLIDE_TIMESTAMPS_JSON
        if timestamps_file.exists():
            print(f"  - slide_timestamps.json")
        if data.slides_json_path and data.slides_json_path.exists():
            print(f"  - {data.slides_json_path.name}")
        if data.markdown_path and data.markdown_path.exists():
            print(f"  - {data.markdown_path.name}")
        analysis_file = config.output_dir / ANALYSIS_MD
        if analysis_file.exists():
            print(f"  - {analysis_file.name} (source survival analysis)")

        if not config.no_slides and not config.analyze_slides and data.slide_images:
            print()
            print("Tip: Run with --analyze-slides to get detailed slide descriptions")

        if (not config.merge_sources or config.no_llm) and data.captions_path and data.captions_path.exists() and not data.merged_transcript_path:
            print("Tip: YouTube captions available - run without --no-merge/--no-llm to create a 'critical text'")

    except Exception as e:
        print()
        print(f"Error: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
