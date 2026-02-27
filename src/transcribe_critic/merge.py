"""
Merge logic for combining multiple transcript sources into a critical text.

Uses wdiff-based alignment and blind/anonymous presentation to an LLM
for unbiased adjudication of differences between transcript sources.
"""

import json
import os
import re
import shutil
import subprocess

from transcribe_critic.shared import (
    tprint as print,
    SpeechConfig, COMMON_WORDS,
    create_llm_client, llm_call_with_retry, is_up_to_date, _save_json,
)


def _write_temp_text(content: str) -> str:
    """Write content to a temporary text file, return its path.

    Caller is responsible for cleanup with os.unlink().
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return f.name


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
    are replaced with an underscore (_) placeholder so alignment maps stay
    in sync with original word positions.
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

    # Normalize both texts for comparison
    a_normalized = _normalize_for_comparison(text_a)
    b_normalized = _normalize_for_comparison(text_b)

    a_file = _write_temp_text(a_normalized)
    b_file = _write_temp_text(b_normalized)

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
    meaningful_diffs = []
    for d in differences:
        if d["type"] == "changed":
            a_words = set(d["a_text"].lower().split())
            b_words = set(d["b_text"].lower().split())
            if not (a_words <= COMMON_WORDS and b_words <= COMMON_WORDS):
                meaningful_diffs.append(d)
        else:
            text = d.get("text", "").lower()
            if text and text not in COMMON_WORDS:
                meaningful_diffs.append(d)
    return meaningful_diffs


# Matches wdiff markup: [-deleted-], {+inserted+}, or common (unmarked) text.
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
    norm_a = _normalize_for_comparison(text_a)
    norm_b = _normalize_for_comparison(text_b)

    a_path = None
    b_path = None
    try:
        a_path = _write_temp_text(norm_a)
        b_path = _write_temp_text(norm_b)

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


# Transcript structure detection patterns
# "Speaker Name [(HH:MM:SS)](url)" or "Speaker Name (HH:MM:SS)" (Lex Fridman style)
LEX_PATTERN = re.compile(r'^(\w[\w\s]+?)\s*\[?\((\d{1,2}:\d{2}:\d{2})\)\]?')
# "[HH:MM:SS] Speaker: text"
BRACKETED_PATTERN = re.compile(r'^\[(\d{1,2}:\d{2}:\d{2})\]\s+(\w[\w\s]+?):')
# "Speaker Name: text" (no timestamps)
SPEAKER_ONLY_PATTERN = re.compile(r'^([A-Z][\w]+(?:\s+[A-Z][\w]+)*)\s*:\s+\S')

# Ordered detection: try each pattern, first match wins
_STRUCTURE_FORMATS = [
    ("lex",          LEX_PATTERN,          True,  True),
    ("bracketed",    BRACKETED_PATTERN,    True,  True),
    ("speaker_only", SPEAKER_ONLY_PATTERN, True,  False),
]


def _detect_transcript_structure(text: str) -> dict:
    """Detect if a transcript has speaker labels and/or timestamps.

    Returns dict with keys: has_speakers, has_timestamps, format.
    Supported formats: "lex", "bracketed", "speaker_only", or None.
    """
    lines = text.strip().split('\n')
    content_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    sample = content_lines[:20]

    for fmt, pattern, has_speakers, has_timestamps in _STRUCTURE_FORMATS:
        if sum(1 for line in sample if pattern.match(line)) >= 2:
            return {"has_speakers": has_speakers, "has_timestamps": has_timestamps, "format": fmt}

    return {"has_speakers": False, "has_timestamps": False, "format": None}


def _extract_segment_fields(fmt: str, match: re.Match, line: str) -> dict:
    """Extract speaker, timestamp, and initial text from a regex match.

    Each format maps its capture groups differently.
    """
    if fmt == "lex":
        speaker = match.group(1).strip()
        timestamp = match.group(2)
        # Text may continue after the [(HH:MM:SS)](url) on the same line
        rest = LEX_PATTERN.sub('', line).strip()
        rest = re.sub(r'^\([^)]*\)\s*', '', rest)
        return {"speaker": speaker, "timestamp": timestamp, "text": rest}
    elif fmt == "bracketed":
        return {"speaker": match.group(2).strip(), "timestamp": match.group(1), "text": match.group(3).strip()}
    else:  # speaker_only
        return {"speaker": match.group(1).strip(), "timestamp": None, "text": match.group(2).strip()}


# Parsing patterns (may differ from detection patterns for full-line extraction)
_PARSE_PATTERNS = {
    "lex": LEX_PATTERN,
    "bracketed": re.compile(r'^\[(\d{1,2}:\d{2}:\d{2})\]\s+(\w[\w\s]+?):\s*(.*)'),
    "speaker_only": re.compile(r'^([A-Z][\w]+(?:\s+[A-Z][\w]+)*)\s*:\s+(.*)'),
}


def _parse_structured_transcript(text: str, fmt: str) -> list:
    """Parse a structured transcript into segments.

    Returns list of dicts: [{"speaker": str, "timestamp": str|None, "text": str}, ...]
    """
    pattern = _PARSE_PATTERNS.get(fmt)
    if pattern is None:
        return []
    segments = []
    current = None

    for line in text.split('\n'):
        m = pattern.match(line)
        if m:
            if current:
                segments.append(current)
            current = _extract_segment_fields(fmt, m, line)
        elif current is not None:
            if line.strip():
                current["text"] += (" " if current["text"] else "") + line.strip()
            elif fmt == "lex" and current["text"]:
                # Lex format preserves paragraph breaks
                current["text"] += "\n"
    if current:
        segments.append(current)

    for seg in segments:
        seg["text"] = seg["text"].strip()

    return segments


MERGE_CHECKPOINT_VERSION = "5"


def _init_merge_chunks_dir(config: SpeechConfig) -> 'Path':
    """Initialize the merge_chunks directory with version tracking.

    Creates the directory, checks the version file, and clears stale
    checkpoints from previous algorithm versions.

    Returns the chunks_dir path.
    """
    from pathlib import Path
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


def _count_fresh_chunks(num_chunks: int, chunks_dir: 'Path',
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


def _load_chunk_checkpoint(chunks_dir: 'Path', chunk_idx: int):
    """Load a checkpoint file and return the parsed JSON data."""
    checkpoint_path = chunks_dir / f"chunk_{chunk_idx:03d}.json"
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def _save_chunk_checkpoint(chunks_dir: 'Path', chunk_idx: int, data):
    """Save data as a checkpoint JSON file."""
    _save_json(chunks_dir / f"chunk_{chunk_idx:03d}.json", data)


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


def _merge_structured(skeleton_segments: list, all_sources: list,
                      config: SpeechConfig, source_paths: list = None,
                      skeleton_source_name: str = "External Transcript") -> list:
    """Merge transcript sources using blind, label-free presentation.

    The skeleton_segments provide structure (speaker labels, timestamps, segment
    boundaries) from the structured source. Text is presented to the LLM
    anonymously — no source names, no speaker labels, no timestamps — so that
    no source receives preferential treatment.

    Alignment between sources uses wdiff rather than proportional word-fraction
    slicing, with cursor-based tracking to prevent segment overlap.

    Chunks are first-class DAG artefacts in merge_chunks/. Each chunk is checked
    for staleness against source_paths and reused when fresh.

    skeleton_segments: parsed segments from _parse_structured_transcript
    all_sources: list of (name, description, text) tuples including the skeleton source
    skeleton_source_name: name of the source whose text matches the skeleton
    source_paths: list of Path objects for staleness checks

    Returns list of merged segments (same structure as skeleton_segments).
    """
    client = create_llm_client(config)

    chunks_dir = _init_merge_chunks_dir(config)

    # Step 1: Separate skeleton source from other sources
    skeleton_text = None
    other_sources = []
    for name, desc, text in all_sources:
        if name == skeleton_source_name:
            skeleton_text = text
        else:
            other_sources.append((name, desc, text))

    if skeleton_text is None:
        raise ValueError(f"Skeleton source '{skeleton_source_name}' not found in all_sources")

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
        print(f"  Reusing {chunks_reused}/{len(chunks)} chunks from checkpoint, processing {len(chunks) - chunks_reused} via LLM...")
        for chunk_idx in range(chunks_reused):
            chunk_data = _load_chunk_checkpoint(chunks_dir, chunk_idx)
            for i, seg_idx in enumerate(chunks[chunk_idx]):
                if i < len(chunk_data):
                    corrected_segments[seg_idx] = chunk_data[i]

    # Step 6: Process remaining chunks via LLM
    for chunk_idx, seg_indices in enumerate(chunks):
        if chunk_idx < chunks_reused:
            continue

        chunk_words = sum(len(seg["text"].split()) for seg in (skeleton_segments[i] for i in seg_indices))
        print(f"  Merging chunk {chunk_idx + 1}/{len(chunks)} via LLM ({len(seg_indices)} passages, ~{chunk_words} words/source)...")

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

        message = llm_call_with_retry(
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


def _merge_multi_source(sources: list,
                        config: SpeechConfig, source_paths: list = None) -> str:
    """Merge multiple transcript sources using LLM with blind presentation.

    Uses wdiff alignment (not proportional slicing) to split sources into
    aligned chunks. Sources are presented anonymously as Source 1, Source 2, etc.

    Chunks are first-class DAG artefacts in merge_chunks/. Each chunk is checked
    for staleness against source_paths and reused when fresh.

    sources: list of (name, description, text) tuples
    source_paths: list of Path objects for staleness checks
    """
    client = create_llm_client(config)

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
        print(f"  Reusing {chunks_reused}/{num_chunks} chunks from checkpoint, processing {num_chunks - chunks_reused} via LLM...")
        for i in range(chunks_reused):
            merged_chunks.append(_load_chunk_checkpoint(chunks_dir, i))

    for chunk_idx, (start, end) in enumerate(chunks):
        if chunk_idx < chunks_reused:
            continue

        chunk_words = end - start
        print(f"  Merging chunk {chunk_idx + 1}/{num_chunks} via LLM (~{chunk_words} words/source)...")

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

        message = llm_call_with_retry(
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

    a_path = None
    b_path = None
    try:
        a_path = _write_temp_text(norm_a)
        b_path = _write_temp_text(norm_b)

        result = subprocess.run(
            ["wdiff", "-s", a_path, b_path],
            capture_output=True, text=True
        )
        # wdiff puts stats on stdout (after the diff output)
        output = result.stdout + result.stderr

        # Parse stats lines: "/path/to/file: N words  M PP% common  ..."
        wdiff_stat_re = re.compile(r'(\d+)\s+words\s+(\d+)\s+(\d+)%\s+common')
        stats = {}
        for key, path in [('a', a_path), ('b', b_path)]:
            for line in output.strip().split('\n'):
                if path in line:
                    m = wdiff_stat_re.search(line)
                    if m:
                        stats[key] = {
                            'words': int(m.group(1)),
                            'common': int(m.group(2)),
                            'common_pct': int(m.group(3))
                        }
                    break
        return stats
    finally:
        if a_path:
            os.unlink(a_path)
        if b_path:
            os.unlink(b_path)
