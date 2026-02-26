"""
Transcription module for the speech transcription pipeline.

Handles speech-to-text transcription with Whisper models, including
multi-model ensembling via wdiff analysis and LLM adjudication.
"""

import json
import re
from pathlib import Path

from transcribe_critic.shared import (
    tprint as print,
    SpeechConfig, SpeechData, is_up_to_date,
    WHISPER_MERGED_TXT, COMMON_WORDS, MLX_MODEL_MAP,
    create_llm_client, llm_call_with_retry,
    run_command, _save_json, _print_reusing, _dry_run_skip, _should_skip,
    check_dependencies, MODEL_SIZES,
)
from transcribe_critic.merge import (
    _normalize_for_comparison, _write_temp_text, _parse_wdiff_tokens,
)


# ---------------------------------------------------------------------------
# Repetition loop detection and collapsing (anti-hallucination)
# ---------------------------------------------------------------------------

def detect_repetition_loops(text: str, min_repeats: int = 4,
                             max_phrase_words: int = 10) -> list[dict]:
    """Detect consecutive repetition loops in text.

    Returns a list of dicts with keys: phrase, count, start_pos, end_pos.
    """
    loops = []
    words = text.split()
    n = len(words)
    i = 0

    while i < n:
        best_phrase_len = 0
        best_count = 0

        # Try phrase lengths from shortest to longest — prefer the shortest
        # phrase with the most repeats (hallucinations repeat short phrases)
        for phrase_len in range(1, min(max_phrase_words, (n - i) // min_repeats) + 1):
            phrase = words[i:i + phrase_len]
            count = 1
            j = i + phrase_len
            while j + phrase_len <= n and words[j:j + phrase_len] == phrase:
                count += 1
                j += phrase_len

            if count >= min_repeats:
                best_phrase_len = phrase_len
                best_count = count
                break  # Shortest repeating phrase found

        if best_count >= min_repeats:
            phrase_text = " ".join(words[i:i + best_phrase_len])
            # Calculate character positions
            prefix = " ".join(words[:i])
            start_pos = len(prefix) + (1 if prefix else 0)
            repeated_text = (" ".join([phrase_text] * best_count))
            end_pos = start_pos + len(repeated_text)

            loops.append({
                "phrase": phrase_text,
                "count": best_count,
                "start_pos": start_pos,
                "end_pos": end_pos,
            })
            i += best_phrase_len * best_count
        else:
            i += 1

    return loops


def collapse_repetition_loops(text: str, min_repeats: int = 4,
                               max_phrase_words: int = 10) -> tuple[str, list[dict]]:
    """Collapse consecutive repetition loops, keeping only 2 occurrences.

    Returns (collapsed_text, loops_found).
    """
    loops = detect_repetition_loops(text, min_repeats, max_phrase_words)
    if not loops:
        return text, []

    # Rebuild text with loops collapsed (work on word array)
    words = text.split()
    result_words = []
    i = 0
    n = len(words)

    for loop in sorted(loops, key=lambda l: l["start_pos"]):
        phrase_words = loop["phrase"].split()
        phrase_len = len(phrase_words)
        total_len = phrase_len * loop["count"]

        # Find the word index where this loop starts
        # Walk forward to find the matching position
        while i < n:
            # Check if current position is the start of this loop
            if (i + total_len <= n and
                    words[i:i + phrase_len] == phrase_words):
                # Verify this is actually the full loop
                is_loop = True
                for k in range(loop["count"]):
                    if words[i + k * phrase_len:i + (k + 1) * phrase_len] != phrase_words:
                        is_loop = False
                        break
                if is_loop:
                    # Keep only 2 occurrences
                    result_words.extend(phrase_words)
                    result_words.extend(phrase_words)
                    i += total_len
                    break
            result_words.append(words[i])
            i += 1

    # Append remaining words
    result_words.extend(words[i:])

    return " ".join(result_words), loops


def _select_largest_model_json(data: SpeechData):
    """Return the JSON path from the largest available Whisper model."""
    for size in MODEL_SIZES:
        if size in data.whisper_transcripts:
            return data.whisper_transcripts[size].get("json")
    return None


# ---------------------------------------------------------------------------
# Positioned wdiff parsing for targeted diff resolution
# ---------------------------------------------------------------------------

def _parse_wdiff_diffs(text_a: str, text_b: str, config: SpeechConfig) -> list[dict]:
    """Run wdiff on two texts and return positioned diffs.

    Each diff has:
        type: "substitution" | "insertion" | "deletion"
        a_text: original words from text_a (empty for insertion)
        b_text: original words from text_b (empty for deletion)
        a_pos: word index in text_a where diff starts
        b_pos: word index in text_b where diff starts
        a_len: number of words in text_a span
        b_len: number of words in text_b span

    Normalization is used for alignment only — the returned a_text/b_text
    are from the original (unnormalized) texts so replacements preserve
    the original casing and punctuation.
    """
    import os
    import subprocess

    norm_a = _normalize_for_comparison(text_a)
    norm_b = _normalize_for_comparison(text_b)
    a_words_orig = text_a.split()
    b_words_orig = text_b.split()

    a_path = _write_temp_text(norm_a)
    b_path = _write_temp_text(norm_b)
    try:
        result = subprocess.run(
            ["wdiff", a_path, b_path],
            capture_output=True, text=True,
        )
        tokens = _parse_wdiff_tokens(result.stdout)
    finally:
        os.unlink(a_path)
        os.unlink(b_path)

    # Walk tokens, tracking positions in both original texts
    diffs = []
    a_pos = 0
    b_pos = 0

    i = 0
    while i < len(tokens):
        tok_type, tok_text = tokens[i]
        n = len(tok_text.split())

        if tok_type == "common":
            a_pos += n
            b_pos += n
            i += 1

        elif tok_type == "deleted" and i + 1 < len(tokens) and tokens[i + 1][0] == "inserted":
            # Adjacent deleted + inserted = substitution
            ins_text = tokens[i + 1][1]
            ins_n = len(ins_text.split())
            diffs.append({
                "type": "substitution",
                "a_text": " ".join(a_words_orig[a_pos:a_pos + n]),
                "b_text": " ".join(b_words_orig[b_pos:b_pos + ins_n]),
                "a_pos": a_pos,
                "b_pos": b_pos,
                "a_len": n,
                "b_len": ins_n,
            })
            a_pos += n
            b_pos += ins_n
            i += 2

        elif tok_type == "deleted":
            # In A only (deletion from B's perspective)
            diffs.append({
                "type": "deletion",
                "a_text": " ".join(a_words_orig[a_pos:a_pos + n]),
                "b_text": "",
                "a_pos": a_pos,
                "b_pos": b_pos,
                "a_len": n,
                "b_len": 0,
            })
            a_pos += n
            i += 1

        elif tok_type == "inserted":
            # In B only (insertion from A's perspective)
            diffs.append({
                "type": "insertion",
                "a_text": "",
                "b_text": " ".join(b_words_orig[b_pos:b_pos + n]),
                "a_pos": a_pos,
                "b_pos": b_pos,
                "a_len": 0,
                "b_len": n,
            })
            b_pos += n
            i += 1

    return diffs


def _filter_trivial_diffs(diffs: list[dict]) -> list[dict]:
    """Remove diffs where both sides are only common/stop words."""
    filtered = []
    for d in diffs:
        a_words = set(d["a_text"].lower().split()) if d["a_text"] else set()
        b_words = set(d["b_text"].lower().split()) if d["b_text"] else set()
        # Keep if either side has a non-trivial word
        if not (a_words <= COMMON_WORDS and b_words <= COMMON_WORDS):
            filtered.append(d)
    return filtered


def _merge_pairwise_diffs(pairwise_diffs: list[tuple[str, list[dict]]],
                          base_model: str,
                          all_models: list[str]) -> list[dict]:
    """Merge pairwise diffs into multi-way diffs with per-model readings.

    Each pairwise entry is (model_name, diffs) where diffs are from wdiff
    of that model against the base. Diffs at the same (b_pos, b_len) are
    merged into a single diff with a ``readings`` dict mapping each model
    name to its text at that position. Models not present in any pairwise
    diff at a position implicitly agree with the base.

    The ``readings`` dict is ordered: non-base models first (in all_models
    order), then the base model last.
    """
    # Index pairwise diffs by (b_pos, b_len)
    by_pos: dict[tuple[int, int], dict] = {}  # key → merged diff
    readings_by_pos: dict[tuple[int, int], dict[str, str]] = {}

    for model_name, diffs in pairwise_diffs:
        for d in diffs:
            key = (d["b_pos"], d["b_len"])
            if key not in by_pos:
                by_pos[key] = dict(d)  # copy first diff as template
                readings_by_pos[key] = {}
            readings_by_pos[key][model_name] = d["a_text"]

    # Build final merged diffs with ordered readings
    merged = []
    non_base = [m for m in all_models if m != base_model]
    for key in sorted(by_pos):
        d = by_pos[key]
        readings = {}
        for m in non_base:
            if m in readings_by_pos[key]:
                readings[m] = readings_by_pos[key][m]
            else:
                # Model agreed with base at this position
                readings[m] = d["b_text"]
        # Base model last
        readings[base_model] = d["b_text"]
        d["readings"] = readings
        merged.append(d)

    return merged


def _cluster_diffs(diffs: list[dict], base_word_count: int,
                   context_words: int = 30,
                   max_cluster_diffs: int = 50) -> list[list[dict]]:
    """Group nearby diffs into clusters for batched LLM resolution.

    Diffs within context_words of each other are grouped together.
    Each cluster is capped at max_cluster_diffs diffs.
    """
    if not diffs:
        return []

    # Sort by position in base text (text_b = medium)
    sorted_diffs = sorted(diffs, key=lambda d: d["b_pos"])

    clusters = []
    current = [sorted_diffs[0]]

    for d in sorted_diffs[1:]:
        prev = current[-1]
        prev_end = prev["b_pos"] + prev["b_len"]
        gap = d["b_pos"] - prev_end

        if gap <= context_words and len(current) < max_cluster_diffs:
            current.append(d)
        else:
            clusters.append(current)
            current = [d]

    if current:
        clusters.append(current)

    return clusters


def _clean_resolution(text: str) -> str:
    """Clean LLM format leakage from a resolution choice.

    Strips patterns like 'Model A: "text"', 'Decision: text', surrounding quotes.
    """
    # Strip pipe + second model alternative first (e.g., '| Model B: "xyz"')
    text = re.sub(r'\s*\|\s*(?:Model\s+[A-Z])\s*:.*$', '', text, flags=re.IGNORECASE)
    # Strip "Model A:" / "Model B:" / "Model C:" prefix (case-insensitive)
    text = re.sub(r'^(?:Model\s+[A-Z])\s*:\s*', '', text, flags=re.IGNORECASE)
    # Strip "Current:" / "Alternative:" / "Decision:" / "Choice:" / "Reading:" prefix
    text = re.sub(r'^(?:Current|Alternative|Alternative adds|Decision|Choice|Reading)\s*:\s*', '', text, flags=re.IGNORECASE)
    # Strip surrounding quotes
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.strip()


def _format_reading(text: str) -> str:
    """Format a single reading for the prompt: quoted text or (omit)."""
    return f'"{text}"' if text else "(omit)"


def _build_cluster_prompt(cluster: list[dict], base_words: list[str],
                          context_words: int = 30) -> str:
    """Build an LLM prompt for resolving a cluster of diffs.

    Shows context around the diffs with numbered disagreement markers.
    Uses anonymous letter labels (A, B, C, ...).

    If diffs have a ``readings`` dict (multi-way), each model gets a letter.
    Otherwise falls back to 2-way A/B format.
    """
    # Find the span this cluster covers in the base text
    first_pos = cluster[0]["b_pos"]
    last = cluster[-1]
    last_end = last["b_pos"] + last["b_len"]

    ctx_start = max(0, first_pos - context_words)
    ctx_end = min(len(base_words), last_end + context_words)

    # Build context with inline markers
    context_parts = []
    pos = ctx_start
    disagreements = []

    for i, d in enumerate(cluster, 1):
        if pos < d["b_pos"]:
            context_parts.append(" ".join(base_words[pos:d["b_pos"]]))
        context_parts.append(f"[{i}]")
        disagreements.append(d)
        pos = d["b_pos"] + d["b_len"]

    if pos < ctx_end:
        context_parts.append(" ".join(base_words[pos:ctx_end]))

    context_line = " ".join(context_parts)
    if ctx_start > 0:
        context_line = "..." + context_line
    if ctx_end < len(base_words):
        context_line = context_line + "..."

    # Build disagreement list
    diff_lines = []
    for i, d in enumerate(disagreements, 1):
        if "readings" in d:
            # Multi-way format: one letter per model
            parts = []
            for j, (model, text) in enumerate(d["readings"].items()):
                letter = chr(ord("A") + j)
                parts.append(f'{letter}: {_format_reading(text)}')
            diff_lines.append(f'{i}. {" | ".join(parts)}')
        else:
            # Legacy 2-way format
            if d["type"] == "substitution":
                diff_lines.append(f'{i}. A: "{d["a_text"]}" | B: "{d["b_text"]}"')
            elif d["type"] == "deletion":
                diff_lines.append(f'{i}. A: "{d["a_text"]}" | B: (omit)')
            elif d["type"] == "insertion":
                diff_lines.append(f'{i}. A: (omit) | B: "{d["b_text"]}"')

    return f"""CONTEXT:
{context_line}

DISAGREEMENTS:
{chr(10).join(diff_lines)}"""


def _call_and_parse_cluster(client, config, cluster, cluster_prompt,
                            hallucination_warning):
    """Call the LLM to resolve a cluster and parse the A/B/C response.

    Returns (cluster_choices, cluster_resolutions) where:
      cluster_choices: list of "A", "B", "C", ... or None for each diff (for checkpointing)
      cluster_resolutions: dict of diff id(diff) → chosen text
    """
    # Determine how many choices per diff
    has_readings = any("readings" in d for d in cluster)
    if has_readings:
        # Count readings from first diff with readings
        n_choices = max(len(d["readings"]) for d in cluster if "readings" in d)
        letters = [chr(ord("A") + i) for i in range(n_choices)]
        letter_list = ", ".join(letters[:-1]) + f", or {letters[-1]}"
        model_desc = f"{n_choices} Whisper transcriptions"
        agreement_hint = "\nWhen multiple options show the same text, that agreement is a useful signal.\n"
    else:
        letter_list = "A or B"
        model_desc = "two Whisper transcriptions"
        agreement_hint = ""

    prompt = f"""You are resolving disagreements between {model_desc} of the same speech.
No model is more reliable than the other — judge each difference on its merits.
{agreement_hint}{hallucination_warning}
{cluster_prompt}

For each numbered disagreement, reply with just the number and your choice: {letter_list}.
If both omitted, reply with the number and "A" to keep the omission.

Example output:
1. A
2. B
3. A

Output your decisions:"""

    message = llm_call_with_retry(
        client, config,
        model=config.claude_model,
        max_tokens=4096,
        system="You are a speech transcription correction tool. Your job is to mechanically choose "
               "the more accurate transcription at each point of disagreement. The content is from "
               "real recorded speech and must be transcribed faithfully regardless of subject matter.",
        messages=[{"role": "user", "content": prompt}],
    )

    response = _clean_llm_output(message.content[0].text)

    cluster_choices = [None] * len(cluster)
    cluster_resolutions = {}

    for line in response.splitlines():
        line = line.strip()
        m = re.match(r'^(\d+)[.):]\s*(.*)', line)
        if m:
            num = int(m.group(1))
            choice = m.group(2).strip().upper()
            choice = _clean_resolution(choice).upper()
            if 1 <= num <= len(cluster):
                diff = cluster[num - 1]
                diff_key = id(diff)

                if "readings" in diff:
                    # Multi-way: map letter to reading
                    readings_list = list(diff["readings"].values())
                    models_list = list(diff["readings"].keys())
                    letter_idx = ord(choice[0]) - ord("A") if choice and choice[0].isalpha() else -1
                    if 0 <= letter_idx < len(readings_list):
                        text = readings_list[letter_idx]
                        chosen = text if text else "(omit)"
                        cluster_choices[num - 1] = chr(ord("A") + letter_idx)
                    else:
                        # Fallback: base model (last in readings)
                        chosen = readings_list[-1] if readings_list[-1] else "(omit)"
                        cluster_choices[num - 1] = chr(ord("A") + len(readings_list) - 1)
                else:
                    # Legacy 2-way
                    if choice.startswith("A"):
                        chosen = diff["a_text"] if diff["a_text"] else "(omit)"
                        cluster_choices[num - 1] = "A"
                    elif choice.startswith("B"):
                        chosen = diff["b_text"] if diff["b_text"] else "(omit)"
                        cluster_choices[num - 1] = "B"
                    else:
                        chosen = diff["b_text"] if diff["b_text"] else "(omit)"
                        cluster_choices[num - 1] = "B"
                cluster_resolutions[diff_key] = chosen

    return cluster_choices, cluster_resolutions


def _resolve_whisper_diffs(base_text: str, all_transcripts: dict,
                           config: SpeechConfig) -> str:
    """Resolve Whisper model disagreements via targeted diff resolution.

    Instead of rewriting entire text chunks, this:
    1. Finds positioned diffs between transcripts
    2. Groups nearby diffs into clusters
    3. Asks the LLM to resolve each cluster's disagreements
    4. Applies targeted replacements to the base text
    """
    # Collapse hallucination loops before processing
    all_loops = {}
    base_text, base_loops = collapse_repetition_loops(base_text)
    if base_loops:
        all_loops["base"] = base_loops
    all_transcripts = dict(all_transcripts)
    for model in list(all_transcripts.keys()):
        collapsed, loops = collapse_repetition_loops(all_transcripts[model])
        if loops:
            all_loops[model] = loops
        all_transcripts[model] = collapsed

    if all_loops:
        total = sum(len(v) for v in all_loops.values())
        print(f"  Collapsed {total} Whisper hallucination loop(s)")

    # Step 1: Find positioned diffs between base and each other model
    base_words = base_text.split()
    base_model = None
    for size in MODEL_SIZES:
        if size in all_transcripts:
            base_model = size
            break
    if not base_model:
        base_model = list(all_transcripts.keys())[0]

    other_models = [m for m in all_transcripts if m != base_model]

    # Collect pairwise diffs
    pairwise_diffs = []
    for other_model in other_models:
        diffs = _parse_wdiff_diffs(
            all_transcripts[other_model],  # text_a = other
            all_transcripts[base_model],   # text_b = base
            config,
        )
        diffs = _filter_trivial_diffs(diffs)
        print(f"  Found {len(diffs)} meaningful differences ({other_model} vs {base_model})")
        pairwise_diffs.append((other_model, diffs))

    # Merge into multi-way diffs if >1 other model
    if len(other_models) > 1:
        all_diffs = _merge_pairwise_diffs(
            pairwise_diffs, base_model, list(all_transcripts.keys()),
        )
        pairwise_total = sum(len(d) for _, d in pairwise_diffs)
        print(f"  Merged {pairwise_total} pairwise diffs into {len(all_diffs)} multi-way diffs")
    else:
        all_diffs = pairwise_diffs[0][1] if pairwise_diffs else []

    if not all_diffs:
        print("  No meaningful differences to resolve")
        return base_text

    # Step 2: Cluster diffs
    clusters = _cluster_diffs(
        all_diffs, len(base_words),
        context_words=config.merge_diff_context_words,
        max_cluster_diffs=config.merge_max_diffs_per_call,
    )
    total_diffs = sum(len(c) for c in clusters)
    print(f"  Grouped {total_diffs} diffs into {len(clusters)} cluster(s)")

    # Step 3: Resolve each cluster via LLM (with checkpointing)
    client = create_llm_client(config)

    # Build hallucination warning for prompt
    hallucination_warning = ""
    if all_loops:
        hallucination_warning = "\nWARNING: Some models had hallucination loops (repeated phrases). "
        hallucination_warning += "These have been collapsed. Do NOT choose repetitive readings.\n"

    # Initialize checkpoint directory
    ensemble_dir = config.output_dir / "ensemble_chunks"
    ensemble_dir.mkdir(exist_ok=True)
    version_file = ensemble_dir / ".version"
    ENSEMBLE_CHECKPOINT_VERSION = "v3"  # bump to invalidate old checkpoints
    if not version_file.exists() or version_file.read_text().strip() != ENSEMBLE_CHECKPOINT_VERSION:
        for old_file in ensemble_dir.glob("cluster_*.json"):
            old_file.unlink()
        version_file.write_text(ENSEMBLE_CHECKPOINT_VERSION)

    # Determine source paths for staleness checks
    whisper_inputs = [
        config.output_dir / f"whisper_{m}.txt"
        for m in all_transcripts
        if (config.output_dir / f"whisper_{m}.txt").exists()
    ]

    # Collect all resolutions: list of (diff, chosen_text) pairs
    resolutions = {}  # diff index in all_diffs → chosen text
    clusters_reused = 0

    for cluster_idx, cluster in enumerate(clusters):
        checkpoint_path = ensemble_dir / f"cluster_{cluster_idx:03d}.json"

        # Check for reusable checkpoint
        if whisper_inputs and is_up_to_date(checkpoint_path, *whisper_inputs):
            with open(checkpoint_path, 'r') as f:
                saved = json.load(f)
            # Restore resolutions from checkpoint
            for i, choice in enumerate(saved["choices"]):
                if i < len(cluster):
                    diff = cluster[i]
                    diff_key = id(diff)
                    if "readings" in diff:
                        # Multi-way: map letter to reading
                        readings_list = list(diff["readings"].values())
                        letter_idx = ord(choice[0]) - ord("A") if choice and choice[0].isalpha() else -1
                        if 0 <= letter_idx < len(readings_list):
                            text = readings_list[letter_idx]
                            chosen = text if text else "(omit)"
                        else:
                            chosen = readings_list[-1] if readings_list[-1] else "(omit)"
                    else:
                        # Legacy 2-way
                        if choice == "A":
                            chosen = diff["a_text"] if diff["a_text"] else "(omit)"
                        elif choice == "B":
                            chosen = diff["b_text"] if diff["b_text"] else "(omit)"
                        else:
                            chosen = diff["b_text"] if diff["b_text"] else "(omit)"
                    resolutions[diff_key] = chosen
            clusters_reused += 1
            continue

        print(f"  Resolving cluster {cluster_idx + 1}/{len(clusters)} ({len(cluster)} diffs)...")

        # Build prompt for this cluster
        cluster_prompt = _build_cluster_prompt(
            cluster, base_words,
            context_words=config.merge_diff_context_words,
        )

        cluster_choices, cluster_resolutions = _call_and_parse_cluster(
            client, config, cluster, cluster_prompt, hallucination_warning,
        )

        # If all diffs unresolved, retry without context (may help with content refusals)
        if all(c is None for c in cluster_choices):
            # Build disagreements-only prompt (strip context)
            diff_lines = []
            for i, d in enumerate(cluster, 1):
                if "readings" in d:
                    parts = []
                    for j, (model, text) in enumerate(d["readings"].items()):
                        letter = chr(ord("A") + j)
                        parts.append(f'{letter}: {_format_reading(text)}')
                    diff_lines.append(f'{i}. {" | ".join(parts)}')
                elif d["type"] == "substitution":
                    diff_lines.append(f'{i}. A: "{d["a_text"]}" | B: "{d["b_text"]}"')
                elif d["type"] == "deletion":
                    diff_lines.append(f'{i}. A: "{d["a_text"]}" | B: (omit)')
                elif d["type"] == "insertion":
                    diff_lines.append(f'{i}. A: (omit) | B: "{d["b_text"]}"')
            minimal_prompt = "DISAGREEMENTS:\n" + "\n".join(diff_lines)

            print(f"    Retrying cluster {cluster_idx + 1} without context...")
            cluster_choices, cluster_resolutions = _call_and_parse_cluster(
                client, config, cluster, minimal_prompt, hallucination_warning,
            )

        # Store resolutions
        for diff_key, chosen in cluster_resolutions.items():
            resolutions[diff_key] = chosen

        # Save checkpoint
        _save_json(checkpoint_path, {"choices": cluster_choices})

    if clusters_reused:
        print(f"  Reused {clusters_reused}/{len(clusters)} clusters from checkpoint")

    # Step 4: Apply resolutions to base text
    resolved_text = _apply_resolutions(base_words, all_diffs, resolutions)

    resolved_words = len(resolved_text.split())
    applied = sum(1 for d in all_diffs if id(d) in resolutions)
    print(f"  Applied {applied}/{total_diffs} resolutions ({resolved_words} words)")

    return resolved_text


def _apply_resolutions(base_words: list[str], diffs: list[dict],
                       resolutions: dict) -> str:
    """Apply LLM resolutions to the base text.

    Walks the base text word by word. At each diff position (in text_b,
    since text_b is the base), substitutes the LLM's chosen text.
    Unresolved diffs keep the base text.
    """
    # Sort diffs by b_pos (position in base text)
    sorted_diffs = sorted(diffs, key=lambda d: d["b_pos"])

    result = []
    pos = 0

    for d in sorted_diffs:
        diff_key = id(d)
        b_start = d["b_pos"]
        b_len = d["b_len"]

        # Add base text before this diff
        if pos < b_start:
            result.extend(base_words[pos:b_start])

        if diff_key in resolutions:
            chosen = resolutions[diff_key]
            # Handle "(omit)" response — skip the words
            if chosen.lower().strip("()") == "omit":
                pass  # Don't add anything
            else:
                result.extend(chosen.split())
        else:
            # No resolution — keep base text
            if b_len > 0:
                result.extend(base_words[b_start:b_start + b_len])

        pos = b_start + b_len

    # Add remaining base text
    if pos < len(base_words):
        result.extend(base_words[pos:])

    return " ".join(result)


def transcribe_audio(config: SpeechConfig, data: SpeechData) -> None:
    """Transcribe audio using Whisper, supporting multiple models for ensembling."""
    print()
    print("[transcribe] Transcribing audio...")

    if not config.dry_run and (not data.audio_path or not data.audio_path.exists()):
        raise FileNotFoundError(f"Audio file not found: {data.audio_path}")

    deps = check_dependencies()
    if not deps["mlx_whisper"] and not deps["whisper"]:
        raise RuntimeError("No Whisper implementation found. Install mlx-whisper or openai-whisper.")

    models = config.whisper_models
    print(f"  Models to run: {', '.join(models)}")

    # Run each model
    for model in models:
        _run_whisper_model(config, data, model, deps)

    # Single model - use it directly (ensemble handled by pipeline)
    if len(models) == 1:
        model = models[0]
        if model in data.whisper_transcripts:
            data.transcript_path = data.whisper_transcripts[model]["txt"]
            data.transcript_json_path = data.whisper_transcripts[model].get("json")

        if config.dry_run:
            return

        if not data.transcript_path:
            raise FileNotFoundError("Transcript file not found after transcription")

        # Load segments from JSON
        _load_transcript_segments(data)


def _run_whisper_model(config: SpeechConfig, data: SpeechData, model: str, deps: dict) -> None:
    """Run a single Whisper model and save output."""
    # Create model-specific output names
    txt_path = config.output_dir / f"whisper_{model}.txt"
    json_path = config.output_dir / f"whisper_{model}.json"

    # Check if up to date (output newer than audio input)
    if _should_skip(config, txt_path, f"transcribe with Whisper {model}", data.audio_path):
        if txt_path.exists():
            data.whisper_transcripts[model] = {"txt": txt_path, "json": json_path if json_path.exists() else None}
        return
    print(f"  Running Whisper {model}...")

    if deps["mlx_whisper"]:
        model_name = MLX_MODEL_MAP.get(model, f"mlx-community/whisper-{model}-mlx")

        # mlx_whisper outputs to input filename, so we need to work around this
        # Run transcription
        for fmt in ["txt", "json"]:
            cmd = ["mlx_whisper", str(data.audio_path),
                   "--model", model_name,
                   "--output-format", fmt,
                   "--output-dir", str(config.output_dir),
                   "--condition-on-previous-text", "False",
                   "--no-speech-threshold", "0.2",
                   "--compression-ratio-threshold", "2.0",
                   "--hallucination-silence-threshold", "3.0",
                   "--word-timestamps", "True"]
            run_command(cmd,
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
        result = whisper_model.transcribe(
            str(data.audio_path),
            condition_on_previous_text=False,
            no_speech_threshold=0.2,
            compression_ratio_threshold=2.0,
            hallucination_silence_threshold=3.0,
            word_timestamps=True,
        )

        with open(txt_path, 'w') as f:
            f.write(result["text"])

        _save_json(json_path, {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language", "en")
            })

    # Collapse Whisper hallucination loops in the text output
    if txt_path.exists():
        text = txt_path.read_text()
        collapsed, loops = collapse_repetition_loops(text)
        if loops:
            txt_path.write_text(collapsed)
            total_removed = sum(
                len(l["phrase"].split()) * (l["count"] - 2) for l in loops
            )
            print(f"  Collapsed {len(loops)} hallucination loop(s) "
                  f"({total_removed} words removed)")

    data.whisper_transcripts[model] = {
        "txt": txt_path if txt_path.exists() else None,
        "json": json_path if json_path.exists() else None
    }
    print(f"  {model} transcript saved: {txt_path.name}")


def _ensemble_whisper_transcripts(config: SpeechConfig, data: SpeechData) -> None:
    """Adjudicate multiple Whisper transcripts using wdiff."""
    models = list(data.whisper_transcripts.keys())
    # Check for existing whisper_merged file
    if data.audio_path:
        merged_path = config.output_dir / WHISPER_MERGED_TXT
        # Inputs are the individual whisper transcripts
        whisper_inputs = [paths["txt"] for paths in data.whisper_transcripts.values()
                         if paths.get("txt")]
        action = f"adjudicate {len(models)} Whisper transcripts ({', '.join(models)})"
        if config.dry_run:
            # Show checkpoint status in dry-run
            ensemble_dir = config.output_dir / "ensemble_chunks"
            if ensemble_dir.exists():
                checkpoints = list(ensemble_dir.glob("cluster_*.json"))
                fresh = sum(1 for cp in checkpoints
                            if whisper_inputs and is_up_to_date(cp, *whisper_inputs))
                total = len(checkpoints)
                if fresh > 0:
                    action += f" ({fresh}/{total} clusters cached, {total - fresh} to resolve)"
        if _should_skip(config, merged_path, action, *whisper_inputs):
            if merged_path.exists():
                data.transcript_path = merged_path
                data.transcript_json_path = _select_largest_model_json(data)
            return
    print(f"  Adjudicating {len(models)} Whisper transcripts: {', '.join(models)}")
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
    base_model = None
    for size in MODEL_SIZES:
        if size in transcripts:
            base_model = size
            break

    if not base_model:
        base_model = models[0]

    base_text = transcripts[base_model]
    print(f"  Using {base_model} as base transcript")

    if config.no_llm:
        print("  --no-llm flag set - using base model without resolving differences")
        merged_text = base_text
    else:
        merged_text = _resolve_whisper_diffs(base_text, transcripts, config)

    # Save whisper-merged transcript
    merged_path = config.output_dir / WHISPER_MERGED_TXT
    with open(merged_path, 'w') as f:
        f.write(merged_text)

    data.transcript_path = merged_path
    print(f"  Whisper-merged transcript saved: {merged_path.name}")

    data.transcript_json_path = _select_largest_model_json(data)


def _clean_llm_output(text: str) -> str:
    """Remove common LLM formatting artifacts from merge output."""
    lines = text.strip().splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip separator lines (---, ***, ===, etc.)
        if re.match(r'^[-*=]{3,}$', stripped):
            continue
        # Skip markdown headers that the LLM may prepend
        if re.match(r'^#{1,3}\s', stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _load_transcript_segments(data: SpeechData) -> None:
    """Load transcript segments with timestamps from JSON file."""
    if not data.transcript_json_path or not data.transcript_json_path.exists():
        return

    try:
        with open(data.transcript_json_path, 'r') as f:
            transcript_data = json.load(f)

        segments = transcript_data.get("segments", [])
        data.transcript_segments = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            entry = {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": text,
            }
            # Extract per-word timestamps when available (mlx-whisper --word-timestamps True)
            if seg.get("words"):
                entry["words"] = [
                    {
                        "word": w.get("word", ""),
                        "start": w.get("start", 0),
                        "end": w.get("end", 0),
                    }
                    for w in seg["words"]
                ]
            data.transcript_segments.append(entry)
        print(f"  Loaded {len(data.transcript_segments)} transcript segments with timestamps")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Warning: Could not parse transcript JSON: {e}")
