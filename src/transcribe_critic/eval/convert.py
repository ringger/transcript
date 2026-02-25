"""Format converters between transcribe-critic outputs and meeteval inputs.

Supported conversions:
- Earnings-21 NLP → STM (reference transcripts)
- diarized.txt → STM (hypothesis with speaker labels)
- structured transcript_merged.txt → STM (hypothesis with speaker labels)
- plain text → STM (single-speaker hypothesis)
- diarization.json → RTTM (hypothesis speaker segments)
"""

import json
import re
from pathlib import Path

from transcribe_critic.shared import (
    WHISPER_MERGED_TXT, DIARIZED_TXT, TRANSCRIPT_MERGED_TXT, DIARIZATION_JSON,
)


def nlp_to_stm(nlp_path: Path, file_id: str) -> str:
    """Convert Earnings-21 .nlp reference file to STM format.

    NLP columns: token|speaker|ts|endTs|punctuation|case|tags|wer_tags
    Note: ts and endTs are always empty in nlp_references; we use synthetic
    timestamps based on token position. cpWER only needs correct speaker
    assignment, not precise timing.

    STM format: filename channel speaker start end transcript
    """
    lines = nlp_path.read_text().splitlines()
    if not lines:
        return ""

    # Skip header row
    if lines[0].startswith("token|"):
        lines = lines[1:]

    # Parse tokens and group by speaker
    segments = []
    current_speaker = None
    current_tokens = []
    token_index = 0

    for line in lines:
        parts = line.split("|")
        if len(parts) < 6:
            continue

        token, speaker, _ts, _end_ts, punct, case_code = parts[:6]

        # Apply casing
        if case_code == "UC":
            token = token[0].upper() + token[1:] if len(token) > 1 else token.upper()
        elif case_code == "LC":
            token = token.lower()
        elif case_code == "CA":
            token = token.upper()
        # MC = mixed case, keep as-is

        # Append punctuation
        if punct and punct not in ("", "<NA>"):
            token = token + punct

        # Break segment on speaker change
        if speaker != current_speaker:
            if current_tokens:
                text = " ".join(current_tokens)
                start = token_index - len(current_tokens)
                segments.append((file_id, f"speaker_{current_speaker}", start, token_index, text))
            current_speaker = speaker
            current_tokens = [token]
        else:
            current_tokens.append(token)

        token_index += 1

    # Flush last segment
    if current_tokens:
        text = " ".join(current_tokens)
        start = token_index - len(current_tokens)
        segments.append((file_id, f"speaker_{current_speaker}", start, token_index, text))

    # Emit STM lines with synthetic timestamps (1 token ≈ 0.3s)
    stm_lines = []
    for fid, speaker, start_idx, end_idx, text in segments:
        start_t = start_idx * 0.3
        end_t = end_idx * 0.3
        stm_lines.append(f"{fid} 1 {speaker} {start_t:.3f} {end_t:.3f} {text}")

    return "\n".join(stm_lines)


def diarized_txt_to_stm(path: Path, file_id: str) -> str:
    """Convert transcribe-critic diarized.txt to STM format.

    Input format: [H:MM:SS] Speaker Name: text
    Segments are separated by blank lines.
    """
    text = path.read_text().strip()
    if not text:
        return ""

    pattern = re.compile(r"^\[(\d+):(\d{2}):(\d{2})\]\s+(.+?):\s+(.*)", re.DOTALL)

    # Split on blank lines to get segments
    raw_segments = re.split(r"\n\n+", text)
    segments = []

    for block in raw_segments:
        block = block.strip()
        if not block:
            continue
        m = pattern.match(block)
        if m:
            h, mm, ss, speaker, body = m.groups()
            start = int(h) * 3600 + int(mm) * 60 + int(ss)
            # Body may span multiple lines within the block
            body = " ".join(body.split())
            safe_speaker = re.sub(r"[^a-zA-Z0-9_]", "_", speaker.strip())
            segments.append((start, safe_speaker, body))

    stm_lines = []
    for i, (start, speaker, body) in enumerate(segments):
        if i + 1 < len(segments):
            end = segments[i + 1][0]
        else:
            # Estimate end from word count (~2.5 words/sec)
            end = start + max(1, len(body.split()) / 2.5)
        stm_lines.append(f"{file_id} 1 {speaker} {start:.3f} {end:.3f} {body}")

    return "\n".join(stm_lines)


def structured_merged_to_stm(path: Path, file_id: str) -> str:
    """Convert structured transcript_merged.txt to STM format.

    Input format:
        **Speaker Name** (HH:MM:SS)

        Paragraph text...

        **Other Speaker** (HH:MM:SS)

        More text...
    """
    text = path.read_text().strip()
    if not text:
        return ""

    header_pattern = re.compile(
        r"^\*\*(.+?)\*\*\s*\((\d+):(\d{2}):(\d{2})\)\s*$", re.MULTILINE
    )

    segments = []
    matches = list(header_pattern.finditer(text))

    for i, m in enumerate(matches):
        speaker = m.group(1).strip()
        h, mm, ss = int(m.group(2)), int(m.group(3)), int(m.group(4))
        start = h * 3600 + mm * 60 + ss

        # Text runs from after this header to before the next header
        text_start = m.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[text_start:text_end].strip()
        body = " ".join(body.split())  # normalize whitespace

        if body:
            safe_speaker = re.sub(r"[^a-zA-Z0-9_]", "_", speaker)
            segments.append((start, safe_speaker, body))

    stm_lines = []
    for i, (start, speaker, body) in enumerate(segments):
        if i + 1 < len(segments):
            end = segments[i + 1][0]
        else:
            end = start + max(1, len(body.split()) / 2.5)
        stm_lines.append(f"{file_id} 1 {speaker} {start:.3f} {end:.3f} {body}")

    return "\n".join(stm_lines)


def plain_text_to_stm(path: Path, file_id: str, speaker: str = "unknown") -> str:
    """Convert a plain text transcript to single-speaker STM.

    Used for Whisper .txt output or flat transcript_merged.txt without
    speaker labels. Only suitable for WER (not cpWER or DER).
    """
    text = path.read_text().strip()
    if not text:
        return ""
    # Collapse whitespace, emit as a single segment
    text = " ".join(text.split())
    return f"{file_id} 1 {speaker} 0.000 999999.000 {text}"


def diarization_json_to_rttm(json_path: Path, file_id: str) -> str:
    """Convert transcribe-critic diarization.json to RTTM format.

    Input: [{start, end, speaker}, ...]
    Output: SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    """
    segments = json.loads(json_path.read_text())
    lines = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        speaker = seg["speaker"]
        lines.append(
            f"SPEAKER {file_id} 1 {seg['start']:.3f} {duration:.3f} "
            f"<NA> <NA> {speaker} <NA> <NA>"
        )
    return "\n".join(lines)


def detect_hypothesis_format(output_dir: Path) -> str:
    """Detect which hypothesis format is available in a pipeline output directory.

    Returns one of: 'diarized', 'structured_merged', 'plain_merged', 'whisper'
    """
    diarized = output_dir / DIARIZED_TXT
    merged = output_dir / TRANSCRIPT_MERGED_TXT
    whisper_merged = output_dir / WHISPER_MERGED_TXT

    if diarized.exists():
        return "diarized"

    if merged.exists():
        text = merged.read_text().strip()
        if re.search(r"^\*\*.+?\*\*\s*\(\d+:\d{2}:\d{2}\)", text, re.MULTILINE):
            return "structured_merged"
        return "plain_merged"

    if whisper_merged.exists():
        return "whisper"

    # Fall back to any Whisper model output
    skip = {DIARIZED_TXT, TRANSCRIPT_MERGED_TXT, WHISPER_MERGED_TXT}
    for txt in sorted(output_dir.glob("*.txt")):
        if txt.name not in skip:
            return "whisper"

    return "none"


def hypothesis_to_stm(output_dir: Path, file_id: str, hypothesis: str = "auto") -> str:
    """Convert the best available hypothesis from a pipeline output dir to STM.

    Args:
        output_dir: Directory containing transcribe-critic pipeline outputs.
        file_id: Identifier for STM file field.
        hypothesis: Which output to use: 'auto', 'merged', 'diarized', 'whisper'.

    Returns:
        STM-formatted string.
    """
    if hypothesis == "auto":
        fmt = detect_hypothesis_format(output_dir)
    elif hypothesis == "merged":
        merged = output_dir / "transcript_merged.txt"
        if not merged.exists():
            raise FileNotFoundError(f"No transcript_merged.txt in {output_dir}")
        text = merged.read_text().strip()
        if re.search(r"^\*\*.+?\*\*\s*\(\d+:\d{2}:\d{2}\)", text, re.MULTILINE):
            fmt = "structured_merged"
        else:
            fmt = "plain_merged"
    elif hypothesis == "diarized":
        fmt = "diarized"
    elif hypothesis == "whisper":
        fmt = "whisper"
    else:
        raise ValueError(f"Unknown hypothesis type: {hypothesis}")

    if fmt == "diarized":
        return diarized_txt_to_stm(output_dir / DIARIZED_TXT, file_id)
    elif fmt == "structured_merged":
        return structured_merged_to_stm(output_dir / TRANSCRIPT_MERGED_TXT, file_id)
    elif fmt == "plain_merged":
        return plain_text_to_stm(output_dir / TRANSCRIPT_MERGED_TXT, file_id)
    elif fmt == "whisper":
        # Prefer whisper_merged, then medium, then small
        for name in (WHISPER_MERGED_TXT, "whisper_medium.txt", "whisper_small.txt"):
            p = output_dir / name
            if p.exists():
                return plain_text_to_stm(p, file_id)
        raise FileNotFoundError(f"No Whisper output found in {output_dir}")
    else:
        raise FileNotFoundError(f"No hypothesis output found in {output_dir}")
