"""Score pipeline outputs against references using meeteval.

Computes WER, cpWER, and DER metrics per file and in aggregate.
Supports scoring all available transcript variants (individual Whisper
models, whisper_merged, transcript_merged) for comparison.
"""

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transcribe_critic.shared import DIARIZATION_JSON, WHISPER_MERGED_TXT, TRANSCRIPT_MERGED_TXT
from transcribe_critic.eval.convert import (
    hypothesis_to_stm, plain_text_to_stm, diarization_json_to_rttm,
)
from transcribe_critic.eval.datasets import load_manifest, filter_manifest, ManifestEntry


@dataclass
class FileResult:
    """Scoring results for a single file, one hypothesis variant."""
    file_id: str
    hypothesis_name: str = ""
    duration_secs: float = 0.0
    wer: Optional[float] = None
    cpwer: Optional[float] = None
    der: Optional[float] = None
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _run_meeteval_wer(ref_stm: Path, hyp_stm: Path, metric: str = "cpwer") -> Optional[float]:
    """Run meeteval-wer and parse the result."""
    try:
        result = subprocess.run(
            ["meeteval-wer", metric, "-r", str(ref_stm), "-h", str(hyp_stm),
             "--per-reco-out", "-"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"  meeteval-wer {metric} failed: {result.stderr.strip()}", file=sys.stderr)
            return None

        data = json.loads(result.stdout)
        # --per-reco-out produces {file_id: {error_rate: ...}}
        if isinstance(data, dict):
            for file_id, scores in data.items():
                if isinstance(scores, dict):
                    return scores.get("error_rate")
                return scores
        return None
    except FileNotFoundError:
        print("Error: meeteval-wer not found. Install with: pip install meeteval", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  meeteval error: {e}", file=sys.stderr)
        return None


def _compute_plain_wer(ref_stm: Path, hyp_stm: Path) -> Optional[float]:
    """Compute WER by collapsing all speakers into one (ignoring diarization)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".stm", delete=False) as rf, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".stm", delete=False) as hf:

        for line in ref_stm.read_text().splitlines():
            parts = line.split(None, 5)
            if len(parts) >= 6:
                parts[2] = "spk"
                rf.write(" ".join(parts) + "\n")

        for line in hyp_stm.read_text().splitlines():
            parts = line.split(None, 5)
            if len(parts) >= 6:
                parts[2] = "spk"
                hf.write(" ".join(parts) + "\n")

        rf.flush()
        hf.flush()

        result = _run_meeteval_wer(Path(rf.name), Path(hf.name), "cpwer")

        Path(rf.name).unlink(missing_ok=True)
        Path(hf.name).unlink(missing_ok=True)

        return result


def _run_meeteval_der(ref_rttm: Path, hyp_rttm: Path, collar: float = 0.25) -> Optional[float]:
    """Run meeteval-der and parse the result."""
    try:
        result = subprocess.run(
            ["meeteval-der", "md_eval_22", "-r", str(ref_rttm), "-h", str(hyp_rttm),
             "--collar", str(collar), "--per-reco-out", "-"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"  meeteval-der failed: {result.stderr.strip()}", file=sys.stderr)
            return None

        data = json.loads(result.stdout)
        if isinstance(data, dict):
            for file_id, scores in data.items():
                if isinstance(scores, dict):
                    return scores.get("error_rate")
                return scores
        return None
    except FileNotFoundError:
        print("Error: meeteval-der not found. Install with: pip install meeteval", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  meeteval-der error: {e}", file=sys.stderr)
        return None


def _discover_hypotheses(output_dir: Path) -> list[tuple[str, Path]]:
    """Discover all available transcript variants in an output directory.

    Returns list of (name, path) tuples for each scorable transcript.
    """
    hypotheses = []

    # Individual Whisper model outputs
    for model in ("tiny", "base", "small", "medium", "large"):
        txt = output_dir / f"whisper_{model}.txt"
        if txt.exists():
            hypotheses.append((f"whisper_{model}", txt))

    # Whisper-merged (adjudicated from multiple models)
    wm = output_dir / WHISPER_MERGED_TXT
    if wm.exists():
        hypotheses.append(("whisper_merged", wm))

    # Source-merged (critical text from all sources)
    tm = output_dir / TRANSCRIPT_MERGED_TXT
    if tm.exists():
        hypotheses.append(("transcript_merged", tm))

    return hypotheses


def _score_stm(
    ref_stm_path: Path,
    hyp_stm_content: str,
    entry: ManifestEntry,
    dataset_dir: Path,
    output_dir: Path,
    metrics: list[str],
    hypothesis_name: str,
) -> FileResult:
    """Score a single hypothesis STM against the reference."""
    result = FileResult(
        file_id=entry.file_id,
        hypothesis_name=hypothesis_name,
        duration_secs=entry.duration_secs,
        metadata=entry.metadata,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".stm", delete=False) as f:
        f.write(hyp_stm_content)
        hyp_stm_path = Path(f.name)

    try:
        if "wer" in metrics:
            result.wer = _compute_plain_wer(ref_stm_path, hyp_stm_path)

        if "cpwer" in metrics:
            if entry.ref_rttm_path is not None:
                result.cpwer = _run_meeteval_wer(ref_stm_path, hyp_stm_path, "cpwer")
            else:
                result.errors.append("cpWER skipped: no speaker labels in reference")

        if "der" in metrics:
            if entry.ref_rttm_path:
                ref_rttm_path = dataset_dir / entry.ref_rttm_path
                hyp_diar_json = output_dir / DIARIZATION_JSON
                if hyp_diar_json.exists():
                    hyp_rttm_content = diarization_json_to_rttm(hyp_diar_json, entry.file_id)
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".rttm", delete=False) as rf:
                        rf.write(hyp_rttm_content)
                        hyp_rttm_path = Path(rf.name)
                    try:
                        result.der = _run_meeteval_der(ref_rttm_path, hyp_rttm_path)
                    finally:
                        hyp_rttm_path.unlink(missing_ok=True)
                else:
                    result.errors.append("DER skipped: no diarization.json in hypothesis")
            else:
                result.errors.append("DER skipped: no reference RTTM")
    finally:
        hyp_stm_path.unlink(missing_ok=True)

    return result


def score_file(
    entry: ManifestEntry,
    dataset_dir: Path,
    run_dir: Path,
    metrics: list[str],
    hypothesis: str = "auto",
) -> list[FileResult]:
    """Score a single file's pipeline output(s) against its reference.

    When hypothesis='all', scores every available transcript variant.
    Returns a list of FileResult (one per variant scored).
    """
    output_dir = run_dir / entry.file_id

    if not output_dir.exists():
        r = FileResult(file_id=entry.file_id, duration_secs=entry.duration_secs,
                       metadata=entry.metadata)
        r.errors.append(f"Output directory not found: {output_dir}")
        return [r]

    ref_stm_path = dataset_dir / entry.ref_stm_path
    if not ref_stm_path.exists():
        r = FileResult(file_id=entry.file_id, duration_secs=entry.duration_secs,
                       metadata=entry.metadata)
        r.errors.append(f"Reference STM not found: {ref_stm_path}")
        return [r]

    if hypothesis == "all":
        # Score every available transcript variant
        variants = _discover_hypotheses(output_dir)
        if not variants:
            r = FileResult(file_id=entry.file_id, duration_secs=entry.duration_secs,
                           metadata=entry.metadata)
            r.errors.append("No transcript outputs found")
            return [r]

        results = []
        for name, path in variants:
            stm_content = plain_text_to_stm(path, entry.file_id)
            r = _score_stm(ref_stm_path, stm_content, entry, dataset_dir,
                           output_dir, metrics, name)
            results.append(r)
        return results
    else:
        # Score a single hypothesis
        try:
            hyp_stm_content = hypothesis_to_stm(output_dir, entry.file_id, hypothesis)
        except (FileNotFoundError, ValueError) as e:
            r = FileResult(file_id=entry.file_id, hypothesis_name=hypothesis,
                           duration_secs=entry.duration_secs, metadata=entry.metadata)
            r.errors.append(f"Hypothesis conversion failed: {e}")
            return [r]

        return [_score_stm(ref_stm_path, hyp_stm_content, entry, dataset_dir,
                           output_dir, metrics, hypothesis)]


def score_results(args):
    """Entry point for the score subcommand."""
    manifest_path = Path(args.manifest)
    dataset_dir = manifest_path.parent
    run_dir = Path(args.run_dir)
    metrics = [m.strip() for m in args.metrics.split(",")]

    file_ids = args.file_ids.split(",") if args.file_ids else None

    entries = load_manifest(manifest_path)
    entries = filter_manifest(
        entries,
        subset=args.subset,
        max_files=args.max_files,
        file_ids=file_ids,
    )

    if not entries:
        print("No files to score after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(entries)} files against {run_dir}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Hypothesis: {args.hypothesis}")
    print()

    all_results = []
    for entry in entries:
        print(f"  Scoring {entry.file_id}...")
        file_results = score_file(entry, dataset_dir, run_dir, metrics, args.hypothesis)
        for r in file_results:
            parts = [f"    {r.hypothesis_name}:"]
            if r.wer is not None:
                parts.append(f"WER={r.wer:.1%}")
            if r.cpwer is not None:
                parts.append(f"cpWER={r.cpwer:.1%}")
            if r.der is not None:
                parts.append(f"DER={r.der:.1%}")
            if r.errors:
                parts.append(f"({'; '.join(r.errors)})")
            print(" ".join(parts))
        all_results.extend(file_results)

    # Generate reports
    from transcribe_critic.eval.report import render_terminal, render_markdown, render_json

    print()
    render_terminal(all_results, metrics)

    # Write markdown and JSON
    output_path = Path(args.output) if args.output else run_dir / "results.md"
    md_content = render_markdown(all_results, metrics, run_dir=str(run_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_content)
    print(f"\nResults written to {output_path}")

    json_path = output_path.with_suffix(".json")
    json_content = render_json(all_results, metrics)
    json_path.write_text(json_content)
