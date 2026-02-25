"""Result formatting for terminal, markdown, and JSON output."""

import json
from collections import defaultdict
from datetime import datetime
from typing import Optional


def _fmt_duration(secs: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _fmt_pct(val: Optional[float]) -> str:
    """Format a float as percentage, or '-' if None."""
    if val is None:
        return "-"
    return f"{val:.1%}"


def _avg(values: list[Optional[float]]) -> Optional[float]:
    """Compute average of non-None values."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _weighted_avg(values: list[Optional[float]], weights: list[float]) -> Optional[float]:
    """Compute weighted average of non-None values."""
    total_w = 0.0
    total_v = 0.0
    for v, w in zip(values, weights):
        if v is not None:
            total_v += v * w
            total_w += w
    if total_w == 0:
        return None
    return total_v / total_w


def _group_by_hypothesis(results: list) -> dict[str, list]:
    """Group results by hypothesis_name. Returns {name: [results]}."""
    groups = defaultdict(list)
    for r in results:
        groups[r.hypothesis_name or ""].append(r)
    return dict(groups)


def _has_multiple_hypotheses(results: list) -> bool:
    """Check if results contain multiple hypothesis variants."""
    names = {r.hypothesis_name for r in results if r.hypothesis_name}
    return len(names) > 1


def render_terminal(results: list, metrics: list[str]):
    """Print results as a formatted terminal table.

    When results contain multiple hypothesis variants (from --hypothesis all),
    prints a separate table per variant with a summary comparison at the end.
    """
    groups = _group_by_hypothesis(results)
    multi = _has_multiple_hypotheses(results)

    if not multi:
        _render_terminal_table(results, metrics)
        return

    # Print per-variant tables
    for name in sorted(groups):
        group = groups[name]
        print(f"\n=== {name} ===\n")
        _render_terminal_table(group, metrics)
        print()

    # Summary comparison table
    print("\n=== Summary: All Variants ===\n")
    _render_terminal_summary(groups, metrics)


def _render_terminal_table(results: list, metrics: list[str]):
    """Print a single results table."""
    cols = [("File ID", 12), ("Duration", 10)]
    if "wer" in metrics:
        cols.append(("WER", 8))
    if "cpwer" in metrics:
        cols.append(("cpWER", 8))
    if "der" in metrics:
        cols.append(("DER", 8))

    header = "".join(name.ljust(width) for name, width in cols)
    sep = "".join("-" * (width - 1) + " " for _, width in cols)
    print(header)
    print(sep)

    for r in results:
        row_parts = [
            r.file_id[:11].ljust(12),
            _fmt_duration(r.duration_secs).ljust(10),
        ]
        if "wer" in metrics:
            row_parts.append(_fmt_pct(r.wer).ljust(8))
        if "cpwer" in metrics:
            row_parts.append(_fmt_pct(r.cpwer).ljust(8))
        if "der" in metrics:
            row_parts.append(_fmt_pct(r.der).ljust(8))
        print("".join(row_parts))

    print(sep)

    durations = [r.duration_secs for r in results]
    avg_parts = ["AVERAGE".ljust(12), "".ljust(10)]
    wavg_parts = ["WEIGHTED".ljust(12), "".ljust(10)]

    for metric in metrics:
        vals = [getattr(r, metric) for r in results]
        avg_parts.append(_fmt_pct(_avg(vals)).ljust(8))
        wavg_parts.append(_fmt_pct(_weighted_avg(vals, durations)).ljust(8))

    print("".join(avg_parts))
    print("".join(wavg_parts))

    for r in results:
        if r.errors:
            for err in r.errors:
                print(f"  [{r.file_id}] {err}")


def _render_terminal_summary(groups: dict[str, list], metrics: list[str]):
    """Print a comparison summary across hypothesis variants."""
    cols = [("Variant", 22)]
    if "wer" in metrics:
        cols.append(("WER (avg)", 12))
        cols.append(("WER (wtd)", 12))
    if "cpwer" in metrics:
        cols.append(("cpWER (avg)", 14))
        cols.append(("cpWER (wtd)", 14))
    if "der" in metrics:
        cols.append(("DER (avg)", 12))
        cols.append(("DER (wtd)", 12))

    header = "".join(name.ljust(width) for name, width in cols)
    sep = "".join("-" * (width - 1) + " " for _, width in cols)
    print(header)
    print(sep)

    for name in sorted(groups):
        group = groups[name]
        durations = [r.duration_secs for r in group]
        row = [name[:21].ljust(22)]

        for metric in metrics:
            vals = [getattr(r, metric) for r in group]
            row.append(_fmt_pct(_avg(vals)).ljust(12 if metric != "cpwer" else 14))
            row.append(_fmt_pct(_weighted_avg(vals, durations)).ljust(12 if metric != "cpwer" else 14))

        print("".join(row))

    print(sep)


def render_markdown(results: list, metrics: list[str], run_dir: str = "") -> str:
    """Render results as a markdown report.

    When results contain multiple hypothesis variants, includes per-variant
    sections and a summary comparison table.
    """
    lines = ["# Evaluation Results", ""]
    lines.append(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if run_dir:
        lines.append(f"- **Run directory**: `{run_dir}`")

    groups = _group_by_hypothesis(results)
    multi = _has_multiple_hypotheses(results)

    # Unique files (count by file_id)
    file_ids = {r.file_id for r in results}
    scored_ids = {r.file_id for r in results
                  if r.wer is not None or r.cpwer is not None or r.der is not None}
    lines.append(f"- **Files scored**: {len(scored_ids)} / {len(file_ids)}")

    total_hours = sum(r.duration_secs for r in results if r.file_id in file_ids) / 3600
    # Avoid double-counting duration for multi-variant results
    if multi:
        unique_durations = {}
        for r in results:
            if r.file_id not in unique_durations:
                unique_durations[r.file_id] = r.duration_secs
        total_hours = sum(unique_durations.values()) / 3600
    lines.append(f"- **Total duration**: {total_hours:.1f} hours")

    if multi:
        variant_names = sorted(groups.keys())
        lines.append(f"- **Variants**: {', '.join(variant_names)}")
    lines.append("")

    if multi:
        # Summary comparison table first
        lines.append("## Summary Comparison")
        lines.append("")
        lines.extend(_render_md_summary(groups, metrics))
        lines.append("")

        # Per-variant detail tables
        for name in sorted(groups):
            lines.append(f"## {name}")
            lines.append("")
            lines.extend(_render_md_table(groups[name], metrics))
            lines.append("")
    else:
        lines.extend(_render_md_table(results, metrics))
        lines.append("")

    # Errors
    has_errors = any(r.errors for r in results)
    if has_errors:
        lines.append("## Warnings")
        lines.append("")
        for r in results:
            for err in r.errors:
                label = f"{r.file_id}"
                if r.hypothesis_name:
                    label += f" ({r.hypothesis_name})"
                lines.append(f"- **{label}**: {err}")
        lines.append("")

    return "\n".join(lines)


def _render_md_table(results: list, metrics: list[str]) -> list[str]:
    """Render a markdown table for a set of results."""
    lines = []

    header_parts = ["| File ID | Duration"]
    sep_parts = ["|---------|----------"]
    if "wer" in metrics:
        header_parts.append("| WER")
        sep_parts.append("|------")
    if "cpwer" in metrics:
        header_parts.append("| cpWER")
        sep_parts.append("|------")
    if "der" in metrics:
        header_parts.append("| DER")
        sep_parts.append("|------")
    header_parts.append("|")
    sep_parts.append("|")

    lines.append(" ".join(header_parts))
    lines.append(" ".join(sep_parts))

    for r in results:
        row = f"| {r.file_id} | {_fmt_duration(r.duration_secs)}"
        if "wer" in metrics:
            row += f" | {_fmt_pct(r.wer)}"
        if "cpwer" in metrics:
            row += f" | {_fmt_pct(r.cpwer)}"
        if "der" in metrics:
            row += f" | {_fmt_pct(r.der)}"
        row += " |"
        lines.append(row)

    # Averages
    durations = [r.duration_secs for r in results]
    avg_parts = []
    wavg_parts = []
    for metric in metrics:
        vals = [getattr(r, metric) for r in results]
        label = metric.upper() if metric != "cpwer" else "cpWER"
        avg_parts.append(f"{label}={_fmt_pct(_avg(vals))}")
        wavg_parts.append(f"{label}={_fmt_pct(_weighted_avg(vals, durations))}")

    lines.append("")
    lines.append("**Macro average**: " + ", ".join(avg_parts))
    lines.append("**Duration-weighted average**: " + ", ".join(wavg_parts))

    return lines


def _render_md_summary(groups: dict[str, list], metrics: list[str]) -> list[str]:
    """Render a markdown comparison table across variants."""
    lines = []

    header_parts = ["| Variant"]
    sep_parts = ["|--------"]
    for metric in metrics:
        label = metric.upper() if metric != "cpwer" else "cpWER"
        header_parts.append(f"| {label} (avg)")
        header_parts.append(f"| {label} (wtd)")
        sep_parts.extend(["|------", "|------"])
    header_parts.append("|")
    sep_parts.append("|")

    lines.append(" ".join(header_parts))
    lines.append(" ".join(sep_parts))

    for name in sorted(groups):
        group = groups[name]
        durations = [r.duration_secs for r in group]
        row = f"| {name}"
        for metric in metrics:
            vals = [getattr(r, metric) for r in group]
            row += f" | {_fmt_pct(_avg(vals))}"
            row += f" | {_fmt_pct(_weighted_avg(vals, durations))}"
        row += " |"
        lines.append(row)

    return lines


def render_json(results: list, metrics: list[str]) -> str:
    """Render results as JSON.

    When results contain multiple hypothesis variants, groups per-file
    entries by variant and includes per-variant aggregates.
    """
    groups = _group_by_hypothesis(results)
    multi = _has_multiple_hypotheses(results)

    if not multi:
        return _render_json_flat(results, metrics)

    variants = {}
    for name in sorted(groups):
        group = groups[name]
        durations = [r.duration_secs for r in group]
        per_file = []
        for r in group:
            entry = {
                "file_id": r.file_id,
                "duration_secs": r.duration_secs,
                "metadata": r.metadata,
            }
            for metric in metrics:
                entry[metric] = getattr(r, metric)
            if r.errors:
                entry["errors"] = r.errors
            per_file.append(entry)

        aggregate = {}
        for metric in metrics:
            vals = [getattr(r, metric) for r in group]
            aggregate[f"{metric}_macro"] = _avg(vals)
            aggregate[f"{metric}_weighted"] = _weighted_avg(vals, durations)

        variants[name] = {
            "per_file": per_file,
            "aggregate": aggregate,
            "num_files": len(group),
        }

    # Unique file count
    file_ids = {r.file_id for r in results}
    unique_durations = {}
    for r in results:
        if r.file_id not in unique_durations:
            unique_durations[r.file_id] = r.duration_secs

    data = {
        "variants": variants,
        "num_files": len(file_ids),
        "num_variants": len(variants),
        "total_duration_hours": sum(unique_durations.values()) / 3600,
    }

    return json.dumps(data, indent=2)


def _render_json_flat(results: list, metrics: list[str]) -> str:
    """Render a flat JSON report (single hypothesis variant)."""
    durations = [r.duration_secs for r in results]

    per_file = []
    for r in results:
        entry = {
            "file_id": r.file_id,
            "duration_secs": r.duration_secs,
            "metadata": r.metadata,
        }
        if r.hypothesis_name:
            entry["hypothesis"] = r.hypothesis_name
        for metric in metrics:
            entry[metric] = getattr(r, metric)
        if r.errors:
            entry["errors"] = r.errors
        per_file.append(entry)

    aggregate = {}
    for metric in metrics:
        vals = [getattr(r, metric) for r in results]
        aggregate[f"{metric}_macro"] = _avg(vals)
        aggregate[f"{metric}_weighted"] = _weighted_avg(vals, durations)

    data = {
        "per_file": per_file,
        "aggregate": aggregate,
        "num_files": len(results),
        "num_scored": sum(1 for r in results if any(
            getattr(r, m) is not None for m in metrics
        )),
        "total_duration_hours": sum(durations) / 3600,
    }

    return json.dumps(data, indent=2)
