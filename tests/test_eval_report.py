"""Tests for eval/report.py — terminal, markdown, and JSON rendering."""

import json
from dataclasses import dataclass, field
from typing import Optional

import pytest

from transcribe_critic.eval.report import (
    _avg,
    _fmt_duration,
    _fmt_pct,
    _weighted_avg,
    render_json,
    render_markdown,
    render_terminal,
)


@dataclass
class _FakeResult:
    """Minimal stand-in for FileResult."""
    file_id: str
    hypothesis_name: str = ""
    duration_secs: float = 600.0
    wer: Optional[float] = None
    cpwer: Optional[float] = None
    der: Optional[float] = None
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _make_results():
    return [
        _FakeResult(file_id="001", duration_secs=600, wer=0.10, cpwer=0.12),
        _FakeResult(file_id="002", duration_secs=1800, wer=0.20, cpwer=0.22),
    ]


class TestHelpers:
    def test_avg_basic(self):
        assert _avg([0.1, 0.2, 0.3]) == pytest.approx(0.2)

    def test_avg_with_nones(self):
        assert _avg([0.1, None, 0.3]) == pytest.approx(0.2)

    def test_avg_all_none(self):
        assert _avg([None, None]) is None

    def test_avg_empty(self):
        assert _avg([]) is None

    def test_weighted_avg_basic(self):
        # 0.1*600 + 0.2*1800 = 60+360 = 420; 420/2400 = 0.175
        assert _weighted_avg([0.1, 0.2], [600, 1800]) == pytest.approx(0.175)

    def test_weighted_avg_with_none(self):
        # Only second value counts: 0.2*1800 / 1800 = 0.2
        assert _weighted_avg([None, 0.2], [600, 1800]) == pytest.approx(0.2)

    def test_weighted_avg_all_none(self):
        assert _weighted_avg([None, None], [600, 1800]) is None

    def test_fmt_duration_minutes(self):
        assert _fmt_duration(125) == "2:05"

    def test_fmt_duration_hours(self):
        assert _fmt_duration(3661) == "1:01:01"

    def test_fmt_pct_value(self):
        assert _fmt_pct(0.123) == "12.3%"

    def test_fmt_pct_none(self):
        assert _fmt_pct(None) == "-"


class TestRenderTerminal:
    def test_single_variant(self, capsys):
        results = _make_results()
        render_terminal(results, ["wer"])
        out = capsys.readouterr().out
        assert "001" in out
        assert "002" in out
        assert "AVERAGE" in out
        assert "WEIGHTED" in out

    def test_multi_variant(self, capsys):
        results = [
            _FakeResult(file_id="001", hypothesis_name="whisper_medium", wer=0.10),
            _FakeResult(file_id="001", hypothesis_name="whisper_merged", wer=0.08),
        ]
        render_terminal(results, ["wer"])
        out = capsys.readouterr().out
        assert "whisper_medium" in out
        assert "whisper_merged" in out
        assert "Summary" in out

    def test_empty_results(self, capsys):
        render_terminal([], ["wer"])
        out = capsys.readouterr().out
        assert "AVERAGE" in out

    def test_errors_shown(self, capsys):
        results = [_FakeResult(file_id="X01", errors=["cpWER skipped: no speakers"])]
        render_terminal(results, ["wer"])
        out = capsys.readouterr().out
        assert "cpWER skipped" in out


class TestRenderMarkdown:
    def test_produces_valid_markdown(self):
        results = _make_results()
        md = render_markdown(results, ["wer", "cpwer"])
        assert "# Evaluation Results" in md
        assert "| File ID" in md
        assert "| WER" in md
        assert "Macro average" in md

    def test_multi_variant_markdown(self):
        results = [
            _FakeResult(file_id="001", hypothesis_name="a", wer=0.10),
            _FakeResult(file_id="001", hypothesis_name="b", wer=0.12),
        ]
        md = render_markdown(results, ["wer"])
        assert "## Summary Comparison" in md
        assert "Variants" in md

    def test_errors_section(self):
        results = [_FakeResult(file_id="E01", errors=["something went wrong"])]
        md = render_markdown(results, ["wer"])
        assert "## Warnings" in md
        assert "something went wrong" in md


class TestRenderJson:
    def test_roundtrip(self):
        results = _make_results()
        text = render_json(results, ["wer", "cpwer"])
        data = json.loads(text)
        assert data["num_files"] == 2
        assert len(data["per_file"]) == 2
        assert "aggregate" in data
        assert "wer_macro" in data["aggregate"]

    def test_multi_variant_json(self):
        results = [
            _FakeResult(file_id="001", hypothesis_name="a", wer=0.10, duration_secs=600),
            _FakeResult(file_id="001", hypothesis_name="b", wer=0.12, duration_secs=600),
        ]
        text = render_json(results, ["wer"])
        data = json.loads(text)
        assert "variants" in data
        assert "a" in data["variants"]
        assert "b" in data["variants"]
        assert data["num_variants"] == 2

    def test_empty_results(self):
        text = render_json([], ["wer"])
        data = json.loads(text)
        assert data["num_files"] == 0
