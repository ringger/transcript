"""Tests for eval/score.py — WER/DER scoring and hypothesis discovery."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_critic.eval.score import (
    FileResult,
    _discover_hypotheses,
    _run_meeteval_der,
    _run_meeteval_wer,
    score_file,
)
from transcribe_critic.eval.datasets import ManifestEntry


def _make_entry(**overrides):
    defaults = dict(
        file_id="001", audio_path="audio/001.wav", ref_stm_path="ref/001.stm",
        duration_secs=600, subset_tags=["all"], metadata={},
    )
    defaults.update(overrides)
    return ManifestEntry(**defaults)


class TestRunMeeTevalWer:
    @patch("transcribe_critic.eval.score.subprocess.run")
    def test_parses_result(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"file001": {"error_rate": 0.15}}),
            stderr="",
        )
        result = _run_meeteval_wer(tmp_path / "ref.stm", tmp_path / "hyp.stm", "cpwer")
        assert result == pytest.approx(0.15)

    @patch("transcribe_critic.eval.score.subprocess.run")
    def test_returns_none_on_failure(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = _run_meeteval_wer(tmp_path / "ref.stm", tmp_path / "hyp.stm")
        assert result is None

    @patch("transcribe_critic.eval.score.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_when_not_installed(self, mock_run, tmp_path):
        result = _run_meeteval_wer(tmp_path / "ref.stm", tmp_path / "hyp.stm")
        assert result is None


class TestRunMeeTevalDer:
    @patch("transcribe_critic.eval.score.subprocess.run")
    def test_parses_result(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"file001": {"error_rate": 0.08}}),
            stderr="",
        )
        result = _run_meeteval_der(tmp_path / "ref.rttm", tmp_path / "hyp.rttm")
        assert result == pytest.approx(0.08)

    @patch("transcribe_critic.eval.score.subprocess.run")
    def test_returns_none_on_failure(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = _run_meeteval_der(tmp_path / "ref.rttm", tmp_path / "hyp.rttm")
        assert result is None


class TestDiscoverHypotheses:
    def test_finds_whisper_models(self, tmp_path):
        (tmp_path / "whisper_medium.txt").write_text("hello")
        (tmp_path / "whisper_large.txt").write_text("hello")
        result = _discover_hypotheses(tmp_path)
        names = [name for name, _ in result]
        assert "whisper_medium" in names
        assert "whisper_large" in names

    def test_finds_merged_outputs(self, tmp_path):
        (tmp_path / "whisper_merged.txt").write_text("hello")
        (tmp_path / "transcript_merged.txt").write_text("hello")
        result = _discover_hypotheses(tmp_path)
        names = [name for name, _ in result]
        assert "whisper_merged" in names
        assert "transcript_merged" in names

    def test_empty_dir(self, tmp_path):
        assert _discover_hypotheses(tmp_path) == []


class TestScoreFile:
    def test_missing_output_dir(self, tmp_path):
        entry = _make_entry()
        results = score_file(entry, tmp_path, tmp_path / "runs", ["wer"])
        assert len(results) == 1
        assert "not found" in results[0].errors[0]

    def test_missing_ref_stm(self, tmp_path):
        # Create output dir but no ref STM
        run_dir = tmp_path / "runs"
        (run_dir / "file001").mkdir(parents=True)
        entry = _make_entry(ref_stm_path="ref/001.stm")
        results = score_file(entry, tmp_path, run_dir, ["wer"])
        assert len(results) == 1
        assert "Reference STM not found" in results[0].errors[0]

    def test_fallback_to_bare_id_dir(self, tmp_path):
        # Uses entry.file_id without "file" prefix as fallback
        run_dir = tmp_path / "runs"
        (run_dir / "001").mkdir(parents=True)
        # Still needs ref STM to exist for further scoring
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        (ref_dir / "001.stm").write_text("file001 1 spk 0.0 10.0 hello world")
        entry = _make_entry(ref_stm_path="ref/001.stm")
        results = score_file(entry, tmp_path, run_dir, ["wer"], hypothesis="all")
        # No transcripts found in bare dir → error about no outputs
        assert any("No transcript outputs found" in e for r in results for e in r.errors)


class TestFileResult:
    def test_default_values(self):
        r = FileResult(file_id="test")
        assert r.wer is None
        assert r.cpwer is None
        assert r.der is None
        assert r.errors == []
        assert r.metadata == {}
