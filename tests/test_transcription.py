"""Tests for transcription.py — Whisper transcription, ensembling, and segment loading."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_critic.shared import SpeechConfig, SpeechData

from transcribe_critic.transcription import (
    _apply_resolutions,
    _build_cluster_prompt,
    _clean_llm_output,
    _clean_resolution,
    _cluster_diffs,
    _ensemble_whisper_transcripts,
    _filter_trivial_diffs,
    _format_reading,
    _load_transcript_segments,
    _merge_pairwise_diffs,
    _parse_wdiff_diffs,
    _run_whisper_model,
    _select_largest_model_json,
    collapse_repetition_loops,
    detect_repetition_loops,
    transcribe_audio,
)


# ---------------------------------------------------------------------------
# _load_transcript_segments
# ---------------------------------------------------------------------------

class TestLoadTranscriptSegments:
    def test_loads_segments_from_json(self, tmp_path):
        json_path = tmp_path / "transcript.json"
        json_path.write_text(json.dumps({
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "Hello world"},
                {"start": 1.5, "end": 3.0, "text": "Second segment"},
            ]
        }))
        data = SpeechData()
        data.transcript_json_path = json_path
        _load_transcript_segments(data)
        assert len(data.transcript_segments) == 2
        assert data.transcript_segments[0]["text"] == "Hello world"
        assert data.transcript_segments[0]["start"] == 0.0
        assert data.transcript_segments[1]["end"] == 3.0

    def test_skips_empty_text_segments(self, tmp_path):
        json_path = tmp_path / "transcript.json"
        json_path.write_text(json.dumps({
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Good"},
                {"start": 1.0, "end": 2.0, "text": "  "},
                {"start": 2.0, "end": 3.0, "text": ""},
            ]
        }))
        data = SpeechData()
        data.transcript_json_path = json_path
        _load_transcript_segments(data)
        assert len(data.transcript_segments) == 1
        assert data.transcript_segments[0]["text"] == "Good"

    def test_no_json_path_is_noop(self):
        data = SpeechData()
        data.transcript_json_path = None
        _load_transcript_segments(data)
        assert data.transcript_segments == []

    def test_missing_file_is_noop(self, tmp_path):
        data = SpeechData()
        data.transcript_json_path = tmp_path / "nonexistent.json"
        _load_transcript_segments(data)
        assert data.transcript_segments == []

    def test_malformed_json_does_not_raise(self, tmp_path):
        json_path = tmp_path / "bad.json"
        json_path.write_text("not json at all {{{")
        data = SpeechData()
        data.transcript_json_path = json_path
        _load_transcript_segments(data)  # should not raise
        assert data.transcript_segments == []

    def test_missing_fields_use_defaults(self, tmp_path):
        json_path = tmp_path / "transcript.json"
        json_path.write_text(json.dumps({
            "segments": [
                {"text": "No times here"},
            ]
        }))
        data = SpeechData()
        data.transcript_json_path = json_path
        _load_transcript_segments(data)
        assert len(data.transcript_segments) == 1
        assert data.transcript_segments[0]["start"] == 0
        assert data.transcript_segments[0]["end"] == 0


# ---------------------------------------------------------------------------
# Whisper ensembling: targeted diff resolution
# ---------------------------------------------------------------------------

class TestParseWdiffDiffs:
    """Test _parse_wdiff_diffs position tracking and diff types."""

    def test_substitution(self, tmp_path):
        text_a = "The quick brown fox jumps"
        text_b = "The quick red fox jumps"
        config = SpeechConfig(url="x", output_dir=tmp_path)
        diffs = _parse_wdiff_diffs(text_a, text_b, config)
        subs = [d for d in diffs if d["type"] == "substitution"]
        assert len(subs) == 1
        assert subs[0]["a_text"] == "brown"
        assert subs[0]["b_text"] == "red"
        assert subs[0]["a_pos"] == 2
        assert subs[0]["b_pos"] == 2

    def test_deletion(self, tmp_path):
        text_a = "The very quick fox"
        text_b = "The quick fox"
        config = SpeechConfig(url="x", output_dir=tmp_path)
        diffs = _parse_wdiff_diffs(text_a, text_b, config)
        dels = [d for d in diffs if d["type"] == "deletion"]
        assert len(dels) == 1
        assert dels[0]["a_text"] == "very"
        assert dels[0]["b_text"] == ""
        assert dels[0]["a_len"] == 1
        assert dels[0]["b_len"] == 0

    def test_insertion(self, tmp_path):
        text_a = "The quick fox"
        text_b = "The quick brown fox"
        config = SpeechConfig(url="x", output_dir=tmp_path)
        diffs = _parse_wdiff_diffs(text_a, text_b, config)
        ins = [d for d in diffs if d["type"] == "insertion"]
        assert len(ins) == 1
        assert ins[0]["a_text"] == ""
        assert ins[0]["b_text"] == "brown"

    def test_identical_texts_no_diffs(self, tmp_path):
        text = "Hello world this is a test"
        config = SpeechConfig(url="x", output_dir=tmp_path)
        diffs = _parse_wdiff_diffs(text, text, config)
        assert diffs == []

    def test_multiple_diffs_positions(self, tmp_path):
        text_a = "The cat sat on the mat and quietly left"
        text_b = "The dog sat on the rug and loudly left"
        config = SpeechConfig(url="x", output_dir=tmp_path)
        diffs = _parse_wdiff_diffs(text_a, text_b, config)
        # cat→dog, mat→rug, quietly→loudly (mat/rug separated by "and")
        assert len(diffs) >= 3


class TestFilterTrivialDiffs:
    def test_keeps_proper_nouns(self):
        diffs = [{"type": "substitution", "a_text": "Progerium", "b_text": "Progeria",
                  "a_pos": 0, "b_pos": 0, "a_len": 1, "b_len": 1}]
        assert len(_filter_trivial_diffs(diffs)) == 1

    def test_removes_stop_word_diffs(self):
        diffs = [{"type": "substitution", "a_text": "the", "b_text": "a",
                  "a_pos": 0, "b_pos": 0, "a_len": 1, "b_len": 1}]
        assert len(_filter_trivial_diffs(diffs)) == 0

    def test_keeps_mixed_diff(self):
        diffs = [{"type": "substitution", "a_text": "the spectral", "b_text": "a special",
                  "a_pos": 0, "b_pos": 0, "a_len": 2, "b_len": 2}]
        # "spectral" and "special" are not stop words
        assert len(_filter_trivial_diffs(diffs)) == 1


class TestClusterDiffs:
    def test_nearby_diffs_grouped(self):
        diffs = [
            {"type": "substitution", "a_text": "cat", "b_text": "dog",
             "a_pos": 5, "b_pos": 5, "a_len": 1, "b_len": 1},
            {"type": "substitution", "a_text": "mat", "b_text": "rug",
             "a_pos": 10, "b_pos": 10, "a_len": 1, "b_len": 1},
        ]
        clusters = _cluster_diffs(diffs, base_word_count=100, context_words=10)
        assert len(clusters) == 1  # close enough to be in one cluster

    def test_distant_diffs_separated(self):
        diffs = [
            {"type": "substitution", "a_text": "cat", "b_text": "dog",
             "a_pos": 5, "b_pos": 5, "a_len": 1, "b_len": 1},
            {"type": "substitution", "a_text": "mat", "b_text": "rug",
             "a_pos": 100, "b_pos": 100, "a_len": 1, "b_len": 1},
        ]
        clusters = _cluster_diffs(diffs, base_word_count=200, context_words=10)
        assert len(clusters) == 2

    def test_empty_diffs(self):
        assert _cluster_diffs([], base_word_count=100) == []

    def test_max_cluster_diffs(self):
        diffs = [
            {"type": "substitution", "a_text": f"w{i}", "b_text": f"x{i}",
             "a_pos": i * 2, "b_pos": i * 2, "a_len": 1, "b_len": 1}
            for i in range(10)
        ]
        clusters = _cluster_diffs(diffs, base_word_count=100,
                                  context_words=50, max_cluster_diffs=5)
        assert all(len(c) <= 5 for c in clusters)


# ---------------------------------------------------------------------------
# _clean_resolution
# ---------------------------------------------------------------------------

class TestCleanResolution:
    def test_strips_model_a_prefix(self):
        assert _clean_resolution('Model A: "progeria"') == "progeria"

    def test_strips_model_b_prefix(self):
        assert _clean_resolution('Model B: "going to"') == "going to"

    def test_strips_model_prefix_no_quotes(self):
        assert _clean_resolution("Model A: can") == "can"

    def test_strips_decision_prefix(self):
        assert _clean_resolution("Decision: like,") == "like,"

    def test_strips_surrounding_quotes(self):
        assert _clean_resolution('"the spectral"') == "the spectral"

    def test_strips_pipe_second_model(self):
        assert _clean_resolution('Model A: "cat" | Model B: "dog"') == "cat"

    def test_plain_text_unchanged(self):
        assert _clean_resolution("progeria") == "progeria"

    def test_omit_unchanged(self):
        assert _clean_resolution("(omit)") == "(omit)"

    def test_case_insensitive(self):
        assert _clean_resolution('model a: "test"') == "test"

    def test_strips_current_prefix(self):
        assert _clean_resolution('Current: "going to"') == "going to"

    def test_strips_alternative_prefix(self):
        assert _clean_resolution('Alternative: "gonna"') == "gonna"

    def test_strips_alternative_adds_prefix(self):
        assert _clean_resolution('Alternative adds: "extra words"') == "extra words"


class TestBuildClusterPrompt:
    def test_contains_context_and_diffs(self):
        cluster = [
            {"type": "substitution", "a_text": "cat", "b_text": "dog",
             "a_pos": 5, "b_pos": 5, "a_len": 1, "b_len": 1},
        ]
        base_words = "The quick brown dog sat on the mat in the room".split()
        prompt = _build_cluster_prompt(cluster, base_words, context_words=3)
        assert "[1]" in prompt
        assert "cat" in prompt
        assert "dog" in prompt
        assert "A:" in prompt
        assert "B:" in prompt

    def test_insertion_shows_omit(self):
        cluster = [
            {"type": "insertion", "a_text": "", "b_text": "extra",
             "a_pos": 3, "b_pos": 3, "a_len": 0, "b_len": 1},
        ]
        base_words = "one two three extra four five".split()
        prompt = _build_cluster_prompt(cluster, base_words)
        assert "(omit)" in prompt

    def test_diverged_positions_uses_b_pos(self):
        """When a_pos and b_pos diverge, context should use b_pos (base/medium)."""
        # Scenario: an earlier insertion shifted a_pos ahead of b_pos
        # The diff is at b_pos=5 in medium but a_pos=7 in small
        cluster = [
            {"type": "substitution", "a_text": "cat", "b_text": "dog",
             "a_pos": 7, "b_pos": 5, "a_len": 1, "b_len": 1},
        ]
        # medium (base): word at position 5 is "dog"
        base_words = "zero one two three four dog six seven eight nine".split()
        prompt = _build_cluster_prompt(cluster, base_words, context_words=2)
        # Context should show words around b_pos=5 in medium: "...four [1] six seven..."
        assert "four" in prompt
        assert "six" in prompt
        # Should NOT show words from a_pos=7 position: "seven" before the marker
        # "seven" can appear after, but "eight" should not be the leading context
        lines = prompt.split("\n")
        context_line = [l for l in lines if "[1]" in l][0]
        # "four" should precede the marker, "six" should follow
        marker_idx = context_line.index("[1]")
        before_marker = context_line[:marker_idx]
        assert "four" in before_marker

    def test_cluster_sorts_by_b_pos(self):
        """_cluster_diffs should group by b_pos, not a_pos."""
        # Two diffs close in b_pos but far in a_pos
        diffs = [
            {"type": "substitution", "a_text": "x", "b_text": "y",
             "a_pos": 100, "b_pos": 5, "a_len": 1, "b_len": 1},
            {"type": "substitution", "a_text": "p", "b_text": "q",
             "a_pos": 5, "b_pos": 10, "a_len": 1, "b_len": 1},
        ]
        clusters = _cluster_diffs(diffs, base_word_count=200, context_words=10)
        # Should be in one cluster (b_pos 5 and 10 are within context_words=10)
        assert len(clusters) == 1
        # Should be sorted by b_pos
        assert clusters[0][0]["b_pos"] == 5
        assert clusters[0][1]["b_pos"] == 10


# ---------------------------------------------------------------------------
# _merge_pairwise_diffs
# ---------------------------------------------------------------------------

class TestMergePairwiseDiffs:
    def test_same_position_merged(self):
        """Diffs at the same (b_pos, b_len) from different models are merged."""
        pairwise = [
            ("small", [
                {"type": "substitution", "a_text": "cat", "b_text": "dog",
                 "a_pos": 2, "b_pos": 2, "a_len": 1, "b_len": 1},
            ]),
            ("medium", [
                {"type": "substitution", "a_text": "cat", "b_text": "dog",
                 "a_pos": 2, "b_pos": 2, "a_len": 1, "b_len": 1},
            ]),
        ]
        result = _merge_pairwise_diffs(pairwise, "large", ["small", "medium", "large"])
        assert len(result) == 1
        assert "readings" in result[0]
        readings = result[0]["readings"]
        assert readings["small"] == "cat"
        assert readings["medium"] == "cat"
        assert readings["large"] == "dog"  # base model gets b_text

    def test_different_positions_separate(self):
        """Diffs at different positions stay separate."""
        pairwise = [
            ("small", [
                {"type": "substitution", "a_text": "cat", "b_text": "dog",
                 "a_pos": 2, "b_pos": 2, "a_len": 1, "b_len": 1},
            ]),
            ("medium", [
                {"type": "substitution", "a_text": "hat", "b_text": "mat",
                 "a_pos": 5, "b_pos": 5, "a_len": 1, "b_len": 1},
            ]),
        ]
        result = _merge_pairwise_diffs(pairwise, "large", ["small", "medium", "large"])
        assert len(result) == 2

    def test_model_not_in_diff_gets_base(self):
        """A model with no diff at a position gets the base text."""
        pairwise = [
            ("small", [
                {"type": "substitution", "a_text": "cat", "b_text": "dog",
                 "a_pos": 2, "b_pos": 2, "a_len": 1, "b_len": 1},
            ]),
            ("medium", []),  # medium agrees with large at all positions
        ]
        result = _merge_pairwise_diffs(pairwise, "large", ["small", "medium", "large"])
        assert len(result) == 1
        readings = result[0]["readings"]
        assert readings["small"] == "cat"
        assert readings["medium"] == "dog"  # agreed with base
        assert readings["large"] == "dog"   # base

    def test_all_three_differ(self):
        """All three models have different text at same position."""
        pairwise = [
            ("small", [
                {"type": "substitution", "a_text": "spectral", "b_text": "special",
                 "a_pos": 2, "b_pos": 2, "a_len": 1, "b_len": 1},
            ]),
            ("medium", [
                {"type": "substitution", "a_text": "spectacular", "b_text": "special",
                 "a_pos": 2, "b_pos": 2, "a_len": 1, "b_len": 1},
            ]),
        ]
        result = _merge_pairwise_diffs(pairwise, "large", ["small", "medium", "large"])
        assert len(result) == 1
        readings = result[0]["readings"]
        assert readings["small"] == "spectral"
        assert readings["medium"] == "spectacular"
        assert readings["large"] == "special"

    def test_empty_diffs(self):
        """All pairwise diffs empty returns empty list."""
        pairwise = [("small", []), ("medium", [])]
        result = _merge_pairwise_diffs(pairwise, "large", ["small", "medium", "large"])
        assert result == []

    def test_readings_order(self):
        """Readings are ordered: non-base models first (in all_models order), base last."""
        pairwise = [
            ("small", [
                {"type": "substitution", "a_text": "x", "b_text": "z",
                 "a_pos": 0, "b_pos": 0, "a_len": 1, "b_len": 1},
            ]),
            ("medium", [
                {"type": "substitution", "a_text": "y", "b_text": "z",
                 "a_pos": 0, "b_pos": 0, "a_len": 1, "b_len": 1},
            ]),
        ]
        result = _merge_pairwise_diffs(pairwise, "large", ["small", "medium", "large"])
        keys = list(result[0]["readings"].keys())
        assert keys == ["small", "medium", "large"]


# ---------------------------------------------------------------------------
# Multi-way prompt and parsing
# ---------------------------------------------------------------------------

class TestMultiWayPrompt:
    def test_multiway_shows_abc(self):
        """Multi-way diffs show A/B/C labels."""
        cluster = [
            {"type": "substitution", "a_text": "cat", "b_text": "dog",
             "a_pos": 2, "b_pos": 2, "a_len": 1, "b_len": 1,
             "readings": {"small": "cat", "medium": "dog", "large": "dog"}},
        ]
        base_words = "The quick dog sat here".split()
        prompt = _build_cluster_prompt(cluster, base_words, context_words=2)
        assert "A:" in prompt
        assert "B:" in prompt
        assert "C:" in prompt
        assert '"cat"' in prompt
        assert '"dog"' in prompt

    def test_multiway_agreeing_models_both_shown(self):
        """When two models agree, both still get their own letter with same text."""
        cluster = [
            {"type": "substitution", "a_text": "cat", "b_text": "dog",
             "a_pos": 2, "b_pos": 2, "a_len": 1, "b_len": 1,
             "readings": {"small": "cat", "medium": "dog", "large": "dog"}},
        ]
        base_words = "The quick dog sat here".split()
        prompt = _build_cluster_prompt(cluster, base_words, context_words=2)
        # B and C should both show "dog"
        lines = prompt.split("\n")
        disagree_line = [l for l in lines if "1." in l][0]
        assert disagree_line.count('"dog"') == 2

    def test_multiway_no_model_names(self):
        """Multi-way prompt should NOT contain model names."""
        cluster = [
            {"type": "substitution", "a_text": "cat", "b_text": "dog",
             "a_pos": 2, "b_pos": 2, "a_len": 1, "b_len": 1,
             "readings": {"small": "cat", "medium": "dog", "large": "dog"}},
        ]
        base_words = "The quick dog sat here".split()
        prompt = _build_cluster_prompt(cluster, base_words, context_words=2)
        assert "small" not in prompt
        assert "medium" not in prompt
        assert "large" not in prompt


class TestApplyResolutions:
    def test_substitution(self):
        base_words = "The quick brown fox jumps".split()
        diffs = [{"type": "substitution", "a_text": "red", "b_text": "brown",
                  "a_pos": 2, "b_pos": 2, "a_len": 1, "b_len": 1}]
        resolutions = {id(diffs[0]): "red"}
        result = _apply_resolutions(base_words, diffs, resolutions)
        assert result == "The quick red fox jumps"

    def test_deletion_chosen(self):
        base_words = "The very quick fox".split()
        diffs = [{"type": "insertion", "a_text": "", "b_text": "very",
                  "a_pos": 1, "b_pos": 1, "a_len": 0, "b_len": 1}]
        resolutions = {id(diffs[0]): "(omit)"}
        result = _apply_resolutions(base_words, diffs, resolutions)
        assert result == "The quick fox"

    def test_unresolved_keeps_base(self):
        base_words = "The quick brown fox".split()
        diffs = [{"type": "substitution", "a_text": "red", "b_text": "brown",
                  "a_pos": 2, "b_pos": 2, "a_len": 1, "b_len": 1}]
        resolutions = {}  # No resolution
        result = _apply_resolutions(base_words, diffs, resolutions)
        assert result == "The quick brown fox"

    def test_multiple_resolutions(self):
        base_words = "The cat sat on the mat".split()
        diffs = [
            {"type": "substitution", "a_text": "dog", "b_text": "cat",
             "a_pos": 1, "b_pos": 1, "a_len": 1, "b_len": 1},
            {"type": "substitution", "a_text": "rug", "b_text": "mat",
             "a_pos": 5, "b_pos": 5, "a_len": 1, "b_len": 1},
        ]
        resolutions = {id(diffs[0]): "dog", id(diffs[1]): "rug"}
        result = _apply_resolutions(base_words, diffs, resolutions)
        assert result == "The dog sat on the rug"

    def test_insertion_adds_words(self):
        base_words = "The fox jumps".split()
        diffs = [{"type": "deletion", "a_text": "brown", "b_text": "",
                  "a_pos": 1, "b_pos": 1, "a_len": 1, "b_len": 0}]
        resolutions = {id(diffs[0]): "brown"}
        result = _apply_resolutions(base_words, diffs, resolutions)
        assert result == "The brown fox jumps"


# ---------------------------------------------------------------------------
# Whisper ensembling: _ensemble_whisper_transcripts
# ---------------------------------------------------------------------------

class TestEnsembleWhisperTranscripts:
    """Test the top-level ensemble function."""

    def _make_whisper_data(self, tmp_path, models=("small", "medium")):
        """Create whisper transcript files and data."""
        data = SpeechData()
        data.audio_path = tmp_path / "audio.mp3"
        data.audio_path.write_text("fake audio")
        data.whisper_transcripts = {}
        for model in models:
            txt = tmp_path / f"{model}.txt"
            txt.write_text(f"transcript from {model} model with words")
            json_path = tmp_path / f"{model}.json"
            json_path.write_text('{"segments": []}')
            data.whisper_transcripts[model] = {"txt": txt, "json": json_path}
        return data

    def test_skips_when_single_model(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        data = self._make_whisper_data(tmp_path, models=("medium",))
        _ensemble_whisper_transcripts(config, data)
        # With only one model, should return without ensembling
        assert not (tmp_path / "whisper_merged.txt").exists()

    def test_reuses_fresh_whisper_merged(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        data = self._make_whisper_data(tmp_path)
        # Create ensembled.txt newer than whisper files
        time.sleep(0.05)
        ensembled = tmp_path / "whisper_merged.txt"
        ensembled.write_text("cached whisper_merged text")
        _ensemble_whisper_transcripts(config, data)
        out = capsys.readouterr().out
        assert "Reusing: whisper_merged.txt" in out
        assert data.transcript_path == ensembled

    def test_no_llm_uses_base(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, no_llm=True,
                              skip_existing=False)
        data = self._make_whisper_data(tmp_path)
        # Write distinct content to trigger differences
        (tmp_path / "whisper_small.txt").write_text("hello world from small")
        (tmp_path / "whisper_medium.txt").write_text("hello world from medium")
        _ensemble_whisper_transcripts(config, data)
        out = capsys.readouterr().out
        # Should use medium (larger) as base without LLM resolution
        ensembled = (tmp_path / "whisper_merged.txt").read_text()
        assert "medium" in ensembled
        assert "--no-llm" in out

    @patch("transcribe_critic.transcription._resolve_whisper_diffs")
    def test_calls_resolve(self, mock_resolve, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        data = self._make_whisper_data(tmp_path)
        mock_resolve.return_value = "resolved text"

        _ensemble_whisper_transcripts(config, data)
        mock_resolve.assert_called_once()
        assert (tmp_path / "whisper_merged.txt").read_text() == "resolved text"

    def test_selects_largest_model_as_base(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, no_llm=True,
                              skip_existing=False)
        data = self._make_whisper_data(tmp_path, models=("tiny", "small", "large"))
        (tmp_path / "whisper_tiny.txt").write_text("tiny text differs here")
        (tmp_path / "whisper_small.txt").write_text("small text differs here")
        (tmp_path / "whisper_large.txt").write_text("large text differs here")
        _ensemble_whisper_transcripts(config, data)
        out = capsys.readouterr().out
        assert "Using large as base" in out

    def test_dry_run_skips(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, dry_run=True,
                              skip_existing=False)
        data = self._make_whisper_data(tmp_path)
        _ensemble_whisper_transcripts(config, data)
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert not (tmp_path / "whisper_merged.txt").exists()


# ---------------------------------------------------------------------------
# Pipeline: transcribe_audio — validation
# ---------------------------------------------------------------------------

class TestRunWhisperModel:
    """Test _run_whisper_model with mocked subprocess and whisper."""

    def test_skip_existing_reuses(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=True)
        data = SpeechData(audio_path=tmp_path / "audio.mp3")
        data.audio_path.write_text("fake")
        # Create existing output that is newer than audio
        time.sleep(0.05)
        txt = tmp_path / "whisper_medium.txt"
        txt.write_text("existing transcript")
        deps = {"mlx_whisper": True, "whisper": False}
        _run_whisper_model(config, data, "medium", deps)
        out = capsys.readouterr().out
        assert "Reusing" in out
        assert data.whisper_transcripts["medium"]["txt"] == txt

    def test_dry_run_skips(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, dry_run=True,
                              skip_existing=False)
        data = SpeechData(audio_path=tmp_path / "audio.mp3")
        data.audio_path.write_text("fake")
        deps = {"mlx_whisper": True, "whisper": False}
        _run_whisper_model(config, data, "small", deps)
        out = capsys.readouterr().out
        assert "[dry-run]" in out

    @patch("transcribe_critic.transcription.run_command")
    def test_mlx_whisper_runs_and_renames(self, mock_run, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        data = SpeechData(audio_path=tmp_path / "audio.mp3")
        data.audio_path.write_text("fake")
        deps = {"mlx_whisper": True, "whisper": False}

        # Simulate mlx_whisper creating default-named output files
        def create_default_files(cmd, desc, verbose=False):
            (tmp_path / "audio.txt").write_text("transcribed text")
            (tmp_path / "audio.json").write_text('{"segments": []}')
            return MagicMock()
        mock_run.side_effect = create_default_files

        _run_whisper_model(config, data, "small", deps)
        # Should have renamed to model-specific names
        assert (tmp_path / "whisper_small.txt").exists()
        assert (tmp_path / "whisper_small.json").exists()
        assert data.whisper_transcripts["small"]["txt"] == tmp_path / "whisper_small.txt"

    @patch("transcribe_critic.transcription.run_command")
    def test_mlx_whisper_unlinks_existing_before_rename(self, mock_run, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        data = SpeechData(audio_path=tmp_path / "audio.mp3")
        data.audio_path.write_text("fake")
        deps = {"mlx_whisper": True, "whisper": False}
        # Pre-create target files (from a previous run)
        (tmp_path / "whisper_small.txt").write_text("old")
        (tmp_path / "whisper_small.json").write_text("old")

        def create_default_files(cmd, desc, verbose=False):
            (tmp_path / "audio.txt").write_text("new text")
            (tmp_path / "audio.json").write_text('{"segments": []}')
            return MagicMock()
        mock_run.side_effect = create_default_files

        _run_whisper_model(config, data, "small", deps)
        assert (tmp_path / "whisper_small.txt").read_text() == "new text"

    def test_openai_whisper_success(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        data = SpeechData(audio_path=tmp_path / "audio.mp3")
        data.audio_path.write_text("fake")
        deps = {"mlx_whisper": False, "whisper": True}

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "hello world",
            "segments": [{"start": 0, "end": 1, "text": "hello world"}],
            "language": "en",
        }
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            _run_whisper_model(config, data, "medium", deps)

        assert (tmp_path / "whisper_medium.txt").read_text() == "hello world"
        assert (tmp_path / "whisper_medium.json").exists()
        assert data.whisper_transcripts["medium"]["txt"] == tmp_path / "whisper_medium.txt"

    @patch("transcribe_critic.transcription.run_command")
    def test_stores_none_when_files_missing(self, mock_run, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        data = SpeechData(audio_path=tmp_path / "audio.mp3")
        data.audio_path.write_text("fake")
        deps = {"mlx_whisper": True, "whisper": False}
        # mlx_whisper runs but doesn't create any files
        mock_run.return_value = MagicMock()
        _run_whisper_model(config, data, "small", deps)
        assert data.whisper_transcripts["small"]["txt"] is None
        assert data.whisper_transcripts["small"]["json"] is None


# ---------------------------------------------------------------------------
# Pipeline: transcribe_audio — validation and branching
# ---------------------------------------------------------------------------

class TestTranscribeAudio:
    def test_raises_without_audio(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        data = SpeechData()
        with pytest.raises(FileNotFoundError):
            transcribe_audio(config, data)

    def test_raises_with_missing_audio_file(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        data = SpeechData(audio_path=tmp_path / "nonexistent.mp3")
        with pytest.raises(FileNotFoundError):
            transcribe_audio(config, data)

    @patch("transcribe_critic.transcription.check_dependencies",
           return_value={"mlx_whisper": False, "whisper": False})
    def test_raises_without_whisper(self, mock_deps, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake")
        data = SpeechData(audio_path=audio)
        with pytest.raises(RuntimeError, match="No Whisper"):
            transcribe_audio(config, data)

    @patch("transcribe_critic.transcription._load_transcript_segments")
    @patch("transcribe_critic.transcription._run_whisper_model")
    @patch("transcribe_critic.transcription.check_dependencies",
           return_value={"mlx_whisper": True, "whisper": False})
    def test_single_model_uses_directly(self, mock_deps, mock_run, mock_load, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, whisper_models=["medium"])
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake")
        txt = tmp_path / "whisper_medium.txt"
        txt.write_text("transcript")
        data = SpeechData(audio_path=audio)

        def populate_transcripts(cfg, d, model, deps):
            d.whisper_transcripts[model] = {"txt": txt, "json": None}
        mock_run.side_effect = populate_transcripts

        transcribe_audio(config, data)
        assert data.transcript_path == txt
        mock_run.assert_called_once()

    @patch("transcribe_critic.transcription._run_whisper_model")
    @patch("transcribe_critic.transcription.check_dependencies",
           return_value={"mlx_whisper": True, "whisper": False})
    def test_multiple_models_does_not_call_ensemble(self, mock_deps, mock_run, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path,
                              whisper_models=["small", "medium"])
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake")
        data = SpeechData(audio_path=audio)

        def populate_transcripts(cfg, d, model, deps):
            txt = tmp_path / f"{model}.txt"
            txt.write_text(f"{model} text")
            d.whisper_transcripts[model] = {"txt": txt, "json": None}
        mock_run.side_effect = populate_transcripts

        transcribe_audio(config, data)
        assert mock_run.call_count == 2
        # transcript_path not set — ensemble (run separately) will set it
        assert data.transcript_path is None

    @patch("transcribe_critic.transcription._load_transcript_segments")
    @patch("transcribe_critic.transcription._run_whisper_model")
    @patch("transcribe_critic.transcription.check_dependencies",
           return_value={"mlx_whisper": True, "whisper": False})
    def test_raises_if_no_transcript_after_run(self, mock_deps, mock_run,
                                                mock_load, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, whisper_models=["medium"])
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake")
        data = SpeechData(audio_path=audio)
        # _run_whisper_model doesn't set transcript_path
        mock_run.return_value = None
        with pytest.raises(FileNotFoundError, match="Transcript file not found"):
            transcribe_audio(config, data)


# ---------------------------------------------------------------------------
# _select_largest_model_json
# ---------------------------------------------------------------------------

class TestSelectLargestModelJson:
    def test_returns_largest_available(self, tmp_path):
        json_medium = tmp_path / "whisper_medium.json"
        json_small = tmp_path / "whisper_small.json"
        data = SpeechData()
        data.whisper_transcripts = {
            "small": {"txt": tmp_path / "whisper_small.txt", "json": json_small},
            "medium": {"txt": tmp_path / "whisper_medium.txt", "json": json_medium},
        }
        assert _select_largest_model_json(data) == json_medium

    def test_returns_none_when_empty(self):
        data = SpeechData()
        assert _select_largest_model_json(data) is None

    def test_returns_none_when_json_missing(self, tmp_path):
        data = SpeechData()
        data.whisper_transcripts = {
            "small": {"txt": tmp_path / "whisper_small.txt", "json": None},
        }
        assert _select_largest_model_json(data) is None


# ---------------------------------------------------------------------------
# _clean_llm_output
# ---------------------------------------------------------------------------

class TestCleanLlmOutput:
    def test_strips_separator_lines(self):
        text = "Hello world\n---\nMore text\n---"
        assert _clean_llm_output(text) == "Hello world\nMore text"

    def test_strips_markdown_headers(self):
        text = "## Merged Transcript\nHello world"
        assert _clean_llm_output(text) == "Hello world"

    def test_preserves_normal_text(self):
        text = "Hello world.\nThis is fine.\nAll good."
        assert _clean_llm_output(text) == text

    def test_strips_multiple_artifact_types(self):
        text = "---\n# Header\nHello\n***\nWorld\n==="
        assert _clean_llm_output(text) == "Hello\nWorld"

    def test_handles_empty_input(self):
        assert _clean_llm_output("") == ""
        assert _clean_llm_output("  \n  ") == ""

    def test_preserves_hyphens_in_words(self):
        """Hyphens within words should not be stripped."""
        text = "well-known fact\nself-driving car"
        assert _clean_llm_output(text) == text

    def test_preserves_short_dashes(self):
        """Short dashes (1-2 chars) are not separators."""
        text = "Hello - world\nA -- B"
        assert _clean_llm_output(text) == text


# ---------------------------------------------------------------------------
# detect_repetition_loops
# ---------------------------------------------------------------------------

class TestDetectRepetitionLoops:
    def test_detects_simple_repeat(self):
        text = "hello hello hello hello world"
        loops = detect_repetition_loops(text, min_repeats=4)
        assert len(loops) == 1
        assert loops[0]["phrase"] == "hello"
        assert loops[0]["count"] == 4

    def test_detects_multiword_phrase(self):
        text = "It is not. It is not. It is not. It is not. Then something else."
        loops = detect_repetition_loops(text, min_repeats=4)
        assert len(loops) == 1
        assert loops[0]["phrase"] == "It is not."
        assert loops[0]["count"] == 4

    def test_no_detection_below_threshold(self):
        text = "hello hello hello world"
        loops = detect_repetition_loops(text, min_repeats=4)
        assert len(loops) == 0

    def test_preserves_non_repetitive_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        loops = detect_repetition_loops(text)
        assert len(loops) == 0

    def test_large_repeat_count(self):
        text = "It is not. " * 63 + "Something else."
        loops = detect_repetition_loops(text, min_repeats=4)
        assert len(loops) == 1
        assert loops[0]["count"] == 63

    def test_multiple_loops_in_text(self):
        text = "ok ok ok ok ok then hello hello hello hello"
        loops = detect_repetition_loops(text, min_repeats=4)
        assert len(loops) == 2
        phrases = {l["phrase"] for l in loops}
        assert phrases == {"ok", "hello"}


# ---------------------------------------------------------------------------
# collapse_repetition_loops
# ---------------------------------------------------------------------------

class TestCollapseRepetitionLoops:
    def test_collapses_to_two_occurrences(self):
        text = "before " + "repeat " * 10 + "after"
        result, loops = collapse_repetition_loops(text, min_repeats=4)
        assert loops[0]["count"] == 10
        assert result.count("repeat") == 2
        assert "before" in result
        assert "after" in result

    def test_returns_loops_metadata(self):
        text = "It is not. " * 20 + "End."
        result, loops = collapse_repetition_loops(text, min_repeats=4)
        assert len(loops) == 1
        assert loops[0]["phrase"] == "It is not."
        assert loops[0]["count"] == 20

    def test_no_change_for_clean_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        result, loops = collapse_repetition_loops(text)
        assert result == text
        assert loops == []

    def test_multiword_collapse(self):
        text = "start oh no oh no oh no oh no oh no end"
        result, loops = collapse_repetition_loops(text, min_repeats=4)
        assert loops[0]["phrase"] == "oh no"
        # Should have exactly 2 "oh no" remaining
        assert result.count("oh no") == 2
        assert result.startswith("start")
        assert result.endswith("end")


# ---------------------------------------------------------------------------
# _run_whisper_model — hallucination collapse on output
# ---------------------------------------------------------------------------

class TestRunWhisperModelHallucinationCollapse:
    @patch("transcribe_critic.transcription.run_command")
    def test_collapses_hallucination_in_output(self, mock_run, tmp_path, capsys):
        """After whisper produces output with repetitions, they get collapsed."""
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        data = SpeechData(audio_path=tmp_path / "audio.mp3")
        data.audio_path.write_text("fake")
        deps = {"mlx_whisper": True, "whisper": False}

        # Simulate mlx_whisper creating output with repetitions
        def create_hallucinated_output(cmd, desc, verbose=False):
            hallucinated = "The unremarkable. " * 100 + "Good morning everyone."
            (tmp_path / "audio.txt").write_text(hallucinated)
            (tmp_path / "audio.json").write_text('{"segments": []}')
            return MagicMock()
        mock_run.side_effect = create_hallucinated_output

        _run_whisper_model(config, data, "small", deps)
        text = (tmp_path / "whisper_small.txt").read_text()
        # Should be collapsed to 2 occurrences
        assert text.count("The unremarkable.") == 2
        assert "Good morning everyone." in text
        out = capsys.readouterr().out
        assert "Collapsed" in out
        assert "hallucination" in out


# ---------------------------------------------------------------------------
# _call_and_parse_cluster — LLM response parsing
# ---------------------------------------------------------------------------

class TestCallAndParseCluster:
    def test_parses_two_way_responses(self):
        from transcribe_critic.transcription import _call_and_parse_cluster
        client = MagicMock()
        config = SpeechConfig(url="x", output_dir=Path("/tmp"), local=True)

        cluster = [
            {"type": "substitution", "a_text": "dog", "b_text": "cat",
             "a_pos": 1, "b_pos": 1, "a_len": 1, "b_len": 1},
            {"type": "substitution", "a_text": "ran", "b_text": "run",
             "a_pos": 3, "b_pos": 3, "a_len": 1, "b_len": 1},
        ]

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="1. A\n2. B")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        with patch("transcribe_critic.transcription.llm_call_with_retry",
                   return_value=mock_response):
            choices, resolutions = _call_and_parse_cluster(
                client, config, cluster, "prompt", "")

        assert choices[0] == "A"
        assert choices[1] == "B"
        assert resolutions[id(cluster[0])] == "dog"
        assert resolutions[id(cluster[1])] == "run"

    def test_parses_multi_way_responses(self):
        from transcribe_critic.transcription import _call_and_parse_cluster
        client = MagicMock()
        config = SpeechConfig(url="x", output_dir=Path("/tmp"), local=True)

        cluster = [
            {"type": "substitution", "b_pos": 1, "b_len": 1,
             "readings": {"small": "dog", "medium": "cat", "distil-large-v3": "cat"}},
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="1. B")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        with patch("transcribe_critic.transcription.llm_call_with_retry",
                   return_value=mock_response):
            choices, resolutions = _call_and_parse_cluster(
                client, config, cluster, "prompt", "")

        assert choices[0] == "B"
        assert resolutions[id(cluster[0])] == "cat"


# ---------------------------------------------------------------------------
# _resolve_whisper_diffs — end-to-end with mocked LLM
# ---------------------------------------------------------------------------

class TestResolveWhisperDiffs:
    @patch("transcribe_critic.transcription._call_and_parse_cluster")
    @patch("transcribe_critic.transcription.create_llm_client")
    def test_resolves_diffs_and_applies(self, mock_client, mock_call, tmp_path):
        from transcribe_critic.transcription import _resolve_whisper_diffs
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)

        base_text = "the quick brown fox jumps over the lazy dog"
        all_transcripts = {
            "medium": base_text,
            "small": "the quick red fox jumps over the lazy cat",
        }

        # Mock cluster resolution: choose base (B) for all diffs
        def resolve_cluster(client, cfg, cluster, prompt, warning):
            choices = []
            resolutions = {}
            for d in cluster:
                choices.append("B")
                resolutions[id(d)] = d["b_text"] if d.get("b_text") else ""
            return choices, resolutions
        mock_call.side_effect = resolve_cluster

        result = _resolve_whisper_diffs(base_text, all_transcripts, config)
        # Base text should be preserved (chose B = base)
        assert "brown" in result
        assert "dog" in result

    @patch("transcribe_critic.transcription.create_llm_client")
    def test_no_diffs_returns_base(self, mock_client, tmp_path, capsys):
        from transcribe_critic.transcription import _resolve_whisper_diffs
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)

        # Same text for both models → no diffs
        text = "the quick brown fox"
        all_transcripts = {"medium": text, "small": text}
        result = _resolve_whisper_diffs(text, all_transcripts, config)
        assert result == text
        out = capsys.readouterr().out
        assert "No meaningful differences" in out


# ---------------------------------------------------------------------------
# _ensemble_whisper_transcripts — dry-run checkpoint status
# ---------------------------------------------------------------------------

class TestEnsembleDryRunCheckpoints:
    def test_dry_run_shows_cached_cluster_count(self, tmp_path, capsys):
        """Dry-run with cached clusters shows 'X/Y clusters cached'."""
        config = SpeechConfig(url="x", output_dir=tmp_path, dry_run=True,
                              skip_existing=False)
        # Create whisper transcripts
        data = SpeechData(audio_path=tmp_path / "audio.mp3")
        data.audio_path.write_text("fake")
        small_txt = tmp_path / "whisper_small.txt"
        medium_txt = tmp_path / "whisper_medium.txt"
        small_txt.write_text("small text")
        medium_txt.write_text("medium text")
        data.whisper_transcripts = {
            "small": {"txt": small_txt, "json": None},
            "medium": {"txt": medium_txt, "json": None},
        }

        # Create ensemble checkpoint dir with some cached clusters
        ensemble_dir = tmp_path / "ensemble_chunks"
        ensemble_dir.mkdir()
        time.sleep(0.05)
        (ensemble_dir / "cluster_000.json").write_text('{"choices": ["A"]}')
        (ensemble_dir / "cluster_001.json").write_text('{"choices": ["B"]}')

        _ensemble_whisper_transcripts(config, data)
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "clusters cached" in out


