"""Tests for transcriber.py — pipeline stages, VTT cleaning, markdown, and CLI logic."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared import SpeechConfig, SpeechData, is_up_to_date

from merge import _format_structured_segments

from transcriber import (
    _dry_run_skip,
    _ensemble_whisper_transcripts,
    _format_paragraph,
    _generate_interleaved_markdown,
    _generate_sequential_markdown,
    _get_best_transcript_text,
    _load_transcript_segments,
    _resolve_whisper_chunk,
    _resolve_whisper_differences,
    _strip_structured_headers,
    analyze_slides_with_vision,
    analyze_source_survival,
    clean_vtt_captions,
    estimate_api_cost,
    extract_slides,
    generate_markdown,
    merge_transcript_sources,
    transcribe_audio,
)


# ---------------------------------------------------------------------------
# clean_vtt_captions
# ---------------------------------------------------------------------------

class TestCleanVttCaptions:
    def test_basic_vtt(self, tmp_path):
        vtt = tmp_path / "captions.vtt"
        vtt.write_text(
            "WEBVTT\n"
            "Kind: captions\n"
            "Language: en\n"
            "\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Hello world.\n"
            "\n"
            "00:00:03.000 --> 00:00:05.000\n"
            "This is a test.\n"
        )
        result = clean_vtt_captions(vtt)
        assert "Hello world." in result
        assert "This is a test." in result
        assert "WEBVTT" not in result
        assert "-->" not in result

    def test_deduplication(self, tmp_path):
        vtt = tmp_path / "captions.vtt"
        vtt.write_text(
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Hello world.\n\n"
            "00:00:03.000 --> 00:00:05.000\n"
            "Hello world.\n"
        )
        result = clean_vtt_captions(vtt)
        assert result.count("Hello world.") == 1

    def test_strips_html_tags(self, tmp_path):
        vtt = tmp_path / "captions.vtt"
        vtt.write_text(
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "<c>Hello</c> <i>world</i>.\n"
        )
        result = clean_vtt_captions(vtt)
        assert "<c>" not in result
        assert "<i>" not in result
        assert "Hello world." in result


# ---------------------------------------------------------------------------
# _dry_run_skip
# ---------------------------------------------------------------------------

class TestDryRunSkip:
    def test_returns_false_when_not_dry_run(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, dry_run=False)
        assert _dry_run_skip(config, "do thing", "out.txt") is False

    def test_returns_true_when_dry_run(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, dry_run=True)
        assert _dry_run_skip(config, "do thing", "out.txt") is True

    def test_prints_message_when_dry_run(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, dry_run=True)
        _dry_run_skip(config, "merge sources", "merged.txt")
        captured = capsys.readouterr()
        assert "[dry-run]" in captured.out
        assert "merge sources" in captured.out
        assert "merged.txt" in captured.out


# ---------------------------------------------------------------------------
# _strip_structured_headers / roundtrip with _format_structured_segments
# ---------------------------------------------------------------------------

class TestStripStructuredHeaders:
    def test_strip_removes_headers_with_timestamp(self):
        text = "**Alice** (0:01:00)\n\nSome text.\n"
        stripped = _strip_structured_headers(text)
        assert "**Alice**" not in stripped
        assert "Some text." in stripped

    def test_strip_removes_headers_without_timestamp(self):
        text = "**Bob**\n\nHi.\n"
        stripped = _strip_structured_headers(text)
        assert "**Bob**" not in stripped
        assert "Hi." in stripped

    def test_strip_preserves_inline_bold(self):
        text = "This has **bold** in the middle.\n"
        stripped = _strip_structured_headers(text)
        assert "**bold**" in stripped

    def test_roundtrip(self):
        segs = [
            {"speaker": "Alice", "timestamp": "0:01:00", "text": "First."},
            {"speaker": "Bob", "timestamp": "0:02:00", "text": "Second."},
        ]
        formatted = _format_structured_segments(segs)
        stripped = _strip_structured_headers(formatted)
        assert "**Alice**" not in stripped
        assert "First." in stripped
        assert "Second." in stripped


# ---------------------------------------------------------------------------
# DAG integration: stage skip/run decisions
# ---------------------------------------------------------------------------

def _make_test_artefacts(tmp_path):
    """Create a minimal set of artefacts for testing stage skip/run logic."""
    whisper = tmp_path / "ensembled.txt"
    whisper.write_text("whisper transcript words " * 200)
    captions_vtt = tmp_path / "captions.en.vtt"
    captions_vtt.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:05.000\n"
        "caption transcript words " * 50 + "\n"
    )
    config = SpeechConfig(
        url="https://example.com/video",
        output_dir=tmp_path,
        skip_existing=True,
        merge_sources=True,
        no_llm=False,
        local=False,
        api_key="fake-key",
    )
    data = SpeechData(
        transcript_path=whisper,
        captions_path=captions_vtt,
    )
    return config, data


class TestDAGStageSkipping:
    """Test that pipeline stages correctly skip when artefacts are fresh."""

    def test_merge_skips_when_fresh(self, tmp_path, capsys):
        config, data = _make_test_artefacts(tmp_path)
        # Create a merged file newer than sources
        time.sleep(0.05)
        merged = tmp_path / "transcript_merged.txt"
        merged.write_text("already merged text")
        merge_transcript_sources(config, data)
        out = capsys.readouterr().out
        assert "Reusing: transcript_merged.txt" in out
        assert data.merged_transcript_path == merged

    def test_merge_runs_when_source_is_newer(self, tmp_path, capsys):
        config, data = _make_test_artefacts(tmp_path)
        merged = tmp_path / "transcript_merged.txt"
        merged.write_text("old merged text")
        time.sleep(0.05)
        # Touch a source to make merged stale
        data.transcript_path.write_text("updated whisper " * 200)
        # Will try to call API and fail — that's fine, we just want to see it didn't skip
        try:
            merge_transcript_sources(config, data)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert "Reusing: transcript_merged.txt" not in out
        assert "Merging" in out

    def test_markdown_skips_when_fresh(self, tmp_path, capsys):
        config, data = _make_test_artefacts(tmp_path)
        merged = tmp_path / "transcript_merged.txt"
        merged.write_text("merged text for markdown")
        data.merged_transcript_path = merged
        time.sleep(0.05)
        md = tmp_path / "transcript.md"
        md.write_text("# existing markdown")
        generate_markdown(config, data)
        out = capsys.readouterr().out
        assert "Reusing: transcript.md" in out

    def test_markdown_regenerates_when_merged_is_newer(self, tmp_path, capsys):
        config, data = _make_test_artefacts(tmp_path)
        md = tmp_path / "transcript.md"
        md.write_text("# old markdown")
        time.sleep(0.05)
        merged = tmp_path / "transcript_merged.txt"
        merged.write_text("updated merged text")
        data.merged_transcript_path = merged
        generate_markdown(config, data)
        out = capsys.readouterr().out
        assert "Reusing: transcript.md" not in out
        assert "Markdown saved: transcript.md" in out

    def test_analysis_skips_when_fresh(self, tmp_path, capsys):
        config, data = _make_test_artefacts(tmp_path)
        merged = tmp_path / "transcript_merged.txt"
        merged.write_text("merged text")
        data.merged_transcript_path = merged
        time.sleep(0.05)
        analysis = tmp_path / "analysis.md"
        analysis.write_text("# existing analysis")
        analyze_source_survival(config, data)
        out = capsys.readouterr().out
        assert "analysis up to date" in out

    def test_analysis_runs_when_merged_is_newer(self, tmp_path, capsys):
        config, data = _make_test_artefacts(tmp_path)
        analysis = tmp_path / "analysis.md"
        analysis.write_text("# old analysis")
        time.sleep(0.05)
        merged = tmp_path / "transcript_merged.txt"
        merged.write_text("updated merged text " * 100)
        data.merged_transcript_path = merged
        analyze_source_survival(config, data)
        out = capsys.readouterr().out
        assert "analysis up to date" not in out

    def test_dry_run_skips_merge(self, tmp_path, capsys):
        config, data = _make_test_artefacts(tmp_path)
        config.dry_run = True
        merge_transcript_sources(config, data)
        out = capsys.readouterr().out
        assert "[dry-run] Would merge" in out

    def test_dry_run_analysis_when_no_merged(self, tmp_path, capsys):
        config, data = _make_test_artefacts(tmp_path)
        config.dry_run = True
        analyze_source_survival(config, data)
        out = capsys.readouterr().out
        assert "[dry-run] Would analyze" in out


# ---------------------------------------------------------------------------
# estimate_api_cost
# ---------------------------------------------------------------------------

class TestEstimateApiCost:
    def test_no_llm_returns_zero(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, no_llm=True)
        costs = estimate_api_cost(config)
        assert costs["total"] == 0.0
        assert costs["details"] == []

    def test_merge_only(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path,
                              analyze_slides=False, whisper_models=["medium"])
        costs = estimate_api_cost(config, transcript_words=6000)
        assert costs["merge_sources"] > 0
        assert costs["analyze_slides"] == 0.0
        assert costs["ensemble_whisper"] == 0.0
        assert costs["total"] == costs["merge_sources"]
        assert len(costs["details"]) == 1

    def test_slides_only(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path,
                              analyze_slides=True, merge_sources=False,
                              whisper_models=["medium"])
        costs = estimate_api_cost(config, num_slides=10)
        assert costs["analyze_slides"] == pytest.approx(0.20)
        assert costs["merge_sources"] == 0.0
        assert costs["total"] == costs["analyze_slides"]

    def test_ensemble_requires_multiple_models(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path,
                              merge_sources=False, whisper_models=["medium"])
        costs = estimate_api_cost(config)
        assert costs["ensemble_whisper"] == 0.0

        config2 = SpeechConfig(url="x", output_dir=tmp_path,
                               merge_sources=False,
                               whisper_models=["small", "medium"])
        costs2 = estimate_api_cost(config2)
        assert costs2["ensemble_whisper"] > 0

    def test_external_transcript_adds_source(self, tmp_path):
        config2 = SpeechConfig(url="x", output_dir=tmp_path,
                               whisper_models=["medium"])
        config3 = SpeechConfig(url="x", output_dir=tmp_path,
                               whisper_models=["medium"],
                               external_transcript="http://example.com")
        cost2 = estimate_api_cost(config2, transcript_words=10000)
        cost3 = estimate_api_cost(config3, transcript_words=10000)
        # 3 sources should cost more than 2
        assert cost3["merge_sources"] > cost2["merge_sources"]

    def test_total_is_sum_of_parts(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path,
                              analyze_slides=True,
                              whisper_models=["small", "medium"])
        costs = estimate_api_cost(config, num_slides=20, transcript_words=5000)
        expected = costs["analyze_slides"] + costs["merge_sources"] + costs["ensemble_whisper"]
        assert costs["total"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _load_transcript_segments
# ---------------------------------------------------------------------------

import json

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
# _format_paragraph
# ---------------------------------------------------------------------------

class TestFormatParagraph:
    def test_short_text_no_splitting(self):
        result = _format_paragraph("One sentence. Two sentence.")
        assert result == "One sentence. Two sentence."

    def test_splits_at_three_sentences(self):
        text = "First. Second. Third. Fourth. Fifth. Sixth."
        result = _format_paragraph(text)
        parts = result.split("\n\n")
        assert len(parts) == 2
        assert "First. Second. Third." in parts[0]
        assert "Fourth. Fifth. Sixth." in parts[1]

    def test_preserves_sentence_boundaries(self):
        text = "Hello world. This is a test. Another one here. And a fourth."
        result = _format_paragraph(text)
        # Should not break mid-sentence
        for part in result.split("\n\n"):
            assert part.rstrip().endswith(".")

    def test_handles_question_and_exclamation(self):
        text = "What is this? I love it! Great. Next one? Yes! Done."
        result = _format_paragraph(text)
        parts = result.split("\n\n")
        assert len(parts) == 2

    def test_single_sentence(self):
        result = _format_paragraph("Just one sentence here.")
        assert result == "Just one sentence here."

    def test_empty_string(self):
        result = _format_paragraph("")
        assert result == ""


# ---------------------------------------------------------------------------
# Whisper ensembling: _resolve_whisper_chunk
# ---------------------------------------------------------------------------

class TestResolveWhisperChunk:
    """Test _resolve_whisper_chunk with mocked LLM."""

    @patch("transcriber.create_llm_client")
    @patch("transcriber.llm_call_with_retry")
    def test_basic_resolution(self, mock_llm, mock_client, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        base_text = "Hello world"
        all_transcripts = {"medium": "Hello world", "small": "Hello worlds"}
        diff_summary = "1. small has 'worlds' vs medium has 'world'"

        mock_resp = MagicMock()
        mock_resp.content = [MagicMock()]
        mock_resp.content[0].text = "Hello world"
        mock_llm.return_value = mock_resp

        result = _resolve_whisper_chunk(base_text, all_transcripts, diff_summary, config)
        assert result == "Hello world"
        mock_llm.assert_called_once()

    @patch("transcriber.create_llm_client")
    @patch("transcriber.llm_call_with_retry")
    def test_prompt_contains_base_and_diffs(self, mock_llm, mock_client, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        base_text = "The quick brown fox"
        all_transcripts = {"medium": base_text, "small": "A quick brown fox"}
        diff_summary = "1. small: 'A' vs medium: 'The'"

        captured_kwargs = {}
        def capture_call(*args, **kwargs):
            captured_kwargs.update(kwargs)
            msg = MagicMock()
            msg.content = [MagicMock()]
            msg.content[0].text = "The quick brown fox"
            return msg
        mock_llm.side_effect = capture_call

        _resolve_whisper_chunk(base_text, all_transcripts, diff_summary, config)
        prompt = captured_kwargs["messages"][0]["content"]
        assert "The quick brown fox" in prompt
        assert "SMALL MODEL" in prompt
        assert diff_summary in prompt

    @patch("transcriber.create_llm_client")
    @patch("transcriber.llm_call_with_retry")
    def test_strips_whitespace(self, mock_llm, mock_client, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock()]
        mock_resp.content[0].text = "  result with spaces  \n"
        mock_llm.return_value = mock_resp

        result = _resolve_whisper_chunk("base", {"m": "base"}, "", config)
        assert result == "result with spaces"


# ---------------------------------------------------------------------------
# Whisper ensembling: _resolve_whisper_differences
# ---------------------------------------------------------------------------

class TestResolveWhisperDifferences:
    """Test _resolve_whisper_differences chunking and diff formatting."""

    @patch("transcriber._resolve_whisper_chunk")
    def test_short_text_single_chunk(self, mock_chunk, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, merge_chunk_words=500)
        base_text = "short text here"
        transcripts = {"medium": base_text, "small": "short texts here"}
        diffs = [{"type": "changed", "a_text": "texts", "b_text": "text",
                  "a_source": "small", "b_source": "medium"}]
        mock_chunk.return_value = "short text here"

        result = _resolve_whisper_differences(base_text, transcripts, diffs, config)
        assert result == "short text here"
        mock_chunk.assert_called_once()

    @patch("transcriber._resolve_whisper_chunk")
    def test_long_text_multiple_chunks(self, mock_chunk, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, merge_chunk_words=10)
        base_text = "word " * 25  # 25 words, should split into 3 chunks
        transcripts = {"medium": base_text, "small": base_text}
        diffs = [{"type": "a_only", "text": "extra", "source": "small"}]
        mock_chunk.return_value = "chunk result"

        result = _resolve_whisper_differences(base_text, transcripts, diffs, config)
        assert mock_chunk.call_count == 3
        assert "chunk result" in result

    @patch("transcriber._resolve_whisper_chunk")
    def test_diff_summary_formatting(self, mock_chunk, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, merge_chunk_words=500)
        diffs = [
            {"type": "changed", "a_text": "foo", "b_text": "bar",
             "a_source": "small", "b_source": "medium"},
            {"type": "a_only", "text": "extra", "source": "small"},
            {"type": "b_only", "text": "bonus", "source": "medium"},
        ]
        mock_chunk.return_value = "result"

        _resolve_whisper_differences("text", {"m": "text"}, diffs, config)
        call_args = mock_chunk.call_args
        diff_summary = call_args[0][2]  # Third positional arg
        assert 'small has "foo"' in diff_summary
        assert 'medium has "bar"' in diff_summary
        assert '"extra"' in diff_summary
        assert '"bonus"' in diff_summary


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
        assert not (tmp_path / "ensembled.txt").exists()

    def test_reuses_fresh_ensembled(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        data = self._make_whisper_data(tmp_path)
        # Create ensembled.txt newer than whisper files
        time.sleep(0.05)
        ensembled = tmp_path / "ensembled.txt"
        ensembled.write_text("cached ensembled text")
        _ensemble_whisper_transcripts(config, data)
        out = capsys.readouterr().out
        assert "Reusing: ensembled.txt" in out
        assert data.transcript_path == ensembled

    def test_no_llm_uses_base(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, no_llm=True,
                              skip_existing=False)
        data = self._make_whisper_data(tmp_path)
        # Write distinct content to trigger differences
        (tmp_path / "small.txt").write_text("hello world from small")
        (tmp_path / "medium.txt").write_text("hello world from medium")
        _ensemble_whisper_transcripts(config, data)
        out = capsys.readouterr().out
        # Should use medium (larger) as base without LLM resolution
        ensembled = (tmp_path / "ensembled.txt").read_text()
        assert "medium" in ensembled
        assert "--no-llm" in out

    @patch("transcriber._resolve_whisper_differences")
    @patch("transcriber._analyze_differences_wdiff")
    def test_calls_resolve_when_diffs_found(self, mock_analyze, mock_resolve, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        data = self._make_whisper_data(tmp_path)
        mock_analyze.return_value = [{"type": "changed", "a_text": "x", "b_text": "y"}]
        mock_resolve.return_value = "resolved text"

        _ensemble_whisper_transcripts(config, data)
        mock_resolve.assert_called_once()
        assert (tmp_path / "ensembled.txt").read_text() == "resolved text"

    @patch("transcriber._analyze_differences_wdiff")
    def test_no_diffs_uses_base(self, mock_analyze, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        data = self._make_whisper_data(tmp_path)
        mock_analyze.return_value = []

        _ensemble_whisper_transcripts(config, data)
        out = capsys.readouterr().out
        assert "No significant differences" in out

    def test_selects_largest_model_as_base(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, no_llm=True,
                              skip_existing=False)
        data = self._make_whisper_data(tmp_path, models=("tiny", "small", "large"))
        (tmp_path / "tiny.txt").write_text("tiny text differs here")
        (tmp_path / "small.txt").write_text("small text differs here")
        (tmp_path / "large.txt").write_text("large text differs here")
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
        assert not (tmp_path / "ensembled.txt").exists()


# ---------------------------------------------------------------------------
# _get_best_transcript_text
# ---------------------------------------------------------------------------

class TestGetBestTranscriptText:
    def test_prefers_merged(self, tmp_path):
        merged = tmp_path / "merged.txt"
        merged.write_text("merged content")
        whisper = tmp_path / "whisper.txt"
        whisper.write_text("whisper content")
        data = SpeechData(merged_transcript_path=merged, transcript_path=whisper)
        assert _get_best_transcript_text(data) == "merged content"

    def test_falls_back_to_whisper(self, tmp_path):
        whisper = tmp_path / "whisper.txt"
        whisper.write_text("whisper content")
        data = SpeechData(transcript_path=whisper)
        assert _get_best_transcript_text(data) == "whisper content"

    def test_returns_empty_when_nothing(self):
        data = SpeechData()
        assert _get_best_transcript_text(data) == ""

    def test_ignores_missing_merged(self, tmp_path):
        whisper = tmp_path / "whisper.txt"
        whisper.write_text("whisper content")
        data = SpeechData(
            merged_transcript_path=tmp_path / "nonexistent.txt",
            transcript_path=whisper,
        )
        assert _get_best_transcript_text(data) == "whisper content"


# ---------------------------------------------------------------------------
# _generate_sequential_markdown
# ---------------------------------------------------------------------------

class TestGenerateSequentialMarkdown:
    def test_basic_no_slides(self, tmp_path):
        transcript = tmp_path / "merged.txt"
        transcript.write_text("This is the transcript.")
        data = SpeechData(title="Test Video", merged_transcript_path=transcript)
        result = _generate_sequential_markdown(data)
        assert "# Test Video" in result
        assert "## Transcript" in result
        assert "This is the transcript." in result
        assert "merged transcript" in result  # source note

    def test_with_slides_gallery(self, tmp_path):
        transcript = tmp_path / "merged.txt"
        transcript.write_text("Transcript text.")
        slide1 = tmp_path / "slide_0001.png"
        slide2 = tmp_path / "slide_0002.png"
        slide1.write_text("img")
        slide2.write_text("img")
        data = SpeechData(
            title="Slides Talk",
            merged_transcript_path=transcript,
            slide_images=[slide1, slide2],
        )
        result = _generate_sequential_markdown(data)
        assert "## Slides" in result
        assert "### Slide 1" in result
        assert "### Slide 2" in result
        assert "slide_0001.png" in result
        assert "Title Slide" in result  # first slide is title slide

    def test_slide_metadata(self, tmp_path):
        transcript = tmp_path / "merged.txt"
        transcript.write_text("Text.")
        slide = tmp_path / "slide_0001.png"
        slide.write_text("img")
        data = SpeechData(
            title="Talk",
            merged_transcript_path=transcript,
            slide_images=[slide],
            slide_metadata=[{"title": "Introduction Slide"}],
        )
        result = _generate_sequential_markdown(data)
        assert "Introduction Slide" in result

    def test_whisper_source_note(self, tmp_path):
        transcript = tmp_path / "whisper.txt"
        transcript.write_text("Whisper text.")
        data = SpeechData(title="Talk", transcript_path=transcript)
        result = _generate_sequential_markdown(data)
        assert "Whisper transcript" in result


# ---------------------------------------------------------------------------
# _generate_interleaved_markdown
# ---------------------------------------------------------------------------

class TestGenerateInterleavedMarkdown:
    def test_text_only(self):
        data = SpeechData(
            title="Test",
            transcript_segments=[
                {"start": 0.0, "end": 2.0, "text": "Hello world."},
                {"start": 2.0, "end": 4.0, "text": "Second segment."},
            ],
        )
        result = _generate_interleaved_markdown(data)
        assert "# Test" in result
        assert "Hello world." in result
        assert "Second segment." in result

    def test_slides_interleaved(self):
        slide_img = Path("/tmp/slide_0001.png")
        data = SpeechData(
            title="Slides Talk",
            transcript_segments=[
                {"start": 0.0, "end": 5.0, "text": "Before the slide."},
                {"start": 10.0, "end": 15.0, "text": "After the slide."},
            ],
            slide_timestamps=[{"slide_number": 1, "timestamp": 6.0}],
            slide_images=[slide_img],
        )
        result = _generate_interleaved_markdown(data)
        lines = result.split("\n")
        # Find positions
        before_idx = next(i for i, l in enumerate(lines) if "Before the slide" in l)
        slide_idx = next(i for i, l in enumerate(lines) if "slide_0001.png" in l)
        after_idx = next(i for i, l in enumerate(lines) if "After the slide" in l)
        assert before_idx < slide_idx < after_idx

    def test_slide_metadata_alt_text(self):
        slide_img = Path("/tmp/slide_0001.png")
        data = SpeechData(
            title="Talk",
            transcript_segments=[{"start": 0.0, "end": 1.0, "text": "Hi."}],
            slide_timestamps=[{"slide_number": 1, "timestamp": 0.5}],
            slide_images=[slide_img],
            slide_metadata=[{"title": "My Custom Title"}],
        )
        result = _generate_interleaved_markdown(data)
        assert "My Custom Title" in result

    def test_merged_note_when_merged_exists(self, tmp_path):
        merged = tmp_path / "merged.txt"
        merged.write_text("merged")
        data = SpeechData(
            title="Talk",
            transcript_segments=[{"start": 0.0, "end": 1.0, "text": "Hi."}],
            merged_transcript_path=merged,
        )
        result = _generate_interleaved_markdown(data)
        assert "merged version available" in result


# ---------------------------------------------------------------------------
# generate_markdown — integration (layout selection)
# ---------------------------------------------------------------------------

class TestGenerateMarkdownLayout:
    def test_selects_sequential_without_timestamps(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        transcript = tmp_path / "merged.txt"
        transcript.write_text("Plain transcript.")
        data = SpeechData(title="Test", merged_transcript_path=transcript)
        generate_markdown(config, data)
        out = capsys.readouterr().out
        assert "sequential layout" in out
        content = (tmp_path / "transcript.md").read_text()
        assert "## Transcript" in content

    def test_selects_interleaved_with_timestamps(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        slide_img = tmp_path / "slides" / "slide_0001.png"
        slide_img.parent.mkdir()
        slide_img.write_text("img")
        data = SpeechData(
            title="Test",
            transcript_segments=[
                {"start": 0.0, "end": 2.0, "text": "Hello."},
            ],
            slide_timestamps=[{"slide_number": 1, "timestamp": 1.0}],
            slide_images=[slide_img],
        )
        generate_markdown(config, data)
        out = capsys.readouterr().out
        assert "timestamp-based" in out


# ---------------------------------------------------------------------------
# Pipeline: extract_slides — early returns and skip logic
# ---------------------------------------------------------------------------

class TestExtractSlides:
    def test_returns_early_without_video(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        data = SpeechData()
        # No video_path set
        extract_slides(config, data)
        assert data.slides_dir is None

    def test_returns_early_with_missing_video(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        data = SpeechData(video_path=tmp_path / "nonexistent.mp4")
        extract_slides(config, data)
        assert data.slides_dir is None

    def test_reuses_fresh_slides(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        # Create video file
        video = tmp_path / "video.mp4"
        video.write_text("fake video")
        data = SpeechData(video_path=video)
        # Create slides directory with existing slides and timestamps
        slides_dir = tmp_path / "slides"
        slides_dir.mkdir()
        slide_img = slides_dir / "slide_0001.png"
        slide_img.write_text("img")
        time.sleep(0.05)
        ts_file = tmp_path / "slide_timestamps.json"
        ts_file.write_text(json.dumps([
            {"slide_number": 1, "filename": "slide_0001.png", "timestamp": 0.0}
        ]))
        extract_slides(config, data)
        out = capsys.readouterr().out
        assert "Reusing" in out
        assert len(data.slide_images) == 1

    def test_dry_run_skips(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, dry_run=True)
        video = tmp_path / "video.mp4"
        video.write_text("fake video")
        data = SpeechData(video_path=video)
        extract_slides(config, data)
        out = capsys.readouterr().out
        assert "[dry-run]" in out


# ---------------------------------------------------------------------------
# Pipeline: analyze_slides_with_vision — early returns
# ---------------------------------------------------------------------------

class TestAnalyzeSlidesWithVision:
    def test_skips_when_disabled(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, analyze_slides=False)
        data = SpeechData()
        analyze_slides_with_vision(config, data)
        assert data.slides_json_path is None

    def test_skips_when_no_llm(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, analyze_slides=True,
                              no_llm=True)
        data = SpeechData(slide_images=[tmp_path / "slide.png"])
        analyze_slides_with_vision(config, data)
        out = capsys.readouterr().out
        assert "--no-llm" in out or data.slides_json_path is None

    def test_skips_when_no_slides(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, analyze_slides=True)
        data = SpeechData()
        analyze_slides_with_vision(config, data)
        assert data.slides_json_path is None

    def test_reuses_fresh_analysis(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, analyze_slides=True)
        slide = tmp_path / "slide_0001.png"
        slide.write_text("img")
        data = SpeechData(slide_images=[slide])
        time.sleep(0.05)
        slides_json = tmp_path / "slides_transcript.json"
        slides_json.write_text(json.dumps({"slides": [{"title": "Cached"}]}))
        analyze_slides_with_vision(config, data)
        out = capsys.readouterr().out
        assert "Reusing" in out
        assert data.slide_metadata == [{"title": "Cached"}]

    def test_dry_run_skips(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, analyze_slides=True,
                              dry_run=True, skip_existing=False)
        slide = tmp_path / "slide_0001.png"
        slide.write_text("img")
        data = SpeechData(slide_images=[slide])
        analyze_slides_with_vision(config, data)
        out = capsys.readouterr().out
        assert "[dry-run]" in out


# ---------------------------------------------------------------------------
# Pipeline: transcribe_audio — validation
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
