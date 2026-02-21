"""Tests for transcriber.py — pure functions and DAG logic."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcriber import (
    MERGE_CHECKPOINT_VERSION,
    SpeechConfig,
    SpeechData,
    _build_alignments,
    _build_wdiff_alignment,
    _compute_chunk_diffs,
    _count_fresh_chunks,
    _detect_transcript_structure,
    _dry_run_skip,
    _extract_aligned_chunk,
    _filter_meaningful_diffs,
    _format_structured_segments,
    _init_merge_chunks_dir,
    _merge_multi_source,
    _merge_structured,
    _normalize_for_comparison,
    _parse_structured_transcript,
    _parse_wdiff_tokens,
    _strip_structured_headers,
    analyze_source_survival,
    api_call_with_retry,
    clean_vtt_captions,
    generate_markdown,
    is_up_to_date,
    merge_transcript_sources,
)


# ---------------------------------------------------------------------------
# is_up_to_date
# ---------------------------------------------------------------------------

class TestIsUpToDate:
    def test_output_missing(self, tmp_path):
        output = tmp_path / "out.txt"
        inp = tmp_path / "in.txt"
        inp.write_text("x")
        assert is_up_to_date(output, inp) is False

    def test_output_newer_than_input(self, tmp_path):
        inp = tmp_path / "in.txt"
        inp.write_text("x")
        time.sleep(0.05)
        output = tmp_path / "out.txt"
        output.write_text("y")
        assert is_up_to_date(output, inp) is True

    def test_input_newer_than_output(self, tmp_path):
        output = tmp_path / "out.txt"
        output.write_text("y")
        time.sleep(0.05)
        inp = tmp_path / "in.txt"
        inp.write_text("x")
        assert is_up_to_date(output, inp) is False

    def test_no_inputs(self, tmp_path):
        output = tmp_path / "out.txt"
        output.write_text("y")
        assert is_up_to_date(output) is True

    def test_none_input_ignored(self, tmp_path):
        output = tmp_path / "out.txt"
        output.write_text("y")
        assert is_up_to_date(output, None) is True

    def test_missing_input_ignored(self, tmp_path):
        output = tmp_path / "out.txt"
        output.write_text("y")
        missing = tmp_path / "gone.txt"
        assert is_up_to_date(output, missing) is True

    def test_multiple_inputs_one_newer(self, tmp_path):
        old = tmp_path / "old.txt"
        old.write_text("x")
        time.sleep(0.05)
        output = tmp_path / "out.txt"
        output.write_text("y")
        time.sleep(0.05)
        new = tmp_path / "new.txt"
        new.write_text("z")
        assert is_up_to_date(output, old, new) is False

    def test_multiple_inputs_all_older(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("x")
        b.write_text("x")
        time.sleep(0.05)
        output = tmp_path / "out.txt"
        output.write_text("y")
        assert is_up_to_date(output, a, b) is True


# ---------------------------------------------------------------------------
# _normalize_for_comparison
# ---------------------------------------------------------------------------

class TestNormalizeForComparison:
    def test_lowercase(self):
        assert _normalize_for_comparison("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert _normalize_for_comparison("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize_for_comparison("  hello   world  ") == "hello world"

    def test_tabs_and_newlines(self):
        assert _normalize_for_comparison("hello\t\nworld") == "hello world"

    def test_apostrophes_removed(self):
        # \w includes letters and digits, ' is not \w
        assert _normalize_for_comparison("don't") == "dont"

    def test_empty_string(self):
        assert _normalize_for_comparison("") == ""


# ---------------------------------------------------------------------------
# _filter_meaningful_diffs
# ---------------------------------------------------------------------------

class TestFilterMeaningfulDiffs:
    def test_keeps_content_words(self):
        diffs = [{"type": "changed", "a_text": "quantum", "b_text": "classical"}]
        assert _filter_meaningful_diffs(diffs) == diffs

    def test_drops_pure_stopwords(self):
        diffs = [{"type": "changed", "a_text": "the", "b_text": "a"}]
        assert _filter_meaningful_diffs(diffs) == []

    def test_keeps_if_one_side_has_content(self):
        diffs = [{"type": "changed", "a_text": "the", "b_text": "Python"}]
        assert len(_filter_meaningful_diffs(diffs)) == 1

    def test_a_only_stopword_dropped(self):
        diffs = [{"type": "a_only", "text": "the"}]
        assert _filter_meaningful_diffs(diffs) == []

    def test_a_only_content_kept(self):
        diffs = [{"type": "a_only", "text": "quantum"}]
        assert len(_filter_meaningful_diffs(diffs)) == 1

    def test_multi_word_stopwords_dropped(self):
        diffs = [{"type": "changed", "a_text": "the a", "b_text": "an"}]
        assert _filter_meaningful_diffs(diffs) == []

    def test_empty_input(self):
        assert _filter_meaningful_diffs([]) == []


# ---------------------------------------------------------------------------
# _detect_transcript_structure
# ---------------------------------------------------------------------------

class TestDetectTranscriptStructure:
    def test_lex_format(self):
        text = (
            "Lex Fridman (0:00:00) Welcome to the podcast.\n"
            "Guest Name (0:01:30) Thanks for having me.\n"
            "Lex Fridman (0:02:00) Let's start.\n"
        )
        result = _detect_transcript_structure(text)
        assert result["format"] == "lex"
        assert result["has_speakers"] is True
        assert result["has_timestamps"] is True

    def test_bracketed_format(self):
        text = (
            "[0:00:00] Host: Welcome everyone.\n"
            "[0:01:30] Guest: Thank you.\n"
            "[0:02:00] Host: Let's begin.\n"
        )
        result = _detect_transcript_structure(text)
        assert result["format"] == "bracketed"
        assert result["has_speakers"] is True
        assert result["has_timestamps"] is True

    def test_speaker_only(self):
        text = (
            "Host: Welcome to the show.\n"
            "Guest: Happy to be here.\n"
            "Host: Let's get started.\n"
        )
        result = _detect_transcript_structure(text)
        assert result["format"] == "speaker_only"
        assert result["has_speakers"] is True
        assert result["has_timestamps"] is False

    def test_unstructured(self):
        text = "This is just a plain paragraph of text with no structure."
        result = _detect_transcript_structure(text)
        assert result["format"] is None
        assert result["has_speakers"] is False

    def test_skips_markdown_headers(self):
        text = (
            "# Title\n"
            "Lex Fridman (0:00:00) Hello.\n"
            "Guest (0:01:00) Hi.\n"
        )
        result = _detect_transcript_structure(text)
        assert result["format"] == "lex"

    def test_needs_two_matches(self):
        """One matching line isn't enough."""
        text = (
            "Lex Fridman (0:00:00) Hello.\n"
            "This is just text without structure.\n"
            "More plain text here.\n"
        )
        result = _detect_transcript_structure(text)
        # Only 1 lex match — not enough
        assert result["format"] is None


# ---------------------------------------------------------------------------
# _parse_structured_transcript
# ---------------------------------------------------------------------------

class TestParseStructuredTranscript:
    def test_lex_basic(self):
        text = (
            "Lex Fridman (0:00:00) Welcome.\n"
            "Guest Name (0:01:30) Thanks.\n"
        )
        segs = _parse_structured_transcript(text, "lex")
        assert len(segs) == 2
        assert segs[0]["speaker"] == "Lex Fridman"
        assert segs[0]["timestamp"] == "0:00:00"
        assert "Welcome" in segs[0]["text"]
        assert segs[1]["speaker"] == "Guest Name"

    def test_lex_continuation_lines(self):
        text = (
            "Speaker (0:00:00) First line.\n"
            "Continuation of the same segment.\n"
            "Speaker (0:01:00) Next segment.\n"
        )
        segs = _parse_structured_transcript(text, "lex")
        assert len(segs) == 2
        assert "Continuation" in segs[0]["text"]

    def test_bracketed_basic(self):
        text = (
            "[0:00:00] Host: Welcome.\n"
            "[0:01:30] Guest: Thanks.\n"
        )
        segs = _parse_structured_transcript(text, "bracketed")
        assert len(segs) == 2
        assert segs[0]["speaker"] == "Host"
        assert segs[0]["timestamp"] == "0:00:00"

    def test_speaker_only(self):
        text = (
            "Host: Welcome.\n"
            "Guest: Thanks.\n"
        )
        segs = _parse_structured_transcript(text, "speaker_only")
        assert len(segs) == 2
        assert segs[0]["timestamp"] is None
        assert segs[0]["speaker"] == "Host"

    def test_unknown_format(self):
        assert _parse_structured_transcript("some text", "unknown") == []

    def test_text_stripped(self):
        text = "Speaker (0:00:00) Hello.  \n"
        segs = _parse_structured_transcript(text, "lex")
        assert segs[0]["text"] == "Hello."


# ---------------------------------------------------------------------------
# _format_structured_segments / _strip_structured_headers (roundtrip)
# ---------------------------------------------------------------------------

class TestFormatAndStripStructuredSegments:
    def test_format_with_timestamp(self):
        segs = [{"speaker": "Alice", "timestamp": "0:01:00", "text": "Hello there."}]
        result = _format_structured_segments(segs)
        assert "**Alice** (0:01:00)" in result
        assert "Hello there." in result

    def test_format_without_timestamp(self):
        segs = [{"speaker": "Bob", "timestamp": None, "text": "Hi."}]
        result = _format_structured_segments(segs)
        assert "**Bob**" in result
        assert "()" not in result  # no empty parens

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
# api_call_with_retry
# ---------------------------------------------------------------------------

class TestApiCallWithRetry:
    @pytest.fixture
    def cfg(self, tmp_path):
        return SpeechConfig(url="x", output_dir=tmp_path)

    def test_success_on_first_try(self, cfg):
        client = MagicMock()
        client.messages.create.return_value = "ok"
        result = api_call_with_retry(client, cfg, model="test", max_tokens=100, messages=[])
        assert result == "ok"
        assert client.messages.create.call_count == 1

    @patch("transcriber.time.sleep")
    def test_retries_on_529(self, mock_sleep, cfg):
        import anthropic

        client = MagicMock()
        error = anthropic.APIStatusError(
            message="Overloaded",
            response=MagicMock(status_code=529, headers={}),
            body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
        client.messages.create.side_effect = [error, error, "ok"]
        result = api_call_with_retry(client, cfg, model="test", max_tokens=100, messages=[])
        assert result == "ok"
        assert client.messages.create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("transcriber.time.sleep")
    def test_exponential_backoff(self, mock_sleep, cfg):
        import anthropic

        client = MagicMock()
        error = anthropic.APIStatusError(
            message="Rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limited"}},
        )
        client.messages.create.side_effect = [error, error, error, "ok"]
        api_call_with_retry(client, cfg, model="test", max_tokens=100, messages=[])
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [5, 10, 20]

    @patch("transcriber.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep, cfg):
        import anthropic

        client = MagicMock()
        error = anthropic.APIStatusError(
            message="Overloaded",
            response=MagicMock(status_code=529, headers={}),
            body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
        client.messages.create.side_effect = error
        with pytest.raises(anthropic.APIStatusError):
            api_call_with_retry(client, cfg, model="test", max_tokens=100, messages=[])
        assert client.messages.create.call_count == 5

    def test_non_retryable_error_raises_immediately(self, cfg):
        import anthropic

        client = MagicMock()
        error = anthropic.APIStatusError(
            message="Bad request",
            response=MagicMock(status_code=400, headers={}),
            body={"type": "error", "error": {"type": "invalid_request_error", "message": "Bad request"}},
        )
        client.messages.create.side_effect = error
        with pytest.raises(anthropic.APIStatusError):
            api_call_with_retry(client, cfg, model="test", max_tokens=100, messages=[])
        assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# DAG integration: checkpoint staleness
# ---------------------------------------------------------------------------

class TestCheckpointStaleness:
    """Test that checkpoints are treated as DAG artefacts."""

    def test_fresh_checkpoint_reused(self, tmp_path):
        """A checkpoint newer than all sources should be reused."""
        source = tmp_path / "source.txt"
        source.write_text("input")
        time.sleep(0.05)
        checkpoint = tmp_path / "chunk_000.json"
        checkpoint.write_text(json.dumps({"text": "cached"}))
        assert is_up_to_date(checkpoint, source) is True

    def test_stale_checkpoint_rejected(self, tmp_path):
        """A checkpoint older than a source should be stale."""
        checkpoint = tmp_path / "chunk_000.json"
        checkpoint.write_text(json.dumps({"text": "old"}))
        time.sleep(0.05)
        source = tmp_path / "source.txt"
        source.write_text("updated input")
        assert is_up_to_date(checkpoint, source) is False

    def test_missing_checkpoint_is_stale(self, tmp_path):
        source = tmp_path / "source.txt"
        source.write_text("input")
        checkpoint = tmp_path / "chunk_000.json"
        assert is_up_to_date(checkpoint, source) is False

    def test_assembled_output_depends_on_chunks(self, tmp_path):
        """transcript_merged.txt should be stale if any chunk is newer."""
        chunk_a = tmp_path / "chunk_000.json"
        chunk_b = tmp_path / "chunk_001.json"
        chunk_a.write_text("a")
        chunk_b.write_text("b")
        time.sleep(0.05)
        merged = tmp_path / "transcript_merged.txt"
        merged.write_text("merged")
        assert is_up_to_date(merged, chunk_a, chunk_b) is True

        # Now touch a chunk — merged should be stale
        time.sleep(0.05)
        chunk_b.write_text("b updated")
        assert is_up_to_date(merged, chunk_a, chunk_b) is False


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
        no_api=False,
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
# _parse_wdiff_tokens
# ---------------------------------------------------------------------------

class TestParseWdiffTokens:
    def test_common_only(self):
        tokens = _parse_wdiff_tokens("hello world")
        assert tokens == [("common", "hello world")]

    def test_deleted_only(self):
        tokens = _parse_wdiff_tokens("[-removed-]")
        assert tokens == [("deleted", "removed")]

    def test_inserted_only(self):
        tokens = _parse_wdiff_tokens("{+added+}")
        assert tokens == [("inserted", "added")]

    def test_mixed(self):
        output = "hello [-old-] {+new+} world"
        tokens = _parse_wdiff_tokens(output)
        assert tokens == [
            ("common", "hello"),
            ("deleted", "old"),
            ("inserted", "new"),
            ("common", "world"),
        ]

    def test_adjacent_delete_insert(self):
        output = "[-foo-]{+bar+}"
        tokens = _parse_wdiff_tokens(output)
        assert tokens == [("deleted", "foo"), ("inserted", "bar")]

    def test_multiword_common(self):
        output = "the quick brown fox [-jumped-] {+leaped+} over the lazy dog"
        tokens = _parse_wdiff_tokens(output)
        assert tokens[0] == ("common", "the quick brown fox")
        assert tokens[1] == ("deleted", "jumped")
        assert tokens[2] == ("inserted", "leaped")
        assert tokens[3] == ("common", "over the lazy dog")

    def test_empty_string(self):
        assert _parse_wdiff_tokens("") == []

    def test_whitespace_only_common_stripped(self):
        tokens = _parse_wdiff_tokens("   ")
        assert tokens == []


# ---------------------------------------------------------------------------
# _build_wdiff_alignment
# ---------------------------------------------------------------------------

class TestBuildWdiffAlignment:
    def test_identical_texts(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        text = "hello world foo bar"
        alignment = _build_wdiff_alignment(text, text, config)
        # Identity mapping: word i in A maps to word i in B
        assert len(alignment) == 5  # 4 words + 1 sentinel
        for i in range(5):
            assert alignment[i] == i

    def test_insertion_in_b(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        text_a = "hello world"
        text_b = "hello beautiful world"
        alignment = _build_wdiff_alignment(text_a, text_b, config)
        assert len(alignment) == 3  # 2 words + sentinel
        assert alignment[0] == 0  # "hello" -> "hello" at 0
        assert alignment[1] == 2  # "world" -> "world" at 2 (after "beautiful")
        assert alignment[2] == 3  # sentinel

    def test_deletion_in_b(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        text_a = "hello beautiful world"
        text_b = "hello world"
        alignment = _build_wdiff_alignment(text_a, text_b, config)
        assert len(alignment) == 4  # 3 words + sentinel
        assert alignment[0] == 0  # "hello" -> "hello" at 0
        assert alignment[1] == 1  # "beautiful" -> position 1 (deleted, maps to current b_pos)
        assert alignment[2] == 1  # "world" -> "world" at 1
        assert alignment[3] == 2  # sentinel

    def test_monotonic(self, tmp_path):
        """Alignment positions should be non-decreasing."""
        config = SpeechConfig(url="x", output_dir=tmp_path)
        text_a = "the quick brown fox jumps over the lazy dog"
        text_b = "a quick red fox leaps over a lazy dog"
        alignment = _build_wdiff_alignment(text_a, text_b, config)
        for i in range(len(alignment) - 1):
            assert alignment[i] <= alignment[i + 1], \
                f"alignment[{i}]={alignment[i]} > alignment[{i+1}]={alignment[i+1]}"

    def test_segment_extraction_via_alignment(self, tmp_path):
        """Verify we can extract aligned segments using the alignment map."""
        config = SpeechConfig(url="x", output_dir=tmp_path)
        # Simulate two segments in text A
        seg1 = "hello world"
        seg2 = "goodbye moon"
        text_a = f"{seg1} {seg2}"
        text_b = "hello beautiful world goodbye bright moon"
        alignment = _build_wdiff_alignment(text_a, text_b, config)
        b_words = text_b.split()

        # Segment 1: words 0-1 in A
        b_start = alignment[0]
        b_end = alignment[2]  # start of seg 2
        seg1_from_b = " ".join(b_words[b_start:b_end])
        assert "hello" in seg1_from_b
        assert "world" in seg1_from_b

        # Segment 2: words 2-3 in A
        b_start = alignment[2]
        b_end = alignment[4]  # sentinel
        seg2_from_b = " ".join(b_words[b_start:b_end])
        assert "goodbye" in seg2_from_b
        assert "moon" in seg2_from_b

    def test_transcript_like_text(self, tmp_path):
        """Alignment with realistic transcript differences (via wdiff subprocess)."""
        config = SpeechConfig(url="x", output_dir=tmp_path)
        # External transcript (authoritative structure)
        text_a = (
            "Welcome to the podcast today we are talking about "
            "artificial intelligence and its impact on society"
        )
        # Whisper transcription (typical errors: missing words, substitutions)
        text_b = (
            "Welcome to the podcast today we are talking about "
            "artificial intelligence and it's impact on the society"
        )
        alignment = _build_wdiff_alignment(text_a, text_b, config)
        a_words = text_a.split()
        b_words = text_b.split()

        assert len(alignment) == len(a_words) + 1
        # First 10 words are identical — should map 1:1
        for i in range(10):
            assert alignment[i] == i, f"word {i} ({a_words[i]}) should map to {i}"

    def test_no_overlap_between_segments(self, tmp_path):
        """Cursor-based extraction: segments should not overlap in target text."""
        config = SpeechConfig(url="x", output_dir=tmp_path)
        # Three segments in external
        segs_a = [
            "the quick brown fox",
            "jumps over the lazy",
            "dog in the park",
        ]
        text_a = " ".join(segs_a)
        # Target with some insertions and substitutions
        text_b = "the fast brown fox leaps over a lazy dog in the garden"
        alignment = _build_wdiff_alignment(text_a, text_b, config)
        b_words = text_b.split()

        # Extract segments using cursor-based approach
        pos = 0
        extracted = []
        for seg_text in segs_a:
            n = len(seg_text.split())
            start, end = pos, pos + n
            b_start = alignment[start]
            b_end = alignment[end]
            extracted.append((b_start, b_end, " ".join(b_words[b_start:b_end])))
            pos = end

        # Verify no overlap: each segment's b_start >= previous segment's b_end
        for i in range(1, len(extracted)):
            prev_end = extracted[i - 1][1]
            curr_start = extracted[i][0]
            assert curr_start >= prev_end, (
                f"Segment {i} starts at {curr_start} but segment {i-1} ends at {prev_end}"
            )

    def test_large_substitution_block(self, tmp_path):
        """Alignment handles a large block of different text in the middle."""
        config = SpeechConfig(url="x", output_dir=tmp_path)
        common_start = "this is the beginning of the text"
        common_end = "and this is how we conclude"
        text_a = f"{common_start} alpha beta gamma delta {common_end}"
        text_b = f"{common_start} one two three four five {common_end}"
        alignment = _build_wdiff_alignment(text_a, text_b, config)

        a_words = text_a.split()
        b_words = text_b.split()

        # The common start (7 words) should map 1:1
        for i in range(7):
            assert alignment[i] == i

        # The common end words should map to the correct positions in B
        # A has 7 + 4 + 6 = 17 words, B has 7 + 5 + 6 = 18 words
        # Common end starts at A[11] and B[12]
        assert alignment[11] == 12  # "and"
        assert alignment[len(a_words)] == len(b_words)  # sentinel

    def test_punctuation_only_tokens_preserve_word_count(self, tmp_path):
        """Standalone punctuation tokens (em-dashes, ellipses) must not
        cause alignment map size to diverge from original word count."""
        config = SpeechConfig(url="x", output_dir=tmp_path)
        text_a = "hello — world ... end"  # 5 words, 2 are punctuation-only
        text_b = "hello world end"
        alignment = _build_wdiff_alignment(text_a, text_b, config)
        # Alignment map must have len(text_a.split()) + 1 entries
        assert len(alignment) == 6  # 5 words + 1 sentinel
        # Accessing all positions should not raise IndexError
        for i in range(len(text_a.split())):
            _ = alignment[i]


# ---------------------------------------------------------------------------
# Integration: _merge_structured end-to-end (with real wdiff)
# ---------------------------------------------------------------------------

class TestMergeStructuredEndToEnd:
    """Test _merge_structured with real wdiff subprocess, only mocking the API."""

    @patch("transcriber.api_call_with_retry")
    def test_blind_merge_with_real_wdiff(self, mock_api, tmp_path):
        """Full pipeline: wdiff alignment + anonymous prompting + response parsing."""
        segments = [
            {"speaker": "Host", "timestamp": "0:00:00",
             "text": "Welcome to the podcast today"},
            {"speaker": "Guest", "timestamp": "0:01:00",
             "text": "Thanks for having me here"},
        ]

        ext_text = " ".join(s["text"] for s in segments)
        all_sources = [
            ("External Transcript", "external", ext_text),
            ("Whisper AI Transcript", "whisper",
             "Welcome to the podcast today Thanks for having me here"),
            ("YouTube Captions", "captions",
             "Welcome to the podcast today Thanks for having me"),
        ]

        source_paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
        for p in source_paths:
            p.write_text("x")

        config = SpeechConfig(url="x", output_dir=tmp_path)

        def fake_api_response(*args, **kwargs):
            # Verify the prompt is anonymous (no source names, no speaker labels)
            prompt = kwargs.get("messages", [{}])[0].get("content", "")
            assert "External Transcript" not in prompt
            assert "Whisper" not in prompt
            assert "YouTube" not in prompt
            assert "Host" not in prompt
            assert "Guest" not in prompt
            assert "Source 1:" in prompt
            assert "Source 2:" in prompt
            assert "PASSAGE 1:" in prompt
            assert "PASSAGE 2:" in prompt

            msg = MagicMock()
            msg.content = [MagicMock()]
            msg.content[0].text = (
                "PASSAGE 1: Welcome to the podcast today\n"
                "PASSAGE 2: Thanks for having me here"
            )
            return msg

        mock_api.side_effect = fake_api_response

        result = _merge_structured("fake-key", segments, all_sources, config, source_paths)

        assert len(result) == 2
        # Speaker/timestamp restored from skeleton
        assert result[0]["speaker"] == "Host"
        assert result[0]["timestamp"] == "0:00:00"
        assert result[1]["speaker"] == "Guest"
        assert result[1]["timestamp"] == "0:01:00"
        # Merged text from API response
        assert result[0]["text"] == "Welcome to the podcast today"
        assert result[1]["text"] == "Thanks for having me here"

    @patch("transcriber.api_call_with_retry")
    def test_alignment_extracts_correct_segments_for_prompt(self, mock_api, tmp_path):
        """Verify wdiff alignment produces sensible per-segment text in the prompt."""
        segments = [
            {"speaker": "A", "timestamp": "0:00:00",
             "text": "the quick brown fox"},
            {"speaker": "B", "timestamp": "0:01:00",
             "text": "jumps over the lazy dog"},
        ]

        ext_text = " ".join(s["text"] for s in segments)
        all_sources = [
            ("External Transcript", "external", ext_text),
            ("Whisper AI Transcript", "whisper",
             "the quick brown fox jumps over the lazy dog"),
            ("YouTube Captions", "captions",
             "a quick brown fox jumps over a lazy dog"),
        ]

        source_paths = [tmp_path / "src.txt"]
        source_paths[0].write_text("x")

        config = SpeechConfig(url="x", output_dir=tmp_path)

        captured_prompts = []

        def fake_api_response(*args, **kwargs):
            prompt = kwargs.get("messages", [{}])[0].get("content", "")
            captured_prompts.append(prompt)
            msg = MagicMock()
            msg.content = [MagicMock()]
            msg.content[0].text = (
                "PASSAGE 1: the quick brown fox\n"
                "PASSAGE 2: jumps over the lazy dog"
            )
            return msg

        mock_api.side_effect = fake_api_response

        _merge_structured("fake-key", segments, all_sources, config, source_paths)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # PASSAGE 1 should contain "fox" from all sources
        # PASSAGE 2 should contain "jumps" from all sources
        assert "PASSAGE 1:" in prompt
        assert "PASSAGE 2:" in prompt
        # Each source should be represented for each passage
        assert prompt.count("Source 1:") == 2
        assert prompt.count("Source 2:") == 2
        assert prompt.count("Source 3:") == 2


# ---------------------------------------------------------------------------
# Shared merge helpers
# ---------------------------------------------------------------------------

class TestInitMergeChunksDir:
    def test_creates_directory_and_version(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        chunks_dir = _init_merge_chunks_dir(config)
        assert chunks_dir.exists()
        assert (chunks_dir / ".version").read_text().strip() == MERGE_CHECKPOINT_VERSION

    def test_clears_old_chunks_on_version_mismatch(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        chunks_dir = tmp_path / "merge_chunks"
        chunks_dir.mkdir()
        (chunks_dir / ".version").write_text("OLD")
        (chunks_dir / "chunk_000.json").write_text("{}")
        _init_merge_chunks_dir(config)
        assert not (chunks_dir / "chunk_000.json").exists()
        assert (chunks_dir / ".version").read_text().strip() == MERGE_CHECKPOINT_VERSION

    def test_preserves_chunks_when_version_matches(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        chunks_dir = tmp_path / "merge_chunks"
        chunks_dir.mkdir()
        (chunks_dir / ".version").write_text(MERGE_CHECKPOINT_VERSION)
        (chunks_dir / "chunk_000.json").write_text("{}")
        _init_merge_chunks_dir(config)
        assert (chunks_dir / "chunk_000.json").exists()


class TestCountFreshChunks:
    def test_no_chunks_exist(self, tmp_path):
        source = tmp_path / "src.txt"
        source.write_text("x")
        assert _count_fresh_chunks(3, tmp_path, [source]) == 0

    def test_all_fresh(self, tmp_path):
        source = tmp_path / "src.txt"
        source.write_text("x")
        time.sleep(0.05)
        for i in range(3):
            (tmp_path / f"chunk_{i:03d}.json").write_text("{}")
        assert _count_fresh_chunks(3, tmp_path, [source]) == 3

    def test_partial_fresh(self, tmp_path):
        source = tmp_path / "src.txt"
        source.write_text("x")
        time.sleep(0.05)
        (tmp_path / "chunk_000.json").write_text("{}")
        # chunk_001 missing → stops at 1
        assert _count_fresh_chunks(3, tmp_path, [source]) == 1


class TestExtractAlignedChunk:
    def test_identity_alignment(self):
        anchor_words = ["hello", "world", "foo", "bar"]
        alignment = list(range(5))  # identity + sentinel
        other_words = [["hello", "world", "foo", "bar"]]
        result = _extract_aligned_chunk(anchor_words, 0, 2, [alignment], [other_words[0]])
        assert result[0] == "hello world"  # anchor
        assert result[1] == "hello world"  # other

    def test_with_insertion(self):
        anchor_words = ["hello", "world"]
        alignment = [0, 2, 3]  # "hello" -> 0, "world" -> 2, sentinel -> 3
        other_words = [["hello", "beautiful", "world"]]
        result = _extract_aligned_chunk(anchor_words, 0, 2, [alignment], [other_words[0]])
        assert result[0] == "hello world"
        assert result[1] == "hello beautiful world"


class TestComputeChunkDiffs:
    def test_identical_texts_no_diffs(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        result = _compute_chunk_diffs(["hello world", "hello world"], config)
        assert result == ""

    def test_different_texts_produce_diffs(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        result = _compute_chunk_diffs(["hello world", "hello earth"], config)
        assert "Source 1" in result
        assert "Source 2" in result


# ---------------------------------------------------------------------------
# Integration: _merge_multi_source end-to-end (with real wdiff)
# ---------------------------------------------------------------------------

class TestMergeMultiSourceEndToEnd:
    """Test _merge_multi_source with real wdiff, only mocking the API."""

    @patch("transcriber.api_call_with_retry")
    def test_anonymous_prompt_no_source_names(self, mock_api, tmp_path):
        """Source names should not appear in the prompt sent to Claude."""
        sources = [
            ("Whisper AI Transcript", "whisper", "hello world foo bar " * 50),
            ("YouTube Captions", "captions", "hello world foo bar " * 50),
        ]
        source_paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
        for p in source_paths:
            p.write_text("x")

        config = SpeechConfig(url="x", output_dir=tmp_path)

        def fake_api_response(*args, **kwargs):
            prompt = kwargs.get("messages", [{}])[0].get("content", "")
            assert "Whisper" not in prompt
            assert "YouTube" not in prompt
            assert "Source 1" in prompt
            assert "Source 2" in prompt
            msg = MagicMock()
            msg.content = [MagicMock()]
            msg.content[0].text = "hello world foo bar " * 50
            return msg

        mock_api.side_effect = fake_api_response

        result = _merge_multi_source("fake-key", sources, config, source_paths)
        assert len(result) > 0

    @patch("transcriber.api_call_with_retry")
    def test_three_sources(self, mock_api, tmp_path):
        """Flat merge with 3 sources should work with wdiff alignment."""
        text = "the quick brown fox jumps over the lazy dog " * 60
        sources = [
            ("Whisper AI Transcript", "whisper", text),
            ("YouTube Captions", "captions", text),
            ("External Transcript", "external", text),
        ]
        source_paths = [tmp_path / "src.txt"]
        source_paths[0].write_text("x")

        config = SpeechConfig(url="x", output_dir=tmp_path)

        def fake_api_response(*args, **kwargs):
            prompt = kwargs.get("messages", [{}])[0].get("content", "")
            assert "Source 1:" in prompt
            assert "Source 2:" in prompt
            assert "Source 3:" in prompt
            msg = MagicMock()
            msg.content = [MagicMock()]
            msg.content[0].text = "merged text chunk"
            return msg

        mock_api.side_effect = fake_api_response

        result = _merge_multi_source("fake-key", sources, config, source_paths)
        assert len(result) > 0

    @patch("transcriber.api_call_with_retry")
    def test_checkpoint_versioning(self, mock_api, tmp_path):
        """Flat merge should use checkpoint versioning."""
        text = "word " * 600
        sources = [
            ("Source A", "a", text),
            ("Source B", "b", text),
        ]
        source_paths = [tmp_path / "src.txt"]
        source_paths[0].write_text("x")
        config = SpeechConfig(url="x", output_dir=tmp_path)

        def fake_api_response(*args, **kwargs):
            msg = MagicMock()
            msg.content = [MagicMock()]
            msg.content[0].text = "merged"
            return msg

        mock_api.side_effect = fake_api_response

        _merge_multi_source("fake-key", sources, config, source_paths)

        chunks_dir = tmp_path / "merge_chunks"
        assert (chunks_dir / ".version").read_text().strip() == MERGE_CHECKPOINT_VERSION


# ---------------------------------------------------------------------------
# Integration: checkpoint file integrity (_merge_structured)
# ---------------------------------------------------------------------------

class TestCheckpointFileIntegrity:
    """Test that _merge_structured writes one checkpoint file per chunk."""

    def _make_segments_and_sources(self, num_segments):
        """Build segments and sources for checkpoint tests.

        Returns (segments, all_sources) where all_sources includes an
        External Transcript (required by the blind merge) plus two others.
        """
        config = SpeechConfig(url="x", output_dir=Path("/tmp"))
        words_per_seg = config.merge_chunk_words // 2  # 2 segments per chunk
        segments = []
        for i in range(num_segments):
            segments.append({
                "speaker": "Speaker",
                "timestamp": f"0:{i:02d}:00",
                "text": f"word{i} " * words_per_seg,
            })

        ext_text = " ".join(s["text"] for s in segments)
        all_sources = [
            ("External Transcript", "external", ext_text),
            ("Whisper AI Transcript", "whisper", ext_text),
            ("YouTube Captions", "captions", ext_text),
        ]
        return segments, all_sources

    def _fake_passage_response(self, num_passages):
        """Return a mock API response with PASSAGE N: format."""
        def responder(*args, **kwargs):
            msg = MagicMock()
            msg.content = [MagicMock()]
            lines = []
            for p in range(1, num_passages + 1):
                lines.append(f"PASSAGE {p}: Merged text for passage {p}.")
            msg.content[0].text = "\n".join(lines)
            return msg
        return responder

    @patch("transcriber._build_wdiff_alignment")
    @patch("transcriber.api_call_with_retry")
    def test_each_chunk_gets_own_checkpoint(self, mock_api, mock_align, tmp_path):
        """Verify the checkpoint_path bug fix: each chunk writes a distinct file."""
        segments, all_sources = self._make_segments_and_sources(6)

        # Mock alignment: identity mapping (same text in all sources)
        word_count = sum(len(s["text"].split()) for s in segments)
        mock_align.return_value = list(range(word_count + 1))

        source_paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
        for p in source_paths:
            p.write_text("x")

        config = SpeechConfig(url="x", output_dir=tmp_path)

        # 2 segments per chunk → 3 chunks → each API call gets 2 passages
        mock_api.side_effect = self._fake_passage_response(2)

        result = _merge_structured("fake-key", segments, all_sources, config, source_paths)

        # Should have 3 chunks (6 segments / 2 per chunk)
        chunks_dir = tmp_path / "merge_chunks"
        assert chunks_dir.exists()
        chunk_files = sorted(chunks_dir.glob("chunk_*.json"))
        assert len(chunk_files) == 3
        assert chunk_files[0].name == "chunk_000.json"
        assert chunk_files[1].name == "chunk_001.json"
        assert chunk_files[2].name == "chunk_002.json"

        # Each file should contain valid JSON with 2 segments
        for f in chunk_files:
            data = json.loads(f.read_text())
            assert len(data) == 2
            assert "speaker" in data[0]
            assert "text" in data[0]

        # 3 API calls (one per chunk, none reused)
        assert mock_api.call_count == 3

        # Version file should exist
        version_file = chunks_dir / ".version"
        assert version_file.exists()
        assert version_file.read_text().strip() == MERGE_CHECKPOINT_VERSION

    @patch("transcriber._build_wdiff_alignment")
    @patch("transcriber.api_call_with_retry")
    def test_partial_resume_skips_fresh_chunks(self, mock_api, mock_align, tmp_path):
        """Pre-existing fresh checkpoints should be loaded, not re-processed."""
        segments, all_sources = self._make_segments_and_sources(4)

        word_count = sum(len(s["text"].split()) for s in segments)
        mock_align.return_value = list(range(word_count + 1))

        # Create source files first
        source_paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
        for p in source_paths:
            p.write_text("x")

        # Pre-create checkpoint for chunk 0 (newer than sources)
        time.sleep(0.05)
        chunks_dir = tmp_path / "merge_chunks"
        chunks_dir.mkdir()
        (chunks_dir / ".version").write_text(MERGE_CHECKPOINT_VERSION)
        pre_existing = [
            {"speaker": "Speaker", "timestamp": "0:00:00", "text": "Cached segment one."},
            {"speaker": "Speaker", "timestamp": "0:01:00", "text": "Cached segment two."},
        ]
        (chunks_dir / "chunk_000.json").write_text(json.dumps(pre_existing))

        config = SpeechConfig(url="x", output_dir=tmp_path)

        mock_api.side_effect = self._fake_passage_response(2)

        result = _merge_structured("fake-key", segments, all_sources, config, source_paths)

        # Only 1 API call — chunk 0 was reused
        assert mock_api.call_count == 1

        # Result should have 4 segments total (2 cached + 2 from API)
        assert len(result) == 4
        assert result[0]["text"] == "Cached segment one."
        assert result[1]["text"] == "Cached segment two."

    @patch("transcriber._build_wdiff_alignment")
    @patch("transcriber.api_call_with_retry")
    def test_stale_checkpoint_is_reprocessed(self, mock_api, mock_align, tmp_path):
        """A checkpoint older than a source should be re-processed via API."""
        segments, all_sources = self._make_segments_and_sources(2)

        word_count = sum(len(s["text"].split()) for s in segments)
        mock_align.return_value = list(range(word_count + 1))

        # Create stale checkpoint BEFORE source files
        chunks_dir = tmp_path / "merge_chunks"
        chunks_dir.mkdir()
        (chunks_dir / ".version").write_text(MERGE_CHECKPOINT_VERSION)
        stale_data = [
            {"speaker": "Speaker", "timestamp": "0:00:00", "text": "Stale."},
            {"speaker": "Speaker", "timestamp": "0:01:00", "text": "Stale."},
        ]
        (chunks_dir / "chunk_000.json").write_text(json.dumps(stale_data))

        time.sleep(0.05)
        source_paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
        for p in source_paths:
            p.write_text("updated source")

        config = SpeechConfig(url="x", output_dir=tmp_path)

        mock_api.side_effect = self._fake_passage_response(2)

        result = _merge_structured("fake-key", segments, all_sources, config, source_paths)

        # API should be called — stale checkpoint not reused
        assert mock_api.call_count == 1
        assert result[0]["text"] == "Merged text for passage 1."

    @patch("transcriber._build_wdiff_alignment")
    @patch("transcriber.api_call_with_retry")
    def test_version_mismatch_clears_old_checkpoints(self, mock_api, mock_align, tmp_path):
        """Checkpoint version mismatch should clear old chunk files."""
        segments, all_sources = self._make_segments_and_sources(2)

        word_count = sum(len(s["text"].split()) for s in segments)
        mock_align.return_value = list(range(word_count + 1))

        source_paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
        for p in source_paths:
            p.write_text("x")

        # Pre-create chunk with OLD version
        chunks_dir = tmp_path / "merge_chunks"
        chunks_dir.mkdir()
        (chunks_dir / ".version").write_text("OLD_VERSION")
        old_chunk = chunks_dir / "chunk_000.json"
        old_chunk.write_text(json.dumps([{"speaker": "S", "timestamp": "0:00:00", "text": "old"}]))

        config = SpeechConfig(url="x", output_dir=tmp_path)

        mock_api.side_effect = self._fake_passage_response(2)

        result = _merge_structured("fake-key", segments, all_sources, config, source_paths)

        # Old chunk should have been cleared and re-processed
        assert mock_api.call_count == 1
        # Version file should be updated
        assert (chunks_dir / ".version").read_text().strip() == MERGE_CHECKPOINT_VERSION
