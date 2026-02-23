"""Tests for transcriber.py — merge orchestration, cost estimation, and CLI logic."""

import re
import time
from unittest.mock import MagicMock, patch

import pytest

from shared import SpeechConfig, SpeechData, is_up_to_date
from merge import _format_structured_segments
from download import clean_vtt_captions
from output import generate_markdown
from transcriber import (
    _load_external_transcript,
    _slugify_title,
    _strip_structured_headers,
    analyze_source_survival,
    estimate_api_cost,
    merge_transcript_sources,
)


# ---------------------------------------------------------------------------
# _slugify_title
# ---------------------------------------------------------------------------

class TestSlugifyTitle:
    def test_basic(self):
        assert _slugify_title("Hello World") == "hello-world"

    def test_strips_special_chars(self):
        assert _slugify_title("Anthropic's Chief on A.I.: 'We Don't Know'") == "anthropics-chief-on-ai-we-dont-know"

    def test_truncates_long_titles(self):
        result = _slugify_title("A" * 100)
        assert len(result) <= 50

    def test_empty_title(self):
        assert _slugify_title("") == ""

    def test_preserves_hyphens(self):
        assert _slugify_title("two-part title") == "two-part-title"


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
        assert "Reusing: analysis.md" in out

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

    def test_opus_costs_more_than_sonnet(self, tmp_path):
        sonnet = SpeechConfig(url="x", output_dir=tmp_path,
                              claude_model="claude-sonnet-4-20250514",
                              whisper_models=["medium"])
        opus = SpeechConfig(url="x", output_dir=tmp_path,
                            claude_model="claude-opus-4-20250514",
                            whisper_models=["medium"])
        cost_s = estimate_api_cost(sonnet, transcript_words=5000)
        cost_o = estimate_api_cost(opus, transcript_words=5000)
        assert cost_o["merge_sources"] > cost_s["merge_sources"]

    def test_haiku_costs_less_than_sonnet(self, tmp_path):
        sonnet = SpeechConfig(url="x", output_dir=tmp_path,
                              claude_model="claude-sonnet-4-20250514",
                              whisper_models=["medium"])
        haiku = SpeechConfig(url="x", output_dir=tmp_path,
                             claude_model="claude-haiku-4-20250514",
                             whisper_models=["medium"])
        cost_s = estimate_api_cost(sonnet, transcript_words=5000)
        cost_h = estimate_api_cost(haiku, transcript_words=5000)
        assert cost_h["merge_sources"] < cost_s["merge_sources"]

    def test_unknown_model_uses_default_pricing(self, tmp_path):
        from transcriber import _get_model_pricing, DEFAULT_PRICING
        pricing = _get_model_pricing("some-unknown-model")
        assert pricing == DEFAULT_PRICING


# ---------------------------------------------------------------------------
# _load_external_transcript
# ---------------------------------------------------------------------------

class TestLoadExternalTranscript:
    def test_local_file_exists(self, tmp_path):
        f = tmp_path / "transcript.txt"
        f.write_text("  Hello world  ")
        config = SpeechConfig(url="x", output_dir=tmp_path,
                              external_transcript=str(f))
        text, label = _load_external_transcript(config)
        assert text == "Hello world"
        assert label == "transcript.txt"

    def test_local_file_missing(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path,
                              external_transcript=str(tmp_path / "nope.txt"))
        text, label = _load_external_transcript(config)
        assert text is None
        assert label == "nope.txt"

    @patch("urllib.request.urlopen")
    def test_url_plain_text(self, mock_urlopen, tmp_path):
        mock_response = MagicMock()
        mock_response.read.return_value = b"Plain transcript text"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = SpeechConfig(url="x", output_dir=tmp_path,
                              external_transcript="https://example.com/transcript.txt")
        text, label = _load_external_transcript(config)
        assert text == "Plain transcript text"
        assert label == "transcript.txt"

    @patch("transcriber._extract_text_from_html", return_value="Extracted text")
    @patch("urllib.request.urlopen")
    def test_url_html_calls_extract(self, mock_urlopen, mock_extract, tmp_path):
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html><body>Some HTML</body></html>"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = SpeechConfig(url="x", output_dir=tmp_path,
                              external_transcript="https://example.com/page")
        text, label = _load_external_transcript(config)
        assert text == "Extracted text"
        mock_extract.assert_called_once()

    @patch("urllib.request.urlopen", side_effect=Exception("Connection refused"))
    def test_url_unreachable(self, mock_urlopen, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path,
                              external_transcript="https://example.com/bad")
        text, label = _load_external_transcript(config)
        assert text is None
        out = capsys.readouterr().out
        assert "Warning" in out

    @patch("urllib.request.urlopen")
    def test_url_source_label_from_path(self, mock_urlopen, tmp_path):
        mock_response = MagicMock()
        mock_response.read.return_value = b"text"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = SpeechConfig(url="x", output_dir=tmp_path,
                              external_transcript="https://example.com/path/my_file.txt")
        _, label = _load_external_transcript(config)
        assert label == "my_file.txt"

    def test_local_source_label_is_filename(self, tmp_path):
        f = tmp_path / "my_notes.txt"
        f.write_text("notes")
        config = SpeechConfig(url="x", output_dir=tmp_path,
                              external_transcript=str(f))
        _, label = _load_external_transcript(config)
        assert label == "my_notes.txt"


# ---------------------------------------------------------------------------
# Dry-run end-to-end: full pipeline creates no files
# ---------------------------------------------------------------------------

class TestDryRunNoSideEffects:
    """Verify that dry-run mode follows the main code path but creates no files."""

    def test_full_pipeline_dry_run_creates_no_files(self, tmp_path, capsys):
        """Run all pipeline stages in dry-run mode and verify no files are created."""
        from download import download_media
        from transcription import transcribe_audio
        from slides import extract_slides, create_basic_slides_json
        from unittest.mock import patch, MagicMock
        import json

        # Mock yt-dlp to avoid network calls
        def mock_run_command(cmd, desc, verbose=False):
            if "--dump-json" in cmd:
                return MagicMock(stdout=json.dumps({
                    "title": "Test", "id": "t1", "duration": 60
                }))
            return MagicMock(stdout="", stderr="")

        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              dry_run=True, skip_existing=False)
        data = SpeechData()

        with patch("download.run_command", side_effect=mock_run_command):
            download_media(config, data)

        transcribe_audio(config, data)
        extract_slides(config, data)
        merge_transcript_sources(config, data)
        generate_markdown(config, data)
        analyze_source_survival(config, data)

        # Verify no files were created
        created_files = list(tmp_path.iterdir())
        assert created_files == [], f"Dry-run created files: {[f.name for f in created_files]}"

        # Verify all stages printed dry-run messages
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "Would download audio" in out
        assert "Would save metadata" in out


# ---------------------------------------------------------------------------
# Diarized skeleton merge routing
# ---------------------------------------------------------------------------

class TestDiarizedSkeletonRouting:
    """Test merge routing when diarized transcript provides structure."""

    def _make_diarized_artefacts(self, tmp_path, *, has_external=False, has_captions=False):
        """Create artefacts for diarized skeleton routing tests."""
        whisper = tmp_path / "ensembled.txt"
        whisper.write_text("Hello world this is a test transcript " * 30)

        diarized = tmp_path / "diarized.txt"
        diarized.write_text(
            "[0:00:00] Alice: Hello world this is a test transcript.\n"
            "[0:01:00] Bob: And here is more content for testing.\n"
        )

        config = SpeechConfig(
            url="https://example.com/video",
            output_dir=tmp_path,
            skip_existing=False,
            merge_sources=True,
            no_llm=False,
            local=False,
            api_key="fake-key",
            diarize=True,
        )
        data = SpeechData(
            transcript_path=whisper,
            diarization_path=diarized,
        )

        if has_captions:
            captions_vtt = tmp_path / "captions.en.vtt"
            captions_vtt.write_text(
                "WEBVTT\n\n"
                "00:00:01.000 --> 00:00:05.000\n"
                "Hello world this is a test transcript\n"
            )
            data.captions_path = captions_vtt

        if has_external:
            config.external_transcript = str(tmp_path / "external.txt")
            ext = tmp_path / "external.txt"
            ext.write_text(
                "Alice (0:00:00) Hello world this is a test transcript.\n"
                "Bob (0:01:00) And here is more content for testing.\n"
            )

        return config, data

    def test_diarized_only_whisper_skips_merge(self, tmp_path, capsys):
        """Diarized skeleton + only Whisper → uses diarized text directly."""
        config, data = self._make_diarized_artefacts(tmp_path)
        merge_transcript_sources(config, data)
        out = capsys.readouterr().out
        assert "Single source with diarized skeleton" in out
        assert data.merged_transcript_path is not None
        content = data.merged_transcript_path.read_text()
        assert "[0:00:00] Alice:" in content

    @patch("transcriber._merge_structured")
    def test_diarized_with_captions_calls_structured_merge(self, mock_merge, tmp_path, capsys):
        """Diarized skeleton + Whisper + captions → structured merge with 'Diarized Transcript'."""
        config, data = self._make_diarized_artefacts(tmp_path, has_captions=True)

        mock_merge.return_value = [
            {"speaker": "Alice", "timestamp": "0:00:00", "text": "Merged text one."},
            {"speaker": "Bob", "timestamp": "0:01:00", "text": "Merged text two."},
        ]

        merge_transcript_sources(config, data)

        mock_merge.assert_called_once()
        call_kwargs = mock_merge.call_args
        assert call_kwargs[1]["skeleton_source_name"] == "Diarized Transcript"

    @patch("transcriber._merge_structured")
    def test_external_takes_priority_over_diarized(self, mock_merge, tmp_path, capsys):
        """External transcript provides skeleton even when diarized exists."""
        config, data = self._make_diarized_artefacts(tmp_path, has_external=True)

        mock_merge.return_value = [
            {"speaker": "Alice", "timestamp": "0:00:00", "text": "From external."},
            {"speaker": "Bob", "timestamp": "0:01:00", "text": "Also from external."},
        ]

        merge_transcript_sources(config, data)

        mock_merge.assert_called_once()
        call_kwargs = mock_merge.call_args
        assert call_kwargs[1]["skeleton_source_name"] == "External Transcript"
        out = capsys.readouterr().out
        assert "external transcript provides structure" in out
