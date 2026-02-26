"""Tests for the summarize pipeline step."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_critic.shared import SpeechConfig, SpeechData, SUMMARY_MD
from transcribe_critic.summarize import (
    _get_best_transcript,
    _resolve_summary_config,
    summarize_transcript,
)


# --- _get_best_transcript ---

class TestGetBestTranscript:
    def test_prefers_diarized(self, tmp_path):
        diarized = tmp_path / "diarized.txt"
        diarized.write_text("[0:00:00] Speaker: diarized text")
        merged = tmp_path / "merged.txt"
        merged.write_text("merged text")

        data = SpeechData(
            diarization_path=diarized,
            merged_transcript_path=merged,
        )
        assert _get_best_transcript(data) == "[0:00:00] Speaker: diarized text"

    def test_falls_back_to_merged(self, tmp_path):
        merged = tmp_path / "merged.txt"
        merged.write_text("merged text")
        whisper = tmp_path / "whisper.txt"
        whisper.write_text("whisper text")

        data = SpeechData(
            merged_transcript_path=merged,
            transcript_path=whisper,
        )
        assert _get_best_transcript(data) == "merged text"

    def test_falls_back_to_transcript(self, tmp_path):
        whisper = tmp_path / "whisper.txt"
        whisper.write_text("whisper text")

        data = SpeechData(transcript_path=whisper)
        assert _get_best_transcript(data) == "whisper text"

    def test_falls_back_to_captions(self, tmp_path):
        captions = tmp_path / "captions.vtt"
        captions.write_text("caption text")

        data = SpeechData(captions_path=captions)
        assert _get_best_transcript(data) == "caption text"

    def test_returns_none_when_empty(self):
        data = SpeechData()
        assert _get_best_transcript(data) is None

    def test_skips_empty_files(self, tmp_path):
        merged = tmp_path / "merged.txt"
        merged.write_text("")
        whisper = tmp_path / "whisper.txt"
        whisper.write_text("whisper text")

        data = SpeechData(
            merged_transcript_path=merged,
            transcript_path=whisper,
        )
        assert _get_best_transcript(data) == "whisper text"


# --- _resolve_summary_config ---

class TestResolveSummaryConfig:
    def test_inherits_main_when_no_overrides(self):
        config = SpeechConfig(url="test", output_dir=Path("/tmp"))
        resolved = _resolve_summary_config(config)
        assert resolved is config  # no copy needed

    def test_overrides_local_to_api(self):
        config = SpeechConfig(
            url="test", output_dir=Path("/tmp"),
            local=True,
            summary_local=False,
        )
        resolved = _resolve_summary_config(config)
        assert resolved.local is False

    def test_overrides_model_for_api(self):
        config = SpeechConfig(
            url="test", output_dir=Path("/tmp"),
            local=False,
            claude_model="claude-sonnet-4-20250514",
            summary_model="claude-opus-4-20250514",
        )
        resolved = _resolve_summary_config(config)
        assert resolved.claude_model == "claude-opus-4-20250514"

    def test_overrides_model_for_local(self):
        config = SpeechConfig(
            url="test", output_dir=Path("/tmp"),
            local=True,
            local_model="qwen2.5:14b",
            summary_model="llama3:70b",
        )
        resolved = _resolve_summary_config(config)
        assert resolved.local_model == "llama3:70b"

    def test_overrides_api_key(self):
        config = SpeechConfig(
            url="test", output_dir=Path("/tmp"),
            api_key="main-key",
            summary_api_key="summary-key",
        )
        resolved = _resolve_summary_config(config)
        assert resolved.api_key == "summary-key"

    def test_summary_api_switches_backend_and_model(self):
        config = SpeechConfig(
            url="test", output_dir=Path("/tmp"),
            local=True,
            local_model="qwen2.5:14b",
            summary_local=False,
            summary_model="claude-opus-4-20250514",
        )
        resolved = _resolve_summary_config(config)
        assert resolved.local is False
        assert resolved.claude_model == "claude-opus-4-20250514"


# --- summarize_transcript ---

class TestSummarizeTranscript:
    def test_skips_when_disabled(self, tmp_path, capsys):
        config = SpeechConfig(url="test", output_dir=tmp_path, summarize=False)
        data = SpeechData()
        summarize_transcript(config, data)
        assert "Skipped (summarization disabled)" in capsys.readouterr().out
        assert data.summary_path is None

    def test_skips_when_no_llm(self, tmp_path, capsys):
        config = SpeechConfig(url="test", output_dir=tmp_path, no_llm=True, skip_existing=False)
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("Hello world")
        data = SpeechData(transcript_path=transcript)
        summarize_transcript(config, data)
        assert "Skipped (--no-llm)" in capsys.readouterr().out

    def test_skips_when_no_transcript(self, tmp_path, capsys):
        config = SpeechConfig(url="test", output_dir=tmp_path, skip_existing=False)
        data = SpeechData()
        summarize_transcript(config, data)
        assert "No transcript available" in capsys.readouterr().out

    def test_dry_run(self, tmp_path, capsys):
        config = SpeechConfig(url="test", output_dir=tmp_path, dry_run=True)
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("Hello world")
        data = SpeechData(transcript_path=transcript)
        summarize_transcript(config, data)
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "summary.md" in out
        assert not (tmp_path / SUMMARY_MD).exists()

    def test_reuses_existing(self, tmp_path, capsys):
        summary = tmp_path / SUMMARY_MD
        summary.write_text("Existing summary")
        config = SpeechConfig(url="test", output_dir=tmp_path)
        data = SpeechData()
        summarize_transcript(config, data)
        assert data.summary_path == summary
        assert "Reusing" in capsys.readouterr().out

    @patch("transcribe_critic.summarize.llm_call_with_retry")
    @patch("transcribe_critic.summarize.create_llm_client")
    def test_calls_llm_and_saves(self, mock_client, mock_llm, tmp_path, capsys):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="## Summary\n\nThis is a summary.")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_llm.return_value = mock_response

        transcript = tmp_path / "transcript.txt"
        transcript.write_text("Some long transcript text here")

        config = SpeechConfig(url="test", output_dir=tmp_path, skip_existing=False)
        data = SpeechData(transcript_path=transcript)

        summarize_transcript(config, data)

        assert data.summary_path == tmp_path / SUMMARY_MD
        assert data.summary_path.read_text().strip() == "## Summary\n\nThis is a summary."
        mock_llm.assert_called_once()
        assert "Summary saved" in capsys.readouterr().out

    @patch("transcribe_critic.summarize.llm_call_with_retry")
    @patch("transcribe_critic.summarize.create_llm_client")
    def test_uses_summary_backend(self, mock_client, mock_llm, tmp_path):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_llm.return_value = mock_response

        transcript = tmp_path / "transcript.txt"
        transcript.write_text("Transcript")

        config = SpeechConfig(
            url="test", output_dir=tmp_path,
            local=True,
            summary_local=False,
            summary_model="claude-opus-4-20250514",
            skip_existing=False,
        )
        data = SpeechData(transcript_path=transcript)

        summarize_transcript(config, data)

        # The resolved config passed to create_llm_client should have local=False
        call_args = mock_client.call_args[0][0]
        assert call_args.local is False
        assert call_args.claude_model == "claude-opus-4-20250514"
