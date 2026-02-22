"""Tests for transcription.py — Whisper transcription, ensembling, and segment loading."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from shared import SpeechConfig, SpeechData

from transcription import (
    _ensemble_whisper_transcripts,
    _load_transcript_segments,
    _resolve_whisper_chunk,
    _resolve_whisper_differences,
    _run_whisper_model,
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
# Whisper ensembling: _resolve_whisper_chunk
# ---------------------------------------------------------------------------

class TestResolveWhisperChunk:
    """Test _resolve_whisper_chunk with mocked LLM."""

    @patch("transcription.create_llm_client")
    @patch("transcription.llm_call_with_retry")
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

    @patch("transcription.create_llm_client")
    @patch("transcription.llm_call_with_retry")
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

    @patch("transcription.create_llm_client")
    @patch("transcription.llm_call_with_retry")
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

    @patch("transcription._resolve_whisper_chunk")
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

    @patch("transcription._resolve_whisper_chunk")
    def test_long_text_multiple_chunks(self, mock_chunk, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, merge_chunk_words=10)
        base_text = "word " * 25  # 25 words, should split into 3 chunks
        transcripts = {"medium": base_text, "small": base_text}
        diffs = [{"type": "a_only", "text": "extra", "source": "small"}]
        mock_chunk.return_value = "chunk result"

        result = _resolve_whisper_differences(base_text, transcripts, diffs, config)
        assert mock_chunk.call_count == 3
        assert "chunk result" in result

    @patch("transcription._resolve_whisper_chunk")
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

    @patch("transcription._resolve_whisper_differences")
    @patch("transcription._analyze_differences_wdiff")
    def test_calls_resolve_when_diffs_found(self, mock_analyze, mock_resolve, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        data = self._make_whisper_data(tmp_path)
        mock_analyze.return_value = [{"type": "changed", "a_text": "x", "b_text": "y"}]
        mock_resolve.return_value = "resolved text"

        _ensemble_whisper_transcripts(config, data)
        mock_resolve.assert_called_once()
        assert (tmp_path / "ensembled.txt").read_text() == "resolved text"

    @patch("transcription._analyze_differences_wdiff")
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
        txt = tmp_path / "medium.txt"
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

    @patch("transcription.run_command")
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
        assert (tmp_path / "small.txt").exists()
        assert (tmp_path / "small.json").exists()
        assert data.whisper_transcripts["small"]["txt"] == tmp_path / "small.txt"

    @patch("transcription.run_command")
    def test_mlx_whisper_unlinks_existing_before_rename(self, mock_run, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        data = SpeechData(audio_path=tmp_path / "audio.mp3")
        data.audio_path.write_text("fake")
        deps = {"mlx_whisper": True, "whisper": False}
        # Pre-create target files (from a previous run)
        (tmp_path / "small.txt").write_text("old")
        (tmp_path / "small.json").write_text("old")

        def create_default_files(cmd, desc, verbose=False):
            (tmp_path / "audio.txt").write_text("new text")
            (tmp_path / "audio.json").write_text('{"segments": []}')
            return MagicMock()
        mock_run.side_effect = create_default_files

        _run_whisper_model(config, data, "small", deps)
        assert (tmp_path / "small.txt").read_text() == "new text"

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

        assert (tmp_path / "medium.txt").read_text() == "hello world"
        assert (tmp_path / "medium.json").exists()
        assert data.whisper_transcripts["medium"]["txt"] == tmp_path / "medium.txt"

    @patch("transcription.run_command")
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

    @patch("transcription.check_dependencies",
           return_value={"mlx_whisper": False, "whisper": False})
    def test_raises_without_whisper(self, mock_deps, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path)
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake")
        data = SpeechData(audio_path=audio)
        with pytest.raises(RuntimeError, match="No Whisper"):
            transcribe_audio(config, data)

    @patch("transcription._load_transcript_segments")
    @patch("transcription._run_whisper_model")
    @patch("transcription.check_dependencies",
           return_value={"mlx_whisper": True, "whisper": False})
    def test_single_model_uses_directly(self, mock_deps, mock_run, mock_load, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, whisper_models=["medium"])
        audio = tmp_path / "audio.mp3"
        audio.write_text("fake")
        txt = tmp_path / "medium.txt"
        txt.write_text("transcript")
        data = SpeechData(audio_path=audio)

        def populate_transcripts(cfg, d, model, deps):
            d.whisper_transcripts[model] = {"txt": txt, "json": None}
        mock_run.side_effect = populate_transcripts

        transcribe_audio(config, data)
        assert data.transcript_path == txt
        mock_run.assert_called_once()

    @patch("transcription._load_transcript_segments")
    @patch("transcription._ensemble_whisper_transcripts")
    @patch("transcription._run_whisper_model")
    @patch("transcription.check_dependencies",
           return_value={"mlx_whisper": True, "whisper": False})
    def test_multiple_models_calls_ensemble(self, mock_deps, mock_run, mock_ensemble,
                                             mock_load, tmp_path):
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

        def set_ensembled(cfg, d):
            d.transcript_path = tmp_path / "ensembled.txt"
            d.transcript_path.write_text("ensembled")
        mock_ensemble.side_effect = set_ensembled

        transcribe_audio(config, data)
        assert mock_run.call_count == 2
        mock_ensemble.assert_called_once()

    @patch("transcription._load_transcript_segments")
    @patch("transcription._run_whisper_model")
    @patch("transcription.check_dependencies",
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
