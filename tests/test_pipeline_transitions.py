"""Tests for data flow between pipeline stages.

Verify that each stage populates SpeechData fields that downstream stages
depend on, catching transition bugs like the ensemble→diarize segment reload
issue.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_critic.shared import (
    SpeechConfig, SpeechData,
    AUDIO_MP3, WHISPER_MERGED_TXT, DIARIZED_TXT, TRANSCRIPT_MERGED_TXT,
)


def _make_whisper_json(path: Path, num_segments: int = 3):
    """Write a realistic Whisper JSON with segments and word timestamps."""
    segments = []
    for i in range(num_segments):
        start = i * 10.0
        end = start + 9.0
        words = [
            {"word": f"word{j}", "start": start + j, "end": start + j + 0.5}
            for j in range(5)
        ]
        segments.append({
            "text": " ".join(w["word"] for w in words),
            "start": start,
            "end": end,
            "words": words,
        })
    data = {"segments": segments, "language": "en"}
    path.write_text(json.dumps(data))
    return segments


def _make_config(tmp_path, **overrides):
    defaults = dict(url="x", output_dir=tmp_path, skip_existing=False)
    defaults.update(overrides)
    return SpeechConfig(**defaults)


# ---------------------------------------------------------------------------
# Ensemble → Diarize: transcript_segments must be populated after ensemble
# ---------------------------------------------------------------------------

class TestEnsembleToDiarize:
    @patch("transcribe_critic.transcription.run_command")
    @patch("transcribe_critic.transcription.create_llm_client")
    @patch("transcribe_critic.transcription.llm_call_with_retry")
    def test_ensemble_reloads_segments_for_diarize(
        self, mock_llm, mock_client, mock_run, tmp_path
    ):
        """After ensemble, transcript_segments should be loaded from the
        largest model's JSON so diarization can use them."""
        from transcribe_critic.transcription import _ensemble_whisper_transcripts

        config = _make_config(tmp_path, whisper_models=["small", "medium"])
        data = SpeechData(audio_path=tmp_path / AUDIO_MP3)

        # Create two model transcripts on disk
        small_txt = tmp_path / "whisper_small.txt"
        medium_txt = tmp_path / "whisper_medium.txt"
        small_txt.write_text("Hello world this is a test transcript")
        medium_txt.write_text("Hello world this is a test transcript")

        small_json = tmp_path / "whisper_small.json"
        medium_json = tmp_path / "whisper_medium.json"
        _make_whisper_json(small_json)
        segs = _make_whisper_json(medium_json)

        data.whisper_transcripts = {
            "small": {"txt": small_txt, "json": small_json},
            "medium": {"txt": medium_txt, "json": medium_json},
        }
        data.transcript_path = medium_txt
        data.transcript_json_path = medium_json

        # wdiff finds no diffs (identical transcripts) → no LLM calls needed
        # But merged file still gets written
        _ensemble_whisper_transcripts(config, data)

        # Key assertions: segments must be loaded for diarizer
        assert data.transcript_segments is not None
        assert len(data.transcript_segments) == len(segs)
        for seg in data.transcript_segments:
            assert "start" in seg
            assert "end" in seg
            assert "text" in seg
            assert "words" in seg
            for w in seg["words"]:
                assert "word" in w
                assert "start" in w
                assert "end" in w


# ---------------------------------------------------------------------------
# _hydrate_data: correct field population from existing files
# ---------------------------------------------------------------------------

class TestHydrateData:
    def test_hydrate_loads_ensemble_and_segments(self, tmp_path):
        """_hydrate_data should load whisper_merged.txt as transcript_path
        and populate transcript_segments from the largest model's JSON."""
        from transcribe_critic.transcriber import _hydrate_data

        config = _make_config(tmp_path)
        data = SpeechData()

        # Create files on disk
        (tmp_path / WHISPER_MERGED_TXT).write_text("merged transcript text")
        (tmp_path / "whisper_medium.txt").write_text("medium transcript text")
        _make_whisper_json(tmp_path / "whisper_medium.json", num_segments=4)

        _hydrate_data(config, data)

        assert data.transcript_path == tmp_path / WHISPER_MERGED_TXT
        assert data.transcript_json_path == tmp_path / "whisper_medium.json"
        assert "medium" in data.whisper_transcripts
        # Segments must be loaded so diarize can use them
        assert len(data.transcript_segments) == 4
        assert "words" in data.transcript_segments[0]

    def test_hydrate_loads_diarization_path(self, tmp_path):
        from transcribe_critic.transcriber import _hydrate_data

        config = _make_config(tmp_path)
        data = SpeechData()

        (tmp_path / "whisper_medium.txt").write_text("text")
        (tmp_path / DIARIZED_TXT).write_text("[0:00:00] Speaker 1: Hello")

        _hydrate_data(config, data)

        assert data.diarization_path == tmp_path / DIARIZED_TXT

    def test_hydrate_loads_merged_transcript(self, tmp_path):
        from transcribe_critic.transcriber import _hydrate_data

        config = _make_config(tmp_path)
        data = SpeechData()

        (tmp_path / "whisper_medium.txt").write_text("text")
        (tmp_path / TRANSCRIPT_MERGED_TXT).write_text("merged text")

        _hydrate_data(config, data)

        assert data.merged_transcript_path == tmp_path / TRANSCRIPT_MERGED_TXT

    def test_hydrate_falls_back_to_largest_model(self, tmp_path):
        """Without whisper_merged.txt, should use the largest available model."""
        from transcribe_critic.transcriber import _hydrate_data

        config = _make_config(tmp_path)
        data = SpeechData()

        # Create small and medium (no merged)
        (tmp_path / "whisper_small.txt").write_text("small")
        (tmp_path / "whisper_medium.txt").write_text("medium")

        _hydrate_data(config, data)

        # medium is larger than small in MODEL_SIZES order
        assert data.transcript_path == tmp_path / "whisper_medium.txt"

    def test_hydrate_audio_path(self, tmp_path):
        from transcribe_critic.transcriber import _hydrate_data

        config = _make_config(tmp_path)
        data = SpeechData()

        (tmp_path / AUDIO_MP3).write_text("fake audio")

        _hydrate_data(config, data)

        assert data.audio_path == tmp_path / AUDIO_MP3


# ---------------------------------------------------------------------------
# Slides → Markdown: slide_images and slide_timestamps populated
# ---------------------------------------------------------------------------

class TestSlidesToMarkdown:
    @patch("transcribe_critic.slides.subprocess.run")
    def test_slides_populate_fields_for_markdown(self, mock_run, tmp_path):
        from transcribe_critic.slides import extract_slides, create_basic_slides_json

        config = _make_config(tmp_path)
        video = tmp_path / "video.mp4"
        video.write_text("fake video")
        data = SpeechData(video_path=video)

        # Simulate ffmpeg creating slides + timestamps
        slides_dir = tmp_path / "slides"
        slides_dir.mkdir()
        (slides_dir / "slide_0001.png").write_bytes(b"img1")
        (slides_dir / "slide_0002.png").write_bytes(b"img2")

        mock_run.return_value = MagicMock(
            returncode=0, stdout="",
            stderr=(
                "[Parsed_showinfo_1 @ 0x1] n:   0 pts_time:5.0\n"
                "[Parsed_showinfo_1 @ 0x1] n:   1 pts_time:30.0\n"
            ),
        )

        extract_slides(config, data)

        # Fields that markdown stage depends on
        assert len(data.slide_images) == 2
        assert len(data.slide_timestamps) == 2
        assert data.slide_timestamps[0]["timestamp"] == 5.0
        assert data.slide_timestamps[1]["filename"] == "slide_0002.png"

        # Now create_basic_slides_json should set slide_metadata
        data.title = "Test"
        create_basic_slides_json(config, data)
        assert data.slides_json_path is not None
        assert len(data.slide_metadata) == 2


# ---------------------------------------------------------------------------
# Merge → Markdown: merged_transcript_path preferred over transcript_path
# ---------------------------------------------------------------------------

class TestMergeToMarkdown:
    def test_markdown_prefers_merged_transcript(self, tmp_path):
        from transcribe_critic.output import generate_markdown

        config = _make_config(tmp_path)
        data = SpeechData()
        data.title = "Test"

        # Create both transcript files
        whisper = tmp_path / "whisper_medium.txt"
        whisper.write_text("whisper only content")
        data.transcript_path = whisper

        merged = tmp_path / TRANSCRIPT_MERGED_TXT
        merged.write_text("merged content with extra sources")
        data.merged_transcript_path = merged

        generate_markdown(config, data)

        md_text = (tmp_path / "transcript.md").read_text()
        assert "merged content with extra sources" in md_text
        assert "whisper only content" not in md_text

    def test_markdown_falls_back_to_whisper(self, tmp_path):
        from transcribe_critic.output import generate_markdown

        config = _make_config(tmp_path)
        data = SpeechData()
        data.title = "Test"

        whisper = tmp_path / "whisper_medium.txt"
        whisper.write_text("whisper fallback content")
        data.transcript_path = whisper
        # No merged_transcript_path

        generate_markdown(config, data)

        md_text = (tmp_path / "transcript.md").read_text()
        assert "whisper fallback content" in md_text


# ---------------------------------------------------------------------------
# Diarize → Merge: diarization_path must be set for structured merge
# ---------------------------------------------------------------------------

class TestDiarizeToMerge:
    def test_diarize_requires_segments(self, tmp_path, capsys):
        """Diarize should warn and return if transcript_segments is empty."""
        from transcribe_critic.diarization import diarize_audio

        config = _make_config(tmp_path, diarize=True)
        audio = tmp_path / AUDIO_MP3
        audio.write_text("fake audio")
        data = SpeechData(audio_path=audio)
        # transcript_segments is empty by default

        diarize_audio(config, data)

        out = capsys.readouterr().out
        assert "No transcript segments" in out
        assert data.diarization_path is None

    def test_diarize_requires_audio(self, tmp_path, capsys):
        """Diarize should warn if no audio file exists."""
        from transcribe_critic.diarization import diarize_audio

        config = _make_config(tmp_path, diarize=True)
        data = SpeechData(audio_path=tmp_path / "nonexistent.mp3")
        data.transcript_segments = [{"start": 0, "end": 1, "text": "hi"}]

        diarize_audio(config, data)

        out = capsys.readouterr().out
        assert "No audio file" in out
