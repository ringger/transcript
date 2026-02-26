"""Tests for slides.py — slide extraction and vision analysis."""

import json
import time
from unittest.mock import MagicMock, patch

from transcribe_critic.shared import SpeechConfig, SpeechData

from transcribe_critic.slides import (
    _load_slide_timestamps,
    analyze_slides_with_vision,
    create_basic_slides_json,
    extract_slides,
)


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
# _load_slide_timestamps
# ---------------------------------------------------------------------------

class TestLoadSlideTimestamps:
    def test_loads_valid_json(self, tmp_path, capsys):
        ts_file = tmp_path / "slide_timestamps.json"
        ts_file.write_text(json.dumps([
            {"slide_number": 1, "filename": "slide_0001.png", "timestamp": 5.0},
            {"slide_number": 2, "filename": "slide_0002.png", "timestamp": 30.0},
        ]))
        data = SpeechData(slide_images=[
            tmp_path / "slide_0001.png",
            tmp_path / "slide_0002.png",
        ])
        _load_slide_timestamps(data, ts_file)
        assert len(data.slide_timestamps) == 2
        assert data.slide_timestamps[0]["timestamp"] == 5.0
        assert data.slide_timestamps[1]["filename"] == "slide_0002.png"

    def test_malformed_json_falls_back(self, tmp_path, capsys):
        ts_file = tmp_path / "slide_timestamps.json"
        ts_file.write_text("not valid json {{")
        slide = tmp_path / "slide_0001.png"
        slide.write_text("img")
        data = SpeechData(slide_images=[slide])
        _load_slide_timestamps(data, ts_file)
        out = capsys.readouterr().out
        assert "Warning" in out
        assert len(data.slide_timestamps) == 1
        assert data.slide_timestamps[0]["timestamp"] == 0.0
        assert data.slide_timestamps[0]["slide_number"] == 1

    def test_missing_file_falls_back(self, tmp_path, capsys):
        ts_file = tmp_path / "nonexistent.json"
        slide = tmp_path / "slide_0001.png"
        slide.write_text("img")
        data = SpeechData(slide_images=[slide])
        _load_slide_timestamps(data, ts_file)
        out = capsys.readouterr().out
        assert "Warning" in out
        assert len(data.slide_timestamps) == 1
        assert data.slide_timestamps[0]["filename"] == "slide_0001.png"

    def test_fallback_preserves_filenames(self, tmp_path):
        ts_file = tmp_path / "bad.json"
        ts_file.write_text("{{{")
        slides = [tmp_path / f"slide_{i:04d}.png" for i in range(1, 4)]
        for s in slides:
            s.write_text("img")
        data = SpeechData(slide_images=slides)
        _load_slide_timestamps(data, ts_file)
        assert [t["filename"] for t in data.slide_timestamps] == [
            "slide_0001.png", "slide_0002.png", "slide_0003.png"
        ]


# ---------------------------------------------------------------------------
# create_basic_slides_json
# ---------------------------------------------------------------------------

class TestCreateBasicSlidesJson:
    def test_creates_json_with_correct_structure(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        slide = tmp_path / "slide_0001.png"
        slide.write_text("img")
        data = SpeechData(slide_images=[slide])
        data.title = "Talk"
        create_basic_slides_json(config, data)
        json_path = tmp_path / "slides_basic.json"
        assert json_path.exists()
        content = json.loads(json_path.read_text())
        assert content["slide_count"] == 1
        assert content["slides"][0]["slide_number"] == 1
        assert content["slides"][0]["type"] == "unknown"
        assert "note" in content

    def test_skip_existing_when_fresh(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=True)
        slide = tmp_path / "slide_0001.png"
        slide.write_text("img")
        data = SpeechData(slide_images=[slide])
        # Create JSON newer than slide
        time.sleep(0.05)
        json_path = tmp_path / "slides_basic.json"
        json_path.write_text('{"old": true}')
        create_basic_slides_json(config, data)
        # Should not have overwritten
        assert json.loads(json_path.read_text()) == {"old": True}

    def test_dry_run_skips(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, dry_run=True,
                              skip_existing=False)
        slide = tmp_path / "slide_0001.png"
        slide.write_text("img")
        data = SpeechData(slide_images=[slide])
        create_basic_slides_json(config, data)
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert not (tmp_path / "slides_basic.json").exists()

    def test_sets_data_paths(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        slides = [tmp_path / f"slide_{i:04d}.png" for i in range(1, 3)]
        for s in slides:
            s.write_text("img")
        data = SpeechData(slide_images=slides)
        data.title = "Talk"
        create_basic_slides_json(config, data)
        assert data.slides_json_path == tmp_path / "slides_basic.json"
        assert len(data.slide_metadata) == 2


# ---------------------------------------------------------------------------
# extract_slides — ffmpeg execution path
# ---------------------------------------------------------------------------

class TestExtractSlidesExecution:
    @patch("transcribe_critic.slides.subprocess.run")
    def test_runs_ffmpeg_and_parses_timestamps(self, mock_run, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        video = tmp_path / "video.mp4"
        video.write_text("fake video")
        data = SpeechData(video_path=video)

        # Simulate ffmpeg creating slide files + stderr with timestamps
        slides_dir = tmp_path / "slides"
        slides_dir.mkdir(exist_ok=True)
        (slides_dir / "slide_0001.png").write_bytes(b"img1")
        (slides_dir / "slide_0002.png").write_bytes(b"img2")

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr=(
                "[Parsed_showinfo_1 @ 0x1234] n:   0 pts:      0 pts_time:5.5\n"
                "[Parsed_showinfo_1 @ 0x1234] n:   1 pts:  90000 pts_time:30.2\n"
            ),
        )

        extract_slides(config, data)
        assert len(data.slide_images) == 2
        assert len(data.slide_timestamps) == 2
        assert data.slide_timestamps[0]["timestamp"] == 5.5
        assert data.slide_timestamps[1]["timestamp"] == 30.2
        # Verify timestamps JSON was saved
        ts_file = tmp_path / "slide_timestamps.json"
        assert ts_file.exists()

    @patch("transcribe_critic.slides.subprocess.run")
    def test_handles_no_timestamps(self, mock_run, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        video = tmp_path / "video.mp4"
        video.write_text("fake video")
        data = SpeechData(video_path=video)

        slides_dir = tmp_path / "slides"
        slides_dir.mkdir(exist_ok=True)
        (slides_dir / "slide_0001.png").write_bytes(b"img1")

        # ffmpeg runs but no pts_time in stderr
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        extract_slides(config, data)
        assert len(data.slide_images) == 1
        assert data.slide_timestamps[0]["timestamp"] == 0.0

    @patch("transcribe_critic.slides.subprocess.run")
    def test_saves_timestamps_json(self, mock_run, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        video = tmp_path / "video.mp4"
        video.write_text("fake video")
        data = SpeechData(video_path=video)

        slides_dir = tmp_path / "slides"
        slides_dir.mkdir(exist_ok=True)
        (slides_dir / "slide_0001.png").write_bytes(b"img")

        mock_run.return_value = MagicMock(
            returncode=0, stdout="",
            stderr="[Parsed_showinfo_1] pts_time:12.3\n",
        )
        extract_slides(config, data)
        ts_data = json.loads((tmp_path / "slide_timestamps.json").read_text())
        assert ts_data[0]["timestamp"] == 12.3
        assert ts_data[0]["filename"] == "slide_0001.png"


# ---------------------------------------------------------------------------
# analyze_slides_with_vision — LLM execution path
# ---------------------------------------------------------------------------

class TestAnalyzeSlidesWithVisionExecution:
    @patch("transcribe_critic.slides.llm_call_with_retry")
    @patch("transcribe_critic.slides.create_llm_client")
    def test_analyzes_slides_with_mocked_llm(self, mock_client, mock_llm, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, analyze_slides=True,
                              skip_existing=False)
        slide = tmp_path / "slide_0001.png"
        slide.write_bytes(b"\x89PNG\r\n\x1a\n")
        data = SpeechData(slide_images=[slide])
        data.title = "Test Talk"

        mock_llm.return_value = MagicMock(
            content=[MagicMock(text='{"type": "title", "title": "Welcome", "description": "Title slide"}')]
        )

        analyze_slides_with_vision(config, data)
        assert data.slides_json_path is not None
        assert len(data.slide_metadata) == 1
        assert data.slide_metadata[0]["type"] == "title"
        assert data.slide_metadata[0]["title"] == "Welcome"
        # Verify JSON saved
        saved = json.loads(data.slides_json_path.read_text())
        assert saved["slide_count"] == 1

    @patch("transcribe_critic.slides.llm_call_with_retry")
    @patch("transcribe_critic.slides.create_llm_client")
    def test_handles_non_json_response(self, mock_client, mock_llm, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, analyze_slides=True,
                              skip_existing=False)
        slide = tmp_path / "slide_0001.png"
        slide.write_bytes(b"\x89PNG\r\n\x1a\n")
        data = SpeechData(slide_images=[slide])
        data.title = "Test"

        # LLM returns plain text, not JSON
        mock_llm.return_value = MagicMock(
            content=[MagicMock(text="This is a content slide with bullet points")]
        )

        analyze_slides_with_vision(config, data)
        assert data.slide_metadata[0]["type"] == "content"
        assert "bullet points" in data.slide_metadata[0]["description"]

    @patch("transcribe_critic.slides.llm_call_with_retry")
    @patch("transcribe_critic.slides.create_llm_client")
    def test_handles_invalid_json(self, mock_client, mock_llm, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, analyze_slides=True,
                              skip_existing=False)
        slide = tmp_path / "slide_0001.png"
        slide.write_bytes(b"\x89PNG\r\n\x1a\n")
        data = SpeechData(slide_images=[slide])
        data.title = "Test"

        # LLM returns something that looks like JSON but isn't valid
        mock_llm.return_value = MagicMock(
            content=[MagicMock(text='{invalid json here}')]
        )

        analyze_slides_with_vision(config, data)
        assert data.slide_metadata[0]["type"] == "unknown"
        assert "Could not parse" in data.slide_metadata[0]["description"]
