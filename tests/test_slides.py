"""Tests for slides.py — slide extraction and vision analysis."""

import json
import time

from transcript_critic.shared import SpeechConfig, SpeechData

from transcript_critic.slides import (
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
