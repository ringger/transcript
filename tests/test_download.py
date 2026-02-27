"""Tests for download.py — VTT caption cleaning and media download logic."""

import json
import subprocess
from unittest.mock import MagicMock, patch

from transcribe_critic.shared import SpeechConfig, SpeechData

from transcribe_critic.download import clean_vtt_captions, download_media


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
# download_media
# ---------------------------------------------------------------------------

def _mock_run_command_for_download(cmd, desc, verbose=False):
    """Mock run_command that returns fake yt-dlp JSON for --dump-json."""
    if "--dump-json" in cmd:
        return MagicMock(stdout=json.dumps({
            "title": "Test Video",
            "id": "abc123",
            "channel": "Test Channel",
            "upload_date": "20240101",
            "duration": 300,
            "description": "A test video",
        }))
    return MagicMock(stdout="", stderr="")


class TestDownloadMedia:
    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_downloads_audio_video_captions(self, mock_run, tmp_path):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=False, skip_existing=False)
        data = SpeechData()
        download_media(config, data)
        assert data.title == "Test Video"
        assert data.audio_path == tmp_path / "audio.mp3"
        assert data.video_path == tmp_path / "video.mp4"
        # Should have called run_command for: info, audio, video, captions
        assert mock_run.call_count == 4

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_reuses_existing_audio(self, mock_run, tmp_path):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              skip_existing=True)
        # Pre-create audio file
        (tmp_path / "audio.mp3").write_text("fake audio")
        data = SpeechData()
        download_media(config, data)
        # Should NOT have called yt-dlp for audio download (only info, video, captions)
        audio_calls = [c for c in mock_run.call_args_list if "-x" in c[0][0]]
        assert len(audio_calls) == 0

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_no_slides_skips_video(self, mock_run, tmp_path, capsys):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=True, skip_existing=False)
        data = SpeechData()
        download_media(config, data)
        out = capsys.readouterr().out
        assert "Skipping video download" in out
        assert data.video_path is None

    @patch("transcribe_critic.download.run_command")
    def test_captions_failure_is_graceful(self, mock_run, tmp_path):
        def side_effect(cmd, desc, verbose=False):
            if "--dump-json" in cmd:
                return MagicMock(stdout=json.dumps({"title": "T", "id": "x"}))
            if "--write-auto-sub" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return MagicMock(stdout="", stderr="")
        mock_run.side_effect = side_effect
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              skip_existing=False)
        data = SpeechData()
        # Should not raise
        download_media(config, data)
        assert data.captions_path is None

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_dry_run_skips_downloads(self, mock_run, tmp_path, capsys):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              dry_run=True, skip_existing=False)
        data = SpeechData()
        download_media(config, data)
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        # Only the info fetch should run (not dry-run gated)
        assert mock_run.call_count == 1
        # Dry-run should not create any files
        assert not (tmp_path / "metadata.json").exists()

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_saves_metadata(self, mock_run, tmp_path):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              skip_existing=False)
        data = SpeechData()
        download_media(config, data)
        metadata_path = tmp_path / "metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["title"] == "Test Video"
        assert metadata["url"] == "https://example.com/v"
        assert metadata["video_id"] == "abc123"

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_external_transcript_in_metadata(self, mock_run, tmp_path):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              skip_existing=False,
                              external_transcript="https://example.com/transcript.txt")
        data = SpeechData()
        download_media(config, data)
        metadata = json.loads((tmp_path / "metadata.json").read_text())
        assert metadata["external_transcript"] == "https://example.com/transcript.txt"

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_podcast_skips_video_and_captions(self, mock_run, tmp_path, capsys):
        config = SpeechConfig(url="https://example.com/podcast/ep1", output_dir=tmp_path,
                              podcast=True, no_slides=True, skip_existing=False)
        data = SpeechData()
        download_media(config, data)
        out = capsys.readouterr().out
        assert "Skipping video download (--podcast)" in out
        assert "Skipping captions download (--podcast)" in out
        assert data.audio_path == tmp_path / "audio.mp3"
        assert data.video_path is None
        # info + audio only (no video, no captions)
        assert mock_run.call_count == 2
