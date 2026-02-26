"""Tests for eval/run.py — pipeline orchestration and output seeding."""

import json

import pytest

from transcribe_critic.eval.run import _is_complete, _seed_output_dir, _fmt_duration
from transcribe_critic.eval.datasets import ManifestEntry


def _make_entry(**overrides):
    defaults = dict(
        file_id="001", audio_path="audio/001.wav", ref_stm_path="ref/001.stm",
        duration_secs=600, subset_tags=["all"], metadata={"company_name": "Acme"},
    )
    defaults.update(overrides)
    return ManifestEntry(**defaults)


class TestIsComplete:
    def test_true_when_transcript_exists(self, tmp_path):
        (tmp_path / "transcript.md").write_text("# done")
        assert _is_complete(tmp_path) is True

    def test_false_when_missing(self, tmp_path):
        assert _is_complete(tmp_path) is False


class TestSeedOutputDir:
    def test_creates_symlink_and_metadata(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        audio = tmp_path / "audio.wav"
        audio.write_text("fake audio")
        entry = _make_entry()

        _seed_output_dir(output_dir, audio, entry)

        audio_link = output_dir / "audio.mp3"
        assert audio_link.is_symlink()
        assert audio_link.resolve() == audio.resolve()

        meta = json.loads((output_dir / "metadata.json").read_text())
        assert meta["video_id"] == "001"
        assert meta["duration_seconds"] == 600

    def test_skips_if_already_exists(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        audio = tmp_path / "audio.wav"
        audio.write_text("fake audio")

        # Pre-create both files
        (output_dir / "audio.mp3").write_text("existing")
        (output_dir / "metadata.json").write_text('{"pre": true}')

        _seed_output_dir(output_dir, audio, _make_entry())

        # Should not overwrite
        assert (output_dir / "audio.mp3").read_text() == "existing"
        assert json.loads((output_dir / "metadata.json").read_text()) == {"pre": True}

    def test_uses_episode_title_metadata(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        audio = tmp_path / "audio.wav"
        audio.write_text("fake audio")
        entry = _make_entry(metadata={"episode_title": "My Episode"})

        _seed_output_dir(output_dir, audio, entry)

        meta = json.loads((output_dir / "metadata.json").read_text())
        assert meta["title"] == "My Episode"


class TestFmtDuration:
    def test_minutes(self):
        assert _fmt_duration(125) == "2:05"

    def test_hours(self):
        assert _fmt_duration(3661) == "1:01:01"
