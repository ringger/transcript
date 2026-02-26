"""Tests for eval/datasets.py — manifest filtering, save/load, ManifestEntry."""

import json

import pytest

from transcribe_critic.eval.datasets import (
    ManifestEntry,
    filter_manifest,
    load_manifest,
    save_manifest,
)


def _make_entries():
    """Create a list of ManifestEntry objects for testing."""
    return [
        ManifestEntry(
            file_id="001", audio_path="audio/001.wav", ref_stm_path="ref/001.stm",
            duration_secs=600, subset_tags=["all", "eval10"],
            metadata={"company_name": "Acme"},
        ),
        ManifestEntry(
            file_id="002", audio_path="audio/002.wav", ref_stm_path="ref/002.stm",
            duration_secs=1800, subset_tags=["all"],
            metadata={"company_name": "Beta"},
        ),
        ManifestEntry(
            file_id="003", audio_path="audio/003.wav", ref_stm_path="ref/003.stm",
            duration_secs=3600, subset_tags=["all", "eval10"],
            metadata={"company_name": "Gamma"},
        ),
    ]


class TestFilterManifest:
    def test_no_filters_returns_all(self):
        entries = _make_entries()
        result = filter_manifest(entries)
        assert len(result) == 3

    def test_filter_by_subset(self):
        entries = _make_entries()
        result = filter_manifest(entries, subset="eval10")
        assert len(result) == 2
        assert all("eval10" in e.subset_tags for e in result)

    def test_filter_by_subset_no_match(self):
        entries = _make_entries()
        result = filter_manifest(entries, subset="nonexistent")
        assert len(result) == 0

    def test_filter_by_max_files(self):
        entries = _make_entries()
        result = filter_manifest(entries, max_files=2)
        assert len(result) == 2
        assert result[0].file_id == "001"

    def test_filter_by_max_hours(self):
        entries = _make_entries()
        # 001=600s + 002=1800s = 2400s = 0.67h; 003=3600s would push to 1.67h
        result = filter_manifest(entries, max_hours=1.0)
        assert len(result) == 2
        assert result[-1].file_id == "002"

    def test_filter_by_max_hours_zero(self):
        entries = _make_entries()
        result = filter_manifest(entries, max_hours=0.0)
        assert len(result) == 0

    def test_filter_by_file_ids(self):
        entries = _make_entries()
        result = filter_manifest(entries, file_ids=["002", "003"])
        assert len(result) == 2
        assert {e.file_id for e in result} == {"002", "003"}

    def test_filter_by_file_ids_nonexistent(self):
        entries = _make_entries()
        result = filter_manifest(entries, file_ids=["999"])
        assert len(result) == 0

    def test_filter_combined(self):
        entries = _make_entries()
        result = filter_manifest(entries, subset="eval10", max_files=1)
        assert len(result) == 1
        assert result[0].file_id == "001"

    def test_empty_entries(self):
        result = filter_manifest([])
        assert result == []


class TestSaveLoadManifest:
    def test_roundtrip(self, tmp_path):
        entries = _make_entries()
        path = tmp_path / "manifest.json"
        save_manifest(entries, path)
        loaded = load_manifest(path)
        assert len(loaded) == len(entries)
        assert loaded[0].file_id == "001"
        assert loaded[0].duration_secs == 600
        assert loaded[0].subset_tags == ["all", "eval10"]

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "subdir" / "manifest.json"
        save_manifest(_make_entries(), path)
        assert path.exists()

    def test_manifest_json_structure(self, tmp_path):
        path = tmp_path / "manifest.json"
        save_manifest(_make_entries()[:1], path)
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert len(data["entries"]) == 1
        assert data["entries"][0]["file_id"] == "001"
