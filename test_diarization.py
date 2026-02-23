"""Tests for diarization.py — speaker diarization, assignment, and identification."""

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shared import SpeechConfig, SpeechData

from diarization import (
    _assign_speakers_to_words,
    _find_speaker_at_time,
    _format_diarized_transcript,
    _format_timestamp,
    _get_intro_text,
    _identify_speakers,
    _apply_speaker_names,
    _make_progress_hook,
    _run_pyannote_steps,
    diarize_audio,
)


# ---------------------------------------------------------------------------
# _format_timestamp
# ---------------------------------------------------------------------------

class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0) == "0:00:00"

    def test_seconds_only(self):
        assert _format_timestamp(45) == "0:00:45"

    def test_minutes_and_seconds(self):
        assert _format_timestamp(125) == "0:02:05"

    def test_hours(self):
        assert _format_timestamp(3661) == "1:01:01"


# ---------------------------------------------------------------------------
# _find_speaker_at_time
# ---------------------------------------------------------------------------

class TestFindSpeakerAtTime:
    def test_exact_match(self):
        segments = [
            {"start": 0, "end": 10, "speaker": "A"},
            {"start": 10, "end": 20, "speaker": "B"},
        ]
        assert _find_speaker_at_time(5, segments) == "A"
        assert _find_speaker_at_time(15, segments) == "B"

    def test_boundary(self):
        segments = [
            {"start": 0, "end": 10, "speaker": "A"},
            {"start": 10, "end": 20, "speaker": "B"},
        ]
        assert _find_speaker_at_time(10, segments) in ("A", "B")

    def test_gap_finds_nearest(self):
        segments = [
            {"start": 0, "end": 5, "speaker": "A"},
            {"start": 15, "end": 20, "speaker": "B"},
        ]
        # 8 is closer to A's end (5) than B's start (15)
        assert _find_speaker_at_time(8, segments) == "A"
        # 12 is closer to B's start (15) than A's end (5)
        assert _find_speaker_at_time(12, segments) == "B"

    def test_empty_segments(self):
        assert _find_speaker_at_time(5, []) == "UNKNOWN"


# ---------------------------------------------------------------------------
# _assign_speakers_to_words
# ---------------------------------------------------------------------------

class TestAssignSpeakersToWords:
    def test_word_level_assignment(self):
        data = SpeechData()
        data.transcript_segments = [
            {
                "start": 0, "end": 5, "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0, "end": 2},
                    {"word": "world", "start": 2, "end": 5},
                ],
            },
        ]
        speaker_segments = [
            {"start": 0, "end": 10, "speaker": "SPEAKER_00"},
        ]
        _assign_speakers_to_words(data, speaker_segments)
        assert data.transcript_segments[0]["speaker"] == "SPEAKER_00"
        assert data.transcript_segments[0]["words"][0]["speaker"] == "SPEAKER_00"

    def test_majority_vote_for_segment(self):
        data = SpeechData()
        data.transcript_segments = [
            {
                "start": 0, "end": 10, "text": "one two three",
                "words": [
                    {"word": "one", "start": 0, "end": 3},
                    {"word": "two", "start": 3, "end": 6},
                    {"word": "three", "start": 6, "end": 10},
                ],
            },
        ]
        speaker_segments = [
            {"start": 0, "end": 4, "speaker": "A"},
            {"start": 4, "end": 20, "speaker": "B"},
        ]
        _assign_speakers_to_words(data, speaker_segments)
        # "one" midpoint=1.5 -> A; "two" midpoint=4.5 -> B; "three" midpoint=8 -> B
        assert data.transcript_segments[0]["speaker"] == "B"

    def test_fallback_segment_midpoint(self):
        """Without word-level data, use segment midpoint."""
        data = SpeechData()
        data.transcript_segments = [
            {"start": 0, "end": 5, "text": "Hello"},
        ]
        speaker_segments = [
            {"start": 0, "end": 10, "speaker": "SPEAKER_01"},
        ]
        _assign_speakers_to_words(data, speaker_segments)
        assert data.transcript_segments[0]["speaker"] == "SPEAKER_01"

    def test_empty_speaker_segments(self):
        data = SpeechData()
        data.transcript_segments = [
            {"start": 0, "end": 5, "text": "Hello"},
        ]
        _assign_speakers_to_words(data, [])
        assert "speaker" not in data.transcript_segments[0]


# ---------------------------------------------------------------------------
# _apply_speaker_names
# ---------------------------------------------------------------------------

class TestApplySpeakerNames:
    def test_renames_segments(self):
        data = SpeechData()
        data.transcript_segments = [
            {"start": 0, "end": 5, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 5, "end": 10, "text": "Hi", "speaker": "SPEAKER_01"},
        ]
        _apply_speaker_names(data, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
        assert data.transcript_segments[0]["speaker"] == "Alice"
        assert data.transcript_segments[1]["speaker"] == "Bob"

    def test_renames_word_speakers(self):
        data = SpeechData()
        data.transcript_segments = [
            {
                "start": 0, "end": 5, "text": "Hi",
                "speaker": "SPEAKER_00",
                "words": [{"word": "Hi", "start": 0, "end": 1, "speaker": "SPEAKER_00"}],
            },
        ]
        _apply_speaker_names(data, {"SPEAKER_00": "Alice"})
        assert data.transcript_segments[0]["words"][0]["speaker"] == "Alice"

    def test_partial_mapping(self):
        data = SpeechData()
        data.transcript_segments = [
            {"start": 0, "end": 5, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 5, "end": 10, "text": "Hi", "speaker": "SPEAKER_01"},
        ]
        _apply_speaker_names(data, {"SPEAKER_00": "Alice"})
        assert data.transcript_segments[0]["speaker"] == "Alice"
        assert data.transcript_segments[1]["speaker"] == "SPEAKER_01"


# ---------------------------------------------------------------------------
# _identify_speakers
# ---------------------------------------------------------------------------

class TestIdentifySpeakers:
    def test_manual_speaker_names(self):
        config = SpeechConfig(
            url="test", output_dir="/tmp",
            speaker_names=["Alice", "Bob"],
        )
        data = SpeechData()
        data.transcript_segments = [
            {"start": 0, "end": 5, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 5, "end": 10, "text": "Hi", "speaker": "SPEAKER_01"},
        ]
        _identify_speakers(config, data)
        assert data.transcript_segments[0]["speaker"] == "Alice"
        assert data.transcript_segments[1]["speaker"] == "Bob"

    def test_no_llm_skips_identification(self):
        config = SpeechConfig(url="test", output_dir="/tmp", no_llm=True)
        data = SpeechData()
        data.transcript_segments = [
            {"start": 0, "end": 5, "text": "Hello", "speaker": "SPEAKER_00"},
        ]
        _identify_speakers(config, data)
        assert data.transcript_segments[0]["speaker"] == "SPEAKER_00"

    @patch("diarization.llm_call_with_retry")
    @patch("diarization.create_llm_client")
    def test_llm_identifies_speakers(self, mock_client, mock_llm):
        mock_llm.return_value = MagicMock(
            content=[MagicMock(text='{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}')],
        )
        config = SpeechConfig(url="test", output_dir="/tmp", local=False)
        data = SpeechData()
        data.transcript_segments = [
            {"start": 0, "end": 5, "text": "I'm Alice and welcome", "speaker": "SPEAKER_00"},
            {"start": 5, "end": 10, "text": "Thanks Alice", "speaker": "SPEAKER_01"},
        ]
        _identify_speakers(config, data)
        assert data.transcript_segments[0]["speaker"] == "Alice"
        assert data.transcript_segments[1]["speaker"] == "Bob"

    def test_no_speakers_is_noop(self):
        config = SpeechConfig(url="test", output_dir="/tmp")
        data = SpeechData()
        data.transcript_segments = [
            {"start": 0, "end": 5, "text": "Hello"},
        ]
        _identify_speakers(config, data)  # Should not raise


# ---------------------------------------------------------------------------
# _get_intro_text
# ---------------------------------------------------------------------------

class TestGetIntroText:
    def test_returns_first_500_words(self):
        data = SpeechData()
        data.transcript_segments = [
            {"start": 0, "end": 5, "text": "word " * 100, "speaker": "A"},
            {"start": 5, "end": 10, "text": "word " * 100, "speaker": "B"},
            {"start": 10, "end": 15, "text": "word " * 100, "speaker": "A"},
            {"start": 15, "end": 20, "text": "word " * 100, "speaker": "B"},
            {"start": 20, "end": 25, "text": "word " * 100, "speaker": "A"},
            {"start": 25, "end": 30, "text": "word " * 200, "speaker": "B"},
        ]
        text = _get_intro_text(data)
        assert "A:" in text
        assert "B:" in text
        # Should stop after ~500 words
        lines = text.strip().split("\n")
        assert len(lines) <= 6


# ---------------------------------------------------------------------------
# _format_diarized_transcript
# ---------------------------------------------------------------------------

class TestFormatDiarizedTranscript:
    def test_basic_formatting(self):
        data = SpeechData()
        data.transcript_segments = [
            {"start": 0, "end": 5, "text": "Hello there.", "speaker": "Alice"},
            {"start": 5, "end": 10, "text": "Hi Alice.", "speaker": "Bob"},
        ]
        result = _format_diarized_transcript(data)
        assert "[0:00:00] Alice: Hello there." in result
        assert "[0:00:05] Bob: Hi Alice." in result

    def test_merges_consecutive_same_speaker(self):
        data = SpeechData()
        data.transcript_segments = [
            {"start": 0, "end": 3, "text": "Hello.", "speaker": "Alice"},
            {"start": 3, "end": 6, "text": "How are you?", "speaker": "Alice"},
            {"start": 6, "end": 10, "text": "I'm fine.", "speaker": "Bob"},
        ]
        result = _format_diarized_transcript(data)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 2
        assert "Hello. How are you?" in lines[0]

    def test_matches_bracketed_pattern(self):
        """Output should match merge.py's BRACKETED_PATTERN."""
        import re
        BRACKETED_PATTERN = re.compile(r'^\[(\d{1,2}:\d{2}:\d{2})\]\s+(\w[\w\s]+?):')
        data = SpeechData()
        data.transcript_segments = [
            {"start": 61, "end": 70, "text": "Welcome.", "speaker": "Host"},
        ]
        result = _format_diarized_transcript(data)
        first_line = result.strip().split("\n")[0]
        assert BRACKETED_PATTERN.match(first_line), f"'{first_line}' doesn't match BRACKETED_PATTERN"

    def test_empty_segments(self):
        data = SpeechData()
        data.transcript_segments = []
        result = _format_diarized_transcript(data)
        assert result == ""


# ---------------------------------------------------------------------------
# diarize_audio (integration)
# ---------------------------------------------------------------------------

class TestDiarizeAudio:
    def test_skips_when_not_enabled(self, tmp_path):
        config = SpeechConfig(url="test", output_dir=tmp_path, diarize=False)
        data = SpeechData()
        diarize_audio(config, data)
        assert data.diarization_path is None

    def test_dry_run(self, tmp_path):
        config = SpeechConfig(url="test", output_dir=tmp_path, diarize=True, dry_run=True)
        data = SpeechData()
        data.audio_path = tmp_path / "audio.mp3"
        data.audio_path.touch()
        diarize_audio(config, data)
        assert data.diarization_path is None

    def test_warns_no_audio(self, tmp_path):
        config = SpeechConfig(
            url="test", output_dir=tmp_path, diarize=True, skip_existing=False,
        )
        data = SpeechData()
        data.audio_path = tmp_path / "nonexistent.mp3"
        diarize_audio(config, data)
        assert data.diarization_path is None

    def test_warns_no_segments(self, tmp_path):
        config = SpeechConfig(
            url="test", output_dir=tmp_path, diarize=True, skip_existing=False,
        )
        data = SpeechData()
        data.audio_path = tmp_path / "audio.mp3"
        data.audio_path.touch()
        data.transcript_segments = []
        diarize_audio(config, data)
        assert data.diarization_path is None

    @patch("diarization._identify_speakers")
    @patch("diarization._run_pyannote")
    @patch.dict("sys.modules", {"pyannote": MagicMock(), "pyannote.audio": MagicMock()})
    def test_full_pipeline(self, mock_pyannote, mock_identify, tmp_path):
        mock_pyannote.return_value = [
            {"start": 0, "end": 5, "speaker": "SPEAKER_00"},
            {"start": 5, "end": 10, "speaker": "SPEAKER_01"},
        ]

        config = SpeechConfig(
            url="test", output_dir=tmp_path, diarize=True, skip_existing=False,
        )
        data = SpeechData()
        data.audio_path = tmp_path / "audio.mp3"
        data.audio_path.touch()
        data.transcript_segments = [
            {"start": 0, "end": 5, "text": "Hello there."},
            {"start": 5, "end": 10, "text": "Hi back."},
        ]

        diarize_audio(config, data)

        assert data.diarization_path is not None
        assert data.diarization_path.exists()
        content = data.diarization_path.read_text()
        assert "SPEAKER_00" in content
        assert "SPEAKER_01" in content

    def test_reuses_cached(self, tmp_path):
        import time
        audio = tmp_path / "audio.mp3"
        audio.touch()
        time.sleep(0.05)
        diarized = tmp_path / "diarized.txt"
        diarized.write_text("[0:00:00] Alice: Hello")

        config = SpeechConfig(url="test", output_dir=tmp_path, diarize=True)
        data = SpeechData()
        data.audio_path = audio
        diarize_audio(config, data)
        assert data.diarization_path == diarized


# ---------------------------------------------------------------------------
# _make_progress_hook
# ---------------------------------------------------------------------------

class TestMakeProgressHook:
    def test_prints_progress(self, capsys):
        hook = _make_progress_hook()
        hook("embeddings", None, completed=5, total=10)
        captured = capsys.readouterr()
        assert "50%" in captured.out
        assert "5/10" in captured.out

    def test_prints_done_at_completion(self, capsys):
        hook = _make_progress_hook()
        hook("embeddings", None, completed=10, total=10)
        captured = capsys.readouterr()
        assert "done" in captured.out

    def test_no_crash_without_progress(self):
        hook = _make_progress_hook()
        # Called without completed/total — should not crash
        hook("segmentation", "some_artifact")


# ---------------------------------------------------------------------------
# _run_pyannote_steps (checkpoint logic)
# ---------------------------------------------------------------------------

def _make_mock_pipeline():
    """Create a mock pyannote pipeline with step methods for testing."""
    pipeline = MagicMock()

    # Segmentation model attributes
    pipeline._segmentation.model.specifications.powerset = True
    pipeline._segmentation.model.receptive_field = MagicMock()

    # Segmentation returns SlidingWindowFeature-like object
    seg_data = np.random.rand(10, 50, 3).astype(np.float32)
    mock_swf = MagicMock()
    mock_swf.data = seg_data
    mock_swf.sliding_window.start = 0.0
    mock_swf.sliding_window.duration = 5.0
    mock_swf.sliding_window.step = 0.5
    pipeline.get_segmentations.return_value = mock_swf

    # Speaker count returns SlidingWindowFeature with non-zero data
    count_swf = MagicMock()
    count_swf.data = np.array([[1], [2], [1]], dtype=np.uint8)
    pipeline.speaker_count.return_value = count_swf

    # Embeddings
    embeddings = np.random.rand(10, 3, 192).astype(np.float32)
    pipeline.get_embeddings.return_value = embeddings
    pipeline.embedding_exclude_overlap = False

    # Clustering
    hard_clusters = np.array([[0, 1, -2]] * 10, dtype=np.int32)
    centroids = np.random.rand(2, 192).astype(np.float32)
    pipeline.clustering.return_value = (hard_clusters, None, centroids)

    # Reconstruction
    mock_annotation = MagicMock()
    mock_annotation.labels.return_value = [0, 1]
    mock_annotation.rename_labels.return_value = mock_annotation
    pipeline.to_annotation.return_value = mock_annotation
    pipeline.reconstruct.return_value = MagicMock()
    pipeline.segmentation.min_duration_off = 0.0
    pipeline.classes.return_value = iter(["SPEAKER_00", "SPEAKER_01"])

    return pipeline


class TestRunPyannoteSteps:
    @patch.dict("sys.modules", {
        "pyannote": MagicMock(),
        "pyannote.core": MagicMock(),
        "pyannote.audio": MagicMock(),
        "pyannote.audio.utils": MagicMock(),
        "pyannote.audio.utils.signal": MagicMock(),
    })
    def test_caches_segmentation(self, tmp_path):
        """Step 1 should save segmentation.npy on first run."""
        pipeline = _make_mock_pipeline()
        config = SpeechConfig(url="test", output_dir=tmp_path)
        data = SpeechData()
        data.audio_path = tmp_path / "audio.mp3"
        data.audio_path.touch()

        _run_pyannote_steps(config, data, pipeline)

        seg_npy = tmp_path / "diarization_segmentation.npy"
        seg_meta = tmp_path / "diarization_segmentation_meta.json"
        emb_npy = tmp_path / "diarization_embeddings.npy"
        assert seg_npy.exists()
        assert seg_meta.exists()
        assert emb_npy.exists()
        # Verify segmentation metadata
        with open(seg_meta) as f:
            meta = json.load(f)
        assert "start" in meta and "duration" in meta and "step" in meta

    @patch.dict("sys.modules", {
        "pyannote": MagicMock(),
        "pyannote.core": MagicMock(),
        "pyannote.audio": MagicMock(),
        "pyannote.audio.utils": MagicMock(),
        "pyannote.audio.utils.signal": MagicMock(),
    })
    def test_reuses_cached_segmentation(self, tmp_path):
        """Step 1 should skip if segmentation.npy is newer than audio."""
        import time

        pipeline = _make_mock_pipeline()
        config = SpeechConfig(url="test", output_dir=tmp_path)
        data = SpeechData()
        data.audio_path = tmp_path / "audio.mp3"
        data.audio_path.touch()

        time.sleep(0.05)

        # Pre-create cached segmentation (newer than audio)
        seg_data = np.random.rand(10, 50, 3).astype(np.float32)
        seg_npy = tmp_path / "diarization_segmentation.npy"
        np.save(seg_npy, seg_data)
        seg_meta = tmp_path / "diarization_segmentation_meta.json"
        with open(seg_meta, 'w') as f:
            json.dump({"start": 0.0, "duration": 5.0, "step": 0.5}, f)

        _run_pyannote_steps(config, data, pipeline)

        # get_segmentations should NOT have been called
        pipeline.get_segmentations.assert_not_called()

    @patch.dict("sys.modules", {
        "pyannote": MagicMock(),
        "pyannote.core": MagicMock(),
        "pyannote.audio": MagicMock(),
        "pyannote.audio.utils": MagicMock(),
        "pyannote.audio.utils.signal": MagicMock(),
    })
    def test_reuses_cached_embeddings(self, tmp_path):
        """Step 4 should skip if embeddings.npy is newer than segmentation.npy."""
        import time

        pipeline = _make_mock_pipeline()
        config = SpeechConfig(url="test", output_dir=tmp_path)
        data = SpeechData()
        data.audio_path = tmp_path / "audio.mp3"
        data.audio_path.touch()

        time.sleep(0.05)

        # Pre-create cached segmentation (newer than audio)
        seg_npy = tmp_path / "diarization_segmentation.npy"
        np.save(seg_npy, np.random.rand(10, 50, 3).astype(np.float32))
        seg_meta = tmp_path / "diarization_segmentation_meta.json"
        with open(seg_meta, 'w') as f:
            json.dump({"start": 0.0, "duration": 5.0, "step": 0.5}, f)

        time.sleep(0.05)

        # Pre-create cached embeddings (newer than segmentation)
        emb_npy = tmp_path / "diarization_embeddings.npy"
        np.save(emb_npy, np.random.rand(10, 3, 192).astype(np.float32))

        _run_pyannote_steps(config, data, pipeline)

        # get_embeddings should NOT have been called
        pipeline.get_embeddings.assert_not_called()

    @patch.dict("sys.modules", {
        "pyannote": MagicMock(),
        "pyannote.core": MagicMock(),
        "pyannote.audio": MagicMock(),
        "pyannote.audio.utils": MagicMock(),
        "pyannote.audio.utils.signal": MagicMock(),
    })
    def test_stale_cache_reruns(self, tmp_path):
        """If audio is newer than cache, re-run the step."""
        import time

        pipeline = _make_mock_pipeline()
        config = SpeechConfig(url="test", output_dir=tmp_path)
        data = SpeechData()

        # Pre-create old cache
        seg_npy = tmp_path / "diarization_segmentation.npy"
        np.save(seg_npy, np.random.rand(10, 50, 3).astype(np.float32))
        seg_meta = tmp_path / "diarization_segmentation_meta.json"
        with open(seg_meta, 'w') as f:
            json.dump({"start": 0.0, "duration": 5.0, "step": 0.5}, f)

        time.sleep(0.05)

        # Create audio AFTER cache (makes cache stale)
        data.audio_path = tmp_path / "audio.mp3"
        data.audio_path.touch()

        _run_pyannote_steps(config, data, pipeline)

        # Segmentation should have been re-run since cache is stale
        pipeline.get_segmentations.assert_called_once()


