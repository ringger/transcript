"""Tests for eval format converters."""

import json
from pathlib import Path

import pytest

from transcribe_critic.eval.convert import (
    nlp_to_stm,
    diarized_txt_to_stm,
    structured_merged_to_stm,
    plain_text_to_stm,
    diarization_json_to_rttm,
    detect_hypothesis_format,
    hypothesis_to_stm,
)


# --- Fixtures ---

NLP_CONTENT = """\
token|speaker|ts|endTs|punctuation|case|tags|wer_tags
Good|0||||UC|[]|[]
morning|0|||,|LC|[]|['4']
gentlemen|0|||.|LC|[]|[]
Welcome|1||||UC|[]|[]
to|1||||LC|[]|[]
our|1||||LC|[]|[]
call|1|||.|LC|[]|[]
Thank|0||||UC|[]|[]
you|0|||.|LC|[]|[]
"""

DIARIZED_CONTENT = """\
[0:00:00] Kevin Roose: Well, Casey, have you ever been defamed on the internet?

[0:00:06] Casey Newton: Oh, probably a few times, but I try to just let it slide off my mind.

[0:00:15] Kevin Roose: That is a very healthy approach.
"""

STRUCTURED_MERGED_CONTENT = """\
**Peter Steinberger** (00:00:00)

I watched my agent happily click the button.

**Lex Fridman** (00:00:31)

You prefer agentic engineering? That is interesting.

**Peter Steinberger** (00:01:05)

Yes, absolutely. It changes everything.
"""

DIARIZATION_JSON = [
    {"start": 0.031, "end": 5.414, "speaker": "SPEAKER_00"},
    {"start": 6.477, "end": 14.821, "speaker": "SPEAKER_01"},
    {"start": 15.100, "end": 20.500, "speaker": "SPEAKER_00"},
]


# --- Tests for nlp_to_stm ---

class TestNlpToStm:
    def test_basic_conversion(self, tmp_path):
        nlp_file = tmp_path / "test.nlp"
        nlp_file.write_text(NLP_CONTENT)

        result = nlp_to_stm(nlp_file, "test_file")
        lines = result.strip().split("\n")

        assert len(lines) == 3  # 3 speaker segments: 0, 1, 0

        # First segment: speaker_0, "Good morning, gentlemen."
        parts = lines[0].split(None, 5)
        assert parts[0] == "test_file"
        assert parts[1] == "1"
        assert parts[2] == "speaker_0"
        assert "Good" in parts[5]
        assert "morning," in parts[5]
        assert "gentlemen." in parts[5]

    def test_speaker_changes(self, tmp_path):
        nlp_file = tmp_path / "test.nlp"
        nlp_file.write_text(NLP_CONTENT)

        result = nlp_to_stm(nlp_file, "test_file")
        lines = result.strip().split("\n")

        speakers = [line.split(None, 5)[2] for line in lines]
        assert speakers == ["speaker_0", "speaker_1", "speaker_0"]

    def test_casing(self, tmp_path):
        nlp_content = "token|speaker|ts|endTs|punctuation|case|tags|wer_tags\n"
        nlp_content += "hello|0||||UC|[]|[]\n"    # UC = capitalize first
        nlp_content += "world|0||||LC|[]|[]\n"    # LC = lowercase
        nlp_content += "IBM|0||||CA|[]|[]\n"      # CA = all caps
        nlp_content += "iPhone|0||||MC|[]|[]\n"   # MC = keep as-is

        nlp_file = tmp_path / "test.nlp"
        nlp_file.write_text(nlp_content)

        result = nlp_to_stm(nlp_file, "test")
        text = result.split(None, 5)[5]
        assert "Hello" in text
        assert "world" in text
        assert "IBM" in text
        assert "iPhone" in text

    def test_punctuation(self, tmp_path):
        nlp_content = "token|speaker|ts|endTs|punctuation|case|tags|wer_tags\n"
        nlp_content += "yes|0|||,|LC|[]|[]\n"
        nlp_content += "indeed|0|||.|LC|[]|[]\n"

        nlp_file = tmp_path / "test.nlp"
        nlp_file.write_text(nlp_content)

        result = nlp_to_stm(nlp_file, "test")
        text = result.split(None, 5)[5]
        assert text == "yes, indeed."

    def test_empty_file(self, tmp_path):
        nlp_file = tmp_path / "test.nlp"
        nlp_file.write_text("")
        assert nlp_to_stm(nlp_file, "test") == ""

    def test_header_only(self, tmp_path):
        nlp_file = tmp_path / "test.nlp"
        nlp_file.write_text("token|speaker|ts|endTs|punctuation|case|tags|wer_tags\n")
        assert nlp_to_stm(nlp_file, "test") == ""


# --- Tests for diarized_txt_to_stm ---

class TestDiarizedTxtToStm:
    def test_basic_conversion(self, tmp_path):
        path = tmp_path / "diarized.txt"
        path.write_text(DIARIZED_CONTENT)

        result = diarized_txt_to_stm(path, "test")
        lines = result.strip().split("\n")

        assert len(lines) == 3

    def test_speaker_names(self, tmp_path):
        path = tmp_path / "diarized.txt"
        path.write_text(DIARIZED_CONTENT)

        result = diarized_txt_to_stm(path, "test")
        lines = result.strip().split("\n")

        speakers = [line.split(None, 5)[2] for line in lines]
        assert speakers == ["Kevin_Roose", "Casey_Newton", "Kevin_Roose"]

    def test_timestamps(self, tmp_path):
        path = tmp_path / "diarized.txt"
        path.write_text(DIARIZED_CONTENT)

        result = diarized_txt_to_stm(path, "test")
        lines = result.strip().split("\n")

        # First segment starts at 0, ends at 6 (start of next)
        parts = lines[0].split(None, 5)
        assert float(parts[3]) == 0.0
        assert float(parts[4]) == 6.0

    def test_empty_file(self, tmp_path):
        path = tmp_path / "diarized.txt"
        path.write_text("")
        assert diarized_txt_to_stm(path, "test") == ""


# --- Tests for structured_merged_to_stm ---

class TestStructuredMergedToStm:
    def test_basic_conversion(self, tmp_path):
        path = tmp_path / "transcript_merged.txt"
        path.write_text(STRUCTURED_MERGED_CONTENT)

        result = structured_merged_to_stm(path, "test")
        lines = result.strip().split("\n")

        assert len(lines) == 3

    def test_speakers_and_timestamps(self, tmp_path):
        path = tmp_path / "transcript_merged.txt"
        path.write_text(STRUCTURED_MERGED_CONTENT)

        result = structured_merged_to_stm(path, "test")
        lines = result.strip().split("\n")

        # First: Peter_Steinberger at 0s
        p = lines[0].split(None, 5)
        assert p[2] == "Peter_Steinberger"
        assert float(p[3]) == 0.0
        assert float(p[4]) == 31.0  # next segment starts at 31

        # Second: Lex_Fridman at 31s
        p = lines[1].split(None, 5)
        assert p[2] == "Lex_Fridman"
        assert float(p[3]) == 31.0
        assert float(p[4]) == 65.0  # next segment starts at 65

    def test_text_content(self, tmp_path):
        path = tmp_path / "transcript_merged.txt"
        path.write_text(STRUCTURED_MERGED_CONTENT)

        result = structured_merged_to_stm(path, "test")
        lines = result.strip().split("\n")

        text = lines[0].split(None, 5)[5]
        assert text == "I watched my agent happily click the button."

    def test_empty_file(self, tmp_path):
        path = tmp_path / "transcript_merged.txt"
        path.write_text("")
        assert structured_merged_to_stm(path, "test") == ""


# --- Tests for plain_text_to_stm ---

class TestPlainTextToStm:
    def test_basic(self, tmp_path):
        path = tmp_path / "transcript.txt"
        path.write_text("Hello world. This is a test.")

        result = plain_text_to_stm(path, "test")
        parts = result.split(None, 5)
        assert parts[0] == "test"
        assert parts[2] == "unknown"
        assert parts[5] == "Hello world. This is a test."

    def test_whitespace_normalization(self, tmp_path):
        path = tmp_path / "transcript.txt"
        path.write_text("Hello   world.\n\nThis is  a   test.")

        result = plain_text_to_stm(path, "test")
        text = result.split(None, 5)[5]
        assert text == "Hello world. This is a test."

    def test_custom_speaker(self, tmp_path):
        path = tmp_path / "transcript.txt"
        path.write_text("Hello")

        result = plain_text_to_stm(path, "test", speaker="narrator")
        assert "narrator" in result

    def test_empty(self, tmp_path):
        path = tmp_path / "transcript.txt"
        path.write_text("")
        assert plain_text_to_stm(path, "test") == ""


# --- Tests for diarization_json_to_rttm ---

class TestDiarizationJsonToRttm:
    def test_basic_conversion(self, tmp_path):
        path = tmp_path / "diarization.json"
        path.write_text(json.dumps(DIARIZATION_JSON))

        result = diarization_json_to_rttm(path, "test")
        lines = result.strip().split("\n")

        assert len(lines) == 3

    def test_rttm_format(self, tmp_path):
        path = tmp_path / "diarization.json"
        path.write_text(json.dumps(DIARIZATION_JSON))

        result = diarization_json_to_rttm(path, "test")
        line = result.strip().split("\n")[0]
        parts = line.split()

        assert parts[0] == "SPEAKER"
        assert parts[1] == "test"
        assert parts[2] == "1"
        assert float(parts[3]) == pytest.approx(0.031, abs=0.001)
        # Duration = 5.414 - 0.031 = 5.383
        assert float(parts[4]) == pytest.approx(5.383, abs=0.001)
        assert parts[5] == "<NA>"
        assert parts[6] == "<NA>"
        assert parts[7] == "SPEAKER_00"
        assert parts[8] == "<NA>"
        assert parts[9] == "<NA>"

    def test_speakers(self, tmp_path):
        path = tmp_path / "diarization.json"
        path.write_text(json.dumps(DIARIZATION_JSON))

        result = diarization_json_to_rttm(path, "test")
        lines = result.strip().split("\n")

        speakers = [line.split()[7] for line in lines]
        assert speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


# --- Tests for detect_hypothesis_format ---

class TestDetectHypothesisFormat:
    def test_diarized(self, tmp_path):
        (tmp_path / "diarized.txt").write_text("[0:00:00] Speaker: Hello")
        assert detect_hypothesis_format(tmp_path) == "diarized"

    def test_structured_merged(self, tmp_path):
        (tmp_path / "transcript_merged.txt").write_text(
            "**Speaker** (00:00:00)\n\nHello"
        )
        assert detect_hypothesis_format(tmp_path) == "structured_merged"

    def test_plain_merged(self, tmp_path):
        (tmp_path / "transcript_merged.txt").write_text("Hello world")
        assert detect_hypothesis_format(tmp_path) == "plain_merged"

    def test_whisper_merged(self, tmp_path):
        (tmp_path / "whisper_merged.txt").write_text("Hello world")
        assert detect_hypothesis_format(tmp_path) == "whisper"

    def test_whisper_model(self, tmp_path):
        (tmp_path / "whisper_medium.txt").write_text("Hello world")
        assert detect_hypothesis_format(tmp_path) == "whisper"

    def test_none(self, tmp_path):
        assert detect_hypothesis_format(tmp_path) == "none"

    def test_diarized_takes_priority(self, tmp_path):
        (tmp_path / "diarized.txt").write_text("[0:00:00] Speaker: Hello")
        (tmp_path / "transcript_merged.txt").write_text("Hello world")
        assert detect_hypothesis_format(tmp_path) == "diarized"


# --- Tests for hypothesis_to_stm ---

class TestHypothesisToStm:
    def test_auto_diarized(self, tmp_path):
        (tmp_path / "diarized.txt").write_text(DIARIZED_CONTENT)
        result = hypothesis_to_stm(tmp_path, "test")
        assert "Kevin_Roose" in result

    def test_auto_whisper(self, tmp_path):
        (tmp_path / "whisper_medium.txt").write_text("Hello world")
        result = hypothesis_to_stm(tmp_path, "test")
        assert "unknown" in result
        assert "Hello world" in result

    def test_explicit_merged(self, tmp_path):
        (tmp_path / "transcript_merged.txt").write_text(STRUCTURED_MERGED_CONTENT)
        (tmp_path / "diarized.txt").write_text(DIARIZED_CONTENT)
        # Explicit merged overrides auto-detection of diarized
        result = hypothesis_to_stm(tmp_path, "test", hypothesis="merged")
        assert "Peter_Steinberger" in result

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            hypothesis_to_stm(tmp_path, "test", hypothesis="merged")
