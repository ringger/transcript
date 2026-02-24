"""Tests for output.py — markdown generation and text formatting."""

from pathlib import Path

from transcribe_critic.shared import SpeechConfig, SpeechData

from transcribe_critic.output import (
    _format_paragraph,
    _generate_interleaved_markdown,
    _generate_sequential_markdown,
    _get_best_transcript_text,
    generate_markdown,
)


# ---------------------------------------------------------------------------
# _format_paragraph
# ---------------------------------------------------------------------------

class TestFormatParagraph:
    def test_short_text_no_splitting(self):
        result = _format_paragraph("One sentence. Two sentence.")
        assert result == "One sentence. Two sentence."

    def test_splits_at_three_sentences(self):
        text = "First. Second. Third. Fourth. Fifth. Sixth."
        result = _format_paragraph(text)
        parts = result.split("\n\n")
        assert len(parts) == 2
        assert "First. Second. Third." in parts[0]
        assert "Fourth. Fifth. Sixth." in parts[1]

    def test_preserves_sentence_boundaries(self):
        text = "Hello world. This is a test. Another one here. And a fourth."
        result = _format_paragraph(text)
        # Should not break mid-sentence
        for part in result.split("\n\n"):
            assert part.rstrip().endswith(".")

    def test_handles_question_and_exclamation(self):
        text = "What is this? I love it! Great. Next one? Yes! Done."
        result = _format_paragraph(text)
        parts = result.split("\n\n")
        assert len(parts) == 2

    def test_single_sentence(self):
        result = _format_paragraph("Just one sentence here.")
        assert result == "Just one sentence here."

    def test_empty_string(self):
        result = _format_paragraph("")
        assert result == ""


# ---------------------------------------------------------------------------
# _get_best_transcript_text
# ---------------------------------------------------------------------------

class TestGetBestTranscriptText:
    def test_prefers_merged(self, tmp_path):
        merged = tmp_path / "merged.txt"
        merged.write_text("merged content")
        whisper = tmp_path / "whisper.txt"
        whisper.write_text("whisper content")
        data = SpeechData(merged_transcript_path=merged, transcript_path=whisper)
        assert _get_best_transcript_text(data) == "merged content"

    def test_falls_back_to_whisper(self, tmp_path):
        whisper = tmp_path / "whisper.txt"
        whisper.write_text("whisper content")
        data = SpeechData(transcript_path=whisper)
        assert _get_best_transcript_text(data) == "whisper content"

    def test_returns_empty_when_nothing(self):
        data = SpeechData()
        assert _get_best_transcript_text(data) == ""

    def test_ignores_missing_merged(self, tmp_path):
        whisper = tmp_path / "whisper.txt"
        whisper.write_text("whisper content")
        data = SpeechData(
            merged_transcript_path=tmp_path / "nonexistent.txt",
            transcript_path=whisper,
        )
        assert _get_best_transcript_text(data) == "whisper content"


# ---------------------------------------------------------------------------
# _generate_sequential_markdown
# ---------------------------------------------------------------------------

class TestGenerateSequentialMarkdown:
    def test_basic_no_slides(self, tmp_path):
        transcript = tmp_path / "merged.txt"
        transcript.write_text("This is the transcript.")
        data = SpeechData(title="Test Video", merged_transcript_path=transcript)
        result = _generate_sequential_markdown(data)
        assert "# Test Video" in result
        assert "## Transcript" in result
        assert "This is the transcript." in result
        assert "merged transcript" in result  # source note

    def test_with_slides_gallery(self, tmp_path):
        transcript = tmp_path / "merged.txt"
        transcript.write_text("Transcript text.")
        slide1 = tmp_path / "slide_0001.png"
        slide2 = tmp_path / "slide_0002.png"
        slide1.write_text("img")
        slide2.write_text("img")
        data = SpeechData(
            title="Slides Talk",
            merged_transcript_path=transcript,
            slide_images=[slide1, slide2],
        )
        result = _generate_sequential_markdown(data)
        assert "## Slides" in result
        assert "### Slide 1" in result
        assert "### Slide 2" in result
        assert "slide_0001.png" in result
        assert "Title Slide" in result  # first slide is title slide

    def test_slide_metadata(self, tmp_path):
        transcript = tmp_path / "merged.txt"
        transcript.write_text("Text.")
        slide = tmp_path / "slide_0001.png"
        slide.write_text("img")
        data = SpeechData(
            title="Talk",
            merged_transcript_path=transcript,
            slide_images=[slide],
            slide_metadata=[{"title": "Introduction Slide"}],
        )
        result = _generate_sequential_markdown(data)
        assert "Introduction Slide" in result

    def test_whisper_source_note(self, tmp_path):
        transcript = tmp_path / "whisper.txt"
        transcript.write_text("Whisper text.")
        data = SpeechData(title="Talk", transcript_path=transcript)
        result = _generate_sequential_markdown(data)
        assert "Whisper transcript" in result


# ---------------------------------------------------------------------------
# _generate_interleaved_markdown
# ---------------------------------------------------------------------------

class TestGenerateInterleavedMarkdown:
    def test_text_only(self):
        data = SpeechData(
            title="Test",
            transcript_segments=[
                {"start": 0.0, "end": 2.0, "text": "Hello world."},
                {"start": 2.0, "end": 4.0, "text": "Second segment."},
            ],
        )
        result = _generate_interleaved_markdown(data)
        assert "# Test" in result
        assert "Hello world." in result
        assert "Second segment." in result

    def test_slides_interleaved(self):
        slide_img = Path("/tmp/slide_0001.png")
        data = SpeechData(
            title="Slides Talk",
            transcript_segments=[
                {"start": 0.0, "end": 5.0, "text": "Before the slide."},
                {"start": 10.0, "end": 15.0, "text": "After the slide."},
            ],
            slide_timestamps=[{"slide_number": 1, "timestamp": 6.0}],
            slide_images=[slide_img],
        )
        result = _generate_interleaved_markdown(data)
        lines = result.split("\n")
        # Find positions
        before_idx = next(i for i, l in enumerate(lines) if "Before the slide" in l)
        slide_idx = next(i for i, l in enumerate(lines) if "slide_0001.png" in l)
        after_idx = next(i for i, l in enumerate(lines) if "After the slide" in l)
        assert before_idx < slide_idx < after_idx

    def test_slide_metadata_alt_text(self):
        slide_img = Path("/tmp/slide_0001.png")
        data = SpeechData(
            title="Talk",
            transcript_segments=[{"start": 0.0, "end": 1.0, "text": "Hi."}],
            slide_timestamps=[{"slide_number": 1, "timestamp": 0.5}],
            slide_images=[slide_img],
            slide_metadata=[{"title": "My Custom Title"}],
        )
        result = _generate_interleaved_markdown(data)
        assert "My Custom Title" in result

    def test_merged_note_when_merged_exists(self, tmp_path):
        merged = tmp_path / "merged.txt"
        merged.write_text("merged")
        data = SpeechData(
            title="Talk",
            transcript_segments=[{"start": 0.0, "end": 1.0, "text": "Hi."}],
            merged_transcript_path=merged,
        )
        result = _generate_interleaved_markdown(data)
        assert "merged version available" in result

    def test_speaker_grouping(self):
        """Consecutive segments from the same speaker should be grouped."""
        data = SpeechData(
            title="Test",
            transcript_segments=[
                {"start": 0.0, "end": 2.0, "text": "First sentence.", "speaker": "Alice"},
                {"start": 2.0, "end": 4.0, "text": "Second sentence.", "speaker": "Alice"},
                {"start": 4.0, "end": 6.0, "text": "Third sentence.", "speaker": "Bob"},
            ],
        )
        result = _generate_interleaved_markdown(data)
        # Alice's sentences should be grouped together
        assert "First sentence. Second sentence." in result
        # Both speakers should appear in bold
        assert "**Alice**" in result
        assert "**Bob**" in result

    def test_slide_out_of_bounds_skipped(self):
        """Slide with index beyond slide_images should be skipped."""
        slide_img = Path("/tmp/slide_0001.png")
        data = SpeechData(
            title="Test",
            transcript_segments=[
                {"start": 0.0, "end": 5.0, "text": "Some text."},
            ],
            slide_timestamps=[
                {"slide_number": 1, "timestamp": 1.0},
                {"slide_number": 5, "timestamp": 3.0},  # out of bounds
            ],
            slide_images=[slide_img],
        )
        result = _generate_interleaved_markdown(data)
        assert "slide_0001.png" in result
        # Out-of-bounds slide should not appear
        assert result.count("![") == 1

    def test_no_speaker_label(self):
        """Segments without speaker labels should render without bold header."""
        data = SpeechData(
            title="Test",
            transcript_segments=[
                {"start": 0.0, "end": 2.0, "text": "No speaker here."},
            ],
        )
        result = _generate_interleaved_markdown(data)
        assert "No speaker here." in result
        assert "**" not in result.split("---")[1].split("---")[0]  # no bold in transcript body


# ---------------------------------------------------------------------------
# generate_markdown — integration (layout selection)
# ---------------------------------------------------------------------------

class TestGenerateMarkdownLayout:
    def test_selects_sequential_without_timestamps(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        transcript = tmp_path / "merged.txt"
        transcript.write_text("Plain transcript.")
        data = SpeechData(title="Test", merged_transcript_path=transcript)
        generate_markdown(config, data)
        out = capsys.readouterr().out
        assert "sequential layout" in out
        content = (tmp_path / "transcript.md").read_text()
        assert "## Transcript" in content

    def test_selects_interleaved_with_timestamps(self, tmp_path, capsys):
        config = SpeechConfig(url="x", output_dir=tmp_path, skip_existing=False)
        slide_img = tmp_path / "slides" / "slide_0001.png"
        slide_img.parent.mkdir()
        slide_img.write_text("img")
        data = SpeechData(
            title="Test",
            transcript_segments=[
                {"start": 0.0, "end": 2.0, "text": "Hello."},
            ],
            slide_timestamps=[{"slide_number": 1, "timestamp": 1.0}],
            slide_images=[slide_img],
        )
        generate_markdown(config, data)
        out = capsys.readouterr().out
        assert "timestamp-based" in out
