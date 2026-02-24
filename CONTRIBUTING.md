# Contributing

Thanks for your interest in contributing!

## Getting Started

1. Fork the repo and clone your fork
2. Install external tools: `brew install ffmpeg wdiff ollama` (macOS) or `apt install ffmpeg wdiff` + [Ollama](https://ollama.com/) (Ubuntu/Debian)
3. Install in editable mode with dev dependencies: `pip install -e .[dev]`
4. Run tests: `pytest -v`

## Making Changes

- Create a feature branch from `main`
- Add tests for new functionality
- Run `pytest -v` and ensure all tests pass before submitting
- Keep PRs focused — one feature or fix per PR

## Code Organization

Source code lives in `src/transcribe_critic/`:

- `shared.py` — Shared types (`SpeechConfig`, `SpeechData`) and utilities
- `transcriber.py` — Pipeline orchestration, CLI, cost estimation, source survival analysis
- `download.py` — Media downloading (yt-dlp)
- `transcription.py` — Whisper transcription and multi-model ensembling
- `diarization.py` — Speaker diarization (pyannote)
- `merge.py` — Merge/alignment logic (wdiff, chunking, LLM adjudication)
- `slides.py` — Slide extraction and vision analysis
- `output.py` — Markdown generation

Tests are in `tests/`, organized by module: `test_shared.py`, `test_transcriber.py`, `test_transcription.py`, `test_download.py`, `test_merge.py`, `test_output.py`, `test_slides.py`, `test_diarization.py`.

## Reporting Issues

Open an issue with steps to reproduce. For transcription quality issues, include the source URL and which merge path was used (structured vs. flat).
