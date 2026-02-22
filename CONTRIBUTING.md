# Contributing

Thanks for your interest in contributing!

## Getting Started

1. Fork the repo and clone your fork
2. Install external tools: `brew install ffmpeg wdiff` (macOS) or `apt install ffmpeg wdiff` (Ubuntu/Debian)
3. Install Python dependencies: `pip install -r requirements.txt` (auto-selects the right Whisper for your platform)
4. Run tests: `pytest -v`

## Making Changes

- Create a feature branch from `main`
- Add tests for new functionality
- Run `pytest -v` and ensure all tests pass before submitting
- Keep PRs focused — one feature or fix per PR

## Code Organization

- `shared.py` — Shared types (`SpeechConfig`, `SpeechData`) and utilities
- `merge.py` — Merge/alignment logic (wdiff, chunking, LLM adjudication)
- `transcriber.py` — Pipeline orchestration, download, transcription, slides, CLI
- `test_transcriber.py` — Test suite

## Reporting Issues

Open an issue with steps to reproduce. For transcription quality issues, include the source URL and which merge path was used (structured vs. flat).
