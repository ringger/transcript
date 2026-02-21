# Contributing

Thanks for your interest in contributing!

## Getting Started

1. Fork the repo and clone your fork
2. Install dependencies: `pip install -r requirements.txt`
3. Install external tools: `brew install ffmpeg wdiff` (macOS)
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
