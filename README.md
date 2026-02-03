# Speech Transcriber

Automated transcription pipeline for speeches from video URLs. Downloads media, transcribes with Whisper, extracts slides, and generates markdown with slides interleaved at correct timestamps.

## Features

- **Multi-source transcription**: Combines YouTube captions + Whisper AI for best accuracy
- **Multi-model ensembling**: Run multiple Whisper models and merge results
- **Slide extraction**: Automatic scene detection to capture presentation slides
- **Slide analysis**: Optional vision API to describe slide content
- **Timestamp alignment**: Places slides inline with transcript at correct moments
- **Cost estimation**: Shows estimated API costs before running
- **Local-only mode**: `--no-api` flag for completely free operation

## Installation

### Dependencies

```bash
# Required tools
brew install ffmpeg
pip install yt-dlp mlx-whisper

# Optional (for wdiff-based comparison)
brew install wdiff

# Optional (for API features)
pip install anthropic
```

### Apple Silicon

This tool is optimized for Apple Silicon Macs using `mlx-whisper`. On other platforms, it falls back to `openai-whisper` (slower, CPU-based):

```bash
pip install openai-whisper  # For non-Apple Silicon
```

## Quick Start

```bash
# Basic usage - Whisper transcript + slides
python speech_transcriber.py "https://youtube.com/watch?v=..."

# Run completely free (no API calls)
python speech_transcriber.py "https://youtube.com/watch?v=..." --no-api
```

## Usage Examples

### Free/Local Operation

```bash
# Single Whisper model (fastest)
python speech_transcriber.py "https://youtube.com/watch?v=..." --no-api

# Ensemble multiple models (more accurate, still free)
python speech_transcriber.py "https://youtube.com/watch?v=..." \
    --whisper-models small,medium --no-api
```

### With API Features

```bash
# Set API key (or use --api-key flag)
export ANTHROPIC_API_KEY="your-key-here"

# Analyze slides with vision API
python speech_transcriber.py "https://youtube.com/watch?v=..." --analyze-slides

# Create "critical text" by merging YouTube + Whisper
python speech_transcriber.py "https://youtube.com/watch?v=..." --merge-sources

# Full pipeline with all features
python speech_transcriber.py "https://youtube.com/watch?v=..." \
    --whisper-models small,medium \
    --merge-sources \
    --analyze-slides
```

### Custom Options

```bash
# Custom output directory
python speech_transcriber.py "https://youtube.com/watch?v=..." -o ./my_transcript

# Use specific Whisper model
python speech_transcriber.py "https://youtube.com/watch?v=..." --whisper-models large

# Adjust slide detection sensitivity (0.0-1.0, lower = more slides)
python speech_transcriber.py "https://youtube.com/watch?v=..." --scene-threshold 0.15

# Force re-processing (ignore existing files)
python speech_transcriber.py "https://youtube.com/watch?v=..." --no-skip

# Verbose output
python speech_transcriber.py "https://youtube.com/watch?v=..." -v
```

## Output Files

```
output_dir/
├── speech.mp3                    # Downloaded audio
├── speech.mp4                    # Downloaded video
├── speech.en.vtt                 # YouTube captions (if available)
├── speech_small.txt              # Whisper small transcript
├── speech_medium.txt             # Whisper medium transcript
├── speech_ensembled.txt          # Ensembled from multiple models
├── speech.json                   # Transcript with timestamps
├── transcript_merged.txt         # Merged with YouTube (if --merge-sources)
├── slides/
│   ├── slide_0001.png
│   ├── slide_0002.png
│   └── ...
├── slide_timestamps.json         # When each slide appears
├── slides_transcript.json        # Slide descriptions (if --analyze-slides)
└── transcript.md                 # Final markdown with interleaved slides
```

## Pipeline Stages

| Stage | Tool | API Required |
|-------|------|--------------|
| 1. Download | yt-dlp | No |
| 2. Transcribe | mlx-whisper | No |
| 3. Extract slides | ffmpeg | No |
| 4. Analyze slides | Claude Vision | Yes (optional) |
| 4b. Merge sources | Claude + wdiff | Yes (optional) |
| 5. Generate markdown | Python | No |

## Cost Estimation

When API features are enabled, the tool displays estimated costs before running:

```
==================================================
ESTIMATED API COSTS
==================================================
  Slide analysis: 45 slides × $0.02 = $0.90
  Source merging: ~12000 input words + 6000 output = $0.13
  Whisper ensemble: 2 models = $0.12

  TOTAL: $1.15 (estimate)
  Note: Actual costs may vary based on transcript length
==================================================
```

### Typical Costs (40-minute speech)

| Feature | Estimated Cost |
|---------|---------------|
| `--analyze-slides` | $0.50 - $2.00 |
| `--merge-sources` | $0.10 - $0.30 |
| `--whisper-models small,medium` | $0.05 - $0.15 |
| All features | $0.65 - $2.50 |
| `--no-api` | **Free** |

## How It Works

### Multi-Model Ensembling

When using multiple Whisper models (e.g., `--whisper-models small,medium`):

1. Runs each model independently
2. Uses `wdiff` to identify differences (normalized: no caps, no punctuation)
3. Claude resolves disagreements, preferring:
   - Real words over transcription errors ("progeria" > "progerium")
   - Proper nouns and technical terms
   - Grammatically correct versions

### Source Merging

When using `--merge-sources`:

1. Compares YouTube captions with Whisper transcript via `wdiff`
2. Identifies meaningful differences (not just formatting)
3. Claude merges, preferring:
   - YouTube for proper nouns, names, technical terms
   - Whisper for punctuation and sentence structure
   - YouTube for content that Whisper missed

### Timestamp-Based Slide Placement

1. Whisper provides word-level timestamps (JSON output)
2. ffmpeg reports when each scene change occurs
3. Slides are inserted into transcript at the moment they appeared in the video

## Command Reference

```
usage: speech_transcriber.py [-h] [-o OUTPUT_DIR]
                             [--whisper-models WHISPER_MODELS]
                             [--scene-threshold SCENE_THRESHOLD]
                             [--analyze-slides] [--merge-sources]
                             [--api-key API_KEY] [--no-api] [--no-skip] [-v]
                             url

positional arguments:
  url                   URL of the speech video

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR         Output directory (default: ./transcripts/<title>)
  --whisper-models      Model(s) to use, comma-separated (default: medium)
                        Options: tiny, base, small, medium, large
  --scene-threshold     Scene detection threshold 0-1 (default: 0.1)
  --analyze-slides      Use Claude vision API to analyze slides
  --merge-sources       Merge YouTube captions with Whisper transcript
  --api-key             Anthropic API key (or set ANTHROPIC_API_KEY env var)
  --no-api              Skip all API-dependent features
  --no-skip             Re-process even if files exist
  -v, --verbose         Show detailed command output
```

## Troubleshooting

### "No Whisper implementation found"

Install either mlx-whisper (Apple Silicon) or openai-whisper:

```bash
pip install mlx-whisper    # Apple Silicon (recommended)
pip install openai-whisper # Other platforms
```

### ffmpeg scene detection captures too few/many slides

Adjust the threshold:
- Lower value (e.g., `0.05`) = more sensitive, more slides
- Higher value (e.g., `0.2`) = less sensitive, fewer slides

```bash
python speech_transcriber.py "..." --scene-threshold 0.05
```

### wdiff not found

The tool works without wdiff but comparison is less precise:

```bash
brew install wdiff  # macOS
apt install wdiff   # Ubuntu/Debian
```

### API rate limits

For long speeches or many slides, you may hit rate limits. The tool processes sequentially to minimize this, but you can:

1. Run with `--no-api` first to get basic transcript
2. Run again with API features (existing files are skipped)

## License

MIT

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [MLX Whisper](https://github.com/ml-explore/mlx-examples) - Apple Silicon optimization
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Media downloading
- [Anthropic Claude](https://www.anthropic.com/) - Vision and text analysis
