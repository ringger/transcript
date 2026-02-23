# Speech Transcriber

Automated pipeline for producing accurate speech transcripts from video URLs. Downloads media, transcribes with multiple Whisper models, and merges all available sources — Whisper, YouTube captions, and optional external transcripts — into a single "critical text" using LLM-based adjudication.

The approach applies principles from [textual criticism](https://en.wikipedia.org/wiki/Textual_criticism): multiple independent "witnesses" to the same speech are aligned, compared, and merged by an LLM that judges each difference on its merits, without knowing which source produced which reading. This builds on earlier work applying similar techniques to OCR ([Ringger & Lund, 2014](https://scholarsarchive.byu.edu/facpub/1647/); [Lund et al., 2013](https://www.researchgate.net/publication/220861175_Error_Correction_with_In-Domain_Training_Across_Multiple_OCR_System_Outputs)), replacing trained classifiers with an LLM as the eclectic editor.

## Features

- **Critical text merging**: Combines 2–3+ transcript sources into the most accurate version using blind, anonymous presentation to an LLM — no source receives preferential treatment
- **wdiff-based alignment**: Uses longest common subsequence alignment (via `wdiff`) to keep chunks properly aligned across sources of different lengths, replacing naive proportional slicing
- **Multi-model Whisper ensembling**: Runs multiple Whisper models (e.g., small + medium) and resolves disagreements via LLM
- **External transcript support**: Merges in human-edited transcripts (e.g., from publisher websites) as an additional source
- **Structured transcript preservation**: When external transcripts have speaker labels and timestamps, the merged output preserves that structure
- **Slide extraction and analysis**: Automatic scene detection for presentation slides, with optional vision API descriptions
- **Make-style DAG pipeline**: Each stage checks whether its outputs are newer than its inputs, skipping unnecessary work
- **Checkpoint resumption**: Long merge operations save per-chunk checkpoints, resuming from where they left off after interruption
- **Cost estimation**: Shows estimated API costs before running (`--dry-run` for estimation only)
- **Local-first LLM**: Uses Ollama by default for free, local operation — no API key needed
- **Speaker diarization**: Identifies who is speaking using pyannote.audio, with automatic or manual speaker naming
- **Whisper-only mode**: `--no-llm` to skip all LLM features and run Whisper only

## Installation

### Dependencies

```bash
# Required tools
brew install ffmpeg wdiff    # macOS
# apt install ffmpeg wdiff   # Ubuntu/Debian

# Install Python dependencies (auto-selects mlx-whisper on Apple Silicon, openai-whisper elsewhere)
pip install -r requirements.txt

# Install Ollama for local LLM (used by default for merging/ensembling)
brew install ollama          # macOS
# curl -fsSL https://ollama.com/install.sh | sh  # Linux

# Pull a model (one-time)
ollama pull qwen2.5
```

## Quick Start

```bash
# Basic: Whisper transcription + local LLM merge (free, uses Ollama)
python transcriber.py "https://youtube.com/watch?v=..."

# With an external human-edited transcript for three-way merge
python transcriber.py "https://youtube.com/watch?v=..." \
    --external-transcript "https://example.com/transcript"

# Use Anthropic Claude API instead of local Ollama (higher quality, costs money)
python transcriber.py "https://youtube.com/watch?v=..." --api

# Whisper only — no LLM merging at all
python transcriber.py "https://youtube.com/watch?v=..." --no-llm
```

## Usage Examples

### Podcast

```bash
# Podcast episode — audio only, no video or captions
python transcriber.py --podcast "https://www.iheart.com/podcast/.../episode/..."
python transcriber.py --podcast "https://podcasts.apple.com/us/podcast/..."
```

### Speaker Diarization

```bash
# Identify who is speaking (requires pyannote.audio and HF_TOKEN)
pip install pyannote.audio
export HF_TOKEN="hf_..."  # HuggingFace token with pyannote model access

# Auto-detect speaker names from introductions
python transcriber.py --diarize --num-speakers 2 --podcast "https://..."

# Manual speaker names (in order of first appearance)
python transcriber.py --diarize --speaker-names "Ross Douthat,Dario Amodei" --podcast "https://..."
```

### Speech-Only (No Slides)

```bash
# YouTube talk or interview — skip slide extraction
python transcriber.py "https://youtube.com/watch?v=..." --no-slides

# With external transcript for higher accuracy
python transcriber.py "https://youtube.com/watch?v=..." \
    --no-slides \
    --external-transcript "https://example.com/transcript"
```

### Presentation with Slides

```bash
# Extract slides and interleave with transcript
python transcriber.py "https://youtube.com/watch?v=..."

# Also describe slide content with vision API
python transcriber.py "https://youtube.com/watch?v=..." --analyze-slides
```

### Custom Options

```bash
# Custom output directory
python transcriber.py "https://youtube.com/watch?v=..." -o ./my_transcript

# Use specific Whisper models
python transcriber.py "https://youtube.com/watch?v=..." --whisper-models large

# Use a different local model
python transcriber.py "https://youtube.com/watch?v=..." --local-model llama3.3

# Adjust slide detection sensitivity (0.0–1.0, lower = more slides)
python transcriber.py "https://youtube.com/watch?v=..." --scene-threshold 0.15

# Force re-processing (ignore existing files)
python transcriber.py "https://youtube.com/watch?v=..." --force

# Verbose output
python transcriber.py "https://youtube.com/watch?v=..." -v
```

## Output Files

```
output_dir/
├── metadata.json                 # Source URL, title, duration, etc.
├── audio.mp3                     # Downloaded audio
├── video.mp4                     # Downloaded video (if slides enabled)
├── captions.en.vtt               # YouTube captions (if available)
├── small.txt                     # Whisper small transcript
├── medium.txt                    # Whisper medium transcript
├── ensembled.txt                 # Ensembled from multiple Whisper models
├── medium.json                   # Transcript with timestamps
├── diarization.json              # Speaker segments (if --diarize)
├── diarized.txt                  # Speaker-labeled transcript (if --diarize)
├── transcript_merged.txt         # Critical text (merged from all sources)
├── analysis.md                   # Source survival analysis
├── transcript.md                 # Final markdown output
├── merge_chunks/                 # Per-chunk checkpoints (resumable)
│   ├── .version
│   ├── chunk_000.json
│   └── ...
├── slide_timestamps.json         # Slide timing data
├── slides_transcript.json        # (if --analyze-slides)
└── slides/                       # (if slides enabled)
    ├── slide_0001.png
    └── ...
```

## Pipeline Stages

Optional stages are skipped based on flags. Stage numbers are fixed regardless of which stages run.

| Stage | Tool | Optional |
|-------|------|----------|
| [1] Download media | yt-dlp | No |
| [2] Transcribe audio | mlx-whisper | No |
| [2b] Speaker diarization | pyannote.audio | Yes (`--diarize`) |
| [3] Extract slides | ffmpeg | Yes (skipped with `--no-slides` / `--podcast`) |
| [4] Analyze slides with vision | LLM + vision | Yes (`--analyze-slides`) |
| [4b] Merge transcript sources | LLM + wdiff | Yes (on by default; `--no-merge` to skip) |
| [5] Generate markdown | Python | No |
| [6] Source survival analysis | wdiff | No |

## How It Works

### Critical Text Merging

The core innovation is treating transcript merging as textual criticism. Given 2–3+ independent "witnesses" to the same speech:

1. **Align** all sources against an anchor text using `wdiff` (longest common subsequence), producing word-position maps that keep chunks synchronized even when sources differ in length
2. **Chunk** the aligned sources into ~500-word segments
3. **Present** each chunk to Claude with **anonymous labels** (Source 1, Source 2, Source 3) — source names are never revealed, preventing provenance bias
4. **Adjudicate** — Claude chooses the best reading at each point of disagreement, preferring proper nouns, grammatical correctness, and contextual fit
5. **Reassemble** the merged chunks, restoring speaker labels and timestamps from the structured source (if available)

When an external transcript has structure (speaker labels, timestamps), the merge preserves that skeleton while improving the text content from all sources.

### Source Survival Analysis

After merging, `wdiff -s` compares each source against the merged output:

```
Source                       Words   Common  % of Merged  % of Source
------------------------- -------- -------- ------------ ------------
Whisper (ensembled)         28,277   27,441          90%          97%
YouTube captions            30,668   28,741          94%          94%
External transcript         33,122   30,245          99%          91%
Merged output               30,524
```

This shows how much each source contributed to the final text and which source the merged output most closely resembles.

### Multi-Model Ensembling

When using multiple Whisper models (default: `small,medium`):

1. Runs each model independently
2. Uses `wdiff` to identify differences (normalized: no caps, no punctuation)
3. Claude resolves disagreements, preferring real words over transcription errors and proper nouns over generic alternatives

### Speaker Diarization

When `--diarize` is enabled, the pipeline identifies who is speaking at each point in the audio by combining two independent signals:

1. **pyannote.audio** runs a neural segmentation model over the audio in sliding ~5-second windows, producing frame-level speaker activity probabilities. A global clustering step stitches local predictions across the full recording into consistent speaker labels (SPEAKER_00, SPEAKER_01, etc.). The model handles overlapping speech natively and operates purely on the audio signal — no linguistic content is used.

2. **Whisper word timestamps** (`--word-timestamps True`) provide per-word `{start, end}` timing from the transcription model.

The pipeline links these by **midpoint matching**: for each word, it finds which speaker segment overlaps the word's temporal midpoint. Each transcript segment is then assigned the majority speaker of its constituent words. The result is a structured transcript in bracketed format (`[H:MM:SS] Speaker: text`) that feeds directly into the existing merge pipeline as a structural skeleton.

**Speaker identification** maps generic labels to real names via three methods (in priority order):
- `--speaker-names "Alice,Bob"` — manual mapping by order of first appearance
- LLM-based detection — reads the first ~500 words and infers names from introductions
- `--no-llm` — keeps generic SPEAKER_00/SPEAKER_01 labels

### Make-Style Staleness Checks

Every stage checks `is_up_to_date(output, *inputs)` — if the output file is newer than all input files, the stage is skipped. This means you can re-run the pipeline after changing options and only the affected stages will execute.

## Cost Estimation

```
==================================================
ESTIMATED API COSTS
==================================================
  Source merging: 3 sources × 59 chunks = $1.03
  Whisper ensemble: 2 models × 59 chunks = $0.92

  TOTAL: $1.95 (estimate)
==================================================
```

### Typical Costs

| Feature | 20-min speech | 3-hour podcast |
|---------|--------------|----------------|
| Whisper ensemble | $0.05–$0.15 | $0.50–$1.00 |
| Source merging (2 sources) | $0.10–$0.30 | $0.50–$1.00 |
| Source merging (3 sources) | $0.15–$0.40 | $1.00–$2.00 |
| Slide analysis | $0.50–$2.00 | N/A |
| Local Ollama (default) | **Free** | **Free** |
| `--no-llm` | **Free** | **Free** |

## Background

This tool applies the principles of [textual criticism](https://en.wikipedia.org/wiki/Textual_criticism) — the scholarly discipline of comparing multiple manuscript witnesses to reconstruct an authoritative text — to the problem of speech transcription.

The approach has roots in earlier work applying noisy-channel models and multi-source correction to speech and OCR:

- **Ringger & Allen (1996)** — [Error Correction via a Post-Processor for Continuous Speech Recognition](https://www.researchgate.net/publication/2321329_Error_Correction_Via_A_Post-Processor_For_Continuous_Speech_Recognition) (ICASSP). Introduced SpeechPP, a noisy-channel post-processor that corrects ASR output using language and channel models with Viterbi beam search, developed as part of the [TRAINS/TRIPS](https://www.cs.rochester.edu/research/trains/) spoken dialogue systems at the University of Rochester. Extended with a fertility channel model in [Ringger & Allen, ICSLP 1996](https://scholarsarchive.byu.edu/facpub/1288/).
- **Ringger & Lund (2014)** — [How Well Does Multiple OCR Error Correction Generalize?](https://scholarsarchive.byu.edu/facpub/1647/) Demonstrated that aligning and merging outputs from multiple OCR engines significantly reduces word error rates.
- **Lund et al. (2013)** — [Error Correction with In-Domain Training Across Multiple OCR System Outputs](https://www.researchgate.net/publication/220861175_Error_Correction_with_In-Domain_Training_Across_Multiple_OCR_System_Outputs). Used A* alignment and trained classifiers (CRFs, MaxEnt) to choose the best reading from multiple OCR witnesses — a 52% relative decrease in word error rate.

The OCR work used A* alignment because page layout provides natural line boundaries, making alignment a series of short, bounded search problems. Speech has no such boundaries — different ASR systems segment a continuous audio stream arbitrarily — so this tool uses `wdiff` (LCS-based global alignment) instead. It also replaces the trained classifiers with an LLM, which brings world knowledge and contextual reasoning without requiring task-specific training data. The blind/anonymous presentation of sources is borrowed from peer review and prevents the LLM from developing source-level biases.

Related work in speech:
- **ROVER** ([Fiscus, 1997](https://ieeexplore.ieee.org/document/659110/)) — Statistical voting across multiple ASR outputs via word transition networks
- **Ensemble Methods for ASR** ([Lehmann](https://github.com/cassandra-lehmann/ensemble_methods_ASR_transcripts)) — Random Forest classifier for selecting words from multiple ASR systems

## Troubleshooting

### "No Whisper implementation found"

```bash
pip install mlx-whisper    # Apple Silicon (recommended)
pip install openai-whisper # Other platforms
```

### wdiff not found

Required for alignment-based merging:

```bash
brew install wdiff  # macOS
apt install wdiff   # Ubuntu/Debian
```

### API timeouts

The tool retries on timeouts (120s per attempt, up to 5 retries with exponential backoff). Long merges save per-chunk checkpoints, so interrupted runs resume from the last completed chunk.

### ffmpeg scene detection captures too few/many slides

```bash
python transcriber.py "..." --scene-threshold 0.05  # More slides
python transcriber.py "..." --scene-threshold 0.20  # Fewer slides
```

## License

MIT

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition
- [MLX Whisper](https://github.com/ml-explore/mlx-examples) — Apple Silicon optimization
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — Media downloading
- [Anthropic Claude](https://www.anthropic.com/) — LLM-based adjudication and vision analysis
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — Speaker diarization
- [wdiff](https://www.gnu.org/software/wdiff/) — Word-level diff for alignment and comparison
