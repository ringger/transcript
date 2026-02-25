# Transcribe Critic

Automated pipeline for producing accurate speech transcripts from video URLs. Downloads media, transcribes with multiple Whisper models, and merges all available sources — Whisper, YouTube captions, and optional external transcripts — into a single "critical text" using LLM-based adjudication.

The approach applies principles from [textual criticism](https://en.wikipedia.org/wiki/Textual_criticism): multiple independent "witnesses" to the same speech are aligned, compared, and merged by an LLM that judges each difference on its merits, without knowing which source produced which reading. This builds on earlier work applying similar techniques to OCR ([Ringger & Lund, 2014](https://scholarsarchive.byu.edu/facpub/1647/); [Lund et al., 2013](https://www.researchgate.net/publication/220861175_Error_Correction_with_In-Domain_Training_Across_Multiple_OCR_System_Outputs)), replacing trained classifiers with an LLM as the eclectic editor.

## How is this different from WhisperX?

[WhisperX](https://github.com/m-bain/whisperX) improves a single Whisper run with voice-activity-detection (VAD) chunking, word-level timestamps, and speaker diarization — but the transcript still comes from one model pass. Transcribe Critic takes a different approach: it runs multiple Whisper models, pulls in YouTube captions and external human-edited transcripts, and treats them all as independent witnesses. An LLM then adjudicates every disagreement blindly, without knowing which source produced which reading. The result is a merged "critical text" that is more accurate than any single source. If you just need fast, well-segmented Whisper output, WhisperX is the right tool; if you want the most accurate transcript possible from multiple sources, this is.

## Features

- **Critical text merging**: Combines 2–3+ transcript sources into the most accurate version using blind, anonymous presentation to an LLM — no source receives preferential treatment
- **wdiff-based alignment**: Uses longest common subsequence alignment (via `wdiff`) to keep chunks properly aligned across sources of different lengths, replacing naive proportional slicing
- **Multi-model Whisper ensembling**: Runs multiple Whisper models (default: small + medium + distil-large-v3) and resolves disagreements via LLM with anonymous A/B/C labels
- **Anti-hallucination**: Whisper runs use `condition_on_previous_text=False` and other flags to prevent cascading hallucination; residual repetition loops are automatically detected and collapsed
- **External transcript support**: Merges in human-edited transcripts (e.g., from publisher websites) as an additional source
- **Structured transcript preservation**: When external transcripts have speaker labels and timestamps, the merged output preserves that structure
- **Slide extraction and analysis**: Automatic scene detection for presentation slides, with optional vision API descriptions
- **Make-style DAG pipeline**: Each stage checks whether its outputs are newer than its inputs, skipping unnecessary work — `--steps` allows re-running specific stages in isolation
- **Checkpoint resumption**: Long operations save checkpoints and resume after interruption — merge chunks, diarization segmentation, and embedding extraction all checkpoint independently
- **Cost estimation**: Shows estimated API costs before running (`--dry-run` for estimation only)
- **Local-first LLM**: Uses Ollama by default for free, local operation — no API key needed
- **Speaker diarization**: Identifies who is speaking using pyannote.audio, with automatic or manual speaker naming — LLM speaker identification uses video metadata (title, description) for correct name spellings
- **Timestamped logging**: All pipeline output prefixed with `[HH:MM:SS]` wall-clock timestamps for log correlation during long runs
- **Whisper-only mode**: `--no-llm` to skip all LLM features and run Whisper only

## Installation

```bash
pip install transcribe-critic
```

### System Dependencies

```bash
# Required tools
brew install ffmpeg wdiff    # macOS
# apt install ffmpeg wdiff   # Ubuntu/Debian

# Install Ollama for local LLM (used by default for merging/ensembling)
brew install ollama          # macOS
# curl -fsSL https://ollama.com/install.sh | sh  # Linux

# Pull a model (one-time)
ollama pull qwen2.5:14b
```

### From Source

```bash
git clone https://github.com/ringger/transcribe-critic.git
cd transcribe-critic
pip install -e .          # editable install
pip install -e .[dev]     # with test dependencies
pip install -e .[diarize] # with speaker diarization
```

## Quick Start

```bash
# Basic: Whisper transcription + local LLM merge (free, uses Ollama)
transcribe-critic "https://youtube.com/watch?v=..."

# With an external human-edited transcript for three-way merge
transcribe-critic "https://youtube.com/watch?v=..." \
    --external-transcript "https://example.com/transcript"

# Use Anthropic Claude API instead of local Ollama (higher quality, costs money)
transcribe-critic "https://youtube.com/watch?v=..." --api

# Whisper only — no LLM merging at all
transcribe-critic "https://youtube.com/watch?v=..." --no-llm
```

## Usage Examples

### Podcast

```bash
# Podcast episode — audio only, no video or captions
transcribe-critic --podcast "https://www.iheart.com/podcast/.../episode/..."
transcribe-critic --podcast "https://podcasts.apple.com/us/podcast/..."
```

### Speaker Diarization

```bash
# Identify who is speaking (requires pyannote.audio and HF_TOKEN)
pip install transcribe-critic[diarize]
export HF_TOKEN="hf_..."  # HuggingFace token with pyannote model access

# Auto-detect speaker names from introductions
transcribe-critic --diarize --num-speakers 2 --podcast "https://..."

# Manual speaker names (in order of first appearance)
transcribe-critic --diarize --speaker-names "Ross Douthat,Dario Amodei" --podcast "https://..."
```

### Speech-Only (No Slides)

```bash
# YouTube talk or interview — skip slide extraction
transcribe-critic "https://youtube.com/watch?v=..." --no-slides

# With external transcript for higher accuracy
transcribe-critic "https://youtube.com/watch?v=..." \
    --no-slides \
    --external-transcript "https://example.com/transcript"
```

### Presentation with Slides

```bash
# Extract slides and interleave with transcript
transcribe-critic "https://youtube.com/watch?v=..."

# Also describe slide content with vision API
transcribe-critic "https://youtube.com/watch?v=..." --analyze-slides
```

### Custom Options

```bash
# Custom output directory
transcribe-critic "https://youtube.com/watch?v=..." -o ./my_transcript

# Use specific Whisper models (default: small,medium,distil-large-v3)
transcribe-critic "https://youtube.com/watch?v=..." --whisper-models medium,distil-large-v3

# Use a different local model
transcribe-critic "https://youtube.com/watch?v=..." --local-model llama3.3

# Adjust slide detection sensitivity (0.0–1.0, lower = more slides)
transcribe-critic "https://youtube.com/watch?v=..." --scene-threshold 0.15

# Force re-processing (ignore existing files)
transcribe-critic "https://youtube.com/watch?v=..." --force

# Re-run only the Whisper ensemble step (uses existing Whisper outputs)
transcribe-critic "https://youtube.com/watch?v=..." --steps ensemble -o ./my_transcript

# Re-run only the merge step (uses existing Whisper outputs)
transcribe-critic "https://youtube.com/watch?v=..." --steps merge -o ./my_transcript

# Re-run transcription and merge only
transcribe-critic "https://youtube.com/watch?v=..." --steps transcribe,merge -o ./my_transcript

# Verbose output
transcribe-critic "https://youtube.com/watch?v=..." -v
```

## Output Files

```
output_dir/
├── metadata.json                 # Source URL, title, duration, etc.
├── audio.mp3                     # Downloaded audio
├── audio.wav                     # Converted for diarization (if --diarize)
├── video.mp4                     # Downloaded video (if slides enabled)
├── captions.en.vtt               # YouTube captions (if available)
├── whisper_small.txt              # Whisper small transcript
├── whisper_small.json             # Whisper small with timestamps
├── whisper_medium.txt             # Whisper medium transcript
├── whisper_medium.json            # Whisper medium with timestamps
├── whisper_distil-large-v3.txt    # Whisper distil-large-v3 transcript
├── whisper_distil-large-v3.json   # Whisper distil-large-v3 with timestamps
├── whisper_merged.txt             # Merged from multiple Whisper models via adjudication
├── diarization.json              # Speaker segments (if --diarize)
├── diarization_segmentation.npy  # Cached segmentation (if --diarize)
├── diarization_embeddings.npy    # Cached embeddings (if --diarize)
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

| Stage | Step name | Tool | Optional |
|-------|-----------|------|----------|
| [1] Download media | `download` | yt-dlp | No |
| [2] Transcribe audio | `transcribe` | mlx-whisper | No |
| [2b] Whisper ensemble | `ensemble` | LLM + wdiff | Yes (on by default with 2+ models; default: 3 models) |
| [2c] Speaker diarization | `diarize` | pyannote.audio | Yes (`--diarize`) |
| [3] Extract slides | `slides` | ffmpeg | Yes (skipped with `--no-slides` / `--podcast`) |
| [4] Analyze slides with vision | `slides` | LLM + vision | Yes (`--analyze-slides`) |
| [4b] Merge transcript sources | `merge` | LLM + wdiff | Yes (on by default; `--no-merge` to skip) |
| [5] Generate markdown | `markdown` | Python | No |
| [6] Source survival analysis | `analysis` | wdiff | No |

Use `--steps <step1>,<step2>,...` to run only specific stages. Existing outputs from skipped stages are loaded automatically. This is useful for re-running just the ensemble or merge after fixing a bug, without re-downloading or re-transcribing.

## How It Works

### Critical Text Merging

The core idea — inspired by textual criticism — is to treat multiple transcripts as independent witnesses to the same speech and adjudicate their differences. Given 2–3+ sources:

1. **Align** all sources against an anchor text using `wdiff` (longest common subsequence), producing word-position maps that keep chunks synchronized even when sources differ in length
2. **Chunk** the aligned sources into ~500-word segments
3. **Present** each chunk to Claude with **anonymous labels** (Source 1, Source 2, Source 3) — source names are never revealed, preventing provenance bias
4. **Adjudicate** — Claude chooses the best reading at each point of disagreement, preferring proper nouns, grammatical correctness, and contextual fit
5. **Reassemble** the merged chunks, restoring speaker labels and timestamps from the structured source (if available)

When an external transcript has structure (speaker labels, timestamps), the merge preserves that skeleton while improving the text content from all sources.

Unlike a traditional critical edition, the pipeline does not produce an apparatus of variants, construct a stemma of source relationships, or preserve editorial rationale for each decision. The goal is a single best-reading transcript, not a scholarly edition.

### Source Survival Analysis

After merging, `wdiff -s` compares each source against the merged output, showing how much each source contributed to the final text. Here is an actual survival analysis from a 3-hour podcast episode transcribed with Whisper (small + medium, merged via adjudication), YouTube auto-captions, and a human-edited external transcript:

```
Source                       Words   Common  Output Coverage  Retention
------------------------- -------- -------- --------------- ----------
Whisper (merged)            28,277   27,441             90%        97%
YouTube captions            30,668   28,741             94%        94%
External transcript         33,122   30,245             99%        91%
Merged output               30,524
```

- **Output Coverage**: what percentage of the merged output's words appear in this source (how much of the final text did this source "cover"?)
- **Retention**: what percentage of this source's words survived into the merged output

No single source matches the merged output — the merged text draws from all three. The external transcript has the highest coverage (99% of merged words present), but the merge still corrects ~1% of its content using the other sources. Whisper contributes readings not found in either captions or the external transcript, and vice versa.

Here are specific corrections the merge made by adjudicating across sources:

| Whisper | YouTube captions | External transcript | Merged (correct) |
|---------|-----------------|--------------------|--------------------|
| "Cloud Opus" | — | "Claude Opus" | **Claude Opus** (product name) |
| "Ross Douthend" | "ross douthat" | "Ross Douthat" | **Ross Douthat** (person name) |
| "GPT 5.3 codecs" | — | "GPT-5.3 Codex" | **GPT 5.3 Codex** (model name, not audio codec) |
| "is source code" | — | "its source code" | **its source code** (grammar) |

Each source alone gets some things right and others wrong. Whisper hallucinates proper nouns ("Cloud" for "Claude", "Douthend" for "Douthat"). YouTube captions lack capitalization and punctuation but sometimes have correct spellings. The external transcript has the best proper nouns but may paraphrase or omit filler words. The merge selects the best reading at each disagreement, producing a transcript more accurate than any individual source.

### Multi-Model Whisper Merging

When using multiple Whisper models (default: `small,medium,distil-large-v3`):

1. Runs each model independently with anti-hallucination flags
2. Uses `wdiff` to identify specific word-level differences between each non-base model and the base (largest model)
3. For 3+ models, merges pairwise diffs at the same positions into unified diffs with per-model readings
4. Clusters nearby differences and presents each cluster to an LLM with anonymous labels (A/B or A/B/C) and surrounding context — model names are never revealed
5. The LLM picks a letter for each disagreement — constrained to choose between actual transcriptions, preventing hallucinated text
6. Chosen readings are surgically applied to the base transcript, leaving uncontested regions untouched

This targeted diff resolution avoids the problems of full-text rewriting (chunk-boundary duplication, errors in uncontested regions, wasted tokens). The implementation runs Whisper-vs-Whisper adjudication first to produce a single merged Whisper witness (`whisper_merged.txt`), which then enters the multi-source merge alongside captions and external transcripts.

### Speaker Diarization

When `--diarize` is enabled, the pipeline identifies who is speaking at each point in the audio by combining two independent signals:

1. **pyannote.audio** runs a neural segmentation model over the audio in sliding ~5-second windows, producing frame-level speaker activity probabilities. A global clustering step stitches local predictions across the full recording into consistent speaker labels (SPEAKER_00, SPEAKER_01, etc.). The model handles overlapping speech natively and operates purely on the audio signal — no linguistic content is used.

2. **Whisper word timestamps** (`--word-timestamps True`) provide per-word `{start, end}` timing from the transcription model.

The pipeline links these by **midpoint matching**: for each word, it finds which speaker segment overlaps the word's temporal midpoint. Each transcript segment is then assigned the majority speaker of its constituent words. The result is a structured transcript in bracketed format (`[H:MM:SS] Speaker: text`) that feeds directly into the existing merge pipeline as a structural skeleton.

**Speaker identification** maps generic labels to real names via three methods (in priority order):
- `--speaker-names "Alice,Bob"` — manual mapping by order of first appearance
- LLM-based detection — reads the first ~500 words and infers names from introductions, using video metadata (title, description, channel) for correct spellings (e.g., corrects Whisper's "Douthit" to "Douthat" when the video description contains the correct name)
- `--no-llm` — keeps generic SPEAKER_00/SPEAKER_01 labels

**Diarization checkpointing** breaks the expensive pyannote pipeline into 6 independently cached steps. Segmentation (the neural model pass, ~50% of runtime) and embedding extraction (the other slow step) both save to `.npy` files. Embeddings checkpoint every 10 batches to a partial file, enabling resume mid-extraction. If any step's output is newer than the audio file, it is skipped on re-run.

### Make-Style Staleness Checks

Every stage checks `is_up_to_date(output, *inputs)` — if the output file is newer than all input files, the stage is skipped. This means you can re-run the pipeline after changing options and only the affected stages will execute.

## Cost Estimation

```
==================================================
ESTIMATED API COSTS
==================================================
  Source merging: 3 sources × 59 chunks = $1.03
  Whisper ensemble: 3 models × 98 clusters = $0.72

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

This tool is inspired by [textual criticism](https://en.wikipedia.org/wiki/Textual_criticism) — the scholarly discipline of comparing multiple manuscript witnesses to reconstruct an authoritative text — applying its core principles (independent witnesses, alignment, adjudication) to speech transcription.

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

`pip install transcribe-critic` automatically installs the right Whisper for your platform (mlx-whisper on Apple Silicon, openai-whisper elsewhere). If you installed from source and see this error:

```bash
pip install mlx-whisper    # Apple Silicon
pip install openai-whisper # Other platforms
```

### wdiff not found

Required for alignment-based merging:

```bash
brew install wdiff  # macOS
apt install wdiff   # Ubuntu/Debian
```

### Diarization fails on short audio clips

pyannote's audio decoder can produce sample-count mismatches with MP3 files, especially short clips. The pipeline automatically converts MP3 to WAV before diarization, so this should be handled transparently. If you still encounter issues, you can manually provide a WAV file:

```bash
ffmpeg -i output_dir/audio.mp3 -ar 16000 -ac 1 output_dir/audio.wav
```

The pipeline will use an existing `audio.wav` over `audio.mp3` for diarization.

### API timeouts

The tool retries on timeouts (120s per attempt, up to 5 retries with exponential backoff). Long merges save per-chunk checkpoints, so interrupted runs resume from the last completed chunk.

### ffmpeg scene detection captures too few/many slides

```bash
transcribe-critic "..." --scene-threshold 0.05  # More slides
transcribe-critic "..." --scene-threshold 0.20  # Fewer slides
```

## License

MIT

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition
- [Distil-Whisper](https://github.com/huggingface/distil-whisper) — Distilled large-v3 model (faster, fewer hallucinations)
- [MLX Whisper](https://github.com/ml-explore/mlx-examples) — Apple Silicon optimization
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — Media downloading
- [Anthropic Claude](https://www.anthropic.com/) — LLM-based adjudication and vision analysis
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — Speaker diarization
- [wdiff](https://www.gnu.org/software/wdiff/) — Word-level diff for alignment and comparison
