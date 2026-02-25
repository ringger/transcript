# Whisper Ensemble Experiment Results

## Dataset

Three files from Rev16 evaluated against reference transcripts using meeteval WER.

## Current Best Results

| File | small | medium | distil-large-v3 | 2-way merged (s+m) | 3-way merged |
|------|-------|--------|-----------------|-------------------|--------------|
| 3 | 31.5% | 29.5% | 29.6% | **28.5%** | 29.4% |
| 4 | 31.3% | 29.2% | 27.6% | 28.2% | 28.5% |
| 9 | 24.1% | 24.5% | 21.6% | 23.1% | 21.7% |
| **Avg** | **28.9%** | **27.7%** | **26.3%** | **26.6%** | **26.5%** |

Best single model: distil-large-v3 (26.3%). Best ensemble: 2-way small+medium (26.6%). 3-way ensemble (26.5%) doesn't improve over distil-large-v3 alone — weaker models dilute its advantage on files where it's clearly better.

All results use anti-hallucination flags and Sonnet adjudicator.

## Experiment History

### 1–9. Architecture and Prompt Evolution

Started with chunk-rewrite (LLM rewrites 500-word chunks), which introduced errors in uncontested regions. Refactored to targeted diff resolution: wdiff finds disagreements, cluster nearby diffs, LLM resolves only the disagreements. Tested with local qwen2.5:7b and 14b — both degraded results due to format leakage (7b) or poor choices (14b).

Key milestones:
- **Targeted diff resolution** replaced chunk-rewrite (exp 2)
- **`_clean_resolution()`** strips LLM format artifacts (exp 3)
- **A/B choice format** replaced text-echo — eliminates invented words, 100% parse rate (exp 10)
- **Bug fixes**: context indexed by wrong positions, clustering sort mismatch (exp 9)

| Variant | File 3 WER | Parse Rate | Notes |
|---------|-----------|-----------|-------|
| Chunk-rewrite | 30.9% | — | Boundary duplication, uncontested errors |
| 7b + text-echo | 29.1% | ~55% | Parsing failures masked as conservatism |
| 14b + text-echo | 31.3% | 98% | More intervention ≠ better |
| 14b + A/B format | 29.3% | 100% | Format solved, model quality the bottleneck |

### 10. Claude Sonnet API

Switched to Claude Sonnet 4 via API. Added checkpointing and retry-without-context for content refusals.

| File | medium | 14b merged | Sonnet merged | Gap to medium |
|------|--------|------------|---------------|---------------|
| 3 | 28.9% | 29.3% | **28.3%** | **-0.6pp** |
| 4 | 27.9% | 28.0% | **27.6%** | **-0.3pp** |
| 9 | 24.8% | 24.9% | **24.8%** | **0.0pp** |
| **Avg** | **27.2%** | **27.4%** | **26.9%** | **-0.3pp** |

**Key finding:** Model quality was the bottleneck. The diff resolution architecture and A/B format were necessary but not sufficient — a capable adjudicator was also required.

### 11. Whisper Large — Catastrophic Hallucination

Whisper large (mlx-whisper) on file 3 produced "The unremarkable." repeated 7,479 times (97% WER). Root cause: `condition_on_previous_text=True` (default) creates a feedback loop. Large models are most susceptible.

### 12. Anti-Hallucination Flags

Added flags to all Whisper runs: `condition_on_previous_text=False`, `no_speech_threshold=0.2`, `compression_ratio_threshold=2.0`, `hallucination_silence_threshold=3.0`.

| Metric | Before flags | After flags | Change |
|--------|-------------|-------------|--------|
| Medium avg | 27.2% | 27.7% | +0.5pp |
| 2-way merged avg | 26.9% | 26.6% | -0.3pp |
| Gap (merged vs medium) | -0.3pp | -1.1pp | -0.8pp |

The flags changed individual WERs slightly, but the ensemble gap widened from -0.3pp to -1.1pp. The flags appear to produce transcripts that differ in more meaningful ways, giving the adjudicator better signal.

### 13. distil-large-v3

Replaced whisper large with `mlx-community/distil-whisper-large-v3` — a distilled model 6x faster than large-v3, within 1% WER, and specifically optimized to reduce hallucinations. No catastrophic failures (only minor hallucination loops of 8–10 words, caught by existing collapsing logic).

**Standalone quality:** distil-large-v3 (26.3% avg) beats medium (27.7%) by 1.4pp and even beats the 2-way small+medium ensemble (26.6%) by 0.3pp.

**3-way ensemble (small + medium + distil-large-v3):**

| File | distil-large-v3 | 2-way merged | 3-way merged |
|------|----------------|--------------|--------------|
| 3 | 29.6% | **28.5%** | 29.4% |
| 4 | **27.6%** | 28.2% | 28.5% |
| 9 | **21.6%** | 23.1% | 21.7% |
| **Avg** | **26.3%** | 26.6% | 26.5% |

The 3-way ensemble doesn't improve over distil-large-v3 alone. On file 4, merging dragged 27.6% up to 28.5% by incorporating medium and small's worse readings. When one model is clearly better, ensembling with weaker models dilutes quality.

## Key Lessons

1. **Constrain the output format.** A/B choice eliminates format leakage and invented words.
2. **Model quality matters most.** Same architecture, same prompt — Sonnet succeeds where local models fail.
3. **Ensembling helps when models are comparable.** 2-way small+medium ensemble beats medium by 1.1pp. But ensembling a strong model with weaker ones can hurt.
4. **Anti-hallucination flags are essential.** They prevent catastrophic failures and improve ensemble signal.
5. **distil-large-v3 is the best single model.** Faster than large, no hallucination, better WER than medium.

## Next Steps

- Try 2-way ensemble: distil-large-v3 + medium (skip small, which is consistently weakest)
- Expand eval to more Rev16 files for statistical significance
- Test with a local adjudicator closer to Sonnet quality (e.g., Llama 3.3 70B)

## Environment

- Hardware: M4 Max, 64GB RAM
- LLM (local): Ollama qwen2.5:14b (9GB)
- LLM (API): Claude Sonnet 4 (claude-sonnet-4-20250514)
- Whisper models: small, medium, distil-large-v3 (mlx-community/distil-whisper-large-v3)
- Scoring: meeteval WER
