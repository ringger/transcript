# Rev16 Test1: Whisper Ensemble Experiment Results

## Dataset

Three files from Rev16 evaluated against reference transcripts using meeteval cpWER.

## Baseline: Individual Whisper Models

| File | whisper_small | whisper_medium |
|------|--------------|----------------|
| 3    | 29.8%        | 28.9%          |
| 4    | 36.0%        | 27.9%          |
| 9    | 25.1%        | 24.8%          |
| **Avg** | **30.3%**  | **27.2%**      |

## Current Best Results (Sonnet + A/B format + anti-hallucination flags)

| File | small | medium | merged | Gap to medium |
|------|-------|--------|--------|---------------|
| 3 | 31.5% | 29.5% | **28.5%** | -1.0pp |
| 4 | 31.3% | 29.2% | **28.2%** | -1.0pp |
| 9 | 24.1% | 24.5% | **23.1%** | -1.4pp |
| **Avg** | **28.9%** | **27.7%** | **26.6%** | **-1.1pp** |

The ensemble beats medium by 1.1pp on average, consistent across all 3 files.

## Experiment History

### 1. Chunk-Rewrite Approach (original)

Split base (medium) transcript into ~500-word chunks, sent each with the other transcript for LLM rewriting. Problems: chunk-boundary duplication, errors in uncontested regions, wasted tokens, base-model bias.

**File 3 WER:** 30.9%

### 2. Targeted Diff Resolution (refactor)

Replaced chunk-rewrite with surgical approach: wdiff to find disagreements, cluster nearby diffs, LLM resolves only the disagreements, apply targeted replacements. Anonymous "Model A" / "Model B" labels.

**File 3 WER:** 29.6%

### 3. Format Leakage Fix

Local qwen2.5:7b was leaking prompt formatting ("Model A", quoted phrases, "Decision:" prefixes) into output. Added `_clean_resolution()` to strip artifacts.

**Avg WER:** 28.0% → 27.5% (-0.5pp)

### 4–7. Prompt Tuning (all counterproductive with 7b)

Tested: fixing a_pos→b_pos context alignment, "prefer Model B" variants with different label orderings. All performed worse than the anonymous baseline. The 7b model's apparent conservatism was later revealed to be parsing failure (~45% of responses unparsed, defaulting to base text).

| Variant | File 3 WER | Parse Rate | Notes |
|---------|-----------|-----------|-------|
| Baseline (anonymous) | 29.1% | ~55% | Best with 7b |
| a_pos→b_pos fix | 29.5% | ~55% | Misalignment helped accidentally |
| "Prefer Model B" variants | 28.9–29.9% | 0–100% | Positional bias dominated |

### 8. qwen2.5:14b (text-echo format)

Larger model found more diffs (1797 vs ~988) and resolved more (98% parse rate), but made worse choices. More intervention ≠ better intervention with unconstrained text output.

**File 3 WER:** 31.3% (+2.4pp vs medium)

### 9. Bug Fixes (a_pos→b_pos, clustering sort)

Code audit revealed: context indexed by wrong positions, clustering/application sort mismatch. Fixed both. The b_pos fix alone didn't improve results (31.4%).

### 10. A/B Choice Format + Few-Shot Example

Key architectural change: LLM outputs "A" or "B" instead of echoing text. Eliminates format leakage, prevents inventing words, achieves 100% parse rate.

| Variant | File 3 WER | Parse Rate | Gap to medium |
|---------|-----------|-----------|---------------|
| 7b + text-echo (old) | 29.1% | ~55% | +0.2pp |
| 14b + text-echo | 31.3% | 98% | +2.4pp |
| **14b + A/B format** | **29.3%** | **100%** | **+0.4pp** |

Full 14b + A/B results: avg 27.4% (+0.2pp vs medium). Still worse than medium alone.

### 11. Claude Sonnet API

Switched to Claude Sonnet 4 via API. Added ensemble checkpointing (`ensemble_chunks/`) and retry-without-context for content refusals (Sonnet refused clusters containing offensive language; stripping context recovered most).

| File | medium | 14b merged | Sonnet merged | Gap to medium |
|------|--------|------------|---------------|---------------|
| 3 | 28.9% | 29.3% | **28.3%** | **-0.6pp** |
| 4 | 27.9% | 28.0% | **27.6%** | **-0.3pp** |
| 9 | 24.8% | 24.9% | **24.8%** | **0.0pp** |
| **Avg** | **27.2%** | **27.4%** | **26.9%** | **-0.3pp** |

**Key finding:** Model quality was the bottleneck. The diff resolution architecture and A/B format were necessary but not sufficient — a capable adjudicator model was also required.

### 12. Whisper Large — Catastrophic Hallucination

Added whisper large (via mlx-whisper) as a third model for 3-way adjudication with majority-vote signal. Extended the diff resolution code to merge pairwise diffs and present anonymous A/B/C choices.

On file 3, whisper large produced 16,160 words — but 14,958 of those were the phrase "The unremarkable." repeated 7,479 times. After hallucination loop collapsing, only 1,198 words remained (vs ~24,000 from small/medium). The model effectively stopped transcribing after the first few minutes.

| File | small | medium | large | 3-way merged | 2-way merged |
|------|-------|--------|-------|--------------|--------------|
| 3 | 29.8% | 28.9% | 97.0% | 29.0% | **28.3%** |

Using large as base (the default, since it's the "largest" model) dragged the merged result to 29.0% — worse than medium alone. The 3-way code works correctly, but garbage-in-garbage-out: a catastrophically broken base model poisons the ensemble.

**Conclusion:** mlx-whisper large is unreliable on long audio (2+ hours). Staying with 2-way (small + medium) until a more robust large model is available. The 3-way adjudication code remains in place for future use.

### 13. Anti-Hallucination Flags

Added Whisper anti-hallucination flags to address the large model failure from experiment 12. These are applied to all model sizes:

- `--condition-on-previous-text False` — breaks the feedback loop where one hallucinated window cascades into all subsequent windows
- `--no-speech-threshold 0.2` — raises the bar for detecting speech vs silence
- `--compression-ratio-threshold 2.0` — flags repetitive/compressed output
- `--hallucination-silence-threshold 3.0` — detects hallucination during silent segments (requires `--word-timestamps True`)

Re-ran the full 3-file eval with small + medium (2-way ensemble, Sonnet adjudicator) using the new flags.

| File | small | medium | merged | Gap to medium |
|------|-------|--------|--------|---------------|
| 3 | 31.5% | 29.5% | **28.5%** | -1.0pp |
| 4 | 31.3% | 29.2% | **28.2%** | -1.0pp |
| 9 | 24.1% | 24.5% | **23.1%** | -1.4pp |
| **Avg** | **28.9%** | **27.7%** | **26.6%** | **-1.1pp** |

Comparison with pre-antihalluc results (experiment 11):

| Metric | Before (test1) | After (antihalluc) | Change |
|--------|---------------|-------------------|--------|
| Medium avg | 27.2% | 27.7% | +0.5pp |
| Merged avg | 26.9% | 26.6% | -0.3pp |
| Gap (merged vs medium) | -0.3pp | -1.1pp | -0.8pp |

**Key finding:** The anti-hallucination flags slightly changed individual model WERs (some up, some down), but the ensemble's gap over medium widened dramatically from -0.3pp to -1.1pp. The merged result now consistently and substantially beats medium on every file. The flags appear to produce transcripts that differ in more meaningful ways, giving the adjudicator better signal.

## Key Lessons

1. **Constrain the output format.** Text-echo lets the LLM invent words and leak formatting. A/B choice is strictly better.
2. **Parse rate ≠ quality.** The 7b model's 55% parse rate accidentally preserved base text; the 14b's 98% parse rate actively degraded it.
3. **Model quality matters most.** Same architecture, same prompt — Sonnet beats medium by 0.3pp while 14b loses by 0.2pp.
4. **Audit your baselines.** The 7b "conservatism" was a parsing bug, not a feature.

## Next Steps

- Re-test whisper large with anti-hallucination flags for 3-way adjudication
- Expand eval to more Rev16 files for statistical significance
- Test with a local model closer to Sonnet quality (e.g., Llama 3.3 70B via Ollama)

## Environment

- Hardware: M4 Max, 64GB RAM
- LLM (local): Ollama qwen2.5:14b (9GB)
- LLM (API): Claude Sonnet 4 (claude-sonnet-4-20250514)
- Whisper models: small, medium
- Scoring: meeteval cpWER
