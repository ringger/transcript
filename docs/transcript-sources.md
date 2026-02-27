# Transcript Source Quality Notes

## YouTube Auto-Captions

YouTube generates automatic captions using Google's Cloud Speech-to-Text API, an
ASR system built on deep neural networks. The system identifies phonemes from the
audio, assembles them into words, then applies NLP for context, grammar, and
punctuation.

### Accuracy

- University of Minnesota research found YouTube auto-captions typically achieve
  **60-70% accuracy** overall.
- Accuracy varies significantly by content type. Clean studio audio with a single
  native English speaker can be much higher; noisy recordings, multiple speakers,
  accents, or technical jargon push accuracy much lower.
- YouTube's ASR has improved over time: neural networks (2015), transformer models
  (2019), LLM-based context (2022).

### Known Limitations

- **Technical terminology**: Domain-specific terms are frequently mangled
  (e.g., "dienophile" → "dinah file", "dyna file", etc.)
- **Proper nouns**: Names of people, companies, and products are often wrong
- **Filler words**: "um", false starts, and repetitions are transcribed literally
- **Punctuation**: Older videos may lack punctuation entirely; newer ones are
  better but still imperfect
- **No speaker labels**: Auto-captions don't identify speakers

### Creator-Uploaded vs Auto-Generated

YouTube captions may be auto-generated or manually uploaded by the creator.
The pipeline downloads whatever is available via yt-dlp. Creator-uploaded
captions are typically much higher quality but less common. The VTT file
metadata doesn't always make it obvious which type you have.

## Whisper (via mlx-whisper)

The pipeline runs multiple Whisper model sizes and ensembles them via LLM
adjudication. See [ensemble-experiments.md](ensemble-experiments.md) for
comparative WER data.

### Model Characteristics

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| small | Fast | Lower | Good for ensemble diversity |
| medium | Moderate | Good | Solid mid-range |
| distil-large-v3 | ~6x faster than large-v3 | Best single-model WER | Recommended as base transcript |
| large-v3 | Slow | Unreliable | Catastrophic hallucination on long audio (2h+); not used |

### Known Limitations

- **Hallucination on silence**: Without anti-hallucination flags, Whisper can
  generate repetitive phantom text during silent passages
- **Long audio**: large-v3 is especially prone to hallucination on 2h+ recordings
- The pipeline applies anti-hallucination flags to all models:
  `condition_on_previous_text=False`, `no_speech_threshold=0.2`,
  `compression_ratio_threshold=2.0`, `hallucination_silence_threshold=3.0`

## Multi-Source Merge

The pipeline's merge step aligns YouTube captions and Whisper output via wdiff,
then uses an LLM to resolve differences chunk by chunk. The `analysis.md` output
reports per-source coverage and retention for each run.

In practice, when audio quality is high (single speaker, studio recording),
YouTube and Whisper agree closely (97-99% overlap). The merge adds the most
value on noisy or technical content where the sources diverge significantly.

## References

- [Use automatic captioning - YouTube Help](https://support.google.com/youtube/answer/6373554?hl=en)
- [The Technology Behind YouTube's Auto-Captioning System - DEV Community](https://dev.to/activejack/the-technology-behind-youtubes-auto-captioning-system-1o45)
- [Fixing YouTube's automatic captioning with AI - Tech, Chem, Scott](https://blog.hartleygroup.org/2025/12/05/fixing-youtubes-automatic-captioning-with-ai/)
