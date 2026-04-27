# Enhancement Recommendations

## Short-Term (Finish Before Submission)

1. Add deterministic fallback when ANTHROPIC_API_KEY is missing.
- Current behavior can fail hard in summarization.
- Add local fallback summarizer (extractive heuristic) to keep demo resilient.

2. Save full run manifest per execution.
- Create outputs/run_manifest.json with:
  - input file name
  - model versions
  - stage latencies
  - prompt variant
  - timestamp

3. Add one-click evaluation runner.
- Wrap evaluation script into a CLI command so you can regenerate metrics quickly.

4. Add sample assets.
- Include 1 short anonymized audio sample and expected output JSONs.

## Mid-Term (Higher Quality)

1. Improve sentiment quality for meetings.
- Current SST-2 model is not meeting-domain specific.
- Consider a dialogue-aware sentiment model or sentence-level aggregation.

2. Better diarization alignment.
- Add overlap-handling and short-turn smoothing to reduce speaker flips.

3. Improve summary robustness.
- Add JSON schema validation + retry loop for malformed LLM output.

4. Caching for speed.
- Cache Whisper and SpeechT5 models to avoid reload on each run.

## Presentation-Ready Metrics to Add

- WER (small vs medium)
- End-to-end latency per stage
- Prompt quality rubric scores
- Noise robustness curve (SNR vs WER)
- Action-item precision on a labeled mini-set

## Risks to Mention Transparently

- API dependency for summarization
- Diarization degrades on overlap and short turns
- TTS quality less natural than premium engines
- Accuracy drop in noisy recordings and heavy accents
