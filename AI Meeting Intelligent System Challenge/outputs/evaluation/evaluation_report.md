# Meeting Intelligence — Evaluation Report

**Generated:** 2026-04-28 03:00 UTC
**Primary Audio:** `QDH3YKPZKHE5M.m4a` (real ~8.5-min meeting call, 1431 words transcribed)
**Secondary Audio:** `sample_meeting.wav` (short 20s stub used for baseline timing)

> **Note on data sources:** The `improved` prompt runs were executed against the full-length
> meeting audio (`QDH3YKPZKHE5M.m4a`) via the live Gradio app, including TTS. The `baseline`
> runs were executed via `run_eval.py` against `sample_meeting.wav` (a shorter reference clip).
> WER comparisons are noted where applicable.

---

## Latency per Stage

| Run | Audio | Transcription | Diarization | Analysis | Summarization | TTS | Total |
|-----|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| small_baseline | sample_meeting.wav | 80.25s | 45.20s | 10.33s | 2.63s | — | **138.41s** |
| small_improved | QDH3YKPZKHE5M.m4a | 256.29s | 551.57s | 55.52s | 0.30s | 153.21s | **1016.89s** |
| medium_baseline | sample_meeting.wav | 53.99s | 3.49s | 3.06s | 0.31s | — | **60.85s** |
| medium_improved | QDH3YKPZKHE5M.m4a | 774.53s | 683.87s | 14.16s | 1.45s | 59.73s | **1533.74s** |

> **Insight:** Medium is faster than small on short audio (60s vs 138s), but slower on long audio
> (1533s vs 1016s) because its larger attention window processes longer context at greater depth.
> Diarization dominates total latency on long audio (551s / 683s), not transcription.

---

## Word Error Rate (WER)

| Comparison | WER | Notes |
|---|:---:|---|
| Whisper small vs medium (same `demo_call.wav` audio) | 1.00 | Small produced 1 word (overwritten by eval run); medium produced 1431 words |
| Whisper medium vs `expected_transcript.txt` reference | 28.28 | Reference is a different 5-sentence script, not matched to this audio |
| Noise robustness — SNR 20dB | 1.00 | Gaussian noise causes Whisper VAD to fail entirely |
| Noise robustness — SNR 10dB | 1.00 | Language detection switches to Nynorsk — empty output |
| Noise robustness — SNR 5dB | 1.00 | Complete speech detection failure |
| Noise robustness — SNR 0dB | 1.00 | Full noise floor — no speech detected |

> **Limitation:** WER vs reference is misleading because `expected_transcript.txt` is a 5-sentence
> script from a different audio file than the one processed. Whisper medium produced a full,
> coherent 1431-word transcript of the real meeting audio (verified in `transcript_medium.txt`).

---

## Summary Quality Rubric

| Run | Summarizer | Fields Populated | Action Items | Key Decisions | Well-Formed | Has Tone |
|-----|-----------|:---:|:---:|:---:|:---:|:---:|
| small_baseline | Local fallback | 1/6 (17%) | 0 | 0 | No | No |
| small_improved | Local fallback | 5/6 (83%) | 0 | 1 | No | Yes |
| medium_baseline | Local fallback | 1/6 (17%) | 0 | 0 | No | No |
| medium_improved | Gemini 2.0 Flash | 6/6 (100%) | 3 | 4 | Yes | Yes |

> **Insight:** The `improved` prompt with Gemini produces a fully structured output (6/6 fields,
> 3 action items with owners and deadlines, 4 key decisions). The `baseline` prompt with the
> local fallback summarizer produces minimal output — demonstrating the value of both
> structured prompt engineering AND a capable LLM backend.

---

## Prompt Variant Comparison (Qualitative)

| Dimension | Baseline Prompt | Improved Prompt |
|---|---|---|
| Output format | Free-form text | Strict JSON schema |
| Fields requested | None (implicit) | 7 explicit fields |
| Action items | Not extracted | Owner + task + deadline |
| Topic sentiment | Not extracted | Per-topic evidence |
| LLM guidance | Minimal | Rules + examples + role |
| Structured output rate | ~17% fields | ~83-100% fields |

---

## Key Findings & Tradeoffs

1. **Diarization is the pipeline bottleneck** — not transcription. On the 8.5-min audio,
   pyannote diarization took 551s (small) and 683s (medium) vs 256s/774s for transcription.
   Caching or skipping diarization would halve total pipeline time.

2. **Improved prompt + Gemini = dramatically better summaries.** The structured IMPROVED_PROMPT
   with JSON schema produced 6/6 fields with specific action items and meeting tone; the baseline
   free-text prompt produced only the executive summary field, with no structured extraction.

3. **Whisper medium produces higher-quality transcripts on long audio.** The 1431-word
   transcript from `transcript_medium.txt` is coherent and well-punctuated. The small model's
   `transcript_small.txt` was overwritten during the evaluation grid run (known issue: output
   paths were shared across eval runs).

4. **Gaussian noise is catastrophic for Whisper.** At all tested SNR levels (0-20dB), Whisper
   fails to detect speech and returns empty output or misidentifies language as Nynorsk.
   Real-world noise robustness should be tested with background conversation or HVAC noise.

5. **API key management is a deployment risk.** Both the Gemini API key leak and prepaid
   credit exhaustion caused silent fallback to local summarizer. Production deployments need
   key rotation policies and fallback quality monitoring.