# Bonus Options (Extra Credit)

These options align with the challenge bonus theme: multimodal chaining.

## Option A: Meeting Translation + Voice Replay

Flow:
- speech -> text (Whisper)
- text -> translated summary (LLM)
- translated text -> speech (SpeechT5)

Deliverables:
- One English input and one non-English output audio
- Side-by-side quality notes (accuracy and naturalness)

## Option B: Speaker-Level Action Item Voice Notes

Flow:
- diarized transcript -> action item extraction per speaker
- per-speaker action items -> separate audio snippets

Deliverables:
- action_items_by_speaker.json
- one audio file per speaker action list

Why this can score bonus:
- shows deeper use of diarization + TTS, not just single summary narration

## Option C: Audio + Visual Executive Brief

Flow:
- summary JSON -> auto-generated 1-page visual brief (PNG/PDF)
- optional voice-over audio attached

Deliverables:
- outputs/executive_brief.png
- outputs/summary_audio.wav

## Option D: Real-Time Chunked Meeting Assistant

Flow:
- process audio in chunks every N seconds
- rolling summary and rolling action items update live

Deliverables:
- latency comparison: full-file vs chunked mode
- screenshot/video showing live update

## Option E: Confidence-Aware Summaries

Flow:
- use Whisper confidence/probability signals
- mark low-confidence transcript regions
- constrain LLM to avoid uncertain spans

Deliverables:
- summary_with_confidence.json
- reduction in hallucination examples

## Easiest High-Impact Bonus Path

Implement Option B first.
- It reuses your current pipeline.
- It is demonstrably multimodal.
- It creates clear, gradable artifacts with low additional engineering cost.
