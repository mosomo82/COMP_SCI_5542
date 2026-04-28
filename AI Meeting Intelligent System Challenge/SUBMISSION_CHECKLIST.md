# Submission Checklist (ScopeOfChallenge)

## 1. Working System (Code)

- [x] Speech/audio input accepted (.mp3/.wav/.m4a) in app
- [x] Pretrained foundation models integrated
- [x] End-to-end pipeline runs: transcription -> diarization -> analysis -> summary -> TTS
- [x] At least 1 real sample run saved in outputs/

Required output artifacts to generate before final submission:
- [x] outputs/transcript.txt
- [x] outputs/diarized.json
- [x] outputs/sentiment.json
- [x] outputs/summary.json
- [x] outputs/summary_audio.wav
- [x] outputs/latency_chart.png (Included as table in evaluation_report.md)

## 2. Evaluation (Required)

- [x] Baseline vs improved design implemented
- [x] WER function implemented for transcription comparison
- [x] Prompt comparison rubric included
- [x] Noise robustness experiment included
- [x] Run evaluation and capture final numeric results table
- [x] Add 3-5 key insights from failures and tradeoffs (See evaluation_report.md)
- [x] Basic smoke tests passing (pytest tests/ -v)

## 3. Presentation Deck (Minimum 10 Slides)

Ensure deck includes all required sections:
- [x] Problem description
- [x] Why interesting / business value
- [x] Dataset or inputs used
- [x] Models used
- [x] Pipeline architecture
- [x] Prompt/input design
- [x] Results
- [x] Evaluation
- [ ] Demo video link
- [x] GitHub link
- [x] Limitations
- [x] AI tools used disclosure

Deck link:
- [ ] Add final URL here:

## 4. Demo Video (1-2 Minutes)

Show:
- [ ] Input sample
- [ ] Model running
- [ ] Output results
- [ ] Key findings

Video link:
- [ ] Add final URL here:

## 5. AI Tools Disclosure (Required)

Document explicitly:
- [x] Which tools were used
- [x] How they were used
- [x] What parts were AI-assisted or AI-generated

Suggested statement template:
"This project used GitHub Copilot and Claude for code scaffolding, debugging suggestions, prompt refinement, and documentation drafting. Final integration, testing, experiment design, and result interpretation were completed and verified by the student."
