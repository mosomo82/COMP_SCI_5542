# Submission Checklist (ScopeOfChallenge)

## 1. Working System (Code)

- [x] Speech/audio input accepted (.mp3/.wav/.m4a) in app
- [x] Pretrained foundation models integrated
- [x] End-to-end pipeline runs: transcription -> diarization -> analysis -> summary -> TTS
- [ ] At least 1 real sample run saved in outputs/

Required output artifacts to generate before final submission:
- [ ] outputs/transcript.txt
- [ ] outputs/diarized.json
- [ ] outputs/sentiment.json
- [ ] outputs/summary.json
- [ ] outputs/summary_audio.wav
- [ ] outputs/latency_chart.png

## 2. Evaluation (Required)

- [x] Baseline vs improved design implemented
- [x] WER function implemented for transcription comparison
- [x] Prompt comparison rubric included
- [x] Noise robustness experiment included
- [ ] Run evaluation and capture final numeric results table
- [ ] Add 3-5 key insights from failures and tradeoffs

## 3. Presentation Deck (Minimum 10 Slides)

Ensure deck includes all required sections:
- [ ] Problem description
- [ ] Why interesting / business value
- [ ] Dataset or inputs used
- [ ] Models used
- [ ] Pipeline architecture
- [ ] Prompt/input design
- [ ] Results
- [ ] Evaluation
- [ ] Demo video link
- [ ] GitHub link
- [ ] Limitations
- [ ] AI tools used disclosure

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
