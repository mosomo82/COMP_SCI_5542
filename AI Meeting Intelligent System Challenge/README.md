# Meeting Intelligence System

An AI-powered pipeline that transforms raw meeting audio into structured summaries, action items, and sentiment insights — with a voiced narration output.

## Pipeline Overview

```
Audio Input → Whisper (STT) → Speaker Diarization → Sentiment + Keywords → LLM Summary → SpeechT5 (TTS) → Gradio UI
```

## Project Structure

```
meeting-intelligence/
├── src/
│   ├── transcribe.py        # Whisper speech-to-text
│   ├── diarize.py           # Speaker diarization (pyannote)
│   ├── analyze.py           # Sentiment + keyword extraction
│   ├── summarize.py         # LLM summary + action items
│   ├── speak.py             # SpeechT5 text-to-speech
│   └── pipeline.py          # Full end-to-end orchestration
├── app.py                   # Gradio UI entry point
├── audio_samples/           # Test audio files
├── outputs/                 # Generated transcripts, summaries, audio
├── notebooks/
│   └── evaluation.ipynb     # Baseline vs improved comparisons
├── tests/
│   └── test_pipeline.py     # Basic smoke tests
├── requirements.txt
├── .env.example
├── SUBMISSION_CHECKLIST.md
├── BONUS_OPTIONS.md
├── ENHANCEMENTS.md
└── README.md
```

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/meeting-intelligence
cd meeting-intelligence
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your keys:
#   GEMINI_API_KEY=...
#   GEMINI_MODEL=gemini-1.5-pro
#   ANTHROPIC_API_KEY=...    (optional fallback)
#   HF_TOKEN=hf_...          (required for pyannote diarization)
```

Windows PowerShell alternative:

```powershell
Copy-Item .env.example .env
```

### 3. Run the app

```bash
python app.py
# Opens at http://localhost:7860
```

## Models Used

| Task | Model | Source |
|------|-------|--------|
| Speech-to-text | `openai/whisper-small` | Hugging Face |
| Speaker diarization | `pyannote/speaker-diarization-3.1` | Hugging Face |
| Sentiment analysis | `distilbert-base-uncased-finetuned-sst-2-english` | Hugging Face |
| Keyword extraction | KeyBERT | PyPI |
| Summarization | Gemini Pro (preferred) / Claude (fallback) | Google / Anthropic |
| Text-to-speech | `microsoft/speecht5_tts` | Hugging Face |

## Evaluation

See `notebooks/evaluation.ipynb` for:
- Whisper small vs medium transcription accuracy (WER)
- Summary quality comparison across prompt variants
- Sentiment usefulness on noisy vs clean audio
- Latency benchmarks per pipeline stage

## Sample Outputs

After running a meeting clip, `outputs/` will contain:
- `transcript.txt` — raw timestamped transcript
- `diarized.json` — speaker-labeled segments
- `summary.json` — structured summary + action items
- `sentiment.json` — per-speaker sentiment scores
- `summary_audio.wav` — voiced narration of the summary

## AI Tools Used

- **Anthropic Claude / GitHub Copilot (GPT-5.3-Codex)** — code scaffolding, prompt engineering, README drafting
- **Hugging Face** — model hosting and inference
- **Gradio** — UI framework

## Challenge Alignment Snapshot

| Scope requirement | Status | Evidence |
|---|---|---|
| Pretrained foundation model(s) | Done | Whisper, pyannote, DistilBERT, SpeechT5 in `src/` and `requirements.txt` |
| Prompt / input engineering | Done | baseline vs improved prompts in `src/summarize.py` |
| Evaluation baseline vs improved | In progress | `notesbooks/evaluation.ipynb` contains WER/prompt/noise/latency workflow |
| Working demo pipeline | Done | end-to-end `src/pipeline.py` + `app.py` |
| GitHub repo with setup and outputs | Done | setup is present; sample outputs have been successfully generated into `outputs/` |
| AI tools disclosure | Done | this section + `SUBMISSION_CHECKLIST.md` |
| 1-2 minute demo video | Pending | add final link in `SUBMISSION_CHECKLIST.md` |
| 10-slide presentation | Pending | prepare and link in `SUBMISSION_CHECKLIST.md` |

Use these companion docs before submission:
- `SUBMISSION_CHECKLIST.md`
- `BONUS_OPTIONS.md`
- `ENHANCEMENTS.md`

## Limitations

- Whisper small struggles with heavy accents and 3+ overlapping speakers
- Diarization accuracy drops below ~10s speaker turns
- SpeechT5 voice quality is robotic compared to commercial TTS
- No real-time streaming — processes full file only
