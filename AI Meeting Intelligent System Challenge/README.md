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
git clone https://github.com/mosomo82/COMP_SCI_5542
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

## Tools & AI Used

### Core Tools

| Tool | Purpose | How It Was Used | Contribution | Limitations |
|------|---------|-----------------|--------------|-------------|
| [Visual Studio Code](https://code.visualstudio.com/) | Primary development environment | Used to edit Python source files, Gradio UI, and evaluation scripts; managed local venv and terminal execution | Central workspace for implementation, pipeline orchestration, and debugging | Editor only; depends on local CPU/GPU for model execution |
| [Google Colab](https://colab.research.google.com/) | Remote API runtime | Used to host the `colab_api_server.ipynb` for testing pipeline execution on remote GPU resources | Enabled testing of transcription and diarization models without local dependency conflicts | Session timeouts and ngrok tunnel dependency |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | Model inference framework | Used to load and execute Whisper, RoBERTa, and SpeechT5 models | Core library for transcription, sentiment analysis, and text-to-speech stages | Heavy memory/compute requirements for local inference |
| [Gradio](https://www.gradio.app/) | Interactive Web UI | Used to create the main user interface for audio uploading and result visualization | Provided a user-friendly way to interact with the full ML pipeline | Limited customization for complex data visualization compared to custom React frontends |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | Speaker Diarization | Used to identify and separate different speakers in the meeting audio | Essential for producing structured, speaker-labeled transcripts | Requires Hugging Face token and model acceptance; compute intensive |
| [Gemini API](https://ai.google.dev/) | LLM Summarization | Used to transform raw transcripts into structured JSON summaries and action items | Provided high-quality reasoning for baseline vs improved prompt evaluation | Dependent on API availability, latency, and token credit limits |

### AI Assistance Disclosure

| Tool | Purpose | How It Assisted | Contribution to This Project | Limitations / Human Verification |
|------|---------|-----------------|------------------------------|----------------------------------|
| [Claude Code](https://www.anthropic.com/claude-code) | Coding workflow assistance | Used to help implement pipeline updates, evaluator/reporting changes, notebook support, and documentation | Accelerated code drafting, refactoring, and integration of the final evaluation framework | All generated code was reviewed, edited, and validated before acceptance |
| [ChatGPT](https://chat.openai.com) | Prompt and writing assistance | Used to brainstorm prompt phrasing, structure README explanations, and articulate failure-case descriptions | Helped refine structured prompt wording and supporting written explanations | Suggestions were treated as drafts and checked against actual project behavior |
| [Antigravity AI](https://antigravity.ai) | Code/documentation assistance | Used for ideation around code organization, terminal troubleshooting, and documentation improvements | Supported project structuring and technical explanation formatting | Output required manual verification for pathing and environment compatibility |

### Human Responsibility Statement

- Final model selection, prompt strategy, evaluation design, and code integration decisions were made manually.
- AI tools were used as assistants for drafting, debugging, and documentation support, not as autonomous substitutes for implementation review.
- All reported results, metrics, and submission artifacts were checked against the actual repository outputs before inclusion.

## Challenge Alignment Snapshot

| Scope requirement | Status | Evidence |
|---|---|---|
| Pretrained foundation model(s) | Done | Whisper, pyannote, DistilBERT, SpeechT5 in `src/` and `requirements.txt` |
| Prompt / input engineering | Done | baseline vs improved prompts in `src/summarize.py` |
| Evaluation baseline vs improved | Done | `outputs/evaluation/evaluation_report.md` contains WER/prompt/noise/latency workflow |
| Working demo pipeline | Done | end-to-end `src/pipeline.py` + `app.py` |
| GitHub repo with setup and outputs | Done | setup is present; specific sample outputs (transcript, diarized, summary, sentiment, audio) are in `outputs/` |
| AI tools disclosure | Done | this section + `SUBMISSION_CHECKLIST.md` |
| 1-2 minute demo video | Done | [YouTube Link](https://youtu.be/oL2RsNLcwUo) |
| 10-slide presentation | Done | [GitHub PPTX Link](https://github.com/mosomo82/COMP_SCI_5542/blob/main/AI%20Meeting%20Intelligent%20System%20Challenge/slides/UMKC_COMP_SCI_5542_Presentation_Meeting_Intelligence.pptx) |


## Limitations

- Whisper small struggles with heavy accents and 3+ overlapping speakers
- Diarization accuracy drops below ~10s speaker turns
- SpeechT5 voice quality is robotic compared to commercial TTS
- No real-time streaming — processes full file only
