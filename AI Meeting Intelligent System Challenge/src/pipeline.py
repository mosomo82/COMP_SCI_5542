"""
pipeline.py — Full end-to-end orchestration

Runs all stages in sequence and returns a unified result dict.
Import this in app.py for the Gradio UI.
"""

import json
import os
import time
import datetime
from pathlib import Path
from src.transcribe import transcribe, save_transcript
from src.diarize    import diarize, format_diarized_transcript, save_diarized
from src.analyze    import analyze_sentiment, extract_keywords, save_analysis
from src.summarize  import summarize, format_summary_for_speech, save_summary
from src.speak      import synthesize_speech


def run_pipeline(
    audio_path: str,
    whisper_model: str = "small",
    prompt_variant: str = "improved",
    generate_audio: bool = True,
) -> dict:
    """
    Run the full Meeting Intelligence pipeline.

    Args:
        audio_path:      Path to meeting audio file
        whisper_model:   "small" (baseline) | "medium" (improved)
        prompt_variant:  "baseline" | "improved" (for evaluation)
        generate_audio:  Whether to run TTS on the summary

    Returns:
        {
            "transcript":        raw transcript string,
            "diarized":          list of speaker-labeled segments,
            "diarized_text":     formatted readable transcript,
            "keywords":          list of top keywords,
            "sentiment":         per-speaker sentiment dict,
            "summary":           structured summary dict,
            "summary_text":      natural language summary string,
            "audio_path":        path to voiced summary .wav (or None),
            "stage_times":       dict of latency per stage (seconds),
        }
    """
    times = {}
    result = {}

    print(f"\n{'='*50}")
    print(f"  Meeting Intelligence Pipeline")
    print(f"  Audio: {audio_path}")
    print(f"  Whisper: {whisper_model} | Prompt: {prompt_variant}")
    print(f"{'='*50}\n")

    # Stage 1: Transcription
    t0 = time.time()
    print("► Stage 1/5: Transcription")
    transcription = transcribe(audio_path, model_size=whisper_model)
    save_transcript(transcription)
    result["transcript"] = transcription["text"]
    times["transcription"] = round(time.time() - t0, 2)

    # Stage 2: Diarization
    t0 = time.time()
    print("\n► Stage 2/5: Speaker diarization")
    diarized = diarize(audio_path, transcription["segments"])
    save_diarized(diarized)
    result["diarized"] = diarized
    result["diarized_text"] = format_diarized_transcript(diarized)
    times["diarization"] = round(time.time() - t0, 2)

    # Stage 3: NLP Analysis
    t0 = time.time()
    print("\n► Stage 3/5: NLP analysis")
    sentiment = analyze_sentiment(diarized)
    keywords  = extract_keywords(transcription["text"])
    save_analysis(sentiment, keywords)
    result["sentiment"] = sentiment
    result["keywords"]  = keywords
    times["analysis"] = round(time.time() - t0, 2)

    # Stage 4: LLM Summary
    t0 = time.time()
    print("\n► Stage 4/5: LLM summarization")
    summary = summarize(
        diarized_transcript=result["diarized_text"],
        keywords=keywords,
        sentiment=sentiment,
        prompt_variant=prompt_variant,
    )
    save_summary(summary)
    result["summary"]      = summary
    result["summary_text"] = format_summary_for_speech(summary)
    times["summarization"] = round(time.time() - t0, 2)

    # Stage 5: TTS (optional — skip for fast iteration)
    if generate_audio:
        t0 = time.time()
        print("\n► Stage 5/5: Text-to-speech narration")
        audio_path_out = synthesize_speech(result["summary_text"])
        result["audio_path"] = audio_path_out
        times["tts"] = round(time.time() - t0, 2)
    else:
        result["audio_path"] = None
        print("\n► Stage 5/5: TTS skipped")

    result["stage_times"] = times
    total = round(sum(times.values()), 2)

    print(f"\n{'='*50}")
    print(f"  Pipeline complete in {total}s")
    for stage, t in times.items():
        print(f"    {stage:<20} {t}s")
    print(f"{'='*50}\n")

    _save_manifest(audio_path, whisper_model, prompt_variant, times)
    return result


def _save_manifest(audio_path: str, whisper_model: str, prompt_variant: str, stage_times: dict) -> None:
    manifest = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "input_file": Path(audio_path).name,
        "model_versions": {
            "whisper": whisper_model,
            "summarizer_primary": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            "summarizer_fallback": "claude-sonnet-4-20250514",
        },
        "prompt_variant": prompt_variant,
        "stage_latencies_sec": stage_times,
        "total_latency_sec": round(sum(stage_times.values()), 2),
    }
    out = Path("outputs/run_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Pipeline] Manifest saved → {out}")
