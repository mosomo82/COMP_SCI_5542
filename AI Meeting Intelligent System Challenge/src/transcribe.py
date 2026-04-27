"""
transcribe.py — Speech-to-text using OpenAI Whisper

Evaluation hook: pass model_size="medium" to compare against "small" baseline.
"""

import whisper
import torch
from pathlib import Path


_WHISPER_MODEL_CACHE: dict[tuple[str, str], object] = {}


def transcribe(audio_path: str, model_size: str = "small") -> dict:
    """
    Transcribe an audio file using Whisper.

    Args:
        audio_path: Path to .wav / .mp3 / .m4a file
        model_size: "tiny" | "small" | "medium" | "large"
                    Use "small" as baseline, "medium" for improved eval.

    Returns:
        {
            "text": full transcript string,
            "segments": [{"start", "end", "text"}, ...],
            "language": detected language code
        }
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_whisper_model(model_size, device)

    result = model.transcribe(
        audio_path,
        verbose=False,
        word_timestamps=True,   # needed for diarization alignment
        fp16=(device == "cuda"),
    )

    print(f"[Whisper] Detected language: {result['language']}")
    print(f"[Whisper] Transcript length: {len(result['text'].split())} words")

    return {
        "text": result["text"].strip(),
        "segments": result["segments"],
        "language": result["language"],
    }


def _get_whisper_model(model_size: str, device: str):
    key = (model_size, device)
    if key not in _WHISPER_MODEL_CACHE:
        print(f"[Whisper] Loading {model_size} on {device}...")
        _WHISPER_MODEL_CACHE[key] = whisper.load_model(model_size, device=device)
    else:
        print(f"[Whisper] Using cached {model_size} model on {device}.")
    return _WHISPER_MODEL_CACHE[key]


def save_transcript(result: dict, output_path: str = "outputs/transcript.txt") -> None:
    """Save timestamped transcript to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            start = f"{seg['start']:.1f}s"
            end   = f"{seg['end']:.1f}s"
            f.write(f"[{start} → {end}] {seg['text'].strip()}\n")
    print(f"[Whisper] Transcript saved → {output_path}")


if __name__ == "__main__":
    # Quick smoke test — swap in any .wav file
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "audio_samples/sample.wav"
    result = transcribe(path)
    print("\n--- Transcript ---")
    print(result["text"][:500], "...")
    save_transcript(result)
