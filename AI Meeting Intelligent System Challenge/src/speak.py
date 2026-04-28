"""
speak.py — Text-to-speech using Microsoft SpeechT5

Converts the meeting summary text into a voiced .wav file.
This is the multimodal bonus: speech in → speech out.
"""

import os
import numpy as np
from pathlib import Path


_TTS_COMPONENT_CACHE: dict[str, dict] = {}


def synthesize_speech(text: str, output_path: str = "outputs/summary_audio.wav") -> str:
    """
    Convert text to speech using SpeechT5 and save as .wav.

    Args:
        text: The text to narrate (use format_summary_for_speech() output)
        output_path: Where to save the .wav file

    Returns:
        Path to saved audio file
    """
    import torch
    import soundfile as sf
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = _get_tts_bundle(device)
    processor = bundle["processor"]
    model = bundle["model"]
    vocoder = bundle["vocoder"]
    speaker_embeddings = bundle["speaker_embeddings"]

    # SpeechT5 has a 600-token limit — chunk long text
    chunks = _chunk_text(text, max_chars=500)
    print(f"[TTS] Synthesizing {len(chunks)} chunk(s)...")

    all_audio = []
    silence = np.zeros(int(0.3 * 16000))  # 0.3s pause between chunks

    for i, chunk in enumerate(chunks):
        print(f"[TTS] Chunk {i+1}/{len(chunks)}: {chunk[:60]}...")
        inputs = processor(text=chunk, return_tensors="pt").to(device)

        with torch.no_grad():
            speech = model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=vocoder,
            )

        all_audio.append(speech.cpu().numpy())
        if i < len(chunks) - 1:
            all_audio.append(silence)

    # Concatenate and save
    final_audio = np.concatenate(all_audio)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, final_audio, samplerate=16000)

    duration = len(final_audio) / 16000
    print(f"[TTS] Audio saved → {output_path} ({duration:.1f}s)")
    return output_path


def _get_tts_bundle(device: str) -> dict:
    import torch
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

    if device in _TTS_COMPONENT_CACHE:
        print(f"[TTS] Using cached SpeechT5 bundle on {device}.")
        return _TTS_COMPONENT_CACHE[device]

    print(f"[TTS] Loading SpeechT5 bundle on {device}...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    print("[TTS] Loading speaker embeddings...")
    # datasets 3.x dropped legacy script support for this dataset — load parquet directly
    import pandas as pd, requests, io as _io
    _url = (
        "https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors"
        "/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
    )
    _resp = requests.get(_url)
    _resp.raise_for_status()
    _df = pd.read_parquet(_io.BytesIO(_resp.content))
    speaker_embeddings = torch.tensor(
        list(_df.iloc[7306]["xvector"])  # slt speaker
    ).unsqueeze(0).to(device)

    bundle = {
        "processor": processor,
        "model": model,
        "vocoder": vocoder,
        "speaker_embeddings": speaker_embeddings,
    }
    _TTS_COMPONENT_CACHE[device] = bundle
    return bundle


def _chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Split text at sentence boundaries to stay within model token limit."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:max_chars]]
