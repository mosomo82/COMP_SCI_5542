"""
diarize.py — Speaker diarization using pyannote.audio

Labels each transcript segment with a speaker ID (SPEAKER_00, SPEAKER_01, ...).
Requires HF_TOKEN in .env and accepting model terms at:
  https://huggingface.co/pyannote/speaker-diarization-3.1
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


SHORT_TURN_SMOOTH_SEC = 1.2
AMBIGUOUS_OVERLAP_RATIO = 0.2


def diarize(audio_path: str, transcript_segments: list) -> list:
    """
    Run speaker diarization and align with Whisper transcript segments.

    Args:
        audio_path: Path to audio file
        transcript_segments: segments list from transcribe() result

    Returns:
        List of dicts: [{"speaker", "start", "end", "text"}, ...]
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("[Diarize] pyannote.audio not installed. Returning mock speakers.")
        return _mock_diarize(transcript_segments)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("[Diarize] HF_TOKEN missing — skipping diarization, using single speaker.")
        return _mock_diarize(transcript_segments)

    print("[Diarize] Loading pyannote speaker-diarization-3.1...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    import torch
    import whisper
    
    wav_numpy = whisper.load_audio(audio_path)
    waveform = torch.from_numpy(wav_numpy).unsqueeze(0)
    
    diarization = pipeline({"waveform": waveform, "sample_rate": 16000})

    # Robustly extract the Annotation object from whatever pyannote returns.
    # Different versions return: Annotation directly, DiarizeOutput.diarization,
    # SpeakerDiarizationOutput, or some other wrapper — so we search all attributes.
    print(f"[Diarize] Pipeline returned type: {type(diarization).__name__}")

    annotation = None
    if hasattr(diarization, "itertracks"):
        # Old versions: returned Annotation directly
        annotation = diarization
    else:
        # New versions: search all attributes for one that IS an Annotation
        for attr_val in vars(diarization).values() if hasattr(diarization, "__dict__") else []:
            if hasattr(attr_val, "itertracks"):
                annotation = attr_val
                break
        # Last resort: check known attribute names explicitly
        if annotation is None:
            for attr_name in ("diarization", "annotation", "output", "result"):
                candidate = getattr(diarization, attr_name, None)
                if candidate is not None and hasattr(candidate, "itertracks"):
                    annotation = candidate
                    break

    if annotation is None:
        print(f"[Diarize] WARNING: Could not extract Annotation. Attrs: {dir(diarization)}")
        return _mock_diarize(transcript_segments)

    # Build speaker timeline: list of (start, end, speaker_label)
    timeline = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]

    # Align each Whisper segment with overlap-aware speaker selection
    diarized = _align_segments_with_timeline(transcript_segments, timeline)

    # Smooth very short speaker flips (A -> B -> A pattern)
    diarized = _smooth_short_turn_flips(diarized, min_turn_sec=SHORT_TURN_SMOOTH_SEC)

    print(f"[Diarize] Found {len(set(d['speaker'] for d in diarized))} speaker(s)")
    return diarized


def _find_speaker(seg_start: float, seg_end: float, timeline: list) -> str:
    """Return the speaker with the most overlap in the given time window."""
    overlap = _compute_overlap(seg_start, seg_end, timeline)
    if not overlap:
        return "SPEAKER_00"
    return max(overlap, key=overlap.get)


def _compute_overlap(seg_start: float, seg_end: float, timeline: list) -> dict:
    """Return overlap duration by speaker for a segment window."""
    overlap = {}
    for (t_start, t_end, speaker) in timeline:
        o = max(0, min(seg_end, t_end) - max(seg_start, t_start))
        overlap[speaker] = overlap.get(speaker, 0) + o
    return overlap


def _align_segments_with_timeline(transcript_segments: list, timeline: list) -> list:
    """Align Whisper segments with overlap-aware handling for ambiguous windows."""
    diarized = []
    prev_speaker = None

    for seg in transcript_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        seg_len = max(0.01, seg_end - seg_start)

        overlap = _compute_overlap(seg_start, seg_end, timeline)
        speaker = _choose_speaker_from_overlap(overlap, seg_len, prev_speaker)

        diarized.append({
            "speaker": speaker,
            "start": round(seg_start, 2),
            "end": round(seg_end, 2),
            "text": seg["text"].strip(),
        })
        prev_speaker = speaker

    return diarized


def _choose_speaker_from_overlap(overlap: dict, seg_len: float, prev_speaker: str | None) -> str:
    """
    Choose speaker with overlap handling:
    - Use dominant speaker when clear.
    - If two speakers are close in overlap, keep previous speaker for continuity.
    """
    if not overlap:
        return prev_speaker or "SPEAKER_00"

    ranked = sorted(overlap.items(), key=lambda kv: kv[1], reverse=True)
    top_speaker, top_overlap = ranked[0]

    if len(ranked) == 1:
        return top_speaker

    second_speaker, second_overlap = ranked[1]
    margin = top_overlap - second_overlap
    ambiguous = margin <= (AMBIGUOUS_OVERLAP_RATIO * max(top_overlap, 1e-6))

    # If overlap is ambiguous, prefer continuity to reduce rapid flip-flops.
    if ambiguous and prev_speaker in {top_speaker, second_speaker}:
        return prev_speaker

    # If previous speaker still has meaningful overlap in this window, keep it.
    if prev_speaker and overlap.get(prev_speaker, 0.0) >= 0.35 * seg_len and margin <= 0.25 * max(top_overlap, 1e-6):
        return prev_speaker

    return top_speaker


def _smooth_short_turn_flips(diarized: list, min_turn_sec: float = 1.2) -> list:
    """
    Smooth A -> B -> A patterns when B is a short turn.
    This reduces unstable speaker flips on brief ambiguous segments.
    """
    if len(diarized) < 3:
        return diarized

    smoothed = [dict(seg) for seg in diarized]

    for i in range(1, len(smoothed) - 1):
        prev_seg = smoothed[i - 1]
        curr_seg = smoothed[i]
        next_seg = smoothed[i + 1]

        curr_duration = max(0.0, curr_seg["end"] - curr_seg["start"])
        if curr_duration > min_turn_sec:
            continue

        if prev_seg["speaker"] == next_seg["speaker"] and curr_seg["speaker"] != prev_seg["speaker"]:
            curr_seg["speaker"] = prev_seg["speaker"]

    return smoothed


def _mock_diarize(segments: list) -> list:
    """Fallback when pyannote is unavailable — assigns single speaker."""
    return [
        {
            "speaker": "SPEAKER_00",
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        for seg in segments
    ]


def format_diarized_transcript(diarized: list) -> str:
    """Return a readable string grouping consecutive same-speaker turns."""
    lines = []
    current_speaker = None
    buffer = []

    for seg in diarized:
        if seg["speaker"] != current_speaker:
            if buffer:
                lines.append(f"**{current_speaker}:** {' '.join(buffer)}")
            current_speaker = seg["speaker"]
            buffer = [seg["text"]]
        else:
            buffer.append(seg["text"])

    if buffer:
        lines.append(f"**{current_speaker}:** {' '.join(buffer)}")

    return "\n\n".join(lines)


def save_diarized(diarized: list, output_path: str = "outputs/diarized.json") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(diarized, f, indent=2)
    print(f"[Diarize] Diarized transcript saved → {output_path}")
