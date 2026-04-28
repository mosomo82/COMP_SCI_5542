"""
tests/test_pipeline.py — Basic smoke tests

Run: python -m pytest tests/ -v
"""

import pytest
import sys
import os
from pathlib import Path

# Add repo root to path so src modules are importable
sys.path.append(str(Path(__file__).parent.parent))

# Ensure ffmpeg is in PATH for transcription tests
FFMPEG_PATH = r"C:\Users\mtuan\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
if FFMPEG_PATH not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH


# ── Transcription ──────────────────────────────────────────────────────────────

def test_transcribe_returns_expected_keys():
    from src.transcribe import transcribe
    # Use a tiny generated audio if no sample exists
    import numpy as np
    import soundfile as sf
    import tempfile, os

    silence = np.zeros(16000, dtype=np.float32)  # 1s silence
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        os.close(fd)
        sf.write(path, silence, 16000)
        result = transcribe(path, model_size="tiny")
    finally:
        if os.path.exists(path):
            os.unlink(path)

    assert "text" in result
    assert "segments" in result
    assert "language" in result
    assert isinstance(result["segments"], list)


# ── Diarization mock ──────────────────────────────────────────────────────────

def test_mock_diarize():
    from src.diarize import _mock_diarize
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello everyone."},
        {"start": 2.5, "end": 5.0, "text": "Let us get started."},
    ]
    result = _mock_diarize(segments)
    assert len(result) == 2
    assert all("speaker" in r for r in result)
    assert all(r["speaker"] == "SPEAKER_00" for r in result)


def test_diarization_short_turn_smoothing_reduces_flips():
    from src.diarize import _smooth_short_turn_flips

    diarized = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0, "text": "Opening discussion."},
        {"speaker": "SPEAKER_01", "start": 3.0, "end": 3.5, "text": "Yes."},
        {"speaker": "SPEAKER_00", "start": 3.5, "end": 8.0, "text": "Continuing details."},
    ]

    smoothed = _smooth_short_turn_flips(diarized, min_turn_sec=1.2)
    assert smoothed[1]["speaker"] == "SPEAKER_00"


# ── Keyword extraction fallback ───────────────────────────────────────────────

def test_keyword_extraction_fallback():
    from src.analyze import extract_keywords
    text = "The product launch is scheduled for next quarter. Marketing needs a budget plan."
    keywords = extract_keywords(text, top_n=5)
    assert isinstance(keywords, list)
    assert len(keywords) <= 5
    assert all(isinstance(k, str) for k in keywords)


def test_sentiment_label_normalization_and_polarity_mapping():
    from src.analyze import _normalize_sentiment_label, _label_confidence_to_polarity

    assert _normalize_sentiment_label("LABEL_2") == "POSITIVE"
    assert _normalize_sentiment_label("LABEL_1") == "NEUTRAL"
    assert _normalize_sentiment_label("LABEL_0") == "NEGATIVE"

    assert _label_confidence_to_polarity("POSITIVE", 0.8) > 0
    assert _label_confidence_to_polarity("NEGATIVE", 0.8) < 0
    assert _label_confidence_to_polarity("NEUTRAL", 0.8) == 0


def test_polarity_thresholding_to_labels():
    from src.analyze import _polarity_to_label_score

    label_pos, score_pos = _polarity_to_label_score(0.55)
    label_neg, score_neg = _polarity_to_label_score(-0.45)
    label_neu, score_neu = _polarity_to_label_score(0.04)

    assert label_pos == "POSITIVE"
    assert label_neg == "NEGATIVE"
    assert label_neu == "NEUTRAL"
    assert 0 <= score_pos <= 1
    assert 0 <= score_neg <= 1
    assert 0 <= score_neu <= 1


# ── Text chunking ─────────────────────────────────────────────────────────────

def test_text_chunking():
    from src.speak import _chunk_text
    
    # Mock processor to simulate token count = char count
    class MockProcessor:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": type('obj', (object,), {'shape': (1, len(text))})}
    
    long_text = "Hello world. " * 50
    chunks = _chunk_text(long_text, processor=MockProcessor(), max_tokens=100)
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks) # allow slight sentence overlap


# ── Summary formatting ────────────────────────────────────────────────────────

def test_format_summary_for_speech():
    from src.summarize import format_summary_for_speech
    summary = {
        "executive_summary": "The team agreed on the Q3 roadmap.",
        "key_decisions": ["Launch on August 15"],
        "action_items": [
            {"owner": "SPEAKER_00", "task": "send calendar invite", "deadline": "Friday"}
        ],
        "topic_sentiment": [
            {"topic": "launch timeline", "sentiment": "positive", "evidence": "team aligned on Aug 15"}
        ],
        "unresolved_questions": ["Budget approval still pending"],
    }
    text = format_summary_for_speech(summary)
    assert "Q3 roadmap" in text
    assert "August 15" in text
    assert "SPEAKER_00" in text
    assert "launch timeline" in text
    assert isinstance(text, str)
    assert len(text) > 50


def test_postprocess_summary_adds_topic_confidence_metrics():
    from src.summarize import _postprocess_summary

    summary = {
        "executive_summary": "Quick summary.",
        "topic_sentiment": [
            {
                "topic": "budget",
                "sentiment": "negative",
                "evidence": "Budget approval was delayed and concern was repeated by two speakers.",
            },
            {
                "topic": "timeline",
                "sentiment": "positive",
                "evidence": "Team aligned on Aug 15 release date.",
            },
        ],
    }
    sentiment = {"overall": {"label": "NEUTRAL", "score": 0.52, "valence": 0.0}}

    enriched = _postprocess_summary(summary, sentiment)

    assert "topic_sentiment_confidence" in enriched
    assert "high_confidence_topics" in enriched
    assert len(enriched["topic_sentiment"]) == 2
    assert all("confidence" in item for item in enriched["topic_sentiment"])
    assert 0 <= enriched["topic_sentiment_confidence"] <= 1


def test_summary_schema_validation_and_coercion():
    from src.summarize import _coerce_summary_shape, _validate_summary_schema

    raw = {
        "executive_summary": "Summary",
        "key_decisions": "Use Aug 15",
        "action_items": [{"owner": "SPEAKER_00", "task": "send invite", "deadline": None}],
        "topics_covered": "timeline",
        "topic_sentiment": [{"topic": "timeline", "sentiment": "positive", "evidence": "agreement"}],
        "unresolved_questions": [],
        "meeting_tone": "productive",
    }

    coerced = _coerce_summary_shape(raw)
    is_valid, _ = _validate_summary_schema(coerced)

    assert is_valid
    assert isinstance(coerced["key_decisions"], list)
    assert isinstance(coerced["topics_covered"], list)
