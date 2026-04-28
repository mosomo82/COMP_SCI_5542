"""
analyze.py — Sentiment analysis + keyword extraction

Models used:
    - cardiffnlp/twitter-roberta-base-sentiment-latest (sentiment, primary)
    - distilbert-base-uncased-finetuned-sst-2-english (sentiment, fallback)
  - KeyBERT (keyword extraction)
"""

import json
from pathlib import Path
from collections import defaultdict


PRIMARY_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
FALLBACK_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def analyze_sentiment(diarized: list) -> dict:
    """
    Run sentiment analysis at dialogue-turn level, then aggregate per speaker.

    Args:
        diarized: output from diarize()

    Returns:
        {
            "SPEAKER_00": {"label": "POSITIVE", "score": 0.94, "segments": [...]},
            "SPEAKER_01": {...},
            "overall": {"label": "POSITIVE", "score": 0.87}
        }
    """
    classifier = _build_sentiment_classifier()

    # Group turns by speaker
    speaker_turns = defaultdict(list)
    for seg in diarized:
        text = str(seg.get("text", "")).strip()
        if text:
            speaker_turns[seg.get("speaker", "SPEAKER_00")].append(text)

    if not speaker_turns:
        return {
            "overall": {
                "label": "NEUTRAL",
                "score": 0.0,
                "valence": 0.0,
                "turns_analyzed": 0,
            }
        }

    results = {}
    overall_weighted_sum = 0.0
    overall_weight_total = 0.0

    for speaker, turns in speaker_turns.items():
        turn_polarities = []
        turn_weights = []

        for turn in turns:
            pred = classifier(turn[:512])[0]
            label = _normalize_sentiment_label(pred.get("label", ""))
            confidence = float(pred.get("score", 0.0))
            polarity = _label_confidence_to_polarity(label, confidence)

            # Weight by turn length (capped) to avoid over-dominating long turns.
            weight = min(len(turn.split()), 40)
            weight = max(weight, 1)

            turn_polarities.append(polarity)
            turn_weights.append(weight)

        weighted_polarity = _weighted_average(turn_polarities, turn_weights)
        sentiment_label, sentiment_score = _polarity_to_label_score(weighted_polarity)

        results[speaker] = {
            "label": sentiment_label,
            "score": round(sentiment_score, 4),
            "valence": round(weighted_polarity, 4),
            "turns_analyzed": len(turns),
            "text_sample": " ".join(turns)[:200],
        }

        overall_weighted_sum += weighted_polarity * len(turns)
        overall_weight_total += len(turns)

    overall_polarity = overall_weighted_sum / overall_weight_total if overall_weight_total else 0.0
    overall_label, overall_score = _polarity_to_label_score(overall_polarity)
    results["overall"] = {
        "label": overall_label,
        "score": round(overall_score, 4),
        "valence": round(overall_polarity, 4),
        "turns_analyzed": int(overall_weight_total),
    }

    print(f"[Analyze] Sentiment — Overall: {results['overall']['label']} ({results['overall']['score']})")
    return results


def _build_sentiment_classifier():
    from transformers import pipeline as hf_pipeline

    try:
        print(f"[Analyze] Loading sentiment model: {PRIMARY_SENTIMENT_MODEL}")
        return hf_pipeline(
            "sentiment-analysis",
            model=PRIMARY_SENTIMENT_MODEL,
            truncation=True,
            max_length=512,
        )
    except Exception as e:
        print(f"[Analyze] Primary model unavailable ({e}). Falling back to {FALLBACK_SENTIMENT_MODEL}")
        return hf_pipeline(
            "sentiment-analysis",
            model=FALLBACK_SENTIMENT_MODEL,
            truncation=True,
            max_length=512,
        )


def _normalize_sentiment_label(raw_label: str) -> str:
    label = raw_label.strip().upper()

    if label in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
        return label
    if label in {"LABEL_2", "2"}:
        return "POSITIVE"
    if label in {"LABEL_1", "1"}:
        return "NEUTRAL"
    if label in {"LABEL_0", "0"}:
        return "NEGATIVE"
    if "POS" in label:
        return "POSITIVE"
    if "NEG" in label:
        return "NEGATIVE"
    if "NEU" in label:
        return "NEUTRAL"
    return "NEUTRAL"


def _label_confidence_to_polarity(label: str, confidence: float) -> float:
    """Map label+confidence into signed valence in [-1, 1]."""
    c = max(0.0, min(float(confidence), 1.0))
    if label == "POSITIVE":
        return c
    if label == "NEGATIVE":
        return -c
    return 0.0


def _weighted_average(values: list[float], weights: list[int]) -> float:
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


def _polarity_to_label_score(polarity: float) -> tuple[str, float]:
    """
    Convert signed polarity to a discrete sentiment label and a display score.

    Threshold window keeps mild valence as NEUTRAL to better match mixed meeting tone.
    """
    if polarity >= 0.15:
        return "POSITIVE", min(abs(polarity), 1.0)
    if polarity <= -0.15:
        return "NEGATIVE", min(abs(polarity), 1.0)
    return "NEUTRAL", min(1.0 - abs(polarity), 1.0)


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """
    Extract top keywords/phrases from the full transcript.

    Args:
        text: full transcript string
        top_n: number of keywords to return

    Returns:
        List of keyword strings, ranked by relevance
    """
    try:
        from keybert import KeyBERT
        print("[Analyze] Extracting keywords with KeyBERT...")
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
        )
        return [kw for kw, score in keywords]

    except ImportError:
        # Fallback: simple frequency-based extraction
        print("[Analyze] KeyBERT not available — using simple frequency extraction.")
        import re
        from collections import Counter
        stopwords = {"the","a","an","is","it","in","on","at","to","for","of","and","or","but","with","this","that","was","are","be","have","has","had","will","would","could","should","we","i","you","they","he","she"}
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        filtered = [w for w in words if w not in stopwords]
        return [w for w, _ in Counter(filtered).most_common(top_n)]


def save_analysis(sentiment: dict, keywords: list, output_path: str = "outputs/sentiment.json") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"sentiment": sentiment, "keywords": keywords}, f, indent=2)
    print(f"[Analyze] Analysis saved -> {output_path}")
