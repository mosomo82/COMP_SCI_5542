"""
summarize.py — LLM-powered meeting summary (Gemini-first, Anthropic fallback)

This is your prompt engineering showcase.
Baseline prompt vs improved structured prompt — compare outputs in evaluation.ipynb.
"""

import os
import json
import importlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


MAX_SUMMARY_RETRIES = 3


# ─── Prompt Variants (for evaluation comparison) ──────────────────────────────

BASELINE_PROMPT = """
Summarize this meeting transcript:

{transcript}
"""

IMPROVED_PROMPT = """
You are an expert meeting analyst for a sales team. Analyze the transcript below and return a JSON object with this exact structure:

{{
  "executive_summary": "2-3 sentence overview of what was discussed and decided",
  "key_decisions": ["decision 1", "decision 2"],
  "action_items": [
    {{"owner": "SPEAKER_00", "task": "...", "deadline": "mentioned or null"}},
    {{"owner": "SPEAKER_01", "task": "...", "deadline": "mentioned or null"}}
  ],
  "topics_covered": ["topic 1", "topic 2"],
    "topic_sentiment": [
        {{"topic": "timeline", "sentiment": "positive|neutral|negative", "evidence": "short quote or reason"}},
        {{"topic": "budget", "sentiment": "positive|neutral|negative", "evidence": "short quote or reason"}}
    ],
  "unresolved_questions": ["question 1"],
  "meeting_tone": "collaborative | tense | productive | unfocused"
}}

Rules:
- Use only information explicitly stated in the transcript
- If a speaker is not identified, use their label (SPEAKER_00, etc.)
- Keep action items specific and actionable
- Return ONLY valid JSON, no preamble

Transcript:
{transcript}
"""


def summarize(
    diarized_transcript: str,
    keywords: list[str],
    sentiment: dict,
    prompt_variant: str = "improved",
) -> dict:
    """
    Generate structured meeting summary using Gemini (preferred)
    or Anthropic (fallback).

    Args:
        diarized_transcript: formatted string from format_diarized_transcript()
        keywords: list from extract_keywords()
        sentiment: dict from analyze_sentiment()
        prompt_variant: "baseline" | "improved" (for evaluation)

    Returns:
        Parsed summary dict
    """
    # Enrich transcript with context
    context = f"""
Keywords detected: {', '.join(keywords)}
Overall sentiment: {sentiment.get('overall', {}).get('label', 'unknown')}
Speaker sentiment signals: {json.dumps({k: v for k, v in sentiment.items() if k != 'overall'})[:1200]}

TRANSCRIPT:
{diarized_transcript}
""".strip()

    if prompt_variant == "baseline":
        user_content = BASELINE_PROMPT.format(transcript=context)
    else:
        user_content = IMPROVED_PROMPT.format(transcript=context)

    if prompt_variant == "improved":
        parsed = _generate_validated_summary(user_content, sentiment, max_retries=MAX_SUMMARY_RETRIES)
        return _postprocess_summary(parsed, sentiment)
    else:
        raw_response = _call_llm(user_content, prompt_variant)
        fallback = {
            "executive_summary": raw_response,
            "action_items": [],
            "key_decisions": [],
            "topics_covered": [],
            "topic_sentiment": [],
        }
        return _postprocess_summary(fallback, sentiment)


def _generate_validated_summary(user_content: str, sentiment: dict, max_retries: int = 3) -> dict:
    """Retry LLM call until a schema-valid summary is produced, then fallback locally."""
    attempt_prompt = user_content
    last_error = "unknown validation error"

    for attempt in range(1, max_retries + 1):
        raw_response = _call_llm(attempt_prompt, prompt_variant="improved")
        parsed = _parse_json_response(raw_response)
        if parsed is None:
            last_error = "response did not contain valid JSON"
            attempt_prompt = _build_retry_prompt(user_content, raw_response, last_error)
            print(f"[Summarize] Retry {attempt}/{max_retries}: {last_error}")
            continue

        candidate = _coerce_summary_shape(parsed)
        valid, reason = _validate_summary_schema(candidate)
        if valid:
            return candidate

        last_error = reason
        attempt_prompt = _build_retry_prompt(user_content, raw_response, reason)
        print(f"[Summarize] Retry {attempt}/{max_retries}: {reason}")

    print(f"[Summarize] Falling back to local summary after retries: {last_error}")
    return _coerce_summary_shape(json.loads(_local_fallback_summary(user_content)))


def _parse_json_response(raw_response: str) -> dict | None:
    """Parse JSON from raw LLM response, including fenced or mixed text outputs."""
    text = raw_response.strip()

    # Common fenced JSON output
    if text.startswith("```"):
        blocks = text.split("```")
        for block in blocks:
            candidate = block.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Direct JSON response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first plausible JSON object from mixed text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def _coerce_summary_shape(summary: dict) -> dict:
    """Coerce partially valid outputs into the target schema shape."""
    out = {
        "executive_summary": str(summary.get("executive_summary", "")).strip(),
        "key_decisions": summary.get("key_decisions", []),
        "action_items": summary.get("action_items", []),
        "topics_covered": summary.get("topics_covered", []),
        "topic_sentiment": summary.get("topic_sentiment", []),
        "unresolved_questions": summary.get("unresolved_questions", []),
        "meeting_tone": str(summary.get("meeting_tone", "productive")).strip().lower() or "productive",
    }

    if not isinstance(out["key_decisions"], list):
        out["key_decisions"] = [str(out["key_decisions"])]
    out["key_decisions"] = [str(x).strip() for x in out["key_decisions"] if str(x).strip()]

    if not isinstance(out["topics_covered"], list):
        out["topics_covered"] = [str(out["topics_covered"])]
    out["topics_covered"] = [str(x).strip() for x in out["topics_covered"] if str(x).strip()]

    if not isinstance(out["unresolved_questions"], list):
        out["unresolved_questions"] = [str(out["unresolved_questions"])]
    out["unresolved_questions"] = [str(x).strip() for x in out["unresolved_questions"] if str(x).strip()]

    if not isinstance(out["action_items"], list):
        out["action_items"] = []
    normalized_actions = []
    for item in out["action_items"]:
        if not isinstance(item, dict):
            continue
        normalized_actions.append({
            "owner": str(item.get("owner", "Unassigned")).strip() or "Unassigned",
            "task": str(item.get("task", "")).strip(),
            "deadline": item.get("deadline") if item.get("deadline") is None else str(item.get("deadline")).strip(),
        })
    out["action_items"] = normalized_actions

    if not isinstance(out["topic_sentiment"], list):
        out["topic_sentiment"] = []
    normalized_topics = []
    for item in out["topic_sentiment"]:
        if not isinstance(item, dict):
            continue
        normalized_topics.append({
            "topic": str(item.get("topic", "")).strip(),
            "sentiment": str(item.get("sentiment", "neutral")).strip().lower(),
            "evidence": str(item.get("evidence", "")).strip(),
        })
    out["topic_sentiment"] = normalized_topics

    return out


def _validate_summary_schema(summary: dict) -> tuple[bool, str]:
    required = [
        "executive_summary",
        "key_decisions",
        "action_items",
        "topics_covered",
        "topic_sentiment",
        "unresolved_questions",
        "meeting_tone",
    ]
    for key in required:
        if key not in summary:
            return False, f"missing required field: {key}"

    if not isinstance(summary["executive_summary"], str):
        return False, "executive_summary must be a string"
    if not isinstance(summary["key_decisions"], list):
        return False, "key_decisions must be a list"
    if not isinstance(summary["action_items"], list):
        return False, "action_items must be a list"
    if not isinstance(summary["topics_covered"], list):
        return False, "topics_covered must be a list"
    if not isinstance(summary["topic_sentiment"], list):
        return False, "topic_sentiment must be a list"
    if not isinstance(summary["unresolved_questions"], list):
        return False, "unresolved_questions must be a list"
    if not isinstance(summary["meeting_tone"], str):
        return False, "meeting_tone must be a string"

    for item in summary["action_items"]:
        if not isinstance(item, dict):
            return False, "each action_items entry must be an object"
        if "owner" not in item or "task" not in item or "deadline" not in item:
            return False, "each action item must include owner, task, deadline"

    valid_topic_sentiments = {"positive", "neutral", "negative"}
    for item in summary["topic_sentiment"]:
        if not isinstance(item, dict):
            return False, "each topic_sentiment entry must be an object"
        if "topic" not in item or "sentiment" not in item or "evidence" not in item:
            return False, "each topic_sentiment entry must include topic, sentiment, evidence"
        if str(item["sentiment"]).lower() not in valid_topic_sentiments:
            return False, "topic_sentiment.sentiment must be positive|neutral|negative"

    return True, "ok"


def _build_retry_prompt(original_prompt: str, bad_response: str, reason: str) -> str:
    """Ask the model to repair malformed response and return strict JSON only."""
    return (
        original_prompt
        + "\n\n"
        + "Your previous response was invalid. Return ONLY JSON that strictly matches the required schema. "
        + f"Validation error: {reason}.\n"
        + "Do not include markdown fences, explanation, or extra keys.\n"
        + "Previous invalid response (for repair context):\n"
        + bad_response[:2000]
    )


def _call_llm(user_content: str, prompt_variant: str) -> str:
    """Call Gemini first, then Anthropic fallback, else deterministic local fallback."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if gemini_key:
        try:
            import google.generativeai as genai
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
            print(f"[Summarize] Sending to Gemini ({model_name}, {prompt_variant} prompt)...")
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(user_content)
            return (response.text or "").strip()
        except Exception as e:
            print(f"[Summarize] Gemini error: {e}. Trying Anthropic/local fallback.")

    if anthropic_key:
        try:
            print(f"[Summarize] Sending to Anthropic ({prompt_variant} prompt)...")
            anthropic_module = importlib.import_module("anthropic")
            client = anthropic_module.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": user_content}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"[Summarize] Anthropic error: {e}. Using local fallback.")

    print("[Summarize] No LLM API key found. Using deterministic local fallback summary.")
    return _local_fallback_summary(user_content)


def _local_fallback_summary(text: str) -> str:
    """
    Extractive heuristic summary — runs fully offline, no API key needed.
    Returns JSON-formatted string matching the IMPROVED_PROMPT schema.
    """
    import re

    # Split header context from transcript body
    if "TRANSCRIPT:" in text:
        header, body = text.split("TRANSCRIPT:", 1)
    else:
        header, body = "", text

    # Extract keywords hint from header
    keywords: list[str] = []
    kw_match = re.search(r"Keywords detected:\s*(.+)", header)
    if kw_match:
        keywords = [k.strip() for k in kw_match.group(1).split(",") if k.strip()]

    # Parse speaker turns
    turns = re.findall(r"(SPEAKER_\w+):\s+(.+?)(?=SPEAKER_\w+:|$)", body, re.DOTALL)
    speakers = list(dict.fromkeys(t[0] for t in turns))

    # Collect (speaker, sentence) pairs
    labeled_sentences: list[tuple[str, str]] = []
    for speaker, utterance in turns:
        for s in re.split(r"(?<=[.!?])\s+", utterance.strip()):
            s = s.strip()
            if len(s) > 15:
                labeled_sentences.append((speaker, s))

    if not labeled_sentences:
        raw = re.split(r"(?<=[.!?])\s+", body.strip())
        labeled_sentences = [("SPEAKER_00", s.strip()) for s in raw if len(s.strip()) > 15]

    # Score sentences by importance keywords + keyword overlap
    _importance = {
        "decision", "decided", "agreed", "will", "action", "deadline", "due",
        "plan", "goal", "launch", "implement", "complete", "review", "approve",
        "submit", "schedule", "next", "budget", "milestone",
    }

    def _score(sentence: str) -> int:
        s = sentence.lower()
        return sum(1 for w in _importance if w in s) + sum(1 for k in keywords if k.lower() in s)

    ranked = sorted(labeled_sentences, key=lambda x: _score(x[1]), reverse=True)
    top_sentences = [s for _, s in ranked[:3]]
    executive_summary = " ".join(top_sentences) if top_sentences else "Meeting transcript processed with local summarizer."

    # Extract action items (owner + action verb + task text)
    action_items: list[dict] = []
    action_re = re.compile(
        r"(SPEAKER_\w+)[^.]*?(?:will|should|needs? to|must|going to)\s+(.{10,80}?)(?:[.!?]|$)",
        re.IGNORECASE,
    )
    for m in action_re.finditer(body):
        task_text = m.group(2).strip().rstrip(".,")
        if len(task_text) > 5:
            action_items.append({"owner": m.group(1), "task": task_text, "deadline": None})
    action_items = action_items[:5]

    # Extract key decisions
    decision_re = re.compile(
        r"(?:decided|agreed|confirmed|approved|conclusion)[:\s]+(.{10,120}?)(?:[.!?]|$)",
        re.IGNORECASE,
    )
    key_decisions = [m.group(1).strip().rstrip(".,") for m in decision_re.finditer(body)][:3]
    if not key_decisions and top_sentences:
        key_decisions = top_sentences[:1]

    result = {
        "executive_summary": executive_summary,
        "key_decisions": key_decisions,
        "action_items": action_items,
        "topics_covered": keywords[:5] if keywords else ["general discussion"],
        "topic_sentiment": [
            {
                "topic": topic,
                "sentiment": "neutral",
                "evidence": "Local fallback uses keyword-only topic extraction without deep sentiment attribution.",
            }
            for topic in (keywords[:5] if keywords else ["general discussion"])
        ],
        "unresolved_questions": [],
        "meeting_tone": "productive",
    }
    return json.dumps(result)


def _postprocess_summary(summary: dict, sentiment: dict) -> dict:
    """Normalize optional fields and add topic sentiment confidence metrics."""
    summary.setdefault("topics_covered", [])
    summary.setdefault("topic_sentiment", [])

    topic_items = []
    for item in summary.get("topic_sentiment", []):
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic", "")).strip()
        if not topic:
            continue
        topic_label = _normalize_topic_label(item.get("sentiment", "neutral"))
        evidence = str(item.get("evidence", "")).strip()
        confidence = _estimate_topic_confidence(topic_label, evidence, sentiment)
        enriched = {
            "topic": topic,
            "sentiment": topic_label,
            "evidence": evidence,
            "confidence": round(confidence, 3),
        }
        topic_items.append(enriched)

    summary["topic_sentiment"] = topic_items

    if topic_items:
        avg_conf = sum(i["confidence"] for i in topic_items) / len(topic_items)
        high_conf = sum(1 for i in topic_items if i["confidence"] >= 0.65)
    else:
        avg_conf = 0.0
        high_conf = 0

    summary["topic_sentiment_confidence"] = round(avg_conf, 3)
    summary["high_confidence_topics"] = high_conf
    return summary


def _normalize_topic_label(label: str) -> str:
    s = str(label).strip().lower()
    if s.startswith("pos"):
        return "positive"
    if s.startswith("neg"):
        return "negative"
    return "neutral"


def _signed_sentiment_value(label: str, score: float) -> float:
    l = str(label).upper()
    sc = max(0.0, min(float(score), 1.0))
    if l == "POSITIVE":
        return sc
    if l == "NEGATIVE":
        return -sc
    return 0.0


def _estimate_topic_confidence(topic_label: str, evidence: str, sentiment: dict) -> float:
    """Heuristic confidence score in [0.05, 0.99] for topic sentiment entries."""
    conf = 0.55

    ev_len = len(evidence)
    if ev_len >= 20:
        conf += 0.12
    if ev_len >= 60:
        conf += 0.08
    if '"' in evidence or "'" in evidence:
        conf += 0.03

    # Compare topic sentiment direction with meeting-level direction for consistency.
    overall = sentiment.get("overall", {}) if isinstance(sentiment, dict) else {}
    overall_valence = overall.get("valence")
    if overall_valence is None:
        overall_valence = _signed_sentiment_value(overall.get("label", "NEUTRAL"), overall.get("score", 0.0))

    topic_sign = 0
    if topic_label == "positive":
        topic_sign = 1
    elif topic_label == "negative":
        topic_sign = -1

    if topic_sign != 0 and overall_valence != 0:
        if topic_sign * float(overall_valence) > 0:
            conf += 0.08
        else:
            conf -= 0.08
    elif topic_sign == 0:
        conf += 0.03

    return max(0.05, min(conf, 0.99))


def format_summary_for_speech(summary: dict) -> str:
    """
    Convert structured summary dict into natural language for TTS narration.
    """
    parts = []

    if summary.get("executive_summary"):
        parts.append(f"Meeting summary. {summary['executive_summary']}")

    if summary.get("key_decisions"):
        decisions = ". ".join(summary["key_decisions"])
        parts.append(f"Key decisions made: {decisions}.")

    if summary.get("action_items"):
        parts.append("Action items:")
        for item in summary["action_items"]:
            owner = item.get("owner", "Unassigned")
            task = item.get("task", "")
            deadline = item.get("deadline")
            line = f"{owner} will {task}"
            if deadline:
                line += f", due {deadline}"
            parts.append(line + ".")

    if summary.get("topic_sentiment"):
        parts.append("Topic sentiment highlights:")
        for item in summary["topic_sentiment"][:3]:
            topic = item.get("topic", "topic")
            tone = item.get("sentiment", "neutral")
            parts.append(f"{topic} was {tone}.")

    if summary.get("unresolved_questions"):
        parts.append("Open questions remaining: " + ". ".join(summary["unresolved_questions"]) + ".")

    return " ".join(parts)


def save_summary(summary: dict, output_path: str = "outputs/summary.json") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summarize] Summary saved → {output_path}")
