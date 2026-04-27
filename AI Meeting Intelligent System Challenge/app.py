"""
app.py — Gradio UI for the Meeting Intelligence System

Run: python app.py
Opens at http://localhost:7860
"""

import gradio as gr
from src.pipeline import run_pipeline


def process_meeting(audio_file, whisper_model, prompt_variant, generate_audio):
    """Gradio handler — called when user clicks Submit."""
    if audio_file is None:
        return "Please upload an audio file.", "", "", "", None

    result = run_pipeline(
        audio_path=audio_file,
        whisper_model=whisper_model,
        prompt_variant=prompt_variant,
        generate_audio=generate_audio,
    )

    # Format sentiment for display
    sentiment_display = _format_sentiment(result["sentiment"])

    # Format summary for display
    summary = result["summary"]
    summary_display = _format_summary(summary)

    # Timing info
    timing = "\n".join(
        f"{k}: {v}s" for k, v in result["stage_times"].items()
    )

    info_display = _format_info(result, timing)

    return (
        result["diarized_text"],         # Transcript tab
        summary_display,                  # Summary tab
        sentiment_display,                # Sentiment tab
        info_display,                     # Info tab
        result["audio_path"],             # Audio output
    )


def _format_sentiment(sentiment: dict) -> str:
    def emoji_for(label: str) -> str:
        if label == "POSITIVE":
            return "😊"
        if label == "NEGATIVE":
            return "😟"
        return "😐"

    lines = []
    for speaker, data in sentiment.items():
        if speaker == "overall":
            continue
        emoji = emoji_for(data.get("label", "NEUTRAL"))
        lines.append(f"{emoji} **{speaker}**: {data['label']} ({data['score']:.0%})")
    overall = sentiment.get("overall", {})
    emoji = emoji_for(overall.get("label", "NEUTRAL"))
    lines.append(f"\n{emoji} **Overall meeting**: {overall.get('label', 'N/A')} ({overall.get('score', 0):.0%})")
    return "\n".join(lines)


def _format_summary(summary: dict) -> str:
    if not summary:
        return "No summary generated."

    parts = []

    if summary.get("executive_summary"):
        parts.append(f"### Summary\n{summary['executive_summary']}")

    if summary.get("key_decisions"):
        parts.append("### Key Decisions\n" + "\n".join(f"- {d}" for d in summary["key_decisions"]))

    if summary.get("action_items"):
        parts.append("### Action Items")
        for item in summary["action_items"]:
            deadline = f" _(due: {item['deadline']})_" if item.get("deadline") else ""
            parts.append(f"- **{item.get('owner', '?')}**: {item.get('task', '')}{deadline}")

    if summary.get("topic_sentiment"):
        parts.append("### Topic Sentiment")
        for item in summary["topic_sentiment"]:
            topic = item.get("topic", "topic")
            tone = str(item.get("sentiment", "neutral")).capitalize()
            evidence = item.get("evidence")
            conf = item.get("confidence")
            conf_txt = f" ({conf:.0%})" if isinstance(conf, (int, float)) else ""
            if evidence:
                parts.append(f"- **{topic}**: {tone}{conf_txt} — {evidence}")
            else:
                parts.append(f"- **{topic}**: {tone}{conf_txt}")

    if summary.get("unresolved_questions"):
        parts.append("### Open Questions\n" + "\n".join(f"- {q}" for q in summary["unresolved_questions"]))

    if summary.get("meeting_tone"):
        parts.append(f"### Tone\n{summary['meeting_tone'].capitalize()}")

    return "\n\n".join(parts)


def _format_info(result: dict, timing: str) -> str:
    keywords = result.get("keywords", [])
    summary = result.get("summary", {}) or {}

    lines = [f"Keywords: {', '.join(keywords)}"]

    topic_items = summary.get("topic_sentiment", [])
    if topic_items:
        avg_conf = summary.get("topic_sentiment_confidence", 0.0)
        high_conf = summary.get("high_confidence_topics", 0)
        lines.append("")
        lines.append("Topic sentiment quality:")
        lines.append(f"- Avg confidence: {float(avg_conf):.0%}")
        lines.append(f"- High-confidence topics (>=65%): {int(high_conf)}/{len(topic_items)}")

    lines.append("")
    lines.append("Latency:")
    lines.append(timing)
    return "\n".join(lines)


# ─── UI Layout ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Meeting Intelligence", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙 Meeting Intelligence System")
    gr.Markdown("Upload a meeting recording to get a transcript, summary, action items, sentiment analysis, and voiced summary.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Meeting audio",
            )
            with gr.Accordion("Settings", open=False):
                whisper_model = gr.Radio(
                    choices=["tiny", "small", "medium"],
                    value="small",
                    label="Whisper model (small = faster, medium = more accurate)",
                )
                prompt_variant = gr.Radio(
                    choices=["baseline", "improved"],
                    value="improved",
                    label="Summary prompt variant",
                )
                generate_audio = gr.Checkbox(
                    value=True,
                    label="Generate voiced summary (slower)",
                )
            submit_btn = gr.Button("Analyze Meeting", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Transcript"):
                    transcript_out = gr.Markdown(label="Diarized transcript")
                with gr.Tab("Summary"):
                    summary_out = gr.Markdown(label="Structured summary")
                with gr.Tab("Sentiment"):
                    sentiment_out = gr.Markdown(label="Speaker sentiment")
                with gr.Tab("Info"):
                    info_out = gr.Markdown(label="Keywords and latency")

            audio_out = gr.Audio(label="Voiced summary", type="filepath")

    submit_btn.click(
        fn=process_meeting,
        inputs=[audio_input, whisper_model, prompt_variant, generate_audio],
        outputs=[transcript_out, summary_out, sentiment_out, info_out, audio_out],
    )

    gr.Markdown(
        "**Sample audio**: Use any short meeting recording (.mp3/.wav/.m4a). "
        "No audio? Record yourself summarizing a project update."
    )


if __name__ == "__main__":
    demo.launch(share=False)
