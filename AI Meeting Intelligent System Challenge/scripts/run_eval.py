#!/usr/bin/env python3
"""
run_eval.py — One-click evaluation runner for the Meeting Intelligence pipeline.

Generates a full evaluation report covering:
  • WER  (Whisper small vs medium, optional reference transcript)
  • Summary quality rubric  (fields populated, action items, key decisions)
  • Action-item precision   (if expected_summary.json is provided)
  • Per-stage latency table
  • Noise robustness        (optional, requires --noise-levels)

Usage:
  # Quick run — baseline audio, small model, both prompt variants
  python scripts/run_eval.py

  # Full run — compare small vs medium Whisper with WER
  python scripts/run_eval.py --audio audio_samples/sample_meeting.wav \\
      --whisper-models small medium \\
      --reference audio_samples/expected_transcript.txt \\
      --expected-summary audio_samples/expected_summary.json

  # Add noise robustness sweep (requires pydub)
  python scripts/run_eval.py --noise-levels 0 5 10 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Allow running from repo root or scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate via dynamic programming."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return round(d[len(ref)][len(hyp)] / len(ref), 4)


def score_summary_quality(summary: dict) -> dict:
    """Rubric: check presence of expected fields and structural quality."""
    fields = [
        "executive_summary", "key_decisions", "action_items",
        "topics_covered", "unresolved_questions", "meeting_tone"
    ]
    present = sum(1 for f in fields if summary.get(f))
    action_items = summary.get("action_items", [])
    action_has_owner = all("owner" in item for item in action_items)
    action_has_task  = all("task"  in item for item in action_items)
    return {
        "fields_populated":        f"{present}/{len(fields)}",
        "fields_populated_pct":    round(present / len(fields), 2),
        "action_items_count":      len(action_items),
        "key_decisions_count":     len(summary.get("key_decisions", [])),
        "action_items_well_formed": action_has_owner and action_has_task,
        "has_unresolved_questions": bool(summary.get("unresolved_questions")),
        "has_meeting_tone":        bool(summary.get("meeting_tone")),
    }


def score_action_item_precision(predicted: list, expected: list) -> dict:
    """
    Soft precision: for each predicted action item, check if any expected
    action item contains its task keywords (token overlap).
    """
    if not expected or not predicted:
        return {"precision": "n/a", "predicted": len(predicted), "expected": len(expected)}

    def tokens(text: str) -> set:
        return set(text.lower().split())

    hits = 0
    for pred in predicted:
        pred_tokens = tokens(pred.get("task", ""))
        for exp in expected:
            exp_tokens = tokens(exp.get("task", ""))
            if pred_tokens and exp_tokens:
                overlap = len(pred_tokens & exp_tokens) / len(pred_tokens | exp_tokens)
                if overlap >= 0.3:
                    hits += 1
                    break

    precision = round(hits / len(predicted), 2) if predicted else 0.0
    return {
        "precision":  precision,
        "hits":       hits,
        "predicted":  len(predicted),
        "expected":   len(expected),
    }


def add_noise_to_audio(audio_path: str, snr_db: int, out_path: str) -> str:
    """Add Gaussian noise at a given SNR level using pydub."""
    try:
        import numpy as np
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        signal_power = np.mean(samples ** 2)
        noise_power  = signal_power / (10 ** (snr_db / 10))
        noise        = np.random.normal(0, np.sqrt(noise_power), len(samples)).astype(np.int16)
        noisy        = np.clip(samples + noise, -32768, 32767).astype(np.int16)
        noisy_seg    = audio._spawn(noisy.tobytes())
        noisy_seg.export(out_path, format="wav")
        return out_path
    except ImportError:
        print("[Eval] pydub/numpy not available — skipping noise augmentation.")
        return audio_path


# ─── Report ───────────────────────────────────────────────────────────────────

def print_table(title: str, rows: list[tuple], headers: list[str]) -> None:
    """Pretty-print a table to stdout."""
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    sep   = "  ".join("-" * w for w in col_widths)
    fmt   = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(f"\n  {title}")
    print(f"  {sep}")
    print(f"  {fmt.format(*headers)}")
    print(f"  {sep}")
    for row in rows:
        print(f"  {fmt.format(*[str(c) for c in row])}")
    print(f"  {sep}")


def save_markdown_report(results: dict, path: Path) -> None:
    """Write a Markdown evaluation report."""
    lines = [
        "# Meeting Intelligence — Evaluation Report",
        f"\n**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Audio:** `{results['audio']}`\n",
        "## Latency per Stage",
        "| Run | Transcription | Diarization | Analysis | Summarization | Total |",
        "|-----|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results["runs"]:
        lat = r.get("stage_latencies_sec", {})
        lines.append(
            f"| {r['whisper_model']}_{r['prompt_variant']} "
            f"| {lat.get('transcription','—')}s "
            f"| {lat.get('diarization','—')}s "
            f"| {lat.get('analysis','—')}s "
            f"| {lat.get('summarization','—')}s "
            f"| **{r['total_latency_sec']}s** |"
        )

    if any("wer" in r for r in results["runs"]):
        lines += [
            "\n## Word Error Rate (WER)",
            "| Run | WER |",
            "|-----|:---:|",
        ]
        for r in results["runs"]:
            lines.append(f"| {r['whisper_model']}_{r['prompt_variant']} | {r.get('wer','n/a')} |")

    lines += [
        "\n## Summary Quality Rubric",
        "| Run | Fields Populated | Action Items | Key Decisions | Well-Formed |",
        "|-----|:---:|:---:|:---:|:---:|",
    ]
    for r in results["runs"]:
        q = r.get("summary_quality", {})
        lines.append(
            f"| {r['whisper_model']}_{r['prompt_variant']} "
            f"| {q.get('fields_populated','—')} "
            f"| {q.get('action_items_count','—')} "
            f"| {q.get('key_decisions_count','—')} "
            f"| {'✅' if q.get('action_items_well_formed') else '❌'} |"
        )

    if any("action_item_precision" in r for r in results["runs"]):
        lines += [
            "\n## Action-Item Precision",
            "| Run | Precision | Hits / Predicted / Expected |",
            "|-----|:---:|:---:|",
        ]
        for r in results["runs"]:
            ap = r.get("action_item_precision", {})
            if isinstance(ap, dict) and ap.get("precision") != "n/a":
                lines.append(
                    f"| {r['whisper_model']}_{r['prompt_variant']} "
                    f"| {ap['precision']:.0%} "
                    f"| {ap['hits']} / {ap['predicted']} / {ap['expected']} |"
                )

    if results.get("noise_robustness"):
        lines += ["\n## Noise Robustness (WER vs SNR)", "| SNR (dB) | WER |", "|:---:|:---:|"]
        for entry in results["noise_robustness"]:
            lines.append(f"| {entry['snr_db']} | {entry['wer']} |")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Eval] Markdown report saved → {path}")


# ─── Core runner ──────────────────────────────────────────────────────────────

def run_evaluation(
    audio_path: str,
    whisper_models: list,
    prompt_variants: list,
    reference_transcript: str | None,
    expected_summary: dict | None,
    output_dir: str,
    noise_levels: list,
) -> dict:
    from src.pipeline import run_pipeline

    results = {
        "audio":            audio_path,
        "generated_at":     datetime.utcnow().isoformat() + "Z",
        "runs":             [],
        "comparison":       {},
        "noise_robustness": [],
    }
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Main grid: model × prompt variant ────────────────────────────────────
    for model in whisper_models:
        for variant in prompt_variants:
            label = f"{model}_{variant}"
            print(f"\n{'='*60}")
            print(f"  RUNNING: whisper={model}  prompt={variant}")
            print(f"{'='*60}")

            t_start = time.time()
            pipeline_result = run_pipeline(
                audio_path=audio_path,
                whisper_model=model,
                prompt_variant=variant,
                generate_audio=False,      # skip TTS for speed
            )
            elapsed = round(time.time() - t_start, 2)

            quality = score_summary_quality(pipeline_result["summary"])

            run_record: dict = {
                "whisper_model":        model,
                "prompt_variant":       variant,
                "stage_latencies_sec":  pipeline_result["stage_times"],
                "total_latency_sec":    elapsed,
                "transcript_word_count": len(pipeline_result["transcript"].split()),
                "keywords_found":       pipeline_result["keywords"],
                "summary_quality":      quality,
            }

            if reference_transcript:
                run_record["wer"] = compute_wer(reference_transcript, pipeline_result["transcript"])

            if expected_summary:
                run_record["action_item_precision"] = score_action_item_precision(
                    pipeline_result["summary"].get("action_items", []),
                    expected_summary.get("action_items", []),
                )

            results["runs"].append(run_record)

            # Save individual summary
            run_file = out_path / f"summary_{label}.json"
            with open(run_file, "w") as f:
                json.dump(pipeline_result["summary"], f, indent=2)
            print(f"  Summary saved → {run_file}")

    # ── Noise robustness sweep ────────────────────────────────────────────────
    if noise_levels and reference_transcript:
        from src.transcribe import transcribe
        import tempfile, os as _os

        print(f"\n{'='*60}")
        print("  NOISE ROBUSTNESS SWEEP")
        print(f"{'='*60}")

        noise_dir = out_path / "noisy_audio"
        noise_dir.mkdir(exist_ok=True)

        for snr in sorted(noise_levels, reverse=True):
            noisy_path = str(noise_dir / f"noisy_{snr}dB.wav")
            actual_path = add_noise_to_audio(audio_path, snr, noisy_path)
            try:
                result = transcribe(actual_path, model_size=whisper_models[0])
                wer = compute_wer(reference_transcript, result["text"])
                print(f"  SNR={snr:>3}dB  WER={wer:.4f}")
                results["noise_robustness"].append({"snr_db": snr, "wer": wer})
            except Exception as e:
                print(f"  SNR={snr}dB  ERROR: {e}")

    # ── Build comparison summary ──────────────────────────────────────────────
    results["comparison"] = {
        "latency_sec": {
            f"{r['whisper_model']}_{r['prompt_variant']}": r["total_latency_sec"]
            for r in results["runs"]
        },
        "summary_quality": {
            f"{r['whisper_model']}_{r['prompt_variant']}": r["summary_quality"]
            for r in results["runs"]
        },
    }
    if any("wer" in r for r in results["runs"]):
        results["comparison"]["wer"] = {
            f"{r['whisper_model']}_{r['prompt_variant']}": r.get("wer", "n/a")
            for r in results["runs"]
        }

    # ── Save JSON report ──────────────────────────────────────────────────────
    json_report = out_path / "evaluation_report.json"
    with open(json_report, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] JSON report saved → {json_report}")

    # ── Save Markdown report ──────────────────────────────────────────────────
    save_markdown_report(results, out_path / "evaluation_report.md")

    # ── Terminal summary table ────────────────────────────────────────────────
    has_wer = any("wer" in r for r in results["runs"])
    headers = ["Run", "Latency", "Fields", "Actions", "Decisions"]
    if has_wer:
        headers.append("WER")

    rows = []
    for r in results["runs"]:
        q   = r["summary_quality"]
        row = [
            f"{r['whisper_model']}_{r['prompt_variant']}",
            f"{r['total_latency_sec']}s",
            q["fields_populated"],
            q["action_items_count"],
            q["key_decisions_count"],
        ]
        if has_wer:
            row.append(r.get("wer", "n/a"))
        rows.append(row)

    print_table("EVALUATION SUMMARY", rows, headers)

    if results["noise_robustness"]:
        noise_rows = [(e["snr_db"], e["wer"]) for e in results["noise_robustness"]]
        print_table("NOISE ROBUSTNESS (WER vs SNR)", noise_rows, ["SNR (dB)", "WER"])

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="One-click evaluation runner for the Meeting Intelligence pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick default run (sample audio, small model, baseline + improved)
  python scripts/run_eval.py

  # Full comparison with WER
  python scripts/run_eval.py --audio audio_samples/sample_meeting.wav \\
      --whisper-models small medium \\
      --reference audio_samples/expected_transcript.txt \\
      --expected-summary audio_samples/expected_summary.json

  # Add noise robustness sweep
  python scripts/run_eval.py --noise-levels 0 5 10 20
        """,
    )
    parser.add_argument(
        "--audio", default="audio_samples/sample_meeting.wav",
        help="Path to input audio file (default: audio_samples/sample_meeting.wav)",
    )
    parser.add_argument(
        "--whisper-models", nargs="+", default=["small"],
        choices=["tiny", "small", "medium"], metavar="MODEL",
        help="Whisper model sizes to evaluate (default: small)",
    )
    parser.add_argument(
        "--prompt-variants", nargs="+", default=["baseline", "improved"],
        choices=["baseline", "improved"], metavar="VARIANT",
        help="Prompt variants to compare (default: baseline improved)",
    )
    parser.add_argument(
        "--reference", default=None,
        help="Path to reference transcript .txt or .json for WER",
    )
    parser.add_argument(
        "--expected-summary", default=None,
        help="Path to expected_summary.json for action-item precision scoring",
    )
    parser.add_argument(
        "--output-dir", default="outputs/evaluation",
        help="Directory for all evaluation outputs (default: outputs/evaluation)",
    )
    parser.add_argument(
        "--noise-levels", nargs="+", type=int, default=[],
        metavar="SNR_DB",
        help="SNR levels in dB for noise robustness sweep (e.g. --noise-levels 0 5 10 20)",
    )

    args = parser.parse_args()

    # Validate audio
    if not Path(args.audio).exists():
        print(f"[Eval] ERROR: Audio file not found: {args.audio}")
        sys.exit(1)

    # Load reference transcript
    reference = None
    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"[Eval] WARNING: Reference file not found: {args.reference}")
        elif ref_path.suffix == ".json":
            data = json.loads(ref_path.read_text())
            reference = data.get("text", "")
        else:
            reference = ref_path.read_text().strip()

    # Load expected summary
    expected_summary = None
    if args.expected_summary:
        es_path = Path(args.expected_summary)
        if es_path.exists():
            expected_summary = json.loads(es_path.read_text())
        else:
            print(f"[Eval] WARNING: Expected summary not found: {args.expected_summary}")

    print("\n" + "="*60)
    print("  Meeting Intelligence — Evaluation Runner")
    print(f"  Audio:   {args.audio}")
    print(f"  Models:  {args.whisper_models}")
    print(f"  Prompts: {args.prompt_variants}")
    print(f"  Output:  {args.output_dir}")
    print("="*60)

    run_evaluation(
        audio_path=args.audio,
        whisper_models=args.whisper_models,
        prompt_variants=args.prompt_variants,
        reference_transcript=reference,
        expected_summary=expected_summary,
        output_dir=args.output_dir,
        noise_levels=args.noise_levels,
    )


if __name__ == "__main__":
    main()
