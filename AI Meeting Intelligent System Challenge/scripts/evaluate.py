#!/usr/bin/env python3
"""
evaluate.py — One-click evaluation runner for the Meeting Intelligence pipeline.

Usage:
    python scripts/evaluate.py --audio audio_samples/sample_meeting.wav
    python scripts/evaluate.py --audio meeting.wav --whisper-models small medium
    python scripts/evaluate.py --audio meeting.wav --reference audio_samples/expected_transcript.txt
    python scripts/evaluate.py --help
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: (substitutions + deletions + insertions) / reference_words."""
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


def _score_summary_quality(summary: dict) -> dict:
    """Simple rubric: check presence and non-emptiness of expected fields."""
    fields = ["executive_summary", "key_decisions", "action_items", "topics_covered",
              "unresolved_questions", "meeting_tone"]
    present = sum(1 for f in fields if summary.get(f))
    action_has_owner = all("owner" in item for item in summary.get("action_items", []))
    return {
        "fields_populated": f"{present}/{len(fields)}",
        "action_items_count": len(summary.get("action_items", [])),
        "key_decisions_count": len(summary.get("key_decisions", [])),
        "action_items_well_formed": action_has_owner,
    }


def run_evaluation(
    audio_path: str,
    whisper_models: list,
    prompt_variants: list,
    reference_transcript: str | None,
    output_dir: str,
) -> dict:
    results = {"audio": audio_path, "runs": [], "comparison": {}}
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for model in whisper_models:
        for variant in prompt_variants:
            print(f"\n{'='*60}")
            print(f"  whisper={model}  prompt={variant}")
            print(f"{'='*60}")

            t_start = time.time()
            pipeline_result = run_pipeline(
                audio_path=audio_path,
                whisper_model=model,
                prompt_variant=variant,
                generate_audio=False,
            )
            elapsed = round(time.time() - t_start, 2)

            run_record = {
                "whisper_model": model,
                "prompt_variant": variant,
                "stage_latencies_sec": pipeline_result["stage_times"],
                "total_latency_sec": elapsed,
                "transcript_word_count": len(pipeline_result["transcript"].split()),
                "keywords_found": pipeline_result["keywords"],
                "summary_quality": _score_summary_quality(pipeline_result["summary"]),
            }

            if reference_transcript:
                run_record["wer"] = _compute_wer(reference_transcript, pipeline_result["transcript"])

            results["runs"].append(run_record)

            run_file = out_path / f"summary_{model}_{variant}.json"
            with open(run_file, "w") as f:
                json.dump(pipeline_result["summary"], f, indent=2)
            print(f"  Summary saved → {run_file}")

    # Build comparison table
    results["comparison"]["latency_sec"] = {
        f"{r['whisper_model']}_{r['prompt_variant']}": r["total_latency_sec"]
        for r in results["runs"]
    }
    results["comparison"]["summary_quality"] = {
        f"{r['whisper_model']}_{r['prompt_variant']}": r["summary_quality"]
        for r in results["runs"]
    }
    if any("wer" in r for r in results["runs"]):
        results["comparison"]["wer"] = {
            f"{r['whisper_model']}_{r['prompt_variant']}": r.get("wer", "n/a")
            for r in results["runs"]
        }

    report_file = out_path / "evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Report saved → {report_file}")

    # Print results table
    print(f"\n{'='*60}")
    print("  EVALUATION RESULTS")
    print(f"{'='*60}")
    has_wer = any("wer" in r for r in results["runs"])
    header = f"  {'Run':<30} {'Latency':>10}" + (f" {'WER':>8}" if has_wer else "")
    print(header)
    print(f"  {'-'*55}")
    for r in results["runs"]:
        run_name = f"{r['whisper_model']}_{r['prompt_variant']}"
        wer_col = f" {r.get('wer', 'n/a'):>8}" if has_wer else ""
        print(f"  {run_name:<30} {r['total_latency_sec']:>9}s{wer_col}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="One-click evaluation runner for Meeting Intelligence pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py --audio audio_samples/sample_meeting.wav
  python scripts/evaluate.py --audio meeting.wav --whisper-models small medium
  python scripts/evaluate.py --audio meeting.wav --prompt-variants baseline improved --reference audio_samples/expected_transcript.txt
        """,
    )
    parser.add_argument("--audio", required=True, help="Path to input audio file")
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
        help="Path to reference transcript .txt for WER calculation",
    )
    parser.add_argument(
        "--output-dir", default="outputs/evaluation",
        help="Directory for evaluation outputs (default: outputs/evaluation)",
    )

    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"[Eval] ERROR: Audio file not found: {args.audio}")
        sys.exit(1)

    reference = None
    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"[Eval] WARNING: Reference file not found: {args.reference}")
        else:
            reference = ref_path.read_text().strip()

    run_evaluation(
        audio_path=args.audio,
        whisper_models=args.whisper_models,
        prompt_variants=args.prompt_variants,
        reference_transcript=reference,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
