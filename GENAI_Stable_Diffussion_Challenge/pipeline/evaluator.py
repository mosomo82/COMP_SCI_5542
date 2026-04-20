"""
evaluator.py
------------
Computes evaluation metrics for generated product images.

Metrics implemented
-------------------
1. CLIP Score          — measures prompt-image alignment (primary, quantitative)
2. Consistency Score   — average pairwise cosine similarity of views per product
3. Diversity Score     — std-dev of CLIP embeddings (naive vs structured)
4. Human Rating Stub   — qualitative template (1-5 scale, filled manually)

Output
------
- results/evaluation_report.csv   : full per-image metrics table
- results/summary.csv             : per-product aggregated stats
- results/evaluation_report.html  : styled per-image report (Colab/notebook friendly)
- results/summary.html            : styled per-product summary (Colab/notebook friendly)
"""
import csv
import json
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from rich.console import Console
import pandas as pd
from rich.table import Table

console = Console()

QUALITY_SCORES_FILE = "quality_scores.csv"


# ---------------------------------------------------------------------------
# CLIP model loader (singleton — loaded once)
# ---------------------------------------------------------------------------
_clip_model     = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        console.print("[cyan]Loading CLIP model for evaluation…[/cyan]")
        _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        console.print("[green]✓ CLIP model ready[/green]")
    return _clip_model, _clip_processor


# ---------------------------------------------------------------------------
# CLIP Score
# ---------------------------------------------------------------------------

def clip_score(image: Image.Image, prompt: str) -> float:
    """
    Compute CLIP cosine similarity between an image and a text prompt.
    Returns a float in [0, 1] — higher means better prompt alignment.
    """
    model, processor = _load_clip()
    inputs  = processor(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        outputs = model(**inputs)
    # logits_per_image is the raw dot product; softmax gives probability
    score = outputs.logits_per_image.squeeze().item()
    # Normalise to [0, 1] range for easier reading
    return round(float(score) / 100.0, 4)


def _image_embedding(image: Image.Image) -> np.ndarray:
    """Return the CLIP visual embedding as a numpy vector."""
    model, processor = _load_clip()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        # Use vision_model directly — get_image_features returns
        # BaseModelOutputWithPooling in newer transformers versions
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        emb = model.visual_projection(vision_out.pooler_output)
    emb = emb / emb.norm(dim=-1, keepdim=True)   # L2 normalise
    return emb.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Consistency Score (pairwise cosine similarity across views)
# ---------------------------------------------------------------------------

def consistency_score(image_paths: list) -> float:
    """
    Measure how visually consistent a set of generated views are.
    Computes the mean pairwise cosine similarity of their CLIP embeddings.
    Range: [0, 1] — 1.0 means all views look identical.
    """
    if len(image_paths) < 2:
        return 1.0

    embeddings = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        embeddings.append(_image_embedding(img))

    embeddings = np.stack(embeddings)  # (N, D)
    scores = []
    n = len(embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            cos = float(np.dot(embeddings[i], embeddings[j]))
            scores.append(cos)
    return round(float(np.mean(scores)), 4)


# ---------------------------------------------------------------------------
# Diversity Score (std-dev of CLIP embeddings)
# ---------------------------------------------------------------------------

def diversity_score(image_paths: list) -> float:
    """
    Measure the diversity of a set of generated views.
    Computes the std-dev of their CLIP embeddings across all dimensions.
    Range: [0, inf) — higher means more diverse (less similar to each other).
    """
    if len(image_paths) < 2:
        return 0.0

    embeddings = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        embeddings.append(_image_embedding(img))

    embeddings = np.stack(embeddings)  # (N, D)
    # Compute std-dev of all embeddings across all dimensions
    diversity = float(np.std(embeddings))
    return round(diversity, 4)


def _normalize_image_key(path_str: str) -> str:
    return Path(path_str).name.strip().lower()


def _load_quality_scores(output_dir: Path) -> dict[str, dict]:
    quality_path = output_dir / QUALITY_SCORES_FILE
    if not quality_path.exists():
        return {}

    with open(quality_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        scores = {}
        for row in reader:
            image_key = row.get("image_name") or row.get("image_path")
            if not image_key:
                continue
            scores[_normalize_image_key(image_key)] = row
        return scores


# ---------------------------------------------------------------------------
# Full evaluation over all generation results
# ---------------------------------------------------------------------------

def evaluate_results(
    generation_results: list,
    output_dir: Path,
) -> list:
    """
    Run all metrics on every generated image.

    Parameters
    ----------
    generation_results : list returned by generator.run_all_products()
    output_dir         : where to save CSV reports

    Returns
    -------
    List of enriched result dicts with metric columns added.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated = []
    quality_scores = _load_quality_scores(output_dir)

    total = len(generation_results)
    console.print(f"\n[cyan]Evaluating {total} images…[/cyan]")

    for idx, result in enumerate(generation_results, 1):
        console.print(
            f"  [{idx}/{total}] Scoring [dim]{Path(result['image_path']).name}[/dim]"
        )
        img   = Image.open(result["image_path"]).convert("RGB")
        score = clip_score(img, result["prompt"])
        quality_row = quality_scores.get(_normalize_image_key(result["image_path"]), {})
        quality_score_value = quality_row.get("quality_score", "")
        try:
            quality_score_value = float(quality_score_value) if quality_score_value != "" else None
        except ValueError:
            quality_score_value = None
        enriched = {
            **result,
            "clip_score": score,
            "quality_score": quality_score_value,
            "quality_notes": quality_row.get("quality_notes", ""),
        }
        evaluated.append(enriched)

    # -----------------------------------------------------------------------
    # Consistency & Diversity: group by (product_id, prompt_type)
    # -----------------------------------------------------------------------
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in evaluated:
        key = (r["product_id"], r["prompt_type"])
        groups[key].append(r["image_path"])

    for r in evaluated:
        key = (r["product_id"], r["prompt_type"])
        r["consistency_score"] = consistency_score(groups[key])
        r["diversity_score"] = diversity_score(groups[key])

    # -----------------------------------------------------------------------
    # Save per-image report
    # -----------------------------------------------------------------------
    csv_path = output_dir / "evaluation_report.csv"
    fieldnames = [
        "product_id", "product_title", "prompt_type",
        "clip_score", "consistency_score", "diversity_score", "quality_score", "quality_notes", "gen_time_s",
        "seed", "image_path", "prompt",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evaluated)
    console.print(f"\n[green]✓ Evaluation report saved:[/green] {csv_path}")

    # -----------------------------------------------------------------------
    # Save summary and styled reports
    # -----------------------------------------------------------------------
    summary_rows = _save_summary(evaluated, output_dir)
    _save_styled_reports(evaluated, summary_rows, output_dir)
    _print_comparison_table(evaluated)

    return evaluated


def _save_summary(evaluated: list, output_dir: Path):
    """Aggregate metrics per (product_id, prompt_type) and write summary CSV."""
    from collections import defaultdict

    agg: dict = defaultdict(list)
    for r in evaluated:
        key = (r["product_id"], r["product_title"], r["prompt_type"])
        agg[key].append(r)

    rows = []
    for (pid, title, ptype), records in agg.items():
        clips   = [r["clip_score"]        for r in records]
        consis  = [r["consistency_score"] for r in records]
        divers  = [r["diversity_score"]   for r in records]
        quality = [r["quality_score"] for r in records if r.get("quality_score") is not None]
        times   = [r["gen_time_s"]        for r in records]
        rows.append({
            "product_id":         pid,
            "product_title":      title[:40],
            "prompt_type":        ptype,
            "mean_clip_score":    round(float(np.mean(clips)),  4),
            "mean_consistency":   round(float(np.mean(consis)), 4),
            "mean_diversity":     round(float(np.mean(divers)), 4),
            "mean_quality_score": round(float(np.mean(quality)), 2) if quality else None,
            "n_quality_scored":   len(quality),
            "mean_gen_time_s":    round(float(np.mean(times)),  2),
            "n_images":           len(records),
        })

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"[green]✓ Summary saved:[/green] {summary_path}")
    
    return rows


def _save_styled_reports(evaluated: list, summary_rows: list, output_dir: Path):
    """Save notebook-friendly styled HTML versions of the CSV reports."""
    report_df = pd.DataFrame(evaluated)
    summary_df = pd.DataFrame(summary_rows)

    report_cols = [
        "product_id", "product_title", "prompt_type", "clip_score",
        "consistency_score", "diversity_score", "quality_score", "quality_notes", "gen_time_s", "seed", "image_path", "prompt",
    ]
    summary_cols = [
        "product_id", "product_title", "prompt_type", "mean_clip_score",
        "mean_consistency", "mean_diversity", "mean_quality_score", "n_quality_scored", "mean_gen_time_s", "n_images",
    ]

    if not report_df.empty:
        report_df = report_df[report_cols]
    if not summary_df.empty:
        summary_df = summary_df[summary_cols]

    style_common = [
        {
            "selector": "th",
            "props": "background-color: #0b3c5d; color: white; font-weight: 600; text-align: center;",
        },
        {
            "selector": "td",
            "props": "font-size: 13px;",
        },
        {
            "selector": "tr:nth-child(even)",
            "props": "background-color: #f8fafc;",
        },
        {
            "selector": "caption",
            "props": "caption-side: top; font-size: 16px; font-weight: 700; color: #0b3c5d; padding: 8px;",
        },
    ]

    if not report_df.empty:
        report_styler = (
            report_df.style
            .format({
                "clip_score": "{:.4f}",
                "consistency_score": "{:.4f}",
                "diversity_score": "{:.4f}",
                "quality_score": "{:.1f}",
                "gen_time_s": "{:.2f}",
            })
            .background_gradient(subset=["clip_score"], cmap="YlGn")
            .background_gradient(subset=["consistency_score"], cmap="PuBuGn")
            .background_gradient(subset=["diversity_score"], cmap="OrRd")
            .background_gradient(subset=["quality_score"], cmap="YlOrBr")
            .set_properties(subset=["prompt", "image_path", "quality_notes"], **{"max-width": "460px", "white-space": "normal"})
            .set_caption("Evaluation Report (Per Image)")
            .set_table_styles(style_common)
            .hide(axis="index")
        )
        report_html_path = output_dir / "evaluation_report.html"
        report_styler.to_html(report_html_path)
        console.print(f"[green]✓ Styled report saved:[/green] {report_html_path}")

    if not summary_df.empty:
        summary_styler = (
            summary_df.style
            .format({
                "mean_clip_score": "{:.4f}",
                "mean_consistency": "{:.4f}",
                "mean_diversity": "{:.4f}",
                "mean_quality_score": "{:.2f}",
                "mean_gen_time_s": "{:.2f}",
            })
            .background_gradient(subset=["mean_clip_score"], cmap="YlGn")
            .background_gradient(subset=["mean_consistency"], cmap="PuBuGn")
            .background_gradient(subset=["mean_diversity"], cmap="OrRd")
            .background_gradient(subset=["mean_quality_score"], cmap="YlOrBr")
            .bar(subset=["mean_gen_time_s"], color="#f4a261")
            .set_caption("Summary Report (Per Product and Prompt Type)")
            .set_table_styles(style_common)
            .hide(axis="index")
        )
        summary_html_path = output_dir / "summary.html"
        summary_styler.to_html(summary_html_path)
        console.print(f"[green]✓ Styled summary saved:[/green] {summary_html_path}")


def _print_comparison_table(evaluated: list):
    """Print a rich CLI table comparing naive vs structured prompt metrics."""
    from collections import defaultdict

    by_product: dict = defaultdict(dict)
    for r in evaluated:
        pid    = r["product_id"]
        ptype  = r["prompt_type"]
        key_cs = f"{ptype}_clip"
        key_co = f"{ptype}_consistency"
        key_dv = f"{ptype}_diversity"
        key_qs = f"{ptype}_quality"
        clips  = by_product[pid].setdefault(key_cs, [])
        cohere = by_product[pid].setdefault(key_co, [])
        divers = by_product[pid].setdefault(key_dv, [])
        quality = by_product[pid].setdefault(key_qs, [])
        clips.append(r["clip_score"])
        cohere.append(r["consistency_score"])
        divers.append(r["diversity_score"])
        if r.get("quality_score") is not None:
            quality.append(r["quality_score"])
        by_product[pid]["title"] = r["product_title"][:30]

    table = Table(
        title="📊 Naive vs Structured — Evaluation Summary",
        show_lines=True,
    )
    table.add_column("Product", style="cyan", max_width=32)
    table.add_column("CLIP (Naive)", justify="right")
    table.add_column("CLIP (Struct.)", justify="right", style="green")
    table.add_column("Δ CLIP", justify="right", style="bold")
    table.add_column("Consist. (Naive)", justify="right")
    table.add_column("Consist. (Struct.)", justify="right", style="green")
    table.add_column("Diversity (Naive)", justify="right")
    table.add_column("Diversity (Struct.)", justify="right", style="green")
    table.add_column("Quality (Naive)", justify="right")
    table.add_column("Quality (Struct.)", justify="right", style="green")

    for pid, data in sorted(by_product.items()):
        naive_clip  = np.mean(data.get("naive_clip",      [0]))
        struct_clip = np.mean(data.get("structured_clip", [0]))
        naive_con   = np.mean(data.get("naive_consistency",      [0]))
        struct_con  = np.mean(data.get("structured_consistency", [0]))
        naive_div   = np.mean(data.get("naive_diversity",      [0]))
        struct_div  = np.mean(data.get("structured_diversity", [0]))
        naive_q     = data.get("naive_quality", [])
        struct_q    = data.get("structured_quality", [])
        delta       = struct_clip - naive_clip
        delta_str   = f"[green]+{delta:.4f}[/green]" if delta > 0 else f"[red]{delta:.4f}[/red]"
        naive_q_str = f"{np.mean(naive_q):.2f}" if naive_q else "-"
        struct_q_str = f"{np.mean(struct_q):.2f}" if struct_q else "-"
        table.add_row(
            data.get("title", pid),
            f"{naive_clip:.4f}",
            f"{struct_clip:.4f}",
            delta_str,
            f"{naive_con:.4f}",
            f"{struct_con:.4f}",
            f"{naive_div:.4f}",
            f"{struct_div:.4f}",
            naive_q_str,
            struct_q_str,
        )

    console.print(table)
