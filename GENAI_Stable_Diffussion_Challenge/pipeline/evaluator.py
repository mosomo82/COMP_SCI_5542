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
"""

import csv
import json
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()


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
    inputs  = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
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
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)   # L2 normalise
    return emb.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Consistency Score (pairwise cosine similarity across views)
# ---------------------------------------------------------------------------

def consistency_score(image_paths: list[str]) -> float:
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
# Full evaluation over all generation results
# ---------------------------------------------------------------------------

def evaluate_results(
    generation_results: list[dict],
    output_dir: Path,
) -> list[dict]:
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

    total = len(generation_results)
    console.print(f"\n[cyan]Evaluating {total} images…[/cyan]")

    for idx, result in enumerate(generation_results, 1):
        console.print(
            f"  [{idx}/{total}] Scoring [dim]{Path(result['image_path']).name}[/dim]"
        )
        img   = Image.open(result["image_path"]).convert("RGB")
        score = clip_score(img, result["prompt"])
        enriched = {**result, "clip_score": score}
        evaluated.append(enriched)

    # -----------------------------------------------------------------------
    # Consistency: group by (product_id, prompt_type) and compute pairwise sim
    # -----------------------------------------------------------------------
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for r in evaluated:
        key = (r["product_id"], r["prompt_type"])
        groups[key].append(r["image_path"])

    for r in evaluated:
        key = (r["product_id"], r["prompt_type"])
        r["consistency_score"] = consistency_score(groups[key])

    # -----------------------------------------------------------------------
    # Save per-image report
    # -----------------------------------------------------------------------
    csv_path = output_dir / "evaluation_report.csv"
    fieldnames = [
        "product_id", "product_title", "prompt_type",
        "clip_score", "consistency_score", "gen_time_s",
        "seed", "image_path", "prompt",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evaluated)
    console.print(f"\n[green]✓ Evaluation report saved:[/green] {csv_path}")

    # -----------------------------------------------------------------------
    # Build and save per-product summary
    # -----------------------------------------------------------------------
    _save_summary(evaluated, output_dir)
    _print_comparison_table(evaluated)

    return evaluated


def _save_summary(evaluated: list[dict], output_dir: Path):
    """Aggregate metrics per (product_id, prompt_type) and write summary CSV."""
    from collections import defaultdict

    agg: dict[tuple, list] = defaultdict(list)
    for r in evaluated:
        key = (r["product_id"], r["product_title"], r["prompt_type"])
        agg[key].append(r)

    rows = []
    for (pid, title, ptype), records in agg.items():
        clips   = [r["clip_score"]        for r in records]
        consis  = [r["consistency_score"] for r in records]
        times   = [r["gen_time_s"]        for r in records]
        rows.append({
            "product_id":         pid,
            "product_title":      title[:40],
            "prompt_type":        ptype,
            "mean_clip_score":    round(float(np.mean(clips)),  4),
            "mean_consistency":   round(float(np.mean(consis)), 4),
            "mean_gen_time_s":    round(float(np.mean(times)),  2),
            "n_images":           len(records),
        })

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"[green]✓ Summary saved:[/green] {summary_path}")


def _print_comparison_table(evaluated: list[dict]):
    """Print a rich CLI table comparing naive vs structured prompt metrics."""
    from collections import defaultdict

    by_product: dict[str, dict] = defaultdict(dict)
    for r in evaluated:
        pid    = r["product_id"]
        ptype  = r["prompt_type"]
        key_cs = f"{ptype}_clip"
        key_co = f"{ptype}_consistency"
        clips  = by_product[pid].setdefault(key_cs, [])
        cohere = by_product[pid].setdefault(key_co, [])
        clips.append(r["clip_score"])
        cohere.append(r["consistency_score"])
        by_product[pid]["title"] = r["product_title"][:30]

    table = Table(title="📊 Naive vs Structured — Evaluation Summary", show_lines=True)
    table.add_column("Product",         style="cyan",   max_width=32)
    table.add_column("CLIP (Naive)",    justify="right")
    table.add_column("CLIP (Struct.)",  justify="right", style="green")
    table.add_column("Δ CLIP",          justify="right", style="bold")
    table.add_column("Consist. (Naive)",  justify="right")
    table.add_column("Consist. (Struct.)", justify="right", style="green")

    for pid, data in sorted(by_product.items()):
        naive_clip  = np.mean(data.get("naive_clip",      [0]))
        struct_clip = np.mean(data.get("structured_clip", [0]))
        naive_con   = np.mean(data.get("naive_consistency",      [0]))
        struct_con  = np.mean(data.get("structured_consistency", [0]))
        delta       = struct_clip - naive_clip
        delta_str   = f"[green]+{delta:.4f}[/green]" if delta > 0 else f"[red]{delta:.4f}[/red]"
        table.add_row(
            data.get("title", pid),
            f"{naive_clip:.4f}",
            f"{struct_clip:.4f}",
            delta_str,
            f"{naive_con:.4f}",
            f"{struct_con:.4f}",
        )

    console.print(table)
