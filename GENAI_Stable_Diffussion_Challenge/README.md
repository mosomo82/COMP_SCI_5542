# CS 5542 — GenAI: Stable Diffusion E-Commerce Image Generation Pipeline

> **Quiz Challenge · 5% credit · Due April 20, 2026**

A local Python pipeline that converts product metadata into high-quality e-commerce product images using Stable Diffusion. Compares **naive prompts** (raw title only) vs **structured prompts** (rich metadata templates) and evaluates with CLIP scores.

---

## Project Structure

```
GENAI_Stable_Diffusion_Challenge/
├── run_pipeline.py          # ← main entry point (CLI)
├── requirements.txt
├── data/
│   └── products.json        # sample product metadata (10 products)
├── pipeline/
│   ├── __init__.py
│   ├── prompt_builder.py    # naive vs structured prompt strategies
│   ├── sd_pipeline.py       # loads SD / ControlNet pipelines
│   ├── generator.py         # generates images, saves PNGs
│   └── evaluator.py         # CLIP score + consistency metrics + CSV reports
├── outputs/                 # generated images (auto-created)
│   └── P001/
│       ├── P001_naive_view01_seed42.png
│       ├── P001_structured_view01_seed42.png
│       └── ...
└── results/                 # evaluation reports (auto-created)
    ├── evaluation_report.csv
    └── summary.csv
```

---

## Setup

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU required** for reasonable speed.  
> SD 1.5 needs ~4 GB VRAM · ControlNet needs ~6 GB · SDXL needs ~8 GB.  
> CPU works but is extremely slow (~10 min/image).

---

## Usage

### Quick test (first 3 products, 2 images each)
```bash
python run_pipeline.py --mode sd --limit 3 --n-images 2
```

### Full run — all 10 products, 4 views each
```bash
python run_pipeline.py --mode sd
```

### Use SDXL (better quality, needs 8 GB VRAM)
```bash
python run_pipeline.py --mode sd --sdxl
```

### Use ControlNet for structural consistency
```bash
python run_pipeline.py --mode controlnet --reference data/reference.jpg
```

### Evaluate existing outputs only (skip generation)
```bash
python run_pipeline.py --eval-only
```

### All flags
| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `sd` | `sd` or `controlnet` |
| `--sdxl` | off | Use SDXL instead of SD 1.5 |
| `--reference` | — | Reference image path (ControlNet only) |
| `--n-images` | 4 | Views per product per prompt type |
| `--steps` | 30 | Denoising steps (30–50 recommended) |
| `--cfg` | 7.5 | Guidance scale (7–9 recommended) |
| `--height/--width` | 512 | Image resolution |
| `--seed` | 42 | Base random seed |
| `--limit` | all | Process only first N products |
| `--eval-only` | off | Skip generation, evaluate existing images |
| `--skip-eval` | off | Skip evaluation after generation |

---

## Outputs

**Images** are saved in `outputs/<product_id>/`:
```
P001_naive_view01_seed42.png
P001_naive_view02_seed43.png
P001_structured_view01_seed42.png
P001_structured_view02_seed43.png
```

**Evaluation reports** in `results/`:
- `evaluation_report.csv` — per-image CLIP score + consistency score
- `summary.csv` — per-product averages comparing naive vs structured

---

## Prompt Strategies

### Naive (Baseline)
```
Running Sneakers with Mesh Upper
```

### Structured (Improved)
```
Professional studio product photography of Navy Blue and White Breathable Mesh, Rubber Sole Running Sneakers with Mesh Upper, category: Athletic Footwear, style: sporty, lightweight, modern, studio lighting, pure white background, sharp focus, 8K ultra resolution, e-commerce style, centered product, no shadows, photorealistic, commercial quality
```

### Negative Prompt (applied to all)
```
blurry, low quality, watermark, text, distorted, dark background, cartoon, ...
```

---

## Evaluation Metrics

| Metric | Method | Notes |
|--------|--------|-------|
| CLIP Score | `openai/clip-vit-base-patch32` | Prompt-image alignment, 0–1 |
| Consistency | Pairwise cosine sim of CLIP embeddings | Across 4 views, 0–1 |
| Human Quality | 1–5 scale | Fill in `results/evaluation_report.csv` manually |

---

## Tools & AI Used

- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Antigravity AI](https://antigravity.ai) — assisted with code structure and documentation
- [ChatGPT](https://chat.openai.com) — assisted with prompt template design

---

## Dataset

Sample data in `data/products.json` — 10 products across categories:
Athletic Footwear, Handbags, Watches, Jackets, T-Shirts, Furniture, Electronics, Pants, Kitchen, Shoes.

For larger experiments, use the [Amazon Product Dataset](https://nijianmo.github.io/amazon/index.html).
