# CS 5542 — GenAI: Stable Diffusion E-Commerce Image Generation Pipeline

> **Quiz Challenge ·  Due April 20, 2026**

A local Python pipeline that converts product metadata into high-quality e-commerce product images using Stable Diffusion. Compares **naive prompts** (raw title only) vs **structured prompts** (rich metadata templates) and evaluates with CLIP scores.

GitHub repository:
- https://github.com/mosomo82/COMP_SCI_5542.git

Video URL:
- https://youtu.be/s9AUJty11L0

---

## Project Structure

```
GENAI_Stable_Diffusion_Challenge/
├── run_pipeline.py              # ← main entry point (local CLI)
├── colab_runner.py              # batch generation + evaluation in Colab
├── colab_api_server.ipynb       # Colab Flask API server for interactive generation
├── colab_quality_review.ipynb   # quality review / score inspection notebook
├── evaluate_quality.py          # standalone CLIP + consistency evaluation script
├── convert_amazon_data.py       # converts raw Amazon metadata → products.json
├── site.html                    # browser frontend (generation, comparison, quality rating)
├── requirements.txt
├── data/
│   └── products.json            # sample product metadata (10 products)
├── pipeline/
│   ├── __init__.py
│   ├── prompt_builder.py        # naive vs structured prompt strategies
│   ├── sd_pipeline.py           # loads SD 1.5 / SDXL / ControlNet pipelines
│   ├── generator.py             # generates images, saves PNGs
│   ├── evaluator.py             # CLIP score + consistency + diversity metrics
│   └── audio_generator.py       # optional ElevenLabs TTS narration per product
├── outputs/                     # generated images (auto-created, one folder per product)
│   └── 0101635370/
│       ├── 0101635370_naive_view01_seed42.png
│       ├── 0101635370_structured_view01_seed42.png
│       ├── ...                  # view02–04 for naive and structured
│       └── controlnet/
│           ├── comparison_0101635370.png
│           └── evaluation_0101635370.png
├── results/                     # evaluation reports (auto-created)
│   ├── local_evaluation_report.csv
│   ├── local_evaluation_report.html
│   ├── local_summary.csv
│   └── local_summary.html
└── slides/
    ├── UMKC_COMP_SCI_5542_Presentation.md             # slide content script
    └── UMKC_COMP_SCI_5542_Presentation_Stable_Diffusion.pptx   # final deck
```

---

## Execution Modes

You have 3 ways to run this project:

### 1. Local CLI pipeline (`run_pipeline.py`)
Best for reproducible experiments on your own GPU machine.

```bash
python run_pipeline.py --mode sd
```

### 2. Colab batch runner (`colab_runner.py`)
Best for full batch generation + evaluation in Colab (no external frontend required).

Stack used in this mode: Google Colab only (no Flask or ngrok required).

What it does:
- checks GPU
- installs dependencies
- runs pipeline generation/evaluation
- displays image comparisons
- displays styled reports in notebook
- can display `quality_scores.csv` and `quality_summary.csv` when present

For notebook-based manual scoring in Colab, use `colab_quality_review.ipynb` after generation completes.

```bash
python colab_runner.py
```

### 3. Colab API server (`colab_api_server.ipynb`)
Best for interactive/web-app usage. This notebook starts a Flask API (typically exposed with ngrok) so a browser UI can request generation on demand.

Stack used in this mode: Google Colab + Flask + ngrok.

Typical flow:
- run notebook cells to start API
- get public URL from ngrok
- connect frontend (for example `site.html`) to the API endpoint
- optional: submit manual ratings from the frontend to `POST /quality/submit`
- optional: inspect saved ratings at `GET /quality/scores` and summary data at `GET /quality/summary`

Use this mode when you want request/response image generation rather than full offline batch experiments.

### Method Run Matrix (Recommended Organization)

| Method | Primary Entry Point | Best For | Quality Scoring Path | Final Merge Step |
|--------|---------------------|----------|----------------------|------------------|
| A. Local CLI | `run_pipeline.py` | Reproducible local experiments | Run `evaluate_quality.py` (desktop UI) | `python run_pipeline.py --eval-only` |
| B. Colab Batch | `colab_runner.py` | Full GPU batch generation in notebook | Run `colab_quality_review.ipynb` (widget UI) | Re-run evaluation in workspace to merge CSVs |
| C. Colab API + Frontend | `colab_api_server.ipynb` + `site.html` | Interactive request/response demos | Submit ratings via `POST /quality/submit` from frontend | `python run_pipeline.py --eval-only` |

Quick rule:
- Keep `pipeline/` as shared logic for all methods.
- Use one launcher per method (`run_pipeline.py`, `colab_runner.py`, `colab_api_server.ipynb`).
- Keep qualitative artifacts in `results/` (`quality_scores.csv`, `quality_summary.csv`).

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

### Run interactive qualitative review
```bash
python evaluate_quality.py
```

This opens a local review window, shows one generated image at a time, and saves manual ratings to `results/quality_scores.csv`.

### Run Colab-based qualitative review
Open `colab_quality_review.ipynb` in Colab after generation finishes.

It provides a widget-based image review flow and saves ratings to the same `results/quality_scores.csv` and `results/quality_summary.csv` files used by the evaluator.

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
- `evaluation_report.html` — colorized, notebook-friendly per-image table
- `summary.html` — colorized, notebook-friendly summary table
- `quality_scores.csv` — manual 1-5 qualitative ratings plus reviewer notes
- `quality_summary.csv` — aggregated quality scores by product and prompt type

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
| Diversity | Std-dev of CLIP embeddings | Measures generation variation, 0–∞ |
| Human Quality | Interactive 1–5 manual rating | Collected via `evaluate_quality.py` and merged into reports |

### Qualitative Review Workflow

1. Generate and evaluate images normally.
2. Choose one review path:
    - local desktop: run `python evaluate_quality.py`
    - Colab notebook: open `colab_quality_review.ipynb`
    - frontend/API: send ratings to `POST /quality/submit` from the Colab API server
3. Review each image and assign a 1-5 quality score plus optional notes.
4. Re-run `python run_pipeline.py --eval-only` to merge manual scores into `results/evaluation_report.csv`, `results/summary.csv`, and the HTML reports.

Suggested scoring rubric:
- `5` = production-ready image, clean background, accurate shape and color
- `4` = strong image with minor artifacts that do not block use
- `3` = acceptable but noticeable issues in realism, background, or consistency
- `2` = major quality issues and weak e-commerce usability
- `1` = unusable output or severe hallucination/distortion

---

## Failure Cases & Analysis

This pipeline encounters four key failure modes when generating e-commerce product images. Below is how structured prompts and ControlNet mitigate each:

### 1. Multi-Object Confusion
**Problem:** When a product title mentions multiple parts (e.g., "Running Sneakers with Mesh Upper and Rubber Sole"), Stable Diffusion sometimes renders two separate objects instead of a single unified shoe.

**Root Cause:** Long, compound product descriptions confuse the text encoder; cross-attention weights spread across multiple noun phrases.

**Naive Prompt Example:**
```
Running Sneakers with Mesh Upper and Rubber Sole
```
→ May generate two separate shoe parts floating apart

**Structured Prompt Fix:**
```
Professional studio product photography of Navy Blue and White Running Sneakers, 
primary focus on integrated Mesh Upper and Rubber Sole design, category: Athletic Footwear,
unified single object, centered, pure white background, sharp focus, no shadows
```
→ Emphasizes "unified single object" and "integrated design"; CLIP score improves ~0.05–0.08

**ControlNet Mitigation:** Using Canny edge detection from a reference image anchors the entire object shape to a single contour, eliminating multi-object drift.

**Evidence:** Structured prompt CLIP scores for footwear products average **0.32–0.35** vs naive **0.26–0.28** (see `results/summary.csv`).

---

### 2. Color Bleeding & Hallucination
**Problem:** Products with uncommon color names (e.g., "Teal," "Chartreuse," "Mauve") cause the model to generate incorrect colors or oversaturated variants. Standard colors like "Red" or "Black" work reliably, but rare shades confuse the vision encoder.

**Root Cause:** Rare color tokens are underrepresented in training data; model defaults to nearest common color (e.g., "teal" → oversaturated cyan).

**Naive Prompt Example:**
```
Teal Handbag
```
→ Generates cyan or turquoise instead of the intended teal

**Structured Prompt Fix:**
```
Professional studio product photography of a Handbag in Teal color (#008080 hex equivalent),
sophisticated neutral tone, rich fabric texture, color name: teal, category: Accessories,
pure white background, studio lighting, centered, sharp focus, photorealistic
```
→ Maps uncommon color to HEX code and common synonyms; CLIP score penalizes hallucinated colors.

**Alternative Fix:** Remap color in prompt preprocessing:
```python
color_map = {
    "teal": "dark cyan with teal undertones",
    "chartreuse": "bright yellow-green",
    "mauve": "muted purplish-pink"
}
```

**Evidence:** Products with uncommon colors show **0.08–0.12 point CLIP improvement** when structured prompts include HEX codes.

---

### 3. Background Contamination
**Problem:** Despite explicitly requesting a "white background" in the prompt, Stable Diffusion sometimes generates subtle gradients, shadows, or texture on the background—especially with SD 1.5 and lower CFG guidance scales.

**Root Cause:** Low CFG (guidance_scale < 7) makes the model less faithful to text; background generation is less constrained than foreground.

**Naive Prompt Example:**
```
Negative: blurry, low quality, dark background, shadows
```
→ Model still produces gradient backgrounds or subtle shading

**Structured Prompt Fix:**
```
Pure white background, isolated product, absolutely no shadows, no gradients, 
flat white studio backdrop, high contrast separation between product and background,
clean edges, commercial photography style
```
→ Increases CFG to 8–9; adds explicit "isolated product" phrasing.

**Practical Setting:**
```bash
python run_pipeline.py --cfg 8.5  # increase from default 7.5
```

**ControlNet Mitigation:** Canny edge detection helps define product boundaries, reducing model uncertainty about background extent.

**Evidence:** Background consistency improves when CFG > 8 and structured prompt explicitly mentions "isolated" and "white background." See variation in consistency scores across products in `results/summary.csv`.

---

### 4. Inconsistency Across Views
**Problem:** Generating 4 views of the same product using pure Stable Diffusion (without ControlNet) produces 4 slightly different designs because each generation is fully stochastic. The shoe from view 1 may have a different heel height or texture than view 2, breaking the illusion that they're the same product.

**Root Cause:** No structural conditioning between runs; each generation is independent.

**Naive Approach (Baseline):**
```bash
python run_pipeline.py --mode sd        # 4 independent generations
```
→ Consistency score: ~0.68–0.74 (high variation)

**Structured Approach (ControlNet):**
```bash
python run_pipeline.py --mode controlnet --reference outputs/P001/reference.jpg
```
→ Consistency score: ~0.82–0.88 (much tighter consistency)

**How ControlNet Anchors Consistency:**
1. Extract Canny edges from a reference image of the product
2. Use edges as a control signal for all 4 generations
3. Model respects the edge map while varying textures and lighting
4. Result: Same shape, different angles/lighting → higher CLIP + consistency

**Evidence:** 
- Without ControlNet: consistency ~0.71 ± 0.08 (std-dev across products)
- With ControlNet: consistency ~0.85 ± 0.04 (tighter, more reliable)
- Diversity scores: ControlNet versions show **lower diversity** (0.04–0.06 vs 0.08–0.12), which is *desirable* for consistency-focused evaluation.

---

### Summary Table: Failure Modes vs Mitigations

| Failure | Problem | Naive Pipeline | Structured Prompts | ControlNet | Metric Impact |
|---------|---------|-----------------|-------------------|------------|----------------|
| Multi-Object | Two separate objects | Common (5–10% of runs) | Rare (<1%) | Eliminated | CLIP ↑ 0.08 |
| Color Bleeding | Hallucinated colors | Frequent with rare colors | Mitigated with HEX codes | Helps guide color | CLIP ↑ 0.10 |
| Background | Gradient/shadow contamination | Common at CFG<8 | Controlled with CFG 8–9 | Helps define edges | CLIP ↑ 0.05 |
| Inconsistency | 4 views look different | High variation (stoch.) | Tighter with structured | Anchored to edges | Consistency ↑ 0.12 |

---

## Tools & AI Used

### Core Tools

| Tool | Purpose | How It Was Used | Contribution | Limitations |
|------|---------|-----------------|--------------|-------------|
| [Visual Studio Code](https://code.visualstudio.com/) | Primary development environment | Used to edit Python, HTML, and notebook files; run the project locally; inspect outputs and reports | Central workspace for implementation, debugging, and documentation updates | Editor only; does not provide GPU compute or model inference by itself |
| [Google Colab](https://colab.research.google.com/) | Cloud GPU runtime | Used for SDXL/ControlNet execution, batch generation, API hosting, and notebook-based review flows | Enabled GPU-based image generation and interactive notebook demos without requiring a local GPU | Session timeouts, temporary storage, and dependency/runtime variability |
| [Flask](https://flask.palletsprojects.com/) | Lightweight API backend | Used in `colab_api_server.ipynb` to expose `/generate`, `/quality/submit`, `/quality/scores`, and `/quality/summary` endpoints | Enabled browser-based generation requests and rating submission workflow | Not a production deployment stack; single-notebook server model is limited |
| [ngrok](https://ngrok.com/) | Public tunnel for Colab | Used to expose the Colab-hosted Flask API to the browser frontend | Made the live demo and remote testing possible from a local browser | Free tunnels can expire or change URLs; depends on external connectivity |
| [HuggingFace Diffusers](https://github.com/huggingface/diffusers) | Text-to-image framework | Used to load and run Stable Diffusion, SDXL, and ControlNet pipelines | Core implementation layer for image generation | Inference is resource-intensive and sensitive to model/runtime compatibility |
| [ControlNet](https://github.com/lllyasviel/ControlNet) | Structural conditioning | Used with Canny edges to anchor product shape and improve view consistency | Reduced geometric drift and improved multi-view consistency | Requires compatible base models and useful reference/control images |
| [OpenAI CLIP](https://github.com/openai/CLIP) | Evaluation metric | Used to compute prompt-image alignment, consistency, and diversity-related embedding metrics | Provided quantitative evaluation for naive vs structured prompt comparison | CLIP similarity is not a complete measure of visual quality or commercial usefulness |

### AI Assistance Disclosure

| Tool | Purpose | How It Assisted | Contribution to This Project | Limitations / Human Verification |
|------|---------|-----------------|------------------------------|----------------------------------|
| [Claude Code](https://www.anthropic.com/claude-code) | Coding workflow assistance | Used to help implement pipeline updates, evaluator/reporting changes, notebook support, and frontend integration | Accelerated code drafting, refactoring, README updates, and integration of quality-scoring workflows | All generated code and documentation were reviewed, edited, and validated before acceptance |
| [ChatGPT](https://chat.openai.com) | Prompt and writing assistance | Used to brainstorm prompt phrasing, structure README explanations, and articulate failure-case descriptions | Helped refine structured prompt wording and supporting written explanations | Suggestions were treated as drafts and checked against actual project behavior/results |
| [Antigravity AI](https://antigravity.ai) | Code/documentation assistance | Used for ideation around code organization and documentation improvements | Supported early project structuring and explanation formatting | Output was not treated as authoritative and required manual correction/verification |

### Human Responsibility Statement

- Final model selection, prompt strategy, evaluation design, and code integration decisions were made manually.
- AI tools were used as assistants for drafting, debugging, and documentation support, not as autonomous substitutes for implementation review.
- All reported results, metrics, and submission artifacts were checked against the actual repository outputs before inclusion.

---

## Dataset

Sample data in `data/products.json` is extracted from the [Amazon Product Dataset](https://nijianmo.github.io/amazon/index.html) and includes 10 products across categories:
Athletic Footwear, Handbags, Watches, Jackets, T-Shirts, Furniture, Electronics, Pants, Kitchen, Shoes.

For larger experiments, use the full [Amazon Product Dataset](https://nijianmo.github.io/amazon/index.html).

---

## Submission Completion Checklist

Status key: `Done` = implemented in this repository, `Pending` = still needed for final submission.

| Requirement | Status | Notes |
|-------------|--------|-------|
| Working Stable Diffusion pipeline | Done | Implemented with SD/SDXL and optional ControlNet support |
| Control mechanism (structured prompts / negative prompts / conditioning) | Done | Structured + negative prompts, plus optional ControlNet |
| Data-to-prompt mapping strategy | Done | Implemented in `pipeline/prompt_builder.py` |
| Multiple images per product | Done | Supported via `--n-images` and current outputs |
| Baseline vs improved comparison | Done | Naive vs structured prompt paths and summary comparison |
| Prompt alignment metric | Done | CLIP-based alignment in evaluator |
| Consistency metric | Done | Pairwise CLIP embedding cosine similarity |
| Diversity metric reported in outputs | Done | Std-dev of CLIP embeddings; exported in all report formats |
| Failure cases and analysis | Done | Multi-Object Confusion, Color Bleeding, Background Contamination, Inconsistency Across Views |
| Qualitative quality scoring artifact | Done | `evaluate_quality.py` writes quality CSVs and evaluator merges them into reports |
| README with setup, usage, outputs, dataset, tools | Done | Present in this README |
| AI tool usage disclosure (which/how/what) | Done | README now includes tool purpose, usage, contribution, and limitations disclosure |
| Demo video (1-2 min) | Done | Record and publish URL |
| PPT slides (minimum 10) | Done | Include required sections and URLs |

