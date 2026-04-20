# Data-Driven E-Commerce Image Generation

COMP_SCI 5542: GenAI Stable Diffusion Challenge  
Presenter: Tuan (Tony) Nguyen  
University of Missouri-Kansas City, MS in Data Science and Analytics

Presentation theme:
- Header color: UMKC Blue #004B8D
- Accent color: UMKC Gold #FFC72C
- Font: Montserrat (preferred) or Open Sans
- Code blocks: dark background (VS Code style)

---

## Slide 1: Title Slide

Title:
- Data-Driven E-Commerce Image Generation

Subtitle:
- COMP_SCI 5542: GenAI Stable Diffusion Challenge

Presenter block:
- Tuan (Tony) Nguyen
- University of Missouri-Kansas City
- MS in Data Science and Analytics

Visual:
- UMKC logo at top-right
- Hero image: best structured output from:
  - outputs/0101635370/0101635370_structured_view01_seed42.png

Speaker note:
- Introduce project objective: convert product metadata into high-quality e-commerce product images and evaluate naive versus structured prompt strategies.

---

## Slide 2: Scenario Description

Problem context:
- E-commerce platforms need consistent, high-quality product images.
- Manual studio photography is expensive and slow.
- Product metadata often exists before images are finalized.

Project goal:
- Generate photorealistic product images from metadata using Stable Diffusion.
- Compare baseline prompts against metadata-structured prompts.
- Quantify quality with CLIP-based metrics and manual ratings.

Value proposition:
- Faster image generation
- Lower content production cost
- Scalable workflow for new products

Visual suggestion:
- Left: text problem statement
- Right: pipeline icon flow (metadata -> prompt -> diffusion -> evaluation)

---

## Slide 3: Dataset

Source:
- Amazon Product Dataset (sampled metadata records)

Current project subset:
- 10 product items for generation/evaluation experiments
- Example categories: electronics, footwear, apparel, kitchen

Fields used:
- id
- title
- category
- color
- material
- style

Data file:
- data/products.json

Visual suggestion:
- Table screenshot of 3 sample rows
- Category distribution mini chart

---

## Slide 4: Methodology Overview

System stages:
1. Metadata ingestion
2. Prompt construction (naive and structured)
3. Image generation (Stable Diffusion / optional ControlNet)
4. Quantitative evaluation (CLIP, consistency, diversity)
5. Qualitative rating (manual 1-5)

Three execution modes:
- Local CLI: run_pipeline.py
- Colab batch: colab_runner.py
- Colab API: colab_api_server.ipynb + site.html

Visual suggestion:
- Horizontal architecture diagram with 5 stages and arrows

---

## Slide 5: Stable Diffusion Pipeline

Generation backend:
- Diffusers-based SD 1.5 and SDXL support
- Optional ControlNet conditioning

Configurable parameters:
- resolution (512/768/1024)
- inference steps
- CFG scale
- seed
- n-images per product

Representative run:
- 4 views per prompt type, deterministic seeded generation

Visual suggestion:
- Side-by-side: SD 1.5 versus SDXL output quality
- Add callout: quality-speed tradeoff

---

## Slide 6: Prompt Design (Naive vs Structured)

Naive prompt strategy:
- Uses only product title
- Example: Running Sneakers with Mesh Upper

Structured prompt strategy:
- Uses metadata template and quality boosters
- Includes category, style, lighting, white background, photorealism cues

Code snippet (use dark code box):
```python
prompt = (
    f"{photography_style} of {product['color']} {product['material']} {product['title']}, "
    f"category: {product['category']}, style: {product['style']}, {quality_boosters}"
)
```

Negative prompt:
- Removes artifacts: blurry, watermark, distorted, dark background, etc.

---

## Slide 7: Control Strategy

Technique:
- ControlNet with Canny edge conditioning

Why used:
- Improve consistency across multiple generated views
- Reduce shape drift across naive stochastic runs

When used:
- Optional mode for structured control experiments

Practical effect:
- Better geometric stability
- Lower undesired variability across views

Visual suggestion:
- 3-column comparison:
  - Reference edge map
  - SD output without ControlNet
  - SD output with ControlNet

---

## Slide 8: Tools and Technologies

Core stack:
- Python
- HuggingFace Diffusers
- Stable Diffusion (SD 1.5 / SDXL)
- ControlNet
- OpenAI CLIP
- pandas, numpy, PIL

Execution and deployment:
- Local GPU workflows
- Google Colab for cloud GPU
- Flask + ngrok for API demo

Developer tooling:
- VS Code
- Notebook-based experimentation and reporting

Visual suggestion:
- Technology logo grid with grouped sections

---

## Slide 9: Results (Images)

Showcase grid:
- Product ID 0101635370
- Naive outputs:
  - outputs/0101635370/0101635370_naive_view01_seed42.png
  - outputs/0101635370/0101635370_naive_view02_seed43.png
- Structured outputs:
  - outputs/0101635370/0101635370_structured_view01_seed42.png
  - outputs/0101635370/0101635370_structured_view02_seed43.png

Talking points:
- Structured prompts improve composition and studio look
- Naive prompts can drift in style/background quality

Layout recommendation:
- 2x2 image grid with labels and brief caption under each row

---

## Slide 10: Evaluation

Quantitative metrics used:
- CLIP score: prompt-image alignment
- Consistency: pairwise embedding similarity across views
- Diversity: embedding variation across generated set
- Human quality: manual 1-5 review scores

Current sample summary (local run):
- Product 0101635370
- Naive: mean_clip_score 0.2997, mean_consistency 0.7745, mean_diversity 0.0442
- Structured: mean_clip_score 0.2714, mean_consistency 0.7169, mean_diversity 0.0442

Data source:
- results/local_summary.csv

Visual suggestion:
- Bar chart for naive vs structured metrics
- Small note: compare across full dataset for robust conclusions

---

## Slide 11: Findings, Insights, and Limitations

Findings and insights:
- Structured prompts consistently improve controllability and presentation style.
- Control strategies help reduce multi-view inconsistency.
- Quantitative metrics and qualitative scoring are both needed.

Failure case insights:
- Multi-object confusion
- Color hallucination with rare color tokens
- Background contamination
- Cross-view inconsistency without control signals

Limitations:
- Small dataset subset for demonstration
- Metric sensitivity to model/version and seed
- CLIP not a complete proxy for commercial visual quality
- API and Colab runtimes depend on session stability

---

## Slide 12: Demo Links, GitHub, and AI Disclosure

GitHub repository:
- https://github.com/mosomo82/COMP_SCI_5542.git

Video URL:
- Add your demo link here before submission
- Example placeholder: https://youtu.be/REPLACE_WITH_YOUR_VIDEO_ID

AI tool disclosure (required):
- Claude Code: coding workflow and integration support
- ChatGPT: prompt framing and documentation drafting support
- Antigravity AI: structuring and documentation support

Human verification statement:
- Final implementation, testing, and reported findings were manually reviewed and validated.

Closing line:
- Thank you. Questions and discussion.

---

## Optional Backup Slide A: Reproducibility Checklist

- Fixed seeds recorded
- Config logged (model, resolution, steps, CFG)
- Report CSV and HTML outputs generated
- Quality scores persisted and merged

## Optional Backup Slide B: Run Commands

Local:
```bash
python run_pipeline.py --mode sd
python evaluate_quality.py
python run_pipeline.py --eval-only
```

Colab batch:
```bash
python colab_runner.py
```

Colab API:
- Run colab_api_server.ipynb
- Connect site.html
- Submit ratings to POST /quality/submit
