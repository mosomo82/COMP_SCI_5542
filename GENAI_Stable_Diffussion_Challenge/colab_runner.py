"""
colab_runner.py
---------------
Run this script in Google Colab to execute the full SD pipeline on GPU.

How to use
----------
In a Colab cell, run:

    !git clone https://github.com/YOUR_USERNAME/COMP_SCI_5542.git
    %cd COMP_SCI_5542/GENAI_Stable_Diffussion_Challenge
    !python colab_runner.py

Or paste this entire file into a Colab cell and run it directly.
"""

import subprocess
import sys
import os

# ── 1. Check GPU ──────────────────────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU detected: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
else:
    print("⚠️  No GPU found. Go to Runtime → Change runtime type → GPU")
    sys.exit(1)

# ── 2. Install dependencies ───────────────────────────────────────────────────
print("\n📦 Installing dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "diffusers>=0.27.2",
    "transformers>=4.39.3",
    "accelerate>=0.28.0",
    "controlnet-aux>=0.0.7",
    "xformers",           # memory-efficient attention (big speed boost on Colab)
    "rich>=13.7.1",
])
print("✅ Dependencies installed.")

# ── 3. (Optional) Mount Google Drive to persist outputs ───────────────────────
SAVE_TO_DRIVE = False   # ← Set to True to save outputs to Google Drive

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_DIR  = "/content/drive/MyDrive/SD_Outputs/outputs"
    RESULTS_DIR = "/content/drive/MyDrive/SD_Outputs/results"
    print(f"💾 Saving to Google Drive: {OUTPUT_DIR}")
else:
    OUTPUT_DIR  = "outputs"   # saved locally in Colab session
    RESULTS_DIR = "results"
    print("📁 Saving to Colab session (outputs/ and results/)")

# ── 4. Run the pipeline ───────────────────────────────────────────────────────
#
#  Adjust these settings as needed:
#    --limit    : number of products to process (use 3 for a quick test)
#    --n-images : views per product per prompt type
#    --steps    : denoising steps (20 = fast,  50 = high quality)
#    --cfg      : guidance scale (7.5 is the sweet spot)
#
#  On Colab Pro A100 (~40 GB):  use --sdxl for best quality
#  On Colab Pro T4 (~16 GB):    use default SD 1.5
#  On Colab Pro L4 (~24 GB):    SD 1.5 or SDXL both work

cmd = [
    sys.executable, "run_pipeline.py",
    "--mode",       "sd",
    "--limit",      "3",          # ← change to 10 for the full run
    "--n-images",   "4",
    "--steps",      "30",
    "--cfg",        "7.5",
    "--height",     "512",
    "--width",      "512",
    "--seed",       "42",
    "--output-dir", OUTPUT_DIR,
    "--results-dir", RESULTS_DIR,
]

# Uncomment to use SDXL (better quality, needs A100 / L4):
# cmd += ["--sdxl"]

print(f"\n🚀 Running: {' '.join(cmd)}\n")
subprocess.check_call(cmd)

# ── 5. Show output images in notebook ─────────────────────────────────────────
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

print("\n🖼  Displaying sample outputs...")

def show_comparison(product_dir: Path, n: int = 4):
    """Show naive vs structured comparison for one product."""
    naive_imgs      = sorted(product_dir.glob("*_naive_*.png"))[:n]
    structured_imgs = sorted(product_dir.glob("*_structured_*.png"))[:n]

    if not naive_imgs and not structured_imgs:
        return

    total = len(naive_imgs) + len(structured_imgs)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))
    fig.suptitle(product_dir.name, fontsize=14, fontweight="bold")

    for col, img_path in enumerate(naive_imgs):
        axes[0][col].imshow(Image.open(img_path))
        axes[0][col].set_title(f"Naive · View {col+1}", fontsize=10, color="orange")
        axes[0][col].axis("off")

    for col, img_path in enumerate(structured_imgs):
        axes[1][col].imshow(Image.open(img_path))
        axes[1][col].set_title(f"Structured · View {col+1}", fontsize=10, color="green")
        axes[1][col].axis("off")

    plt.tight_layout()
    plt.show()

output_path = Path(OUTPUT_DIR)
for product_dir in sorted(output_path.iterdir()):
    if product_dir.is_dir():
        show_comparison(product_dir)

# ── 6. Show results CSV ───────────────────────────────────────────────────────
import pandas as pd

report_path = Path(RESULTS_DIR) / "evaluation_report.csv"
summary_path = Path(RESULTS_DIR) / "summary.csv"

if summary_path.exists():
    print("\n📊 Evaluation Summary:")
    df = pd.read_csv(summary_path)
    print(df.to_string(index=False))

print("\n✅ Done! Check the outputs/ and results/ folders.")
