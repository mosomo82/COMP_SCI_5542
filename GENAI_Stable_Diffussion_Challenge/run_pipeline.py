"""
run_pipeline.py
---------------
Main entry point for the CS 5542 Stable Diffusion E-Commerce pipeline.

Usage examples
--------------
# Generate images for all products (plain SD, 4 views each):
    python run_pipeline.py --mode sd

# Use SDXL instead of SD 1.5 (better quality, needs more VRAM):
    python run_pipeline.py --mode sd --sdxl

# Use ControlNet with a reference image for structural consistency:
    python run_pipeline.py --mode controlnet --reference data/reference.jpg

# Evaluate only (skip generation, use existing outputs):
    python run_pipeline.py --eval-only

# Limit to first 3 products for quick testing:
    python run_pipeline.py --mode sd --limit 3 --n-images 2

# Custom output directory and seed:
    python run_pipeline.py --mode sd --output-dir outputs/run2 --seed 123
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel   import Panel

console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR     = Path(__file__).parent
DATA_DIR     = ROOT_DIR / "data"
PRODUCTS_FILE = DATA_DIR / "products.json"
OUTPUT_DIR   = ROOT_DIR / "outputs"       # images saved here
RESULTS_DIR  = ROOT_DIR / "results"       # CSV reports saved here


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CS 5542 — Stable Diffusion E-Commerce Image Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pipeline mode
    parser.add_argument(
        "--mode", choices=["sd", "controlnet"], default="sd",
        help="Pipeline mode: 'sd' for vanilla SD, 'controlnet' for ControlNet conditioning",
    )
    parser.add_argument("--sdxl",      action="store_true", help="Use SDXL instead of SD 1.5")

    # ControlNet specific
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Path to reference image for ControlNet edge conditioning",
    )
    parser.add_argument("--canny-low",  type=int, default=100, help="Canny low threshold")
    parser.add_argument("--canny-high", type=int, default=200, help="Canny high threshold")

    # Generation parameters
    parser.add_argument("--n-images",   type=int, default=4,   help="Images per product per prompt type")
    parser.add_argument("--steps",      type=int, default=30,  help="Diffusion inference steps")
    parser.add_argument("--cfg",        type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--height",     type=int, default=512, help="Image height in pixels")
    parser.add_argument("--width",      type=int, default=512, help="Image width in pixels")
    parser.add_argument("--seed",       type=int, default=42,  help="Base random seed (None = random)")

    # Data
    parser.add_argument("--products",   type=str, default=str(PRODUCTS_FILE), help="Path to products.json")
    parser.add_argument("--limit",      type=int, default=None, help="Only process first N products (for testing)")

    # Output
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Where to save generated images")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR), help="Where to save CSV reports")

    # Flags
    parser.add_argument("--eval-only",  action="store_true", help="Skip generation, only run evaluation on existing outputs")
    parser.add_argument("--skip-eval",  action="store_true", help="Skip evaluation step after generation")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_products(path: str, limit: int | None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        products = json.load(f)
    if limit:
        products = products[:limit]
    console.print(f"[cyan]Loaded {len(products)} products from:[/cyan] {path}")
    return products


def collect_existing_results(output_dir: Path, products: list[dict]) -> list[dict]:
    """
    Build generation_results list from already-saved PNG files.
    Used when --eval-only is passed.
    """
    from pipeline.prompt_builder import naive_prompt, structured_prompt
    results = []
    for product in products:
        pid     = product["id"]
        pdir    = output_dir / pid
        if not pdir.exists():
            continue
        for prompt_type, prompt_fn in [("naive", naive_prompt), ("structured", structured_prompt)]:
            ptext = prompt_fn(product)
            for png in sorted(pdir.glob(f"{pid}_{prompt_type}_*.png")):
                results.append({
                    "product_id":    pid,
                    "product_title": product["title"],
                    "prompt_type":   prompt_type,
                    "prompt":        ptext,
                    "image_path":    str(png),
                    "seed":          None,
                    "gen_time_s":    0.0,
                })
    console.print(f"[cyan]Found {len(results)} existing images for evaluation.[/cyan]")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir  = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    console.print(Panel.fit(
        "[bold cyan]CS 5542 — Stable Diffusion E-Commerce Pipeline[/bold cyan]\n"
        f"Mode: [yellow]{args.mode}[/yellow]  |  "
        f"SDXL: [yellow]{args.sdxl}[/yellow]  |  "
        f"Images/product/prompt: [yellow]{args.n_images}[/yellow]  |  "
        f"Steps: [yellow]{args.steps}[/yellow]  |  "
        f"CFG: [yellow]{args.cfg}[/yellow]",
        title="🖼  GenAI Pipeline",
    ))

    # -----------------------------------------------------------------------
    # Load products
    # -----------------------------------------------------------------------
    products = load_products(args.products, args.limit)

    generation_results = []

    # -----------------------------------------------------------------------
    # GENERATION PHASE
    # -----------------------------------------------------------------------
    if not args.eval_only:
        from pipeline.sd_pipeline  import load_sd_pipeline, load_controlnet_pipeline
        from pipeline.generator    import run_all_products, build_canny_control_image

        gen_params = {
            "num_inference_steps": args.steps,
            "guidance_scale":      args.cfg,
            "height":              args.height,
            "width":               args.width,
            "use_controlnet":      args.mode == "controlnet",
        }

        control_image = None

        if args.mode == "controlnet":
            pipe, device = load_controlnet_pipeline()
            if args.reference is not None:
                control_image = build_canny_control_image(
                    args.reference,
                    low_threshold=args.canny_low,
                    high_threshold=args.canny_high,
                    target_size=(args.width, args.height),
                )
            else:
                console.print("[cyan]No --reference provided. Using per-product 'image_url' for Canny edge map.[/cyan]")
        else:
            pipe, device = load_sd_pipeline(use_sdxl=args.sdxl)

        generation_results = run_all_products(
            products=products,
            pipe=pipe,
            device=device,
            output_dir=output_dir,
            n_images=args.n_images,
            seed=args.seed,
            gen_params=gen_params,
            control_image=control_image,
        )

        # Free VRAM before running CLIP evaluation
        del pipe
        if device == "cuda":
            import torch; torch.cuda.empty_cache()

    else:
        # Collect existing PNG files for eval-only mode
        generation_results = collect_existing_results(output_dir, products)

    if not generation_results:
        console.print("[red]No images found to evaluate. Run generation first.[/red]")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # EVALUATION PHASE
    # -----------------------------------------------------------------------
    if not args.skip_eval:
        from pipeline.evaluator import evaluate_results
        evaluate_results(generation_results, results_dir)

    console.print("\n[bold green]✅ Pipeline finished![/bold green]")
    console.print(f"  Images  → [cyan]{output_dir}[/cyan]")
    console.print(f"  Reports → [cyan]{results_dir}[/cyan]")


if __name__ == "__main__":
    main()
