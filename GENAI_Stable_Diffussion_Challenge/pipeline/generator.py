"""
generator.py
------------
Generates product images using a loaded pipeline.

Features
--------
- Generates N images per product × 2 prompt strategies (naive + structured)
- Optional ControlNet mode with Canny edge maps from a reference image
- Saves outputs as PNG files with structured naming
- Returns metadata dict for downstream evaluation
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm
from rich.console import Console

from pipeline.prompt_builder import NEGATIVE_PROMPT, naive_prompt, structured_prompt

console = Console()

# ---------------------------------------------------------------------------
# Default generation parameters (tuned for e-commerce product images)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "num_inference_steps": 30,    # 30 is a good speed/quality balance
    "guidance_scale":       7.5,  # CFG — stay in prompt alignment sweet spot
    "height":               512,
    "width":                512,
}


def _make_generator(seed: Optional[int], device: str):
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def generate_for_product(
    product:     dict,
    pipe,                        # StableDiffusionPipeline or ControlNet variant
    device:      str,
    output_dir:  Path,
    n_images:    int  = 4,
    seed:        Optional[int] = 42,
    gen_params:  dict = None,
    control_image: Optional[Image.Image] = None,  # for ControlNet mode
) -> list[dict]:
    """
    Generate images for a single product using both naive and structured prompts.

    Returns
    -------
    List of result dicts, one per generated image:
        {
          "product_id", "product_title",
          "prompt_type",  # "naive" | "structured"
          "prompt",
          "image_path",
          "seed",
          "gen_time_s",
        }
    """
    params = {**DEFAULT_PARAMS, **(gen_params or {})}
    results = []

    prompts = {
        "naive":      naive_prompt(product),
        "structured": structured_prompt(product),
    }

    pid = product["id"]
    product_dir = output_dir / pid
    product_dir.mkdir(parents=True, exist_ok=True)

    for prompt_type, prompt_text in prompts.items():
        console.print(
            f"\n  [bold]{prompt_type.upper()}[/bold] prompt for "
            f"[cyan]{product['title']}[/cyan]"
        )
        console.print(f"  [dim]{prompt_text[:120]}{'…' if len(prompt_text)>120 else ''}[/dim]")

        for i in range(n_images):
            # Use different seeds per view for diversity
            view_seed = (seed + i) if seed is not None else None
            generator = _make_generator(view_seed, device)

            t0 = time.time()

            if control_image is not None:
                # ControlNet mode
                images = pipe(
                    prompt=prompt_text,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=control_image,
                    generator=generator,
                    **params,
                ).images
            else:
                # Plain SD mode
                images = pipe(
                    prompt=prompt_text,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_images_per_prompt=1,
                    generator=generator,
                    **params,
                ).images

            elapsed = time.time() - t0
            img = images[0]

            # Save image
            filename  = f"{pid}_{prompt_type}_view{i+1:02d}_seed{view_seed}.png"
            save_path = product_dir / filename
            img.save(save_path)

            console.print(f"    [green]✓[/green] View {i+1}/{n_images} saved → {filename}  ({elapsed:.1f}s)")

            results.append({
                "product_id":    pid,
                "product_title": product["title"],
                "prompt_type":   prompt_type,
                "prompt":        prompt_text,
                "image_path":    str(save_path),
                "seed":          view_seed,
                "gen_time_s":    round(elapsed, 2),
            })

    return results


def build_canny_control_image(
    reference_path: str,
    low_threshold:  int = 100,
    high_threshold: int = 200,
    target_size:    tuple[int, int] = (512, 512),
) -> Image.Image:
    """
    Create a Canny edge map from a reference image for ControlNet conditioning.

    Parameters
    ----------
    reference_path  : path to reference product image
    low_threshold   : lower Canny threshold
    high_threshold  : upper Canny threshold
    target_size     : resize before edge detection (must match pipeline resolution)
    """
    from controlnet_aux import CannyDetector

    ref = Image.open(reference_path).convert("RGB").resize(target_size)
    canny = CannyDetector()
    edge_map = canny(ref, low_threshold=low_threshold, high_threshold=high_threshold)
    console.print(f"[green]✓ Canny edge map created from:[/green] {reference_path}")
    return edge_map


def run_all_products(
    products:     list[dict],
    pipe,
    device:       str,
    output_dir:   Path,
    n_images:     int  = 4,
    seed:         Optional[int] = 42,
    gen_params:   dict = None,
    control_image: Optional[Image.Image] = None,
) -> list[dict]:
    """
    Iterate over all products and generate images for each.

    Returns a flat list of all result dicts (one per generated image).
    """
    all_results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, product in enumerate(products, 1):
        console.rule(
            f"[bold yellow]Product {idx}/{len(products)} — {product['id']}[/bold yellow]"
        )
        results = generate_for_product(
            product=product,
            pipe=pipe,
            device=device,
            output_dir=output_dir,
            n_images=n_images,
            seed=seed,
            gen_params=gen_params,
            control_image=control_image,
        )
        all_results.extend(results)

    console.rule("[bold green]Generation complete[/bold green]")
    console.print(
        f"[green]Total images generated:[/green] {len(all_results)} "
        f"({len(products)} products × 2 prompts × {n_images} views)"
    )
    return all_results
