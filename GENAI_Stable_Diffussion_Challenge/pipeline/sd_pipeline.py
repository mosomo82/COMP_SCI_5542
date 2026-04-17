"""
sd_pipeline.py
--------------
Loads and returns a Stable Diffusion or ControlNet pipeline.

Supported modes
---------------
  "sd"          : vanilla StableDiffusionPipeline (SD 1.5 or SDXL)
  "controlnet"  : StableDiffusionControlNetPipeline with Canny edges

GPU memory guidance
-------------------
  SD 1.5  → ~4 GB VRAM  (works on most consumer cards)
  SDXL    → ~8 GB VRAM  (higher quality, slower)
  ControlNet + SD 1.5  → ~6 GB VRAM
"""

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Default model IDs — override via config / CLI flags
# ---------------------------------------------------------------------------
SD_MODEL_ID         = "runwayml/stable-diffusion-v1-5"
SDXL_MODEL_ID       = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-canny"


def _choose_device() -> str:
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(
            f"[green]✓ GPU detected:[/green] {torch.cuda.get_device_name(0)} "
            f"({vram_gb:.1f} GB VRAM)"
        )
        return "cuda"
    console.print("[yellow]⚠ No GPU found — running on CPU (very slow).[/yellow]")
    return "cpu"


def _apply_memory_optimisations(pipe, device: str):
    """Enable attention slicing and xformers if available to reduce VRAM usage."""
    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            console.print("[dim]  xformers memory-efficient attention enabled[/dim]")
        except Exception:
            pass  # xformers not installed — continue without it
    return pipe


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_sd_pipeline(
    model_id: str = SD_MODEL_ID,
    use_sdxl: bool = False,
) -> tuple[StableDiffusionPipeline, str]:
    """
    Load a vanilla Stable Diffusion pipeline.

    Returns
    -------
    (pipeline, device_str)
    """
    if use_sdxl:
        model_id = SDXL_MODEL_ID

    device = _choose_device()
    dtype  = torch.float16 if device == "cuda" else torch.float32

    console.print(f"[cyan]Loading SD pipeline:[/cyan] {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,       # disable NSFW filter for product images
        requires_safety_checker=False,
    )

    # Use fast DPM++ scheduler (better quality at 30 steps vs default PNDM)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    pipe = pipe.to(device)
    pipe = _apply_memory_optimisations(pipe, device)

    console.print("[green]✓ SD pipeline ready[/green]")
    return pipe, device


def load_controlnet_pipeline(
    sd_model_id: str = SD_MODEL_ID,
    controlnet_model_id: str = CONTROLNET_MODEL_ID,
) -> tuple[StableDiffusionControlNetPipeline, str]:
    """
    Load a ControlNet-augmented pipeline using Canny edge conditioning.

    Returns
    -------
    (pipeline, device_str)
    """
    device = _choose_device()
    dtype  = torch.float16 if device == "cuda" else torch.float32

    console.print(f"[cyan]Loading ControlNet:[/cyan] {controlnet_model_id}")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id, torch_dtype=dtype
    )

    console.print(f"[cyan]Loading SD backbone:[/cyan] {sd_model_id}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    pipe = pipe.to(device)
    pipe = _apply_memory_optimisations(pipe, device)

    console.print("[green]✓ ControlNet pipeline ready[/green]")
    return pipe, device
