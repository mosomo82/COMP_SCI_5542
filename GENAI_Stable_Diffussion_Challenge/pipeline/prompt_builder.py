"""
prompt_builder.py
-----------------
Converts product metadata dicts into text prompts for Stable Diffusion.

Two strategies are implemented and compared:
  - naive_prompt    : baseline — raw product title only
  - structured_prompt : full metadata template + quality keywords
"""

# ---------------------------------------------------------------------------
# Negative prompt applied to every generation run
# ---------------------------------------------------------------------------
NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, watermark, text, logo, "
    "distorted, deformed, ugly, disfigured, dark background, busy background, "
    "cartoon, sketch, drawing, illustration, painting, 3d render, fake, "
    "duplicate, cropped, out of frame"
)


def naive_prompt(product: dict) -> str:
    """
    Baseline strategy: use only the raw product title.
    Represents what a non-expert user would enter.
    """
    return product["title"]


def structured_prompt(product: dict) -> str:
    """
    Improved strategy: build a rich, template-driven prompt from all
    available metadata fields.

    Template structure
    ------------------
    [photography style] of [color] [material] [title],
    category: [category], style: [style keywords],
    [quality boosters]
    """
    photography_style = "Professional studio product photography"
    quality_boosters  = (
        "studio lighting, pure white background, sharp focus, "
        "8K ultra resolution, e-commerce style, centered product, "
        "no shadows, photorealistic, commercial quality"
    )

    prompt = (
        f"{photography_style} of "
        f"{product['color']} {product['material']} {product['title']}, "
        f"category: {product['category']}, "
        f"style: {product['style']}, "
        f"{quality_boosters}"
    )
    return prompt


def build_both(product: dict) -> dict[str, str]:
    """Return both prompt variants for a single product dict."""
    return {
        "naive":      naive_prompt(product),
        "structured": structured_prompt(product),
        "negative":   NEGATIVE_PROMPT,
    }
