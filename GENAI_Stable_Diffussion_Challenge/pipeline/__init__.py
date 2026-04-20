# pipeline/__init__.py
# Exposes the main public API of the pipeline package.

from pipeline.prompt_builder import naive_prompt, structured_prompt, build_both, NEGATIVE_PROMPT
from pipeline.sd_pipeline    import load_sd_pipeline, load_controlnet_pipeline
from pipeline.generator      import generate_for_product, run_all_products, build_canny_control_image
from pipeline.evaluator      import clip_score, consistency_score, evaluate_results
from pipeline.audio_generator  import generate_product_audio

__all__ = [
    "naive_prompt",
    "structured_prompt",
    "build_both",
    "NEGATIVE_PROMPT",
    "load_sd_pipeline",
    "load_controlnet_pipeline",
    "generate_for_product",
    "run_all_products",
    "build_canny_control_image",
    "clip_score",
    "consistency_score",
    "evaluate_results",
    "generate_product_audio",
]
