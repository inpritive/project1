"""
Image generation using HuggingFace Diffusers.
Uses Stable Diffusion img2img to modify outfit while preserving the face.
"""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

import config

# ---------------------------------------------------------------------------
# Lazy model loading: pipeline is loaded on first use and cached.
# This avoids loading ~4GB of weights at import time.
# ---------------------------------------------------------------------------
_PIPELINE = None


def _get_pipeline():
    """Load Stable Diffusion img2img pipeline once and reuse (optimized loading)."""
    global _PIPELINE
    if _PIPELINE is None:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        _PIPELINE = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.GENERATOR_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,  # Optional: disable for faster inference
        )
        _PIPELINE.to("cuda" if torch.cuda.is_available() else "cpu")
    return _PIPELINE


def _to_pil(image: Union[Path, str, np.ndarray, Image.Image]) -> Image.Image:
    """Convert various input types to PIL Image (RGB, 512x512 for SD)."""
    if isinstance(image, (Path, str)):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        # OpenCV uses BGR; convert to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV BGR → RGB
        pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Stable Diffusion v1 expects 512x512; resize preserving aspect then center-crop if needed
    pil = pil.resize((512, 512), Image.Resampling.LANCZOS)
    return pil


def generate_styled_image(
    original_image: Union[Path, str, np.ndarray, Image.Image],
    clothing_color: str,
    hairstyle: str,
    sneaker_type: str,
    output_dir: Union[Path, str, None] = None,
    output_filename: str = "styled_output.png",
    guidance_scale: float = config.GENERATOR_GUIDANCE_SCALE,
    num_inference_steps: int = config.GENERATOR_STEPS,
    strength: float = config.GENERATOR_STRENGTH,
) -> Path:
    """
    Generate a styled portrait preserving the face while changing outfit.

    Uses img2img: starts from the original image, adds noise, then denoises
    guided by the prompt. Low strength keeps the face similar; the prompt
    steers outfit/hairstyle/sneakers toward the description.

    Diffusion basics:
    - Diffusion models learn to denoise: they reverse a gradual noising process.
    - At each step, the model predicts noise and subtracts it to get a cleaner image.
    - The text prompt is encoded and used to "guide" which direction to denoise
      (via classifier-free guidance). Higher guidance_scale = stronger adherence
      to the prompt but risk of oversaturation.
    - In img2img, we start from a noised version of the input. Strength controls
      how much noise is added: lower (0.3-0.5) = more preservation of the input,
      higher (0.6-0.8) = more creative change.

    Prompt engineering:
    - Order and wording matter: leading terms ("Portrait photo of the same person")
      anchor identity; later terms describe the desired changes.
    - Specific adjectives ("modern", "stylish", "realistic") reduce artifacts.
    - Face preservation works best when we emphasize "same person" and use
      moderate strength so the model doesn't fully reimagine the image.

    Args:
        original_image: Input photo (path, numpy array, or PIL Image).
        clothing_color: Recommended color for outfit (e.g. "olive green").
        hairstyle: Recommended hairstyle (e.g. "layered cuts with volume").
        sneaker_type: Recommended sneaker (e.g. "chunky sneakers").
        output_dir: Directory to save output (default: config.OUTPUT_DIR).
        output_filename: Output file name (default: styled_output.png).
        guidance_scale: How strongly to follow the prompt (7–10 typical).
        num_inference_steps: Denoising steps; more = higher quality, slower.
        strength: img2img strength (0.3–0.55 for face preservation).

    Returns:
        Path to the saved generated image.
    """
    import torch

    pipeline = _get_pipeline()
    init_image = _to_pil(original_image)

    # Prompt structure: anchor identity first, then outfit details.
    # "same person" encourages face preservation; specifics guide outfit.
    prompt = (
        f"Portrait photo of the same person wearing {clothing_color} modern outfit, "
        f"{hairstyle}, stylish {sneaker_type}, realistic, high detail, natural lighting"
    )

    # Negative prompt reduces common artifacts (blurry, cartoon, etc.)
    negative_prompt = "blurry, low quality, cartoon, distorted, deformed"

    output_dir = Path(output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
        )

    out_image = result.images[0]
    out_image.save(output_path)
    return output_path


def apply_style_to_outfit(
    base_image_path: Path,
    style_description: str,
    output_path: Path,
) -> Path:
    """
    Apply a style/outfit description to a base image (thin wrapper).

    Args:
        base_image_path: Path to person/base image.
        style_description: Full text description of outfit to apply.
        output_path: Where to save the result.

    Returns:
        Path to the generated image.
    """
    output_dir = Path(output_path).parent
    return generate_styled_image(
        base_image_path,
        clothing_color=style_description,
        hairstyle="natural",
        sneaker_type="stylish sneakers",
        output_dir=output_dir,
        output_filename=Path(output_path).name,
    )
