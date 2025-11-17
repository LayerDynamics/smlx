"""
Utility functions for SMLX evaluation benchmarks.

This module provides shared utilities for running evaluations on vision-language models
and other multimodal models using the MLX framework on Apple Silicon.
"""

from typing import Optional, Union

from PIL import Image


def inference(
    model,
    processor,
    question: str,
    image: Optional[Union[Image.Image, list[Image.Image]]] = None,
    max_tokens: int = 3000,
    temperature: float = 0.0,
    resize_shape: Optional[tuple] = None,
    verbose: bool = False,
) -> str:
    """
    Run inference on a vision-language model with a question and optional image(s).

    This function wraps the MLX-VLM generation pipeline, applying chat templates
    and handling image preprocessing for evaluation benchmarks.

    Args:
        model: The loaded MLX vision-language model
        processor: The model's processor/tokenizer
        question: The text question/prompt to ask the model
        image: Single PIL Image or list of PIL Images (optional for text-only questions)
        max_tokens: Maximum number of tokens to generate (default: 3000)
        temperature: Sampling temperature (0.0 for deterministic, default: 0.0)
        resize_shape: Optional tuple (height, width) to resize images before processing
        verbose: If True, print detailed generation information

    Returns:
        str: The model's text response

    Example:
        >>> from mlx_vlm import load
        >>> from PIL import Image
        >>> model, processor = load("mlx-community/SmolVLM-256M-Instruct")
        >>> image = Image.open("test.jpg")
        >>> response = inference(model, processor, "What is in this image?", image)
    """
    try:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as e:
        raise ImportError(
            "mlx-vlm is required for evaluations. "
            "Install it with: pip install 'smlx[evals]' or pip install mlx-vlm"
        ) from e

    # Fix processor configuration if patch_size is None (common issue with some models)
    if hasattr(processor, "patch_size") and processor.patch_size is None:
        # Default patch size for most vision transformers
        processor.patch_size = 14
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "patch_size"):
        if processor.image_processor.patch_size is None:
            processor.image_processor.patch_size = 14

    # Determine number of images for chat template
    if image is None:
        num_images = 0
    elif isinstance(image, list):
        num_images = len(image)
    else:
        num_images = 1

    # Apply chat template to format the prompt correctly for the model
    prompt = apply_chat_template(processor, model.config, question, num_images=num_images)

    # Generate response using MLX-VLM
    response = generate(
        model,
        processor,
        prompt,  # type: ignore[arg-type]
        image=image,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        resize_shape=resize_shape,
        verbose=verbose,
    )

    return response.text


def load_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """
    Load a vision-language model for evaluation.

    This is a convenience wrapper around mlx_vlm.load that provides
    consistent model loading across SMLX evaluations.

    Args:
        model_path: Path or HuggingFace model ID (e.g., "mlx-community/SmolVLM-256M-Instruct")
        adapter_path: Optional path to LoRA/adapter weights
        trust_remote_code: Whether to trust remote code in model files (default: True)

    Returns:
        tuple: (model, processor) ready for inference

    Example:
        >>> model, processor = load_model("mlx-community/SmolVLM-256M-Instruct")
        >>> # Model is now ready for evaluation
    """
    try:
        from mlx_vlm import load
    except ImportError as e:
        raise ImportError(
            "mlx-vlm is required for evaluations. "
            "Install it with: pip install 'smlx[evals]' or pip install mlx-vlm"
        ) from e

    return load(model_path, adapter_path=adapter_path, trust_remote_code=trust_remote_code)
