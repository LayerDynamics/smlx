"""
Utility functions for SMLX evaluation benchmarks.

This module provides shared utilities for running evaluations on vision-language models
and other multimodal models using the MLX framework on Apple Silicon.
"""

from typing import Any, Optional, Union

from PIL import Image


def resolve_eval_dataset(
    local_name: str,
    hf_dataset: str,
    split: str,
    *,
    prefer_local: bool = True,
    streaming: bool = False,
    verbose: bool = True,
) -> tuple[Any, str]:
    """Resolve an evaluation dataset, preferring the bundled local copy.

    Loads from the in-repo ``data/`` tree (via :mod:`smlx.data.local`) when all
    of the following hold, otherwise falls back to ``datasets.load_dataset``:

    - ``prefer_local`` is True (callers set this to False when the user
      overrides ``--dataset`` away from its default, so an explicit choice is
      always honoured),
    - ``streaming`` was not requested (the local copies are materialised on
      disk, not streamed),
    - a local copy named ``local_name`` is present and exposes ``split``, and
    - that local copy is a HuggingFace ``Dataset`` (i.e. drop-in compatible
      with the evaluators' ``.select`` / row-iteration). Local copies stored in
      a non-HF layout -- e.g. OCRBench's JSON index -- are skipped and the HF
      source is used instead.

    Args:
        local_name: Dataset key in the ``smlx.data.local`` registry
            (e.g. ``"mathvista"``).
        hf_dataset: HuggingFace repo id to fall back to (usually ``args.dataset``).
        split: Split name to load.
        prefer_local: Whether to attempt the local copy at all.
        streaming: Whether the caller requested a streaming dataset.
        verbose: Print which source was selected.

    Returns:
        ``(dataset, source_description)`` where ``source_description`` records
        whether the local or HuggingFace source was used.
    """
    from datasets import load_dataset

    if prefer_local and not streaming:
        try:
            from smlx.data import local as local_data

            if local_data.is_available(local_name) and split in local_data.available_splits(
                local_name
            ):
                dataset = local_data.load(local_name, split=split)
                # Only a HuggingFace Dataset is drop-in compatible with the
                # evaluators (needs column_names / select / row iteration).
                if hasattr(dataset, "column_names"):
                    rel = local_data.local_path(local_name).relative_to(local_data.data_dir())
                    source = f"local data/{rel} (split={split})"
                    if verbose:
                        print(f"  Dataset source: {source}")
                    return dataset, source
        except Exception as exc:
            if verbose:
                print(f"  (local dataset unavailable: {exc}; using HuggingFace source)")

    dataset = load_dataset(hf_dataset, split=split, streaming=streaming)
    source = f"HuggingFace {hf_dataset} (split={split})"
    if verbose:
        print(f"  Dataset source: {source}")
    return dataset, source


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
