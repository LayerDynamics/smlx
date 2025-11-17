#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model loader for nanoVLM.

Handles downloading and loading models from HuggingFace Hub.
"""

from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx

from .config import DEFAULT_CONFIG, NanoVLMConfig, load_config
from .image_processor import ImageProcessor, create_image_processor
from .model import NanoVLM


class Processor:
    """
    Processor for nanoVLM that combines image processing and tokenization.

    Attributes:
        image_processor: ImageProcessor instance
        tokenizer: HuggingFace tokenizer
    """

    def __init__(self, image_processor: ImageProcessor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, text: str = None, image=None, **kwargs):
        """
        Process text and/or image.

        Args:
            text: Input text (optional)
            image: Input image (optional)
            **kwargs: Additional arguments

        Returns:
            Dictionary with processed inputs
        """
        outputs = {}

        if text is not None:
            # Tokenize text
            outputs["input_ids"] = self.tokenizer.encode(text, return_tensors="np")

        if image is not None:
            # Process image
            outputs["pixel_values"] = self.image_processor(image)

        return outputs


def get_model_path(model_name: str, force_download: bool = False) -> Path:
    """
    Download model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model ID (e.g., "lusxvr/nanoVLM-222M")
        force_download: Force re-download even if cached

    Returns:
        Path to downloaded model directory

    Example:
        >>> path = get_model_path("lusxvr/nanoVLM-222M")
        >>> print(path)
        PosixPath('/Users/.../.cache/huggingface/hub/models--lusxvr--nanoVLM-222M/...')
    """
    from huggingface_hub import snapshot_download

    try:
        # Download model
        path = snapshot_download(
            repo_id=model_name,
            force_download=force_download,
        )
        return Path(path)
    except Exception as e:
        print(f"Could not download model from {model_name}: {e}")
        return None


def load_weights(model_path: Path) -> dict:
    """
    Load model weights from directory.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary of model weights

    Note:
        Supports .safetensors and .npz formats.
    """
    # Try safetensors first
    safetensors_path = model_path / "model.safetensors"
    if safetensors_path.exists():
        weights = mx.load(str(safetensors_path))
        return weights

    # Try npz format
    npz_path = model_path / "model.npz"
    if npz_path.exists():
        weights = mx.load(str(npz_path))
        return weights

    # Try individual safetensors shards
    safetensors_index = model_path / "model.safetensors.index.json"
    if safetensors_index.exists():
        import json

        with open(safetensors_index) as f:
            index = json.load(f)

        weights = {}
        for shard_file in set(index["weight_map"].values()):
            shard_path = model_path / shard_file
            shard_weights = mx.load(str(shard_path))
            weights.update(shard_weights)

        return weights

    print(f"Warning: No weights found in {model_path}")
    return None


def load(
    model_name: str = "lusxvr/nanoVLM-222M",
    force_download: bool = False,
) -> Tuple[NanoVLM, Processor]:
    """
    Load nanoVLM model and processor.

    Args:
        model_name: HuggingFace model ID or local path
        force_download: Force re-download from Hub

    Returns:
        Tuple of (model, processor)

    Example:
        >>> model, processor = load("lusxvr/nanoVLM-222M")
        >>> print(f"Model loaded with {sum(p.size for p in model.parameters())/1e6:.1f}M parameters")
    """
    # Download/get model path
    if Path(model_name).exists():
        model_path = Path(model_name)
    else:
        print(f"Downloading {model_name} from HuggingFace Hub...")
        model_path = get_model_path(model_name, force_download)

        if model_path is None:
            raise ValueError(f"Could not load model from {model_name}")

    # Load config
    config = load_config(str(model_path))

    # Create model
    print("Creating model...")
    model = NanoVLM(config)

    # Load weights
    print("Loading weights...")
    weights = load_weights(model_path)

    if weights is not None:
        # Sanitize weights to match model parameter names
        sanitized_weights = NanoVLM.sanitize(weights)

        # Update model with sanitized weights (strict=False allows missing/extra parameters)
        # nanoVLM vision model doesn't use position embeddings in pretrained weights
        model.update(sanitized_weights, strict=False)
        print("✓ Weights loaded (some parameters randomly initialized)")
    else:
        print("Warning: Model initialized with random weights")

    # Set to eval mode
    model.eval()

    # Create processor
    image_processor = create_image_processor(config.vision_config.image_size)

    # Load tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Using default tokenizer...")
        # Fallback to SmolLM2 tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    processor = Processor(image_processor, tokenizer)

    print(f"✓ Model loaded successfully")

    return model, processor


def save_model(
    model: NanoVLM,
    output_path: str,
    config: Optional[NanoVLMConfig] = None,
):
    """
    Save model weights and configuration.

    Args:
        model: NanoVLM model instance
        output_path: Directory to save model
        config: Optional config (uses model.config if not provided)

    Example:
        >>> model, processor = load("lusxvr/nanoVLM-222M")
        >>> save_model(model, "./my_nanovlm")
    """
    from .config import save_config

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    if config is None:
        config = model.config
    save_config(config, str(output_path))

    # Save weights
    weights = dict(model.parameters())
    weights_path = output_path / "model.safetensors"
    mx.save_safetensors(str(weights_path), weights)

    print(f"✓ Model saved to {output_path}")
