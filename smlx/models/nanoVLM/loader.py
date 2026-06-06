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

        # Special handling for weight shapes
        for weight_name in list(sanitized_weights.keys()):
            weight_value = sanitized_weights[weight_name]

            # 1. Position embedding: (1, num_positions, embed_dim) -> (num_positions, embed_dim)
            if 'position_embedding.weight' in weight_name:
                if len(weight_value.shape) == 3 and weight_value.shape[0] == 1:
                    sanitized_weights[weight_name] = weight_value.squeeze(0)  # Remove batch dimension

            # 2. Patch embedding Conv2d: PyTorch (out, in, h, w) -> MLX (out, h, w, in)
            if 'patch_embedding.weight' in weight_name and len(weight_value.shape) == 4:
                # Transpose from (out_channels, in_channels, height, width) to (out_channels, height, width, in_channels)
                sanitized_weights[weight_name] = mx.transpose(weight_value, (0, 2, 3, 1))

        # Load weights using MLX's built-in method (same as mlx-vlm, TinyLLaVA, Moondream2)
        # This is more robust than manual setattr approach
        # Use strict=False to allow missing/extra weights (sanitize may not catch everything)
        unmatched = model.load_weights(list(sanitized_weights.items()), strict=False)
        if unmatched:
            print(f"Warning: {len(unmatched)} unmatched weights (expected for nanoVLM)")
        print("✓ Weights loaded successfully")
    else:
        print("Warning: Model initialized with random weights")

    # Set to eval mode
    model.eval()

    # Create processor
    image_processor = create_image_processor(config.vision_config.image_size)

    # Load tokenizer
    # First, try to use the tokenizer specified in config
    tokenizer = None
    tokenizer_path = None

    # Check if config specifies a tokenizer
    if hasattr(config, 'lm_tokenizer') and config.lm_tokenizer:
        tokenizer_path = config.lm_tokenizer
        print(f"Using tokenizer from config: {tokenizer_path}")

    # Try loading tokenizer
    from transformers import AutoTokenizer

    if tokenizer_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"✓ Loaded tokenizer: {tokenizer_path}")
        except Exception as e:
            print(f"Warning: Could not load specified tokenizer {tokenizer_path}: {e}")

    # If no tokenizer loaded yet, try from model_path
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            print(f"✓ Loaded tokenizer from model path")
        except Exception as e:
            print(f"Warning: Could not load tokenizer from model path: {e}")

    # Final fallback - raise error instead of using wrong tokenizer
    if tokenizer is None:
        raise ValueError(
            "Could not load tokenizer for nanoVLM!\n"
            "Please ensure the model config specifies 'lm_tokenizer' field,\n"
            "or that the model directory contains a valid tokenizer.\n"
            "Using an incorrect tokenizer will cause gibberish outputs."
        )

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
