#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model loading utilities for TinyLLaVA.

Handles downloading from HuggingFace Hub, loading weights, and
initializing tokenizer + image processor.
"""

import glob
import json
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
from huggingface_hub import snapshot_download
from transformers import LlamaTokenizerFast

from .config import ModelConfig, DEFAULT_CONFIG_1_5B, DEFAULT_CONFIG_2_0B, DEFAULT_CONFIG_3_1B
from .image_processor import ImageProcessor
from .model import TinyLLaVA


class Processor:
    """Combined tokenizer and image processor."""

    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, text=None, images=None, **kwargs):
        """Process text and/or images."""
        result = {}

        if text is not None:
            result.update(self.tokenizer(text, **kwargs))

        if images is not None:
            processed_images = self.image_processor(images)
            result["pixel_values"] = processed_images

        return result


def get_model_path(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """Ensure model is available locally."""
    model_path = Path(path_or_hf_repo)

    if not model_path.exists() or force_download:
        print(f"Downloading model from HuggingFace Hub: {path_or_hf_repo}")
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.model",
                    "*.tiktoken",
                ],
                force_download=force_download,
            )
        )
        print(f"Model downloaded to: {model_path}")

    return model_path


def load_config(model_path: Path, variant: str = "1.5b") -> ModelConfig:
    """Load model configuration."""
    config_path = model_path / "config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        try:
            config = ModelConfig.from_dict(config_dict)
            # FIXME: HuggingFace config incorrectly says 27 vision layers, but weights have 26
            # Override to match actual weights
            if config.vision_config.num_hidden_layers == 27:
                from dataclasses import replace
                config = replace(
                    config,
                    vision_config=replace(config.vision_config, num_hidden_layers=26)
                )
            return config
        except Exception:
            print("Warning: Could not load config, using defaults")

    # Use default config based on variant
    if variant == "2.0b":
        return DEFAULT_CONFIG_2_0B
    elif variant == "3.1b":
        return DEFAULT_CONFIG_3_1B
    else:
        return DEFAULT_CONFIG_1_5B


def load_weights(model_path: Path) -> dict:
    """Load model weights from safetensors files."""
    weight_files = glob.glob(str(model_path / "*.safetensors"))

    if not weight_files:
        raise FileNotFoundError(
            f"No safetensors files found in {model_path}"
        )

    print(f"Loading weights from {len(weight_files)} files...")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    return weights


def load(
    path_or_hf_repo: str = "bczhou/TinyLLaVA-1.5B",
    variant: str = "1.5b",
    revision: Optional[str] = None,
    lazy: bool = False,
    force_download: bool = False,
) -> Tuple[TinyLLaVA, Processor]:
    """Load TinyLLaVA model and processor.

    Args:
        path_or_hf_repo: HuggingFace repo ID or local path
        variant: Model variant ("1.5b", "2.0b", "3.1b")
        revision: Git revision
        lazy: If False, eagerly load all weights
        force_download: Force re-download

    Returns:
        Tuple of (model, processor)

    Example:
        >>> from smlx.models.TinyLLaVA import load
        >>> model, processor = load("bczhou/TinyLLaVA-1.5B")
    """
    # Map variants to repos
    variant_map = {
        "1.5b": "bczhou/TinyLLaVA-1.5B",
        "2.0b": "bczhou/TinyLLaVA-2.0B",
        "3.1b": "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",
    }

    if path_or_hf_repo is None:
        path_or_hf_repo = variant_map.get(variant, "bczhou/TinyLLaVA-1.5B")

    # Get model path
    model_path = get_model_path(path_or_hf_repo, revision, force_download)

    # Load config
    config = load_config(model_path, variant)

    # Load tokenizer FIRST (before model weights to avoid memory interaction issues)
    # Use LlamaTokenizerFast explicitly to avoid sentencepiece segfault in pytest
    print("Loading tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(str(model_path))

    # Add <image> special token if not already present
    # LLaVA uses IMAGE_TOKEN_INDEX = vocab_size (32000 for TinyLlama)
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        config.image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        print(f"Added <image> token with ID: {config.image_token_index}")
    else:
        config.image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        print(f"Using existing <image> token with ID: {config.image_token_index}")

    # Initialize model
    print(f"Initializing TinyLLaVA-{variant.upper()} model...")
    model = TinyLLaVA(config)

    # Load weights
    weights = load_weights(model_path)

    # Sanitize weights
    print("Sanitizing weights...")
    weights = TinyLLaVA.sanitize(weights)

    # Load into model
    print("Loading weights into model...")
    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()

    # Create image processor
    image_processor = ImageProcessor(config.vision_config)

    # Create combined processor
    processor = Processor(tokenizer, image_processor)

    print("✓ Model loaded successfully!")
    return model, processor


__all__ = ["load", "Processor", "get_model_path"]
