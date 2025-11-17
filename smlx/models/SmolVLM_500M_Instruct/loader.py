#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model loading utilities for SmolVLM-500M-Instruct.

Handles downloading from HuggingFace Hub, loading weights, and
initializing tokenizer + image processor.
"""

import glob
import json
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .config import ModelConfig
from .image_processor import ImageProcessor
from .language import LanguageModel
from .model import Model
from .vision import VisionModel


class Processor:
    """Combined tokenizer and image processor.

    Mimics transformers AutoProcessor interface for compatibility.
    """

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
    """Ensure model is available locally.

    Downloads from HuggingFace Hub if not found locally.

    Args:
        path_or_hf_repo: Local path or HuggingFace repo ID
        revision: Git revision (branch, tag, or commit hash)
        force_download: Force re-download even if exists

    Returns:
        Path to model directory
    """
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
                    "*.py",
                    "*.model",
                    "*.tiktoken",
                    "*.txt",
                ],
                force_download=force_download,
            )
        )
        print(f"Model downloaded to: {model_path}")

    return model_path


def load_config(model_path: Path) -> dict:
    """Load model configuration from config.json.

    Args:
        model_path: Path to model directory

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config.json not found
    """
    config_path = model_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found in {model_path}. "
            f"Make sure the model directory contains config.json"
        )

    with open(config_path) as f:
        config = json.load(f)

    return config


def load_weights(model_path: Path) -> dict:
    """Load model weights from safetensors files.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary of weights

    Raises:
        FileNotFoundError: If no safetensors files found
    """
    weight_files = glob.glob(str(model_path / "*.safetensors"))

    if not weight_files:
        raise FileNotFoundError(
            f"No safetensors files found in {model_path}. "
            f"The model needs to be converted to MLX format first."
        )

    print(f"Loading weights from {len(weight_files)} files...")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    return weights


def load(
    path_or_hf_repo: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
    revision: Optional[str] = None,
    lazy: bool = False,
    force_download: bool = False,
) -> Tuple[Model, Processor]:
    """Load SmolVLM-500M model and processor.

    Args:
        path_or_hf_repo: Local path or HuggingFace repo ID
        revision: Git revision (branch, tag, or commit hash)
        lazy: If False, eagerly load all weights into memory
        force_download: Force re-download from HuggingFace Hub

    Returns:
        Tuple of (model, processor)

    Example:
        >>> from smlx.models.SmolVLM_500M_Instruct import load
        >>> model, processor = load()
        >>> # Or from local path:
        >>> model, processor = load("/path/to/model")
    """
    # Ensure model is available locally
    model_path = get_model_path(path_or_hf_repo, revision, force_download)

    # Load configuration
    config_dict = load_config(model_path)

    # Create model config
    model_config = ModelConfig.from_dict(config_dict)

    # Initialize model
    print("Initializing SmolVLM-500M model...")
    model = Model(model_config)

    # Load weights
    weights = load_weights(model_path)

    # Sanitize weights (PyTorch -> MLX conversion)
    print("Sanitizing weights...")
    weights = model.sanitize(weights)
    weights = VisionModel.sanitize(weights)
    weights = LanguageModel.sanitize(weights)

    # Load weights into model
    print("Loading weights into model...")
    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Create image processor
    print("Creating image processor...")
    image_processor = ImageProcessor()

    # Create combined processor
    processor = Processor(tokenizer=tokenizer, image_processor=image_processor)

    print("✓ Model loaded successfully!")
    return model, processor


def save_model(
    model: Model,
    save_path: str,
    tokenizer=None,
) -> None:
    """Save model weights and config.

    Args:
        model: SmolVLM model to save
        save_path: Directory to save model
        tokenizer: Optional tokenizer to save alongside model
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    print(f"Saving model weights to {save_path}...")
    weights = dict(model.parameters())
    mx.save_safetensors(str(save_path / "model.safetensors"), weights)

    # Save config
    config_dict = {
        "model_type": model.config.model_type,
        "vocab_size": model.config.vocab_size,
        "scale_factor": model.config.scale_factor,
        "image_token_id": model.config.image_token_id,
        "text_config": model.config.text_config.__dict__,
        "vision_config": model.config.vision_config.__dict__,
    }

    with open(save_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save tokenizer if provided
    if tokenizer is not None:
        print("Saving tokenizer...")
        tokenizer.save_pretrained(str(save_path))

    print(f"✓ Model saved to {save_path}")


__all__ = ["load", "save_model", "Processor"]
