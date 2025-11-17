#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model loading utilities for MiniLM sentence transformers.

Handles downloading from HuggingFace Hub and loading weights.
"""

import glob
import json
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .config import ModelConfig, DEFAULT_CONFIG_L6
from .model import MiniLM


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
                    "*.txt",
                    "*.model",
                    "*.tiktoken",
                ],
                force_download=force_download,
            )
        )
        print(f"Model downloaded to: {model_path}")

    return model_path


def load_config(model_path: Path) -> ModelConfig:
    """Load model configuration from config.json.

    Args:
        model_path: Path to model directory

    Returns:
        ModelConfig instance
    """
    config_path = model_path / "config.json"

    if not config_path.exists():
        print(
            f"Warning: config.json not found in {model_path}, using default L6 config"
        )
        return DEFAULT_CONFIG_L6

    with open(config_path) as f:
        config_dict = json.load(f)

    # Check for pooling config (sentence-transformers format)
    pooling_config_path = model_path / "1_Pooling" / "config.json"
    if pooling_config_path.exists():
        with open(pooling_config_path) as f:
            pooling_config = json.load(f)
            config_dict.update(pooling_config)

    return ModelConfig.from_dict(config_dict)


def load_weights(model_path: Path) -> dict:
    """Load model weights from safetensors or npz files.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary of weights
    """
    # Try safetensors first (HuggingFace format)
    weight_files = glob.glob(str(model_path / "*.safetensors"))

    if not weight_files:
        # Try npz (MLX format)
        weight_files = glob.glob(str(model_path / "*.npz"))

    if not weight_files:
        raise FileNotFoundError(
            f"No weight files found in {model_path}. "
            f"Expected .safetensors or .npz files."
        )

    print(f"Loading weights from {len(weight_files)} file(s)...")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    return weights


def load(
    variant: str = "all-MiniLM-L6-v2",
    path_or_hf_repo: Optional[str] = None,
    revision: Optional[str] = None,
    lazy: bool = False,
    force_download: bool = False,
) -> Tuple[MiniLM, AutoTokenizer]:
    """Load MiniLM model and tokenizer.

    Args:
        variant: Model variant name (e.g., "all-MiniLM-L6-v2", "all-MiniLM-L12-v2")
        path_or_hf_repo: Override with custom path or HF repo
        revision: Git revision (branch, tag, or commit hash)
        lazy: If False, eagerly load all weights into memory
        force_download: Force re-download from HuggingFace Hub

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> from smlx.models.MiniLM import load
        >>> model, tokenizer = load("all-MiniLM-L6-v2")
    """
    # Map variant names to HuggingFace repos
    variant_map = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
        "paraphrase-MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    }

    if path_or_hf_repo is None:
        if variant in variant_map:
            path_or_hf_repo = variant_map[variant]
        else:
            # Assume it's a HuggingFace repo or local path
            path_or_hf_repo = variant

    # Ensure model is available locally
    model_path = get_model_path(path_or_hf_repo, revision, force_download)

    # Load configuration
    config = load_config(model_path)

    # Initialize model
    print(f"Initializing {variant} model...")
    model = MiniLM(config)

    # Load weights
    weights = load_weights(model_path)

    # Sanitize weights (PyTorch -> MLX conversion)
    print("Sanitizing weights...")
    weights = MiniLM.sanitize(weights)

    # Load weights into model
    print("Loading weights into model...")
    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    print("✓ Model loaded successfully!")
    return model, tokenizer


__all__ = ["load", "get_model_path", "load_config", "load_weights"]
