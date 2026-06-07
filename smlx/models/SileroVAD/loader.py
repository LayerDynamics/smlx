#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model loading utilities for Silero VAD.

Handles downloading from HuggingFace Hub and loading weights.
"""

import glob
import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download

from .config import DEFAULT_CONFIG_16K, VADConfig
from .model import SileroVAD


def get_model_path(
    model_name: str = "silero-vad",
    force_download: bool = False,
) -> Path:
    """Get path to Silero VAD model.

    Args:
        model_name: Model name/variant
        force_download: Force re-download

    Returns:
        Path to model directory
    """
    # Silero VAD models on HuggingFace
    repo_map = {
        "silero-vad": "silero/silero-vad",
        "silero-vad-v3": "silero/silero-vad",
        "silero-vad-v4": "silero/silero-vad",
    }

    repo_id = repo_map.get(model_name, "silero/silero-vad")

    try:
        # Download model file
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename="silero_vad.onnx",  # ONNX format
            force_download=force_download,
        )

        return Path(model_file).parent

    except Exception as e:
        print(f"Warning: Could not download from HuggingFace: {e}")
        print("Using default model configuration without pretrained weights")
        return Path(".")


def load_config(model_path: Path, sample_rate: int = 16000) -> VADConfig:
    """Load VAD configuration.

    Args:
        model_path: Path to model directory
        sample_rate: Target sample rate

    Returns:
        VADConfig instance
    """
    config_path = model_path / "config.json"

    if config_path.exists():
        try:
            with open(config_path) as f:
                config_dict = json.load(f)
            return VADConfig.from_dict(config_dict)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")

    # Use default config
    if sample_rate == 8000:
        from .config import DEFAULT_CONFIG_8K

        return DEFAULT_CONFIG_8K
    else:
        return DEFAULT_CONFIG_16K


def load_weights(model_path: Path) -> Optional[dict]:
    """Load model weights.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary of weights or None
    """
    # Try different weight file formats
    weight_patterns = [
        "*.safetensors",
        "*.npz",
        "*.onnx",
        "*.pth",
    ]

    weight_files = []
    for pattern in weight_patterns:
        weight_files.extend(glob.glob(str(model_path / pattern)))

    if not weight_files:
        print("Warning: No weight files found")
        return None

    # Load weights (prefer safetensors or npz)
    for wf in weight_files:
        if wf.endswith((".safetensors", ".npz")):
            try:
                weights = mx.load(wf)
                print(f"Loaded weights from {wf}")
                return weights
            except Exception as e:
                print(f"Warning: Could not load {wf}: {e}")

    # If only ONNX is available (Silero's native format), convert its weights to
    # MLX. This requires the optional `onnx` package; if it is missing or the file
    # has no weights we raise a clear error rather than silently returning None
    # (which would leave the VAD running on random, untrained weights).
    onnx_files = [wf for wf in weight_files if wf.endswith(".onnx")]
    if onnx_files:
        print(f"ONNX format detected ({onnx_files[0]}). Converting weights to MLX...")
        weights = _convert_onnx_weights(onnx_files[0])
        print(f"Converted {len(weights)} weight tensors from ONNX")
        return weights

    return None


def _convert_onnx_weights(onnx_path: str) -> dict:
    """Convert the trained weights in an ONNX model to an MLX weight dict.

    Reads every initializer (the trained parameters) from the ONNX graph and
    converts it to an ``mx.array`` keyed by its ONNX tensor name.

    Args:
        onnx_path: Path to the .onnx file.

    Returns:
        Dict mapping ONNX initializer names to MLX arrays.

    Raises:
        RuntimeError: If the optional ``onnx`` package is unavailable, or the
            file contains no initializer weights to convert.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as e:
        raise RuntimeError(
            "Silero VAD weights are in ONNX format, but the optional 'onnx' package "
            "is not installed. Install it with `pip install onnx` to convert them, "
            "or place pre-converted .safetensors/.npz weights in the model directory."
        ) from e

    onnx_model = onnx.load(onnx_path)
    weights = {
        initializer.name: mx.array(np.array(numpy_helper.to_array(initializer)))
        for initializer in onnx_model.graph.initializer
    }
    if not weights:
        raise RuntimeError(f"ONNX file '{onnx_path}' contains no initializer weights to convert.")
    return weights


def load(
    model_name: str = "silero-vad",
    sample_rate: int = 16000,
    force_download: bool = False,
) -> SileroVAD:
    """Load Silero VAD model.

    Args:
        model_name: Model variant name
        sample_rate: Audio sample rate (8000 or 16000)
        force_download: Force re-download

    Returns:
        Loaded SileroVAD model

    Example:
        >>> from smlx.models.SileroVAD import load
        >>> vad = load(sample_rate=16000)
        >>> # Use for voice activity detection
    """
    print(f"Loading Silero VAD model (sample_rate={sample_rate})...")

    # Get model path
    model_path = get_model_path(model_name, force_download)

    # Load configuration
    config = load_config(model_path, sample_rate)

    # Initialize model
    model = SileroVAD(config)

    # Load weights if available
    weights = load_weights(model_path)
    weights_loaded = False

    if weights:
        # Sanitize weights
        weights = SileroVAD.sanitize(weights)

        # Load into model
        try:
            model.load_weights(list(weights.items()))
            weights_loaded = True
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load weights into model: {e}")
            print("Using randomly initialized weights")

    else:
        print("Note: Using randomly initialized weights")
        print("For pretrained weights, you can:")
        print("  1. Download from https://github.com/snakers4/silero-vad")
        print("  2. Convert ONNX to MLX format")
        print("  3. Place in model directory")

    # Expose whether real pretrained weights were loaded so callers (e.g. the
    # unified runner) can honestly report output as trained vs pipeline-only.
    model.weights_loaded = weights_loaded

    model.eval()

    print("✓ Model initialized successfully!")
    return model


__all__ = ["load", "get_model_path", "load_config", "load_weights"]
