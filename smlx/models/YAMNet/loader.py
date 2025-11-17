#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
YAMNet Model Loading.

Handles downloading and loading YAMNet weights from HuggingFace Hub or
converting from PyTorch (torch_audioset) format.

Weight Sources:
    - Primary: mlx-community/yamnet (if uploaded to HuggingFace)
    - Fallback: PyTorch weights from w-hc/torch_audioset, auto-converted to MLX
    - Original: Google TensorFlow Hub (requires manual conversion)

License: Apache 2.0 (original YAMNet from Google)
"""

import json
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from .config import YAMNetConfig, DEFAULT_CONFIG, AUDIOSET_CLASSES
from .model import YAMNet
from .weights import get_pytorch_to_mlx_mapping, validate_weight_shapes


PYTORCH_WEIGHTS_URL = "https://github.com/w-hc/torch_audioset/releases/download/v0.1/yamnet.pth"


def download_pytorch_weights(
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """Download pre-converted PyTorch YAMNet weights from torch_audioset.

    Args:
        cache_dir: Directory to cache weights (default: ~/.cache/smlx/yamnet)
        force_download: Force re-download even if cached

    Returns:
        Path to downloaded PyTorch weights file
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "smlx" / "yamnet"

    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / "yamnet.pth"

    if weights_path.exists() and not force_download:
        return weights_path

    print(f"Downloading YAMNet PyTorch weights from torch_audioset...")
    print(f"  Source: {PYTORCH_WEIGHTS_URL}")
    print(f"  Target: {weights_path}")

    try:
        urllib.request.urlretrieve(PYTORCH_WEIGHTS_URL, weights_path)
        print(f"  ✓ Downloaded ({weights_path.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        raise RuntimeError(f"Failed to download PyTorch weights: {e}")

    return weights_path


def convert_pytorch_to_mlx(pytorch_path: Path) -> Dict:
    """Convert PyTorch YAMNet weights to MLX format.

    Args:
        pytorch_path: Path to PyTorch .pth file

    Returns:
        Dictionary of MLX weight arrays
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for weight conversion. "
            "Install with: pip install torch\n"
            "Or use pre-converted weights from HuggingFace Hub."
        )

    print(f"Converting PyTorch weights to MLX format...")

    # Load PyTorch state_dict
    state_dict = torch.load(pytorch_path, map_location='cpu')

    # Get name mapping
    name_mapping = get_pytorch_to_mlx_mapping()

    # Convert weights
    mlx_weights = {}

    for pytorch_key, pytorch_tensor in state_dict.items():
        # Skip PyTorch-specific tracking variables
        if 'num_batches_tracked' in pytorch_key:
            continue

        # Map to MLX key
        if pytorch_key in name_mapping:
            mlx_key = name_mapping[pytorch_key]

            # Convert PyTorch → NumPy → MLX
            numpy_array = pytorch_tensor.numpy()
            mlx_array = mx.array(numpy_array)

            mlx_weights[mlx_key] = mlx_array

    print(f"  ✓ Converted {len(mlx_weights)} weight tensors")

    # Validate shapes
    is_valid, errors = validate_weight_shapes(mlx_weights, strict=False)
    if not is_valid:
        print(f"  ⚠ Weight validation warnings:")
        for error in errors:
            print(f"    {error}")

    return mlx_weights


def get_model_path(
    model_name: str = "yamnet",
    force_download: bool = False,
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Get path to model files, downloading if necessary.

    Args:
        model_name: Model name or HuggingFace repo identifier
        force_download: Force re-download even if cached
        cache_dir: Cache directory for weights

    Returns:
        Path to model directory, or None if download fails
    """
    # Try HuggingFace Hub first (if MLX version exists)
    if model_name.startswith("mlx-community/") or model_name == "yamnet-mlx":
        try:
            from huggingface_hub import hf_hub_download

            repo_id = model_name if "/" in model_name else "mlx-community/yamnet"

            model_file = hf_hub_download(
                repo_id=repo_id,
                filename="yamnet.npz",
                force_download=force_download,
            )

            return Path(model_file).parent

        except ImportError:
            print("Warning: huggingface_hub not available. Install with: pip install huggingface-hub")
            return None
        except Exception as e:
            print(f"Note: Could not load from HuggingFace Hub ({repo_id}): {e}")
            print("Will download and convert from PyTorch instead.")
            return None

    # For default "yamnet", use PyTorch conversion path
    return None


def load_config(
    model_path: Optional[Path] = None,
) -> YAMNetConfig:
    """Load model configuration.

    Args:
        model_path: Path to model directory

    Returns:
        Model configuration
    """
    if model_path and (model_path / "config.json").exists():
        with open(model_path / "config.json") as f:
            config_dict = json.load(f)
        return YAMNetConfig(**config_dict)

    # Use default config
    return DEFAULT_CONFIG


def load_weights(
    model_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Dict:
    """Load model weights from HuggingFace Hub or convert from PyTorch.

    This function handles the complete weight loading pipeline:
    1. Try loading cached MLX weights
    2. Try downloading from HuggingFace Hub (if MLX version uploaded)
    3. Download PyTorch weights and convert to MLX
    4. Cache the converted MLX weights for future use

    Args:
        model_path: Path to model directory (if pre-downloaded)
        cache_dir: Cache directory for weights
        force_download: Force re-download and re-conversion

    Returns:
        Dictionary of MLX weight arrays

    Raises:
        RuntimeError: If weights cannot be loaded or converted
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "smlx" / "yamnet"

    # If model_path provided, try loading from there
    if model_path:
        npz_path = model_path / "yamnet.npz"
        if npz_path.exists():
            print(f"Loading weights from: {npz_path}")
            weights = mx.load(str(npz_path))
            return dict(weights)

        # Try safetensors
        safetensors_path = model_path / "model.safetensors"
        if safetensors_path.exists():
            try:
                print(f"Loading weights from: {safetensors_path}")
                weights = mx.load(str(safetensors_path))
                return dict(weights)
            except Exception as e:
                print(f"Warning: Could not load safetensors: {e}")

    # Check if MLX weights already cached
    cache_dir.mkdir(parents=True, exist_ok=True)
    mlx_cache_path = cache_dir / "yamnet_mlx.npz"

    if mlx_cache_path.exists() and not force_download:
        print(f"Loading cached MLX weights from: {mlx_cache_path}")
        weights = mx.load(str(mlx_cache_path))
        return dict(weights)

    # Download PyTorch weights
    print("No cached MLX weights found. Will download and convert from PyTorch...")
    pytorch_path = download_pytorch_weights(cache_dir, force_download)

    # Convert to MLX
    mlx_weights = convert_pytorch_to_mlx(pytorch_path)

    # Cache the converted weights
    print(f"Caching MLX weights to: {mlx_cache_path}")
    mx.savez(str(mlx_cache_path), **mlx_weights)
    print(f"  ✓ Cached for future use ({mlx_cache_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return mlx_weights


def load(
    model_name: str = "yamnet",
    force_download: bool = False,
    cache_dir: Optional[Path] = None,
) -> YAMNet:
    """Load YAMNet model with pre-trained weights.

    This function loads the YAMNet audio event classifier with real pre-trained weights.
    Weights are automatically downloaded and converted from PyTorch format on first use,
    then cached for subsequent loads.

    Weight Sources:
        - Primary: mlx-community/yamnet on HuggingFace (if uploaded)
        - Fallback: PyTorch weights from w-hc/torch_audioset (auto-converted)
        - License: Apache 2.0 (original YAMNet from Google)

    Args:
        model_name: Model identifier:
            - "yamnet" (default): Download and convert from PyTorch
            - "mlx-community/yamnet": Load from HuggingFace Hub (if available)
            - Custom HuggingFace repo ID
        force_download: Force re-download and re-conversion of weights
        cache_dir: Cache directory for weights (default: ~/.cache/smlx/yamnet)

    Returns:
        YAMNet model with pre-trained weights loaded

    Raises:
        RuntimeError: If weights cannot be downloaded or converted
        ImportError: If required dependencies are missing

    Example:
        >>> from smlx.models.YAMNet import load, classify
        >>> model = load()  # Downloads and converts weights on first run
        >>> predictions = classify(model, "audio.wav", top_k=5)
        >>> for pred in predictions:
        ...     print(f"{pred.label}: {pred.score:.3f}")

    Note:
        First-time load requires PyTorch for weight conversion:
            pip install torch

        Or use pre-converted weights from HuggingFace Hub:
            model = load("mlx-community/yamnet")
    """
    # Try to get model path from HuggingFace Hub
    model_path = get_model_path(model_name, force_download, cache_dir)

    # Load config
    config = load_config(model_path)

    # Load weights (handles download + conversion if needed)
    weights = load_weights(model_path, cache_dir, force_download)

    # Create model
    model = YAMNet(config)

    # Apply weights
    model.update(weights)
    print(f"✓ Loaded YAMNet model with {len(weights)} weight tensors")

    # Set to eval mode
    model.eval()

    return model


def save_model(
    model: YAMNet,
    save_path: Path,
    save_config: bool = True,
) -> None:
    """Save model weights and config.

    Args:
        model: YAMNet model to save
        save_path: Directory to save model
        save_config: Whether to save config file
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = save_path / "yamnet.npz"
    weights = dict(model.parameters())
    mx.save_safetensors(str(weights_path), weights)

    print(f"Saved model weights to {weights_path}")

    # Save config
    if save_config:
        config_path = save_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(model.config.__dict__, f, indent=2)
        print(f"Saved model config to {config_path}")


def load_class_names() -> list[str]:
    """Load AudioSet class names.

    Returns:
        List of 521 AudioSet class names
    """
    return AUDIOSET_CLASSES


__all__ = [
    "load",
    "save_model",
    "load_class_names",
    "load_weights",
    "get_model_path",
    "download_pytorch_weights",
    "convert_pytorch_to_mlx",
]
