# Copyright © 2025 SMLX Project

"""
Model loading utilities for all SMLX models.

Provides standardized functions for loading models, weights, and tokenizers
from local paths or HuggingFace Hub, with support for various weight formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

# Optional imports for HuggingFace integration
try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


def resolve_model_path(
    model_name_or_path: str | Path,
    revision: str | None = None,
    cache_dir: Path | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> Path:
    """
    Resolve model path from local or HuggingFace Hub.

    Args:
        model_name_or_path: Either local path or HuggingFace model ID
        revision: Specific git revision/tag/branch
        cache_dir: Custom cache directory
        allow_patterns: File patterns to include
        ignore_patterns: File patterns to exclude

    Returns:
        Absolute path to model directory

    Example:
        >>> # Local path
        >>> path = resolve_model_path("./models/smollm2-135m")
        >>> # HuggingFace Hub
        >>> path = resolve_model_path("mlx-community/SmolLM2-135M-Instruct")
    """
    model_path = Path(model_name_or_path)

    # Check if it's a local path
    if model_path.exists():
        return model_path.resolve()

    # Try to download from HuggingFace Hub
    if snapshot_download is None:
        raise ImportError(
            "huggingface_hub is required for downloading models. "
            "Install with: pip install huggingface_hub"
        )

    print(f"Downloading model from HuggingFace Hub: {model_name_or_path}")

    # Default patterns optimized for MLX models
    if allow_patterns is None:
        allow_patterns = [
            "*.safetensors",
            "*.npz",
            "*.json",
            "*.txt",
            "*.py",
            "*.model",
            "tokenizer*",
        ]

    if ignore_patterns is None:
        ignore_patterns = [
            "*.bin",  # PyTorch weights
            "*.pt",  # PyTorch checkpoints
            "*.onnx",  # ONNX models
            "*.msgpack",  # Flax weights
        ]

    try:
        from smlx.tools.download import get_cache_dir

        if cache_dir is None:
            cache_dir = get_cache_dir() / "models"

        local_path = Path(
            snapshot_download(
                repo_id=str(model_name_or_path),
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_files_only=False,
            )
        )
        print(f"✓ Downloaded to: {local_path}")
        return local_path

    except Exception as e:
        raise ValueError(
            f"Could not find model at '{model_name_or_path}' locally or on HuggingFace Hub. "
            f"Error: {e}"
        ) from e


def load_weights(
    model_path: str | Path,
    lazy: bool = False,
) -> dict[str, mx.array]:
    """
    Load model weights from safetensors or npz file.

    Supports both single-file and sharded weights.

    Args:
        model_path: Path to weights file or directory containing weights
        lazy: If True, weights are loaded on-demand (lazy loading)

    Returns:
        dictionary mapping parameter names to MLX arrays

    Raises:
        FileNotFoundError: If no weights file is found
        ValueError: If weights format is unsupported

    Example:
        >>> weights = load_weights("./models/smollm2-135m")
        >>> print(weights.keys())
        dict_keys(['model.embed_tokens.weight', ...])
    """
    model_path = Path(model_path)

    # Find weights file
    if model_path.is_file():
        weights_path = model_path
    else:
        # Look for weights in directory
        weights_path = None

        # Check for sharded weights (model.safetensors.index.json)
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            return load_sharded_weights(model_path, lazy=lazy)

        # Look for single weight file
        for ext in [".safetensors", ".npz"]:
            candidates = list(model_path.glob(f"*{ext}"))
            if candidates:
                # Prefer model.safetensors or weights.safetensors
                for name in ["model.safetensors", "weights.safetensors", "model.npz", "weights.npz"]:
                    if (model_path / name).exists():
                        weights_path = model_path / name
                        break
                if weights_path is None:
                    weights_path = candidates[0]
                break

        if weights_path is None:
            raise FileNotFoundError(
                f"No weights file (.safetensors or .npz) found in {model_path}"
            )

    # Load weights
    if weights_path.suffix in [".safetensors", ".npz"]:
        weights = mx.load(str(weights_path))
    else:
        raise ValueError(f"Unsupported weights format: {weights_path.suffix}")

    # Ensure weights is a dict
    if not isinstance(weights, dict):
        raise ValueError(f"Expected weights to be a dict, got {type(weights)}")

    # Evaluate weights if not lazy
    if not lazy:
        weights = {k: mx.array(v) for k, v in weights.items()}
        mx.eval(weights)

    return weights


def load_sharded_weights(
    model_path: Path,
    lazy: bool = False,
) -> dict[str, mx.array]:
    """
    Load sharded model weights.

    Reads model.safetensors.index.json to determine which parameters
    are in which shard files, then loads all shards.

    Args:
        model_path: Path to directory containing sharded weights
        lazy: If True, use lazy loading

    Returns:
        dictionary of all weights from all shards
    """
    index_path = model_path / "model.safetensors.index.json"

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Get unique shard files
    shard_files = set(weight_map.values())

    # Load all shards
    all_weights = {}
    for shard_file in shard_files:
        shard_path = model_path / shard_file
        shard_weights = mx.load(str(shard_path))

        if not isinstance(shard_weights, dict):
            raise ValueError(f"Expected shard to be a dict, got {type(shard_weights)}")

        all_weights.update(shard_weights)

    # Evaluate if not lazy
    if not lazy:
        all_weights = {k: mx.array(v) for k, v in all_weights.items()}
        mx.eval(all_weights)

    return all_weights


def save_weights(
    weights: dict[str, mx.array],
    save_path: str | Path,
    max_shard_size_gb: float = 5.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save model weights with automatic sharding for large models.

    Args:
        weights: dictionary of weights to save
        save_path: Directory or file path to save to
        max_shard_size_gb: Maximum size per shard in GB (default: 5GB)
        metadata: Optional metadata to include

    Example:
        >>> save_weights(model.parameters(), "./my_model")
    """
    save_path = Path(save_path)

    # If save_path is a directory, create model.safetensors
    if save_path.is_dir() or not save_path.suffix:
        save_path.mkdir(parents=True, exist_ok=True)
        weights_path = save_path / "model.safetensors"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path = save_path

    # Evaluate weights before saving
    mx.eval(weights)

    # Calculate total size
    total_size = sum(w.nbytes for w in weights.values())
    max_shard_size = int(max_shard_size_gb * 1024**3)

    # Check if we need sharding
    if total_size <= max_shard_size:
        # Save as single file
        mx.save_safetensors(str(weights_path), weights, metadata=metadata)
    else:
        # Save as sharded files
        save_sharded_weights(
            weights, weights_path.parent, max_shard_size, metadata=metadata
        )


def save_sharded_weights(
    weights: dict[str, mx.array],
    save_dir: Path,
    max_shard_size: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save weights as multiple sharded files.

    Creates model-00001-of-NNNNN.safetensors files and an index file.

    Args:
        weights: dictionary of weights
        save_dir: Directory to save to
        max_shard_size: Maximum shard size in bytes
        metadata: Optional metadata
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sort weights by size (largest first) for better packing
    sorted_weights = sorted(weights.items(), key=lambda x: x[1].nbytes, reverse=True)

    # Pack weights into shards
    shards = []
    current_shard = {}
    current_size = 0

    for name, weight in sorted_weights:
        weight_size = weight.nbytes

        if current_size + weight_size > max_shard_size and current_shard:
            # Start new shard
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[name] = weight
        current_size += weight_size

    # Add last shard
    if current_shard:
        shards.append(current_shard)

    # Save shards
    weight_map = {}
    for i, shard in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        shard_path = save_dir / shard_filename

        mx.save_safetensors(str(shard_path), shard, metadata=metadata)

        # Update weight map
        for name in shard.keys():
            weight_map[name] = shard_filename

    # Save index file
    index = {
        "metadata": metadata or {},
        "weight_map": weight_map,
    }

    index_path = save_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def sanitize_weights(
    weights: dict[str, mx.array],
    remove_patterns: list[str] | None = None,
    model: Any | None = None,
) -> dict[str, mx.array]:
    """
    Remove unnecessary keys from weights dictionary.

    Some weights downloaded from HuggingFace may contain keys that are
    not needed for inference (e.g., optimizer states, unused parameters).

    Args:
        weights: dictionary of weights
        remove_patterns: List of patterns to remove (e.g., ["rotary_emb.inv_freq"])
        model: Optional model instance to check against

    Returns:
        Sanitized weights dictionary

    Example:
        >>> weights = sanitize_weights(weights, remove_patterns=["rotary_emb.inv_freq"])
    """
    if remove_patterns is None:
        remove_patterns = [
            "rotary_emb.inv_freq",  # RoPE inverse frequencies (computed on-the-fly)
            "_orig_mod.",  # PyTorch compilation artifacts
        ]

    # Remove patterns
    sanitized = {}
    for key, value in weights.items():
        should_remove = any(pattern in key for pattern in remove_patterns)

        if not should_remove:
            sanitized[key] = value

    # If model is provided, filter to only parameters that exist in model
    if model is not None:
        model_params = set(dict(model.named_parameters()).keys())
        sanitized = {k: v for k, v in sanitized.items() if k in model_params}

    return sanitized


def load_tokenizer(
    model_path: str | Path,
    trust_remote_code: bool = False,
):
    """
    Load tokenizer using transformers library.

    Args:
        model_path: Path to directory containing tokenizer files
        trust_remote_code: Whether to trust remote code in tokenizer

    Returns:
        Loaded tokenizer (transformers.PreTrainedTokenizer)

    Example:
        >>> tokenizer = load_tokenizer("mlx-community/SmolLM2-135M-Instruct")
    """
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for tokenizer loading. "
            "Install with: pip install transformers"
        )

    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
    )
    return tokenizer


def detect_quantization(weights: dict[str, mx.array]) -> dict[str, Any] | None:
    """
    Detect if weights contain pre-quantized layers.

    Checks for quantization metadata (.scales, .biases keys) in the weights
    dictionary to determine if the model was pre-quantized.

    Args:
        weights: dictionary of model weights

    Returns:
        Dictionary with quantization info if detected, None otherwise:
        - is_quantized: bool
        - num_quantized_layers: int
        - quantized_layers: list of layer names
        - estimated_bits: int (estimated from scales shape)

    Example:
        >>> weights = load_weights("./quantized_model")
        >>> quant_info = detect_quantization(weights)
        >>> if quant_info:
        ...     print(f"Model has {quant_info['num_quantized_layers']} quantized layers")
    """
    quantized_layers = []

    # Look for layers with .scales suffix (indicates quantization)
    for key in weights.keys():
        if key.endswith(".scales"):
            # Extract base layer name
            layer_name = key[:-7]  # Remove ".scales"

            # Verify corresponding .biases exists
            if f"{layer_name}.biases" in weights:
                quantized_layers.append(layer_name)

    if not quantized_layers:
        return None

    # Try to estimate bits from first quantized layer
    estimated_bits = None
    first_layer = quantized_layers[0]

    # Check for explicit bits metadata
    if f"{first_layer}.bits" in weights:
        estimated_bits = int(weights[f"{first_layer}.bits"])
    else:
        # Estimate from weight/scales ratio (rough approximation)
        if f"{first_layer}.weight" in weights:
            weight_shape = weights[f"{first_layer}.weight"].shape
            scales_shape = weights[f"{first_layer}.scales"].shape

            # Common pattern: 4-bit uses 8x compression
            if weight_shape[0] // scales_shape[0] >= 7:
                estimated_bits = 4
            else:
                estimated_bits = 8

    return {
        "is_quantized": True,
        "num_quantized_layers": len(quantized_layers),
        "quantized_layers": quantized_layers,
        "estimated_bits": estimated_bits,
    }


def get_quantized_layers(weights: dict[str, mx.array]) -> dict[str, dict[str, Any]]:
    """
    Get detailed information about quantized layers in weights.

    Args:
        weights: dictionary of model weights

    Returns:
        Dictionary mapping layer names to their quantization info:
        {
            "layer.name": {
                "has_scales": bool,
                "has_biases": bool,
                "has_group_size": bool,
                "has_bits": bool,
                "scales_shape": tuple,
                "biases_shape": tuple,
                "weight_shape": tuple,
            }
        }

    Example:
        >>> quant_layers = get_quantized_layers(weights)
        >>> for layer, info in quant_layers.items():
        ...     print(f"{layer}: {info['scales_shape']}")
    """
    quantized_info = {}

    # Find all layers with .scales
    for key in weights.keys():
        if key.endswith(".scales"):
            layer_name = key[:-7]  # Remove ".scales"

            info = {
                "has_scales": True,
                "has_biases": f"{layer_name}.biases" in weights,
                "has_group_size": f"{layer_name}.group_size" in weights,
                "has_bits": f"{layer_name}.bits" in weights,
                "scales_shape": weights[key].shape,
            }

            if info["has_biases"]:
                info["biases_shape"] = weights[f"{layer_name}.biases"].shape

            if f"{layer_name}.weight" in weights:
                info["weight_shape"] = weights[f"{layer_name}.weight"].shape

            # Extract metadata if present
            if info["has_group_size"]:
                info["group_size"] = int(weights[f"{layer_name}.group_size"])

            if info["has_bits"]:
                info["bits"] = int(weights[f"{layer_name}.bits"])

            quantized_info[layer_name] = info

    return quantized_info


def verify_weights(
    weights: dict[str, mx.array],
    expected_keys: list[str] | None = None,
    model: Any | None = None,
) -> bool:
    """
    Verify that weights dictionary is complete and valid.

    Args:
        weights: dictionary of weights to verify
        expected_keys: Optional list of expected keys
        model: Optional model to check against

    Returns:
        True if weights are valid

    Raises:
        ValueError: If weights are invalid

    Example:
        >>> verify_weights(weights, expected_keys=["model.embed_tokens.weight", ...])
    """
    # Check that weights is a dict
    if not isinstance(weights, dict):
        raise ValueError(f"weights must be a dict, got {type(weights)}")

    # Check that weights is not empty
    if not weights:
        raise ValueError("weights dictionary is empty")

    # Check expected keys
    if expected_keys is not None:
        missing_keys = set(expected_keys) - set(weights.keys())
        if missing_keys:
            raise ValueError(f"Missing expected keys: {missing_keys}")

    # Check against model
    if model is not None:
        model_params = set(dict(model.named_parameters()).keys())
        weight_keys = set(weights.keys())

        missing = model_params - weight_keys
        unexpected = weight_keys - model_params

        if missing:
            raise ValueError(f"Missing keys in weights: {missing}")
        if unexpected:
            # Unexpected keys are just a warning, not an error
            print(f"Warning: Unexpected keys in weights: {unexpected}")

    return True


__all__ = [
    "resolve_model_path",
    "load_weights",
    "load_sharded_weights",
    "save_weights",
    "save_sharded_weights",
    "sanitize_weights",
    "load_tokenizer",
    "detect_quantization",
    "get_quantized_layers",
    "verify_weights",
    "snapshot_download",
    "AutoTokenizer",
]
