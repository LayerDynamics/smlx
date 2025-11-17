"""
Model loading utilities for Whisper-tiny.

Handles downloading models from HuggingFace Hub and loading weights.
"""

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from .model import ModelConfig, Whisper
from .tokenizer import WhisperTokenizer, get_tokenizer


def load(
    model_path: str = "mlx-community/whisper-tiny",
    dtype: mx.Dtype = mx.float16,
) -> tuple[Whisper, WhisperTokenizer]:
    """Load Whisper model and tokenizer.

    Args:
        model_path: HuggingFace model ID or local path
        dtype: Model dtype (float16, float32, or bfloat16)

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load("mlx-community/whisper-tiny")
        >>> # Model is ready for inference
        >>> audio_features = model.encode_audio(mel_spectrogram)
    """
    # Convert dtype to string for config
    dtype_map = {
        mx.float16: "float16",
        mx.float32: "float32",
        mx.bfloat16: "bfloat16",
    }
    dtype_str = dtype_map.get(dtype, "float16")

    # Load model
    model = load_model(model_path, dtype_str)

    # Load tokenizer
    tokenizer = load_tokenizer(model_path)

    return model, tokenizer


def load_model(
    path_or_hf_repo: str,
    dtype: str = "float16",
) -> Whisper:
    """Load Whisper model from path or HuggingFace Hub.

    Args:
        path_or_hf_repo: Local path or HuggingFace repo ID
        dtype: Model dtype ('float16', 'float32', or 'bfloat16')

    Returns:
        Loaded Whisper model

    Example:
        >>> model = load_model("mlx-community/whisper-tiny")
        >>> model.is_multilingual
        True
    """
    # Resolve model path
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        print(f"Downloading model from HuggingFace Hub: {path_or_hf_repo}")
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.npz", "*.safetensors"],
            )
        )

    # Load configuration
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    # Remove non-config fields
    config_dict.pop("model_type", None)
    quantization_config = config_dict.pop("quantization", None)

    # Add dtype to config
    config_dict["dtype"] = dtype

    # Create model config
    config = ModelConfig.from_dict(config_dict)

    # Load weights
    weights_path = model_path / "weights.npz"
    if not weights_path.exists():
        # Try .safetensors
        weights_path = model_path / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No weights found at {model_path}. "
                f"Expected weights.npz or model.safetensors"
            )

    print(f"Loading weights from {weights_path}")
    weights = mx.load(str(weights_path))
    # mx.load returns a dict-like object with .items() method
    weights = tree_unflatten(list(weights.items()))  # type: ignore[attr-defined]

    # Create model
    model = Whisper(config)

    # Apply quantization if specified
    if quantization_config is not None:
        print(f"Applying quantization: {quantization_config}")

        def is_quantizable_layer(path: str, module) -> bool:
            """Check if a layer should be quantized."""
            return isinstance(module, (nn.Linear, nn.Embedding)) and f"{path}.scales" in weights

        nn.quantize(model, **quantization_config, class_predicate=is_quantizable_layer)

    # Load weights
    model.update(weights)
    mx.eval(model.parameters())

    # Count parameters (flatten nested parameter structure)
    def count_params(params):
        """Recursively count parameters in nested dict/list structure."""
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, list):
            for item in params:
                total += count_params(item)
        else:
            # MLX array - has .size attribute
            total += params.size  # type: ignore[attr-defined]
        return total

    param_count = count_params(model.parameters())
    print(f"Model loaded: {config.n_audio_layer}-layer encoder, "
          f"{config.n_text_layer}-layer decoder, "
          f"{param_count / 1e6:.1f}M parameters")

    return model


def load_tokenizer(path_or_hf_repo: str) -> WhisperTokenizer:
    """Load Whisper tokenizer.

    Args:
        path_or_hf_repo: Local path or HuggingFace repo ID

    Returns:
        WhisperTokenizer instance

    Example:
        >>> tokenizer = load_tokenizer("mlx-community/whisper-tiny")
        >>> tokens = tokenizer.encode("Hello, world!")
    """
    # Resolve model path
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json"],
            )
        )

    # Load config to determine if multilingual
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    # Determine if model is multilingual
    # Multilingual models have 51865 vocab tokens, English-only have 51864
    n_vocab = config_dict.get("n_vocab", 51865)
    is_multilingual = n_vocab >= 51865

    # Get tokenizer using get_tokenizer (doesn't require tokenizer files)
    return get_tokenizer(
        multilingual=is_multilingual,
        num_languages=99 if is_multilingual else 1,
        language=None,  # Will be set when transcribing
        task="transcribe",  # Default task
    )


def save_model(
    model: Whisper,
    save_path: str | Path,
    tokenizer: Optional[WhisperTokenizer] = None,
):
    """Save Whisper model to disk.

    Args:
        model: Model to save
        save_path: Directory to save model
        tokenizer: Optional tokenizer to save

    Example:
        >>> model, tokenizer = load("mlx-community/whisper-tiny")
        >>> save_model(model, "my_whisper_model", tokenizer)
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = save_path / "config.json"
    with open(config_path, "w") as f:
        config_dict = model.config.to_dict()
        config_dict["model_type"] = "whisper"
        json.dump(config_dict, f, indent=2)

    # Save weights
    weights = dict(tree_flatten(model.parameters()))
    weights_path = save_path / "weights.npz"
    mx.savez(str(weights_path), **weights)

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(str(save_path))

    print(f"Model saved to {save_path}")


def tree_flatten(tree, prefix=""):
    """Flatten nested dictionary into flat dictionary with dot-separated keys.

    Args:
        tree: Nested dictionary
        prefix: Key prefix

    Returns:
        Flattened dictionary
    """
    flat = {}
    for key, value in tree.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(tree_flatten(value, new_key))
        else:
            flat[new_key] = value
    return flat


def load_model_from_path(model_path: str) -> Whisper:
    """Load model from local directory.

    Convenience function for loading from local path.

    Args:
        model_path: Path to model directory

    Returns:
        Loaded Whisper model

    Example:
        >>> model = load_model_from_path("./my_whisper_model")
    """
    return load_model(model_path)
