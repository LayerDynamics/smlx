# Copyright © 2025 SMLX Project

"""
Model loading utilities for SmolLM2-135M-Instruct.

Handles downloading from HuggingFace Hub, loading weights, and initializing tokenizers.
"""

import json
from pathlib import Path
from typing import Literal, Optional, Union

import mlx.core as mx

from smlx.utils.loading import (
    load_tokenizer as utils_load_tokenizer,
    load_weights as utils_load_weights,
    resolve_model_path,
    save_weights as utils_save_weights,
)

from .config import load_config, validate_config
from .model import Model, ModelArgs

QuantizePreset = Literal["auto", "4bit", "8bit", "gptq", "awq", "dwq"]


def load_model_from_path(
    model_path: Union[str, Path],
    lazy: bool = False,
) -> tuple[Model, ModelArgs]:
    """
    Load model from a local path containing weights and config.

    Args:
        model_path: Path to directory containing model files
            - config.json: Model configuration
            - weights.npz or *.safetensors: Model weights
        lazy: If True, use lazy loading for weights

    Returns:
        Tuple of (model, config)

    Example:
        >>> model, config = load_model_from_path("./models/smollm2-135m")
    """
    model_path = Path(model_path)

    # Load configuration
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    config = load_config(config_dict)
    validate_config(config)

    # Create model
    model = Model(config)

    # Load weights using utils
    weights = utils_load_weights(model_path, lazy=lazy)

    # Sanitize weights (remove unused keys)
    weights = model.sanitize(weights)

    # Load weights into model
    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())

    return model, config


def load_tokenizer_from_path(
    model_path: Union[str, Path],
):
    """
    Load tokenizer from a local path.

    Args:
        model_path: Path to directory containing tokenizer files

    Returns:
        Loaded tokenizer (transformers.PreTrainedTokenizer)

    Example:
        >>> tokenizer = load_tokenizer_from_path("./models/smollm2-135m")
    """
    # Use utils function
    return utils_load_tokenizer(model_path)


def _apply_quantization(
    model: Model,
    quantize: QuantizePreset,
    quantization_config: Optional[dict] = None,
) -> Model:
    """
    Apply quantization to a loaded model.

    Args:
        model: Model to quantize
        quantize: Quantization preset
        quantization_config: Optional configuration dict

    Returns:
        Quantized model

    Raises:
        ValueError: If quantization preset is not recognized
    """
    # Import quantization functions lazily
    from smlx.quant import (
        autoquant,
        awq_quantize,
        dwq_quantize_simple,
        gptq_quantize,
        quantize_4bit,
        quantize_8bit,
    )

    # Default configuration
    default_config = {"bits": 4, "group_size": 64}
    config = {**default_config, **(quantization_config or {})}

    if quantize == "auto":
        # Use automatic quantization selection
        print("Selecting optimal quantization strategy...")
        quantized_model = autoquant(model)
        print("Auto-quantization complete!")
        return quantized_model

    elif quantize == "4bit":
        # Standard 4-bit quantization
        print("Applying 4-bit quantization...")
        quantize_4bit(model)  # In-place quantization
        print("4-bit quantization complete!")
        return model

    elif quantize == "8bit":
        # Standard 8-bit quantization
        print("Applying 8-bit quantization...")
        quantize_8bit(model)  # In-place quantization
        print("8-bit quantization complete!")
        return model

    elif quantize == "gptq":
        # GPTQ quantization
        print(f"Applying GPTQ quantization ({config['bits']}-bit, group_size={config['group_size']})...")
        quantized_model = gptq_quantize(
            model=model,
            bits=config["bits"],
            group_size=config["group_size"],
        )
        print("GPTQ quantization complete!")
        return quantized_model

    elif quantize == "awq":
        # AWQ quantization
        print(f"Applying AWQ quantization ({config['bits']}-bit, group_size={config['group_size']})...")
        quantized_model = awq_quantize(
            model=model,
            bits=config["bits"],
            group_size=config["group_size"],
        )
        print("AWQ quantization complete!")
        return quantized_model

    elif quantize == "dwq":
        # DWQ quantization
        print(f"Applying DWQ quantization ({config['bits']}-bit, group_size={config['group_size']})...")
        quantized_model = dwq_quantize_simple(
            model=model,
            bits=config["bits"],
            group_size=config["group_size"],
        )
        print("DWQ quantization complete!")
        return quantized_model

    else:
        raise ValueError(
            f"Unknown quantization preset: {quantize}. "
            f"Valid options: 'auto', '4bit', '8bit', 'gptq', 'awq', 'dwq'"
        )


def load(
    model_path: Union[str, Path] = "mlx-community/SmolLM2-135M-Instruct",
    lazy: bool = False,
    quantize: Optional[QuantizePreset] = None,
    quantization_config: Optional[dict] = None,
):
    """
    Load SmolLM2-135M model and tokenizer with optional quantization.

    This is the main entry point for loading the model. It handles both:
    1. Loading from local path
    2. Downloading from HuggingFace Hub
    3. Optional quantization for memory efficiency

    Args:
        model_path: Either:
            - Local path to model directory
            - HuggingFace model ID (e.g., "mlx-community/SmolLM2-135M-Instruct")
        lazy: If True, use lazy loading for weights
        quantize: Quantization preset:
            - None: No quantization (default)
            - "auto": Automatic quantization selection based on hardware
            - "4bit": Standard 4-bit quantization
            - "8bit": Standard 8-bit quantization
            - "gptq": GPTQ 4-bit quantization (high quality)
            - "awq": AWQ 4-bit quantization (activation-aware)
            - "dwq": DWQ quantization with knowledge distillation
        quantization_config: Advanced quantization configuration dict:
            - bits: Bits per weight (default: 4 for 4bit/gptq/awq, 8 for 8bit)
            - group_size: Quantization group size (default: 64)
            - Additional method-specific parameters

    Returns:
        Tuple of (model, tokenizer)

    Examples:
        >>> # Load without quantization
        >>> model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        >>>
        >>> # Load with 4-bit quantization
        >>> model, tokenizer = load(
        ...     "mlx-community/SmolLM2-135M-Instruct",
        ...     quantize="4bit"
        ... )
        >>>
        >>> # Load with GPTQ quantization and custom config
        >>> model, tokenizer = load(
        ...     "mlx-community/SmolLM2-135M-Instruct",
        ...     quantize="gptq",
        ...     quantization_config={"bits": 4, "group_size": 64}
        ... )
        >>>
        >>> # Automatic quantization selection
        >>> model, tokenizer = load(
        ...     "mlx-community/SmolLM2-135M-Instruct",
        ...     quantize="auto"
        ... )
    """
    # Use utils to resolve model path (handles both local and HF Hub)
    local_path = resolve_model_path(model_path)

    # Load model and tokenizer
    model, config = load_model_from_path(local_path, lazy=lazy)
    tokenizer = load_tokenizer_from_path(local_path)

    # Apply quantization if requested
    if quantize is not None:
        model = _apply_quantization(model, quantize, quantization_config)

    return model, tokenizer


def load_weights_only(
    model_path: Union[str, Path],
) -> dict[str, mx.array]:
    """
    Load only the weights without initializing the model.

    Useful for inspection or custom initialization.

    Args:
        model_path: Path to weights file or directory

    Returns:
        Dictionary of weights

    Example:
        >>> weights = load_weights_only("./models/smollm2-135m/weights.safetensors")
        >>> print(weights.keys())
    """
    # Use utils function
    return utils_load_weights(model_path)


def save_model(
    model: Model,
    save_path: Union[str, Path],
    config: Optional[ModelArgs] = None,
):
    """
    Save model weights and configuration to disk.

    Args:
        model: Model to save
        save_path: Directory to save to
        config: Optional config to save (uses model.args if not provided)

    Example:
        >>> model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        >>> save_model(model, "./my_model")
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save weights using utils (handles sharding automatically)
    weights = model.parameters()
    utils_save_weights(weights, save_path / "model.safetensors")

    # Save config
    if config is None:
        config = model.args

    config_path = save_path / "config.json"
    config_dict = {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "rope_traditional": config.rope_traditional,
        "rope_scaling": config.rope_scaling,
        "attention_bias": config.attention_bias,
        "mlp_bias": config.mlp_bias,
        "tie_word_embeddings": config.tie_word_embeddings,
        "no_rope_layer_interval": config.no_rope_layer_interval,
        "no_rope_layers": config.no_rope_layers,
    }

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Model saved to {save_path}")
