# Copyright © 2025 SMLX Project

"""
Configuration for SmolLM2-360M-Instruct model.

Provides default configurations and utilities for loading from HuggingFace.
Uses shared utils from smlx.utils.config.
"""

from typing import Any

from smlx.utils.config import (
    load_config as utils_load_config,
    print_config as utils_print_config,
    validate_config as utils_validate_config,
)

from .model import ModelArgs

# Default configuration for SmolLM2-360M-Instruct
# Based on HuggingFace model card: mlx-community/SmolLM2-360M-Instruct
DEFAULT_CONFIG = {
    "model_type": "smollm",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 960,
    "num_hidden_layers": 32,
    "intermediate_size": 2560,
    "num_attention_heads": 15,
    "num_key_value_heads": 5,
    "vocab_size": 49152,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "rope_traditional": False,
    "rope_scaling": None,
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": True,
    "head_dim": None,  # Will be computed as hidden_size // num_attention_heads
    "layer_types": None,  # Will default to all full_attention
    "sliding_window": None,
    # SmolLM3-specific NoPE (No Positional Encoding)
    "no_rope_layer_interval": 4,
    "no_rope_layers": None,  # Will be auto-generated based on interval
}


def load_config(config_dict: dict[str, Any]) -> ModelArgs:
    """
    Load model configuration from a dictionary (e.g., from config.json).

    This handles HuggingFace config format and converts it to ModelArgs.

    Args:
        config_dict: Configuration dictionary from HuggingFace or custom source

    Returns:
        ModelArgs instance

    Example:
        >>> import json
        >>> with open("config.json") as f:
        ...     config_dict = json.load(f)
        >>> args = load_config(config_dict)
    """
    # Start with defaults
    config = DEFAULT_CONFIG.copy()

    # Override with provided values
    config.update(config_dict)

    # Use utils function to load config
    return utils_load_config(config_dict=config, config_class=ModelArgs)


def get_default_config() -> ModelArgs:
    """
    Get default configuration for SmolLM2-360M-Instruct.

    Returns:
        ModelArgs with default configuration

    Example:
        >>> args = get_default_config()
        >>> print(f"Model has {args.num_hidden_layers} layers")
        Model has 32 layers
    """
    return ModelArgs.from_dict(DEFAULT_CONFIG)


def validate_config(args: ModelArgs) -> None:
    """
    Validate model configuration for common issues.

    Args:
        args: ModelArgs to validate

    Raises:
        ValueError: If configuration is invalid

    Example:
        >>> args = get_default_config()
        >>> validate_config(args)  # Should pass without errors
    """
    # Use utils function for basic validation
    utils_validate_config(args)

    # SmolLM2-specific validations
    # Check NoPE configuration
    if args.no_rope_layers is not None:
        assert len(args.no_rope_layers) == args.num_hidden_layers, (
            f"no_rope_layers length ({len(args.no_rope_layers)}) "
            f"must match num_hidden_layers ({args.num_hidden_layers})"
        )

    # Check layer types
    if args.layer_types is not None:
        assert len(args.layer_types) == args.num_hidden_layers, (
            f"layer_types length ({len(args.layer_types)}) "
            f"must match num_hidden_layers ({args.num_hidden_layers})"
        )


def print_config(args: ModelArgs) -> None:
    """
    Pretty-print model configuration.

    Args:
        args: ModelArgs to print

    Example:
        >>> args = get_default_config()
        >>> print_config(args)
        SmolLM2-360M Configuration:
        ===========================
        Model Type: smollm
        Hidden Size: 960
        ...
    """
    # Use utils function for printing
    utils_print_config(args, model_name="SmolLM2-360M")
