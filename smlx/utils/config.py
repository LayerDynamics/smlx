# Copyright © 2025 SMLX Project

"""
Configuration utilities for all SMLX models.

Provides base classes and utilities for loading, validating, and managing
model configurations across language, vision-language, audio, and embedding models.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T", bound="BaseModelArgs")


@dataclass
class BaseModelArgs:
    """
    Base class for model configuration arguments.

    All model-specific configuration classes should inherit from this class.
    Provides utilities for loading from dictionaries and converting to dictionaries.

    Example:
        >>> @dataclass
        ... class MyModelArgs(BaseModelArgs):
        ...     hidden_size: int = 768
        ...     num_layers: int = 12
        ...
        >>> config = {"hidden_size": 512, "num_layers": 6, "unknown": "ignored"}
        >>> args = MyModelArgs.from_dict(config)
        >>> print(args.hidden_size)
        512
    """

    @classmethod
    def from_dict(cls: type[T], params: dict[str, Any]) -> T:
        """
        Create model args from a dictionary, filtering to valid parameters.

        This automatically filters out any keys that are not valid parameters
        for the dataclass, making it safe to load from HuggingFace config.json
        files that may contain extra metadata.

        Args:
            params: Configuration dictionary (e.g., from config.json)

        Returns:
            Instance of the model args class

        Example:
            >>> args = ModelArgs.from_dict({"hidden_size": 768, "extra_key": "ignored"})
        """
        return cls(
            **{k: v for k, v in params.items() if k in inspect.signature(cls).parameters}
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert model args to a dictionary.

        Returns:
            Dictionary representation of configuration

        Example:
            >>> args = ModelArgs(hidden_size=768)
            >>> config_dict = args.to_dict()
        """
        from dataclasses import asdict

        return asdict(self)

    def save(self, path: Path | str) -> None:
        """
        Save configuration to a JSON file.

        Args:
            path: Path to save config.json

        Example:
            >>> args.save("config.json")
        """
        path = Path(path)
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls: type[T], path: Path | str) -> T:
        """
        Load configuration from a JSON file.

        Args:
            path: Path to config.json

        Returns:
            Instance of the model args class

        Example:
            >>> args = ModelArgs.load("config.json")
        """
        path = Path(path)
        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def load_config(
    config_path: Path | str | None = None,
    config_dict: dict[str, Any] | None = None,
    config_class: type[T] = BaseModelArgs,
    default_config: dict[str, Any] | None = None,
) -> T:
    """
    Load model configuration from various sources.

    Can load from:
    1. A file path (config.json)
    2. A dictionary
    3. A combination of default config + overrides

    Args:
        config_path: Path to config.json file
        config_dict: Configuration dictionary
        config_class: Configuration class to instantiate
        default_config: Default configuration to use as base

    Returns:
        Instance of config_class with loaded configuration

    Raises:
        ValueError: If neither config_path nor config_dict is provided

    Example:
        >>> args = load_config(config_path="model/config.json", config_class=ModelArgs)
        >>> args = load_config(config_dict={"hidden_size": 768}, config_class=ModelArgs)
    """
    if config_path is not None:
        path = Path(config_path)
        with open(path) as f:
            config = json.load(f)
    elif config_dict is not None:
        config = config_dict.copy()
    else:
        raise ValueError("Either config_path or config_dict must be provided")

    # Merge with defaults if provided
    if default_config is not None:
        merged_config = default_config.copy()
        merged_config.update(config)
        config = merged_config

    return config_class.from_dict(config)


def merge_configs(
    base_config: dict[str, Any], override_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge two configuration dictionaries with override priority.

    Args:
        base_config: Base configuration
        override_config: Override configuration (takes priority)

    Returns:
        Merged configuration dictionary

    Example:
        >>> base = {"hidden_size": 768, "num_layers": 12}
        >>> override = {"hidden_size": 512}
        >>> merged = merge_configs(base, override)
        >>> print(merged["hidden_size"])
        512
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged


def validate_config(config: BaseModelArgs, strict: bool = True) -> None:
    """
    Validate model configuration for common issues.

    Performs general validation that applies to most transformer models.
    Model-specific validation should be done in the model's config module.

    Args:
        config: Configuration to validate
        strict: If True, raise ValueError on issues. If False, only warn.

    Raises:
        ValueError: If configuration is invalid and strict=True

    Example:
        >>> validate_config(args)  # Raises ValueError if invalid
        >>> validate_config(args, strict=False)  # Only warns
    """
    # Common validations for transformer models
    if hasattr(config, "hidden_size") and config.hidden_size <= 0:
        msg = "hidden_size must be positive"
        if strict:
            raise ValueError(msg)
        print(f"Warning: {msg}")

    if hasattr(config, "num_hidden_layers") and config.num_hidden_layers <= 0:
        msg = "num_hidden_layers must be positive"
        if strict:
            raise ValueError(msg)
        print(f"Warning: {msg}")

    if hasattr(config, "vocab_size") and config.vocab_size <= 0:
        msg = "vocab_size must be positive"
        if strict:
            raise ValueError(msg)
        print(f"Warning: {msg}")

    # Attention head validation (if applicable)
    if hasattr(config, "num_attention_heads") and hasattr(config, "num_key_value_heads"):
        if config.num_attention_heads < config.num_key_value_heads:
            msg = "num_attention_heads must be >= num_key_value_heads"
            if strict:
                raise ValueError(msg)
            print(f"Warning: {msg}")

        if config.num_attention_heads % config.num_key_value_heads != 0:
            msg = "num_attention_heads must be divisible by num_key_value_heads"
            if strict:
                raise ValueError(msg)
            print(f"Warning: {msg}")

    # Hidden size divisibility (if applicable)
    if (
        hasattr(config, "hidden_size")
        and hasattr(config, "num_attention_heads")
        and hasattr(config, "head_dim")
    ):
        if config.head_dim is None:
            if config.hidden_size % config.num_attention_heads != 0:
                msg = (
                    "hidden_size must be divisible by num_attention_heads "
                    "when head_dim is None"
                )
                if strict:
                    raise ValueError(msg)
                print(f"Warning: {msg}")


def print_config(
    config: BaseModelArgs,
    model_name: str = "Model",
    estimate_params: bool = True,
    verbose: bool = True,
) -> None:
    """
    Pretty-print model configuration with optional parameter estimation.

    Args:
        config: Configuration to print
        model_name: Name of the model for display
        estimate_params: Whether to estimate parameter count
        verbose: If True, print all attributes. If False, only key attributes.

    Example:
        >>> print_config(args, model_name="SmolLM2-135M")
        SmolLM2-135M Configuration:
        ====================================
        Model Type: smollm
        Hidden Size: 576
        ...
        Estimated Parameters: 135.0M
    """
    title = f"{model_name} Configuration:"
    print(title)
    print("=" * len(title))

    # Print all config attributes
    config_dict = config.to_dict()

    # Key attributes to always show (if present)
    key_attributes = [
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
    ]

    if verbose:
        # Show all attributes
        for key, value in config_dict.items():
            print(f"{key}: {value}")
    else:
        # Show only key attributes
        for key in key_attributes:
            if key in config_dict:
                print(f"{key}: {config_dict[key]}")

    print("=" * len(title))

    # Estimate parameters if requested
    if estimate_params:
        params = estimate_parameters(config)
        if params > 0:
            print(f"\nEstimated Parameters: {params / 1e6:.1f}M")


def estimate_parameters(config: BaseModelArgs) -> int:
    """
    Estimate model parameter count from configuration.

    This provides a rough estimate for transformer-based models. The actual
    parameter count may differ based on implementation details.

    Args:
        config: Model configuration

    Returns:
        Estimated number of parameters

    Example:
        >>> params = estimate_parameters(args)
        >>> print(f"Model has approximately {params / 1e6:.1f}M parameters")
    """
    total_params = 0

    # Check if this is a transformer model
    if not hasattr(config, "hidden_size") or not hasattr(config, "num_hidden_layers"):
        return 0  # Can't estimate

    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers

    # Embedding parameters
    if hasattr(config, "vocab_size"):
        embed_params = config.vocab_size * hidden_size
        total_params += embed_params

    # Determine head dimension
    head_dim = None
    if hasattr(config, "head_dim") and config.head_dim is not None:
        head_dim = config.head_dim
    elif hasattr(config, "num_attention_heads"):
        head_dim = hidden_size // config.num_attention_heads

    # Per-layer parameters
    if head_dim is not None and hasattr(config, "num_attention_heads"):
        num_q_heads = config.num_attention_heads
        num_kv_heads = (
            config.num_key_value_heads
            if hasattr(config, "num_key_value_heads")
            else num_q_heads
        )

        # Attention parameters
        attn_params = (
            hidden_size * num_q_heads * head_dim  # Q projection
            + hidden_size * num_kv_heads * head_dim  # K projection
            + hidden_size * num_kv_heads * head_dim  # V projection
            + num_q_heads * head_dim * hidden_size  # O projection
        )

        # MLP parameters
        intermediate_size = (
            config.intermediate_size
            if hasattr(config, "intermediate_size")
            else hidden_size * 4
        )
        mlp_params = (
            hidden_size * intermediate_size  # Gate/Up
            + hidden_size * intermediate_size  # Up (SwiGLU has two)
            + intermediate_size * hidden_size  # Down
        )

        # Normalization parameters (typically 2 LayerNorms per layer)
        norm_params = hidden_size * 2

        layer_params = attn_params + mlp_params + norm_params
        total_params += layer_params * num_layers

    # Output head (LM head)
    if hasattr(config, "vocab_size") and hasattr(config, "tie_word_embeddings"):
        if not config.tie_word_embeddings:
            total_params += config.vocab_size * hidden_size

    return total_params


# Common configuration defaults for different model types

LANGUAGE_MODEL_DEFAULTS = {
    "model_type": "transformer",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "vocab_size": 50257,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": True,
}

VISION_LANGUAGE_MODEL_DEFAULTS = {
    "model_type": "vlm",
    "text_config": LANGUAGE_MODEL_DEFAULTS.copy(),
    "vision_config": {
        "model_type": "vision_transformer",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "image_size": 224,
        "patch_size": 16,
    },
}

AUDIO_MODEL_DEFAULTS = {
    "model_type": "audio",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_mel_bins": 80,
    "max_source_positions": 1500,
}


# Alias for backwards compatibility and alternative naming convention
BaseModelConfig = BaseModelArgs


__all__ = [
    "BaseModelArgs",
    "BaseModelConfig",
    "load_config",
    "merge_configs",
    "validate_config",
    "print_config",
    "estimate_parameters",
    "LANGUAGE_MODEL_DEFAULTS",
    "VISION_LANGUAGE_MODEL_DEFAULTS",
    "AUDIO_MODEL_DEFAULTS",
]
