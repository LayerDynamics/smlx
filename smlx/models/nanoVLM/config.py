#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
nanoVLM Configuration.

Defines configuration for the minimal 222M parameter vision-language model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionConfig:
    """SigLIP vision encoder configuration."""

    model_type: str = "siglip_vision_model"
    hidden_size: int = 768  # SigLIP-base
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224  # nanoVLM uses smaller 224x224
    patch_size: int = 16
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.image_size % self.patch_size == 0


@dataclass
class LanguageConfig:
    """SmolLM2-135M language model configuration."""

    model_type: str = "smollm2"
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3  # GQA
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = True
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: int = 0
    attention_bias: bool = False
    mlp_bias: bool = False
    layer_types: Optional[list] = None  # Layer types for SmolLM2 compatibility
    sliding_window: Optional[int] = None  # Optional sliding window size
    head_dim: Optional[int] = None  # Head dimension (computed if None)
    no_rope_layer_interval: int = 4  # Interval for NoPE
    no_rope_layers: Optional[list] = None  # Binary list for NoPE

    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0


@dataclass
class ProjectionConfig:
    """Vision-to-language projection configuration."""

    vision_hidden_size: int = 768  # SigLIP output
    language_hidden_size: int = 576  # SmolLM2 input
    num_layers: int = 2  # 2-layer MLP
    activation: str = "gelu"

    def __post_init__(self):
        """Validate configuration."""
        assert self.num_layers >= 1


@dataclass
class NanoVLMConfig:
    """Complete nanoVLM model configuration."""

    # Model type
    model_type: str = "nanovlm"

    # Component configs
    vision_config: VisionConfig = None
    language_config: LanguageConfig = None
    projection_config: ProjectionConfig = None

    # Architecture
    num_image_tokens: int = 196  # 14x14 patches for 224x224 image

    # Generation defaults
    max_length: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0

    # Special tokens
    image_token_id: int = 49150  # Special token for image

    def __post_init__(self):
        """Initialize sub-configs if not provided."""
        if self.vision_config is None:
            self.vision_config = VisionConfig()
        if self.language_config is None:
            self.language_config = LanguageConfig()
        if self.projection_config is None:
            self.projection_config = ProjectionConfig(
                vision_hidden_size=self.vision_config.hidden_size,
                language_hidden_size=self.language_config.hidden_size,
            )

        # Validate image tokens
        num_patches = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        assert self.num_image_tokens == num_patches, (
            f"num_image_tokens ({self.num_image_tokens}) must match "
            f"number of patches ({num_patches})"
        )


# Default configuration for nanoVLM-222M
DEFAULT_CONFIG = NanoVLMConfig(
    vision_config=VisionConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=224,
        patch_size=16,
    ),
    language_config=LanguageConfig(
        vocab_size=49152,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
    ),
    projection_config=ProjectionConfig(
        vision_hidden_size=768,
        language_hidden_size=576,
        num_layers=2,
    ),
    num_image_tokens=196,
)


def load_config(model_path: str) -> NanoVLMConfig:
    """
    Load configuration from model directory.

    Args:
        model_path: Path to model directory

    Returns:
        NanoVLMConfig instance
    """
    import json
    from pathlib import Path

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return DEFAULT_CONFIG

    with open(config_path) as f:
        config_dict = json.load(f)

    # Parse vision config
    vision_dict = config_dict.get("vision_config", {})
    vision_config = VisionConfig(**vision_dict)

    # Parse language config
    language_dict = config_dict.get("language_config", {})
    language_config = LanguageConfig(**language_dict)

    # Parse projection config
    projection_dict = config_dict.get("projection_config", {})
    projection_config = ProjectionConfig(**projection_dict)

    # Create main config
    config = NanoVLMConfig(
        vision_config=vision_config,
        language_config=language_config,
        projection_config=projection_config,
        num_image_tokens=config_dict.get("num_image_tokens", 196),
        max_length=config_dict.get("max_length", 2048),
        temperature=config_dict.get("temperature", 1.0),
        top_p=config_dict.get("top_p", 0.95),
        top_k=config_dict.get("top_k", 50),
        repetition_penalty=config_dict.get("repetition_penalty", 1.0),
        image_token_id=config_dict.get("image_token_id", 49150),
    )

    return config


def save_config(config: NanoVLMConfig, model_path: str):
    """
    Save configuration to model directory.

    Args:
        config: NanoVLMConfig instance
        model_path: Path to model directory
    """
    import json
    from pathlib import Path

    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "model_type": config.model_type,
        "vision_config": config.vision_config.__dict__,
        "language_config": config.language_config.__dict__,
        "projection_config": config.projection_config.__dict__,
        "num_image_tokens": config.num_image_tokens,
        "max_length": config.max_length,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": config.repetition_penalty,
        "image_token_id": config.image_token_id,
    }

    config_path = model_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
