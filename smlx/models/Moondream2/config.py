#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Configuration for Moondream2 Vision-Language Model.

Moondream2 uses a Phi-based language model with a custom vision encoder
that supports crop-based tiling for efficient high-resolution image processing.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from smlx.utils.config import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    """Configuration for Moondream2 vision encoder.

    The vision encoder uses a custom architecture with:
    - Crop-based tiling for efficient high-resolution processing
    - Global and local feature extraction
    - Multi-layer hierarchical features
    """

    model_type: str = "moondream_vision"
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    image_size: int = 378
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-6

    # Crop-based tiling settings
    use_tiling: bool = True
    max_crops: int = 4
    crop_overlap: int = 14  # Overlap between crops for reconstruction


@dataclass
class TextConfig(BaseModelConfig):
    """Configuration for Phi language model used in Moondream2.

    Phi is a Microsoft-developed transformer model optimized for
    efficient reasoning and instruction following.
    """

    model_type: str = "phi"
    vocab_size: int = 51200
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # No GQA in base Phi

    # Positional embeddings
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    partial_rotary_factor: float = 0.5  # Phi uses partial RoPE

    # Normalization
    rms_norm_eps: float = 1e-5

    # Regularization
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0

    # Activation
    hidden_act: str = "gelu_new"  # Phi uses GELU variant

    # Special tokens
    # CRITICAL: Moondream2 uses starmie-v1 tokenizer where BOS/EOS/PAD all use token ID 0
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: int = 0

    # RoPE scaling (optional)
    rope_scaling: Optional[Dict] = None


@dataclass
class RegionConfig(BaseModelConfig):
    """Configuration for region/detection modules.

    Moondream2 supports object detection and pointing through
    specialized region encoding/decoding modules.
    """

    # Coordinate encoding
    use_fourier_features: bool = True
    fourier_feature_dim: int = 256

    # Detection settings
    max_detections: int = 100
    confidence_threshold: float = 0.5

    # Grounding tokens
    grounding_start_token: str = "<|grounding|>"
    grounding_end_token: str = "</|grounding|>"
    coordinate_token: str = "<|coordinate|>"
    size_token: str = "<|size|>"

    # Special mode tokens
    thinking_token: str = "<|thinking|>"
    answer_token: str = "<|answer|>"


@dataclass
class ModelConfig(BaseModelConfig):
    """Complete Moondream2 model configuration."""

    model_type: str = "moondream"

    # Sub-configurations
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    text_config: TextConfig = field(default_factory=TextConfig)
    region_config: RegionConfig = field(default_factory=RegionConfig)

    # Vision-language integration
    vision_feature_select_strategy: str = "multi_layer"  # Use multiple vision layers
    vision_feature_layers: list = field(default_factory=lambda: [3, 7, 15, 23, 27])

    # Image tokens
    image_token_id: int = 50000

    # Model variant
    variant: str = "2b"  # "2b" or "0.5b"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create config from dictionary (HuggingFace format)."""
        # Extract sub-configs
        vision_dict = config_dict.pop("vision_config", {})
        text_dict = config_dict.pop("text_config", {})
        region_dict = config_dict.pop("region_config", {})

        # Create sub-configs
        vision_config = VisionConfig(**vision_dict) if vision_dict else VisionConfig()
        text_config = TextConfig(**text_dict) if text_dict else TextConfig()
        region_config = RegionConfig(**region_dict) if region_dict else RegionConfig()

        # Create main config
        return cls(
            vision_config=vision_config,
            text_config=text_config,
            region_config=region_config,
            **config_dict
        )


# Default configurations for different variants
DEFAULT_CONFIG_2B = ModelConfig(
    variant="2b",
    vision_config=VisionConfig(),
    text_config=TextConfig(),
    region_config=RegionConfig(),
)

DEFAULT_CONFIG_05B = ModelConfig(
    variant="0.5b",
    vision_config=VisionConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    ),
    text_config=TextConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
    ),
    region_config=RegionConfig(),
)


__all__ = [
    "VisionConfig",
    "TextConfig",
    "RegionConfig",
    "ModelConfig",
    "DEFAULT_CONFIG_2B",
    "DEFAULT_CONFIG_05B",
]
