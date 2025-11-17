#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Configuration classes for SmolVLM-500M-Instruct.

SmolVLM combines:
- SigLIP 93M vision encoder (768 hidden size, 12 heads, 12 layers)
- SmolLM2-360M language model (960 hidden size, 15 heads, 32 layers)
- Idefics3 connector with pixel shuffle (scale=4)
"""

import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BaseModelConfig:
    """Base configuration class providing from_dict() method."""

    @classmethod
    def from_dict(cls, params: Dict):
        """Create config from dictionary, filtering unknown params."""
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class VisionConfig(BaseModelConfig):
    """Configuration for SigLIP vision encoder."""

    model_type: str = "siglip_vision_model"
    hidden_size: int = 768  # SigLIP 93M
    num_attention_heads: int = 12
    patch_size: int = 16
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    image_size: int = 512
    num_channels: int = 3
    layer_norm_eps: float = 1e-6


@dataclass
class TextConfig(BaseModelConfig):
    """Configuration for SmolLM2-360M language model."""

    model_type: str = "smolvlm"
    hidden_size: int = 960  # SmolLM2-360M
    intermediate_size: int = 2560
    num_attention_heads: int = 15
    rms_norm_eps: float = 1e-5
    vocab_size: int = 49280
    num_key_value_heads: int = 5  # Grouped Query Attention (GQA)
    rope_theta: float = 100000.0
    num_hidden_layers: int = 32
    rope_traditional: bool = False
    max_position_embeddings: int = 8192
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class ModelConfig(BaseModelConfig):
    """Configuration for complete SmolVLM-500M model."""

    text_config: TextConfig = field(default_factory=lambda: TextConfig())
    vision_config: VisionConfig = field(default_factory=lambda: VisionConfig())
    model_type: str = "smolvlm"
    ignore_index: int = -100
    vocab_size: int = 49280
    scale_factor: int = 4  # Pixel shuffle scale
    image_token_id: int = 49190  # Special <image> token
    image_token_index: Optional[int] = None
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id

        # Handle nested configs from dict loading
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)


# Default configuration for SmolVLM-500M-Instruct
DEFAULT_CONFIG = ModelConfig(
    text_config=TextConfig(
        model_type="smolvlm",
        hidden_size=960,
        intermediate_size=2560,
        num_attention_heads=15,
        rms_norm_eps=1e-5,
        vocab_size=49280,
        num_key_value_heads=5,
        rope_theta=100000.0,
        num_hidden_layers=32,
        rope_traditional=False,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
    ),
    vision_config=VisionConfig(
        model_type="siglip_vision_model",
        hidden_size=768,
        num_attention_heads=12,
        patch_size=16,
        num_hidden_layers=12,
        intermediate_size=3072,
        image_size=512,
        num_channels=3,
        layer_norm_eps=1e-6,
    ),
    model_type="smolvlm",
    scale_factor=4,
    image_token_id=49190,
    vocab_size=49280,
)


__all__ = ["ModelConfig", "TextConfig", "VisionConfig", "DEFAULT_CONFIG"]
