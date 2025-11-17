#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Configuration classes for SmolVLM-256M-Instruct.

SmolVLM combines:
- SigLIP-SO400M vision encoder (1152 hidden size, 16 heads, 27 layers)
- SmolLM2-135M language model (576 hidden size, 9 heads, 30 layers)
- Idefics3 connector with pixel shuffle (scale=2)
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
    hidden_size: int = 1152  # SigLIP-SO400M
    num_attention_heads: int = 16
    patch_size: int = 14
    num_hidden_layers: int = 27
    intermediate_size: int = 4304
    image_size: int = 384
    num_channels: int = 3
    layer_norm_eps: float = 1e-6


@dataclass
class TextConfig(BaseModelConfig):
    """Configuration for SmolLM2-135M language model."""

    model_type: str = "smolvlm"
    hidden_size: int = 576  # SmolLM2-135M
    intermediate_size: int = 1536
    num_attention_heads: int = 9
    rms_norm_eps: float = 1e-5
    vocab_size: int = 49152
    num_key_value_heads: int = 3  # Grouped Query Attention (GQA)
    rope_theta: float = 1000000.0
    num_hidden_layers: int = 30
    rope_traditional: bool = False
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class ModelConfig(BaseModelConfig):
    """Configuration for complete SmolVLM-256M model."""

    text_config: TextConfig = field(default_factory=lambda: TextConfig())
    vision_config: VisionConfig = field(default_factory=lambda: VisionConfig())
    model_type: str = "smolvlm"
    ignore_index: int = -100
    vocab_size: int = 49152
    scale_factor: int = 2  # Pixel shuffle scale
    image_token_id: int = 49153  # Special <image> token
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


# Default configuration for SmolVLM-256M-Instruct
DEFAULT_CONFIG = ModelConfig(
    text_config=TextConfig(
        model_type="smolvlm",
        hidden_size=576,
        intermediate_size=1536,
        num_attention_heads=9,
        rms_norm_eps=1e-5,
        vocab_size=49152,
        num_key_value_heads=3,
        rope_theta=1000000.0,
        num_hidden_layers=30,
        rope_traditional=False,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    ),
    vision_config=VisionConfig(
        model_type="siglip_vision_model",
        hidden_size=1152,
        num_attention_heads=16,
        patch_size=14,
        num_hidden_layers=27,
        intermediate_size=4304,
        image_size=384,
        num_channels=3,
        layer_norm_eps=1e-6,
    ),
    model_type="smolvlm",
    scale_factor=2,
    image_token_id=49153,
    vocab_size=49152,
)


__all__ = ["ModelConfig", "TextConfig", "VisionConfig", "DEFAULT_CONFIG"]
