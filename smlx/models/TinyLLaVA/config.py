#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Configuration for TinyLLaVA Vision-Language Model.

TinyLLaVA uses SigLIP vision encoder with TinyLlama/Phi-2/StableLM-2
language models, connected via a simple MLP projector.
"""

from dataclasses import dataclass, field
from typing import Optional

from smlx.utils.config import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    """Configuration for SigLIP vision encoder.

    Uses the same SigLIP-so400m architecture as SmolVLM.
    """

    model_type: str = "siglip_vision_model"
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    image_size: int = 384
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "gelu_pytorch_tanh"


@dataclass
class TextConfig(BaseModelConfig):
    """Configuration for TinyLlama language model.

    TinyLlama is a 1.1B LLaMA-based model optimized for efficiency.
    """

    model_type: str = "llama"
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4  # Grouped Query Attention

    # Positional embeddings
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[dict] = None

    # Normalization
    rms_norm_eps: float = 1e-5

    # Activation
    hidden_act: str = "silu"

    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0


@dataclass
class ProjectorConfig(BaseModelConfig):
    """Configuration for vision-language projector.

    Simple MLP projector that maps vision features to language space.
    Optionally supports Perceiver Resampler with cross-attention.
    """

    projector_type: str = "mlp2x_gelu"  # 2-layer MLP with GELU
    projector_hidden_act: str = "gelu"

    # Optional resampler config (for advanced variants)
    use_resampler: bool = False
    num_query_tokens: int = 128
    resampler_n_layers: int = 3
    resampler_hidden_size: int = 768

    # Resampler attention parameters
    resampler_n_heads: int = 16  # Number of attention heads
    resampler_head_dim: int = 96  # Dimension per head
    num_key_value_heads: int = 4  # For grouped query attention (GQA)
    rms_norm_eps: float = 1e-6  # Layer norm epsilon


@dataclass
class ModelConfig(BaseModelConfig):
    """Complete TinyLLaVA model configuration."""

    model_type: str = "tinyllava"

    # Sub-configurations
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    text_config: TextConfig = field(default_factory=TextConfig)
    projector_config: ProjectorConfig = field(default_factory=ProjectorConfig)

    # Vision feature selection
    vision_feature_layer: int = -2  # Use second-to-last layer
    vision_feature_select_strategy: str = "patch"  # Or "cls"

    # Image processing
    image_aspect_ratio: str = "pad"  # or "square", "anyres"

    # Image token
    image_token_index: int = -200

    # Model variant
    variant: str = "1.5b"  # "1.5b", "2.0b", "3.1b"

    def __init__(self, **kwargs):
        """Initialize ModelConfig, supporting both projector_config and connector_config."""
        # Handle connector_config alias
        if "connector_config" in kwargs and "projector_config" not in kwargs:
            kwargs["projector_config"] = kwargs.pop("connector_config")
        elif "connector_config" in kwargs:
            # Remove connector_config if projector_config is also present
            kwargs.pop("connector_config")

        # Call parent __init__
        self.__dict__.update(kwargs)
        # Set defaults for missing fields
        if not hasattr(self, "model_type"):
            self.model_type = "tinyllava"
        if not hasattr(self, "vision_config"):
            self.vision_config = VisionConfig()
        if not hasattr(self, "text_config"):
            self.text_config = TextConfig()
        if not hasattr(self, "projector_config"):
            self.projector_config = ProjectorConfig()
        if not hasattr(self, "vision_feature_layer"):
            self.vision_feature_layer = -2
        if not hasattr(self, "vision_feature_select_strategy"):
            self.vision_feature_select_strategy = "patch"
        if not hasattr(self, "image_aspect_ratio"):
            self.image_aspect_ratio = "pad"
        if not hasattr(self, "image_token_index"):
            self.image_token_index = -200
        if not hasattr(self, "variant"):
            self.variant = "1.5b"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create config from dictionary (HuggingFace format)."""
        # Extract sub-configs
        vision_dict = config_dict.pop("vision_config", {})
        text_dict = config_dict.pop("text_config", {})
        # Support both projector_config and connector_config for backwards compatibility
        projector_dict = config_dict.pop("projector_config", config_dict.pop("connector_config", {}))

        # Create sub-configs
        vision_config = VisionConfig(**vision_dict) if vision_dict else VisionConfig()
        text_config = TextConfig(**text_dict) if text_dict else TextConfig()
        projector_config = (
            ProjectorConfig(**projector_dict) if projector_dict else ProjectorConfig()
        )

        # Create main config
        return cls(
            vision_config=vision_config,
            text_config=text_config,
            projector_config=projector_config,
            **config_dict,
        )


# Default configurations for different variants
DEFAULT_CONFIG_1_5B = ModelConfig(
    variant="1.5b",
    vision_config=VisionConfig(),
    text_config=TextConfig(
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=4,
    ),
    projector_config=ProjectorConfig(),
)

DEFAULT_CONFIG_2_0B = ModelConfig(
    variant="2.0b",
    vision_config=VisionConfig(),
    text_config=TextConfig(
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=4,
        vocab_size=100352,  # StableLM-2 vocab
    ),
    projector_config=ProjectorConfig(),
)

DEFAULT_CONFIG_3_1B = ModelConfig(
    variant="3.1b",
    vision_config=VisionConfig(),
    text_config=TextConfig(
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,  # Phi-2 doesn't use GQA
        vocab_size=51200,  # Phi-2 vocab
        hidden_act="gelu_new",
    ),
    projector_config=ProjectorConfig(),
)


# Alias for backward compatibility and naming consistency with documentation
ConnectorConfig = ProjectorConfig


__all__ = [
    "VisionConfig",
    "TextConfig",
    "ProjectorConfig",
    "ConnectorConfig",
    "ModelConfig",
    "DEFAULT_CONFIG_1_5B",
    "DEFAULT_CONFIG_2_0B",
    "DEFAULT_CONFIG_3_1B",
]
