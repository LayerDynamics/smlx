#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Donut-base Configuration.

Configuration for OCR-free document understanding model with Swin Transformer
vision encoder and BART text decoder.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SwinConfig:
    """Swin Transformer vision encoder configuration."""

    model_type: str = "swin"
    image_size: tuple = (224, 224)  # Input image size
    patch_size: int = 4  # Patch size for initial embedding
    num_channels: int = 3  # RGB
    embed_dim: int = 128  # Initial embedding dimension
    depths: tuple = (2, 2, 18, 2)  # Depths of each stage
    num_heads: tuple = (4, 8, 16, 32)  # Attention heads per stage
    window_size: int = 7  # Window size for shifted window attention
    mlp_ratio: float = 4.0  # MLP hidden dim = embed_dim * mlp_ratio
    qkv_bias: bool = True  # Bias in attention QKV projections
    drop_rate: float = 0.0  # Dropout rate
    attn_drop_rate: float = 0.0  # Attention dropout rate
    drop_path_rate: float = 0.1  # Stochastic depth rate
    layer_norm_eps: float = 1e-5  # Layer normalization epsilon

    def __post_init__(self):
        """Validate configuration."""
        assert len(self.depths) == len(self.num_heads)
        assert len(self.depths) == 4  # Swin has 4 stages


@dataclass
class BARTConfig:
    """BART decoder configuration."""

    model_type: str = "bart"
    vocab_size: int = 50265  # BART vocabulary
    max_position_embeddings: int = 1024  # Max sequence length
    encoder_layers: int = 12  # Not used (vision encoder instead)
    encoder_ffn_dim: int = 4096  # Not used
    encoder_attention_heads: int = 16  # Not used
    decoder_layers: int = 12  # Number of decoder layers
    decoder_ffn_dim: int = 4096  # FFN dimension in decoder
    decoder_attention_heads: int = 16  # Attention heads in decoder
    activation_function: str = "gelu"  # Activation function
    d_model: int = 1024  # Model dimension
    dropout: float = 0.1  # Dropout rate
    attention_dropout: float = 0.0  # Attention dropout
    activation_dropout: float = 0.0  # Activation dropout
    init_std: float = 0.02  # Initialization std
    decoder_start_token_id: int = 2  # <s> token
    pad_token_id: int = 1  # <pad> token
    bos_token_id: int = 0  # <s> token
    eos_token_id: int = 2  # </s> token
    forced_eos_token_id: int = 2  # Force EOS token
    scale_embedding: bool = False  # Whether to scale embeddings
    use_cache: bool = True  # Use KV cache for generation

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.decoder_attention_heads == 0


@dataclass
class DonutConfig:
    """
    Complete Donut model configuration.

    Combines Swin Transformer vision encoder with BART decoder for
    OCR-free document understanding.
    """

    model_type: str = "donut"

    # Component configs
    encoder_config: Optional[SwinConfig] = None
    decoder_config: Optional[BARTConfig] = None

    # Architecture
    encoder_hidden_size: int = 1024  # Swin output dimension
    decoder_hidden_size: int = 1024  # BART dimension

    # Generation defaults
    max_length: int = 512  # Max generated sequence length
    min_length: int = 1  # Min generated sequence length
    num_beams: int = 1  # Beam search beams
    length_penalty: float = 1.0  # Length penalty
    early_stopping: bool = True  # Stop when all beams finish

    def __post_init__(self):
        """Initialize sub-configs if not provided."""
        if self.encoder_config is None:
            self.encoder_config = SwinConfig()
        if self.decoder_config is None:
            self.decoder_config = BARTConfig()

        # Ensure dimensions match
        assert self.encoder_hidden_size == self.decoder_config.d_model


# Default sub-configurations
SWIN_CONFIG = SwinConfig(
    image_size=(224, 224),
    patch_size=4,
    embed_dim=128,
    depths=(2, 2, 18, 2),
    num_heads=(4, 8, 16, 32),
    window_size=7,
)

BART_CONFIG = BARTConfig(
    vocab_size=50265,
    max_position_embeddings=1024,
    decoder_layers=12,
    decoder_ffn_dim=4096,
    decoder_attention_heads=16,
    d_model=1024,
)

# Default configuration for Donut-base
DEFAULT_CONFIG = DonutConfig(
    encoder_config=SWIN_CONFIG,
    decoder_config=BART_CONFIG,
    encoder_hidden_size=1024,
    decoder_hidden_size=1024,
)


def load_config(model_path: str) -> DonutConfig:
    """
    Load configuration from model directory.

    Args:
        model_path: Path to model directory

    Returns:
        DonutConfig instance
    """
    import json
    from pathlib import Path

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return DEFAULT_CONFIG

    with open(config_path) as f:
        config_dict = json.load(f)

    # Parse encoder config (support both encoder_config and vision_config for backwards compat)
    encoder_dict = config_dict.get("encoder_config", config_dict.get("vision_config", {}))
    if "image_size" in encoder_dict and isinstance(encoder_dict["image_size"], list):
        encoder_dict["image_size"] = tuple(encoder_dict["image_size"])
    if "depths" in encoder_dict and isinstance(encoder_dict["depths"], list):
        encoder_dict["depths"] = tuple(encoder_dict["depths"])
    if "num_heads" in encoder_dict and isinstance(encoder_dict["num_heads"], list):
        encoder_dict["num_heads"] = tuple(encoder_dict["num_heads"])
    encoder_config = SwinConfig(**encoder_dict)

    # Parse decoder config
    decoder_dict = config_dict.get("decoder_config", {})
    decoder_config = BARTConfig(**decoder_dict)

    # Create main config
    config = DonutConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        encoder_hidden_size=config_dict.get("encoder_hidden_size", 1024),
        decoder_hidden_size=config_dict.get("decoder_hidden_size", 1024),
        max_length=config_dict.get("max_length", 512),
        min_length=config_dict.get("min_length", 1),
        num_beams=config_dict.get("num_beams", 1),
        length_penalty=config_dict.get("length_penalty", 1.0),
        early_stopping=config_dict.get("early_stopping", True),
    )

    return config


def save_config(config: DonutConfig, model_path: str):
    """
    Save configuration to model directory.

    Args:
        config: DonutConfig instance
        model_path: Path to model directory
    """
    import json
    from pathlib import Path

    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    encoder_dict = config.encoder_config.__dict__.copy()
    # Convert tuples to lists for JSON
    encoder_dict["image_size"] = list(encoder_dict["image_size"])
    encoder_dict["depths"] = list(encoder_dict["depths"])
    encoder_dict["num_heads"] = list(encoder_dict["num_heads"])

    config_dict = {
        "model_type": config.model_type,
        "encoder_config": encoder_dict,
        "decoder_config": config.decoder_config.__dict__,
        "encoder_hidden_size": config.encoder_hidden_size,
        "decoder_hidden_size": config.decoder_hidden_size,
        "max_length": config.max_length,
        "min_length": config.min_length,
        "num_beams": config.num_beams,
        "length_penalty": config.length_penalty,
        "early_stopping": config.early_stopping,
    }

    config_path = model_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
