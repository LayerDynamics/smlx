#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
TrOCR-small Configuration.

TrOCR combines a BEiT vision encoder with a RoBERTa text decoder
for optical character recognition.
"""

from dataclasses import dataclass


@dataclass
class TrOCRVisionConfig:
    """Configuration for BEiT vision encoder.

    BEiT (BERT Pre-Training of Image Transformers) is used
    as the vision encoder to extract features from images.
    """

    # Model architecture
    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0

    # Image parameters
    image_size: int = 384
    patch_size: int = 16
    num_channels: int = 3

    # Layer norm
    layer_norm_eps: float = 1e-12

    @property
    def num_patches(self) -> int:
        """Calculate number of patches."""
        return (self.image_size // self.patch_size) ** 2


@dataclass
class TrOCRDecoderConfig:
    """Configuration for RoBERTa text decoder.

    RoBERTa decoder generates text autoregressively from
    vision encoder features.
    """

    # Model architecture
    vocab_size: int = 64044  # XLMRoberta vocab (TrOCR-specific)
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # Decoder-specific
    max_position_embeddings: int = 512
    is_decoder: bool = True
    add_cross_attention: bool = True

    # Special tokens
    bos_token_id: int = 0
    eos_token_id: int = 2
    pad_token_id: int = 1

    # Layer norm
    layer_norm_eps: float = 1e-5


@dataclass
class TrOCRConfig:
    """Complete TrOCR configuration.

    Combines vision encoder and text decoder configurations.
    """

    model_type: str = "trocr"

    # Sub-configs
    vision_config: TrOCRVisionConfig = None
    decoder_config: TrOCRDecoderConfig = None

    # Generation parameters
    max_length: int = 128
    num_beams: int = 1
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    # Variant
    variant: str = "printed"  # "printed" or "handwritten"

    def __post_init__(self):
        """Initialize sub-configs if not provided."""
        if self.vision_config is None:
            self.vision_config = TrOCRVisionConfig()
        if self.decoder_config is None:
            self.decoder_config = TrOCRDecoderConfig()


# Default configurations
DEFAULT_CONFIG_PRINTED = TrOCRConfig(variant="printed")
DEFAULT_CONFIG_HANDWRITTEN = TrOCRConfig(variant="handwritten")


def create_config_from_dict(config_dict: dict) -> TrOCRConfig:
    """Create TrOCR config from dictionary.

    Args:
        config_dict: Configuration dictionary from HuggingFace

    Returns:
        TrOCR configuration
    """
    # Extract vision config (support both vision_config and encoder for backwards compat)
    vision_dict = config_dict.get("vision_config", config_dict.get("encoder", {}))
    vision_config = TrOCRVisionConfig(
        hidden_size=vision_dict.get("hidden_size", 384),
        num_hidden_layers=vision_dict.get("num_hidden_layers", 12),
        num_attention_heads=vision_dict.get("num_attention_heads", 6),
        intermediate_size=vision_dict.get("intermediate_size", 1536),
        image_size=vision_dict.get("image_size", 384),
        patch_size=vision_dict.get("patch_size", 16),
    )

    # Extract decoder config (support both decoder_config and decoder for backwards compat)
    decoder_dict = config_dict.get("decoder_config", config_dict.get("decoder", {}))
    decoder_config = TrOCRDecoderConfig(
        vocab_size=decoder_dict.get("vocab_size", 64044),
        hidden_size=decoder_dict.get("hidden_size", 384),
        num_hidden_layers=decoder_dict.get("num_hidden_layers", 6),
        num_attention_heads=decoder_dict.get("num_attention_heads", 6),
        intermediate_size=decoder_dict.get("intermediate_size", 1536),
        max_position_embeddings=decoder_dict.get("max_position_embeddings", 512),
        bos_token_id=decoder_dict.get("bos_token_id", 0),
        eos_token_id=decoder_dict.get("eos_token_id", 2),
        pad_token_id=decoder_dict.get("pad_token_id", 1),
    )

    # Create complete config
    config = TrOCRConfig(
        vision_config=vision_config,
        decoder_config=decoder_config,
        variant=config_dict.get("variant", "printed"),
    )

    return config


__all__ = [
    "TrOCRConfig",
    "TrOCRVisionConfig",
    "TrOCRDecoderConfig",
    "DEFAULT_CONFIG_PRINTED",
    "DEFAULT_CONFIG_HANDWRITTEN",
    "create_config_from_dict",
]
