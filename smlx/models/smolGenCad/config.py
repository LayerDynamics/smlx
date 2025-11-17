#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Configuration for smolGenCad model.

This module defines the configuration for the smolGenCad model, including
encoder, decoder, and CAD vocabulary settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smlx.utils.config import BaseModelArgs


@dataclass
class CADVocabularyConfig:
    """
    Configuration for CAD command vocabulary.

    Defines the vocabulary size and command types for CAD generation.
    """

    # Vocabulary size
    num_commands: int = 50  # Total CAD command types
    num_sketch_commands: int = 10  # 2D sketch operations
    num_3d_commands: int = 15  # 3D feature operations
    num_refinement_commands: int = 15  # Refinement operations
    num_control_commands: int = 10  # Control flow commands

    # Sequence parameters
    max_sequence_length: int = 272  # Max CAD operations per model
    max_parameters_per_command: int = 16  # Max params per operation

    # Parameter ranges
    min_coordinate: float = -1000.0  # Minimum coordinate value (mm)
    max_coordinate: float = 1000.0  # Maximum coordinate value (mm)
    min_distance: float = 0.1  # Minimum distance/radius (mm)
    max_distance: float = 1000.0  # Maximum distance/radius (mm)
    angle_unit: str = "degrees"  # "degrees" or "radians"

    def __post_init__(self):
        """Validate configuration."""
        assert self.num_commands > 0, "num_commands must be positive"
        assert self.max_sequence_length > 0, "max_sequence_length must be positive"
        assert (
            self.max_parameters_per_command > 0
        ), "max_parameters_per_command must be positive"


@dataclass
class EncoderConfig(BaseModelArgs):
    """
    Configuration for text encoder.

    Uses SmolLM2-135M as the text encoder.
    """

    # Model architecture (SmolLM2-135M)
    model_type: str = "smollm2"
    hidden_size: int = 576  # SmolLM2-135M hidden size
    num_hidden_layers: int = 30  # SmolLM2-135M layers
    num_attention_heads: int = 9  # SmolLM2-135M attention heads
    num_key_value_heads: int = 3  # GQA with 3 KV heads
    intermediate_size: int = 1536  # MLP hidden size
    vocab_size: int = 49152  # SmolLM2 vocab size
    max_position_embeddings: int = 2048  # Max sequence length
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False


@dataclass
class DecoderConfig(BaseModelArgs):
    """
    Configuration for CAD sequence decoder.

    Custom 8-layer transformer decoder optimized for CAD generation.
    Based on Text2CAD architecture (23M parameters).
    """

    # Model architecture
    model_type: str = "cad_decoder"
    hidden_size: int = 256  # Decoder hidden dimension
    num_hidden_layers: int = 8  # 8 transformer layers
    num_attention_heads: int = 8  # 8 attention heads
    num_key_value_heads: int = 8  # Full attention (no GQA in decoder)
    intermediate_size: int = 1024  # MLP hidden size (4x hidden_size)
    max_position_embeddings: int = 512  # Max decoder sequence length
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    dropout: float = 0.1  # Dropout rate

    # Cross-attention settings
    encoder_hidden_size: int = 576  # Must match encoder hidden size
    cross_attention: bool = True  # Enable cross-attention to encoder


@dataclass
class SmolGenCadConfig(BaseModelArgs):
    """
    Complete configuration for smolGenCad model.

    This is an encoder-decoder model with:
    - Text encoder: SmolLM2-135M (135M parameters)
    - CAD decoder: 8-layer transformer (23M parameters)
    - Total: ~158M parameters

    Architecture pattern follows Text2CAD (NeurIPS 2024).

    Example:
        >>> config = SmolGenCadConfig()
        >>> print(f"Total params: {config.total_parameters()}M")
        Total params: 158M

        >>> # Custom configuration
        >>> config = SmolGenCadConfig(
        ...     decoder=DecoderConfig(num_hidden_layers=6, hidden_size=128)
        ... )
    """

    # Model type
    model_type: str = "smolGenCad"

    # Sub-configurations
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    vocabulary: CADVocabularyConfig = field(default_factory=CADVocabularyConfig)

    # Generation settings
    max_new_tokens: int = 272  # Max tokens to generate
    temperature: float = 0.8  # Sampling temperature
    top_p: float = 0.95  # Nucleus sampling
    top_k: int = 50  # Top-k sampling
    repetition_penalty: float = 1.0  # Repetition penalty

    # Special tokens
    bos_token_id: int = 1  # Beginning of sequence
    eos_token_id: int = 2  # End of sequence
    pad_token_id: int = 0  # Padding token

    def __post_init__(self):
        """Post-initialization validation."""
        # Ensure encoder and decoder configs are instantiated
        if isinstance(self.encoder, dict):
            self.encoder = EncoderConfig(**self.encoder)
        if isinstance(self.decoder, dict):
            self.decoder = DecoderConfig(**self.decoder)
        if isinstance(self.vocabulary, dict):
            self.vocabulary = CADVocabularyConfig(**self.vocabulary)

        # Validate cross-attention dimensions match
        assert (
            self.decoder.encoder_hidden_size == self.encoder.hidden_size
        ), f"Decoder encoder_hidden_size ({self.decoder.encoder_hidden_size}) must match encoder hidden_size ({self.encoder.hidden_size})"

    def total_parameters(self) -> int:
        """
        Estimate total model parameters.

        Returns:
            Approximate parameter count in millions
        """
        # Encoder parameters (SmolLM2-135M)
        encoder_params = 135_000_000  # ~135M

        # Decoder parameters (8-layer transformer)
        # Approximation: 12 * num_layers * hidden_size^2
        decoder_params = (
            12
            * self.decoder.num_hidden_layers
            * (self.decoder.hidden_size**2)
        )

        total = encoder_params + decoder_params
        return total // 1_000_000  # Return in millions

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> SmolGenCadConfig:
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            SmolGenCadConfig instance
        """
        # Handle nested configs
        if "encoder" in config_dict and isinstance(config_dict["encoder"], dict):
            config_dict["encoder"] = EncoderConfig(**config_dict["encoder"])
        if "decoder" in config_dict and isinstance(config_dict["decoder"], dict):
            config_dict["decoder"] = DecoderConfig(**config_dict["decoder"])
        if "vocabulary" in config_dict and isinstance(
            config_dict["vocabulary"], dict
        ):
            config_dict["vocabulary"] = CADVocabularyConfig(
                **config_dict["vocabulary"]
            )

        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        from dataclasses import asdict

        return asdict(self)


# Default configuration
DEFAULT_CONFIG = SmolGenCadConfig()
