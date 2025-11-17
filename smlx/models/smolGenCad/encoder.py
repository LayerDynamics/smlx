#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Text encoder for smolGenCad.

Wraps SmolLM2-135M as the text encoder for encoding natural language
CAD descriptions into embeddings.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from smlx.models.SmolLM2_135M.model import LlamaModel
from smlx.models.SmolLM2_135M.model import ModelArgs as SmolLM2Args

from .config import EncoderConfig


class TextEncoder(nn.Module):
    """
    Text encoder using SmolLM2-135M.

    Encodes natural language descriptions of CAD models into
    continuous embeddings for the CAD decoder.

    Example:
        >>> config = EncoderConfig()
        >>> encoder = TextEncoder(config)
        >>> # Input: "Create a cylinder with radius 5cm"
        >>> input_ids = mx.array([[...]])  # Tokenized text
        >>> embeddings = encoder(input_ids)
        >>> embeddings.shape
        (1, seq_len, 576)  # 576 = SmolLM2-135M hidden size
    """

    def __init__(self, config: EncoderConfig):
        """
        Initialize text encoder.

        Args:
            config: Encoder configuration (SmolLM2-135M config)
        """
        super().__init__()
        self.config = config

        # Convert EncoderConfig to SmolLM2Args
        smollm2_args = SmolLM2Args(
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            rope_traditional=config.rope_traditional,
            attention_bias=config.attention_bias,
            mlp_bias=config.mlp_bias,
        )

        # Use SmolLM2-135M as the encoder backbone
        self.model = LlamaModel(smollm2_args)

    def __call__(
        self,
        input_ids: mx.array,
        cache=None,
    ) -> mx.array:
        """
        Encode text to embeddings.

        Args:
            input_ids: Token IDs [batch, seq_len]
            cache: Optional KV cache (not typically used for encoder)

        Returns:
            Embeddings [batch, seq_len, hidden_size]
        """
        return self.model(input_ids, cache=cache)

    @property
    def hidden_size(self) -> int:
        """Get encoder hidden size."""
        return self.config.hidden_size
