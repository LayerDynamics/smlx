#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
CAD sequence decoder.

Custom 8-layer transformer decoder for autoregressive CAD sequence generation
with cross-attention to text encoder outputs.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .config import DecoderConfig
from .tokenizer import CAD_VOCAB_SIZE


class CADDecoderAttention(nn.Module):
    """
    Multi-head attention for CAD decoder.

    Supports both self-attention and cross-attention to encoder outputs.
    """

    def __init__(self, config: DecoderConfig, is_cross_attention: bool = False):
        """
        Initialize attention layer.

        Args:
            config: Decoder configuration
            is_cross_attention: Whether this is cross-attention layer
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.is_cross_attention = is_cross_attention

        # For cross-attention, key/value come from encoder
        kv_size = config.encoder_hidden_size if is_cross_attention else config.hidden_size

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(kv_size, config.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(kv_size, config.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        # RoPE for positional encoding (only for self-attention)
        if not is_cross_attention:
            self.rope = nn.RoPE(
                self.head_dim,
                traditional=config.rope_traditional,
                base=config.rope_theta,
            )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        """
        Apply attention.

        Args:
            hidden_states: Query hidden states [batch, seq_len, hidden_size]
            encoder_hidden_states: Key/value states for cross-attention [batch, enc_len, enc_hidden]
            mask: Attention mask
            cache: KV cache for generation

        Returns:
            Attention output [batch, seq_len, hidden_size]
        """
        B, L, D = hidden_states.shape

        # Project queries
        queries = self.q_proj(hidden_states)
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

        # Project keys and values
        if self.is_cross_attention:
            # Cross-attention: K,V from encoder
            assert (
                encoder_hidden_states is not None
            ), "encoder_hidden_states required for cross-attention"
            keys = self.k_proj(encoder_hidden_states)
            values = self.v_proj(encoder_hidden_states)
            enc_len = encoder_hidden_states.shape[1]
            keys = keys.reshape(B, enc_len, self.num_heads, -1).transpose(0, 2, 1, 3)
            values = values.reshape(B, enc_len, self.num_heads, -1).transpose(0, 2, 1, 3)
        else:
            # Self-attention: K,V from decoder
            keys = self.k_proj(hidden_states)
            values = self.v_proj(hidden_states)
            keys = keys.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

            # Apply RoPE to queries and keys
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

        # Scaled dot-product attention
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask if not self.is_cross_attention else None,
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class CADDecoderMLP(nn.Module):
    """
    MLP block for CAD decoder.

    Uses SwiGLU activation (same as SmolLM).
    """

    def __init__(self, config: DecoderConfig):
        """
        Initialize MLP.

        Args:
            config: Decoder configuration
        """
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply MLP with SwiGLU activation."""
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class CADDecoderBlock(nn.Module):
    """
    Single transformer decoder block.

    Architecture:
        x -> LayerNorm -> Self-Attention -> Add
          -> LayerNorm -> Cross-Attention -> Add
          -> LayerNorm -> MLP -> Add
    """

    def __init__(self, config: DecoderConfig):
        """
        Initialize decoder block.

        Args:
            config: Decoder configuration
        """
        super().__init__()

        # Self-attention
        self.self_attn = CADDecoderAttention(config, is_cross_attention=False)
        self.self_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Cross-attention (if enabled)
        self.cross_attn_enabled = config.cross_attention
        if self.cross_attn_enabled:
            self.cross_attn = CADDecoderAttention(config, is_cross_attention=True)
            self.cross_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP
        self.mlp = CADDecoderMLP(config)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        """
        Apply decoder block.

        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            encoder_hidden_states: Encoder outputs for cross-attention
            mask: Causal attention mask
            cache: KV cache

        Returns:
            Output hidden states [batch, seq_len, hidden_size]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask=mask, cache=cache)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Cross-attention with residual (if enabled)
        if self.cross_attn_enabled and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_norm(hidden_states)
            hidden_states = self.cross_attn(
                hidden_states, encoder_hidden_states=encoder_hidden_states
            )
            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CADDecoder(nn.Module):
    """
    CAD sequence decoder.

    8-layer transformer decoder with self-attention and cross-attention
    for autoregressive CAD command sequence generation.

    Architecture follows Text2CAD decoder design (23M parameters).

    Example:
        >>> config = DecoderConfig()
        >>> decoder = CADDecoder(config)
        >>> # Input: CAD token embeddings
        >>> input_embeds = mx.random.normal((1, 50, 256))  # [batch, seq, hidden]
        >>> encoder_outputs = mx.random.normal((1, 20, 576))  # [batch, enc_len, enc_hidden]
        >>> output = decoder(input_embeds, encoder_hidden_states=encoder_outputs)
        >>> output.shape
        (1, 50, 256)
    """

    def __init__(self, config: DecoderConfig):
        """
        Initialize CAD decoder.

        Args:
            config: Decoder configuration
        """
        super().__init__()
        self.config = config

        # Embedding layer for CAD tokens. Sized to the tokenizer's vocabulary so
        # every emittable token id (max 1103) is representable.
        self.vocab_size = CAD_VOCAB_SIZE
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size)

        # Transformer decoder layers
        self.layers = [CADDecoderBlock(config) for _ in range(config.num_hidden_layers)]

        # Final layer norm
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        """
        Decode CAD sequence.

        Args:
            input_ids: CAD token IDs [batch, seq_len]
            encoder_hidden_states: Encoder outputs [batch, enc_len, enc_hidden]
            cache: KV cache for generation

        Returns:
            Decoder hidden states [batch, seq_len, hidden_size]
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        # Create causal mask
        seq_len = input_ids.shape[1]
        if seq_len == 1:
            mask = None
        else:
            mask = "causal"

        # Apply decoder layers
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                mask=mask,
                cache=layer_cache,
            )

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states
