#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Phi language model for Moondream2.

Phi is a Microsoft-developed transformer model optimized for
efficient reasoning and instruction following.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from smlx.utils.cache import KVCache

from .config import TextConfig


class PhiAttention(nn.Module):
    """Multi-head attention with optional grouped-query attention."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        # QKV projection
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # Partial RoPE
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        self.rope = nn.RoPE(
            self.rotary_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply partial RoPE (only to rotary_dim portion of head_dim)
        if self.rotary_dim > 0:
            q_rot = q[..., : self.rotary_dim]
            k_rot = k[..., : self.rotary_dim]

            # Use explicit position_ids if provided, otherwise fall back to cache.offset
            if position_ids is not None:
                # position_ids: [B, L] or [L] -> use first position for offset
                if len(position_ids.shape) == 2:
                    offset = int(position_ids[0, 0].item())
                else:
                    offset = int(position_ids[0].item())
                q_rot = self.rope(q_rot, offset=offset)
                k_rot = self.rope(k_rot, offset=offset)
            elif cache is not None:
                q_rot = self.rope(q_rot, offset=cache.offset)
                k_rot = self.rope(k_rot, offset=cache.offset)
            else:
                q_rot = self.rope(q_rot)
                k_rot = self.rope(k_rot)

            # Concatenate rotated and non-rotated parts
            q = mx.concatenate([q_rot, q[..., self.rotary_dim :]], axis=-1)
            k = mx.concatenate([k_rot, k[..., self.rotary_dim :]], axis=-1)

        # Update cache
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Repeat KV heads if using GQA
        if self.num_kv_heads != self.num_heads:
            k = mx.repeat(k, self.num_heads // self.num_kv_heads, axis=1)
            v = mx.repeat(v, self.num_heads // self.num_kv_heads, axis=1)

        # Scaled dot-product attention
        attn_output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(attn_output)

        return output


class PhiMLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

        # Phi uses gelu_new variant
        if config.hidden_act == "gelu_new":
            self.activation = nn.GELU()
        else:
            self.activation = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class PhiDecoderLayer(nn.Module):
    """Transformer decoder layer with pre-normalization."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = PhiAttention(config)
        self.mlp = PhiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, cache=cache, position_ids=position_ids)
        x = residual + x

        # Pre-norm MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class PhiModel(nn.Module):
    """Phi language model."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [PhiDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        x = self.embed_tokens(input_ids)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, mask=mask, cache=layer_cache, position_ids=position_ids)

        x = self.final_layernorm(x)

        return x


__all__ = [
    "PhiModel",
    "PhiDecoderLayer",
    "PhiAttention",
    "PhiMLP",
]
