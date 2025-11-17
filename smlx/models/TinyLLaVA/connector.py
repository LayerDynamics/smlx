#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Vision-Language Connector for TinyLLaVA.

Projects vision features from the vision encoder into the
language model's embedding space using a simple MLP.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import ProjectorConfig


class MLPProjector(nn.Module):
    """2-layer MLP projector with GELU activation.

    Maps vision features (1152-dim) to language features (2048-dim).
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_act: str = "gelu",
    ):
        super().__init__()

        # Two-layer MLP: vision_dim -> text_dim -> text_dim
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=True)

        # Activation function
        if projector_hidden_act == "gelu":
            self.activation = nn.GELU()
        elif projector_hidden_act == "relu":
            self.activation = nn.ReLU()
        elif projector_hidden_act == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()  # Default to GELU

    def __call__(self, vision_features: mx.array) -> mx.array:
        """
        Args:
            vision_features: Vision encoder output [B, num_patches, vision_dim]

        Returns:
            Projected features [B, num_patches, text_dim]
        """
        x = self.linear_1(vision_features)
        x = self.activation(x)
        x = self.linear_2(x)

        return x


class PerceiverCrossAttention(nn.Module):
    """Cross-attention layer for Perceiver Resampler.

    Implements multi-head cross-attention with grouped query attention (GQA),
    where queries come from learnable latent tokens and keys/values come from
    vision features.

    Reference: Based on Idefics2PerceiverAttention from mlx-vlm.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        """
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads for queries
            num_kv_heads: Number of key/value heads (for GQA)
            head_dim: Dimension of each attention head
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        # Query projection (from latents)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)

        # Key/value projections (from vision features + latents)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def __call__(
        self,
        latents: mx.array,
        context: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            latents: Query tokens [B, num_queries, hidden_size]
            context: Context features (vision) [B, num_patches, hidden_size]
            mask: Optional attention mask

        Returns:
            Attention output [B, num_queries, hidden_size]
        """
        B, L_q, _ = latents.shape
        L_kv = context.shape[1]

        # Concatenate context and latents for keys/values
        # This allows latents to attend to both vision features and themselves
        hidden_states = mx.concatenate([context, latents], axis=1)  # [B, L_kv + L_q, hidden]

        # Project queries (only from latents)
        queries = self.q_proj(latents)  # [B, L_q, num_heads * head_dim]

        # Project keys and values (from concatenated context + latents)
        keys = self.k_proj(hidden_states)  # [B, L_kv + L_q, num_kv_heads * head_dim]
        values = self.v_proj(hidden_states)  # [B, L_kv + L_q, num_kv_heads * head_dim]

        # Reshape for multi-head attention
        queries = queries.reshape(B, L_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L_kv + L_q, self.num_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L_kv + L_q, self.num_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Scaled dot-product attention
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        # Reshape output
        output = output.transpose(0, 2, 1, 3).reshape(B, L_q, -1)

        return self.o_proj(output)


class GatedMLP(nn.Module):
    """Gated MLP with SiLU activation.

    Uses gating mechanism: down_proj(silu(gate_proj(x)) * up_proj(x))
    This is the same MLP structure used in LLaMA and other modern transformers.

    Reference: Based on Idefics2 MLP from mlx-vlm.
    """

    def __init__(self, dim: int, hidden_dim: int, output_size: int):
        """
        Args:
            dim: Input dimension
            hidden_dim: Hidden dimension (typically 4x input dim)
            output_size: Output dimension
        """
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor [B, seq_len, dim]

        Returns:
            Output tensor [B, seq_len, output_size]
        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class PerceiverLayer(nn.Module):
    """Single Perceiver Resampler layer.

    Combines cross-attention and feedforward MLP with pre-normalization
    and residual connections.

    Architecture:
        latents_norm = RMSNorm(latents)
        context_norm = RMSNorm(context)
        latents = latents + CrossAttention(latents_norm, context_norm)
        latents = latents + MLP(RMSNorm(latents))

    Reference: Based on Idefics2PerceiverLayer from mlx-vlm.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
    ):
        """
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            num_kv_heads: Number of key/value heads (for GQA)
            head_dim: Dimension per head
            rms_norm_eps: Epsilon for RMSNorm
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Pre-normalization for cross-attention
        self.input_latents_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_context_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        # Cross-attention layer
        self.cross_attn = PerceiverCrossAttention(
            hidden_size, num_heads, num_kv_heads, head_dim
        )

        # Pre-normalization for MLP
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        # Gated MLP (4x hidden expansion)
        self.mlp = GatedMLP(hidden_size, hidden_size * 4, hidden_size)

    def __call__(
        self,
        latents: mx.array,
        context: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            latents: Query tokens [B, num_queries, hidden_size]
            context: Context features (vision) [B, num_patches, hidden_size]
            mask: Optional attention mask

        Returns:
            Updated latents [B, num_queries, hidden_size]
        """
        # Pre-norm for cross-attention
        latents_norm = self.input_latents_norm(latents)
        context_norm = self.input_context_norm(context)

        # Cross-attention with residual
        attn_out = self.cross_attn(latents_norm, context_norm, mask=mask)
        latents = latents + attn_out

        # MLP with residual
        mlp_input = self.post_attention_layernorm(latents)
        mlp_out = self.mlp(mlp_input)
        latents = latents + mlp_out

        return latents


class ResamplerProjector(nn.Module):
    """Perceiver Resampler projector with cross-attention.

    Uses learnable query tokens and cross-attention layers to compress
    variable-length vision features into a fixed number of tokens.

    Architecture:
        1. Project vision features to resampler hidden size
        2. Initialize learnable latent query tokens
        3. Apply N layers of Perceiver (cross-attention + MLP)
        4. Final normalization
        5. Project to language model hidden size

    Reference: Based on Idefics2PerceiverResampler from mlx-vlm.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_query_tokens: int = 128,
        num_layers: int = 3,
        resampler_hidden_size: int = 768,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 96,
        rms_norm_eps: float = 1e-6,
    ):
        """
        Args:
            vision_hidden_size: Vision encoder hidden size (e.g., 1152 for SigLIP)
            text_hidden_size: Language model hidden size (e.g., 2048 for TinyLlama)
            num_query_tokens: Number of learnable query tokens (default: 128)
            num_layers: Number of Perceiver layers (default: 3)
            resampler_hidden_size: Hidden size for resampler (default: 768)
            num_heads: Number of attention heads (default: 16)
            num_kv_heads: Number of key/value heads for GQA (default: 4)
            head_dim: Dimension per attention head (default: 96)
            rms_norm_eps: Epsilon for RMSNorm (default: 1e-6)
        """
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.resampler_hidden_size = resampler_hidden_size

        # Learnable latent query tokens (initialized to ones, not zeros!)
        # These will be trained to extract relevant visual information
        self.latents = mx.ones((num_query_tokens, resampler_hidden_size))

        # Vision projection: map vision features to resampler hidden size
        self.vision_projection = nn.Linear(vision_hidden_size, resampler_hidden_size)

        # Stack of Perceiver layers
        self.layers = [
            PerceiverLayer(
                resampler_hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
            )
            for _ in range(num_layers)
        ]

        # Final normalization
        self.norm = nn.RMSNorm(resampler_hidden_size, eps=rms_norm_eps)

        # Output projection: map to language model hidden size
        self.output_projection = nn.Linear(resampler_hidden_size, text_hidden_size)

    def __call__(self, vision_features: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            vision_features: Vision encoder output [B, num_patches, vision_dim]
            mask: Optional attention mask

        Returns:
            Resampled features [B, num_query_tokens, text_dim]
        """
        B = vision_features.shape[0]

        # Project vision features to resampler hidden size
        vision_proj = self.vision_projection(vision_features)  # [B, num_patches, hidden]

        # Expand learnable latents to batch size
        latents = mx.expand_dims(self.latents, axis=0)  # [1, num_queries, hidden]
        latents = mx.repeat(latents, B, axis=0)  # [B, num_queries, hidden]

        # Apply all Perceiver layers
        # Each layer performs cross-attention between latents and vision features
        for layer in self.layers:
            latents = layer(latents, vision_proj, mask=mask)

        # Final normalization
        output = self.norm(latents)  # [B, num_queries, hidden]

        # Project to language model hidden size
        output = self.output_projection(output)  # [B, num_queries, text_dim]

        return output


def build_projector(
    config: ProjectorConfig,
    vision_hidden_size: int,
    text_hidden_size: int,
) -> nn.Module:
    """Build the appropriate projector based on config.

    Args:
        config: ProjectorConfig
        vision_hidden_size: Vision encoder hidden size
        text_hidden_size: Language model hidden size

    Returns:
        Projector module (MLPProjector or ResamplerProjector)
    """
    if config.projector_type == "mlp2x_gelu":
        return MLPProjector(
            vision_hidden_size,
            text_hidden_size,
            config.projector_hidden_act,
        )
    elif config.projector_type == "resampler" and config.use_resampler:
        return ResamplerProjector(
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=text_hidden_size,
            num_query_tokens=config.num_query_tokens,
            num_layers=config.resampler_n_layers,
            resampler_hidden_size=config.resampler_hidden_size,
            num_heads=config.resampler_n_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.resampler_head_dim,
            rms_norm_eps=config.rms_norm_eps,
        )
    else:
        # Default to simple MLP
        return MLPProjector(
            vision_hidden_size,
            text_hidden_size,
            config.projector_hidden_act,
        )


__all__ = [
    "MLPProjector",
    "PerceiverCrossAttention",
    "GatedMLP",
    "PerceiverLayer",
    "ResamplerProjector",
    "build_projector",
]
