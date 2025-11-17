#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
SigLIP Vision Encoder for SmolVLM-256M-Instruct.

The vision encoder processes images into patch embeddings:
- Conv2d patch embedding (14x14 patches from 384x384 images)
- Position embeddings for each patch
- 27-layer transformer encoder with LayerNorm
- Post-LayerNorm for final pooling
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


def check_array_shape(arr: mx.array) -> bool:
    """Check if conv2d weight array is in correct MLX format."""
    shape = arr.shape

    if len(shape) != 4:
        return False

    out_channels, kH, kW, _ = shape

    # MLX expects: [out_channels, kH, kW, in_channels]
    # Check if out_channels is the largest, and kH and kW are the same
    if (out_channels >= kH) and (out_channels >= kW) and (kH == kW):
        return True
    else:
        return False


class Attention(nn.Module):
    """Multi-head self-attention for vision encoder."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                f"The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    """Feed-forward network for vision encoder."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm architecture."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Pre-norm architecture
        r = self.self_attn(self.layer_norm1(x), mask)
        h = x + r
        r = self.mlp(self.layer_norm2(h))
        return h + r


class Encoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
        mask: Optional[mx.array] = None,
    ) -> tuple:
        encoder_states = (x,) if output_hidden_states else None
        h = x
        for layer in self.layers:
            x = layer(x, mask=mask)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)
            h = x

        return (h, encoder_states)


class VisionEmbeddings(nn.Module):
    """Convert images to patch embeddings with position encoding."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Conv2d patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Position embeddings
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C] (MLX format)
        # Conv2d output: [B, H', W', embed_dim]
        patch_embeddings = self.patch_embedding(x)

        # Flatten spatial dimensions: [B, num_patches, embed_dim]
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)

        # Add position embeddings for the actual number of patches
        num_patches = patch_embeddings.shape[1]
        position_ids = mx.array(mx.arange(num_patches)[None, :])
        embeddings = patch_embeddings
        embeddings += self.position_embedding(position_ids)
        return embeddings


class VisionModel(nn.Module):
    """SigLIP Vision Encoder.

    Processes images through:
    1. Patch embedding + position encoding
    2. 27-layer transformer encoder
    3. Post-LayerNorm

    Args:
        config: VisionConfig with model hyperparameters

    Returns:
        pooler_output: Final normalized embeddings [B, num_patches, hidden_size]
        x: Initial embeddings (before encoder)
        encoder_states: Tuple of hidden states from all layers (if requested)
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type not in [
            "siglip_vision_model",
            "idefics3",
            "idefics3_vision",
            "smolvlm_vision",
        ]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> tuple:
        # Convert to patch embeddings
        x = self.embeddings(x)
        x = x.astype(self.embeddings.patch_embedding.weight.dtype)

        # Encode through transformer layers
        encoder_outputs = self.encoder(
            x=x, output_hidden_states=output_hidden_states, mask=None
        )

        # Final normalization
        pooler_output = self.post_layernorm(encoder_outputs[0])

        return pooler_output, x, encoder_outputs[-1]

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Convert PyTorch weights to MLX format.

        PyTorch conv2d weight shape: [out_channels, in_channels, kH, kW]
        MLX conv2d weight shape: [out_channels, kH, kW, in_channels]
        """
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embedding.weight" in k:
                # Transpose conv2d weights if needed
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights


__all__ = [
    "VisionModel",
    "VisionEmbeddings",
    "Encoder",
    "EncoderLayer",
    "Attention",
    "MLP",
]
