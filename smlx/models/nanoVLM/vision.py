#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
SigLIP-base Vision Encoder for nanoVLM.

Lightweight vision encoder that processes 224x224 images into patch embeddings.
Uses SigLIP-base (12 layers, 768 hidden) instead of the larger SigLIP-SO400M.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


class Attention(nn.Module):
    """Multi-head self-attention for vision encoder."""

    def __init__(self, dims: int, num_heads: int, bias: bool = True):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                f"dims ({dims}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        # Combined QKV projection (SigLIP-style)
        self.qkv = nn.Linear(dims, dims * 3, bias=bias)
        self.proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, D = x.shape

        # Project to Q, K, V together
        qkv = self.qkv(x)  # (B, L, 3*D)

        # Split into Q, K, V
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1)  # (B, L, 3, num_heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, num_heads, L, head_dim)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        # Reshape and project
        output = output.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.proj(output)


class MLP(nn.Module):
    """Feed-forward network for vision encoder."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.activation = nn.GELU(approx="precise")
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm architecture."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Self-attention with pre-norm
        r = self.self_attn(self.layer_norm1(x), mask)
        h = x + r

        # MLP with pre-norm
        r = self.mlp(self.layer_norm2(h))
        return h + r


class Encoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class VisionEmbeddings(nn.Module):
    """Patch embeddings for vision encoder."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Conv2d for patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=True,
        )

        # Number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        # Position embeddings
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Convert image to patch embeddings.

        Args:
            x: Image tensor
                Shape: (batch_size, height, width, channels) - NHWC format for MLX
                Example: (1, 224, 224, 3)

        Returns:
            Patch embeddings with position encoding
                Shape: (batch_size, num_patches, hidden_size)
                Example: (1, 196, 768) for 224x224 image with 16x16 patches
        """
        batch_size = x.shape[0]

        # Patch embedding via Conv2d
        # Input: (B, H, W, C) = (B, 224, 224, 3) - NHWC format for MLX
        # Output: (B, H/patch, W/patch, hidden) = (B, 14, 14, 768)
        patch_embeds = self.patch_embedding(x)

        # Reshape to sequence
        # (B, 14, 14, 768) -> (B, 196, 768)
        embeddings = patch_embeds.reshape(batch_size, -1, self.embed_dim)

        # Add position embeddings
        position_ids = mx.arange(self.num_positions)[None, :]  # (1, 196)
        position_embeds = self.position_embedding(position_ids)  # (1, 196, 768)

        embeddings = embeddings + position_embeds

        return embeddings


class VisionModel(nn.Module):
    """
    SigLIP-base vision encoder.

    Processes 224x224 images into patch embeddings using:
    - Conv2d patch embedding (16x16 patches)
    - 12-layer transformer encoder
    - Layer normalization

    Args:
        config: VisionConfig instance

    Example:
        >>> config = VisionConfig(
        ...     hidden_size=768,
        ...     num_hidden_layers=12,
        ...     num_attention_heads=12,
        ...     image_size=224,
        ...     patch_size=16
        ... )
        >>> model = VisionModel(config)
        >>> image = mx.random.normal((1, 3, 224, 224))
        >>> features = model(image)
        >>> print(features.shape)  # (1, 196, 768)
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self,
        pixel_values: mx.array,
        output_hidden_states: bool = False,
    ) -> mx.array:
        """
        Forward pass of vision encoder.

        Args:
            pixel_values: Image tensor
                Shape: (batch_size, height, width, channels) - NHWC format for MLX
                Values: Normalized [0, 1] or [-1, 1]
            output_hidden_states: Whether to return all hidden states

        Returns:
            Vision features
                Shape: (batch_size, num_patches, hidden_size)
                Example: (1, 196, 768)

        Note:
            For 224x224 images with 16x16 patches:
            - 14 x 14 = 196 patches
            - Each patch is embedded to 768 dimensions
        """
        # Embed patches with position encoding
        hidden_states = self.embeddings(pixel_values)

        # Encode through transformer layers
        hidden_states = self.encoder(hidden_states)

        # Final layer norm
        hidden_states = self.post_layernorm(hidden_states)

        return hidden_states
