#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Vision encoder for Moondream2.

Uses a custom architecture with crop-based tiling for efficient
high-resolution image processing. Extracts both global and local
features from multiple layers.
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from .config import VisionConfig


def create_patches(x: mx.array, patch_size: int) -> mx.array:
    """Create patches from image tensor.

    Converts image from [B, C, H, W] to [B, num_patches, patch_dim]
    where patch_dim = C * patch_size * patch_size.

    This matches the HuggingFace Moondream2 implementation.

    Args:
        x: Image tensor [B, C, H, W]
        patch_size: Size of each patch (e.g., 14)

    Returns:
        Patches tensor [B, (H/P)*(W/P), C*P*P]
    """
    B, C, H, W = x.shape
    P = patch_size

    # Step 1: Split H and W dimensions into patches
    # [B, C, H, W] -> [B, C, H/P, P, W/P, P]
    x = x.reshape(B, C, H // P, P, W // P, P)

    # Step 2: Rearrange dimensions
    # [B, C, H/P, P, W/P, P] -> [B, H/P, W/P, C, P, P]
    x = x.transpose(0, 2, 4, 1, 3, 5)

    # Step 3: Flatten patches
    # [B, H/P, W/P, C, P, P] -> [B, (H/P)*(W/P), C*P*P]
    x = x.reshape(B, (H // P) * (W // P), C * P * P)

    return x


class VisionAttention(nn.Module):
    """Multi-head self-attention for vision encoder."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn, axis=-1)

        # Apply attention to values
        x = mx.matmul(attn, v)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)

        return self.proj(x)


class VisionMLP(nn.Module):
    """MLP for vision encoder."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class VisionEncoderLayer(nn.Module):
    """Transformer encoder layer for vision."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Pre-norm architecture (like ViT)
        x = x + self.attention(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class VisionEmbeddings(nn.Module):
    """Patch embedding for images using Linear projection.

    Following HuggingFace Moondream2 implementation:
    - Use create_patches() to manually extract patches
    - Apply Linear projection instead of Conv2d
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.num_patches = (config.image_size // config.patch_size) ** 2

        # Patch embedding using Linear projection (NOT Conv2d)
        # patch_dim = patch_size * patch_size * num_channels
        # For Moondream2: 14 * 14 * 3 = 588
        patch_dim = config.patch_size * config.patch_size * config.num_channels
        self.patch_embedding = nn.Linear(patch_dim, config.hidden_size, bias=True)

        # Position embeddings
        self.position_embeddings = mx.zeros((1, self.num_patches, config.hidden_size))

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """
        Args:
            pixel_values: Image tensor [B, C, H, W]

        Returns:
            Patch embeddings [B, num_patches, hidden_size]
        """
        batch_size = pixel_values.shape[0]

        # Create patches: [B, C, H, W] -> [B, num_patches, patch_dim]
        x = create_patches(pixel_values, self.patch_size)

        # Linear projection: [B, num_patches, patch_dim] -> [B, num_patches, hidden_size]
        x = self.patch_embedding(x)

        # Add position embeddings
        x = x + self.position_embeddings

        return x


class VisionEncoder(nn.Module):
    """Vision encoder with multi-layer feature extraction."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = VisionEmbeddings(config)
        self.layers = [
            VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        pixel_values: mx.array,
        output_hidden_states: bool = True,
    ) -> Tuple[mx.array, Optional[List[mx.array]]]:
        """
        Args:
            pixel_values: [B, C, H, W] image tensor
            output_hidden_states: Whether to return all layer outputs

        Returns:
            Final hidden states and all hidden states for feature selection
        """
        x = self.embeddings(pixel_values)

        hidden_states = []
        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.post_layernorm(x)

        return x, hidden_states if output_hidden_states else None


def prepare_crops(
    image: mx.array,
    image_size: int = 378,
    max_crops: int = 4,
) -> Tuple[mx.array, List[Tuple[int, int, int, int]]]:
    """Prepare image crops for tiled processing.

    Args:
        image: Input image [C, H, W]
        image_size: Target size for each crop
        max_crops: Maximum number of crops to generate

    Returns:
        Cropped images [num_crops, C, image_size, image_size] and crop coordinates
    """
    C, H, W = image.shape

    # If image is small enough, just resize
    if H <= image_size and W <= image_size:
        # Pad to square
        max_dim = max(H, W)
        padded = mx.zeros((C, max_dim, max_dim))
        padded[:, :H, :W] = image

        # Resize to target size using PIL
        # Input is already normalized, so we work in float32 directly
        # Convert from [C, H, W] to [H, W, C] for PIL
        padded_hwc = padded.transpose(1, 2, 0)  # [H, W, C]
        padded_np = np.array(padded_hwc)

        # Use scipy's zoom for float resizing (preserves normalized values)
        scale_h = image_size / padded_np.shape[0]
        scale_w = image_size / padded_np.shape[1]
        resized_np = zoom(padded_np, (scale_h, scale_w, 1), order=3)  # bicubic

        # Convert back to MLX array in [C, H, W] format
        resized = mx.array(resized_np.astype(np.float32)).transpose(2, 0, 1)  # [C, H, W]

        return resized[None, :, :, :], [(0, 0, H, W)]

    # Calculate crop positions
    # Simple tiling strategy - divide into grid
    crop_h = image_size
    crop_w = image_size

    stride_h = H // int(np.sqrt(max_crops))
    stride_w = W // int(np.sqrt(max_crops))

    crops = []
    crop_coords = []

    for y in range(0, H - crop_h + 1, stride_h):
        for x in range(0, W - crop_w + 1, stride_w):
            if len(crops) >= max_crops:
                break

            crop = image[:, y : y + crop_h, x : x + crop_w]
            crops.append(crop)
            crop_coords.append((x, y, x + crop_w, y + crop_h))

        if len(crops) >= max_crops:
            break

    # Stack crops
    crops_array = mx.stack(crops, axis=0)  # [num_crops, C, H, W]

    return crops_array, crop_coords


def reconstruct_from_crops(
    crop_features: List[mx.array],
    crop_coords: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    overlap: int = 14,
) -> mx.array:
    """Reconstruct features from crops with overlap handling.

    Args:
        crop_features: List of feature arrays [N_patches, hidden_size]
        crop_coords: Crop coordinates (x1, y1, x2, y2)
        image_shape: Original image (H, W)
        overlap: Overlap size in pixels for blending

    Returns:
        Reconstructed features
    """
    # For simplicity, just concatenate all crop features
    # A more sophisticated approach would handle overlaps and reconstruction
    all_features = mx.concatenate(crop_features, axis=0)

    return all_features


__all__ = [
    "VisionEncoder",
    "VisionEmbeddings",
    "VisionEncoderLayer",
    "prepare_crops",
    "reconstruct_from_crops",
]
