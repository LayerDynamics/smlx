#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Vision-to-Language Projection Layer with Pixel Shuffle.

Projects vision encoder outputs to language model input space using:
1. Pixel shuffle to reduce spatial dimensions and increase channels
2. Linear projection to language hidden size
"""

import mlx.core as mx
import mlx.nn as nn

from .config import ProjectionConfig


class MLPProjection(nn.Module):
    """
    Multi-layer perceptron with pixel shuffle for projecting vision features to language space.

    Matches HuggingFace nanoVLM architecture:
    1. Pixel shuffle: Reduces spatial resolution by factor 2 (196 → 49 patches)
       - Increases channels: 768 → 3072 (768 × 2²)
    2. Linear projection: 3072 → 576 (language hidden size)

    Architecture:
        vision_hidden (768) -> pixel_shuffle(factor=2) -> 3072 -> Linear -> language_hidden (576)

    Args:
        config: ProjectionConfig with dimensions and activation
        pixel_shuffle_factor: Downsampling factor for pixel shuffle (default: 2 for nanoVLM)
    """

    def __init__(self, config: ProjectionConfig, pixel_shuffle_factor: int = 2):
        super().__init__()
        self.config = config
        self.pixel_shuffle_factor = pixel_shuffle_factor

        # Calculate intermediate size after pixel shuffle
        # vision_hidden × (factor²) = 768 × 4 = 3072
        intermediate_size = config.vision_hidden_size * (pixel_shuffle_factor ** 2)

        # Single linear projection from shuffled features to language space
        # 3072 → 576 (no bias - HF model doesn't have bias for this layer)
        self.proj = nn.Linear(intermediate_size, config.language_hidden_size, bias=False)

    def pixel_shuffle(self, x: mx.array, scale_factor: int = 2) -> mx.array:
        """
        Pixel shuffle operation to reduce spatial dimensions.

        Reduces spatial resolution by scale_factor while increasing channel depth
        by scale_factor².

        Args:
            x: Input tensor [B, seq, embed_dim] where seq = height × width
            scale_factor: Downsampling factor (default: 2)

        Returns:
            Downsampled tensor [B, seq/(scale_factor²), embed_dim × (scale_factor²)]

        Example:
            Input: [1, 196, 768] (14×14 patches)
            scale_factor=2
            Output: [1, 49, 3072] (7×7 patches, 768×4=3072 channels)
        """
        bsz, seq, embed_dim = x.shape
        height = width = int(seq**0.5)

        # Reshape to spatial grid
        x = x.reshape(bsz, height, width, embed_dim)

        # Downsample width
        x = x.reshape(bsz, height, int(width / scale_factor), embed_dim * scale_factor)

        # Transpose and downsample height
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(
            bsz,
            int(width / scale_factor),
            int(height / scale_factor),
            embed_dim * (scale_factor**2),
        )

        # Transpose back and flatten
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def __call__(self, vision_features: mx.array) -> mx.array:
        """
        Project vision features to language space.

        Args:
            vision_features: Vision encoder output
                Shape: (batch_size, num_patches, vision_hidden_size)
                Example: (1, 196, 768) for SigLIP-base with 14×14 patches

        Returns:
            Projected features in language model space
                Shape: (batch_size, num_patches_reduced, language_hidden_size)
                Example: (1, 49, 576) for SmolLM2-135M after pixel shuffle

        Processing:
            1. Pixel shuffle: (1, 196, 768) → (1, 49, 3072)
               - Reduces 14×14 patches to 7×7
               - Increases channels from 768 to 3072
            2. Linear projection: (1, 49, 3072) → (1, 49, 576)
        """
        # Step 1: Apply pixel shuffle to reduce spatial dimensions
        # Input: (batch, 196, 768) - 14×14 patches
        # Output: (batch, 49, 3072) - 7×7 patches with 4× channels
        shuffled = self.pixel_shuffle(vision_features, self.pixel_shuffle_factor)

        # Step 2: Project to language hidden size
        # Input: (batch, 49, 3072)
        # Output: (batch, 49, 576)
        projected = self.proj(shuffled)

        # Return projection output directly (no normalization/scaling here)
        # Scaling will be applied in model.py before concatenation to avoid
        # disrupting pretrained weights
        return projected


def create_projection(config: ProjectionConfig) -> MLPProjection:
    """
    Create MLP projection layer.

    Args:
        config: ProjectionConfig instance

    Returns:
        MLPProjection module
    """
    return MLPProjection(config)
