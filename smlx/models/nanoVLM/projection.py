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
        intermediate_size = config.vision_hidden_size * (pixel_shuffle_factor**2)

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
        # Exact port of huggingface/nanoVLM ModalityProjector.pixel_shuffle so the
        # channel ordering matches what the trained proj Linear expects:
        #   view(b, h, w, e) -> reshape(b, h_out, s, w_out, s, e)
        #   -> permute(0, 1, 3, 2, 4, 5) -> reshape(b, h_out*w_out, e*s^2)
        # A different reshape/transpose order scrambles the channels and corrupts
        # the projection (image features become garbled).
        bsz, seq, embed_dim = x.shape
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq, f"sequence length {seq} is not a perfect square"
        assert (
            seq_root % scale_factor == 0
        ), f"grid {seq_root} not divisible by pixel-shuffle factor {scale_factor}"

        height = width = seq_root
        h_out = height // scale_factor
        w_out = width // scale_factor

        x = x.reshape(bsz, height, width, embed_dim)
        x = x.reshape(bsz, h_out, scale_factor, w_out, scale_factor, embed_dim)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(bsz, h_out * w_out, embed_dim * scale_factor**2)
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

        # Return the projected features directly. No extra scaling anywhere: the
        # trained proj Linear already maps vision features into the language
        # embedding space (matches huggingface/nanoVLM).
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
