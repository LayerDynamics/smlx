#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Idefics3 Connector for SmolVLM-256M-Instruct.

The connector bridges vision and language modalities:
1. Pixel shuffle: Reduces spatial dimensions while increasing channel depth
2. MLP projection: Projects from vision hidden size to language hidden size

For SmolVLM-256M:
- Input: [B, 729, 1152] (27x27 patches from SigLIP at hidden_size=1152)
- After pixel shuffle (scale=2): [B, 182, 4608] (floor(27/2)^2=182, 1152*4=4608)
- After MLP: [B, 182, 576] (SmolLM2 hidden_size=576)
"""

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


class MLP(nn.Module):
    """Simple MLP projection layer without bias."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class Idefics3Connector(nn.Module):
    """Connector that transforms vision features to language model space.

    Uses pixel shuffle to reduce spatial dimensions while increasing channels,
    then projects to language model hidden size.

    Args:
        config: ModelConfig with scale_factor and hidden sizes

    Example:
        Input: [1, 729, 1152] (27x27 patches, 1152 vision hidden size)
        After pixel_shuffle(scale=2):
            - Reshape to [1, 27, 27, 1152]
            - Downsample spatial by 2x: [1, 13, 13, 4608] (13=floor(27/2), 4608=1152*4)
            - Flatten: [1, 169, 4608]
        After MLP: [1, 169, 576] (576 language hidden size)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = MLP(config)

    def pixel_shuffle(self, x: mx.array, scale_factor: int = 2) -> mx.array:
        """Pixel shuffle operation to reduce spatial dimensions.

        Reduces spatial resolution by scale_factor while increasing channel depth
        by scale_factor^2.

        Args:
            x: Input tensor [B, seq, embed_dim] where seq = height * width
            scale_factor: Downsampling factor (default: 2)

        Returns:
            Downsampled tensor [B, seq/(scale_factor^2), embed_dim * (scale_factor^2)]
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

    def __call__(self, image_hidden_states: mx.array) -> mx.array:
        """Transform vision features to language space.

        Args:
            image_hidden_states: Vision features [B, num_patches, vision_hidden_size]

        Returns:
            Language-compatible features [B, num_patches', text_hidden_size]
        """
        # Reduce spatial dimensions
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)

        # Project to language model dimensions
        image_hidden_states = self.modality_projection(image_hidden_states)

        return image_hidden_states


__all__ = ["Idefics3Connector", "MLP"]
