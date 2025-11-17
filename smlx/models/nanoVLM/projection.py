#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Vision-to-Language Projection Layer.

Simple MLP that projects vision encoder outputs to language model input space.
"""

import mlx.core as mx
import mlx.nn as nn

from .config import ProjectionConfig


class MLPProjection(nn.Module):
    """
    Multi-layer perceptron for projecting vision features to language space.

    This simple connector projects SigLIP vision features (768-dim) to
    SmolLM2 language model input space (576-dim).

    Architecture:
        vision_hidden (768) -> MLP -> language_hidden (576)

    Args:
        config: ProjectionConfig with dimensions and activation
    """

    def __init__(self, config: ProjectionConfig):
        super().__init__()
        self.config = config

        # Build MLP layers
        layers = []

        # First layer
        layers.append(
            nn.Linear(config.vision_hidden_size, config.language_hidden_size)
        )

        # Hidden layers (if num_layers > 1)
        for _ in range(config.num_layers - 1):
            # Activation
            if config.activation == "gelu":
                layers.append(nn.GELU())
            elif config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "silu":
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown activation: {config.activation}")

            # Linear layer
            layers.append(
                nn.Linear(config.language_hidden_size, config.language_hidden_size)
            )

        self.mlp = nn.Sequential(*layers)

    def __call__(self, vision_features: mx.array) -> mx.array:
        """
        Project vision features to language space.

        Args:
            vision_features: Vision encoder output
                Shape: (batch_size, num_patches, vision_hidden_size)
                Example: (1, 196, 768) for SigLIP-base

        Returns:
            Projected features in language model space
                Shape: (batch_size, num_patches, language_hidden_size)
                Example: (1, 196, 576) for SmolLM2-135M

        Note:
            Each image patch becomes a token in the language model.
            For 224x224 image with 16x16 patches: 196 patches (14x14)
        """
        # Project features
        # Input: (batch, 196, 768)
        # Output: (batch, 196, 576)
        projected = self.mlp(vision_features)

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
