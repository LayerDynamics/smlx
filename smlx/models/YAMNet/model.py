#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
YAMNet Model Architecture.

MobileNet-v1 based audio classifier with depthwise-separable
convolutions for efficiency. Trained on AudioSet with 521 classes.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import YAMNetConfig, DEFAULT_CONFIG


class DepthwiseSeparableConv2D(nn.Module):
    """Depthwise separable convolution.

    Efficient convolution that factors a standard convolution into
    a depthwise convolution and a pointwise (1x1) convolution.

    This reduces parameters and computation while maintaining
    similar representational power.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
        depth_multiplier: float = 1.0,
    ):
        """Initialize depthwise separable convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolving kernel
            stride: Stride of convolution
            padding: Zero-padding added to input
            depth_multiplier: Multiplier for depthwise channels
        """
        super().__init__()

        # Depthwise convolution (one filter per input channel)
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * depth_multiplier),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Pointwise convolution (1x1 conv to combine channels)
        self.pointwise = nn.Conv2d(
            in_channels=int(in_channels * depth_multiplier),
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )

        # Batch normalization for each conv
        self.bn_depthwise = nn.BatchNorm(int(in_channels * depth_multiplier))
        self.bn_pointwise = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, height, width, channels)

        Returns:
            Output tensor
        """
        # Depthwise
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = nn.relu(x)

        # Pointwise
        x = self.pointwise(x)
        x = self.bn_pointwise(x)
        x = nn.relu(x)

        return x


class YAMNet(nn.Module):
    """YAMNet audio classifier.

    MobileNet-v1 architecture for audio event classification.
    Processes mel spectrogram patches and outputs probabilities
    for 521 AudioSet classes.

    Architecture:
    - Input: (batch, 96, 64, 1) mel spectrogram patches
    - Conv layers with depthwise separable convolutions
    - Global average pooling
    - Dense layers for classification
    - Output: (batch, 521) class probabilities
    """

    def __init__(self, config: YAMNetConfig = DEFAULT_CONFIG):
        """Initialize YAMNet model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Initial standard convolution
        self.conv1 = nn.Conv2d(
            in_channels=1,  # Single channel (mel spectrogram)
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.bn1 = nn.BatchNorm(32)

        # Depthwise separable convolution blocks
        # Following MobileNet-v1 architecture
        self.blocks = [
            # (in_channels, out_channels, stride)
            (32, 64, (1, 1)),
            (64, 128, (2, 2)),
            (128, 128, (1, 1)),
            (128, 256, (2, 2)),
            (256, 256, (1, 1)),
            (256, 512, (2, 2)),
            # 5x 512 -> 512 blocks
            (512, 512, (1, 1)),
            (512, 512, (1, 1)),
            (512, 512, (1, 1)),
            (512, 512, (1, 1)),
            (512, 512, (1, 1)),
            (512, 1024, (2, 2)),
            (1024, 1024, (1, 1)),
        ]

        self.conv_blocks = []
        for in_ch, out_ch, stride in self.blocks:
            block = DepthwiseSeparableConv2D(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                depth_multiplier=config.depth_multiplier,
            )
            self.conv_blocks.append(block)

        # Embedding layer (before final classification)
        self.embedding = nn.Linear(1024, config.embedding_size)

        # Classification head
        self.classifier = nn.Linear(config.embedding_size, config.num_classes)

    def extract_features(self, x: mx.array) -> mx.array:
        """Extract convolutional features.

        Args:
            x: Input patches (batch, height, width, channels)

        Returns:
            Feature maps before global pooling
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)

        # Depthwise separable blocks
        for block in self.conv_blocks:
            x = block(x)

        return x

    def extract_embeddings(self, x: mx.array) -> mx.array:
        """Extract audio embeddings.

        Args:
            x: Input patches (batch, height, width, channels)

        Returns:
            Audio embeddings (batch, embedding_size)
        """
        # Convolutional features
        features = self.extract_features(x)

        # Global average pooling
        # features shape: (batch, height, width, channels)
        pooled = mx.mean(features, axis=(1, 2))  # (batch, channels)

        # Embedding layer
        embeddings = self.embedding(pooled)

        return embeddings

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass for classification.

        Args:
            x: Input patches (batch, height, width, channels)
               Expected shape: (batch, 96, 64, 1)

        Returns:
            Logits (batch, num_classes)
        """
        # Get embeddings
        embeddings = self.extract_embeddings(x)

        # Classification
        logits = self.classifier(embeddings)

        return logits

    def predict_proba(self, x: mx.array) -> mx.array:
        """Predict class probabilities.

        Args:
            x: Input patches

        Returns:
            Probabilities (batch, num_classes)
        """
        logits = self(x)
        probs = mx.softmax(logits, axis=-1)
        return probs


def count_parameters(model: YAMNet) -> int:
    """Count total parameters in model.

    Args:
        model: YAMNet model

    Returns:
        Total number of parameters
    """
    total = 0
    for name, param in model.parameters().items():
        if hasattr(param, "size"):
            total += param.size
    return total


__all__ = [
    "YAMNet",
    "DepthwiseSeparableConv2D",
    "count_parameters",
]
