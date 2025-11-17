#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Region modules for Moondream2 object detection and pointing.

Handles encoding and decoding of spatial coordinates and bounding boxes
using Fourier features for continuous coordinate representation.
"""

from typing import List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import RegionConfig


class FourierFeatures(nn.Module):
    """Fourier feature encoding for continuous coordinates.

    Maps continuous coordinates (x, y) to high-dimensional features
    for better neural network processing.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        # Random Fourier features
        self.freqs = mx.random.normal((dim // 2, 2)) * np.pi

    def __call__(self, coords: mx.array) -> mx.array:
        """
        Args:
            coords: Coordinates [B, 2] where coords are in range [0, 1]

        Returns:
            Fourier features [B, dim]
        """
        # coords @ freqs^T gives [B, dim/2]
        proj = mx.matmul(coords, self.freqs.T)

        # Concatenate sin and cos
        features = mx.concatenate([mx.sin(proj), mx.cos(proj)], axis=-1)

        return features


class CoordinateEncoder(nn.Module):
    """Encodes spatial coordinates into embeddings."""

    def __init__(self, config: RegionConfig, hidden_size: int = 2048):
        super().__init__()
        self.use_fourier = config.use_fourier_features

        if self.use_fourier:
            self.fourier = FourierFeatures(config.fourier_feature_dim)
            self.projection = nn.Linear(config.fourier_feature_dim, hidden_size)
        else:
            # Simple linear encoding
            self.projection = nn.Linear(2, hidden_size)

    def __call__(self, coords: mx.array) -> mx.array:
        """
        Args:
            coords: Normalized coordinates [B, 2] in range [0, 1]

        Returns:
            Coordinate embeddings [B, hidden_size]
        """
        if self.use_fourier:
            features = self.fourier(coords)
            embeddings = self.projection(features)
        else:
            embeddings = self.projection(coords)

        return embeddings


class CoordinateDecoder(nn.Module):
    """Decodes embeddings into spatial coordinates."""

    def __init__(self, hidden_size: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 2)  # Output (x, y)
        self.activation = nn.GELU()

    def __call__(self, embeddings: mx.array) -> mx.array:
        """
        Args:
            embeddings: Hidden state embeddings [B, hidden_size]

        Returns:
            Predicted coordinates [B, 2] in range [0, 1]
        """
        x = self.fc1(embeddings)
        x = self.activation(x)
        coords = self.fc2(x)

        # Sigmoid to ensure [0, 1] range
        coords = mx.sigmoid(coords)

        return coords


class BoxEncoder(nn.Module):
    """Encodes bounding boxes (x1, y1, x2, y2) into embeddings."""

    def __init__(self, config: RegionConfig, hidden_size: int = 2048):
        super().__init__()
        # Encode two coordinate pairs
        self.coord_encoder = CoordinateEncoder(config, hidden_size // 2)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def __call__(self, boxes: mx.array) -> mx.array:
        """
        Args:
            boxes: Bounding boxes [B, 4] (x1, y1, x2, y2) normalized to [0, 1]

        Returns:
            Box embeddings [B, hidden_size]
        """
        # Split into two coordinate pairs
        top_left = boxes[:, :2]  # (x1, y1)
        bottom_right = boxes[:, 2:]  # (x2, y2)

        # Encode each pair
        tl_emb = self.coord_encoder(top_left)
        br_emb = self.coord_encoder(bottom_right)

        # Concatenate and project
        box_emb = mx.concatenate([tl_emb, br_emb], axis=-1)
        box_emb = self.projection(box_emb)

        return box_emb


class BoxDecoder(nn.Module):
    """Decodes embeddings into bounding boxes."""

    def __init__(self, hidden_size: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 4)  # Output (x1, y1, x2, y2)
        self.activation = nn.GELU()

    def __call__(self, embeddings: mx.array) -> mx.array:
        """
        Args:
            embeddings: Hidden state embeddings [B, hidden_size]

        Returns:
            Predicted boxes [B, 4] (x1, y1, x2, y2) in range [0, 1]
        """
        x = self.fc1(embeddings)
        x = self.activation(x)
        boxes = self.fc2(x)

        # Sigmoid to ensure [0, 1] range
        boxes = mx.sigmoid(boxes)

        # Ensure x2 > x1 and y2 > y1
        x1, y1, x2, y2 = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
        x2 = mx.maximum(x2, x1 + 0.01)  # Minimum box width
        y2 = mx.maximum(y2, y1 + 0.01)  # Minimum box height

        boxes = mx.concatenate([x1, y1, x2, y2], axis=-1)

        return boxes


class DetectionHead(nn.Module):
    """Detection head for object detection with confidence scores."""

    def __init__(self, hidden_size: int = 2048, max_detections: int = 100):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_detections = max_detections

        # Bounding box decoder
        self.box_decoder = BoxDecoder(hidden_size)

        # Confidence score predictor
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def __call__(
        self, embeddings: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            embeddings: Hidden states [B, seq_len, hidden_size]

        Returns:
            boxes: Predicted boxes [B, max_detections, 4]
            confidences: Confidence scores [B, max_detections]
        """
        # Take last max_detections tokens
        if embeddings.shape[1] > self.max_detections:
            embeddings = embeddings[:, -self.max_detections :, :]

        # Predict boxes and confidences
        boxes = self.box_decoder(embeddings)  # [B, N, 4]
        confidences = self.confidence(embeddings).squeeze(-1)  # [B, N]
        confidences = mx.sigmoid(confidences)

        return boxes, confidences


def parse_coordinates_from_text(
    text: str, image_size: Tuple[int, int]
) -> Optional[List[Tuple[int, int]]]:
    """Parse coordinate tokens from generated text.

    Looks for patterns like: <|coordinate|>0.5,0.3</|coordinate|>

    Args:
        text: Generated text containing coordinate tokens
        image_size: Image (width, height) for denormalization

    Returns:
        List of (x, y) pixel coordinates, or None if no coordinates found
    """
    import re

    # Find all coordinate patterns
    pattern = r"<\|coordinate\|>([\d.]+),([\d.]+)</\|coordinate\|>"
    matches = re.findall(pattern, text)

    if not matches:
        return None

    width, height = image_size
    coords = []

    for x_str, y_str in matches:
        x_norm = float(x_str)
        y_norm = float(y_str)

        # Denormalize to pixel coordinates
        x_pixel = int(x_norm * width)
        y_pixel = int(y_norm * height)

        coords.append((x_pixel, y_pixel))

    return coords


def parse_boxes_from_text(
    text: str, image_size: Tuple[int, int]
) -> Optional[List[Tuple[int, int, int, int]]]:
    """Parse bounding box tokens from generated text.

    Looks for patterns like: <|grounding|>0.1,0.2,0.5,0.6</|grounding|>

    Args:
        text: Generated text containing box tokens
        image_size: Image (width, height) for denormalization

    Returns:
        List of (x1, y1, x2, y2) pixel boxes, or None if no boxes found
    """
    import re

    # Find all grounding patterns
    pattern = r"<\|grounding\|>([\d.]+),([\d.]+),([\d.]+),([\d.]+)</\|grounding\|>"
    matches = re.findall(pattern, text)

    if not matches:
        return None

    width, height = image_size
    boxes = []

    for x1_str, y1_str, x2_str, y2_str in matches:
        x1_norm, y1_norm = float(x1_str), float(y1_str)
        x2_norm, y2_norm = float(x2_str), float(y2_str)

        # Denormalize to pixel coordinates
        x1 = int(x1_norm * width)
        y1 = int(y1_norm * height)
        x2 = int(x2_norm * width)
        y2 = int(y2_norm * height)

        boxes.append((x1, y1, x2, y2))

    return boxes


__all__ = [
    "FourierFeatures",
    "CoordinateEncoder",
    "CoordinateDecoder",
    "BoxEncoder",
    "BoxDecoder",
    "DetectionHead",
    "parse_coordinates_from_text",
    "parse_boxes_from_text",
]
