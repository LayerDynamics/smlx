#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Image Processor for nanoVLM.

Handles image loading, resizing, and normalization for SigLIP vision encoder.
"""

from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np
from PIL import Image


class ImageProcessor:
    """
    Process images for nanoVLM vision encoder.

    Handles:
    - Loading images from files or URLs
    - Resizing to 224x224
    - Normalization for SigLIP

    Args:
        image_size: Target image size (default: 224)
        image_mean: Mean for normalization (default: SigLIP mean)
        image_std: Std for normalization (default: SigLIP std)
    """

    def __init__(
        self,
        image_size: int = 224,
        image_mean: tuple = (0.5, 0.5, 0.5),
        image_std: tuple = (0.5, 0.5, 0.5),
    ):
        self.image_size = image_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)

    def __call__(self, image: Union[str, Path, Image.Image]) -> mx.array:
        """
        Process image for model input.

        Args:
            image: Image file path, URL, or PIL Image

        Returns:
            Processed image tensor
                Shape: (1, 224, 224, 3) - NHWC format for MLX
                Values: Normalized [-1, 1] range

        Example:
            >>> processor = ImageProcessor()
            >>> image_tensor = processor("photo.jpg")
            >>> print(image_tensor.shape)  # (1, 224, 224, 3)
        """
        # Load image if path/URL
        if isinstance(image, (str, Path)):
            image = load_image(image)

        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")

        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array and normalize
        image_np = np.array(image, dtype=np.float32) / 255.0

        # Normalize with mean and std
        image_np = (image_np - self.image_mean) / self.image_std

        # Convert to MLX array (keep HWC format for MLX Conv2d)
        # Shape: (224, 224, 3) - MLX expects channel-last format
        image_mx = mx.array(image_np)

        # Add batch dimension
        # Shape: (224, 224, 3) -> (1, 224, 224, 3)
        image_mx = mx.expand_dims(image_mx, axis=0)

        return image_mx

    def batch_process(self, images: list) -> mx.array:
        """
        Process multiple images in a batch.

        Args:
            images: List of image paths, URLs, or PIL Images

        Returns:
            Batch of processed images
                Shape: (batch_size, 224, 224, 3) - NHWC format for MLX

        Example:
            >>> processor = ImageProcessor()
            >>> images = ["photo1.jpg", "photo2.jpg"]
            >>> batch = processor.batch_process(images)
            >>> print(batch.shape)  # (2, 224, 224, 3)
        """
        processed = [self(img) for img in images]
        return mx.concatenate(processed, axis=0)


def load_image(image_path: Union[str, Path, Image.Image]) -> Image.Image:
    """
    Load image from file, URL, or PIL Image.

    Args:
        image_path: Path to image file, URL, or PIL Image

    Returns:
        PIL Image

    Example:
        >>> image = load_image("photo.jpg")
        >>> image = load_image("https://example.com/photo.jpg")
        >>> image = load_image(pil_image)  # Returns as-is
    """
    # If already a PIL Image, return it
    if isinstance(image_path, Image.Image):
        return image_path

    image_path = str(image_path)

    # Check if URL
    if image_path.startswith(("http://", "https://")):
        import requests
        from io import BytesIO

        response = requests.get(image_path)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        # Load from file
        image = Image.open(image_path)

    return image


def create_image_processor(image_size: int = 224) -> ImageProcessor:
    """
    Create image processor with default settings.

    Args:
        image_size: Target image size (default: 224)

    Returns:
        ImageProcessor instance
    """
    return ImageProcessor(
        image_size=image_size,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
