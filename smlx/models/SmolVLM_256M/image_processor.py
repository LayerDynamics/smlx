#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Image Processor for SmolVLM-256M-Instruct.

Handles image preprocessing for the SigLIP vision encoder:
- Convert to RGB
- Resize to 384x384
- Rescale pixel values from [0, 255] to [0, 1]
- Normalize with mean=(0.5, 0.5, 0.5) and std=(0.5, 0.5, 0.5)
- Convert to MLX array format
"""

from typing import List, Union

import numpy as np
from PIL import Image


class ImageProcessor:
    """Process images for SmolVLM vision encoder.

    Preprocessing steps:
    1. Convert to RGB
    2. Resize to (384, 384)
    3. Rescale from [0, 255] to [0, 1]
    4. Normalize with mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    5. Convert to channel-first format [C, H, W]

    Args:
        image_mean: Mean for normalization (default: (0.5, 0.5, 0.5))
        image_std: Std for normalization (default: (0.5, 0.5, 0.5))
        size: Target size as (height, width) (default: (384, 384))
        rescale_factor: Rescaling factor (default: 1/255)
    """

    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        rescale_factor=1 / 255,
    ):
        self.image_mean = np.array(image_mean, dtype=np.float32).reshape(3, 1, 1)
        self.image_std = np.array(image_std, dtype=np.float32).reshape(3, 1, 1)
        self.size = size
        self.rescale_factor = rescale_factor

    def preprocess(
        self, images: Union[Image.Image, List[Image.Image]]
    ) -> List[np.ndarray]:
        """Preprocess images for vision encoder.

        Args:
            images: Single PIL Image or list of PIL Images

        Returns:
            List of preprocessed image arrays in [C, H, W] format
        """
        if isinstance(images, Image.Image):
            images = [images]

        processed_images = []
        for image in images:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize to target size
            image = image.resize(self.size, Image.BICUBIC)

            # Convert to numpy array [H, W, C]
            pixel_values = np.array(image, dtype=np.float32)

            # Convert to channel-first [C, H, W]
            pixel_values = np.transpose(pixel_values, (2, 0, 1))

            # Rescale from [0, 255] to [0, 1]
            pixel_values = pixel_values * self.rescale_factor

            # Normalize
            pixel_values = (pixel_values - self.image_mean) / self.image_std

            processed_images.append(pixel_values)

        return processed_images

    def __call__(
        self, images: Union[Image.Image, List[Image.Image]], **kwargs
    ) -> List[np.ndarray]:
        """Convenience method for preprocessing."""
        return self.preprocess(images, **kwargs)


def load_image(image_source: Union[str, Image.Image]) -> Image.Image:
    """Load image from URL or file path.

    Args:
        image_source: URL, file path, or PIL Image

    Returns:
        PIL Image in RGB format
    """
    if isinstance(image_source, Image.Image):
        return image_source

    if image_source.startswith(("http://", "https://")):
        # Load from URL
        import requests
        from io import BytesIO

        response = requests.get(image_source, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        # Load from file path
        image = Image.open(image_source)

    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


__all__ = ["ImageProcessor", "load_image"]
