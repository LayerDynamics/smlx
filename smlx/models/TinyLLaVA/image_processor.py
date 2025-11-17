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
    """Process images for TinyLLaVA vision encoder.

    Preprocessing steps:
    1. Convert to RGB
    2. Resize to (384, 384)
    3. Rescale from [0, 255] to [0, 1]
    4. Normalize with mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    5. Keep in MLX channel-last format [H, W, C]

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
        image_size=None,  # Alias for size for backwards compatibility
    ):
        # Handle VisionConfig object being passed as first argument
        if hasattr(image_mean, "image_size") and not isinstance(image_mean, (list, tuple)):
            # image_mean is actually a VisionConfig object
            vision_config = image_mean
            # Extract image_size from config, use default normalization for SigLIP
            config_image_size = vision_config.image_size
            image_mean = (0.5, 0.5, 0.5)  # SigLIP default
            image_std = (0.5, 0.5, 0.5)   # SigLIP default
            size = (config_image_size, config_image_size)
            image_size = config_image_size
            rescale_factor = 1 / 255

        # MLX uses channel-last format [H, W, C], so reshape to [1, 1, 3]
        self.image_mean = np.array(image_mean, dtype=np.float32).reshape(1, 1, 3)
        self.image_std = np.array(image_std, dtype=np.float32).reshape(1, 1, 3)
        # Support both size and image_size parameters
        if image_size is not None:
            self.size = (image_size, image_size) if isinstance(image_size, int) else image_size
            self.image_size = image_size  # Store for compatibility
        else:
            self.size = size
            self.image_size = size[0] if isinstance(size, tuple) else size  # Extract scalar for compatibility
        self.rescale_factor = rescale_factor

    def preprocess(
        self, images: Union[Image.Image, List[Image.Image]]
    ) -> List[np.ndarray]:
        """Preprocess images for vision encoder.

        Args:
            images: Single PIL Image or list of PIL Images

        Returns:
            List of preprocessed image arrays in MLX [H, W, C] format
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

            # Convert to numpy array [H, W, C] (MLX channel-last format)
            pixel_values = np.array(image, dtype=np.float32)

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
