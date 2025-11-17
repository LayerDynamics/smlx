"""
Vision utilities for SMLX - Image loading, preprocessing, and batch processing.

This module provides common image processing utilities for vision-language models,
including loading images from various sources and preprocessing pipelines.
"""

from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np
import requests
from PIL import Image, ImageOps


def load_image(
    image_source: Union[str, Path, BytesIO, Image.Image], timeout: int = 10
) -> Image.Image:
    """
    Load an image from a file path, URL, BytesIO, base64 data URI, or PIL Image.

    Args:
        image_source: Source of the image. Can be:
            - File path (str or Path)
            - URL (str starting with http:// or https://)
            - BytesIO object
            - Base64 data URI (str starting with data:image/)
            - PIL Image object (returned as-is after conversion to RGB)
        timeout: Timeout in seconds for HTTP requests (default: 10)

    Returns:
        PIL Image in RGB format with EXIF orientation applied

    Raises:
        ValueError: If the image source is invalid or cannot be loaded
    """
    # If already a PIL Image, just convert to RGB and return
    if isinstance(image_source, Image.Image):
        image = ImageOps.exif_transpose(image_source)
        return image.convert("RGB")

    # Handle BytesIO, base64 data URIs, or file paths
    if (
        isinstance(image_source, BytesIO)
        or (isinstance(image_source, str) and image_source.startswith("data:image/"))
        or Path(image_source).is_file()
    ):
        try:
            # Handle base64 encoded data URIs
            if isinstance(image_source, str) and image_source.startswith("data:image/"):
                import base64

                if "," not in image_source:
                    raise ValueError("Invalid data URI format - missing comma separator")

                _, data = image_source.split(",", 1)
                image_source = BytesIO(base64.b64decode(data))

            image = Image.open(image_source)
        except OSError as e:
            raise ValueError(
                f"Failed to load image from {image_source} with error: {e}"
            ) from e

    # Handle HTTP(S) URLs
    elif isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True, timeout=timeout)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            raise ValueError(
                f"Failed to load image from URL: {image_source} with error {e}"
            ) from e
    else:
        raise ValueError(
            f"The image {image_source} must be a valid URL, file path, BytesIO, "
            f"base64 data URI, or PIL Image."
        )

    # Apply EXIF orientation and convert to RGB
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def resize_image(
    image: Image.Image, max_size: tuple[int, int], keep_aspect_ratio: bool = True
) -> Image.Image:
    """
    Resize an image to fit within max_size while optionally preserving aspect ratio.

    Args:
        image: PIL Image to resize
        max_size: Maximum (width, height) tuple
        keep_aspect_ratio: If True, preserve aspect ratio (default: True)

    Returns:
        Resized PIL Image
    """
    if keep_aspect_ratio:
        ratio = min(max_size[0] / image.width, max_size[1] / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
    else:
        new_size = max_size

    return image.resize(new_size, Image.Resampling.BICUBIC)


def preprocess_image(
    image: Image.Image,
    target_size: Optional[tuple[int, int]] = None,
    resize_mode: str = "shortest_edge",
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
    rescale_factor: float = 1.0 / 255.0,
) -> mx.array:
    """
    Preprocess an image for model input with standard vision transformations.

    Standard preprocessing pipeline:
    1. Resize (if target_size provided)
    2. Convert to numpy array [H, W, C]
    3. Transpose to [C, H, W]
    4. Rescale pixel values (default: [0, 255] → [0, 1])
    5. Normalize with mean and std (if provided)
    6. Convert to MLX array

    Args:
        image: PIL Image to preprocess
        target_size: Target (width, height) for resizing, or None to skip resize
        resize_mode: Resize mode - "shortest_edge", "longest_edge", or "exact"
        mean: Mean values for normalization per channel (default: None, no normalization)
        std: Standard deviation values for normalization per channel (default: None)
        rescale_factor: Factor to rescale pixel values (default: 1/255)

    Returns:
        MLX array of shape [C, H, W] with preprocessed image
    """
    # Resize if target size provided
    if target_size is not None:
        if resize_mode == "shortest_edge":
            # Resize so shortest edge matches target, preserving aspect ratio
            scale = max(target_size[0] / image.width, target_size[1] / image.height)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.BICUBIC)
        elif resize_mode == "longest_edge":
            # Resize so longest edge matches target, preserving aspect ratio
            scale = min(target_size[0] / image.width, target_size[1] / image.height)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.BICUBIC)
        elif resize_mode == "exact":
            # Resize to exact dimensions, ignoring aspect ratio
            image = image.resize(target_size, Image.Resampling.BICUBIC)
        else:
            raise ValueError(
                f"Invalid resize_mode '{resize_mode}'. "
                f"Must be 'shortest_edge', 'longest_edge', or 'exact'"
            )

    # Convert to numpy array [H, W, C]
    img_array = np.array(image)

    # Transpose to [C, H, W]
    img_array = img_array.transpose(2, 0, 1)

    # Rescale pixel values (typically [0, 255] → [0, 1])
    if rescale_factor != 1.0:
        img_array = img_array.astype(np.float32) * rescale_factor
    else:
        img_array = img_array.astype(np.float32)

    # Normalize with mean and std if provided
    if mean is not None:
        mean_array = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        img_array = img_array - mean_array

    if std is not None:
        std_array = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
        img_array = img_array / std_array

    # Convert to MLX array
    return mx.array(img_array)


def batch_images(images: list[mx.array], padding_value: float = 0.0) -> mx.array:
    """
    Batch multiple images into a single tensor with padding.

    Images are padded to match the largest width and height in the batch.

    Args:
        images: List of MLX arrays with shape [C, H, W]
        padding_value: Value to use for padding (default: 0.0)

    Returns:
        MLX array of shape [B, C, H_max, W_max] where B is batch size
    """
    if not images:
        raise ValueError("Cannot batch empty list of images")

    # Get max dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    num_channels = images[0].shape[0]

    # Create batch tensor
    batch_size = len(images)
    batched = mx.full((batch_size, num_channels, max_h, max_w), padding_value)

    # Copy each image into the batch tensor
    for i, img in enumerate(images):
        c, h, w = img.shape
        batched[i, :, :h, :w] = img

    return batched


def expand2square(image: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    """
    Expand an image to a square by adding padding.

    Useful for models that expect square inputs. Padding is added symmetrically
    to maintain the image in the center.

    Args:
        image: PIL Image to expand
        background_color: RGB tuple for padding color (e.g., (255, 255, 255) for white)

    Returns:
        Square PIL Image with padding
    """
    width, height = image.size
    if width == height:
        return image

    max_side = max(width, height)
    result = Image.new(image.mode, (max_side, max_side), background_color)

    # Center the original image
    offset = ((max_side - width) // 2, (max_side - height) // 2)
    result.paste(image, offset)

    return result


def center_crop(image: Image.Image, crop_size: tuple[int, int]) -> Image.Image:
    """
    Perform center crop on an image.

    Args:
        image: PIL Image to crop
        crop_size: (width, height) of the crop

    Returns:
        Cropped PIL Image
    """
    width, height = image.size
    crop_w, crop_h = crop_size

    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    return image.crop((left, top, right, bottom))


def load_and_preprocess_image(
    image_source: Union[str, Path, BytesIO, Image.Image],
    target_size: Optional[tuple[int, int]] = (224, 224),
    resize_mode: str = "shortest_edge",
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
    rescale_factor: float = 1.0 / 255.0,
) -> mx.array:
    """
    Convenience function to load and preprocess an image in one step.

    Combines load_image() and preprocess_image() for common workflows.

    Args:
        image_source: Source of the image (file path, URL, BytesIO, etc.)
        target_size: Target (width, height) for resizing
        resize_mode: Resize mode - "shortest_edge", "longest_edge", or "exact"
        mean: Mean values for normalization per channel
        std: Standard deviation values for normalization per channel
        rescale_factor: Factor to rescale pixel values (default: 1/255)

    Returns:
        MLX array of shape [C, H, W] with preprocessed image
    """
    image = load_image(image_source)
    return preprocess_image(image, target_size, resize_mode, mean, std, rescale_factor)
