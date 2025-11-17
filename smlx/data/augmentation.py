"""
Data augmentation for SMLX - Image and audio augmentation transforms.

This module provides augmentation transforms for training data,
helping improve model robustness and generalization.

Adapted from:
- resources/mlx-examples/cifar/dataset.py (image augmentation)
- Common augmentation practices for vision and audio
"""

import random
from typing import Optional

import mlx.core as mx
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class ImageAugmentation:
    """
    Image augmentation pipeline for training.

    Applies random transformations to images to improve model robustness.

    Args:
        random_flip: Probability of horizontal flip (default: 0.5)
        random_crop: Whether to apply random crop (default: False)
        crop_size: Crop size if random_crop enabled
        random_brightness: Range for brightness adjustment (default: None)
        random_contrast: Range for contrast adjustment (default: None)
        random_saturation: Range for saturation adjustment (default: None)
        random_rotation: Maximum rotation degrees (default: 0)
        random_blur: Probability of Gaussian blur (default: 0.0)

    Example:
        >>> aug = ImageAugmentation(
        ...     random_flip=0.5,
        ...     random_brightness=(0.8, 1.2),
        ...     random_rotation=15
        ... )
        >>> image = Image.open("photo.jpg")
        >>> augmented = aug(image)
    """

    def __init__(
        self,
        random_flip: float = 0.5,
        random_crop: bool = False,
        crop_size: Optional[tuple[int, int]] = None,
        random_brightness: Optional[tuple[float, float]] = None,
        random_contrast: Optional[tuple[float, float]] = None,
        random_saturation: Optional[tuple[float, float]] = None,
        random_rotation: float = 0.0,
        random_blur: float = 0.0,
    ):
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast
        self.random_saturation = random_saturation
        self.random_rotation = random_rotation
        self.random_blur = random_blur

    def horizontal_flip(self, image: Image.Image) -> Image.Image:
        """Randomly flip image horizontally."""
        if random.random() < self.random_flip:
            return ImageOps.mirror(image)
        return image

    def random_crop_transform(self, image: Image.Image) -> Image.Image:
        """Apply random crop to image."""
        if not self.random_crop or self.crop_size is None:
            return image

        width, height = image.size
        crop_w, crop_h = self.crop_size

        if width < crop_w or height < crop_h:
            # Pad if image is smaller than crop size
            pad_w = max(0, crop_w - width)
            pad_h = max(0, crop_h - height)
            image = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=0)
            width, height = image.size

        # Random crop
        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        right = left + crop_w
        bottom = top + crop_h

        return image.crop((left, top, right, bottom))

    def adjust_brightness(self, image: Image.Image) -> Image.Image:
        """Randomly adjust image brightness."""
        if self.random_brightness is None:
            return image

        factor = random.uniform(*self.random_brightness)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def adjust_contrast(self, image: Image.Image) -> Image.Image:
        """Randomly adjust image contrast."""
        if self.random_contrast is None:
            return image

        factor = random.uniform(*self.random_contrast)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def adjust_saturation(self, image: Image.Image) -> Image.Image:
        """Randomly adjust image saturation."""
        if self.random_saturation is None:
            return image

        factor = random.uniform(*self.random_saturation)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def rotate(self, image: Image.Image) -> Image.Image:
        """Randomly rotate image."""
        if self.random_rotation == 0:
            return image

        angle = random.uniform(-self.random_rotation, self.random_rotation)
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))

    def blur(self, image: Image.Image) -> Image.Image:
        """Randomly apply Gaussian blur."""
        if random.random() < self.random_blur:
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        return image

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentation pipeline to image.

        Args:
            image: PIL Image to augment

        Returns:
            Augmented PIL Image
        """
        # Apply transforms in sequence
        image = self.horizontal_flip(image)
        image = self.random_crop_transform(image)
        image = self.adjust_brightness(image)
        image = self.adjust_contrast(image)
        image = self.adjust_saturation(image)
        image = self.rotate(image)
        image = self.blur(image)

        return image


class AudioAugmentation:
    """
    Audio augmentation pipeline for training.

    Applies random transformations to audio to improve model robustness.

    Args:
        add_noise: Whether to add random noise (default: False)
        noise_level: Noise level as fraction of signal (default: 0.01)
        time_stretch: Whether to apply time stretching (default: False)
        stretch_range: Range for time stretch factor (default: (0.8, 1.2))
        pitch_shift: Whether to apply pitch shifting (default: False)
        shift_steps: Range for pitch shift in semitones (default: (-2, 2))
        volume_change: Range for volume adjustment (default: None)

    Example:
        >>> aug = AudioAugmentation(
        ...     add_noise=True,
        ...     noise_level=0.01,
        ...     volume_change=(0.8, 1.2)
        ... )
        >>> audio = mx.random.randn((16000,))
        >>> augmented = aug(audio)
    """

    def __init__(
        self,
        add_noise: bool = False,
        noise_level: float = 0.01,
        time_stretch: bool = False,
        stretch_range: tuple[float, float] = (0.8, 1.2),
        pitch_shift: bool = False,
        shift_steps: tuple[int, int] = (-2, 2),
        volume_change: Optional[tuple[float, float]] = None,
    ):
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.time_stretch = time_stretch
        self.stretch_range = stretch_range
        self.pitch_shift = pitch_shift
        self.shift_steps = shift_steps
        self.volume_change = volume_change

    def add_noise_transform(self, audio: mx.array) -> mx.array:
        """Add random Gaussian noise to audio."""
        if not self.add_noise:
            return audio

        noise = mx.random.normal(shape=audio.shape) * self.noise_level
        return audio + noise

    def time_stretch_transform(self, audio: mx.array) -> mx.array:
        """Apply time stretching to audio."""
        if not self.time_stretch:
            return audio

        # Simple time stretching via resampling
        stretch_factor = random.uniform(*self.stretch_range)
        audio_np = np.array(audio)

        # Resample to stretch/compress time
        new_length = int(len(audio_np) * stretch_factor)
        indices = np.linspace(0, len(audio_np) - 1, new_length)
        stretched = np.interp(indices, np.arange(len(audio_np)), audio_np)

        return mx.array(stretched.astype(np.float32))

    def pitch_shift_transform(self, audio: mx.array) -> mx.array:
        """Apply pitch shifting to audio."""
        if not self.pitch_shift:
            return audio

        # Simple pitch shift via resampling
        # Note: This is a basic implementation; librosa would be better
        steps = random.randint(*self.shift_steps)
        factor = 2 ** (steps / 12.0)  # Semitone to frequency ratio

        audio_np = np.array(audio)
        new_length = int(len(audio_np) * factor)
        indices = np.linspace(0, len(audio_np) - 1, new_length)
        shifted = np.interp(indices, np.arange(len(audio_np)), audio_np)

        # Trim or pad to original length
        if len(shifted) > len(audio_np):
            shifted = shifted[: len(audio_np)]
        elif len(shifted) < len(audio_np):
            shifted = np.pad(shifted, (0, len(audio_np) - len(shifted)))

        return mx.array(shifted.astype(np.float32))

    def adjust_volume(self, audio: mx.array) -> mx.array:
        """Randomly adjust audio volume."""
        if self.volume_change is None:
            return audio

        factor = random.uniform(*self.volume_change)
        return audio * factor

    def __call__(self, audio: mx.array) -> mx.array:
        """
        Apply augmentation pipeline to audio.

        Args:
            audio: MLX array containing audio samples

        Returns:
            Augmented MLX audio array
        """
        # Apply transforms in sequence
        audio = self.add_noise_transform(audio)
        audio = self.time_stretch_transform(audio)
        audio = self.pitch_shift_transform(audio)
        audio = self.adjust_volume(audio)

        return audio


class Compose:
    """
    Compose multiple transforms together.

    Args:
        transforms: List of transform functions or objects

    Example:
        >>> from PIL import Image
        >>> aug1 = ImageAugmentation(random_flip=0.5)
        >>> aug2 = ImageAugmentation(random_brightness=(0.8, 1.2))
        >>> composed = Compose([aug1, aug2])
        >>> image = Image.open("photo.jpg")
        >>> augmented = composed(image)
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x):
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            x = transform(x)
        return x


class RandomApply:
    """
    Apply a transform with given probability.

    Args:
        transform: Transform to apply
        p: Probability of applying transform (default: 0.5)

    Example:
        >>> aug = ImageAugmentation(random_rotation=30)
        >>> random_aug = RandomApply(aug, p=0.5)
        >>> image = Image.open("photo.jpg")
        >>> maybe_rotated = random_aug(image)
    """

    def __init__(self, transform, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, x):
        """Apply transform with probability p."""
        if random.random() < self.p:
            return self.transform(x)
        return x


class RandomChoice:
    """
    Apply one of several transforms at random.

    Args:
        transforms: List of transforms to choose from

    Example:
        >>> aug1 = ImageAugmentation(random_flip=1.0)
        >>> aug2 = ImageAugmentation(random_rotation=30)
        >>> choice = RandomChoice([aug1, aug2])
        >>> image = Image.open("photo.jpg")
        >>> augmented = choice(image)  # Applies either flip or rotation
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x):
        """Apply one randomly chosen transform."""
        transform = random.choice(self.transforms)
        return transform(x)
