"""
Tests for smlx.data.augmentation module.

Tests augmentation transforms for images and audio.
"""

import mlx.core as mx
import pytest
from PIL import Image

from smlx.data.augmentation import (
    AudioAugmentation,
    Compose,
    ImageAugmentation,
    RandomApply,
    RandomChoice,
)


class TestImageAugmentation:
    """Tests for ImageAugmentation."""

    def test_image_augmentation_basic(self):
        """Test basic image augmentation."""
        aug = ImageAugmentation(
            random_flip=0.5, random_brightness=(0.8, 1.2), random_rotation=15
        )

        img = Image.new("RGB", (224, 224), color="red")
        result = aug(img)

        assert isinstance(result, Image.Image)
        assert result.size == img.size or result.size[0] <= img.size[0]

    def test_image_augmentation_horizontal_flip(self):
        """Test horizontal flip."""
        aug = ImageAugmentation(random_flip=1.0)  # Always flip

        # Create image with distinct left/right
        img = Image.new("RGB", (100, 100), color="white")
        # Draw a red rectangle on left half
        pixels = img.load()
        for i in range(50):
            for j in range(100):
                pixels[i, j] = (255, 0, 0)

        result = aug(img)

        # Image should be flipped (though hard to verify without pixel comparison)
        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_image_augmentation_no_flip(self):
        """Test with flip probability 0."""
        aug = ImageAugmentation(random_flip=0.0)  # Never flip

        img = Image.new("RGB", (224, 224), color="blue")
        result = aug(img)

        assert result.size == img.size

    def test_image_augmentation_brightness(self):
        """Test brightness adjustment."""
        aug = ImageAugmentation(random_brightness=(0.5, 1.5))

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = aug(img)

        assert isinstance(result, Image.Image)

    def test_image_augmentation_contrast(self):
        """Test contrast adjustment."""
        aug = ImageAugmentation(random_contrast=(0.5, 1.5))

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = aug(img)

        assert isinstance(result, Image.Image)

    def test_image_augmentation_saturation(self):
        """Test saturation adjustment."""
        aug = ImageAugmentation(random_saturation=(0.5, 1.5))

        img = Image.new("RGB", (224, 224), color="red")
        result = aug(img)

        assert isinstance(result, Image.Image)

    def test_image_augmentation_rotation(self):
        """Test rotation."""
        aug = ImageAugmentation(random_rotation=30)

        img = Image.new("RGB", (224, 224), color="green")
        result = aug(img)

        assert isinstance(result, Image.Image)

    def test_image_augmentation_blur(self):
        """Test Gaussian blur."""
        aug = ImageAugmentation(random_blur=1.0)  # Always blur

        img = Image.new("RGB", (224, 224), color="yellow")
        result = aug(img)

        assert isinstance(result, Image.Image)

    def test_image_augmentation_random_crop(self):
        """Test random crop."""
        aug = ImageAugmentation(random_crop=True, crop_size=(100, 100))

        img = Image.new("RGB", (224, 224), color="cyan")
        result = aug(img)

        # Result should be cropped to 100x100
        assert result.size == (100, 100)

    def test_image_augmentation_crop_with_padding(self):
        """Test crop on smaller image (should pad)."""
        aug = ImageAugmentation(random_crop=True, crop_size=(300, 300))

        img = Image.new("RGB", (200, 200), color="magenta")
        result = aug(img)

        # Should be padded then cropped
        assert result.size == (300, 300)

    def test_image_augmentation_combined(self):
        """Test multiple augmentations together."""
        aug = ImageAugmentation(
            random_flip=0.5,
            random_brightness=(0.8, 1.2),
            random_contrast=(0.8, 1.2),
            random_rotation=15,
        )

        img = Image.new("RGB", (224, 224), color="purple")
        result = aug(img)

        assert isinstance(result, Image.Image)


class TestAudioAugmentation:
    """Tests for AudioAugmentation."""

    def test_audio_augmentation_basic(self):
        """Test basic audio augmentation."""
        aug = AudioAugmentation(add_noise=True, noise_level=0.01)

        audio = mx.random.normal(shape=(16000,))
        result = aug(audio)

        assert isinstance(result, mx.array)
        assert result.shape == audio.shape

    def test_audio_augmentation_add_noise(self):
        """Test adding noise."""
        aug = AudioAugmentation(add_noise=True, noise_level=0.1)

        audio = mx.zeros((16000,))
        result = aug(audio)

        # Result should have noise (not all zeros)
        assert not mx.all(result == 0).item()

    def test_audio_augmentation_no_noise(self):
        """Test without noise."""
        aug = AudioAugmentation(add_noise=False)

        audio = mx.ones((16000,))
        result = aug(audio)

        # Should be unchanged
        assert mx.allclose(result, audio).item()

    def test_audio_augmentation_time_stretch(self):
        """Test time stretching."""
        aug = AudioAugmentation(time_stretch=True, stretch_range=(0.8, 1.2))

        audio = mx.random.normal(shape=(16000,))
        result = aug(audio)

        assert isinstance(result, mx.array)
        # Length might change
        assert result.size > 0

    def test_audio_augmentation_pitch_shift(self):
        """Test pitch shifting."""
        aug = AudioAugmentation(pitch_shift=True, shift_steps=(-2, 2))

        audio = mx.random.normal(shape=(16000,))
        result = aug(audio)

        assert isinstance(result, mx.array)
        # Length should be same (padded/truncated)
        assert result.shape == audio.shape

    def test_audio_augmentation_volume_change(self):
        """Test volume adjustment."""
        aug = AudioAugmentation(volume_change=(0.5, 2.0))

        audio = mx.ones((16000,)) * 0.5
        result = aug(audio)

        assert isinstance(result, mx.array)
        assert result.shape == audio.shape

    def test_audio_augmentation_combined(self):
        """Test multiple augmentations together."""
        aug = AudioAugmentation(
            add_noise=True,
            noise_level=0.01,
            volume_change=(0.8, 1.2),
        )

        audio = mx.random.normal(shape=(16000,))
        result = aug(audio)

        assert isinstance(result, mx.array)
        assert result.shape == audio.shape


class TestCompose:
    """Tests for Compose."""

    def test_compose_image_transforms(self):
        """Test composing multiple image transforms."""
        aug1 = ImageAugmentation(random_flip=0.5)
        aug2 = ImageAugmentation(random_brightness=(0.8, 1.2))

        composed = Compose([aug1, aug2])

        img = Image.new("RGB", (224, 224), color="red")
        result = composed(img)

        assert isinstance(result, Image.Image)

    def test_compose_audio_transforms(self):
        """Test composing multiple audio transforms."""
        aug1 = AudioAugmentation(add_noise=True, noise_level=0.01)
        aug2 = AudioAugmentation(volume_change=(0.8, 1.2))

        composed = Compose([aug1, aug2])

        audio = mx.random.normal(shape=(16000,))
        result = composed(audio)

        assert isinstance(result, mx.array)

    def test_compose_empty(self):
        """Test compose with empty list."""
        composed = Compose([])

        img = Image.new("RGB", (224, 224), color="blue")
        result = composed(img)

        # Should return unchanged
        assert result == img


class TestRandomApply:
    """Tests for RandomApply."""

    def test_random_apply_always(self):
        """Test with probability 1.0."""
        aug = ImageAugmentation(random_rotation=90)
        random_aug = RandomApply(aug, p=1.0)

        img = Image.new("RGB", (224, 224), color="green")
        result = random_aug(img)

        # Transform should always be applied
        assert isinstance(result, Image.Image)

    def test_random_apply_never(self):
        """Test with probability 0.0."""
        aug = ImageAugmentation(random_rotation=90)
        random_aug = RandomApply(aug, p=0.0)

        img = Image.new("RGB", (224, 224), color="yellow")
        result = random_aug(img)

        # Should return unchanged
        assert result == img


class TestRandomChoice:
    """Tests for RandomChoice."""

    def test_random_choice_image(self):
        """Test random choice between image transforms."""
        aug1 = ImageAugmentation(random_flip=1.0)
        aug2 = ImageAugmentation(random_rotation=30)
        aug3 = ImageAugmentation(random_brightness=(0.5, 1.5))

        choice = RandomChoice([aug1, aug2, aug3])

        img = Image.new("RGB", (224, 224), color="purple")
        result = choice(img)

        # One of the transforms should be applied
        assert isinstance(result, Image.Image)

    def test_random_choice_audio(self):
        """Test random choice between audio transforms."""
        aug1 = AudioAugmentation(add_noise=True, noise_level=0.1)
        aug2 = AudioAugmentation(volume_change=(0.5, 1.5))

        choice = RandomChoice([aug1, aug2])

        audio = mx.random.normal(shape=(16000,))
        result = choice(audio)

        assert isinstance(result, mx.array)

    def test_random_choice_single(self):
        """Test with single transform."""
        aug = ImageAugmentation(random_flip=1.0)
        choice = RandomChoice([aug])

        img = Image.new("RGB", (224, 224), color="cyan")
        result = choice(img)

        # Single transform should always be applied
        assert isinstance(result, Image.Image)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
