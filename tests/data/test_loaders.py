"""
Tests for smlx.data.loaders module.

Basic tests to verify data loading functionality.
"""

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from smlx.data.loaders import load_image, load_text, resample_audio


class TestImageLoader:
    """Tests for image loading."""

    def test_load_from_pil_image(self):
        """Test loading from PIL Image object."""
        # Create a test image
        img = Image.new("RGB", (100, 100), color="red")

        # Load it
        loaded = load_image(img)

        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)

    def test_load_from_file(self):
        """Test loading from file path."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create and save test image
            img = Image.new("RGB", (50, 50), color="blue")
            img.save(f.name)

            # Load it
            loaded = load_image(f.name)

            assert isinstance(loaded, Image.Image)
            assert loaded.mode == "RGB"

            # Cleanup
            Path(f.name).unlink()

    def test_load_converts_to_rgb(self):
        """Test that images are converted to RGB."""
        # Create grayscale image
        img = Image.new("L", (100, 100), color=128)

        # Load it
        loaded = load_image(img)

        assert loaded.mode == "RGB"


class TestTextLoader:
    """Tests for text loading."""

    def test_load_text_file(self):
        """Test loading text from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            test_text = "Hello, world!\nThis is a test."
            f.write(test_text)
            f.flush()

            # Load text
            loaded = load_text(f.name)

            assert loaded == test_text

            # Cleanup
            Path(f.name).unlink()


class TestAudioResampler:
    """Tests for audio resampling."""

    def test_resample_audio(self):
        """Test audio resampling."""
        # Create test audio (1 second at 48kHz)
        audio_48k = np.random.randn(48000).astype(np.float32)

        # Resample to 16kHz
        audio_16k = resample_audio(audio_48k, orig_sr=48000, target_sr=16000)

        # Check shape
        assert len(audio_16k) == 16000

    def test_resample_no_change(self):
        """Test that resampling with same rate returns original."""
        audio = np.random.randn(16000).astype(np.float32)

        resampled = resample_audio(audio, orig_sr=16000, target_sr=16000)

        np.testing.assert_array_equal(audio, resampled)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
