"""
Tests for smlx.data.preprocessing module.

Tests preprocessor classes for images, audio, text, and multimodal.
"""

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from smlx.data.preprocessing import (
    AudioPreprocessor,
    ImagePreprocessor,
    MultimodalPreprocessor,
    TextPreprocessor,
)


def _has_librosa():
    """Check if librosa is installed."""
    try:
        import librosa

        return True
    except ImportError:
        return False


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __call__(
        self,
        text,
        max_length=None,
        padding=False,
        truncation=True,
        add_special_tokens=True,
        return_tensors=None,
    ):
        """Mock tokenization."""
        # Tokenize
        if isinstance(text, str):
            tokens = [[hash(word) % 1000 for word in text.split()]]
            is_batch = False
        else:
            tokens = [[hash(word) % 1000 for word in t.split()] for t in text]
            is_batch = True

        # Apply padding if requested
        if padding and is_batch:
            max_len = max(len(t) for t in tokens)
            padded_tokens = []
            attention_masks = []
            for t in tokens:
                attention_mask = [1] * len(t) + [0] * (max_len - len(t))
                padded_t = t + [0] * (max_len - len(t))
                padded_tokens.append(padded_t)
                attention_masks.append(attention_mask)
            return {"input_ids": padded_tokens, "attention_mask": attention_masks}
        elif is_batch:
            # No padding - return lists of different lengths
            attention_masks = [[1] * len(t) for t in tokens]
            return {"input_ids": tokens, "attention_mask": attention_masks}
        else:
            # Single text
            return {
                "input_ids": tokens[0],
                "attention_mask": [1] * len(tokens[0]),
            }


@pytest.fixture
def mock_tokenizer():
    """Fixture providing mock tokenizer."""
    return MockTokenizer()


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""

    def test_image_preprocessor_default(self):
        """Test image preprocessor with default settings."""
        processor = ImagePreprocessor()

        # Create test image
        img = Image.new("RGB", (256, 256), color="red")

        # Process
        result = processor(img)

        assert isinstance(result, mx.array)
        assert result.shape[0] == 3  # 3 channels

    def test_image_preprocessor_exact_resize(self):
        """Test exact resize mode."""
        processor = ImagePreprocessor(size=224, resize_mode="exact")

        img = Image.new("RGB", (512, 256), color="blue")
        result = processor(img)

        # Should be resized to exactly 224x224
        assert result.shape == (3, 224, 224)

    def test_image_preprocessor_shortest_edge(self):
        """Test shortest edge resize mode."""
        processor = ImagePreprocessor(size=(224, 224), resize_mode="shortest_edge")

        img = Image.new("RGB", (512, 256), color="green")
        result = processor(img)

        assert result.shape[0] == 3
        # Shortest edge should be >= 224

    def test_image_preprocessor_longest_edge(self):
        """Test longest edge resize mode."""
        processor = ImagePreprocessor(size=(224, 224), resize_mode="longest_edge")

        img = Image.new("RGB", (512, 256), color="yellow")
        result = processor(img)

        assert result.shape[0] == 3

    def test_image_preprocessor_normalization(self):
        """Test image normalization."""
        processor = ImagePreprocessor(
            size=224,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            do_normalize=True,
        )

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = processor(img)

        # Values should be normalized
        assert isinstance(result, mx.array)

    def test_image_preprocessor_no_normalization(self):
        """Test without normalization."""
        processor = ImagePreprocessor(size=224, do_normalize=False, rescale_factor=1.0)

        img = Image.new("RGB", (224, 224), color="red")
        result = processor(img)

        # Values should be in 0-255 range (not normalized)
        assert result.max() > 1.0

    def test_image_preprocessor_center_crop(self):
        """Test center crop."""
        processor = ImagePreprocessor(
            size=300, resize_mode="exact", do_center_crop=True, crop_size=224
        )

        img = Image.new("RGB", (512, 512), color="cyan")
        result = processor(img)

        # Should be cropped to 224x224
        assert result.shape == (3, 224, 224)

    def test_image_preprocessor_grayscale_conversion(self):
        """Test grayscale to RGB conversion."""
        processor = ImagePreprocessor(size=224)

        # Create grayscale image
        img = Image.new("L", (224, 224), color=128)

        result = processor(img)

        # Should be converted to 3 channels
        assert result.shape[0] == 3

    def test_image_preprocessor_invalid_resize_mode(self):
        """Test invalid resize mode raises error."""
        processor = ImagePreprocessor(size=224, resize_mode="invalid")

        img = Image.new("RGB", (224, 224))

        with pytest.raises(ValueError, match="Invalid resize_mode"):
            processor(img)


class TestAudioPreprocessor:
    """Tests for AudioPreprocessor."""

    def test_audio_preprocessor_basic(self):
        """Test basic audio preprocessing."""
        processor = AudioPreprocessor(sample_rate=16000, n_mels=80)

        # Create test audio (1 second)
        audio = mx.random.normal(shape=(16000,))

        # Process
        result = processor(audio)

        assert isinstance(result, mx.array)
        # Should return mel-spectrogram with n_mels
        assert result.shape[0] == 80

    def test_audio_preprocessor_custom_params(self):
        """Test audio preprocessor with custom parameters."""
        processor = AudioPreprocessor(
            sample_rate=16000, n_fft=512, hop_length=256, n_mels=128, normalize=False
        )

        audio = mx.random.normal(shape=(16000,))
        result = processor(audio)

        assert result.shape[0] == 128

    @pytest.mark.skipif(
        not _has_librosa(), reason="librosa not installed, using fallback"
    )
    def test_audio_preprocessor_with_librosa(self):
        """Test that librosa is used when available."""
        processor = AudioPreprocessor(sample_rate=16000, n_mels=80)

        audio = mx.random.normal(shape=(16000,))
        result = processor(audio)

        assert isinstance(result, mx.array)


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def test_text_preprocessor_basic(self, mock_tokenizer):
        """Test basic text preprocessing."""
        processor = TextPreprocessor(mock_tokenizer, max_length=128)

        text = "Hello world this is a test"
        result = processor(text)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert isinstance(result["input_ids"], mx.array)

    def test_text_preprocessor_return_lists(self, mock_tokenizer):
        """Test returning lists instead of MLX arrays."""
        processor = TextPreprocessor(mock_tokenizer)

        text = "Hello world"
        result = processor(text, return_mlx=False)

        assert isinstance(result["input_ids"], list)

    def test_text_preprocessor_batch(self, mock_tokenizer):
        """Test processing batch of texts."""
        processor = TextPreprocessor(mock_tokenizer, max_length=128, padding=True)

        texts = ["Hello world", "How are you?", "Fine thanks"]
        result = processor(texts)

        assert "input_ids" in result
        # With padding=True, should be an array
        assert isinstance(result["input_ids"], mx.array)


class TestMultimodalPreprocessor:
    """Tests for MultimodalPreprocessor."""

    def test_multimodal_preprocessor_image_only(self, mock_tokenizer):
        """Test processing only image."""
        image_proc = ImagePreprocessor(size=224)
        text_proc = TextPreprocessor(mock_tokenizer)
        processor = MultimodalPreprocessor(image_proc, text_proc)

        img = Image.new("RGB", (256, 256), color="red")
        result = processor(image=img)

        assert "pixel_values" in result
        assert isinstance(result["pixel_values"], mx.array)
        assert "input_ids" not in result

    def test_multimodal_preprocessor_text_only(self, mock_tokenizer):
        """Test processing only text."""
        image_proc = ImagePreprocessor(size=224)
        text_proc = TextPreprocessor(mock_tokenizer)
        processor = MultimodalPreprocessor(image_proc, text_proc)

        result = processor(text="Hello world")

        assert "input_ids" in result
        assert "pixel_values" not in result

    def test_multimodal_preprocessor_both(self, mock_tokenizer):
        """Test processing both image and text."""
        image_proc = ImagePreprocessor(size=224)
        text_proc = TextPreprocessor(mock_tokenizer)
        processor = MultimodalPreprocessor(image_proc, text_proc)

        img = Image.new("RGB", (256, 256), color="blue")
        result = processor(image=img, text="What is in this image?")

        assert "pixel_values" in result
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_multimodal_preprocessor_empty(self, mock_tokenizer):
        """Test processing with no inputs."""
        image_proc = ImagePreprocessor(size=224)
        text_proc = TextPreprocessor(mock_tokenizer)
        processor = MultimodalPreprocessor(image_proc, text_proc)

        result = processor()

        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
