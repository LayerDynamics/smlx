#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for TrOCR-small optical character recognition.

Tests text recognition for printed and handwritten text.

Run with:
    python -m pytest tests/integration/test_trocr_small.py -v
"""

import gc

import mlx.core as mx
import pytest
from PIL import Image, ImageDraw, ImageFont

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
]


def create_text_image(text="Hello World", size=(400, 100)):
    """Create a simple image with text."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fallback to basic font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except:
        font = ImageFont.load_default()

    # Calculate text position (center)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    # Draw text
    draw.text((x, y), text, fill="black", font=font)

    return img


@pytest.fixture(scope="module")
def trocr_printed_model():
    """
    Load TrOCR printed text model once for all tests.

    Memory Requirements:
    - Model size: ~200MB (100M parameters in FP16)
    - Peak memory: ~400MB with activations
    """
    from smlx.models.TrOCR_small import load

    model, processor = load("printed")

    yield model, processor

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up TrOCR printed model...")
    del model
    del processor
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print("Cleanup complete. Memory freed.")


@pytest.fixture(scope="module")
def trocr_handwritten_model():
    """
    Load TrOCR handwritten text model once for all tests.

    Memory Requirements:
    - Model size: ~200MB (100M parameters in FP16)
    - Peak memory: ~400MB with activations
    """
    from smlx.models.TrOCR_small import load

    try:
        model, processor = load("handwritten")
    except Exception as e:
        pytest.skip(f"Handwritten model not available: {e}")

    yield model, processor

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up TrOCR handwritten model...")
    del model
    del processor
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print("Cleanup complete. Memory freed.")


def test_printed_model_loading(trocr_printed_model):
    """Test that TrOCR printed model loads successfully."""
    model, processor = trocr_printed_model

    assert model is not None, "Model should not be None"
    assert processor is not None, "Processor should not be None"
    assert hasattr(model, "encoder"), "Model should have encoder"
    assert hasattr(model, "decoder"), "Model should have decoder"


def test_handwritten_model_loading(trocr_handwritten_model):
    """Test that TrOCR handwritten model loads successfully."""
    model, processor = trocr_handwritten_model

    assert model is not None, "Model should not be None"
    assert processor is not None, "Processor should not be None"
    assert hasattr(model, "encoder"), "Model should have encoder"
    assert hasattr(model, "decoder"), "Model should have decoder"


def test_basic_recognition(trocr_printed_model):
    """Test basic text recognition."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Create test image with text
    test_image = create_text_image("Hello")

    # Recognize text
    text = recognize(model, processor, test_image)

    assert text is not None, "Recognized text should not be None"
    assert isinstance(text, str), "Recognized text should be a string"


def test_simple_words(trocr_printed_model):
    """Test recognition of simple words."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Test common words
    words = ["Hello", "World", "Test", "Python", "Code"]

    for word in words:
        test_image = create_text_image(word)
        text = recognize(model, processor, test_image)

        assert text is not None, f"Should recognize '{word}'"
        assert isinstance(text, str), f"Text for '{word}' should be string"


def test_numbers(trocr_printed_model):
    """Test recognition of numbers."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Test numbers
    numbers = ["123", "456", "789", "2024"]

    for num in numbers:
        test_image = create_text_image(num)
        text = recognize(model, processor, test_image)

        assert text is not None, f"Should recognize '{num}'"


def test_mixed_alphanumeric(trocr_printed_model):
    """Test recognition of mixed alphanumeric text."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Test mixed text
    mixed_texts = ["ABC123", "Test2024", "Code42"]

    for txt in mixed_texts:
        test_image = create_text_image(txt)
        text = recognize(model, processor, test_image)

        assert text is not None, f"Should recognize '{txt}'"


def test_batch_recognition(trocr_printed_model):
    """Test batch recognition of multiple images."""
    from smlx.models.TrOCR_small import recognize_batch

    model, processor = trocr_printed_model

    # Create multiple test images
    texts = ["Hello", "World", "Test"]
    images = [create_text_image(t) for t in texts]

    # Recognize batch
    results = recognize_batch(model, processor, images)

    assert len(results) == len(images), "Should have result for each image"

    for i, result in enumerate(results):
        assert result is not None, f"Result {i} should not be None"
        assert isinstance(result, str), f"Result {i} should be string"


def test_recognition_with_confidence(trocr_printed_model):
    """Test recognition with confidence scores."""
    from smlx.models.TrOCR_small import recognize_with_confidence

    model, processor = trocr_printed_model

    # Create test image
    test_image = create_text_image("Test")

    # Recognize with confidence
    text, confidence = recognize_with_confidence(model, processor, test_image)

    assert text is not None, "Text should not be None"
    assert isinstance(text, str), "Text should be string"
    assert confidence is not None, "Confidence should not be None"
    assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"


def test_image_preprocessing():
    """Test image preprocessing."""
    from smlx.models.TrOCR_small import preprocess_image

    # Create test image
    test_image = create_text_image("Test")

    # Preprocess
    processed = preprocess_image(test_image)

    assert processed is not None, "Preprocessed image should not be None"


def test_pil_image_input(trocr_printed_model):
    """Test that PIL Image input works correctly."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Create PIL Image
    pil_image = create_text_image("Test")

    # Should accept PIL Image
    text = recognize(model, processor, pil_image)

    assert text is not None, "Should work with PIL Image"


def test_file_path_input(trocr_printed_model, tmp_path):
    """Test that file path input works correctly."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Create and save test image
    test_image = create_text_image("Path")
    image_path = tmp_path / "test_text.png"
    test_image.save(image_path)

    # Should accept file path
    text = recognize(model, processor, str(image_path))

    assert text is not None, "Should work with file path"


def test_different_image_sizes(trocr_printed_model):
    """Test with different image sizes."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Test different sizes
    sizes = [(200, 50), (400, 100), (600, 150)]

    for width, height in sizes:
        test_image = create_text_image("Test", size=(width, height))
        text = recognize(model, processor, test_image)

        assert text is not None, f"Should work with size {width}x{height}"


def test_uppercase_text(trocr_printed_model):
    """Test recognition of uppercase text."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Test uppercase
    test_image = create_text_image("UPPERCASE")
    text = recognize(model, processor, test_image)

    assert text is not None, "Should recognize uppercase"


def test_lowercase_text(trocr_printed_model):
    """Test recognition of lowercase text."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Test lowercase
    test_image = create_text_image("lowercase")
    text = recognize(model, processor, test_image)

    assert text is not None, "Should recognize lowercase"


def test_mixed_case(trocr_printed_model):
    """Test recognition of mixed case text."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Test mixed case
    test_image = create_text_image("MixedCase")
    text = recognize(model, processor, test_image)

    assert text is not None, "Should recognize mixed case"


def test_special_characters(trocr_printed_model):
    """Test recognition of text with special characters."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Test with common special characters
    special_texts = ["Test!", "Hello?", "Test-123"]

    for txt in special_texts:
        test_image = create_text_image(txt)
        text = recognize(model, processor, test_image)

        assert text is not None, f"Should handle '{txt}'"


def test_empty_image(trocr_printed_model):
    """Test with blank/empty image."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Create blank image
    blank_image = Image.new("RGB", (400, 100), color="white")

    # Should handle gracefully (may return empty string or placeholder)
    text = recognize(model, processor, blank_image)

    assert text is not None, "Should handle blank image"


def test_noisy_image(trocr_printed_model):
    """Test with slightly noisy image."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Create text image
    test_image = create_text_image("Noisy")

    # Add some noise (dots)
    draw = ImageDraw.Draw(test_image)
    import random

    for _ in range(50):
        x = random.randint(0, test_image.width - 1)
        y = random.randint(0, test_image.height - 1)
        draw.point((x, y), fill="gray")

    # Should still attempt recognition
    text = recognize(model, processor, test_image)

    assert text is not None, "Should handle noisy image"


def test_handwritten_recognition(trocr_handwritten_model):
    """Test handwritten text recognition."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_handwritten_model

    # Create simple "handwritten-like" image (still printed for test)
    test_image = create_text_image("Hello")

    # Recognize with handwritten model
    text = recognize(model, processor, test_image)

    assert text is not None, "Handwritten model should recognize text"


def test_processor_configuration(trocr_printed_model):
    """Test processor configuration."""
    from smlx.models.TrOCR_small import TrOCRProcessor

    model, processor = trocr_printed_model

    assert isinstance(processor, TrOCRProcessor), "Should be TrOCRProcessor"
    assert processor is not None, "Processor should be configured"


def test_model_config(trocr_printed_model):
    """Test model configuration."""
    from smlx.models.TrOCR_small import TrOCRConfig

    model, processor = trocr_printed_model

    # Model should have config
    assert hasattr(model, "config"), "Model should have config"

    config = model.config
    assert config is not None, "Config should not be None"


def test_default_configs():
    """Test default configurations."""
    from smlx.models.TrOCR_small import (
        DEFAULT_CONFIG_PRINTED,
        DEFAULT_CONFIG_HANDWRITTEN,
    )

    assert DEFAULT_CONFIG_PRINTED is not None, "Printed config should exist"
    assert (
        DEFAULT_CONFIG_HANDWRITTEN is not None
    ), "Handwritten config should exist"

    # Check config attributes
    assert hasattr(
        DEFAULT_CONFIG_PRINTED, "vision_config"
    ), "Should have vision_config"
    assert hasattr(
        DEFAULT_CONFIG_PRINTED, "decoder_config"
    ), "Should have decoder_config"


def test_vision_encoder(trocr_printed_model):
    """Test vision encoder component."""
    model, processor = trocr_printed_model

    encoder = model.encoder
    assert encoder is not None, "Vision encoder should not be None"


def test_text_decoder(trocr_printed_model):
    """Test text decoder component."""
    model, processor = trocr_printed_model

    decoder = model.decoder
    assert decoder is not None, "Text decoder should not be None"


def test_multiple_recognitions(trocr_printed_model):
    """Test multiple sequential recognitions."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # Perform multiple recognitions
    texts = ["First", "Second", "Third"]

    for txt in texts:
        test_image = create_text_image(txt)
        result = recognize(model, processor, test_image)

        assert result is not None, f"Recognition {txt} should work"


def test_long_text(trocr_printed_model):
    """Test with longer text (still single line)."""
    from smlx.models.TrOCR_small import recognize

    model, processor = trocr_printed_model

    # TrOCR is designed for single-line text, so keep it reasonable
    long_text = "Long Text Example"
    test_image = create_text_image(long_text, size=(600, 100))

    text = recognize(model, processor, test_image)

    assert text is not None, "Should handle longer text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
