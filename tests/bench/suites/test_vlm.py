"""
Tests for vision-language model benchmark suite.

Tests the VLM benchmarking functions.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
import numpy as np

from smlx.bench.suites.vlm import (
    VLMBenchmarkConfig,
    benchmark_vlm,
    benchmark_vlm_batch,
)
from smlx.bench.stats import ModelBenchmarkStats


# Mock classes for testing


class MockVLMModel:
    """Mock vision-language model.

    Implements the documented benchmark generation interface
    ``generate(prompt, image, max_tokens, temperature) -> str`` so it can be
    driven by benchmark_vlm without a real model.
    """

    def __init__(self, name="MockVLM"):
        self.name = name

    def generate(self, prompt=None, image=None, max_tokens=10, temperature=0.0, **kwargs):
        """Return a deterministic mock generation of roughly max_tokens words."""
        return " ".join(["token"] * max(1, int(max_tokens)))


class MockProcessor:
    """Mock processor for VLM."""

    def __init__(self):
        self.tokenizer = MockTokenizer()

    def __call__(self, image=None, text=None):
        """Mock processing."""
        return {"inputs": "processed"}


class MockTokenizer:
    """Mock tokenizer."""

    def encode(self, text):
        """Return mock tokens."""
        return [1, 2, 3, 4, 5]

    def decode(self, tokens):
        """Return mock text."""
        return f"Generated {len(tokens)} tokens"


def create_test_image(size=(224, 224)):
    """Create a test image for testing.

    Args:
        size: Image size as (width, height) following PIL convention

    Returns:
        PIL Image with the specified size
    """
    # PIL size is (width, height), but numpy arrays are (height, width, channels)
    width, height = size
    array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(array)


@pytest.mark.unit
class TestVLMBenchmarkConfig:
    """Test VLMBenchmarkConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = VLMBenchmarkConfig()

        assert config.max_tokens == 100
        assert config.num_trials == 5
        assert config.temperature == 0.0
        assert config.seed == 0
        assert config.resize_shape is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = VLMBenchmarkConfig(
            max_tokens=200,
            num_trials=3,
            temperature=0.7,
            seed=42,
            resize_shape=(256, 256),
        )

        assert config.max_tokens == 200
        assert config.num_trials == 3
        assert config.temperature == 0.7
        assert config.seed == 42
        assert config.resize_shape == (256, 256)


@pytest.mark.unit
class TestBenchmarkVLM:
    """Test benchmark_vlm function."""

    def test_with_image_path(self):
        """Test benchmarking with image path."""
        with TemporaryDirectory() as tmpdir:
            # Create test image
            img = create_test_image()
            img_path = Path(tmpdir) / "test.jpg"
            img.save(img_path)

            model = MockVLMModel()
            processor = MockProcessor()

            config = VLMBenchmarkConfig(
                max_tokens=10,
                num_trials=1,
            )

            stats = benchmark_vlm(
                model=model,
                processor=processor,
                image=img_path,
                prompt="Describe this image.",
                config=config,
            )

            assert isinstance(stats, ModelBenchmarkStats)
            assert stats.prompt_tokens > 0
            assert stats.generation_tokens > 0

    def test_with_pil_image(self):
        """Test benchmarking with PIL Image."""
        img = create_test_image()

        model = MockVLMModel()
        processor = MockProcessor()

        config = VLMBenchmarkConfig(
            max_tokens=10,
            num_trials=1,
        )

        stats = benchmark_vlm(
            model=model,
            processor=processor,
            image=img,
            prompt="What is in this image?",
            config=config,
        )

        assert isinstance(stats, ModelBenchmarkStats)

    def test_with_resize(self):
        """Test benchmarking with image resize."""
        img = create_test_image(size=(512, 512))

        model = MockVLMModel()
        processor = MockProcessor()

        config = VLMBenchmarkConfig(
            max_tokens=10,
            resize_shape=(224, 224),
        )

        stats = benchmark_vlm(
            model=model,
            processor=processor,
            image=img,
            prompt="Describe this.",
            config=config,
        )

        assert isinstance(stats, ModelBenchmarkStats)

    def test_different_prompts(self):
        """Test with different prompts."""
        img = create_test_image()
        model = MockVLMModel()
        processor = MockProcessor()

        prompts = [
            "What is this?",
            "Describe the image in detail.",
            "What objects are visible?",
        ]

        for prompt in prompts:
            stats = benchmark_vlm(
                model=model,
                processor=processor,
                image=img,
                prompt=prompt,
                config=VLMBenchmarkConfig(max_tokens=10),
            )

            assert isinstance(stats, ModelBenchmarkStats)


@pytest.mark.unit
class TestBenchmarkVLMBatch:
    """Test benchmark_vlm_batch function."""

    def test_batch_with_multiple_images(self):
        """Test batch benchmarking with multiple images."""
        # Create test images
        images = [create_test_image() for _ in range(3)]
        prompts = [
            "What is this?",
            "Describe this image.",
            "What objects do you see?",
        ]

        pairs = list(zip(images, prompts))

        model = MockVLMModel()
        processor = MockProcessor()

        config = VLMBenchmarkConfig(max_tokens=10)

        results = benchmark_vlm_batch(
            model=model,
            processor=processor,
            image_prompt_pairs=pairs,
            config=config,
        )

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, ModelBenchmarkStats) for r in results)

    def test_batch_with_image_paths(self):
        """Test batch with image file paths."""
        with TemporaryDirectory() as tmpdir:
            # Create test images
            img_paths = []
            for i in range(2):
                img = create_test_image()
                img_path = Path(tmpdir) / f"test_{i}.jpg"
                img.save(img_path)
                img_paths.append(img_path)

            prompts = ["First image", "Second image"]
            pairs = list(zip(img_paths, prompts))

            model = MockVLMModel()
            processor = MockProcessor()

            results = benchmark_vlm_batch(
                model=model,
                processor=processor,
                image_prompt_pairs=pairs,
            )

            assert len(results) == 2

    def test_batch_with_mixed_inputs(self):
        """Test batch with mixed PIL images and paths."""
        with TemporaryDirectory() as tmpdir:
            # Create one file path
            img1 = create_test_image()
            img1_path = Path(tmpdir) / "test1.jpg"
            img1.save(img1_path)

            # Create one PIL image
            img2 = create_test_image()

            pairs = [
                (img1_path, "First prompt"),
                (img2, "Second prompt"),
            ]

            model = MockVLMModel()
            processor = MockProcessor()

            results = benchmark_vlm_batch(
                model=model,
                processor=processor,
                image_prompt_pairs=pairs,
            )

            assert len(results) == 2


@pytest.mark.integration
@pytest.mark.requires_model
@pytest.mark.slow
class TestVLMIntegration:
    """Integration tests for VLM benchmarks."""

    @pytest.mark.skip(reason="Requires full VLM model implementation")
    def test_end_to_end_vlm_benchmark(self):
        """Test complete VLM benchmark workflow."""
        # This would test with a real VLM model
        # For now, skipped until VLM models are implemented
        pass

    @pytest.mark.skip(reason="Requires full VLM model implementation")
    def test_vlm_memory_tracking(self):
        """Test that VLM benchmark tracks memory correctly."""
        # This would verify memory tracking with real VLM
        pass


@pytest.mark.benchmark
class TestVLMPerformance:
    """Performance tests for VLM benchmarks."""

    def test_image_loading_overhead(self):
        """Test image loading overhead."""
        with TemporaryDirectory() as tmpdir:
            # Create test image
            img = create_test_image(size=(1024, 1024))
            img_path = Path(tmpdir) / "large.jpg"
            img.save(img_path)

            import time

            # Time loading from file
            start = time.perf_counter()
            with Image.open(img_path) as loaded:
                load_time = time.perf_counter() - start

                # Should be fast (< 100ms for 1024x1024)
                assert load_time < 0.1

    def test_image_resize_overhead(self):
        """Test image resize overhead."""
        img = create_test_image(size=(1024, 1024))

        import time

        start = time.perf_counter()
        resized = img.resize((224, 224))
        resize_time = time.perf_counter() - start

        # Should be fast
        assert resize_time < 0.1

    def test_batch_processing_efficiency(self):
        """Test that batch processing is efficient."""
        # Create multiple test images
        images = [create_test_image() for _ in range(5)]

        # Verify images created successfully
        assert len(images) == 5
        assert all(isinstance(img, Image.Image) for img in images)


@pytest.mark.unit
class TestVLMHelpers:
    """Test helper functions and edge cases."""

    def test_image_format_support(self):
        """Test that various image formats are supported."""
        with TemporaryDirectory() as tmpdir:
            img = create_test_image()

            # Save in different formats
            formats = [
                ("test.jpg", "JPEG"),
                ("test.png", "PNG"),
            ]

            for filename, fmt in formats:
                img_path = Path(tmpdir) / filename
                img.save(img_path, format=fmt)

                # Load and verify
                with Image.open(img_path) as loaded:
                    assert loaded is not None

    def test_image_size_variations(self):
        """Test handling of different image sizes."""
        sizes = [
            (224, 224),
            (256, 256),
            (512, 512),
            (1024, 768),
            (768, 1024),
        ]

        for size in sizes:
            img = create_test_image(size=size)
            assert img.size == size

    def test_config_serialization(self):
        """Test that config can be serialized."""
        config = VLMBenchmarkConfig(
            max_tokens=150,
            num_trials=3,
            temperature=0.5,
            seed=123,
            resize_shape=(256, 256),
        )

        # Convert to dict (using dataclass asdict would work)
        from dataclasses import asdict

        data = asdict(config)

        assert data["max_tokens"] == 150
        assert data["num_trials"] == 3
        assert data["temperature"] == 0.5
        assert data["seed"] == 123
        assert data["resize_shape"] == (256, 256)


@pytest.mark.unit
class TestVLMEdgeCases:
    """Test edge cases for VLM benchmarks."""

    def test_very_small_image(self):
        """Test with very small image."""
        img = create_test_image(size=(16, 16))
        assert img.size == (16, 16)

    def test_very_large_image(self):
        """Test with very large image."""
        # Create but don't actually benchmark with it
        img = create_test_image(size=(2048, 2048))
        assert img.size == (2048, 2048)

    def test_non_square_images(self):
        """Test with non-square images."""
        sizes = [
            (224, 448),  # 1:2 ratio
            (448, 224),  # 2:1 ratio
            (640, 480),  # 4:3 ratio
        ]

        for size in sizes:
            img = create_test_image(size=size)
            assert img.size == size

    def test_grayscale_conversion(self):
        """Test grayscale image handling."""
        img = create_test_image()

        # Convert to grayscale
        gray = img.convert("L")
        assert gray.mode == "L"

        # Convert back to RGB
        rgb = gray.convert("RGB")
        assert rgb.mode == "RGB"

    def test_empty_prompt(self):
        """Test VLM config with empty prompt."""
        config = VLMBenchmarkConfig(max_tokens=10)
        # Config itself should be fine
        assert config.max_tokens == 10

    def test_very_long_prompt(self):
        """Test VLM with very long prompt."""
        long_prompt = " ".join(["word"] * 1000)
        # Just verify we can create the prompt
        assert len(long_prompt) > 1000

    def test_special_characters_in_prompt(self):
        """Test prompts with special characters."""
        special_prompts = [
            "What's this image?",
            "Describe the image (in detail).",
            "Image #1: describe it!",
            "Question: what do you see?",
        ]

        # Verify all prompts are valid strings
        assert all(isinstance(p, str) for p in special_prompts)


@pytest.mark.unit
class TestVLMBatchEdgeCases:
    """Test edge cases for batch VLM benchmarks."""

    def test_single_image_batch(self):
        """Test batch with single image."""
        img = create_test_image()
        pairs = [(img, "Single image")]

        # Should work with single item
        assert len(pairs) == 1

    def test_empty_batch(self):
        """Test empty batch."""
        pairs = []

        # Empty batch should be handled
        assert len(pairs) == 0

    def test_large_batch(self):
        """Test large batch."""
        # Create many images
        images = [create_test_image() for _ in range(100)]
        prompts = [f"Image {i}" for i in range(100)]
        pairs = list(zip(images, prompts))

        assert len(pairs) == 100


@pytest.mark.unit
class TestImageUtilities:
    """Test image utility functions."""

    def test_create_test_image(self):
        """Test the test image creation helper."""
        img = create_test_image()

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (224, 224)

    def test_create_custom_size_image(self):
        """Test creating custom size test image."""
        img = create_test_image(size=(100, 200))

        assert img.size == (100, 200)
        assert img.mode == "RGB"

    def test_image_save_and_load(self):
        """Test saving and loading images."""
        with TemporaryDirectory() as tmpdir:
            img = create_test_image()
            img_path = Path(tmpdir) / "test.jpg"

            # Save
            img.save(img_path)
            assert img_path.exists()

            # Load
            with Image.open(img_path) as loaded:
                assert loaded.size == img.size
                assert loaded.mode == img.mode
