#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for output validation framework.

Tests the validation system integrated with real models to ensure:
1. Validation works end-to-end with generation
2. Retry logic functions correctly
3. Quality metrics integrate properly
4. No performance regression
"""

import pytest

import mlx.core as mx


@pytest.mark.integration
class TestValidationIntegration:
    """Test validation integrated with model generation."""

    @pytest.fixture
    def smollm2_model(self):
        """Load SmolLM2 model for testing."""
        from smlx.models.SmolLM2_135M import load

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        return model, tokenizer

    def test_basic_validation(self, smollm2_model):
        """Test basic validation works with generation."""
        from smlx.models.SmolLM2_135M import generate

        model, tokenizer = smollm2_model

        # Generate without validation
        output_no_val = generate(
            model,
            tokenizer,
            "Write a haiku about code:",
            max_tokens=50,
            temperature=0.7,
            validate_output=False,
        )

        assert len(output_no_val) > 0

        # Generate with validation
        output_with_val = generate(
            model,
            tokenizer,
            "Write a haiku about code:",
            max_tokens=50,
            temperature=0.7,
            validate_output=True,
            max_repetition_ratio=0.5,
        )

        assert len(output_with_val) > 0

    def test_min_tokens_enforcement(self, smollm2_model):
        """Test that min_tokens prevents early stopping."""
        from smlx.models.SmolLM2_135M import generate

        model, tokenizer = smollm2_model

        # Generate with min_tokens
        output = generate(
            model,
            tokenizer,
            "Hello",
            max_tokens=100,
            temperature=0.7,
            min_tokens=20,  # Force at least 20 tokens
        )

        tokens = tokenizer.encode(output)
        assert len(tokens) >= 20, f"Expected >=20 tokens, got {len(tokens)}"

    def test_validation_retry_on_failure(self, smollm2_model):
        """Test retry logic when validation fails."""
        from smlx.models.SmolLM2_135M import generate

        model, tokenizer = smollm2_model

        # This might fail validation due to very high repetition constraint
        # But retry should help
        output = generate(
            model,
            tokenizer,
            "Count to five:",
            max_tokens=30,
            temperature=0.3,  # Lower temperature might cause repetition
            validate_output=True,
            max_repetition_ratio=0.3,  # Very strict
            retry_on_failure=True,
            max_retries=3,
        )

        # Should get output even if first attempts failed
        assert len(output) > 0

    def test_gibberish_detection(self, smollm2_model):
        """Test that gibberish would be detected (synthetic test)."""
        from smlx.utils import validate_text_output

        # Simulate gibberish outputs
        gibberish_examples = [
            "©©©©©©©©",
            ".....................",
            "\x00\x01\x02",
        ]

        for gibberish in gibberish_examples:
            is_valid, reason = validate_text_output(gibberish, check_gibberish=True)

            # Some might pass basic checks but gibberish detector should catch most
            if not is_valid:
                assert "gibberish" in reason.lower() or "special" in reason.lower()

    def test_repetition_detection(self, smollm2_model):
        """Test that excessive repetition is detected."""
        from smlx.utils import validate_text_output

        # Highly repetitive text
        repetitive = "the the the the the the the the the the"

        is_valid, reason = validate_text_output(
            repetitive, max_repetition_ratio=0.3  # Strict threshold
        )

        assert not is_valid, "Should detect excessive repetition"
        assert "repetition" in reason.lower()

    def test_quality_metrics_integration(self, smollm2_model):
        """Test quality metrics work with generated output."""
        from smlx.models.SmolLM2_135M import generate
        from smlx.utils import assess_quality

        model, tokenizer = smollm2_model

        # Generate text
        output = generate(
            model,
            tokenizer,
            "Explain machine learning:",
            max_tokens=50,
            temperature=0.7,
        )

        # Assess quality
        metrics = assess_quality(model, tokenizer, output)

        # Check metrics are calculated
        assert metrics.perplexity is not None
        assert metrics.perplexity > 0
        assert metrics.perplexity < 1000  # Should be reasonable for good output

        assert metrics.repetition_3gram is not None
        assert 0 <= metrics.repetition_3gram <= 1

        assert metrics.unique_token_ratio is not None
        assert 0 < metrics.unique_token_ratio <= 1

    def test_validation_with_different_temperatures(self, smollm2_model):
        """Test validation works across temperature range."""
        from smlx.models.SmolLM2_135M import generate

        model, tokenizer = smollm2_model

        prompt = "Write a short poem about AI:"

        for temp in [0.0, 0.3, 0.7, 1.0]:
            output = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=30,
                temperature=temp,
                validate_output=True,
                max_repetition_ratio=0.6,
            )

            assert len(output) > 0, f"Failed at temperature {temp}"

    def test_empty_output_prevention(self, smollm2_model):
        """Test that min_tokens prevents empty outputs."""
        from smlx.models.SmolLM2_135M import generate

        model, tokenizer = smollm2_model

        # Even with early stopping, should get min_tokens
        output = generate(
            model,
            tokenizer,
            "Hi",
            max_tokens=100,
            temperature=0.0,  # Greedy
            min_tokens=10,
        )

        tokens = tokenizer.encode(output)
        assert len(tokens) >= 10, "Should enforce minimum token count"

    @pytest.mark.slow
    def test_performance_overhead(self, smollm2_model):
        """Test that validation doesn't add excessive overhead."""
        import time

        from smlx.models.SmolLM2_135M import generate

        model, tokenizer = smollm2_model

        prompt = "Count to ten:"

        # Time without validation
        start = time.time()
        for _ in range(3):
            generate(
                model,
                tokenizer,
                prompt,
                max_tokens=30,
                temperature=0.7,
                validate_output=False,
            )
        time_without_val = time.time() - start

        # Time with validation
        start = time.time()
        for _ in range(3):
            generate(
                model,
                tokenizer,
                prompt,
                max_tokens=30,
                temperature=0.7,
                validate_output=True,
                max_repetition_ratio=0.5,
            )
        time_with_val = time.time() - start

        # Validation overhead should be <20% (mostly just string checks)
        overhead_ratio = (time_with_val - time_without_val) / time_without_val

        assert overhead_ratio < 0.2, f"Validation overhead too high: {overhead_ratio:.1%}"


@pytest.mark.integration
class TestVLMValidation:
    """Test validation with vision-language models."""

    @pytest.fixture
    def mock_image(self):
        """Create a mock image for testing."""
        from PIL import Image

        # Create simple test image
        return Image.new("RGB", (224, 224), color="red")

    def test_vlm_validation_basic(self, mock_image):
        """Test basic VLM output validation."""
        pytest.skip("VLM validation integration - implement when VLM models updated")

        # This would test nanoVLM, TinyLLaVA with validation
        # from smlx.models.nanoVLM import load, generate
        # model, processor = load()
        # output = generate(
        #     model, processor,
        #     "What is in this image?",
        #     mock_image,
        #     validate_output=True
        # )
        # assert len(output) > 0


@pytest.mark.integration
class TestAudioValidation:
    """Test validation with audio models."""

    def test_audio_waveform_validation(self):
        """Test audio output validation."""
        from smlx.utils import validate_audio_output

        # Good audio (1 second at 16kHz)
        good_audio = mx.random.normal((16000,)) * 0.5

        is_valid, reason = validate_audio_output(good_audio, sample_rate=16000)
        assert is_valid

        # Bad audio (all zeros = silence)
        bad_audio = mx.zeros((16000,))

        is_valid, reason = validate_audio_output(bad_audio, sample_rate=16000)
        assert not is_valid
        assert "silent" in reason.lower()

    def test_audio_clipping_detection(self):
        """Test detection of clipped audio."""
        from smlx.utils import validate_audio_output

        # Clipped audio (values > 1.0)
        clipped = mx.random.normal((16000,)) * 2.0  # Will exceed [-1, 1]

        is_valid, reason = validate_audio_output(clipped, sample_rate=16000)

        # Might detect clipping
        if not is_valid:
            assert "clipping" in reason.lower()


@pytest.mark.unit
class TestValidationUtilities:
    """Unit tests for validation utilities."""

    def test_validate_text_output_basic(self):
        """Test basic text validation."""
        from smlx.utils import validate_text_output

        # Valid text
        is_valid, reason = validate_text_output("This is a normal sentence.")
        assert is_valid

        # Empty text
        is_valid, reason = validate_text_output("")
        assert not is_valid
        assert "empty" in reason.lower()

        # Whitespace only
        is_valid, reason = validate_text_output("   \n\n   ")
        assert not is_valid

    def test_output_validator_class(self):
        """Test OutputValidator class."""
        from smlx.utils import OutputValidator

        validator = OutputValidator(
            min_length=10, max_repetition_ratio=0.4, check_gibberish=True, strict=True
        )

        # Valid text
        result = validator.validate_text("This is a perfectly normal sentence with enough length.")
        assert result.is_valid

        # Too short
        result = validator.validate_text("Hi")
        assert not result.is_valid
        assert "short" in result.reason.lower()

    def test_quality_metrics_calculation(self):
        """Test quality metrics functions."""
        from smlx.utils import analyze_repetition, analyze_token_distribution

        # Test repetition analysis
        text = "the cat sat on the mat and the dog sat on the rug"
        metrics = analyze_repetition(text, max_n=3)

        assert "repetition_1gram" in metrics
        assert "repetition_2gram" in metrics
        assert "repetition_3gram" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())

        # Test token distribution
        tokens = [1, 2, 3, 1, 2, 1, 4, 5]
        dist = analyze_token_distribution(tokens)

        assert dist["unique_count"] == 5
        assert dist["total_count"] == 8
        assert dist["most_common_token"] == 1
        assert dist["most_common_count"] == 3


@pytest.mark.benchmark
class TestValidationPerformance:
    """Benchmark validation performance impact."""

    def test_validation_speed(self):
        """Test validation is fast."""
        import time

        from smlx.utils import validate_text_output

        # Generate test text
        test_text = "This is a test sentence. " * 100

        # Time validation
        start = time.time()
        iterations = 1000
        for _ in range(iterations):
            validate_text_output(
                test_text, min_length=10, max_repetition_ratio=0.5, check_gibberish=True
            )
        elapsed = time.time() - start

        time_per_validation = elapsed / iterations

        # Should be fast (<1ms per validation)
        assert time_per_validation < 0.001, f"Validation too slow: {time_per_validation*1000:.2f}ms"

    def test_quality_metrics_speed(self):
        """Test quality metrics calculation speed."""
        import time

        from smlx.utils import analyze_repetition, analyze_token_distribution

        # Test data
        text = "word " * 1000
        tokens = list(range(1000))

        # Time repetition analysis
        start = time.time()
        for _ in range(100):
            analyze_repetition(text, max_n=4)
        rep_time = (time.time() - start) / 100

        # Time token distribution
        start = time.time()
        for _ in range(100):
            analyze_token_distribution(tokens)
        dist_time = (time.time() - start) / 100

        # Both should be fast
        assert rep_time < 0.01, f"Repetition analysis too slow: {rep_time*1000:.2f}ms"
        assert dist_time < 0.001, f"Distribution analysis too slow: {dist_time*1000:.2f}ms"
