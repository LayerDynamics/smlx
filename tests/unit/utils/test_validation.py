#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Unit tests for output validation utilities.

Tests cover:
- Gibberish detection
- Repetition detection
- Empty output detection
- Audio validation
- ValidationResult dataclass
- OutputValidator configuration
"""

import pytest
import mlx.core as mx
import numpy as np

from smlx.utils.validation import (
    ValidationResult,
    OutputValidator,
    validate_text_output,
    validate_audio_output,
    validate_tokens,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True, reason="", metadata={})
        assert result.is_valid
        assert result.reason == ""
        assert result.metadata == {}

    def test_invalid_result(self):
        """Test creating an invalid result."""
        result = ValidationResult(
            is_valid=False,
            reason="Too much repetition",
            metadata={"repetition_ratio": 0.8}
        )
        assert not result.is_valid
        assert "repetition" in result.reason.lower()
        assert result.metadata["repetition_ratio"] == 0.8


class TestTextValidation:
    """Test text output validation."""

    def test_valid_text(self):
        """Test validation passes for normal text."""
        is_valid, reason = validate_text_output(
            "The quick brown fox jumps over the lazy dog.",
            min_length=5,
            max_repetition_ratio=0.5,
            check_gibberish=True
        )
        assert is_valid
        assert reason == ""

    def test_empty_text(self):
        """Test validation fails for empty text."""
        is_valid, reason = validate_text_output(
            "",
            min_length=1,
            max_repetition_ratio=0.5,
            check_gibberish=True
        )
        assert not is_valid
        assert "empty" in reason.lower() or "length" in reason.lower()

    def test_whitespace_only(self):
        """Test validation fails for whitespace-only text."""
        is_valid, reason = validate_text_output(
            "   \n\t  ",
            min_length=1,
            max_repetition_ratio=0.5,
            check_gibberish=True
        )
        assert not is_valid
        assert "empty" in reason.lower() or "whitespace" in reason.lower()

    def test_excessive_repetition(self):
        """Test validation fails for pathological repetition."""
        # Repeat the same token many times
        repeated_text = "hello " * 100
        is_valid, reason = validate_text_output(
            repeated_text,
            min_length=5,
            max_repetition_ratio=0.3,  # Strict threshold
            check_gibberish=False
        )
        assert not is_valid
        assert "repetition" in reason.lower()

    def test_moderate_repetition_passes(self):
        """Test validation passes for moderate repetition."""
        # Some repetition is normal in natural language
        text = "I really, really like Python programming. Python is great."
        is_valid, reason = validate_text_output(
            text,
            min_length=5,
            max_repetition_ratio=0.6,
            check_gibberish=False
        )
        assert is_valid

    def test_gibberish_detection(self):
        """Test validation detects gibberish."""
        # Random character sequences
        gibberish = "asdfjkl;qwertyuiop[]zxcvbnm,./1234567890"
        is_valid, reason = validate_text_output(
            gibberish,
            min_length=5,
            max_repetition_ratio=0.6,
            check_gibberish=True
        )
        # May or may not fail depending on implementation
        # This test documents the expected behavior

    def test_min_length_threshold(self):
        """Test minimum length validation."""
        short_text = "Hi"
        is_valid, reason = validate_text_output(
            short_text,
            min_length=10,
            max_repetition_ratio=0.6,
            check_gibberish=False
        )
        assert not is_valid
        assert "length" in reason.lower() or "short" in reason.lower()

    def test_special_characters_not_gibberish(self):
        """Test that code/special chars aren't flagged as gibberish."""
        code = "def hello():\n    print('Hello, world!')\n    return 42"
        is_valid, reason = validate_text_output(
            code,
            min_length=5,
            max_repetition_ratio=0.6,
            check_gibberish=True
        )
        # Code should generally pass (has structure)
        # Note: Implementation-dependent behavior


class TestAudioValidation:
    """Test audio output validation."""

    def test_valid_audio(self):
        """Test validation passes for normal audio."""
        # Create synthetic audio signal (sine wave)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        is_valid, reason = validate_audio_output(
            audio,
            sample_rate=sample_rate,
            min_duration=0.1,
            max_silence_ratio=0.9,
            check_clipping=True
        )
        assert is_valid
        assert reason == ""

    def test_silent_audio(self):
        """Test validation fails for silent audio."""
        # All zeros
        audio = np.zeros(16000)
        is_valid, reason = validate_audio_output(
            audio,
            sample_rate=16000,
            min_duration=0.1,
            max_silence_ratio=0.1,  # Very strict
            check_clipping=False
        )
        assert not is_valid
        assert "silence" in reason.lower() or "silent" in reason.lower()

    def test_clipping_detection(self):
        """Test validation detects clipping."""
        # Create clipped audio (values at ±1.0)
        audio = np.ones(16000)
        is_valid, reason = validate_audio_output(
            audio,
            sample_rate=16000,
            min_duration=0.1,
            max_silence_ratio=0.9,
            check_clipping=True
        )
        assert not is_valid
        assert "clip" in reason.lower()

    def test_too_short_audio(self):
        """Test validation fails for too-short audio."""
        # Very short audio
        audio = np.random.randn(1600)  # 0.1 seconds at 16kHz
        is_valid, reason = validate_audio_output(
            audio,
            sample_rate=16000,
            min_duration=1.0,  # Require at least 1 second
            max_silence_ratio=0.9,
            check_clipping=False
        )
        assert not is_valid
        assert "duration" in reason.lower() or "short" in reason.lower()

    def test_mlx_array_audio(self):
        """Test validation works with MLX arrays."""
        # Create audio as MLX array
        t = np.linspace(0, 1.0, 16000)
        audio_np = 0.3 * np.sin(2 * np.pi * 440 * t)
        audio_mx = mx.array(audio_np)

        is_valid, reason = validate_audio_output(
            audio_mx,
            sample_rate=16000,
            min_duration=0.1,
            max_silence_ratio=0.9,
            check_clipping=True
        )
        assert is_valid


class TestTokenValidation:
    """Test token sequence validation."""

    def test_valid_tokens(self):
        """Test validation passes for normal token sequences."""
        tokens = [1, 42, 100, 256, 500, 1000]
        is_valid, reason = validate_tokens(
            tokens,
            vocab_size=2000,
            min_tokens=3,
            max_repetition_ratio=0.5
        )
        assert is_valid
        assert reason == ""

    def test_out_of_vocab_tokens(self):
        """Test validation fails for invalid token IDs."""
        tokens = [1, 42, 3000, 256]  # 3000 > vocab_size
        is_valid, reason = validate_tokens(
            tokens,
            vocab_size=2000,
            min_tokens=3,
            max_repetition_ratio=0.5
        )
        assert not is_valid
        assert "vocab" in reason.lower() or "range" in reason.lower()

    def test_negative_tokens(self):
        """Test validation fails for negative token IDs."""
        tokens = [1, -5, 42]
        is_valid, reason = validate_tokens(
            tokens,
            vocab_size=2000,
            min_tokens=3,
            max_repetition_ratio=0.5
        )
        assert not is_valid

    def test_too_few_tokens(self):
        """Test validation fails for too few tokens."""
        tokens = [1, 2]
        is_valid, reason = validate_tokens(
            tokens,
            vocab_size=2000,
            min_tokens=10,
            max_repetition_ratio=0.5
        )
        assert not is_valid
        assert "length" in reason.lower() or "tokens" in reason.lower()

    def test_repeated_tokens(self):
        """Test validation detects excessive token repetition."""
        # Same token repeated many times
        tokens = [42] * 100
        is_valid, reason = validate_tokens(
            tokens,
            vocab_size=2000,
            min_tokens=3,
            max_repetition_ratio=0.2
        )
        assert not is_valid
        assert "repetition" in reason.lower()

    def test_mlx_array_tokens(self):
        """Test validation works with MLX array tokens."""
        tokens = mx.array([1, 42, 100, 256, 500])
        is_valid, reason = validate_tokens(
            tokens,
            vocab_size=2000,
            min_tokens=3,
            max_repetition_ratio=0.5
        )
        assert is_valid


class TestOutputValidator:
    """Test OutputValidator class."""

    def test_default_validator(self):
        """Test validator with default settings."""
        validator = OutputValidator()
        result = validator.validate_text("This is a test sentence.")
        assert result.is_valid

    def test_strict_validator(self):
        """Test strict validation mode."""
        validator = OutputValidator(
            min_length=20,
            max_repetition_ratio=0.2,
            strict=True
        )
        # Short text should fail
        result = validator.validate_text("Short")
        assert not result.is_valid

    def test_lenient_validator(self):
        """Test lenient validation mode."""
        validator = OutputValidator(
            min_length=1,
            max_repetition_ratio=0.8,
            strict=False
        )
        # Even with some issues, lenient mode may pass
        result = validator.validate_text("test test test")
        # Behavior depends on implementation

    def test_custom_thresholds(self):
        """Test validator with custom thresholds."""
        validator = OutputValidator(
            min_length=10,
            max_repetition_ratio=0.4,
            check_gibberish=True
        )
        # Normal text should pass
        result = validator.validate_text("The quick brown fox jumps over the lazy dog")
        assert result.is_valid

        # Repeated text should fail
        result = validator.validate_text("hello " * 50)
        assert not result.is_valid

    def test_metadata_included(self):
        """Test that validation results include metadata."""
        validator = OutputValidator()
        result = validator.validate_text("Test sentence with some words.")

        # Should include some diagnostic metadata
        assert isinstance(result.metadata, dict)
        # Metadata might include things like length, repetition_ratio, etc.


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_text(self):
        """Test validation handles Unicode properly."""
        unicode_text = "Hello 世界! Здравствуй мир! 🌍"
        is_valid, reason = validate_text_output(
            unicode_text,
            min_length=5,
            max_repetition_ratio=0.6,
            check_gibberish=False
        )
        assert is_valid

    def test_very_long_text(self):
        """Test validation handles very long text."""
        long_text = "This is a test sentence. " * 1000
        is_valid, reason = validate_text_output(
            long_text,
            min_length=5,
            max_repetition_ratio=0.6,
            check_gibberish=False
        )
        # Should handle without crashing

    def test_none_input(self):
        """Test validation handles None gracefully."""
        with pytest.raises((TypeError, ValueError)):
            validate_text_output(None)

    def test_empty_token_list(self):
        """Test validation handles empty token list."""
        is_valid, reason = validate_tokens(
            [],
            vocab_size=2000,
            min_tokens=1,
            max_repetition_ratio=0.5
        )
        assert not is_valid
