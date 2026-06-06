#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Output validation utilities for ensuring model output quality.

Provides validation functions to detect and prevent common model output issues:
- Gibberish text (nonsensical character sequences)
- Empty or whitespace-only outputs
- Excessive repetition (pathological loops)
- Invalid tokens or malformed outputs
- Audio quality issues (silence, clipping, noise)

These validators are integrated with generation functions to provide automatic
quality assurance and can retry generation with adjusted parameters if needed.

Example:
    >>> from smlx.utils.validation import validate_text_output, OutputValidator
    >>>
    >>> # Quick validation
    >>> is_valid, reason = validate_text_output("Hello world!")
    >>> assert is_valid
    >>>
    >>> # Configurable validator
    >>> validator = OutputValidator(
    ...     min_length=10,
    ...     max_repetition_ratio=0.3,
    ...     strict=True
    ... )
    >>> result = validator.validate_text("This is a test.")
"""

import logging
import re
import string
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of output validation.

    Attributes:
        is_valid: Whether output passed validation
        reason: Reason for validation failure (if failed)
        confidence: Confidence score in validation (0-1)
        metadata: Additional validation metadata
    """

    is_valid: bool
    reason: Optional[str] = None
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OutputValidator:
    """
    Configurable validator for model outputs.

    Provides flexible validation with customizable thresholds and strictness
    levels. Can be used for text, audio, and token validation with automatic
    quality checks.

    Args:
        min_length: Minimum output length (characters for text, samples for audio)
        max_length: Maximum output length (None = unlimited)
        max_repetition_ratio: Maximum allowed ratio of repeated n-grams (0-1)
        min_unique_ratio: Minimum ratio of unique tokens (0-1)
        check_gibberish: Enable gibberish detection
        check_special_chars: Check for excessive special characters
        strict: Enable strict validation mode
        custom_validators: Additional custom validation functions

    Example:
        >>> validator = OutputValidator(min_length=10, strict=True)
        >>> result = validator.validate_text("Hello world!")
        >>> assert result.is_valid
    """

    def __init__(
        self,
        min_length: int = 1,
        max_length: Optional[int] = None,
        max_repetition_ratio: float = 0.5,
        min_unique_ratio: float = 0.3,
        check_gibberish: bool = True,
        check_special_chars: bool = True,
        strict: bool = False,
        custom_validators: Optional[list] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_repetition_ratio = max_repetition_ratio
        self.min_unique_ratio = min_unique_ratio
        self.check_gibberish = check_gibberish
        self.check_special_chars = check_special_chars
        self.strict = strict
        self.custom_validators = custom_validators or []

    def validate_text(self, text: str) -> ValidationResult:
        """
        Validate text output quality.

        Args:
            text: Text to validate

        Returns:
            ValidationResult with validation outcome

        Example:
            >>> validator = OutputValidator()
            >>> result = validator.validate_text("This is a test.")
            >>> assert result.is_valid
        """
        # Check for None or non-string
        if text is None or not isinstance(text, str):
            return ValidationResult(
                is_valid=False,
                reason="Output is not a string",
                confidence=1.0,
            )

        # Check for empty or whitespace-only
        if len(text.strip()) == 0:
            return ValidationResult(
                is_valid=False,
                reason="Output is empty or whitespace-only",
                confidence=1.0,
            )

        # Check minimum length
        if len(text) < self.min_length:
            return ValidationResult(
                is_valid=False,
                reason=f"Output too short ({len(text)} < {self.min_length})",
                confidence=0.9,
            )

        # Check maximum length
        if self.max_length and len(text) > self.max_length:
            return ValidationResult(
                is_valid=False,
                reason=f"Output too long ({len(text)} > {self.max_length})",
                confidence=0.9,
            )

        # Check for gibberish
        if self.check_gibberish:
            is_gibberish, gibberish_reason = _is_gibberish(text)
            if is_gibberish:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Gibberish detected: {gibberish_reason}",
                    confidence=0.8,
                )

        # Check repetition
        repetition_ratio = _calculate_repetition_ratio(text)
        if repetition_ratio > self.max_repetition_ratio:
            return ValidationResult(
                is_valid=False if self.strict else True,
                reason=f"Excessive repetition ({repetition_ratio:.2f} > {self.max_repetition_ratio})",
                confidence=1.0 - repetition_ratio,
                metadata={'repetition_ratio': repetition_ratio},
            )

        # Check special characters
        if self.check_special_chars:
            special_char_ratio = _calculate_special_char_ratio(text)
            if special_char_ratio > 0.5:  # More than 50% special chars
                return ValidationResult(
                    is_valid=False,
                    reason=f"Too many special characters ({special_char_ratio:.2%})",
                    confidence=0.7,
                )

        # Run custom validators
        for validator_func in self.custom_validators:
            try:
                is_valid, reason = validator_func(text)
                if not is_valid:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Custom validation failed: {reason}",
                        confidence=0.8,
                    )
            except Exception as e:
                logger.warning(f"Custom validator failed: {e}")

        # All checks passed
        return ValidationResult(
            is_valid=True,
            confidence=1.0,
            metadata={
                'length': len(text),
                'repetition_ratio': repetition_ratio,
            },
        )

    def validate_audio(self, waveform: mx.array, sample_rate: int = 16000) -> ValidationResult:
        """
        Validate audio output quality.

        Args:
            waveform: Audio waveform as MLX array
            sample_rate: Sample rate in Hz

        Returns:
            ValidationResult with validation outcome

        Example:
            >>> import mlx.core as mx
            >>> validator = OutputValidator()
            >>> waveform = mx.random.normal((16000,))  # 1 second of audio
            >>> result = validator.validate_audio(waveform)
        """
        # Check for None or invalid type
        if waveform is None:
            return ValidationResult(
                is_valid=False,
                reason="Waveform is None",
                confidence=1.0,
            )

        # Convert to numpy for analysis
        waveform_np = np.array(waveform)

        # Check for all zeros (silence)
        if np.all(waveform_np == 0):
            return ValidationResult(
                is_valid=False,
                reason="Audio is completely silent (all zeros)",
                confidence=1.0,
            )

        # Check for NaN or inf
        if np.any(np.isnan(waveform_np)) or np.any(np.isinf(waveform_np)):
            return ValidationResult(
                is_valid=False,
                reason="Audio contains NaN or infinite values",
                confidence=1.0,
            )

        # Check for clipping (values outside [-1, 1])
        if np.any(waveform_np > 1.0) or np.any(waveform_np < -1.0):
            return ValidationResult(
                is_valid=False,
                reason="Audio is clipping (values outside [-1, 1])",
                confidence=0.9,
            )

        # Check minimum length
        if len(waveform_np) < self.min_length:
            return ValidationResult(
                is_valid=False,
                reason=f"Audio too short ({len(waveform_np)} samples < {self.min_length})",
                confidence=0.9,
            )

        # Check for excessive silence (more than 95% near-zero values)
        silence_threshold = 0.01
        silence_ratio = np.mean(np.abs(waveform_np) < silence_threshold)
        if silence_ratio > 0.95:
            return ValidationResult(
                is_valid=False,
                reason=f"Audio is mostly silent ({silence_ratio:.1%} silence)",
                confidence=0.8,
                metadata={'silence_ratio': silence_ratio},
            )

        # All checks passed
        return ValidationResult(
            is_valid=True,
            confidence=1.0,
            metadata={
                'length_samples': len(waveform_np),
                'duration_seconds': len(waveform_np) / sample_rate,
                'rms': float(np.sqrt(np.mean(waveform_np**2))),
                'peak': float(np.max(np.abs(waveform_np))),
            },
        )

    def validate_tokens(
        self, tokens: list[int], vocab_size: int, eos_token_id: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate token sequence.

        Args:
            tokens: Token IDs
            vocab_size: Vocabulary size
            eos_token_id: End-of-sequence token ID (optional)

        Returns:
            ValidationResult with validation outcome

        Example:
            >>> validator = OutputValidator()
            >>> tokens = [1, 2, 3, 4, 5]
            >>> result = validator.validate_tokens(tokens, vocab_size=50000)
            >>> assert result.is_valid
        """
        # Check for empty
        if not tokens or len(tokens) == 0:
            return ValidationResult(
                is_valid=False,
                reason="Token sequence is empty",
                confidence=1.0,
            )

        # Check for invalid token IDs
        for i, token_id in enumerate(tokens):
            if not isinstance(token_id, (int, np.integer)):
                return ValidationResult(
                    is_valid=False,
                    reason=f"Token {i} is not an integer: {type(token_id)}",
                    confidence=1.0,
                )
            if token_id < 0 or token_id >= vocab_size:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Token {i} out of vocabulary range: {token_id} (vocab_size={vocab_size})",
                    confidence=1.0,
                )

        # Check minimum length
        if len(tokens) < self.min_length:
            return ValidationResult(
                is_valid=False,
                reason=f"Token sequence too short ({len(tokens)} < {self.min_length})",
                confidence=0.9,
            )

        # Check for pathological repetition (same token repeated many times)
        if len(tokens) > 10:
            # Check if any token appears more than 50% of the time
            from collections import Counter

            token_counts = Counter(tokens)
            most_common_token, most_common_count = token_counts.most_common(1)[0]

            # Exclude EOS token from repetition check
            if eos_token_id is not None and most_common_token == eos_token_id:
                if len(token_counts) > 1:
                    most_common_token, most_common_count = token_counts.most_common(2)[1]
                else:
                    most_common_count = 0

            repetition_ratio = most_common_count / len(tokens)
            if repetition_ratio > 0.5:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Pathological repetition: token {most_common_token} appears {repetition_ratio:.1%} of time",
                    confidence=0.9,
                    metadata={'repetition_ratio': repetition_ratio},
                )

        # All checks passed
        unique_ratio = len(set(tokens)) / len(tokens)
        return ValidationResult(
            is_valid=True,
            confidence=1.0,
            metadata={
                'length': len(tokens),
                'unique_tokens': len(set(tokens)),
                'unique_ratio': unique_ratio,
            },
        )


# Convenience functions

def validate_text_output(
    text: str,
    min_length: int = 1,
    max_repetition_ratio: float = 0.5,
    check_gibberish: bool = True,
) -> tuple[bool, Optional[str]]:
    """
    Quick validation of text output.

    Args:
        text: Text to validate
        min_length: Minimum acceptable length
        max_repetition_ratio: Maximum repetition ratio
        check_gibberish: Enable gibberish detection

    Returns:
        Tuple of (is_valid, reason_if_invalid)

    Example:
        >>> is_valid, reason = validate_text_output("Hello world!")
        >>> assert is_valid
        >>>
        >>> is_valid, reason = validate_text_output("")
        >>> assert not is_valid
        >>> assert "empty" in reason.lower()
    """
    validator = OutputValidator(
        min_length=min_length,
        max_repetition_ratio=max_repetition_ratio,
        check_gibberish=check_gibberish,
    )
    result = validator.validate_text(text)
    return result.is_valid, result.reason


def validate_audio_output(waveform: mx.array, sample_rate: int = 16000) -> tuple[bool, Optional[str]]:
    """
    Quick validation of audio output.

    Args:
        waveform: Audio waveform as MLX array
        sample_rate: Sample rate in Hz

    Returns:
        Tuple of (is_valid, reason_if_invalid)

    Example:
        >>> import mlx.core as mx
        >>> waveform = mx.random.normal((16000,)) * 0.5
        >>> is_valid, reason = validate_audio_output(waveform)
    """
    validator = OutputValidator()
    result = validator.validate_audio(waveform, sample_rate)
    return result.is_valid, result.reason


def validate_tokens(
    tokens: list[int], vocab_size: int, eos_token_id: Optional[int] = None
) -> tuple[bool, Optional[str]]:
    """
    Quick validation of token sequence.

    Args:
        tokens: Token IDs
        vocab_size: Vocabulary size
        eos_token_id: End-of-sequence token ID

    Returns:
        Tuple of (is_valid, reason_if_invalid)

    Example:
        >>> is_valid, reason = validate_tokens([1, 2, 3], vocab_size=50000)
        >>> assert is_valid
    """
    validator = OutputValidator()
    result = validator.validate_tokens(tokens, vocab_size, eos_token_id)
    return result.is_valid, result.reason


# Internal helper functions

def _is_gibberish(text: str, threshold: float = 0.6) -> tuple[bool, Optional[str]]:
    """
    Detect if text is gibberish.

    Uses multiple heuristics:
    - Vowel ratio (natural text has ~40% vowels)
    - Character distribution
    - Repeating patterns
    - Unicode category distribution

    Args:
        text: Text to check
        threshold: Confidence threshold for gibberish detection

    Returns:
        Tuple of (is_gibberish, reason)
    """
    if len(text) < 3:
        return False, None

    # Check vowel ratio
    vowels = 'aeiouAEIOU'
    vowel_count = sum(1 for c in text if c in vowels)
    alpha_count = sum(1 for c in text if c.isalpha())

    if alpha_count > 0:
        vowel_ratio = vowel_count / alpha_count
        # Natural English text has vowel ratio around 0.35-0.45
        if vowel_ratio < 0.1 or vowel_ratio > 0.8:
            return True, f"Abnormal vowel ratio: {vowel_ratio:.2f}"

    # Check for excessive special characters at start
    if len(text) > 5:
        first_chars = text[:5]
        if sum(1 for c in first_chars if c in string.punctuation) >= 4:
            return True, "Starts with excessive special characters"

    # Check for Unicode control characters
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\t\r')
    if control_chars > 0:
        return True, "Contains control characters"

    # Check for pathological patterns (same char repeated many times)
    for i in range(len(text) - 4):
        if len(set(text[i : i + 5])) == 1:  # Same char 5 times in a row
            return True, f"Pathological repetition: '{text[i]}' repeated"

    # Check for excessive numbers/punctuation
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) > 10 and non_alnum / len(text) > 0.5:
        return True, f"Too many non-alphanumeric characters ({non_alnum}/{len(text)})"

    return False, None


def _calculate_repetition_ratio(text: str, n: int = 3) -> float:
    """
    Calculate repetition ratio using n-grams.

    Args:
        text: Text to analyze
        n: N-gram size

    Returns:
        Repetition ratio (0-1, higher = more repetitive)
    """
    if len(text) < n:
        return 0.0

    # Extract n-grams
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i : i + n])

    if not ngrams:
        return 0.0

    # Calculate uniqueness ratio
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)

    # Repetition ratio = 1 - (unique / total)
    repetition_ratio = 1.0 - (unique_ngrams / total_ngrams)

    return repetition_ratio


def _calculate_special_char_ratio(text: str) -> float:
    """
    Calculate ratio of special characters in text.

    Args:
        text: Text to analyze

    Returns:
        Ratio of special characters (0-1)
    """
    if len(text) == 0:
        return 0.0

    special_chars = sum(1 for c in text if c in string.punctuation or ord(c) > 127)
    return special_chars / len(text)


__all__ = [
    'ValidationResult',
    'OutputValidator',
    'validate_text_output',
    'validate_audio_output',
    'validate_tokens',
]
