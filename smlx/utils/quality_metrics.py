#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Quality metrics for evaluating model output quality.

Provides quantitative metrics to measure the quality of model outputs including:
- Perplexity (language model confidence)
- Entropy (token distribution diversity)
- Repetition analysis (n-gram repetition detection)
- Token distribution metrics
- Output diversity measures

These metrics are used to:
1. Detect degraded model outputs (gibberish, low quality)
2. Compare pre/post-quantization quality
3. Monitor model behavior in production
4. Validate generation parameters

Example:
    >>> from smlx.utils.quality_metrics import calculate_perplexity, analyze_repetition
    >>> from smlx.models.SmolLM2_135M import load
    >>>
    >>> model, tokenizer = load()
    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> perplexity = calculate_perplexity(model, tokenizer, text)
    >>> repetition_metrics = analyze_repetition(text)
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """
    Comprehensive quality metrics for model output.

    Attributes:
        perplexity: Language model perplexity (lower = better)
        entropy: Token distribution entropy (higher = more diverse)
        repetition_1gram: 1-gram repetition ratio
        repetition_2gram: 2-gram repetition ratio
        repetition_3gram: 3-gram repetition ratio
        unique_token_ratio: Ratio of unique tokens to total tokens
        vocab_diversity: Number of unique tokens used
        avg_token_probability: Average probability of generated tokens
        is_high_quality: Overall quality assessment
        metadata: Additional metadata
    """

    perplexity: Optional[float] = None
    entropy: Optional[float] = None
    repetition_1gram: Optional[float] = None
    repetition_2gram: Optional[float] = None
    repetition_3gram: Optional[float] = None
    unique_token_ratio: Optional[float] = None
    vocab_diversity: Optional[int] = None
    avg_token_probability: Optional[float] = None
    is_high_quality: bool = True
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'perplexity': self.perplexity,
            'entropy': self.entropy,
            'repetition_1gram': self.repetition_1gram,
            'repetition_2gram': self.repetition_2gram,
            'repetition_3gram': self.repetition_3gram,
            'unique_token_ratio': self.unique_token_ratio,
            'vocab_diversity': self.vocab_diversity,
            'avg_token_probability': self.avg_token_probability,
            'is_high_quality': self.is_high_quality,
            **self.metadata,
        }


def calculate_perplexity(
    model: Any,
    tokenizer: Any,
    text: str,
    context: Optional[str] = None,
) -> float:
    """
    Calculate perplexity of text using language model.

    Perplexity measures how well the model predicts the text. Lower perplexity
    indicates the text is more likely according to the model. Very high perplexity
    (>1000) often indicates gibberish or out-of-distribution text.

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Text to evaluate
        context: Optional context/prompt (for conditional perplexity)

    Returns:
        Perplexity value (lower = better, typical range: 10-100 for good text)

    Example:
        >>> from smlx.models.SmolLM2_135M import load
        >>> model, tokenizer = load()
        >>> ppl = calculate_perplexity(model, tokenizer, "Hello world!")
        >>> assert ppl < 100  # Good text should have low perplexity
    """
    try:
        # Encode text
        if context:
            full_text = context + text
            context_tokens = tokenizer.encode(context)
            context_len = len(context_tokens)
        else:
            full_text = text
            context_len = 0

        tokens = tokenizer.encode(full_text)

        if len(tokens) <= context_len:
            logger.warning("Text is too short for perplexity calculation")
            return float('inf')

        # Convert to MLX array
        tokens_mx = mx.array(tokens)[None, :]  # Add batch dimension

        # Get model logits
        with mx.no_grad():
            logits = model(tokens_mx)

        # Calculate cross-entropy loss
        # Shift logits and tokens for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_tokens = tokens_mx[:, 1:]

        # Calculate log probabilities
        log_probs = mx.log_softmax(shift_logits, axis=-1)

        # Get log prob of actual next tokens
        batch_size, seq_len, vocab_size = shift_logits.shape
        token_log_probs = log_probs[0, range(seq_len), shift_tokens[0]]

        # Only use tokens after context
        if context_len > 0:
            token_log_probs = token_log_probs[context_len:]

        # Calculate mean negative log-likelihood
        nll = -mx.mean(token_log_probs)

        # Perplexity = exp(nll)
        perplexity = float(mx.exp(nll).item())

        return perplexity

    except Exception as e:
        logger.error(f"Failed to calculate perplexity: {e}")
        return float('inf')


def calculate_entropy(logits: mx.array, temperature: float = 1.0) -> float:
    """
    Calculate entropy of token distribution.

    Entropy measures the diversity/randomness of the token distribution.
    Higher entropy means more diverse predictions (less confident model).
    Lower entropy means peaked distribution (very confident model).

    Args:
        logits: Model logits [vocab_size] or [batch, vocab_size]
        temperature: Temperature for scaling logits

    Returns:
        Entropy value in bits (typical range: 1-15 for language models)

    Example:
        >>> import mlx.core as mx
        >>> logits = mx.random.normal((50000,))  # Random logits
        >>> entropy = calculate_entropy(logits)
        >>> assert entropy > 0  # Entropy should be positive
    """
    # Handle batch dimension
    if logits.ndim > 1:
        logits = logits[0]

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Calculate probabilities
    probs = mx.softmax(logits, axis=-1)

    # Calculate entropy: -sum(p * log2(p))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    log_probs = mx.log2(probs + epsilon)
    entropy = -mx.sum(probs * log_probs)

    return float(entropy.item())


def analyze_repetition(
    text: str, max_n: int = 4
) -> dict[str, float]:
    """
    Analyze repetition at multiple n-gram levels.

    Measures how repetitive the text is by calculating the ratio of unique
    n-grams to total n-grams. Lower ratios indicate more repetition.

    Args:
        text: Text to analyze
        max_n: Maximum n-gram size to analyze

    Returns:
        Dictionary with repetition metrics for each n-gram size

    Example:
        >>> text = "the the the cat sat on the mat"
        >>> metrics = analyze_repetition(text)
        >>> assert metrics['repetition_1gram'] > 0  # "the" repeats
    """
    words = text.split()

    if len(words) == 0:
        return {f'repetition_{n}gram': 0.0 for n in range(1, max_n + 1)}

    metrics = {}

    for n in range(1, max_n + 1):
        if len(words) < n:
            metrics[f'repetition_{n}gram'] = 0.0
            continue

        # Extract n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i : i + n])
            ngrams.append(ngram)

        if not ngrams:
            metrics[f'repetition_{n}gram'] = 0.0
            continue

        # Calculate repetition ratio
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        repetition_ratio = 1.0 - (unique_ngrams / total_ngrams)

        metrics[f'repetition_{n}gram'] = repetition_ratio

    return metrics


def analyze_token_distribution(tokens: list[int]) -> dict[str, Any]:
    """
    Analyze statistical properties of token distribution.

    Args:
        tokens: Token IDs

    Returns:
        Dictionary with distribution metrics

    Example:
        >>> tokens = [1, 2, 3, 1, 2, 1]
        >>> metrics = analyze_token_distribution(tokens)
        >>> assert metrics['unique_ratio'] < 1.0  # Has repeated tokens
    """
    if not tokens:
        return {
            'unique_count': 0,
            'total_count': 0,
            'unique_ratio': 0.0,
            'most_common_token': None,
            'most_common_count': 0,
            'most_common_ratio': 0.0,
        }

    token_counts = Counter(tokens)
    unique_count = len(token_counts)
    total_count = len(tokens)

    most_common_token, most_common_count = token_counts.most_common(1)[0]

    return {
        'unique_count': unique_count,
        'total_count': total_count,
        'unique_ratio': unique_count / total_count,
        'most_common_token': most_common_token,
        'most_common_count': most_common_count,
        'most_common_ratio': most_common_count / total_count,
    }


def calculate_diversity_score(
    tokens: list[int],
    text: str,
    vocab_size: int,
) -> float:
    """
    Calculate overall diversity score for generated text.

    Combines multiple metrics into a single score measuring how diverse and
    non-repetitive the output is. Score ranges from 0 (very repetitive) to
    1 (very diverse).

    Args:
        tokens: Generated token IDs
        text: Generated text
        vocab_size: Total vocabulary size

    Returns:
        Diversity score (0-1, higher = more diverse)

    Example:
        >>> tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> text = "one two three four five six seven eight nine ten"
        >>> score = calculate_diversity_score(tokens, text, vocab_size=50000)
        >>> assert score > 0.5  # Diverse text
    """
    if not tokens or not text:
        return 0.0

    # Token-level diversity
    unique_ratio = len(set(tokens)) / len(tokens)

    # N-gram diversity (use 2-grams and 3-grams)
    repetition_metrics = analyze_repetition(text, max_n=3)
    avg_repetition = (
        repetition_metrics.get('repetition_2gram', 0) +
        repetition_metrics.get('repetition_3gram', 0)
    ) / 2.0

    # Vocabulary coverage (what fraction of vocab is used)
    vocab_coverage = len(set(tokens)) / vocab_size

    # Combine metrics (weighted average)
    diversity_score = (
        0.4 * unique_ratio +              # 40% weight on unique tokens
        0.4 * (1.0 - avg_repetition) +    # 40% weight on low repetition
        0.2 * vocab_coverage              # 20% weight on vocab coverage
    )

    return float(np.clip(diversity_score, 0.0, 1.0))


def assess_quality(
    model: Any,
    tokenizer: Any,
    text: str,
    tokens: Optional[list[int]] = None,
    perplexity_threshold: float = 500.0,
    repetition_threshold: float = 0.6,
    min_diversity: float = 0.2,
) -> QualityMetrics:
    """
    Comprehensive quality assessment of generated text.

    Combines multiple metrics to determine if output is high quality.
    Returns detailed metrics and overall quality boolean.

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Generated text
        tokens: Generated tokens (optional, will encode if not provided)
        perplexity_threshold: Maximum acceptable perplexity
        repetition_threshold: Maximum acceptable repetition ratio
        min_diversity: Minimum acceptable diversity score

    Returns:
        QualityMetrics with comprehensive quality assessment

    Example:
        >>> from smlx.models.SmolLM2_135M import load
        >>> model, tokenizer = load()
        >>> text = "The cat sat on the mat."
        >>> metrics = assess_quality(model, tokenizer, text)
        >>> assert metrics.is_high_quality
    """
    # Encode tokens if not provided
    if tokens is None:
        tokens = tokenizer.encode(text)

    # Calculate perplexity
    try:
        perplexity = calculate_perplexity(model, tokenizer, text)
    except Exception as e:
        logger.warning(f"Could not calculate perplexity: {e}")
        perplexity = None

    # Analyze repetition
    repetition_metrics = analyze_repetition(text, max_n=3)

    # Analyze token distribution
    token_metrics = analyze_token_distribution(tokens)

    # Calculate diversity
    try:
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50000
        diversity_score = calculate_diversity_score(tokens, text, vocab_size)
    except Exception:
        diversity_score = None

    # Determine if high quality
    is_high_quality = True
    quality_reasons = []

    if perplexity is not None and perplexity > perplexity_threshold:
        is_high_quality = False
        quality_reasons.append(f"High perplexity ({perplexity:.1f})")

    if repetition_metrics.get('repetition_3gram', 0) > repetition_threshold:
        is_high_quality = False
        quality_reasons.append(
            f"High repetition ({repetition_metrics['repetition_3gram']:.2f})"
        )

    if diversity_score is not None and diversity_score < min_diversity:
        is_high_quality = False
        quality_reasons.append(f"Low diversity ({diversity_score:.2f})")

    if token_metrics['most_common_ratio'] > 0.5:
        is_high_quality = False
        quality_reasons.append(
            f"One token dominates ({token_metrics['most_common_ratio']:.1%})"
        )

    return QualityMetrics(
        perplexity=perplexity,
        entropy=None,  # Could add if logits available
        repetition_1gram=repetition_metrics.get('repetition_1gram'),
        repetition_2gram=repetition_metrics.get('repetition_2gram'),
        repetition_3gram=repetition_metrics.get('repetition_3gram'),
        unique_token_ratio=token_metrics['unique_ratio'],
        vocab_diversity=token_metrics['unique_count'],
        avg_token_probability=None,
        is_high_quality=is_high_quality,
        metadata={
            'quality_reasons': quality_reasons if not is_high_quality else [],
            'text_length': len(text),
            'token_count': len(tokens),
        },
    )


def compare_quality(
    metrics1: QualityMetrics,
    metrics2: QualityMetrics,
    tolerance: float = 0.1,
) -> dict[str, Any]:
    """
    Compare two quality metric sets (e.g., before/after quantization).

    Args:
        metrics1: First metrics (e.g., full precision)
        metrics2: Second metrics (e.g., quantized)
        tolerance: Acceptable degradation ratio (0.1 = 10%)

    Returns:
        Dictionary with comparison results

    Example:
        >>> # Compare original vs quantized model output
        >>> original_metrics = assess_quality(model_fp16, tokenizer, text)
        >>> quant_metrics = assess_quality(model_4bit, tokenizer, text)
        >>> comparison = compare_quality(original_metrics, quant_metrics)
        >>> assert comparison['acceptable']  # Quality maintained
    """
    comparison = {
        'acceptable': True,
        'degradations': [],
        'improvements': [],
    }

    # Compare perplexity
    if metrics1.perplexity and metrics2.perplexity:
        ppl_change = (metrics2.perplexity - metrics1.perplexity) / metrics1.perplexity
        if ppl_change > tolerance:
            comparison['acceptable'] = False
            comparison['degradations'].append(
                f"Perplexity increased {ppl_change:.1%} "
                f"({metrics1.perplexity:.1f} → {metrics2.perplexity:.1f})"
            )
        elif ppl_change < -tolerance:
            comparison['improvements'].append(
                f"Perplexity decreased {abs(ppl_change):.1%}"
            )

    # Compare repetition
    if metrics1.repetition_3gram and metrics2.repetition_3gram:
        rep_change = metrics2.repetition_3gram - metrics1.repetition_3gram
        if rep_change > tolerance:
            comparison['acceptable'] = False
            comparison['degradations'].append(
                f"Repetition increased {rep_change:.1%}"
            )

    # Compare diversity
    if metrics1.unique_token_ratio and metrics2.unique_token_ratio:
        div_change = (
            (metrics2.unique_token_ratio - metrics1.unique_token_ratio) /
            metrics1.unique_token_ratio
        )
        if div_change < -tolerance:
            comparison['acceptable'] = False
            comparison['degradations'].append(
                f"Token diversity decreased {abs(div_change):.1%}"
            )

    return comparison


__all__ = [
    'QualityMetrics',
    'calculate_perplexity',
    'calculate_entropy',
    'analyze_repetition',
    'analyze_token_distribution',
    'calculate_diversity_score',
    'assess_quality',
    'compare_quality',
]
