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
    diversity_score: Optional[float] = None
    unique_token_ratio: Optional[float] = None
    vocab_diversity: Optional[int] = None
    avg_token_probability: Optional[float] = None
    avg_token_prob: Optional[float] = None
    quality_score: Optional[float] = None
    is_high_quality: bool = True
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # ``avg_token_prob`` is the short public alias for ``avg_token_probability``;
        # keep the two in sync regardless of which one the caller supplied.
        if self.avg_token_prob is None and self.avg_token_probability is not None:
            self.avg_token_prob = self.avg_token_probability
        elif self.avg_token_probability is None and self.avg_token_prob is not None:
            self.avg_token_probability = self.avg_token_prob

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "perplexity": self.perplexity,
            "entropy": self.entropy,
            "repetition_1gram": self.repetition_1gram,
            "repetition_2gram": self.repetition_2gram,
            "repetition_3gram": self.repetition_3gram,
            "diversity_score": self.diversity_score,
            "unique_token_ratio": self.unique_token_ratio,
            "vocab_diversity": self.vocab_diversity,
            "avg_token_probability": self.avg_token_probability,
            "avg_token_prob": self.avg_token_prob,
            "quality_score": self.quality_score,
            "is_high_quality": self.is_high_quality,
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
            return float("inf")

        # Convert to MLX array
        tokens_mx = mx.array(tokens)[None, :]  # Add batch dimension

        # Get model logits. MLX has no `mx.no_grad()` context manager (and is
        # lazy with no autograd tape unless gradients are explicitly requested),
        # so a plain forward pass is already gradient-free.
        logits = model(tokens_mx)

        # Calculate cross-entropy loss
        # Shift logits and tokens for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_tokens = tokens_mx[:, 1:]

        # A 1-token sequence has nothing to predict after shifting; perplexity is
        # undefined, so report infinity rather than reducing over an empty array.
        if shift_logits.shape[1] == 0:
            logger.warning("Sequence too short for perplexity (need >1 token)")
            return float("inf")

        # Calculate log probabilities. MLX core has no `mx.log_softmax`, so
        # compute it stably as logits - logsumexp(logits).
        log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)

        # Get log prob of actual next tokens. MLX cannot index with a Python
        # ``range``; gather each position's target-token log-prob with
        # ``take_along_axis`` (works on the seq x vocab matrix).
        seq_log_probs = log_probs[0]  # (seq_len, vocab)
        targets = shift_tokens[0][:, None]  # (seq_len, 1)
        token_log_probs = mx.take_along_axis(seq_log_probs, targets, axis=-1)[:, 0]

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
        return float("inf")


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
        Entropy value in nats — natural-log units, so a uniform distribution
        over ``V`` tokens has entropy ``ln(V)`` (typical range: ~1-12 for
        language-model vocabularies).

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

    # Calculate entropy in nats: -sum(p * ln(p)). Natural log (not log2) so the
    # uniform-distribution entropy equals ln(vocab_size), the convention callers
    # and tests expect. Add a small epsilon to avoid log(0).
    epsilon = 1e-10
    log_probs = mx.log(probs + epsilon)
    entropy = -mx.sum(probs * log_probs)

    return float(entropy.item())


def analyze_repetition(text: str, max_n: int = 4) -> dict[str, float]:
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
        # Repetition is undefined for empty/whitespace-only text; signal the
        # caller error rather than returning misleading all-zero ratios.
        raise ValueError("Cannot analyze repetition of empty text")

    metrics = {}

    for n in range(1, max_n + 1):
        if len(words) < n:
            metrics[f"repetition_{n}gram"] = 0.0
            continue

        # Extract n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i : i + n])
            ngrams.append(ngram)

        if not ngrams:
            metrics[f"repetition_{n}gram"] = 0.0
            continue

        # Calculate repetition ratio
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        repetition_ratio = 1.0 - (unique_ngrams / total_ngrams)

        metrics[f"repetition_{n}gram"] = repetition_ratio

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
    # Normalize array-like inputs (MLX/NumPy) to a plain list of Python ints so
    # Counter/len behave correctly (iterating an MLX array yields 0-d arrays,
    # which are unhashable and break Counter).
    if isinstance(tokens, mx.array):
        tokens = tokens.tolist()
    elif isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()

    if not tokens or len(tokens) == 0:
        return {
            # New canonical keys
            "num_unique_tokens": 0,
            "total_tokens": 0,
            "unique_token_ratio": 0.0,
            # Back-compat aliases (used by assess_quality)
            "unique_count": 0,
            "total_count": 0,
            "unique_ratio": 0.0,
            "most_common_token": None,
            "most_common_count": 0,
            "most_common_ratio": 0.0,
        }

    token_counts = Counter(tokens)
    unique_count = len(token_counts)
    total_count = len(tokens)
    unique_ratio = unique_count / total_count

    most_common_token, most_common_count = token_counts.most_common(1)[0]

    return {
        # New canonical keys
        "num_unique_tokens": unique_count,
        "total_tokens": total_count,
        "unique_token_ratio": unique_ratio,
        # Back-compat aliases (used by assess_quality)
        "unique_count": unique_count,
        "total_count": total_count,
        "unique_ratio": unique_ratio,
        "most_common_token": most_common_token,
        "most_common_count": most_common_count,
        "most_common_ratio": most_common_count / total_count,
    }


def calculate_diversity_score(
    text: str,
    tokens: Optional[list[int]] = None,
    vocab_size: Optional[int] = None,
) -> float:
    """
    Calculate an overall diversity score for generated text.

    Measures how diverse and non-repetitive the output is, from 0 (very
    repetitive) to 1 (very diverse). The score is computed from the text itself
    (unique-word ratio + low n-gram repetition). When token IDs and a vocab
    size are also supplied, token-level uniqueness and vocabulary coverage are
    blended in for a richer score.

    Args:
        text: Generated text (required).
        tokens: Generated token IDs (optional; enables token-level signals).
        vocab_size: Total vocabulary size (optional; enables vocab coverage).

    Returns:
        Diversity score (0-1, higher = more diverse).

    Raises:
        ValueError: If ``text`` is empty or whitespace-only.

    Example:
        >>> score = calculate_diversity_score("one two three four five")
        >>> assert score > 0.5  # Diverse text
    """
    words = text.split() if text else []
    if not words:
        # Diversity is undefined for empty text; surface the caller error.
        raise ValueError("Cannot compute diversity score for empty text")

    # Word-level diversity (case-insensitive so "The"/"the" count as one word)
    unique_word_ratio = len({w.lower() for w in words}) / len(words)

    # N-gram repetition (2- and 3-grams); higher = more repetitive
    repetition_metrics = analyze_repetition(text, max_n=3)
    avg_repetition = (
        repetition_metrics.get("repetition_2gram", 0.0)
        + repetition_metrics.get("repetition_3gram", 0.0)
    ) / 2.0

    if tokens is not None and len(tokens) > 0 and vocab_size:
        # Richer token-aware score (mirrors the original behaviour): blend token
        # uniqueness, low repetition, and vocabulary coverage.
        unique_token_ratio = len(set(tokens)) / len(tokens)
        vocab_coverage = len(set(tokens)) / vocab_size
        diversity_score = (
            0.4 * unique_token_ratio + 0.4 * (1.0 - avg_repetition) + 0.2 * vocab_coverage
        )
    else:
        # Text-only score: unique-word ratio + low repetition.
        diversity_score = 0.6 * unique_word_ratio + 0.4 * (1.0 - avg_repetition)

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
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50000
        diversity_score = calculate_diversity_score(text, tokens=tokens, vocab_size=vocab_size)
    except Exception:
        diversity_score = None

    # Determine if high quality
    is_high_quality = True
    quality_reasons = []

    if perplexity is not None and perplexity > perplexity_threshold:
        is_high_quality = False
        quality_reasons.append(f"High perplexity ({perplexity:.1f})")

    if repetition_metrics.get("repetition_3gram", 0) > repetition_threshold:
        is_high_quality = False
        quality_reasons.append(f"High repetition ({repetition_metrics['repetition_3gram']:.2f})")

    if diversity_score is not None and diversity_score < min_diversity:
        is_high_quality = False
        quality_reasons.append(f"Low diversity ({diversity_score:.2f})")

    if token_metrics["most_common_ratio"] > 0.5:
        is_high_quality = False
        quality_reasons.append(f"One token dominates ({token_metrics['most_common_ratio']:.1%})")

    return QualityMetrics(
        perplexity=perplexity,
        entropy=None,  # Could add if logits available
        repetition_1gram=repetition_metrics.get("repetition_1gram"),
        repetition_2gram=repetition_metrics.get("repetition_2gram"),
        repetition_3gram=repetition_metrics.get("repetition_3gram"),
        diversity_score=diversity_score,
        unique_token_ratio=token_metrics["unique_ratio"],
        vocab_diversity=token_metrics["unique_count"],
        avg_token_probability=None,
        is_high_quality=is_high_quality,
        metadata={
            "quality_reasons": quality_reasons if not is_high_quality else [],
            "text_length": len(text),
            "token_count": len(tokens),
        },
    )


def _quality_goodness(m: QualityMetrics) -> float:
    """Collapse a QualityMetrics into a single 0-1 'goodness' score.

    Prefers an explicit ``quality_score`` when present; otherwise builds a
    composite from whatever individual signals are available (lower perplexity
    and repetition are better; higher entropy, diversity, and token uniqueness
    are better).
    """
    if m.quality_score is not None:
        return float(m.quality_score)

    parts = []
    if m.diversity_score is not None:
        parts.append(float(m.diversity_score))
    if m.unique_token_ratio is not None:
        parts.append(float(m.unique_token_ratio))
    if m.entropy is not None:
        # Map entropy (nats) into ~[0, 1]; ln(vocab) for common vocabs is ~10.
        parts.append(float(min(m.entropy / 10.0, 1.0)))
    if m.repetition_3gram is not None:
        parts.append(1.0 - float(m.repetition_3gram))
    if m.perplexity is not None:
        # Lower perplexity -> higher goodness; 50 is a "decent text" anchor.
        parts.append(1.0 / (1.0 + float(m.perplexity) / 50.0))

    return sum(parts) / len(parts) if parts else 0.0


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
        tolerance: Acceptable degradation ratio / goodness gap (0.1 = 10%).

    Returns:
        A dict describing the comparison. Always contains ``acceptable`` (bool),
        ``degradations`` / ``improvements`` (lists), the per-set ``goodness``
        scores, and a ``verdict`` string. It additionally carries the verdict as
        membership keys — ``"better"``/``"worse"`` when one set wins, or
        ``"similar"``/``"identical"``/``"same"`` when they are equivalent — so
        callers can write ``"better" in result`` as well as ``result["acceptable"]``.

    Example:
        >>> original_metrics = assess_quality(model_fp16, tokenizer, text)
        >>> quant_metrics = assess_quality(model_4bit, tokenizer, text)
        >>> comparison = compare_quality(original_metrics, quant_metrics)
        >>> assert comparison["acceptable"]
    """
    g1 = _quality_goodness(metrics1)
    g2 = _quality_goodness(metrics2)
    diff = g1 - g2

    degradations: list[str] = []
    improvements: list[str] = []

    # Perplexity (lower is better)
    if metrics1.perplexity and metrics2.perplexity:
        ppl_change = (metrics2.perplexity - metrics1.perplexity) / metrics1.perplexity
        if ppl_change > tolerance:
            degradations.append(
                f"Perplexity increased {ppl_change:.1%} "
                f"({metrics1.perplexity:.1f} -> {metrics2.perplexity:.1f})"
            )
        elif ppl_change < -tolerance:
            improvements.append(f"Perplexity decreased {abs(ppl_change):.1%}")

    # Repetition (lower is better)
    if metrics1.repetition_3gram is not None and metrics2.repetition_3gram is not None:
        rep_change = metrics2.repetition_3gram - metrics1.repetition_3gram
        if rep_change > tolerance:
            degradations.append(f"Repetition increased {rep_change:.1%}")
        elif rep_change < -tolerance:
            improvements.append(f"Repetition decreased {abs(rep_change):.1%}")

    # Token diversity (higher is better)
    if metrics1.unique_token_ratio and metrics2.unique_token_ratio:
        div_change = (
            metrics2.unique_token_ratio - metrics1.unique_token_ratio
        ) / metrics1.unique_token_ratio
        if div_change < -tolerance:
            degradations.append(f"Token diversity decreased {abs(div_change):.1%}")
        elif div_change > tolerance:
            improvements.append(f"Token diversity increased {div_change:.1%}")

    result: dict[str, Any] = {
        "goodness_first": g1,
        "goodness_second": g2,
        "difference": diff,
        "degradations": degradations,
        "improvements": improvements,
        "acceptable": len(degradations) == 0,
    }

    # Encode the verdict both as a string and as membership keys so callers can
    # use either ``result["acceptable"]`` (dict access) or ``"better" in result``
    # (key membership) styles.
    if abs(diff) < tolerance:
        result["verdict"] = "similar"
        result["similar"] = True
        if g1 == g2:
            result["identical"] = True
            result["same"] = True
    elif diff > 0:
        result["verdict"] = "first_better"
        result["better"] = "first"
        result["worse"] = "second"
    else:
        result["verdict"] = "second_better"
        result["better"] = "second"
        result["worse"] = "first"

    return result


__all__ = [
    "QualityMetrics",
    "calculate_perplexity",
    "calculate_entropy",
    "analyze_repetition",
    "analyze_token_distribution",
    "calculate_diversity_score",
    "assess_quality",
    "compare_quality",
]
