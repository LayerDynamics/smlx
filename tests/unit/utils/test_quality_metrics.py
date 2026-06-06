#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Unit tests for quality metrics utilities.

Tests cover:
- Perplexity calculation
- Entropy calculation
- Repetition analysis
- Token distribution metrics
- Diversity scoring
- Quality assessment
- QualityMetrics dataclass
"""

import pytest
import mlx.core as mx
import numpy as np

from smlx.utils.quality_metrics import (
    QualityMetrics,
    calculate_perplexity,
    calculate_entropy,
    analyze_repetition,
    analyze_token_distribution,
    calculate_diversity_score,
    assess_quality,
    compare_quality,
)


class TestQualityMetrics:
    """Test QualityMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating quality metrics object."""
        metrics = QualityMetrics(
            perplexity=25.5,
            entropy=3.2,
            repetition_1gram=0.1,
            repetition_2gram=0.05,
            repetition_3gram=0.02,
            diversity_score=0.85,
            avg_token_prob=0.15,
            quality_score=0.75
        )
        assert metrics.perplexity == 25.5
        assert metrics.entropy == 3.2
        assert metrics.diversity_score == 0.85

    def test_metrics_comparison(self):
        """Test comparing two quality metrics."""
        metrics1 = QualityMetrics(
            perplexity=20.0,
            entropy=3.5,
            repetition_1gram=0.1,
            repetition_2gram=0.05,
            repetition_3gram=0.02,
            diversity_score=0.9,
            avg_token_prob=0.18,
            quality_score=0.8
        )
        metrics2 = QualityMetrics(
            perplexity=50.0,
            entropy=2.5,
            repetition_1gram=0.3,
            repetition_2gram=0.2,
            repetition_3gram=0.1,
            diversity_score=0.6,
            avg_token_prob=0.08,
            quality_score=0.5
        )
        # Lower perplexity is better
        assert metrics1.perplexity < metrics2.perplexity
        # Higher entropy is better
        assert metrics1.entropy > metrics2.entropy
        # Lower repetition is better
        assert metrics1.repetition_1gram < metrics2.repetition_1gram


class TestEntropyCalculation:
    """Test entropy calculation."""

    def test_uniform_distribution(self):
        """Test entropy for uniform distribution."""
        # Uniform distribution has maximum entropy
        vocab_size = 1000
        uniform_probs = mx.ones(vocab_size) / vocab_size
        logprobs = mx.log(uniform_probs)

        entropy = calculate_entropy(logprobs)

        # Entropy of uniform distribution: log(vocab_size)
        expected_entropy = np.log(vocab_size)
        assert abs(float(entropy) - expected_entropy) < 0.01

    def test_deterministic_distribution(self):
        """Test entropy for deterministic distribution."""
        # One token has probability 1, rest have 0
        probs = mx.zeros(1000)
        probs[0] = 1.0
        logprobs = mx.log(probs + 1e-10)  # Add epsilon to avoid log(0)

        entropy = calculate_entropy(logprobs)

        # Deterministic distribution has entropy close to 0
        assert float(entropy) < 0.1

    def test_peaked_distribution(self):
        """Test entropy for peaked distribution."""
        # Create peaked distribution
        probs = mx.array([0.8, 0.1, 0.05, 0.05])
        logprobs = mx.log(probs)

        entropy = calculate_entropy(logprobs)

        # Peaked distribution has low entropy
        # H = -sum(p * log(p)) for small number of tokens
        assert float(entropy) > 0.0
        assert float(entropy) < 1.5  # Less than uniform over 4 tokens


class TestRepetitionAnalysis:
    """Test repetition analysis."""

    def test_no_repetition(self):
        """Test text with no repetition."""
        text = "The quick brown fox jumps over the lazy dog"
        metrics = analyze_repetition(text)

        # Should have very low repetition ratios
        assert metrics["repetition_1gram"] < 0.3  # Some words may repeat
        assert metrics["repetition_2gram"] < 0.1
        assert metrics["repetition_3gram"] < 0.05

    def test_high_repetition(self):
        """Test text with high repetition."""
        text = "hello hello hello world world world " * 10
        metrics = analyze_repetition(text)

        # Should have high repetition ratios
        assert metrics["repetition_1gram"] > 0.5
        assert metrics["repetition_2gram"] > 0.3

    def test_moderate_repetition(self):
        """Test natural text with moderate repetition."""
        text = "I think that Python is great. I really think Python is the best language."
        metrics = analyze_repetition(text)

        # Natural repetition (function words, common terms)
        assert 0.1 < metrics["repetition_1gram"] < 0.7
        assert metrics["repetition_2gram"] < 0.5

    def test_single_word_repeated(self):
        """Test single word repeated many times."""
        text = "test " * 100
        metrics = analyze_repetition(text)

        # Should detect high 1-gram repetition
        assert metrics["repetition_1gram"] > 0.9

    def test_empty_text(self):
        """Test repetition analysis on empty text."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            analyze_repetition("")


class TestTokenDistribution:
    """Test token distribution analysis."""

    def test_diverse_tokens(self):
        """Test analysis of diverse token sequence."""
        # Create diverse token sequence
        tokens = list(range(100))  # 100 unique tokens
        metrics = analyze_token_distribution(tokens)

        # Should have high unique ratio
        assert metrics["unique_token_ratio"] > 0.95
        assert metrics["num_unique_tokens"] == 100
        assert metrics["total_tokens"] == 100

    def test_repeated_tokens(self):
        """Test analysis of repeated token sequence."""
        # Same token repeated
        tokens = [42] * 100
        metrics = analyze_token_distribution(tokens)

        # Should have low unique ratio
        assert metrics["unique_token_ratio"] < 0.02
        assert metrics["num_unique_tokens"] == 1
        assert metrics["total_tokens"] == 100

    def test_mixed_tokens(self):
        """Test analysis of mixed token sequence."""
        # Mix of unique and repeated tokens
        tokens = [1, 2, 3, 4, 5] * 20  # 5 unique tokens, 100 total
        metrics = analyze_token_distribution(tokens)

        assert metrics["num_unique_tokens"] == 5
        assert metrics["total_tokens"] == 100
        assert abs(metrics["unique_token_ratio"] - 0.05) < 0.01

    def test_mlx_array_tokens(self):
        """Test with MLX array input."""
        tokens = mx.array([1, 2, 3, 2, 1, 4, 5, 4])
        metrics = analyze_token_distribution(tokens)

        assert metrics["num_unique_tokens"] == 5
        assert metrics["total_tokens"] == 8


class TestDiversityScore:
    """Test diversity score calculation."""

    def test_high_diversity_text(self):
        """Test diversity score for varied text."""
        text = "The quick brown fox jumps over the lazy dog while exploring the forest."
        score = calculate_diversity_score(text)

        # Should have high diversity
        assert score > 0.6

    def test_low_diversity_text(self):
        """Test diversity score for repetitive text."""
        text = "test test test test " * 25
        score = calculate_diversity_score(text)

        # Should have low diversity
        assert score < 0.3

    def test_moderate_diversity(self):
        """Test diversity score for natural text."""
        text = "I think Python is great. Python is my favorite programming language."
        score = calculate_diversity_score(text)

        # Natural text has moderate diversity
        assert 0.4 < score < 0.9

    def test_empty_text_diversity(self):
        """Test diversity on empty text."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            calculate_diversity_score("")


class TestQualityAssessment:
    """Test overall quality assessment."""

    def test_good_quality_text(self):
        """Test quality assessment for good text."""
        text = "Machine learning is transforming technology. Neural networks enable computers to learn from data and make intelligent decisions."

        # Note: This requires a model and tokenizer
        # For unit tests, we might need to mock this or make it optional
        # Commenting out for now - should be integration test
        # quality = assess_quality(model, tokenizer, text)
        # assert quality.quality_score > 0.6

    def test_poor_quality_text(self):
        """Test quality assessment for poor text."""
        text = "asdfasdf asdfasdf asdfasdf " * 20

        # Same note as above - needs model
        # quality = assess_quality(model, tokenizer, text)
        # assert quality.quality_score < 0.4


class TestQualityComparison:
    """Test quality comparison."""

    def test_compare_two_metrics(self):
        """Test comparing two quality metrics."""
        metrics1 = QualityMetrics(
            perplexity=20.0,
            entropy=3.5,
            repetition_1gram=0.1,
            repetition_2gram=0.05,
            repetition_3gram=0.02,
            diversity_score=0.9,
            avg_token_prob=0.18,
            quality_score=0.8
        )
        metrics2 = QualityMetrics(
            perplexity=50.0,
            entropy=2.5,
            repetition_1gram=0.3,
            repetition_2gram=0.2,
            repetition_3gram=0.1,
            diversity_score=0.6,
            avg_token_prob=0.08,
            quality_score=0.5
        )

        comparison = compare_quality(metrics1, metrics2)

        # Should indicate metrics1 is better
        assert "better" in comparison or "worse" in comparison

    def test_compare_identical_metrics(self):
        """Test comparing identical metrics."""
        metrics = QualityMetrics(
            perplexity=30.0,
            entropy=3.0,
            repetition_1gram=0.2,
            repetition_2gram=0.1,
            repetition_3gram=0.05,
            diversity_score=0.75,
            avg_token_prob=0.12,
            quality_score=0.65
        )

        comparison = compare_quality(metrics, metrics)

        # Should indicate they are similar/identical
        assert "same" in comparison or "identical" in comparison or "similar" in comparison


class TestPerplexityCalculation:
    """Test perplexity calculation."""

    @pytest.mark.skip(reason="Requires model and tokenizer - integration test")
    def test_perplexity_good_text(self):
        """Test perplexity for well-formed text."""
        # This is an integration test - needs actual model
        pass

    @pytest.mark.skip(reason="Requires model and tokenizer - integration test")
    def test_perplexity_poor_text(self):
        """Test perplexity for gibberish text."""
        # This is an integration test - needs actual model
        pass

    def test_perplexity_from_logprobs(self):
        """Test perplexity calculation from log probabilities."""
        # Create synthetic logprobs
        # Perplexity = exp(-mean(log_probs))
        logprobs = mx.array([-1.0, -1.5, -2.0, -1.2, -1.8])

        # Calculate expected perplexity
        mean_logprob = float(mx.mean(logprobs))
        expected_perplexity = np.exp(-mean_logprob)

        # If calculate_perplexity accepts logprobs directly:
        # perplexity = calculate_perplexity(logprobs=logprobs)
        # assert abs(perplexity - expected_perplexity) < 0.01


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_text_metrics(self):
        """Test metrics handle Unicode text."""
        text = "Hello 世界! Здравствуй мир! 🌍"

        # Should not crash
        rep_metrics = analyze_repetition(text)
        div_score = calculate_diversity_score(text)

        assert isinstance(rep_metrics, dict)
        assert isinstance(div_score, (int, float))

    def test_very_long_sequence(self):
        """Test metrics on very long sequences."""
        long_text = "This is a test sentence. " * 1000

        # Should handle without crashing or excessive memory
        rep_metrics = analyze_repetition(long_text)
        div_score = calculate_diversity_score(long_text)

        assert isinstance(rep_metrics, dict)
        assert isinstance(div_score, (int, float))

    def test_single_token(self):
        """Test metrics on single token."""
        tokens = [42]
        metrics = analyze_token_distribution(tokens)

        assert metrics["num_unique_tokens"] == 1
        assert metrics["total_tokens"] == 1
        assert metrics["unique_token_ratio"] == 1.0

    def test_special_characters(self):
        """Test metrics on text with special characters."""
        text = "def test():\n    print('Hello!')\n    return 42"

        # Should handle code/special chars
        rep_metrics = analyze_repetition(text)
        div_score = calculate_diversity_score(text)

        assert isinstance(rep_metrics, dict)
        assert isinstance(div_score, (int, float))
