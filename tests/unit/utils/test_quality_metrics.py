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

import mlx.core as mx
import numpy as np
import pytest

from smlx.utils.quality_metrics import (
    QualityMetrics,
    analyze_repetition,
    analyze_token_distribution,
    assess_quality,
    calculate_diversity_score,
    calculate_entropy,
    calculate_perplexity,
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


class _WordTokenizer:
    """Deterministic word-level tokenizer for unit tests.

    Each distinct word maps to a stable integer id (assigned by first
    appearance), so repeated text collapses to repeated ids — exactly the
    signal ``assess_quality`` uses for repetition/diversity. ``encode`` is pure,
    so the two calls ``assess_quality`` makes (one direct, one inside
    ``calculate_perplexity``) agree.
    """

    vocab_size = 256

    def encode(self, text):
        vocab = {}
        ids = []
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1  # ids start at 1, all < vocab_size
            ids.append(vocab[word])
        return ids or [1]


class _OracleModel:
    """Toy language model that puts almost all probability on the *actual* next
    token, so a well-formed sequence scores a near-1 (low) perplexity. Lets the
    unit tests exercise the real ``assess_quality`` perplexity path without
    downloading a model.
    """

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, tokens):
        toks = tokens[0].tolist()
        seq_len = len(toks)
        logits = np.full((1, seq_len, self.vocab_size), -20.0, dtype=np.float32)
        for i in range(seq_len):
            # Peak each position on the token that actually follows it (the last
            # position has no successor, so peak on itself — it is never scored).
            target = toks[i + 1] if i + 1 < seq_len else toks[i]
            logits[0, i, target] = 20.0
        return mx.array(logits)


class TestQualityAssessment:
    """Test overall quality assessment."""

    def test_good_quality_text(self):
        """Well-formed text with a confident model is rated high quality."""
        text = (
            "Machine learning is transforming technology. Neural networks enable "
            "computers to learn from data and make intelligent decisions."
        )

        tokenizer = _WordTokenizer()
        model = _OracleModel(tokenizer.vocab_size)
        quality = assess_quality(model, tokenizer, text)

        # The confident oracle yields a low perplexity, and diverse, non-repetitive
        # prose clears the repetition/diversity/dominant-token gates.
        assert quality.perplexity is not None
        assert quality.perplexity < 100.0
        assert quality.repetition_3gram < 0.6
        assert quality.is_high_quality
        assert quality.metadata["quality_reasons"] == []

    def test_poor_quality_text(self):
        """Highly repetitive text is rated low quality regardless of the model."""
        text = "asdfasdf asdfasdf asdfasdf " * 20

        tokenizer = _WordTokenizer()
        model = _OracleModel(tokenizer.vocab_size)
        quality = assess_quality(model, tokenizer, text)

        # One token repeated 60× drives repetition_3gram and the dominant-token
        # ratio past their thresholds, so quality must be flagged with reasons.
        assert not quality.is_high_quality
        assert quality.repetition_3gram > 0.6
        reasons = quality.metadata["quality_reasons"]
        assert reasons, "expected explicit quality_reasons for degenerate text"
        assert any("repetition" in r.lower() or "dominate" in r.lower() for r in reasons)


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

        # Should indicate metrics1 (the first set) is better
        assert comparison.first_better
        assert not comparison.second_better
        assert comparison.verdict == "first_better"

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
        assert comparison.similar
        assert comparison.identical
        assert comparison.verdict == "similar"

    def test_comparison_exposes_per_metric_changes(self):
        """Regression: per-metric deltas must be real numeric fields.

        ``compare_quality`` computed perplexity/repetition/diversity changes
        internally but only ever folded them into the human-readable strings,
        so callers reading ``perplexity_change`` / ``repetition_change`` /
        ``diversity_change`` (e.g. the GPTQ/AWQ examples) hit a ``KeyError``.
        They are now first-class fields on the typed result.
        """
        first = QualityMetrics(
            perplexity=20.0,
            repetition_3gram=0.02,
            unique_token_ratio=0.80,
            quality_score=0.8,
        )
        second = QualityMetrics(
            perplexity=30.0,
            repetition_3gram=0.10,
            unique_token_ratio=0.50,
            quality_score=0.5,
        )

        comparison = compare_quality(first, second)

        # Perplexity: relative increase (30-20)/20 = +0.5
        assert comparison.perplexity_change == pytest.approx(0.5)
        # Repetition: absolute increase 0.10 - 0.02 = +0.08
        assert comparison.repetition_change == pytest.approx(0.08)
        # Diversity: relative drop (0.50-0.80)/0.80 = -0.375
        assert comparison.diversity_change == pytest.approx(-0.375)

    def test_comparison_changes_none_when_scores_missing(self):
        """Per-metric changes degrade to ``None`` when a score is absent."""
        bare = QualityMetrics(quality_score=0.5)

        comparison = compare_quality(bare, bare)

        assert comparison.perplexity_change is None
        assert comparison.repetition_change is None
        assert comparison.diversity_change is None

    def test_comparison_perplexity_change_none_when_infinite(self):
        """An infinite perplexity (the failure sentinel) is uncomparable.

        ``calculate_perplexity`` returns ``math.inf`` when it cannot score the
        text. ``inf`` is truthy, so a naive ``if a and b`` would compute
        ``(inf - inf) / inf == nan``; the ``math.isfinite`` guard must instead
        leave ``perplexity_change`` as ``None`` while still comparing the
        finite metrics.
        """
        finite = QualityMetrics(perplexity=20.0, repetition_3gram=0.02, quality_score=0.8)
        broken = QualityMetrics(
            perplexity=float("inf"), repetition_3gram=0.10, quality_score=0.5
        )

        comparison = compare_quality(finite, broken)

        assert comparison.perplexity_change is None
        # The finite repetition metric is still compared.
        assert comparison.repetition_change == pytest.approx(0.08)


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
        """calculate_perplexity must equal exp(-mean(target log-probs))."""
        # Target per-token log-probs we want the model to assign to the gold tokens.
        logprobs = mx.array([-1.0, -1.5, -2.0, -1.2, -1.8])

        # Perplexity is defined as exp of the mean negative log-likelihood.
        mean_logprob = float(mx.mean(logprobs))
        expected_perplexity = float(np.exp(-mean_logprob))

        lp_list = logprobs.tolist()

        class _FixedLogProbTokenizer:
            """Encodes any text to N+1 gold tokens (all id 0) → N scored positions."""

            def encode(self, text):
                return [0] * (len(lp_list) + 1)

        class _FixedLogProbModel:
            """Two-token vocab. At position i the log-prob of the gold token
            (index 0) is exactly ``lp_list[i]``: with logit_0 = 0, choosing
            logit_1 = log(exp(-L) - 1) gives log_softmax([0, logit_1])[0] = L.
            """

            def __call__(self, tokens):
                seq_len = tokens.shape[1]
                logits = np.zeros((1, seq_len, 2), dtype=np.float32)
                for i in range(seq_len - 1):
                    L = lp_list[i]
                    logits[0, i, 1] = float(np.log(np.exp(-L) - 1.0))
                return mx.array(logits)

        perplexity = calculate_perplexity(
            _FixedLogProbModel(), _FixedLogProbTokenizer(), "ignored text"
        )
        assert perplexity == pytest.approx(expected_perplexity, rel=1e-3)

    def test_programmer_error_propagates(self):
        """A real bug (AttributeError from a broken model) must NOT be masked as inf.

        Regression for the narrowed ``except`` in ``calculate_perplexity``: the old
        ``except Exception`` swallowed every error and returned ``float('inf')``,
        hiding broken models/tokenizers behind a plausible-looking metric.
        """

        class _OkTokenizer:
            def encode(self, text):
                return [1, 2, 3, 4]

        class _BuggyModel:
            """Has a programming bug — should surface, not become infinite perplexity."""

            def __call__(self, tokens):
                raise AttributeError("'_BuggyModel' object has no attribute 'layers'")

        with pytest.raises(AttributeError):
            calculate_perplexity(_BuggyModel(), _OkTokenizer(), "hello world")

    def test_expected_failure_returns_inf(self):
        """Expected, recoverable failures still degrade gracefully to inf."""

        class _OkTokenizer:
            def encode(self, text):
                return [1, 2, 3, 4]

        # Tokenizer rejecting the input (ValueError) -> unscorable -> inf.
        class _RejectingTokenizer:
            def encode(self, text):
                raise ValueError("cannot tokenize input")

        # MLX-style runtime failure during the forward pass -> inf.
        class _RuntimeFailModel:
            def __call__(self, tokens):
                raise RuntimeError("MLX evaluation failed")

        assert calculate_perplexity(object(), _RejectingTokenizer(), "x") == float("inf")
        assert calculate_perplexity(_RuntimeFailModel(), _OkTokenizer(), "x") == float("inf")


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
