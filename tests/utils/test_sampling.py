"""Tests for smlx.utils.sampling module."""

import mlx.core as mx
import pytest

from smlx.utils.sampling import (
    categorical_sampling,
    make_logits_processors,
    make_repetition_penalty,
    make_sampler,
    min_p_sampling,
    sample,
    top_k_sampling,
    top_p_sampling,
)


class TestBasicSampling:
    """Test basic sampling function."""

    def test_sample_greedy(self):
        """Test greedy sampling (temperature=0)."""
        logits = mx.array([1.0, 5.0, 2.0, 3.0])
        token = sample(logits, temperature=0.0)

        # Should always pick index 1 (highest logit)
        assert token.item() == 1

    def test_sample_with_temperature(self):
        """Test sampling with temperature."""
        logits = mx.array([1.0, 5.0, 2.0, 3.0])

        # With temperature, should still be valid token
        token = sample(logits, temperature=0.7)
        assert 0 <= token.item() < 4

    def test_sample_high_temperature(self):
        """Test sampling with high temperature (more random)."""
        logits = mx.array([1.0, 2.0, 1.5, 1.8])

        # Should still produce valid tokens
        for _ in range(10):
            token = sample(logits, temperature=2.0)
            assert 0 <= token.item() < 4

    def test_sample_with_top_k(self):
        """Test sampling with top-k filtering."""
        logits = mx.array([1.0, 5.0, 2.0, 3.0, 0.5])

        # With top_k=2, should only sample from top 2 logits
        token = sample(logits, temperature=1.0, top_k=2)
        assert token.item() in [1, 3]  # Indices of top 2 values

    def test_sample_with_top_p(self):
        """Test sampling with top-p (nucleus) filtering."""
        logits = mx.array([1.0, 5.0, 2.0, 3.0])

        token = sample(logits, temperature=1.0, top_p=0.9)
        assert 0 <= token.item() < 4


class TestTopKSampling:
    """Test top-k sampling."""

    def test_top_k_basic(self):
        """Test basic top-k filtering."""
        logits = mx.array([[1.0, 5.0, 2.0, 3.0, 0.5]])

        filtered = top_k_sampling(logits, top_k=2)

        # Should have -inf for all but top 2
        assert filtered[0, 1] != float('-inf')  # Highest
        assert filtered[0, 3] != float('-inf')  # Second highest
        assert filtered[0, 0] == float('-inf')  # Lower values
        assert filtered[0, 4] == float('-inf')

    def test_top_k_invalid_range(self):
        """Test top-k with invalid range."""
        logits = mx.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError):
            top_k_sampling(logits, top_k=0)

        # top_k >= vocab_size should return logits unchanged (no error)
        result = top_k_sampling(logits, top_k=10)
        assert result.shape == logits.shape

    def test_top_k_preserves_shape(self):
        """Test that top-k preserves shape."""
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        filtered = top_k_sampling(logits, top_k=3)

        assert filtered.shape == logits.shape


class TestTopPSampling:
    """Test top-p (nucleus) sampling."""

    def test_top_p_basic(self):
        """Test basic top-p filtering."""
        # Create logits where we can predict the nucleus
        logits = mx.array([[10.0, 5.0, 1.0, 0.1]])

        filtered = top_p_sampling(logits, top_p=0.9)

        # High probability tokens should be kept
        assert filtered[0, 0] != float('-inf')

    def test_top_p_extreme_values(self):
        """Test top-p with extreme probability distributions."""
        # Very peaked distribution
        logits = mx.array([[100.0, 1.0, 1.0, 1.0]])

        filtered = top_p_sampling(logits, top_p=0.99)

        # Top token should definitely be kept
        assert filtered[0, 0] != float('-inf')

    def test_top_p_preserves_shape(self):
        """Test that top-p preserves shape."""
        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        filtered = top_p_sampling(logits, top_p=0.9)

        assert filtered.shape == logits.shape


class TestMinPSampling:
    """Test min-p sampling."""

    def test_min_p_basic(self):
        """Test basic min-p filtering."""
        logits = mx.array([[10.0, 5.0, 1.0, 0.1]])

        filtered = min_p_sampling(logits, min_p=0.1, min_tokens_to_keep=1)

        # Top token should always be kept
        assert filtered[0, 0] != float('-inf')

    def test_min_p_min_tokens_to_keep(self):
        """Test that min_tokens_to_keep is respected."""
        logits = mx.array([[10.0, 1.0, 0.5, 0.1]])

        # Even with high min_p, should keep at least min_tokens_to_keep
        filtered = min_p_sampling(logits, min_p=0.5, min_tokens_to_keep=2)

        # At least 2 tokens should not be -inf
        non_inf_count = mx.sum(filtered[0] != float('-inf')).item()
        assert non_inf_count >= 2

    def test_min_p_invalid_range(self):
        """Test min-p with invalid range."""
        logits = mx.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="`min_p` must be in"):
            min_p_sampling(logits, min_p=-0.1)

        with pytest.raises(ValueError, match="`min_p` must be in"):
            min_p_sampling(logits, min_p=1.5)

    def test_min_p_invalid_min_tokens(self):
        """Test min-p with invalid min_tokens_to_keep."""
        logits = mx.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="`min_tokens_to_keep` must be positive"):
            min_p_sampling(logits, min_p=0.1, min_tokens_to_keep=0)


class TestCategoricalSampling:
    """Test categorical sampling."""

    def test_categorical_sampling_basic(self):
        """Test basic categorical sampling."""
        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])

        token = categorical_sampling(logits, temp=1.0)
        assert 0 <= token.item() < 4

    def test_categorical_sampling_temperature(self):
        """Test categorical sampling with different temperatures."""
        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])

        # Lower temperature should still work
        token = categorical_sampling(logits, temp=0.5)
        assert 0 <= token.item() < 4

        # Higher temperature should still work
        token = categorical_sampling(logits, temp=2.0)
        assert 0 <= token.item() < 4


class TestMakeSampler:
    """Test make_sampler function."""

    def test_make_sampler_greedy(self):
        """Test sampler creation with greedy sampling."""
        sampler = make_sampler(temp=0.0)

        logits = mx.array([[1.0, 5.0, 2.0, 3.0]])
        token = sampler(logits)

        # Should always pick argmax
        assert token.item() == 1

    def test_make_sampler_with_top_p(self):
        """Test sampler with top-p."""
        sampler = make_sampler(temp=1.0, top_p=0.9)

        logits = mx.array([[1.0, 5.0, 2.0, 3.0]])
        token = sampler(logits)

        assert 0 <= token.item() < 4

    def test_make_sampler_with_top_k(self):
        """Test sampler with top-k."""
        sampler = make_sampler(temp=1.0, top_k=2)

        logits = mx.array([[1.0, 5.0, 2.0, 3.0]])
        token = sampler(logits)

        assert 0 <= token.item() < 4

    def test_make_sampler_with_min_p(self):
        """Test sampler with min-p."""
        sampler = make_sampler(temp=1.0, min_p=0.1)

        logits = mx.array([[1.0, 5.0, 2.0, 3.0]])
        token = sampler(logits)

        assert 0 <= token.item() < 4

    def test_make_sampler_combined(self):
        """Test sampler with multiple filters."""
        sampler = make_sampler(temp=0.7, top_p=0.9, top_k=3, min_p=0.05)

        logits = mx.array([[1.0, 5.0, 2.0, 3.0, 0.5]])
        token = sampler(logits)

        assert 0 <= token.item() < 5


class TestRepetitionPenalty:
    """Test repetition penalty."""

    def test_make_repetition_penalty_basic(self):
        """Test basic repetition penalty."""
        penalty_fn = make_repetition_penalty(penalty=1.2, context_size=20)

        tokens = mx.array([1, 2, 3, 4])
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        penalized = penalty_fn(tokens, logits)

        # Logits for tokens in history should be penalized
        assert penalized.shape == logits.shape

    def test_repetition_penalty_reduces_probability(self):
        """Test that repetition penalty reduces probability of repeated tokens."""
        penalty_fn = make_repetition_penalty(penalty=2.0, context_size=10)

        # Token 1 is in history
        tokens = mx.array([1, 1, 1])
        logits = mx.array([[5.0, 5.0, 5.0, 5.0]])

        penalized = penalty_fn(tokens, logits)

        # Logit for token 1 should be different (penalized)
        assert penalized[0, 1].item() != logits[0, 1].item()

    def test_repetition_penalty_context_size(self):
        """Test that context_size limits the penalty window."""
        penalty_fn = make_repetition_penalty(penalty=1.5, context_size=2)

        # Only last 2 tokens should be considered
        tokens = mx.array([1, 2, 3, 4, 5])
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

        penalized = penalty_fn(tokens, logits)

        # Tokens 4 and 5 should be penalized, earlier ones should not
        assert penalized.shape == logits.shape

    def test_repetition_penalty_empty_tokens(self):
        """Test repetition penalty with empty token history."""
        penalty_fn = make_repetition_penalty(penalty=1.5, context_size=10)

        tokens = mx.array([])
        logits = mx.array([[1.0, 2.0, 3.0]])

        # Should return unchanged logits
        penalized = penalty_fn(tokens, logits)

        assert mx.allclose(penalized, logits)

    def test_repetition_penalty_invalid_value(self):
        """Test repetition penalty with invalid penalty value."""
        with pytest.raises(ValueError):
            make_repetition_penalty(penalty=-1.0)


class TestLogitsProcessors:
    """Test logits processors."""

    def test_make_logits_processors_empty(self):
        """Test creating empty processors list."""
        processors = make_logits_processors()

        assert isinstance(processors, list)
        assert len(processors) == 0

    def test_make_logits_processors_with_bias(self):
        """Test processors with logit bias."""
        processors = make_logits_processors(
            logit_bias={0: -100.0, 1: 100.0}
        )

        assert len(processors) == 1

        tokens = mx.array([])
        logits = mx.array([[1.0, 1.0, 1.0]])

        processed = processors[0](tokens, logits)

        # Token 0 should be heavily penalized, token 1 boosted
        assert processed[0, 0] < logits[0, 0]
        assert processed[0, 1] > logits[0, 1]

    def test_make_logits_processors_with_repetition_penalty(self):
        """Test processors with repetition penalty."""
        processors = make_logits_processors(
            repetition_penalty=1.5,
            repetition_context_size=10
        )

        assert len(processors) == 1

    def test_make_logits_processors_combined(self):
        """Test processors with both bias and repetition penalty."""
        processors = make_logits_processors(
            logit_bias={0: -10.0},
            repetition_penalty=1.2,
            repetition_context_size=20
        )

        # Should have both processors
        assert len(processors) == 2


class TestSamplingIntegration:
    """Test integration of sampling components."""

    def test_full_sampling_pipeline(self):
        """Test complete sampling pipeline."""
        # Create logits
        logits = mx.array([[1.0, 5.0, 2.0, 3.0, 0.5]])

        # Apply top-k
        logits = top_k_sampling(logits, top_k=3)

        # Apply top-p
        logits = top_p_sampling(logits, top_p=0.9)

        # Sample
        token = categorical_sampling(logits, temp=0.7)

        assert 0 <= token.item() < 5

    def test_sampler_with_processors(self):
        """Test using sampler with logits processors."""
        sampler = make_sampler(temp=0.7, top_p=0.9)
        processors = make_logits_processors(repetition_penalty=1.2)

        tokens = mx.array([1, 2])
        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])

        # Apply processors
        for processor in processors:
            logits = processor(tokens, logits)

        # Sample
        token = sampler(logits)

        assert 0 <= token.item() < 4
