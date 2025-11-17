# Copyright © 2025 SMLX Project

"""
Integration tests for SmolLM2-135M text generation.

Tests the complete generation pipeline including sampling, caching, and generation functions.
"""

import mlx.core as mx
import pytest

from smlx.models.SmolLM2_135M import (
    Model,
    generate_step,
    get_default_config,
    make_cache,
    sample,
)
from smlx.utils.cache import reset_cache


# ============================================================================
# Sampling Tests
# ============================================================================


@pytest.mark.unit
class TestSamplingStrategies:
    """Test different sampling strategies."""

    def test_greedy_sampling(self):
        """Test greedy sampling (temperature=0)."""
        # Create logits with clear argmax
        logits = mx.array([1.0, 5.0, 2.0, 0.5])

        token = sample(logits, temperature=0.0)

        # Should select argmax (index 1)
        assert int(token.item()) == 1

    def test_temperature_sampling(self):
        """Test temperature-based sampling."""
        logits = mx.array([1.0, 2.0, 3.0, 1.0])

        # With temperature=1.0, should sample from distribution
        token = sample(logits, temperature=1.0)

        # Should be a valid token
        assert 0 <= int(token.item()) < len(logits)

    def test_top_k_sampling(self):
        """Test top-k sampling."""
        logits = mx.array([1.0, 5.0, 2.0, 0.5, 3.0])

        # Sample from top-2
        samples = []
        mx.random.seed(42)
        for _ in range(20):
            token = sample(logits, temperature=1.0, top_k=2)
            samples.append(int(token.item()))

        # Should only sample from indices 1 (5.0) or 4 (3.0)
        assert all(s in [1, 4] for s in samples)

    def test_top_p_sampling(self):
        """Test nucleus (top-p) sampling."""
        logits = mx.array([1.0, 5.0, 2.0, 0.5])

        # Sample with top_p
        token = sample(logits, temperature=1.0, top_p=0.9)

        # Should be a valid token
        assert 0 <= int(token.item()) < len(logits)


# ============================================================================
# Mock Tokenizer for Testing
# ============================================================================


class MockTokenizer:
    """Simple mock tokenizer for testing."""

    def __init__(self, vocab_size=49152):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.bos_token_id = 1

    def encode(self, text):
        """Mock encode - just returns token IDs based on text length."""
        # Simple hash-based encoding for testing
        return [hash(text) % (self.vocab_size - 10) + 2 for _ in range(min(len(text.split()), 10))]

    def decode(self, token_ids):
        """Mock decode - returns placeholder text."""
        return " ".join([f"token_{tid}" for tid in token_ids])


# ============================================================================
# Generation Step Tests
# ============================================================================


@pytest.mark.integration
class TestGenerationStep:
    """Test the generation step function."""

    @pytest.fixture
    def model(self):
        """Create a small test model with cleanup."""
        # Clear memory before creating model
        mx.clear_cache()

        config = get_default_config()
        model = Model(config)

        yield model

        # Cleanup after each test
        del model
        mx.clear_cache()

    def test_generate_step_basic(self, model):
        """Test basic generation step."""
        # Create short prompt
        prompt_tokens = mx.array([1, 2, 3, 4, 5])

        # Create a fresh cache for this test
        cache = make_cache(model)

        # Generate a few tokens
        max_tokens = 3
        generated_tokens = []
        for i, (token_id, _) in enumerate(generate_step(
            model=model,
            prompt_tokens=prompt_tokens,
            temperature=1.0,
            cache=cache,
        )):
            generated_tokens.append(token_id)
            # Clear after each iteration to reduce memory pressure
            mx.eval(token_id)

            # Break after max_tokens
            if i + 1 >= max_tokens:
                break

        # Should generate exactly 3 tokens
        assert len(generated_tokens) == 3

        # All tokens should be valid (in vocabulary)
        for token_id in generated_tokens:
            assert 0 <= token_id < model.args.vocab_size

        # Clean up cache after test
        reset_cache(cache)
        mx.clear_cache()

    def test_generate_step_with_cache(self, model):
        """Test generation with explicit cache."""
        prompt_tokens = mx.array([1, 2, 3])

        # Create cache
        cache = make_cache(model)

        # Generate with cache
        max_tokens = 5
        generated_tokens = []
        for i, (token_id, _) in enumerate(generate_step(
            model=model,
            prompt_tokens=prompt_tokens,
            temperature=1.0,
            cache=cache,
        )):
            generated_tokens.append(token_id)
            mx.eval(token_id)

            # Break after max_tokens
            if i + 1 >= max_tokens:
                break

        assert len(generated_tokens) == 5

        # Cache should have been updated
        for c in cache:
            assert c.offset > 0

        # Clean up cache after test
        reset_cache(cache)
        mx.clear_cache()

    def test_generate_step_deterministic(self, model):
        """Test that greedy sampling is deterministic."""
        prompt_tokens = mx.array([1, 2, 3, 4])

        # Create cache for first run
        cache1 = make_cache(model)

        # Generate twice with temperature=0 (greedy)
        max_tokens = 10
        tokens1 = []
        for i, (token_id, _) in enumerate(generate_step(
            model=model,
            prompt_tokens=prompt_tokens,
            temperature=0.0,
            cache=cache1,
        )):
            tokens1.append(token_id)
            mx.eval(token_id)

            # Break after max_tokens
            if i + 1 >= max_tokens:
                break

        # Clean up first cache
        reset_cache(cache1)
        mx.clear_cache()

        # Create fresh cache for second run
        cache2 = make_cache(model)

        tokens2 = []
        for i, (token_id, _) in enumerate(generate_step(
            model=model,
            prompt_tokens=prompt_tokens,
            temperature=0.0,
            cache=cache2,
        )):
            tokens2.append(token_id)
            mx.eval(token_id)

            # Break after max_tokens
            if i + 1 >= max_tokens:
                break

        # Should be identical
        assert tokens1 == tokens2

        # Clean up second cache
        reset_cache(cache2)
        mx.clear_cache()


# ============================================================================
# Enhanced Cache Tests (New KV Cache Module)
# ============================================================================


@pytest.mark.integration
class TestEnhancedCache:
    """Test new kv_cache module features."""

    @pytest.fixture
    def model(self):
        """Create a small test model with cleanup."""
        mx.clear_cache()
        config = get_default_config()
        model = Model(config)
        yield model
        del model
        mx.clear_cache()

    def test_cache_auto_mode(self, model):
        """Test automatic cache type selection."""
        # Auto mode without extra params should work in legacy mode
        cache = make_cache(model)
        assert len(cache) == len(model.layers)

        # All caches should have offset and update_and_fetch
        for c in cache:
            assert hasattr(c, "offset")
            assert hasattr(c, "update_and_fetch")

        reset_cache(cache)
        mx.clear_cache()

    def test_cache_rotating_mode(self, model):
        """Test rotating cache mode."""
        # Create rotating cache with explicit size
        cache = make_cache(model, cache_type="rotating", max_kv_size=512)
        assert len(cache) == len(model.layers)

        # Generate some tokens to populate cache
        prompt_tokens = mx.array([1, 2, 3])
        max_tokens = 10

        for i, (token_id, _) in enumerate(
            generate_step(model=model, prompt_tokens=prompt_tokens, temperature=1.0, cache=cache)
        ):
            mx.eval(token_id)
            if i + 1 >= max_tokens:
                break

        # Cache should be populated
        for c in cache:
            assert c.offset > 0

        reset_cache(cache)
        mx.clear_cache()

    def test_cache_quantized_mode(self, model):
        """Test quantized cache mode."""
        # Create quantized cache
        cache = make_cache(
            model,
            cache_type="quantized",
            enable_quantization=True,
            quantization_bits=4,
            max_kv_size=1024,
        )
        assert len(cache) == len(model.layers)

        # Generate with quantized cache
        prompt_tokens = mx.array([1, 2, 3, 4, 5])
        max_tokens = 5

        generated = []
        for i, (token_id, _) in enumerate(
            generate_step(model=model, prompt_tokens=prompt_tokens, temperature=1.0, cache=cache)
        ):
            generated.append(token_id)
            mx.eval(token_id)
            if i + 1 >= max_tokens:
                break

        assert len(generated) == max_tokens

        # Clean up
        reset_cache(cache)
        mx.clear_cache()

    def test_cache_with_monitoring(self, model):
        """Test cache creation with memory pressure monitoring."""
        from smlx.models.SmolLM2_135M.cache import make_cache_with_monitoring

        # Create cache with monitoring
        cache, breaker = make_cache_with_monitoring(model, target_memory_gb=32.0)

        assert len(cache) == len(model.layers)
        assert breaker is not None
        assert breaker.enabled

        # Generate some tokens
        prompt_tokens = mx.array([1, 2, 3])
        max_tokens = 5

        for i, (token_id, _) in enumerate(
            generate_step(model=model, prompt_tokens=prompt_tokens, temperature=1.0, cache=cache)
        ):
            # Monitor during generation
            intervention = breaker.monitor_and_intervene(current_step=i)
            # For this small test, should not trigger intervention
            # (unless memory is very tight)
            mx.eval(token_id)

            if i + 1 >= max_tokens:
                break

        # Get statistics
        stats = breaker.get_statistics()
        assert "total_interventions" in stats
        assert "enabled" in stats
        assert stats["enabled"] is True

        # Clean up
        reset_cache(cache)
        mx.clear_cache()

    def test_cache_backwards_compatibility(self, model):
        """Test that legacy API still works."""
        # Old style: simple max_kv_size parameter
        cache1 = make_cache(model)
        cache2 = make_cache(model, max_kv_size=1024)

        assert len(cache1) == len(model.layers)
        assert len(cache2) == len(model.layers)

        # Both should work with generate_step
        prompt_tokens = mx.array([1, 2, 3])

        # Test cache1
        for i, (token_id, _) in enumerate(
            generate_step(model=model, prompt_tokens=prompt_tokens, temperature=1.0, cache=cache1)
        ):
            mx.eval(token_id)
            if i >= 2:
                break

        # Test cache2
        reset_cache(cache1)
        for i, (token_id, _) in enumerate(
            generate_step(model=model, prompt_tokens=prompt_tokens, temperature=1.0, cache=cache2)
        ):
            mx.eval(token_id)
            if i >= 2:
                break

        reset_cache(cache1)
        reset_cache(cache2)
        mx.clear_cache()

    def test_cache_enable_monitoring_flag(self, model):
        """Test enable_monitoring flag in make_cache."""
        # Create cache with monitoring enabled
        cache = make_cache(model, cache_type="standard", enable_monitoring=True)

        assert len(cache) == len(model.layers)

        # Caches should have tracing capabilities
        for c in cache:
            assert hasattr(c, "enable_tracing")

        reset_cache(cache)
        mx.clear_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
