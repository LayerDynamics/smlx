"""Tests for smlx.kv_cache.alibi module."""

import mlx.core as mx
import pytest

from smlx.kv_cache.alibi import ALiBiCache, initialize_alibi_cache
from smlx.kv_cache.mlx_kv import MLXKVCache, MLXRotatingKVCache


class TestALiBiCache:
    """Test ALiBiCache class."""

    def test_alibi_cache_init(self):
        """Test ALiBiCache initialization."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=12)

        assert alibi_cache.cache is cache
        assert alibi_cache.num_heads == 12
        assert alibi_cache.slopes.shape == (12,)

    def test_alibi_cache_with_rotating_cache(self):
        """Test ALiBiCache with rotating KV cache."""
        cache = MLXRotatingKVCache(max_size=1024, keep=64)
        alibi_cache = ALiBiCache(cache, num_heads=8)

        assert isinstance(alibi_cache.cache, MLXRotatingKVCache)
        assert alibi_cache.num_heads == 8

    @pytest.mark.unit
    def test_alibi_slopes_computation(self):
        """Test that slopes are computed correctly."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=8)

        # Slopes should be decreasing (more negative for higher heads)
        slopes = alibi_cache.slopes

        assert slopes.shape == (8,)
        # All slopes should be positive (they become negative when applied to distance)
        assert mx.all(slopes > 0).item()
        # Slopes should be in descending order
        for i in range(len(slopes) - 1):
            assert slopes[i].item() > slopes[i + 1].item()

    @pytest.mark.unit
    def test_compute_bias_single_query(self):
        """Test computing bias for single query token."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=12)

        bias = alibi_cache.compute_bias(seq_len=100, query_len=1)

        # Shape should be [num_heads, query_len, seq_len]
        assert bias.shape == (12, 1, 100)

    @pytest.mark.unit
    def test_compute_bias_multiple_queries(self):
        """Test computing bias for multiple query tokens."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=12)

        bias = alibi_cache.compute_bias(seq_len=100, query_len=10)

        # Shape should be [num_heads, query_len, seq_len]
        assert bias.shape == (12, 10, 100)

    @pytest.mark.unit
    def test_compute_bias_default_query_len(self):
        """Test computing bias with default query_len (1)."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=8)

        bias = alibi_cache.compute_bias(seq_len=50)

        # Should default to query_len=1
        assert bias.shape == (8, 1, 50)

    @pytest.mark.unit
    def test_bias_values_causal(self):
        """Test that bias values are correct for causal attention."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=4)

        bias = alibi_cache.compute_bias(seq_len=10, query_len=1)

        # For the last position attending to all positions:
        # bias should be 0 for current position and negative for past positions
        # bias[h, 0, 9] should be 0 (attending to self)
        # bias[h, 0, 0] should be negative (attending to far past)
        for h in range(4):
            # Self-attention (position 9 attending to 9) should be 0
            assert bias[h, 0, 9].item() == 0.0
            # Past positions should be negative
            assert bias[h, 0, 0].item() < 0.0

    @pytest.mark.unit
    def test_update_and_fetch_with_bias(self):
        """Test updating cache and getting bias."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=12)

        keys = mx.random.normal((1, 12, 10, 64))
        values = mx.random.normal((1, 12, 10, 64))

        all_keys, all_values, bias = alibi_cache.update_and_fetch_with_bias(keys, values)

        # Keys and values should be returned
        assert all_keys.shape == (1, 12, 10, 64)
        assert all_values.shape == (1, 12, 10, 64)

        # Bias should be computed for the sequence
        assert bias.shape == (12, 10, 10)

    @pytest.mark.unit
    def test_update_and_fetch_incremental(self):
        """Test incremental updates with bias."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=8)

        # First update
        keys1 = mx.random.normal((1, 8, 10, 64))
        values1 = mx.random.normal((1, 8, 10, 64))
        all_keys1, _, bias1 = alibi_cache.update_and_fetch_with_bias(keys1, values1)

        assert all_keys1.shape[2] == 10
        assert bias1.shape == (8, 10, 10)

        # Second update (single token)
        keys2 = mx.random.normal((1, 8, 1, 64))
        values2 = mx.random.normal((1, 8, 1, 64))
        all_keys2, _, bias2 = alibi_cache.update_and_fetch_with_bias(keys2, values2)

        # Should have 11 tokens total
        assert all_keys2.shape[2] == 11
        # Bias should be for attending from new token to all 11
        assert bias2.shape == (8, 1, 11)

    @pytest.mark.unit
    def test_offset_property(self):
        """Test offset property."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=12)

        keys = mx.random.normal((1, 12, 10, 64))
        values = mx.random.normal((1, 12, 10, 64))
        alibi_cache.update_and_fetch_with_bias(keys, values)

        assert alibi_cache.offset == 10

    @pytest.mark.unit
    def test_state_getter(self):
        """Test getting state."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=12)

        keys = mx.random.normal((1, 12, 10, 64))
        values = mx.random.normal((1, 12, 10, 64))
        alibi_cache.update_and_fetch_with_bias(keys, values)

        state = alibi_cache.state

        assert "cache_state" in state
        assert state["num_heads"] == 12
        assert state["offset"] == 10

    @pytest.mark.unit
    def test_state_setter(self):
        """Test setting state."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=12)

        # Create state
        keys = mx.random.normal((1, 12, 20, 64))
        values = mx.random.normal((1, 12, 20, 64))
        alibi_cache.update_and_fetch_with_bias(keys, values)

        old_state = alibi_cache.state

        # Create new cache and restore state
        new_cache = MLXKVCache()
        new_alibi_cache = ALiBiCache(new_cache, num_heads=8)  # Different num_heads initially
        new_alibi_cache.state = old_state

        # State should be restored
        assert new_alibi_cache.num_heads == 12
        assert new_alibi_cache.offset == 20

    @pytest.mark.unit
    def test_reset(self):
        """Test resetting cache."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=12)

        keys = mx.random.normal((1, 12, 10, 64))
        values = mx.random.normal((1, 12, 10, 64))
        alibi_cache.update_and_fetch_with_bias(keys, values)

        alibi_cache.reset()

        assert alibi_cache.offset == 0
        assert alibi_cache.cache.keys is None

    @pytest.mark.unit
    def test_get_trace_summary(self):
        """Test getting trace summary."""
        cache = MLXKVCache(enable_tracing=True)
        alibi_cache = ALiBiCache(cache, num_heads=12)

        keys = mx.random.normal((1, 12, 10, 64))
        values = mx.random.normal((1, 12, 10, 64))
        alibi_cache.update_and_fetch_with_bias(keys, values)

        summary = alibi_cache.get_trace_summary()

        assert summary["enabled"] is True

    @pytest.mark.unit
    def test_clear_trace(self):
        """Test clearing trace."""
        cache = MLXKVCache(enable_tracing=True)
        alibi_cache = ALiBiCache(cache, num_heads=12)

        keys = mx.random.normal((1, 12, 10, 64))
        values = mx.random.normal((1, 12, 10, 64))
        alibi_cache.update_and_fetch_with_bias(keys, values)

        alibi_cache.clear_trace()

        summary = alibi_cache.get_trace_summary()
        assert len(summary.get("events", [])) == 0


class TestInitializeALiBiCache:
    """Test initialize_alibi_cache function."""

    def test_initialize_alibi_cache_standard(self):
        """Test creating standard ALiBi caches."""
        caches = initialize_alibi_cache(num_heads=12, num_layers=6)

        assert len(caches) == 6
        assert all(isinstance(c, ALiBiCache) for c in caches)
        assert all(c.num_heads == 12 for c in caches)
        assert all(isinstance(c.cache, MLXKVCache) for c in caches)

    def test_initialize_alibi_cache_rotating(self):
        """Test creating rotating ALiBi caches."""
        caches = initialize_alibi_cache(
            num_heads=12,
            num_layers=6,
            max_kv_size=2048,
            keep=256,
        )

        assert len(caches) == 6
        assert all(isinstance(c.cache, MLXRotatingKVCache) for c in caches)
        # Type narrow for attribute access
        for c in caches:
            assert isinstance(c.cache, MLXRotatingKVCache)
            assert c.cache.max_size == 2048
            assert c.cache.keep == 256

    def test_initialize_alibi_cache_with_tracing(self):
        """Test creating ALiBi caches with tracing enabled."""
        caches = initialize_alibi_cache(
            num_heads=8,
            num_layers=4,
            enable_tracing=True,
        )

        for cache in caches:
            assert cache.cache.enable_tracing is True

    def test_initialize_alibi_cache_custom_step(self):
        """Test creating ALiBi caches with custom step."""
        caches = initialize_alibi_cache(
            num_heads=8,
            num_layers=4,
            step=128,
        )

        for cache in caches:
            assert cache.cache.step == 128

    def test_initialize_alibi_cache_single_layer(self):
        """Test creating single layer cache."""
        caches = initialize_alibi_cache(num_heads=12, num_layers=1)

        assert len(caches) == 1
        assert isinstance(caches[0], ALiBiCache)


class TestALiBiCacheIntegration:
    """Test integration scenarios for ALiBi cache."""

    @pytest.mark.unit
    def test_multi_layer_generation(self):
        """Test multi-layer generation with ALiBi."""
        num_layers = 6
        caches = initialize_alibi_cache(num_heads=12, num_layers=num_layers)

        # Simulate forward pass through all layers
        for layer_idx in range(num_layers):
            keys = mx.random.normal((1, 12, 10, 64))
            values = mx.random.normal((1, 12, 10, 64))
            _, _, bias = caches[layer_idx].update_and_fetch_with_bias(keys, values)

            assert bias.shape == (12, 10, 10)

        # All caches should have same offset
        assert all(c.offset == 10 for c in caches)

    @pytest.mark.unit
    def test_autoregressive_generation(self):
        """Test autoregressive generation with ALiBi."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=8)

        # Initial prompt
        keys = mx.random.normal((1, 8, 10, 64))
        values = mx.random.normal((1, 8, 10, 64))
        alibi_cache.update_and_fetch_with_bias(keys, values)

        # Generate 5 tokens
        for i in range(5):
            keys = mx.random.normal((1, 8, 1, 64))
            values = mx.random.normal((1, 8, 1, 64))
            all_keys, _, bias = alibi_cache.update_and_fetch_with_bias(keys, values)

            expected_len = 10 + i + 1
            assert all_keys.shape[2] == expected_len
            # Bias should attend from new token to all tokens
            assert bias.shape == (8, 1, expected_len)

    @pytest.mark.unit
    def test_alibi_with_rotating_cache_long_sequence(self):
        """Test ALiBi with rotating cache for long sequences."""
        cache = MLXRotatingKVCache(max_size=100, keep=10)
        alibi_cache = ALiBiCache(cache, num_heads=8)

        # Initial prompt
        keys = mx.random.normal((1, 8, 10, 64))
        values = mx.random.normal((1, 8, 10, 64))
        alibi_cache.update_and_fetch_with_bias(keys, values)

        # Generate 200 tokens (exceeds max_size)
        for _ in range(200):
            keys = mx.random.normal((1, 8, 1, 64))
            values = mx.random.normal((1, 8, 1, 64))
            all_keys, _, bias = alibi_cache.update_and_fetch_with_bias(keys, values)

        # Cache should stay at max_size
        assert all_keys.shape[2] == 100
        # Bias should be computed for max_size
        assert bias.shape == (8, 1, 100)


class TestALiBiCacheEdgeCases:
    """Test edge cases for ALiBi cache."""

    @pytest.mark.unit
    def test_alibi_single_head(self):
        """Test ALiBi with single head."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=1)

        assert alibi_cache.slopes.shape == (1,)

        bias = alibi_cache.compute_bias(seq_len=10, query_len=1)
        assert bias.shape == (1, 1, 10)

    @pytest.mark.unit
    def test_alibi_large_num_heads(self):
        """Test ALiBi with large number of heads."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=64)

        assert alibi_cache.slopes.shape == (64,)

        bias = alibi_cache.compute_bias(seq_len=100, query_len=1)
        assert bias.shape == (64, 1, 100)

    @pytest.mark.unit
    def test_alibi_bias_symmetry(self):
        """Test that bias is symmetric for same query/key positions."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=8)

        # Compute bias for sequence length 50
        bias1 = alibi_cache.compute_bias(seq_len=50, query_len=10)
        bias2 = alibi_cache.compute_bias(seq_len=50, query_len=10)

        # Should be identical
        assert mx.array_equal(bias1, bias2).item()

    @pytest.mark.unit
    def test_alibi_trace_disabled(self):
        """Test ALiBi with tracing disabled."""
        cache = MLXKVCache(enable_tracing=False)
        alibi_cache = ALiBiCache(cache, num_heads=8)

        summary = alibi_cache.get_trace_summary()
        assert summary["enabled"] is False

    @pytest.mark.unit
    def test_slopes_different_for_different_heads(self):
        """Test that different heads have different slopes."""
        cache = MLXKVCache()
        alibi_cache = ALiBiCache(cache, num_heads=8)

        slopes = alibi_cache.slopes

        # All slopes should be unique
        for i in range(len(slopes)):
            for j in range(i + 1, len(slopes)):
                assert slopes[i].item() != slopes[j].item()
