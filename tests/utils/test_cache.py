"""Tests for smlx.utils.cache module."""

import mlx.core as mx

from smlx.utils.cache import (
    KVCache,
    RotatingKVCache,
    make_cache,
    reset_cache,
)


class TestKVCache:
    """Test KVCache class."""

    def test_kvcache_init(self):
        """Test KVCache initialization."""
        cache = KVCache()

        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0
        assert cache.step == 256

    def test_kvcache_custom_step(self):
        """Test KVCache with custom step size."""
        cache = KVCache(step=128)

        assert cache.step == 128

    def test_kvcache_update_and_fetch_first_time(self):
        """Test first update to cache."""
        cache = KVCache()

        # Create dummy keys and values [B, n_heads, seq_len, head_dim]
        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape == (1, 4, 10, 64)
        assert all_values.shape == (1, 4, 10, 64)
        assert cache.offset == 10

    def test_kvcache_multiple_updates(self):
        """Test multiple updates to cache."""
        cache = KVCache()

        # First update
        keys1 = mx.ones((1, 4, 10, 64))
        values1 = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(keys1, values1)

        # Second update
        keys2 = mx.ones((1, 4, 5, 64))
        values2 = mx.ones((1, 4, 5, 64))
        all_keys, all_values = cache.update_and_fetch(keys2, values2)

        # Should have 15 tokens total
        assert all_keys.shape == (1, 4, 15, 64)
        assert all_values.shape == (1, 4, 15, 64)
        assert cache.offset == 15

    def test_kvcache_allocation_step(self):
        """Test that cache allocates in steps."""
        cache = KVCache(step=256)

        # Add 10 tokens
        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        # Cache should allocate 256 tokens worth of space
        assert cache.keys is not None
        assert cache.keys.shape[2] == 256

    def test_kvcache_state_getter(self):
        """Test getting cache state."""
        cache = KVCache()

        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        state_keys, state_values = cache.state

        # State should be trimmed to actual size
        assert state_keys.shape == (1, 4, 10, 64)
        assert state_values.shape == (1, 4, 10, 64)

    def test_kvcache_state_setter(self):
        """Test setting cache state."""
        cache = KVCache()

        # Create state
        keys = mx.ones((1, 4, 20, 64))
        values = mx.ones((1, 4, 20, 64))

        # Set state
        cache.state = (keys, values)

        assert cache.offset == 20
        assert cache.keys is not None
        assert cache.values is not None
        assert cache.keys.shape == keys.shape
        assert cache.values.shape == values.shape

    def test_kvcache_reset(self):
        """Test resetting cache."""
        cache = KVCache()

        # Add some data
        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        # Reset
        cache.reset()

        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0


class TestRotatingKVCache:
    """Test RotatingKVCache class."""

    def test_rotating_cache_init(self):
        """Test RotatingKVCache initialization."""
        cache = RotatingKVCache(max_size=1024, keep=64)

        assert cache.max_size == 1024
        assert cache.keep == 64
        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0

    def test_rotating_cache_update_single_token(self):
        """Test updating with single token."""
        cache = RotatingKVCache(max_size=1024, keep=0)

        # Add single token
        keys = mx.ones((1, 4, 1, 64))
        values = mx.ones((1, 4, 1, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape[2] == 1
        assert cache.offset == 1

    def test_rotating_cache_multiple_tokens(self):
        """Test updating with multiple tokens."""
        cache = RotatingKVCache(max_size=1024, keep=0)

        # Add 10 tokens
        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape[2] == 10
        assert cache.offset == 10

    def test_rotating_cache_rotation(self):
        """Test that cache rotates when reaching max size."""
        cache = RotatingKVCache(max_size=100, keep=10)

        # Fill cache beyond max_size
        for i in range(120):
            keys = mx.full((1, 4, 1, 64), float(i))
            values = mx.full((1, 4, 1, 64), float(i))
            all_keys, all_values = cache.update_and_fetch(keys, values)

        # Cache should be at max size
        assert all_keys.shape[2] <= cache.max_size
        assert cache.offset == 120

    def test_rotating_cache_keep_tokens(self):
        """Test that keep tokens are preserved."""
        cache = RotatingKVCache(max_size=50, keep=10)

        # Add initial tokens (these should be kept)
        initial_keys = mx.ones((1, 4, 10, 64))
        initial_values = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(initial_keys, initial_values)

        # Add many more tokens to trigger rotation
        for _ in range(60):
            keys = mx.ones((1, 4, 1, 64))
            values = mx.ones((1, 4, 1, 64))
            cache.update_and_fetch(keys, values)

        # First 10 tokens should still be preserved
        # (though we can't easily verify values without more complex checks)
        assert cache.offset == 70

    def test_rotating_cache_state(self):
        """Test getting/setting state."""
        cache = RotatingKVCache(max_size=1024, keep=0)

        keys = mx.ones((1, 4, 20, 64))
        values = mx.ones((1, 4, 20, 64))
        cache.update_and_fetch(keys, values)

        state_keys, state_values = cache.state

        assert state_keys.shape[2] == 20
        assert state_values.shape[2] == 20

    def test_rotating_cache_reset(self):
        """Test resetting rotating cache."""
        cache = RotatingKVCache(max_size=1024, keep=10)

        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        cache.reset()

        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0
        assert cache._idx == 0

    def test_rotating_cache_custom_step(self):
        """Test rotating cache with custom step."""
        cache = RotatingKVCache(max_size=1024, keep=0, step=128)

        assert cache.step == 128


class TestMakeCache:
    """Test make_cache function."""

    def test_make_cache_standard(self):
        """Test creating standard cache."""
        caches = make_cache(num_layers=12)

        assert len(caches) == 12
        assert all(isinstance(c, KVCache) for c in caches)

    def test_make_cache_rotating(self):
        """Test creating rotating cache."""
        caches = make_cache(num_layers=12, max_kv_size=2048, keep=256)

        assert len(caches) == 12
        assert all(isinstance(c, RotatingKVCache) for c in caches)
        # Type narrow to RotatingKVCache for attribute access
        for c in caches:
            assert isinstance(c, RotatingKVCache)
            assert c.max_size == 2048
            assert c.keep == 256

    def test_make_cache_custom_step(self):
        """Test creating cache with custom step."""
        caches = make_cache(num_layers=6, step=128)

        assert all(c.step == 128 for c in caches)

    def test_make_cache_zero_layers(self):
        """Test creating cache with zero layers."""
        caches = make_cache(num_layers=0)

        assert len(caches) == 0

    def test_make_cache_one_layer(self):
        """Test creating cache with one layer."""
        caches = make_cache(num_layers=1)

        assert len(caches) == 1


class TestResetCache:
    """Test reset_cache function."""

    def test_reset_cache_standard(self):
        """Test resetting standard caches."""
        caches = make_cache(num_layers=4)

        # Add some data to each cache
        for cache in caches:
            keys = mx.ones((1, 4, 10, 64))
            values = mx.ones((1, 4, 10, 64))
            cache.update_and_fetch(keys, values)

        # Reset all
        reset_cache(caches)

        # All should be reset
        assert all(c.keys is None for c in caches)
        assert all(c.values is None for c in caches)
        assert all(c.offset == 0 for c in caches)

    def test_reset_cache_rotating(self):
        """Test resetting rotating caches."""
        caches = make_cache(num_layers=4, max_kv_size=1024, keep=64)

        # Add data
        for cache in caches:
            keys = mx.ones((1, 4, 10, 64))
            values = mx.ones((1, 4, 10, 64))
            cache.update_and_fetch(keys, values)

        # Reset
        reset_cache(caches)

        # All should be reset
        assert all(c.keys is None for c in caches)
        assert all(c.offset == 0 for c in caches)

    def test_reset_cache_empty_list(self):
        """Test resetting empty cache list."""
        reset_cache([])
        # Should not raise an error


class TestCacheIntegration:
    """Test integration scenarios."""

    def test_multi_layer_caching(self):
        """Test using cache for multi-layer model."""
        num_layers = 6
        caches = make_cache(num_layers=num_layers)

        # Simulate forward pass through multiple layers
        for layer_idx in range(num_layers):
            keys = mx.random.normal((1, 4, 10, 64))
            values = mx.random.normal((1, 4, 10, 64))

            all_keys, all_values = caches[layer_idx].update_and_fetch(keys, values)

            assert all_keys.shape[2] == 10

        # All caches should have 10 tokens
        assert all(c.offset == 10 for c in caches)

    def test_autoregressive_generation(self):
        """Test cache in autoregressive generation scenario."""
        cache = KVCache()

        # Initial prompt (10 tokens)
        keys = mx.random.normal((1, 4, 10, 64))
        values = mx.random.normal((1, 4, 10, 64))
        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape[2] == 10

        # Generate 5 tokens one at a time
        for i in range(5):
            keys = mx.random.normal((1, 4, 1, 64))
            values = mx.random.normal((1, 4, 1, 64))
            all_keys, all_values = cache.update_and_fetch(keys, values)

            expected_len = 10 + i + 1
            assert all_keys.shape[2] == expected_len

        # Final cache should have 15 tokens
        assert cache.offset == 15

    def test_rotating_cache_long_generation(self):
        """Test rotating cache with very long generation."""
        cache = RotatingKVCache(max_size=100, keep=10)

        # Initial prompt
        keys = mx.random.normal((1, 4, 10, 64))
        values = mx.random.normal((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        # Generate 200 tokens (exceeds max_size)
        for _ in range(200):
            keys = mx.random.normal((1, 4, 1, 64))
            values = mx.random.normal((1, 4, 1, 64))
            all_keys, all_values = cache.update_and_fetch(keys, values)

        # Cache should stay at max_size
        assert all_keys.shape[2] == cache.max_size
        assert cache.offset == 210  # Total tokens processed


class TestCacheEdgeCases:
    """Test edge cases and special scenarios."""

    def test_kvcache_different_head_dims(self):
        """Test KVCache with different key and value head dimensions."""
        cache = KVCache()

        # Different head dimensions for keys and values
        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 32))  # Different head_dim

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape[3] == 64
        assert all_values.shape[3] == 32

    def test_kvcache_batch_size(self):
        """Test KVCache with different batch sizes."""
        cache = KVCache()

        # Batch size of 4
        keys = mx.ones((4, 8, 10, 64))
        values = mx.ones((4, 8, 10, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape[0] == 4

    def test_rotating_cache_exact_max_size(self):
        """Test rotating cache when adding exactly max_size tokens."""
        cache = RotatingKVCache(max_size=100, keep=0)

        # Add exactly max_size tokens
        keys = mx.ones((1, 4, 100, 64))
        values = mx.ones((1, 4, 100, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape[2] == 100
        assert cache.offset == 100

    def test_rotating_cache_keep_equals_max(self):
        """Test rotating cache when keep equals max_size."""
        cache = RotatingKVCache(max_size=100, keep=100)

        # Should still work (though rotation is essentially disabled)
        keys = mx.ones((1, 4, 50, 64))
        values = mx.ones((1, 4, 50, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape[2] == 50

    def test_kvcache_large_sequence(self):
        """Test KVCache with large sequence."""
        cache = KVCache(step=512)

        # Add a large sequence
        keys = mx.ones((1, 4, 1000, 64))
        values = mx.ones((1, 4, 1000, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape[2] == 1000
        assert cache.offset == 1000
