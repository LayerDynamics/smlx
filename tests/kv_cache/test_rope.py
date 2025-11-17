"""Tests for smlx.kv_cache.rope module."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.kv_cache.mlx_kv import MLXKVCache, MLXRotatingKVCache
from smlx.kv_cache.rope import (
    RoPECache,
    create_rope_module,
    initialize_rope_cache,
)


class TestRoPECache:
    """Test RoPECache class."""

    def test_rope_cache_init(self):
        """Test RoPECache initialization."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        assert rope_cache.cache is cache
        assert rope_cache.rope is rope

    def test_rope_cache_with_rotating_cache(self):
        """Test RoPECache with rotating KV cache."""
        cache = MLXRotatingKVCache(max_size=1024, keep=64)
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        assert isinstance(rope_cache.cache, MLXRotatingKVCache)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_apply_rope_and_update(self):
        """Test applying RoPE and updating cache."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        queries = mx.random.normal((1, 12, 10, 64))
        keys = mx.random.normal((1, 12, 10, 64))
        values = mx.random.normal((1, 12, 10, 64))

        queries_out, all_keys, all_values = rope_cache.apply_rope_and_update(
            queries, keys, values
        )

        # Should return transformed queries and all cached keys/values
        assert queries_out.shape == (1, 12, 10, 64)
        assert all_keys.shape == (1, 12, 10, 64)
        assert all_values.shape == (1, 12, 10, 64)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_apply_rope_offset_tracking(self):
        """Test that RoPE offset is tracked correctly."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        # First update
        queries1 = mx.random.normal((1, 8, 10, 64))
        keys1 = mx.random.normal((1, 8, 10, 64))
        values1 = mx.random.normal((1, 8, 10, 64))
        rope_cache.apply_rope_and_update(queries1, keys1, values1)

        assert rope_cache.offset == 10

        # Second update (single token)
        queries2 = mx.random.normal((1, 8, 1, 64))
        keys2 = mx.random.normal((1, 8, 1, 64))
        values2 = mx.random.normal((1, 8, 1, 64))
        _, all_keys, _ = rope_cache.apply_rope_and_update(queries2, keys2, values2)

        # Offset should be 11
        assert rope_cache.offset == 11
        # All keys should have 11 tokens
        assert all_keys.shape[2] == 11

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_apply_rope_no_cache(self):
        """Test applying RoPE without caching."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        queries = mx.random.normal((1, 8, 10, 64))
        keys = mx.random.normal((1, 8, 10, 64))

        queries_out, keys_out = rope_cache.apply_rope_no_cache(queries, keys)

        # Should return transformed queries and keys
        assert queries_out.shape == (1, 8, 10, 64)
        assert keys_out.shape == (1, 8, 10, 64)
        # Cache should not be updated
        assert rope_cache.offset == 0

    @pytest.mark.unit
    def test_offset_property(self):
        """Test offset property."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        assert rope_cache.offset == 0

        # Update cache manually
        cache.offset = 10
        assert rope_cache.offset == 10

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_state_getter(self):
        """Test getting state."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        queries = mx.random.normal((1, 8, 10, 64))
        keys = mx.random.normal((1, 8, 10, 64))
        values = mx.random.normal((1, 8, 10, 64))
        rope_cache.apply_rope_and_update(queries, keys, values)

        state = rope_cache.state

        # Should return underlying cache state
        assert isinstance(state, tuple)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_state_setter(self):
        """Test setting state."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        # Create state
        keys = mx.random.normal((1, 8, 20, 64))
        values = mx.random.normal((1, 8, 20, 64))
        saved_state = (keys, values)

        # Set state
        rope_cache.state = saved_state

        assert rope_cache.offset == 20

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_reset(self):
        """Test resetting cache."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        queries = mx.random.normal((1, 8, 10, 64))
        keys = mx.random.normal((1, 8, 10, 64))
        values = mx.random.normal((1, 8, 10, 64))
        rope_cache.apply_rope_and_update(queries, keys, values)

        rope_cache.reset()

        assert rope_cache.offset == 0
        assert rope_cache.cache.keys is None

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_trace_summary(self):
        """Test getting trace summary."""
        cache = MLXKVCache(enable_tracing=True)
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        queries = mx.random.normal((1, 8, 10, 64))
        keys = mx.random.normal((1, 8, 10, 64))
        values = mx.random.normal((1, 8, 10, 64))
        rope_cache.apply_rope_and_update(queries, keys, values)

        summary = rope_cache.get_trace_summary()

        assert summary["enabled"] is True

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_clear_trace(self):
        """Test clearing trace."""
        cache = MLXKVCache(enable_tracing=True)
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        queries = mx.random.normal((1, 8, 10, 64))
        keys = mx.random.normal((1, 8, 10, 64))
        values = mx.random.normal((1, 8, 10, 64))
        rope_cache.apply_rope_and_update(queries, keys, values)

        rope_cache.clear_trace()

        summary = rope_cache.get_trace_summary()
        assert len(summary.get("events", [])) == 0


class TestInitializeRoPECache:
    """Test initialize_rope_cache function."""

    def test_initialize_rope_cache_standard(self):
        """Test creating standard RoPE caches."""
        caches = initialize_rope_cache(dims=64, num_layers=6)

        assert len(caches) == 6
        assert all(isinstance(c, RoPECache) for c in caches)
        assert all(isinstance(c.cache, MLXKVCache) for c in caches)

    def test_initialize_rope_cache_rotating(self):
        """Test creating rotating RoPE caches."""
        caches = initialize_rope_cache(
            dims=64,
            num_layers=6,
            max_kv_size=2048,
            keep=256,
        )

        assert len(caches) == 6
        assert all(isinstance(c.cache, MLXRotatingKVCache) for c in caches)
        for c in caches:
            assert isinstance(c.cache, MLXRotatingKVCache)
            assert c.cache.max_size == 2048
            assert c.cache.keep == 256

    def test_initialize_rope_cache_custom_base(self):
        """Test creating RoPE caches with custom base."""
        caches = initialize_rope_cache(
            dims=64,
            base=50000.0,
            num_layers=4,
        )

        assert len(caches) == 4
        # All should share same RoPE module (check base)
        for cache in caches:
            assert cache.rope.base == 50000.0

    def test_initialize_rope_cache_traditional(self):
        """Test creating RoPE caches with traditional mode."""
        caches = initialize_rope_cache(
            dims=64,
            traditional=True,
            num_layers=4,
        )

        for cache in caches:
            assert cache.rope.traditional is True

    def test_initialize_rope_cache_with_tracing(self):
        """Test creating RoPE caches with tracing enabled."""
        caches = initialize_rope_cache(
            dims=64,
            num_layers=4,
            enable_tracing=True,
        )

        for cache in caches:
            assert cache.cache.enable_tracing is True

    def test_initialize_rope_cache_custom_step(self):
        """Test creating RoPE caches with custom step."""
        caches = initialize_rope_cache(
            dims=64,
            num_layers=4,
            step=128,
        )

        for cache in caches:
            assert cache.cache.step == 128

    def test_initialize_rope_cache_single_layer(self):
        """Test creating single layer cache."""
        caches = initialize_rope_cache(dims=64, num_layers=1)

        assert len(caches) == 1
        assert isinstance(caches[0], RoPECache)


class TestCreateRoPEModule:
    """Test create_rope_module function."""

    def test_create_rope_module_default(self):
        """Test creating default RoPE module."""
        rope = create_rope_module(dims=64)

        assert isinstance(rope, nn.RoPE)
        assert rope.dims == 64
        assert rope.base == 10000.0
        assert rope.traditional is False

    def test_create_rope_module_custom_base(self):
        """Test creating RoPE module with custom base."""
        rope = create_rope_module(dims=64, base=50000.0)

        assert rope.base == 50000.0

    def test_create_rope_module_traditional(self):
        """Test creating traditional RoPE module."""
        rope = create_rope_module(dims=64, traditional=True)

        assert rope.traditional is True

    def test_create_rope_module_linear_scaling(self):
        """Test creating RoPE module with linear scaling."""
        rope = create_rope_module(
            dims=64,
            scaling_config={"type": "linear", "factor": 2.0},
        )

        assert isinstance(rope, nn.RoPE)
        # Scale should be 1 / factor = 0.5
        assert rope.scale == 0.5

    def test_create_rope_module_linear_scaling_alt_key(self):
        """Test creating RoPE module with linear scaling (rope_type key)."""
        rope = create_rope_module(
            dims=64,
            scaling_config={"rope_type": "linear", "factor": 4.0},
        )

        assert rope.scale == 0.25

    def test_create_rope_module_default_scaling(self):
        """Test creating RoPE module with default scaling."""
        rope = create_rope_module(
            dims=64,
            scaling_config={"type": "default"},
        )

        assert rope.scale == 1.0

    def test_create_rope_module_unsupported_scaling(self):
        """Test creating RoPE module with unsupported scaling type."""
        with pytest.raises(ValueError, match="Unsupported RoPE type"):
            create_rope_module(
                dims=64,
                scaling_config={"type": "unsupported"},
            )


class TestRoPECacheIntegration:
    """Test integration scenarios for RoPE cache."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_multi_layer_generation(self):
        """Test multi-layer generation with RoPE."""
        num_layers = 6
        caches = initialize_rope_cache(dims=64, num_layers=num_layers)

        # Simulate forward pass through all layers
        for layer_idx in range(num_layers):
            queries = mx.random.normal((1, 12, 10, 64))
            keys = mx.random.normal((1, 12, 10, 64))
            values = mx.random.normal((1, 12, 10, 64))
            queries_out, all_keys, _ = caches[layer_idx].apply_rope_and_update(
                queries, keys, values
            )

            assert queries_out.shape == (1, 12, 10, 64)
            assert all_keys.shape == (1, 12, 10, 64)

        # All caches should have same offset
        assert all(c.offset == 10 for c in caches)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_autoregressive_generation(self):
        """Test autoregressive generation with RoPE."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        # Initial prompt
        queries = mx.random.normal((1, 8, 10, 64))
        keys = mx.random.normal((1, 8, 10, 64))
        values = mx.random.normal((1, 8, 10, 64))
        rope_cache.apply_rope_and_update(queries, keys, values)

        # Generate 5 tokens
        for i in range(5):
            queries = mx.random.normal((1, 8, 1, 64))
            keys = mx.random.normal((1, 8, 1, 64))
            values = mx.random.normal((1, 8, 1, 64))
            _, all_keys, _ = rope_cache.apply_rope_and_update(queries, keys, values)

            expected_len = 10 + i + 1
            assert all_keys.shape[2] == expected_len

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_rope_with_rotating_cache_long_sequence(self):
        """Test RoPE with rotating cache for long sequences."""
        cache = MLXRotatingKVCache(max_size=100, keep=10)
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        # Initial prompt
        queries = mx.random.normal((1, 8, 10, 64))
        keys = mx.random.normal((1, 8, 10, 64))
        values = mx.random.normal((1, 8, 10, 64))
        rope_cache.apply_rope_and_update(queries, keys, values)

        # Generate 200 tokens (exceeds max_size)
        for _ in range(200):
            queries = mx.random.normal((1, 8, 1, 64))
            keys = mx.random.normal((1, 8, 1, 64))
            values = mx.random.normal((1, 8, 1, 64))
            _, all_keys, _ = rope_cache.apply_rope_and_update(queries, keys, values)

        # Cache should stay at max_size
        assert all_keys.shape[2] == 100
        # Offset should track total tokens
        assert rope_cache.offset == 210


class TestRoPECacheEdgeCases:
    """Test edge cases for RoPE cache."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_rope_single_token(self):
        """Test RoPE with single token."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        queries = mx.random.normal((1, 8, 1, 64))
        keys = mx.random.normal((1, 8, 1, 64))
        values = mx.random.normal((1, 8, 1, 64))

        queries_out, all_keys, _ = rope_cache.apply_rope_and_update(queries, keys, values)

        assert queries_out.shape == (1, 8, 1, 64)
        assert all_keys.shape == (1, 8, 1, 64)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_rope_different_dims(self):
        """Test RoPE with different head dimensions."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=128, traditional=False)
        rope_cache = RoPECache(cache, rope)

        queries = mx.random.normal((1, 8, 10, 128))
        keys = mx.random.normal((1, 8, 10, 128))
        values = mx.random.normal((1, 8, 10, 128))

        queries_out, all_keys, _ = rope_cache.apply_rope_and_update(queries, keys, values)

        assert queries_out.shape == (1, 8, 10, 128)
        assert all_keys.shape == (1, 8, 10, 128)

    @pytest.mark.unit
    def test_rope_trace_disabled(self):
        """Test RoPE with tracing disabled."""
        cache = MLXKVCache(enable_tracing=False)
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        summary = rope_cache.get_trace_summary()
        assert summary["enabled"] is False

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_rope_cache_batch_processing(self):
        """Test RoPE cache with batch processing."""
        cache = MLXKVCache()
        rope = nn.RoPE(dims=64, traditional=False)
        rope_cache = RoPECache(cache, rope)

        # Batch size of 4
        queries = mx.random.normal((4, 8, 10, 64))
        keys = mx.random.normal((4, 8, 10, 64))
        values = mx.random.normal((4, 8, 10, 64))

        queries_out, all_keys, all_values = rope_cache.apply_rope_and_update(
            queries, keys, values
        )

        assert queries_out.shape == (4, 8, 10, 64)
        assert all_keys.shape == (4, 8, 10, 64)
        assert all_values.shape == (4, 8, 10, 64)

    def test_rope_module_scaling_none(self):
        """Test creating RoPE module with no scaling config."""
        rope = create_rope_module(dims=64, scaling_config=None)

        assert rope.scale == 1.0
