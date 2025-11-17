"""Tests for smlx.kv_cache.mlx_kv module."""

import mlx.core as mx
import pytest

from smlx.kv_cache.mlx_kv import (
    MLXKVCache,
    MLXRotatingKVCache,
    QuantizedMLXKVCache,
)


class TestMLXKVCache:
    """Test MLXKVCache class."""

    def test_mlx_kv_cache_init(self):
        """Test MLXKVCache initialization."""
        cache = MLXKVCache()

        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0
        assert cache.step == 256
        assert cache.enable_tracing is False
        assert cache.trace_log == []

    def test_mlx_kv_cache_with_tracing(self):
        """Test MLXKVCache with tracing enabled."""
        cache = MLXKVCache(enable_tracing=True)

        assert cache.enable_tracing is True
        assert isinstance(cache.trace_log, list)

    def test_mlx_kv_cache_custom_step(self):
        """Test MLXKVCache with custom step size."""
        cache = MLXKVCache(step=128)

        assert cache.step == 128

    @pytest.mark.unit
    def test_mlx_kv_cache_update_and_fetch(self):
        """Test update_and_fetch without tracing."""
        cache = MLXKVCache()

        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape == (1, 4, 10, 64)
        assert all_values.shape == (1, 4, 10, 64)
        assert cache.offset == 10

    @pytest.mark.unit
    def test_mlx_kv_cache_update_with_tracing(self):
        """Test update_and_fetch with tracing enabled."""
        cache = MLXKVCache(enable_tracing=True)

        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))

        cache.update_and_fetch(keys, values)

        # Should have one trace event
        assert len(cache.trace_log) == 1
        event = cache.trace_log[0]

        assert event["type"] == "update"
        assert event["old_offset"] == 0
        assert event["new_offset"] == 10
        assert event["keys_shape"] == (1, 4, 10, 64)
        assert event["values_shape"] == (1, 4, 10, 64)
        assert "elapsed_ms" in event
        assert "timestamp" in event

    @pytest.mark.unit
    def test_mlx_kv_cache_get_trace_summary(self):
        """Test getting trace summary."""
        cache = MLXKVCache(enable_tracing=True)

        # Add some updates
        for _ in range(5):
            keys = mx.ones((1, 4, 1, 64))
            values = mx.ones((1, 4, 1, 64))
            cache.update_and_fetch(keys, values)

        summary = cache.get_trace_summary()

        assert summary["enabled"] is True
        assert summary["total_updates"] == 5
        assert summary["current_offset"] == 5
        assert "total_time_ms" in summary
        assert "avg_time_ms" in summary
        assert "events" in summary

    @pytest.mark.unit
    def test_mlx_kv_cache_get_trace_summary_disabled(self):
        """Test getting trace summary when tracing is disabled."""
        cache = MLXKVCache(enable_tracing=False)

        summary = cache.get_trace_summary()

        assert summary["enabled"] is False

    @pytest.mark.unit
    def test_mlx_kv_cache_clear_trace(self):
        """Test clearing trace log."""
        cache = MLXKVCache(enable_tracing=True)

        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        assert len(cache.trace_log) > 0

        cache.clear_trace()

        assert len(cache.trace_log) == 0

    @pytest.mark.unit
    def test_mlx_kv_cache_reset(self):
        """Test resetting cache."""
        cache = MLXKVCache(enable_tracing=True)

        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        cache.reset()

        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0
        assert len(cache.trace_log) == 0


class TestMLXRotatingKVCache:
    """Test MLXRotatingKVCache class."""

    def test_mlx_rotating_cache_init(self):
        """Test MLXRotatingKVCache initialization."""
        cache = MLXRotatingKVCache(max_size=1024, keep=64)

        assert cache.max_size == 1024
        assert cache.keep == 64
        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0
        assert cache.enable_tracing is False

    def test_mlx_rotating_cache_with_tracing(self):
        """Test MLXRotatingKVCache with tracing enabled."""
        cache = MLXRotatingKVCache(max_size=1024, enable_tracing=True)

        assert cache.enable_tracing is True

    @pytest.mark.unit
    def test_mlx_rotating_cache_update(self):
        """Test updating rotating cache."""
        cache = MLXRotatingKVCache(max_size=1024, keep=0)

        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape[2] == 10
        assert cache.offset == 10

    @pytest.mark.unit
    def test_mlx_rotating_cache_rotation(self):
        """Test cache rotation."""
        cache = MLXRotatingKVCache(max_size=100, keep=10)

        # Fill beyond max_size
        for i in range(120):
            keys = mx.full((1, 4, 1, 64), float(i))
            values = mx.full((1, 4, 1, 64), float(i))
            cache.update_and_fetch(keys, values)

        # Cache should be at max size
        assert cache.offset == 120

    @pytest.mark.unit
    def test_mlx_rotating_cache_trace_rotation(self):
        """Test tracing rotation events."""
        cache = MLXRotatingKVCache(max_size=50, keep=10, enable_tracing=True)

        # Add enough tokens to trigger rotation
        for _ in range(60):
            keys = mx.ones((1, 4, 1, 64))
            values = mx.ones((1, 4, 1, 64))
            cache.update_and_fetch(keys, values)

        # Check for rotation events
        rotations = [e for e in cache.trace_log if e.get("rotated", False)]

        # Should have some rotations
        assert len(rotations) > 0

    @pytest.mark.unit
    def test_mlx_rotating_cache_get_trace_summary(self):
        """Test getting trace summary for rotating cache."""
        cache = MLXRotatingKVCache(max_size=100, keep=10, enable_tracing=True)

        # Add tokens
        for _ in range(150):
            keys = mx.ones((1, 4, 1, 64))
            values = mx.ones((1, 4, 1, 64))
            cache.update_and_fetch(keys, values)

        summary = cache.get_trace_summary()

        assert summary["enabled"] is True
        assert summary["total_updates"] == 150
        assert summary["total_rotations"] > 0
        assert summary["max_size"] == 100
        assert summary["keep"] == 10

    @pytest.mark.unit
    def test_mlx_rotating_cache_reset(self):
        """Test resetting rotating cache."""
        cache = MLXRotatingKVCache(max_size=1024, keep=10, enable_tracing=True)

        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        cache.reset()

        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0
        assert len(cache.trace_log) == 0


class TestQuantizedMLXKVCache:
    """Test QuantizedMLXKVCache class."""

    def test_quantized_cache_init(self):
        """Test QuantizedMLXKVCache initialization."""
        cache = QuantizedMLXKVCache(bits=4, group_size=64)

        assert cache.bits == 4
        assert cache.group_size == 64
        assert cache.quantize_threshold == 256
        assert cache.is_quantized is False
        assert cache.quantized_keys is None
        assert cache.quantized_values is None

    def test_quantized_cache_with_max_size(self):
        """Test QuantizedMLXKVCache with max_size (rotating)."""
        cache = QuantizedMLXKVCache(bits=4, max_size=1024, keep=64)

        assert isinstance(cache.cache, MLXRotatingKVCache)

    def test_quantized_cache_without_max_size(self):
        """Test QuantizedMLXKVCache without max_size (standard)."""
        cache = QuantizedMLXKVCache(bits=4)

        assert isinstance(cache.cache, MLXKVCache)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_quantized_cache_update_before_threshold(self):
        """Test update before quantization threshold."""
        cache = QuantizedMLXKVCache(bits=4, quantize_threshold=256)

        # Add less than threshold
        keys = mx.random.normal((1, 4, 100, 64))
        values = mx.random.normal((1, 4, 100, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        # Should not be quantized yet
        assert cache.is_quantized is False
        assert all_keys.shape[2] == 100

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_quantized_cache_update_after_threshold(self):
        """Test update after quantization threshold."""
        cache = QuantizedMLXKVCache(bits=4, quantize_threshold=100)

        # Add more than threshold
        keys = mx.random.normal((1, 4, 150, 64))
        values = mx.random.normal((1, 4, 150, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        # Should be quantized now
        assert cache.is_quantized is True
        assert cache.quantized_keys is not None
        assert cache.quantized_values is not None

    @pytest.mark.unit
    def test_quantized_cache_offset_property(self):
        """Test offset property."""
        cache = QuantizedMLXKVCache(bits=4)

        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        assert cache.offset == 10

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_quantized_cache_state_unquantized(self):
        """Test getting state when not quantized."""
        cache = QuantizedMLXKVCache(bits=4, quantize_threshold=1000)

        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        state = cache.state

        assert state["quantized"] is False
        assert "keys" in state
        assert "values" in state

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_quantized_cache_state_quantized(self):
        """Test getting state when quantized."""
        cache = QuantizedMLXKVCache(bits=4, quantize_threshold=10)

        keys = mx.random.normal((1, 4, 20, 64))
        values = mx.random.normal((1, 4, 20, 64))
        cache.update_and_fetch(keys, values)

        state = cache.state

        assert state["quantized"] is True
        assert state["bits"] == 4
        assert "quantized_keys" in state
        assert "quantized_values" in state

    @pytest.mark.unit
    def test_quantized_cache_reset(self):
        """Test resetting quantized cache."""
        cache = QuantizedMLXKVCache(bits=4, quantize_threshold=10)

        keys = mx.random.normal((1, 4, 20, 64))
        values = mx.random.normal((1, 4, 20, 64))
        cache.update_and_fetch(keys, values)

        cache.reset()

        assert cache.offset == 0
        assert cache.is_quantized is False
        assert cache.quantized_keys is None
        assert cache.quantized_values is None

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_quantized_cache_get_trace_summary(self):
        """Test getting trace summary from quantized cache."""
        cache = QuantizedMLXKVCache(bits=4, enable_tracing=True)

        keys = mx.ones((1, 4, 10, 64))
        values = mx.ones((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        summary = cache.get_trace_summary()

        assert "quantized" in summary


class TestMLXKVCacheIntegration:
    """Test integration scenarios for MLX KV caches."""

    @pytest.mark.unit
    def test_multi_layer_caching_with_tracing(self):
        """Test multi-layer caching with trace enabled."""
        num_layers = 6
        caches = [MLXKVCache(enable_tracing=True) for _ in range(num_layers)]

        # Simulate forward pass
        for layer_idx in range(num_layers):
            keys = mx.random.normal((1, 4, 10, 64))
            values = mx.random.normal((1, 4, 10, 64))
            caches[layer_idx].update_and_fetch(keys, values)

        # All caches should have trace
        for cache in caches:
            summary = cache.get_trace_summary()
            assert summary["total_updates"] == 1

    @pytest.mark.unit
    def test_autoregressive_with_rotating_cache(self):
        """Test autoregressive generation with rotating cache."""
        cache = MLXRotatingKVCache(max_size=100, keep=10)

        # Initial prompt
        keys = mx.random.normal((1, 4, 10, 64))
        values = mx.random.normal((1, 4, 10, 64))
        cache.update_and_fetch(keys, values)

        # Generate tokens
        for _ in range(100):
            keys = mx.random.normal((1, 4, 1, 64))
            values = mx.random.normal((1, 4, 1, 64))
            all_keys, all_values = cache.update_and_fetch(keys, values)

        # Cache should be at max size
        assert all_keys.shape[2] == cache.max_size

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_quantized_cache_memory_savings(self):
        """Test that quantized cache uses less memory (conceptually)."""
        # This is a conceptual test - actual memory measurement would require
        # deeper integration with MLX memory tracking
        cache = QuantizedMLXKVCache(bits=4, quantize_threshold=100)

        keys = mx.random.normal((1, 4, 200, 64))
        values = mx.random.normal((1, 4, 200, 64))
        cache.update_and_fetch(keys, values)

        # Should be quantized
        assert cache.is_quantized is True

        # Quantized representation should exist
        assert cache.quantized_keys is not None


class TestMLXKVCacheEdgeCases:
    """Test edge cases for MLX KV caches."""

    @pytest.mark.unit
    def test_mlx_kv_cache_empty_trace_summary(self):
        """Test trace summary with no updates."""
        cache = MLXKVCache(enable_tracing=True)

        summary = cache.get_trace_summary()

        assert summary["enabled"] is True
        assert summary["total_updates"] == 0
        assert summary["avg_time_ms"] == 0

    @pytest.mark.unit
    def test_mlx_rotating_cache_exact_max_size(self):
        """Test rotating cache with exactly max_size tokens."""
        cache = MLXRotatingKVCache(max_size=100, keep=0)

        keys = mx.ones((1, 4, 100, 64))
        values = mx.ones((1, 4, 100, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape[2] == 100

    @pytest.mark.unit
    def test_quantized_cache_8bit(self):
        """Test that 8-bit quantization raises an appropriate error."""
        # MLX group-based quantization only supports 4-bit
        with pytest.raises(ValueError, match="only supports 4-bit"):
            QuantizedMLXKVCache(bits=8, quantize_threshold=50)

    @pytest.mark.unit
    def test_mlx_kv_cache_clear_trace_when_disabled(self):
        """Test clearing trace when tracing is disabled."""
        cache = MLXKVCache(enable_tracing=False)

        # Should not raise error
        cache.clear_trace()

    @pytest.mark.unit
    def test_mlx_rotating_cache_keep_equals_max(self):
        """Test rotating cache when keep equals max_size."""
        cache = MLXRotatingKVCache(max_size=100, keep=100)

        keys = mx.ones((1, 4, 50, 64))
        values = mx.ones((1, 4, 50, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)

        assert all_keys.shape[2] == 50
