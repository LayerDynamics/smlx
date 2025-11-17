"""Tests for smlx.kv_cache.kv_manager module."""

import mlx.core as mx
import pytest

from smlx.kv_cache.kv_manager import KVCacheManager
from smlx.kv_cache.mlx_kv import (
    MLXKVCache,
    MLXRotatingKVCache,
    QuantizedMLXKVCache,
)


class TestKVCacheManager:
    """Test KVCacheManager class."""

    def test_kv_cache_manager_init(self):
        """Test KVCacheManager initialization."""
        caches = [MLXKVCache() for _ in range(6)]
        manager = KVCacheManager(
            num_layers=6,
            cache_type="standard",
            caches=caches,
        )

        assert manager.num_layers == 6
        assert manager.cache_type == "standard"
        assert len(manager.caches) == 6
        assert manager.memory_monitor is not None

    def test_kv_cache_manager_init_no_monitoring(self):
        """Test KVCacheManager initialization without memory monitoring."""
        caches = [MLXKVCache() for _ in range(6)]
        manager = KVCacheManager(
            num_layers=6,
            cache_type="standard",
            caches=caches,
            enable_memory_monitoring=False,
        )

        assert manager.memory_monitor is None

    @pytest.mark.unit
    def test_create_standard(self):
        """Test creating standard cache manager."""
        manager = KVCacheManager.create_standard(num_layers=6)

        assert manager.num_layers == 6
        assert manager.cache_type == "standard"
        assert all(isinstance(c, MLXKVCache) for c in manager.caches)

    @pytest.mark.unit
    def test_create_standard_custom_step(self):
        """Test creating standard cache manager with custom step."""
        manager = KVCacheManager.create_standard(
            num_layers=6,
            step=128,
        )

        for cache in manager.caches:
            assert isinstance(cache, MLXKVCache)
            assert cache.step == 128

    @pytest.mark.unit
    def test_create_standard_with_tracing(self):
        """Test creating standard cache manager with tracing."""
        manager = KVCacheManager.create_standard(
            num_layers=6,
            enable_tracing=True,
        )

        assert all(c.enable_tracing is True for c in manager.caches)

    @pytest.mark.unit
    def test_create_rotating(self):
        """Test creating rotating cache manager."""
        manager = KVCacheManager.create_rotating(
            num_layers=6,
            max_kv_size=2048,
            keep=256,
        )

        assert manager.num_layers == 6
        assert manager.cache_type == "rotating"
        assert all(isinstance(c, MLXRotatingKVCache) for c in manager.caches)

        for cache in manager.caches:
            assert isinstance(cache, MLXRotatingKVCache)
            assert cache.max_size == 2048
            assert cache.keep == 256

    @pytest.mark.unit
    def test_create_quantized(self):
        """Test creating quantized cache manager."""
        manager = KVCacheManager.create_quantized(
            num_layers=6,
            bits=4,
            group_size=64,
        )

        assert manager.num_layers == 6
        assert manager.cache_type == "quantized"
        assert all(isinstance(c, QuantizedMLXKVCache) for c in manager.caches)

        for cache in manager.caches:
            assert isinstance(cache, QuantizedMLXKVCache)
            assert cache.bits == 4
            assert cache.group_size == 64

    @pytest.mark.unit
    def test_create_quantized_with_rotation(self):
        """Test creating quantized cache manager with rotation."""
        manager = KVCacheManager.create_quantized(
            num_layers=6,
            bits=4,
            max_size=4096,
            keep=256,
        )

        for cache in manager.caches:
            assert isinstance(cache, QuantizedMLXKVCache)
            assert isinstance(cache.cache, MLXRotatingKVCache)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_create_auto_standard(self):
        """Test auto-configuration choosing standard cache."""
        manager = KVCacheManager.create_auto(
            num_layers=6,
            model_size_gb=0.1,  # Very small model
            target_memory_gb=32.0,
            num_heads=8,
            head_dim=64,
        )

        # Should choose standard cache for plenty of memory
        assert manager.cache_type == "standard"

    @pytest.mark.unit
    def test_reset_all(self):
        """Test resetting all caches."""
        manager = KVCacheManager.create_standard(num_layers=6)

        # Add data to caches
        for cache in manager.caches:
            keys = mx.ones((1, 4, 10, 64))
            values = mx.ones((1, 4, 10, 64))
            cache.update_and_fetch(keys, values)

        # Reset all
        manager.reset_all()

        # All caches should be reset
        for cache in manager.caches:
            assert isinstance(cache, MLXKVCache)
            assert cache.keys is None
        assert all(c.offset == 0 for c in manager.caches)

    @pytest.mark.unit
    def test_get_state_dict(self):
        """Test getting state dict."""
        manager = KVCacheManager.create_standard(num_layers=4)

        # Add data to caches
        for cache in manager.caches:
            keys = mx.ones((1, 4, 10, 64))
            values = mx.ones((1, 4, 10, 64))
            cache.update_and_fetch(keys, values)

        state_dict = manager.get_state_dict()

        assert state_dict["num_layers"] == 4
        assert state_dict["cache_type"] == "standard"
        assert len(state_dict["caches"]) == 4

        for i, cache_state in enumerate(state_dict["caches"]):
            assert cache_state["layer_idx"] == i
            assert cache_state["offset"] == 10

    @pytest.mark.unit
    def test_load_state_dict(self):
        """Test loading state dict."""
        # Create manager and add data
        manager1 = KVCacheManager.create_standard(num_layers=4)

        for cache in manager1.caches:
            keys = mx.ones((1, 4, 10, 64))
            values = mx.ones((1, 4, 10, 64))
            cache.update_and_fetch(keys, values)

        state_dict = manager1.get_state_dict()

        # Create new manager and load state
        manager2 = KVCacheManager.create_standard(num_layers=4)
        manager2.load_state_dict(state_dict)

        # State should be restored
        assert all(c.offset == 10 for c in manager2.caches)

    @pytest.mark.unit
    def test_load_state_dict_wrong_num_layers(self):
        """Test loading state dict with wrong number of layers."""
        manager1 = KVCacheManager.create_standard(num_layers=4)
        state_dict = manager1.get_state_dict()

        manager2 = KVCacheManager.create_standard(num_layers=6)

        with pytest.raises(ValueError, match="layers"):
            manager2.load_state_dict(state_dict)

    @pytest.mark.unit
    def test_load_state_dict_wrong_cache_type(self):
        """Test loading state dict with wrong cache type."""
        manager1 = KVCacheManager.create_standard(num_layers=4)
        state_dict = manager1.get_state_dict()

        manager2 = KVCacheManager.create_rotating(
            num_layers=4,
            max_kv_size=2048,
        )

        with pytest.raises(ValueError, match="cache"):
            manager2.load_state_dict(state_dict)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_check_memory_pressure(self):
        """Test checking memory pressure."""
        manager = KVCacheManager.create_standard(
            num_layers=6,
            enable_memory_monitoring=True,
        )

        status = manager.check_memory_pressure()

        assert status is not None
        assert "status" in status

    @pytest.mark.unit
    def test_check_memory_pressure_disabled(self):
        """Test checking memory pressure when monitoring disabled."""
        manager = KVCacheManager.create_standard(
            num_layers=6,
            enable_memory_monitoring=False,
        )

        status = manager.check_memory_pressure()

        assert status is None

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_memory_trend(self):
        """Test getting memory trend."""
        manager = KVCacheManager.create_standard(
            num_layers=6,
            enable_memory_monitoring=True,
        )

        # Need some checks first
        manager.check_memory_pressure()
        manager.check_memory_pressure()

        trend = manager.get_memory_trend(last_n=2)

        assert trend in ["increasing", "stable", "decreasing"]

    @pytest.mark.unit
    def test_get_trace_summary(self):
        """Test getting trace summary."""
        manager = KVCacheManager.create_standard(
            num_layers=6,
            enable_tracing=True,
        )

        # Add data to caches
        for cache in manager.caches:
            keys = mx.ones((1, 4, 10, 64))
            values = mx.ones((1, 4, 10, 64))
            cache.update_and_fetch(keys, values)

        summaries = manager.get_trace_summary()

        assert len(summaries) == 6
        for i, summary in enumerate(summaries):
            assert summary["layer_idx"] == i

    @pytest.mark.unit
    def test_clear_traces(self):
        """Test clearing traces."""
        manager = KVCacheManager.create_standard(
            num_layers=6,
            enable_tracing=True,
        )

        # Add data
        for cache in manager.caches:
            keys = mx.ones((1, 4, 10, 64))
            values = mx.ones((1, 4, 10, 64))
            cache.update_and_fetch(keys, values)

        manager.clear_traces()

        # All traces should be cleared
        for cache in manager.caches:
            assert isinstance(cache, MLXKVCache)
            assert len(cache.trace_log) == 0

    @pytest.mark.unit
    def test_getitem(self):
        """Test getting cache by index."""
        manager = KVCacheManager.create_standard(num_layers=6)

        cache = manager[0]

        assert isinstance(cache, MLXKVCache)

    @pytest.mark.unit
    def test_len(self):
        """Test getting number of caches."""
        manager = KVCacheManager.create_standard(num_layers=6)

        assert len(manager) == 6

    @pytest.mark.unit
    def test_iter(self):
        """Test iterating over caches."""
        manager = KVCacheManager.create_standard(num_layers=6)

        count = 0
        for cache in manager:
            assert isinstance(cache, MLXKVCache)
            count += 1

        assert count == 6


class TestKVCacheManagerIntegration:
    """Test integration scenarios for KV cache manager."""

    @pytest.mark.unit
    def test_multi_layer_generation(self):
        """Test multi-layer generation with cache manager."""
        manager = KVCacheManager.create_standard(num_layers=6)

        # Simulate forward pass through all layers
        for cache in manager:
            keys = mx.random.normal((1, 12, 10, 64))
            values = mx.random.normal((1, 12, 10, 64))
            cache.update_and_fetch(keys, values)

        # All caches should have same offset
        assert all(c.offset == 10 for c in manager.caches)

    @pytest.mark.unit
    def test_rotating_cache_long_generation(self):
        """Test rotating cache with long generation."""
        manager = KVCacheManager.create_rotating(
            num_layers=6,
            max_kv_size=100,
            keep=10,
        )

        # Generate many tokens
        for _ in range(200):
            for cache in manager:
                keys = mx.random.normal((1, 8, 1, 64))
                values = mx.random.normal((1, 8, 1, 64))
                cache.update_and_fetch(keys, values)

        # All caches should be at max size
        for cache in manager.caches:
            assert isinstance(cache, MLXRotatingKVCache)
            # Get current keys to check size
            keys = mx.ones((1, 8, 1, 64))
            values = mx.ones((1, 8, 1, 64))
            all_keys, _ = cache.update_and_fetch(keys, values)
            assert all_keys.shape[2] <= cache.max_size

    @pytest.mark.unit
    def test_state_persistence(self):
        """Test saving and loading state."""
        manager1 = KVCacheManager.create_standard(num_layers=4)

        # Run some generation
        for _ in range(10):
            for cache in manager1:
                keys = mx.random.normal((1, 8, 1, 64))
                values = mx.random.normal((1, 8, 1, 64))
                cache.update_and_fetch(keys, values)

        # Save state
        state = manager1.get_state_dict()

        # Create new manager and restore
        manager2 = KVCacheManager.create_standard(num_layers=4)
        manager2.load_state_dict(state)

        # Continue generation from restored state
        for cache in manager2:
            keys = mx.random.normal((1, 8, 1, 64))
            values = mx.random.normal((1, 8, 1, 64))
            all_keys, _ = cache.update_and_fetch(keys, values)
            # Should have 11 tokens (10 from saved + 1 new)
            assert all_keys.shape[2] == 11


class TestKVCacheManagerEdgeCases:
    """Test edge cases for KV cache manager."""

    @pytest.mark.unit
    def test_create_manager_zero_layers(self):
        """Test creating manager with zero layers."""
        manager = KVCacheManager.create_standard(num_layers=0)

        assert len(manager) == 0

    @pytest.mark.unit
    def test_create_manager_one_layer(self):
        """Test creating manager with one layer."""
        manager = KVCacheManager.create_standard(num_layers=1)

        assert len(manager) == 1

    @pytest.mark.unit
    def test_reset_all_empty_manager(self):
        """Test resetting empty manager."""
        manager = KVCacheManager.create_standard(num_layers=6)

        # Reset without adding data
        manager.reset_all()

        # Should not raise error
        assert all(c.offset == 0 for c in manager.caches)

    @pytest.mark.unit
    def test_get_trace_summary_tracing_disabled(self):
        """Test getting trace summary when tracing is disabled."""
        manager = KVCacheManager.create_standard(
            num_layers=6,
            enable_tracing=False,
        )

        summaries = manager.get_trace_summary()

        # Should return empty list or list with disabled summaries
        assert isinstance(summaries, list)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_create_auto_small_memory(self):
        """Test auto-configuration with small memory."""
        manager = KVCacheManager.create_auto(
            num_layers=24,
            model_size_gb=1.0,
            target_memory_gb=2.0,  # Very small target
            num_heads=12,
            head_dim=64,
        )

        # Should choose quantized or rotating cache
        assert manager.cache_type in ["quantized", "rotating"]

    @pytest.mark.unit
    def test_iteration_over_caches(self):
        """Test multiple iterations over caches."""
        manager = KVCacheManager.create_standard(num_layers=4)

        # First iteration
        count1 = sum(1 for _ in manager)
        # Second iteration
        count2 = sum(1 for _ in manager)

        assert count1 == 4
        assert count2 == 4
