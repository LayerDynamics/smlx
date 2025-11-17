"""
Tests for Model Lifecycle Manager.

This module tests ModelLifecycleManager, ModelCache, ModelTelemetry, and configuration.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from smlx.models.smlx_manager import (
    CacheConfig,
    ModelCache,
    ModelLifecycleManager,
    ModelStats,
    ModelTelemetry,
    TelemetryConfig,
    get_manager,
)


# ============================================================================
# CacheConfig Tests
# ============================================================================


@pytest.mark.unit
def test_cache_config_defaults():
    """Test default CacheConfig values."""
    config = CacheConfig()

    assert config.max_models == 3
    assert config.max_memory_gb == 24.0
    assert config.min_free_memory_gb == 4.0
    assert config.enable_eviction is True
    assert config.eviction_threshold == 0.8


@pytest.mark.unit
def test_cache_config_custom():
    """Test custom CacheConfig."""
    config = CacheConfig(
        max_models=5,
        max_memory_gb=32.0,
        min_free_memory_gb=8.0,
        enable_eviction=False,
        eviction_threshold=0.9,
    )

    assert config.max_models == 5
    assert config.max_memory_gb == 32.0
    assert config.min_free_memory_gb == 8.0
    assert config.enable_eviction is False
    assert config.eviction_threshold == 0.9


@pytest.mark.unit
def test_cache_config_validate_valid():
    """Test validation of valid config."""
    config = CacheConfig()
    config.validate()  # Should not raise


@pytest.mark.unit
def test_cache_config_validate_invalid_max_models():
    """Test validation with invalid max_models."""
    config = CacheConfig(max_models=0)

    with pytest.raises(ValueError, match="max_models must be >= 1"):
        config.validate()


@pytest.mark.unit
def test_cache_config_validate_invalid_max_memory():
    """Test validation with invalid max_memory_gb."""
    config = CacheConfig(max_memory_gb=-1.0)

    with pytest.raises(ValueError, match="max_memory_gb must be > 0"):
        config.validate()


@pytest.mark.unit
def test_cache_config_validate_invalid_min_free_memory():
    """Test validation with invalid min_free_memory_gb."""
    config = CacheConfig(min_free_memory_gb=-1.0)

    with pytest.raises(ValueError, match="min_free_memory_gb must be >= 0"):
        config.validate()


@pytest.mark.unit
def test_cache_config_validate_invalid_eviction_threshold():
    """Test validation with invalid eviction_threshold."""
    config = CacheConfig(eviction_threshold=1.5)

    with pytest.raises(ValueError, match="eviction_threshold must be in"):
        config.validate()


# ============================================================================
# TelemetryConfig Tests
# ============================================================================


@pytest.mark.unit
def test_telemetry_config_defaults():
    """Test default TelemetryConfig values."""
    config = TelemetryConfig()

    assert config.enable_telemetry is True
    assert config.track_latency is True
    assert config.track_memory is True
    assert config.track_errors is True
    assert config.retention_hours == 24


@pytest.mark.unit
def test_telemetry_config_custom():
    """Test custom TelemetryConfig."""
    config = TelemetryConfig(
        enable_telemetry=False,
        track_latency=False,
        track_memory=False,
        track_errors=False,
        retention_hours=48,
    )

    assert config.enable_telemetry is False
    assert config.track_latency is False
    assert config.track_memory is False
    assert config.track_errors is False
    assert config.retention_hours == 48


# ============================================================================
# ModelStats Tests
# ============================================================================


@pytest.mark.unit
def test_model_stats_defaults():
    """Test default ModelStats values."""
    stats = ModelStats(model_id="test-model")

    assert stats.model_id == "test-model"
    assert stats.load_count == 0
    assert stats.inference_count == 0
    assert stats.error_count == 0
    assert stats.total_latency_ms == 0.0
    assert stats.memory_mb == 0.0
    assert stats.avg_latency_ms == 0.0


@pytest.mark.unit
def test_model_stats_avg_latency():
    """Test average latency calculation."""
    stats = ModelStats(model_id="test-model")

    stats.inference_count = 10
    stats.total_latency_ms = 100.0

    assert stats.avg_latency_ms == 10.0


@pytest.mark.unit
def test_model_stats_to_dict():
    """Test conversion to dictionary."""
    stats = ModelStats(
        model_id="test-model",
        load_count=5,
        inference_count=10,
        error_count=2,
        total_latency_ms=100.0,
        memory_mb=256.0,
    )

    result = stats.to_dict()

    assert result["model_id"] == "test-model"
    assert result["load_count"] == 5
    assert result["inference_count"] == 10
    assert result["error_count"] == 2
    assert result["avg_latency_ms"] == 10.0
    assert result["memory_mb"] == 256.0
    assert "last_used" in result
    assert "first_loaded" in result


# ============================================================================
# ModelTelemetry Tests
# ============================================================================


@pytest.mark.unit
def test_telemetry_initialization():
    """Test ModelTelemetry initialization."""
    telemetry = ModelTelemetry()

    assert telemetry.config.enable_telemetry is True
    assert len(telemetry._stats) == 0


@pytest.mark.unit
def test_telemetry_record_load():
    """Test recording load events."""
    telemetry = ModelTelemetry()

    telemetry.record_load("model-1", memory_mb=100.0)
    telemetry.record_load("model-1", memory_mb=100.0)
    telemetry.record_load("model-2", memory_mb=200.0)

    stats = telemetry.get_stats()
    assert stats["total_models"] == 2
    assert stats["total_loads"] == 3

    model_1_stats = telemetry.get_stats("model-1")
    assert model_1_stats.load_count == 2
    assert model_1_stats.memory_mb == 100.0


@pytest.mark.unit
def test_telemetry_record_inference():
    """Test recording inference events."""
    telemetry = ModelTelemetry()

    telemetry.record_inference("model-1", latency_ms=10.0)
    telemetry.record_inference("model-1", latency_ms=20.0)

    stats = telemetry.get_stats("model-1")
    assert stats.inference_count == 2
    assert stats.total_latency_ms == 30.0
    assert stats.avg_latency_ms == 15.0


@pytest.mark.unit
def test_telemetry_record_error():
    """Test recording error events."""
    telemetry = ModelTelemetry()

    telemetry.record_error("model-1")
    telemetry.record_error("model-1")

    stats = telemetry.get_stats("model-1")
    assert stats.error_count == 2


@pytest.mark.unit
def test_telemetry_disabled():
    """Test telemetry when disabled."""
    config = TelemetryConfig(enable_telemetry=False)
    telemetry = ModelTelemetry(config=config)

    telemetry.record_load("model-1")
    telemetry.record_inference("model-1", latency_ms=10.0)
    telemetry.record_error("model-1")

    stats = telemetry.get_stats()
    assert stats["total_models"] == 0


@pytest.mark.unit
def test_telemetry_reset_stats():
    """Test resetting statistics."""
    telemetry = ModelTelemetry()

    telemetry.record_load("model-1")
    telemetry.record_load("model-2")

    assert len(telemetry._stats) == 2

    # Reset specific model
    telemetry.reset_stats("model-1")
    assert len(telemetry._stats) == 1
    assert "model-2" in telemetry._stats

    # Reset all
    telemetry.reset_stats()
    assert len(telemetry._stats) == 0


# ============================================================================
# ModelCache Tests
# ============================================================================


@pytest.mark.unit
def test_cache_initialization():
    """Test ModelCache initialization."""
    config = CacheConfig(max_models=2)
    cache = ModelCache(config=config)

    assert cache.config.max_models == 2
    assert len(cache._cache) == 0


@pytest.mark.unit
def test_cache_put_and_get():
    """Test caching models."""
    cache = ModelCache()

    model = MagicMock()
    tokenizer = MagicMock()

    # Cache miss
    result = cache.get("model-1")
    assert result is None

    # Put model
    cache.put("model-1", model, tokenizer)

    # Cache hit
    result = cache.get("model-1")
    assert result is not None
    assert result[0] == model
    assert result[1] == tokenizer


@pytest.mark.unit
def test_cache_lru_eviction():
    """Test LRU eviction when cache is full."""
    config = CacheConfig(max_models=2)
    cache = ModelCache(config=config)

    # Add 2 models (fill cache)
    cache.put("model-1", MagicMock(), MagicMock())
    cache.put("model-2", MagicMock(), MagicMock())

    # Add 3rd model (should evict model-1)
    cache.put("model-3", MagicMock(), MagicMock())

    assert cache.get("model-1") is None  # Evicted
    assert cache.get("model-2") is not None
    assert cache.get("model-3") is not None


@pytest.mark.unit
def test_cache_lru_access_order():
    """Test LRU access order."""
    config = CacheConfig(max_models=2)
    cache = ModelCache(config=config)

    cache.put("model-1", MagicMock(), MagicMock())
    cache.put("model-2", MagicMock(), MagicMock())

    # Access model-1 (makes it most recently used)
    cache.get("model-1")

    # Add model-3 (should evict model-2, not model-1)
    cache.put("model-3", MagicMock(), MagicMock())

    assert cache.get("model-1") is not None  # Still in cache
    assert cache.get("model-2") is None  # Evicted
    assert cache.get("model-3") is not None


@pytest.mark.unit
def test_cache_remove():
    """Test removing models from cache."""
    cache = ModelCache()

    cache.put("model-1", MagicMock(), MagicMock())

    # Remove existing model
    result = cache.remove("model-1")
    assert result is True
    assert cache.get("model-1") is None

    # Remove non-existent model
    result = cache.remove("model-2")
    assert result is False


@pytest.mark.unit
def test_cache_clear():
    """Test clearing cache."""
    cache = ModelCache()

    cache.put("model-1", MagicMock(), MagicMock())
    cache.put("model-2", MagicMock(), MagicMock())

    assert len(cache._cache) == 2

    cache.clear()

    assert len(cache._cache) == 0
    assert cache.get("model-1") is None
    assert cache.get("model-2") is None


@pytest.mark.unit
def test_cache_get_cached_models():
    """Test getting list of cached models."""
    cache = ModelCache()

    cache.put("model-1", MagicMock(), MagicMock())
    cache.put("model-2", MagicMock(), MagicMock())
    cache.put("model-3", MagicMock(), MagicMock())

    models = cache.get_cached_models()

    assert len(models) == 3
    assert "model-3" == models[0]  # Most recent first
    assert "model-2" == models[1]
    assert "model-1" == models[2]


# ============================================================================
# ModelLifecycleManager Tests
# ============================================================================


@pytest.mark.unit
def test_manager_initialization():
    """Test ModelLifecycleManager initialization."""
    manager = ModelLifecycleManager()

    assert manager.cache is not None
    assert manager.telemetry is not None


@pytest.mark.unit
def test_manager_load_model():
    """Test loading model with manager."""
    manager = ModelLifecycleManager()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = (mock_model, mock_tokenizer)

    with patch("smlx.models.smlx_manager.infer_model_type", return_value="smollm2-135m"):
        with patch("smlx.models.smlx_manager.get_model_loader", return_value=mock_loader):
            model, tokenizer = manager.load_model("test-model")

            assert model == mock_model
            assert tokenizer == mock_tokenizer
            mock_loader.load.assert_called_once()


@pytest.mark.unit
def test_manager_load_model_with_quantization():
    """Test loading model with quantization."""
    manager = ModelLifecycleManager()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = (mock_model, mock_tokenizer)

    with patch("smlx.models.smlx_manager.infer_model_type", return_value="smollm2-135m"):
        with patch("smlx.models.smlx_manager.get_model_loader", return_value=mock_loader):
            model, tokenizer = manager.load_model("test-model", quantization="4bit")

            assert model == mock_model
            assert tokenizer == mock_tokenizer
            mock_loader.load.assert_called_with("test-model", quantization="4bit")


@pytest.mark.unit
def test_manager_load_model_caching():
    """Test model caching."""
    manager = ModelLifecycleManager()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = (mock_model, mock_tokenizer)

    with patch("smlx.models.smlx_manager.infer_model_type", return_value="smollm2-135m"):
        with patch("smlx.models.smlx_manager.get_model_loader", return_value=mock_loader):
            # Load model first time
            model1, tokenizer1 = manager.load_model("test-model")

            # Load model second time (should use cache)
            model2, tokenizer2 = manager.load_model("test-model")

            # Should be same instances
            assert model1 == model2
            assert tokenizer1 == tokenizer2

            # Loader should only be called once
            assert mock_loader.load.call_count == 1


@pytest.mark.unit
def test_manager_load_model_force_reload():
    """Test force reloading model."""
    manager = ModelLifecycleManager()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = (mock_model, mock_tokenizer)

    with patch("smlx.models.smlx_manager.infer_model_type", return_value="smollm2-135m"):
        with patch("smlx.models.smlx_manager.get_model_loader", return_value=mock_loader):
            # Load model first time
            manager.load_model("test-model")

            # Force reload
            manager.load_model("test-model", force_reload=True)

            # Loader should be called twice
            assert mock_loader.load.call_count == 2


@pytest.mark.unit
def test_manager_load_model_error():
    """Test error handling during model load."""
    manager = ModelLifecycleManager()

    mock_loader = MagicMock()
    mock_loader.load.side_effect = RuntimeError("Load failed")

    with patch("smlx.models.smlx_manager.infer_model_type", return_value="smollm2-135m"):
        with patch("smlx.models.smlx_manager.get_model_loader", return_value=mock_loader):
            with pytest.raises(RuntimeError, match="Load failed"):
                manager.load_model("test-model")

            # Error should be recorded in telemetry
            stats = manager.get_stats()
            assert stats["total_errors"] == 1


@pytest.mark.unit
def test_manager_unload_model():
    """Test unloading model."""
    manager = ModelLifecycleManager()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = (mock_model, mock_tokenizer)

    with patch("smlx.models.smlx_manager.infer_model_type", return_value="smollm2-135m"):
        with patch("smlx.models.smlx_manager.get_model_loader", return_value=mock_loader):
            # Load model
            manager.load_model("test-model")

            # Verify it's cached
            assert len(manager.get_cached_models()) == 1

            # Unload model
            result = manager.unload_model("test-model")

            assert result is True
            assert len(manager.get_cached_models()) == 0


@pytest.mark.unit
def test_manager_get_cached_models():
    """Test getting cached models."""
    manager = ModelLifecycleManager()

    mock_loader = MagicMock()
    mock_loader.load.return_value = (MagicMock(), MagicMock())

    with patch("smlx.models.smlx_manager.infer_model_type", return_value="smollm2-135m"):
        with patch("smlx.models.smlx_manager.get_model_loader", return_value=mock_loader):
            manager.load_model("model-1")
            manager.load_model("model-2")

            cached = manager.get_cached_models()

            assert len(cached) == 2
            assert "model-2:none" in cached
            assert "model-1:none" in cached


@pytest.mark.unit
def test_manager_get_stats():
    """Test getting telemetry statistics."""
    manager = ModelLifecycleManager()

    mock_loader = MagicMock()
    mock_loader.load.return_value = (MagicMock(), MagicMock())

    with patch("smlx.models.smlx_manager.infer_model_type", return_value="smollm2-135m"):
        with patch("smlx.models.smlx_manager.get_model_loader", return_value=mock_loader):
            manager.load_model("test-model")

            stats = manager.get_stats()

            assert stats["total_models"] == 1
            assert stats["total_loads"] == 1


@pytest.mark.unit
def test_manager_clear_cache():
    """Test clearing cache."""
    manager = ModelLifecycleManager()

    mock_loader = MagicMock()
    mock_loader.load.return_value = (MagicMock(), MagicMock())

    with patch("smlx.models.smlx_manager.infer_model_type", return_value="smollm2-135m"):
        with patch("smlx.models.smlx_manager.get_model_loader", return_value=mock_loader):
            manager.load_model("model-1")
            manager.load_model("model-2")

            assert len(manager.get_cached_models()) == 2

            manager.clear_cache()

            assert len(manager.get_cached_models()) == 0


@pytest.mark.unit
def test_manager_get_memory_info():
    """Test getting memory information."""
    manager = ModelLifecycleManager()

    info = manager.get_memory_info()

    assert "total_gb" in info
    assert "available_gb" in info
    assert "used_gb" in info
    assert "percent" in info
    assert "cached_models" in info
    assert info["cached_models"] == 0


# ============================================================================
# Singleton Tests
# ============================================================================


@pytest.mark.unit
def test_get_manager_singleton():
    """Test that get_manager returns singleton instance."""
    manager1 = get_manager(force_new=True)
    manager2 = get_manager()

    assert manager1 is manager2
    assert isinstance(manager1, ModelLifecycleManager)


@pytest.mark.unit
def test_get_manager_multiple_calls():
    """Test multiple calls to get_manager return same instance."""
    # Force new for clean test
    managers = [get_manager(force_new=True)]
    managers.extend([get_manager() for _ in range(4)])

    # All should be the same instance
    for manager in managers[1:]:
        assert manager is managers[0]


@pytest.mark.unit
def test_get_manager_force_new():
    """Test forcing new manager instance."""
    manager1 = get_manager(force_new=True)
    manager2 = get_manager(force_new=True)

    # Should be different instances when forced
    assert manager1 is not manager2


@pytest.mark.unit
def test_get_manager_with_config():
    """Test get_manager with custom config."""
    cache_config = CacheConfig(max_models=10)
    telemetry_config = TelemetryConfig(enable_telemetry=False)

    manager = get_manager(
        cache_config=cache_config,
        telemetry_config=telemetry_config,
        force_new=True,
    )

    assert manager.cache.config.max_models == 10
    assert manager.telemetry.config.enable_telemetry is False
