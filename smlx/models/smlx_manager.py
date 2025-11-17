#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model Lifecycle Manager for SMLX.

This module provides model loading, caching, telemetry, and lifecycle management
for efficient model execution.

Architecture:
    ModelCache - LRU cache with memory awareness
    ModelTelemetry - Usage and performance tracking
    ModelLifecycleManager - Unified lifecycle management

Example:
    >>> from smlx.models.smlx_manager import get_manager
    >>>
    >>> # Get manager instance
    >>> manager = get_manager()
    >>>
    >>> # Load model (cached automatically)
    >>> model, tokenizer = manager.load_model(
    ...     model_id="mlx-community/SmolLM2-135M-Instruct",
    ...     quantization="4bit"
    ... )
    >>>
    >>> # Get telemetry
    >>> stats = manager.get_stats()
    >>> print(f"Total loads: {stats['total_loads']}")
"""

from __future__ import annotations

import gc
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import mlx.core as mx
import psutil

from .registry import get_model_loader, infer_model_type

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class CacheConfig:
    """
    Configuration for model cache.

    Attributes:
        max_models: Maximum number of models to cache (default: 3)
        max_memory_gb: Maximum memory usage in GB (default: 24)
        min_free_memory_gb: Minimum free memory to maintain (default: 4)
        enable_eviction: Enable automatic eviction on memory pressure
        eviction_threshold: Memory threshold to trigger eviction (0.8 = 80%)
    """

    max_models: int = 3
    max_memory_gb: float = 24.0
    min_free_memory_gb: float = 4.0
    enable_eviction: bool = True
    eviction_threshold: float = 0.8

    def validate(self):
        """Validate cache configuration."""
        if self.max_models < 1:
            raise ValueError(f"max_models must be >= 1, got {self.max_models}")

        if self.max_memory_gb <= 0:
            raise ValueError(f"max_memory_gb must be > 0, got {self.max_memory_gb}")

        if self.min_free_memory_gb < 0:
            raise ValueError(
                f"min_free_memory_gb must be >= 0, got {self.min_free_memory_gb}"
            )

        if not 0 < self.eviction_threshold <= 1:
            raise ValueError(
                f"eviction_threshold must be in (0, 1], got {self.eviction_threshold}"
            )


@dataclass
class TelemetryConfig:
    """
    Configuration for telemetry.

    Attributes:
        enable_telemetry: Enable telemetry tracking
        track_latency: Track inference latency
        track_memory: Track memory usage
        track_errors: Track error counts
        retention_hours: How long to keep telemetry data (hours)
    """

    enable_telemetry: bool = True
    track_latency: bool = True
    track_memory: bool = True
    track_errors: bool = True
    retention_hours: int = 24


# ============================================================================
# Telemetry
# ============================================================================


@dataclass
class ModelStats:
    """Statistics for a single model."""

    model_id: str
    load_count: int = 0
    inference_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    memory_mb: float = 0.0
    last_used: float = field(default_factory=time.time)
    first_loaded: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        """Average inference latency in milliseconds."""
        if self.inference_count == 0:
            return 0.0
        return self.total_latency_ms / self.inference_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "load_count": self.load_count,
            "inference_count": self.inference_count,
            "error_count": self.error_count,
            "avg_latency_ms": self.avg_latency_ms,
            "memory_mb": self.memory_mb,
            "last_used": self.last_used,
            "first_loaded": self.first_loaded,
        }


class ModelTelemetry:
    """
    Track model usage and performance metrics.

    Thread-safe telemetry tracking for models.
    """

    def __init__(self, config: TelemetryConfig | None = None):
        """Initialize telemetry."""
        self.config = config or TelemetryConfig()
        self._stats: dict[str, ModelStats] = {}
        self._lock = Lock()

    def record_load(self, model_id: str, memory_mb: float = 0.0):
        """Record a model load event."""
        if not self.config.enable_telemetry:
            return

        with self._lock:
            if model_id not in self._stats:
                self._stats[model_id] = ModelStats(model_id=model_id)

            stats = self._stats[model_id]
            stats.load_count += 1
            stats.memory_mb = memory_mb
            stats.last_used = time.time()

    def record_inference(self, model_id: str, latency_ms: float = 0.0):
        """Record an inference event."""
        if not self.config.enable_telemetry:
            return

        with self._lock:
            if model_id not in self._stats:
                self._stats[model_id] = ModelStats(model_id=model_id)

            stats = self._stats[model_id]
            stats.inference_count += 1
            if self.config.track_latency:
                stats.total_latency_ms += latency_ms
            stats.last_used = time.time()

    def record_error(self, model_id: str):
        """Record an error event."""
        if not self.config.enable_telemetry or not self.config.track_errors:
            return

        with self._lock:
            if model_id not in self._stats:
                self._stats[model_id] = ModelStats(model_id=model_id)

            stats = self._stats[model_id]
            stats.error_count += 1
            stats.last_used = time.time()

    def get_stats(self, model_id: str | None = None) -> dict[str, Any] | ModelStats:
        """
        Get telemetry statistics.

        Args:
            model_id: Optional model ID. If None, return all stats.

        Returns:
            ModelStats for specific model or dict of all stats
        """
        with self._lock:
            if model_id is not None:
                return self._stats.get(model_id, ModelStats(model_id=model_id))

            return {
                "total_models": len(self._stats),
                "total_loads": sum(s.load_count for s in self._stats.values()),
                "total_inferences": sum(s.inference_count for s in self._stats.values()),
                "total_errors": sum(s.error_count for s in self._stats.values()),
                "models": {mid: stats.to_dict() for mid, stats in self._stats.items()},
            }

    def reset_stats(self, model_id: str | None = None):
        """
        Reset statistics.

        Args:
            model_id: Optional model ID. If None, reset all stats.
        """
        with self._lock:
            if model_id is not None:
                if model_id in self._stats:
                    del self._stats[model_id]
            else:
                self._stats.clear()


# ============================================================================
# Model Cache
# ============================================================================


class ModelCache:
    """
    LRU cache for loaded models with memory awareness.

    Thread-safe caching with automatic eviction based on:
    - Maximum number of models
    - Memory pressure
    - LRU eviction policy
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize cache."""
        self.config = config or CacheConfig()
        self.config.validate()

        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._lock = Lock()

        # Get system memory info
        self._total_memory_gb = psutil.virtual_memory().total / (1024**3)

        logger.info(
            f"Initialized ModelCache: max_models={self.config.max_models}, "
            f"max_memory={self.config.max_memory_gb}GB, "
            f"system_memory={self._total_memory_gb:.1f}GB"
        )

    def get(self, model_id: str) -> tuple[Any, Any] | None:
        """
        Get cached model and tokenizer.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (model, tokenizer) or None if not cached
        """
        with self._lock:
            if model_id in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(model_id)
                logger.debug(f"Cache hit: {model_id}")
                return self._cache[model_id]

            logger.debug(f"Cache miss: {model_id}")
            return None

    def put(self, model_id: str, model: Any, tokenizer: Any):
        """
        Add model to cache.

        Args:
            model_id: Model identifier
            model: Model instance
            tokenizer: Tokenizer instance
        """
        with self._lock:
            # Check if we need to evict
            while len(self._cache) >= self.config.max_models:
                evicted_id, _ = self._cache.popitem(last=False)
                logger.info(f"Evicted model from cache (LRU): {evicted_id}")
                self._cleanup_model()

            # Check memory pressure
            if self.config.enable_eviction:
                self._check_memory_pressure()

            # Add to cache
            self._cache[model_id] = (model, tokenizer)
            logger.info(f"Cached model: {model_id} (cache size: {len(self._cache)})")

    def remove(self, model_id: str) -> bool:
        """
        Remove model from cache.

        Args:
            model_id: Model identifier

        Returns:
            True if model was cached and removed, False otherwise
        """
        with self._lock:
            if model_id in self._cache:
                del self._cache[model_id]
                self._cleanup_model()
                logger.info(f"Removed model from cache: {model_id}")
                return True

            return False

    def clear(self):
        """Clear all cached models."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._cleanup_model()
            logger.info(f"Cleared cache: removed {count} models")

    def get_cached_models(self) -> list[str]:
        """
        Get list of cached model IDs.

        Returns:
            List of model IDs in cache (most recent first)
        """
        with self._lock:
            return list(reversed(self._cache.keys()))

    def _check_memory_pressure(self):
        """Check memory pressure and evict if needed."""
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        available_gb = mem.available / (1024**3)

        # Check if we're using too much memory
        if used_gb > self.config.max_memory_gb * self.config.eviction_threshold:
            logger.warning(
                f"Memory pressure detected: {used_gb:.1f}GB used "
                f"(threshold: {self.config.max_memory_gb * self.config.eviction_threshold:.1f}GB)"
            )
            self._evict_oldest()

        # Check if we have too little free memory
        if available_gb < self.config.min_free_memory_gb:
            logger.warning(
                f"Low free memory: {available_gb:.1f}GB "
                f"(minimum: {self.config.min_free_memory_gb}GB)"
            )
            self._evict_oldest()

    def _evict_oldest(self):
        """Evict oldest (least recently used) model."""
        if len(self._cache) > 0:
            evicted_id, _ = self._cache.popitem(last=False)
            logger.info(f"Evicted model due to memory pressure: {evicted_id}")
            self._cleanup_model()

    def _cleanup_model(self):
        """Force garbage collection and MLX cleanup."""
        gc.collect()
        # MLX doesn't have explicit cleanup, but evaluating empty array can help
        mx.eval(mx.array([]))


# ============================================================================
# Model Lifecycle Manager
# ============================================================================


class ModelLifecycleManager:
    """
    Unified model lifecycle management.

    Provides:
    - Lazy model loading with caching
    - Telemetry and monitoring
    - Memory pressure awareness
    - Quantization support
    - Thread-safe operations

    Example:
        >>> manager = ModelLifecycleManager()
        >>> model, tokenizer = manager.load_model("mlx-community/SmolLM2-135M-Instruct")
        >>> stats = manager.get_stats()
    """

    def __init__(
        self,
        cache_config: CacheConfig | None = None,
        telemetry_config: TelemetryConfig | None = None,
    ):
        """
        Initialize lifecycle manager.

        Args:
            cache_config: Optional cache configuration
            telemetry_config: Optional telemetry configuration
        """
        self.cache = ModelCache(cache_config)
        self.telemetry = ModelTelemetry(telemetry_config)
        self._lock = Lock()

        logger.info("Initialized ModelLifecycleManager")

    def load_model(
        self,
        model_id: str,
        quantization: str | None = None,
        force_reload: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """
        Load model with caching and telemetry.

        Args:
            model_id: Model identifier (HuggingFace ID or local path)
            quantization: Optional quantization ("4bit", "8bit")
            force_reload: Force reload even if cached
            **kwargs: Additional arguments passed to model loader

        Returns:
            Tuple of (model, tokenizer)

        Example:
            >>> model, tokenizer = manager.load_model(
            ...     "mlx-community/SmolLM2-135M-Instruct",
            ...     quantization="4bit"
            ... )
        """
        # Create cache key including quantization
        cache_key = f"{model_id}:{quantization or 'none'}"

        # Check cache first
        if not force_reload:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Using cached model: {cache_key}")
                return cached

        # Load model
        logger.info(f"Loading model: {model_id} (quantization: {quantization})")
        start_time = time.time()

        try:
            # Get model type and loader
            model_type = infer_model_type(model_id)
            if model_type is None:
                raise ValueError(f"Could not infer model type from: {model_id}")

            loader_module = get_model_loader(model_type)

            # Load model
            if quantization:
                # Check if loader supports quantization parameter
                model, tokenizer = loader_module.load(
                    model_id, quantization=quantization, **kwargs
                )
            else:
                model, tokenizer = loader_module.load(model_id, **kwargs)

            # Calculate memory usage (approximate)
            memory_mb = self._estimate_memory_usage()

            # Cache model
            self.cache.put(cache_key, model, tokenizer)

            # Record telemetry
            load_time_ms = (time.time() - start_time) * 1000
            self.telemetry.record_load(cache_key, memory_mb=memory_mb)

            logger.info(
                f"Loaded model: {model_id} in {load_time_ms:.1f}ms "
                f"(~{memory_mb:.1f}MB)"
            )

            return model, tokenizer

        except Exception as e:
            self.telemetry.record_error(cache_key)
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def unload_model(self, model_id: str, quantization: str | None = None) -> bool:
        """
        Unload model from cache.

        Args:
            model_id: Model identifier
            quantization: Optional quantization used when loading

        Returns:
            True if model was unloaded, False if not in cache
        """
        cache_key = f"{model_id}:{quantization or 'none'}"
        removed = self.cache.remove(cache_key)

        if removed:
            logger.info(f"Unloaded model: {cache_key}")

        return removed

    def get_cached_models(self) -> list[str]:
        """
        Get list of currently cached models.

        Returns:
            List of cached model identifiers
        """
        return self.cache.get_cached_models()

    def get_stats(self, model_id: str | None = None) -> dict[str, Any]:
        """
        Get telemetry statistics.

        Args:
            model_id: Optional model ID for specific stats

        Returns:
            Dictionary of statistics
        """
        if model_id is not None:
            stats = self.telemetry.get_stats(model_id)
            if isinstance(stats, ModelStats):
                return stats.to_dict()
            return stats

        return self.telemetry.get_stats()

    def clear_cache(self):
        """Clear all cached models."""
        self.cache.clear()
        logger.info("Cleared all cached models")

    def get_memory_info(self) -> dict[str, float]:
        """
        Get current memory information.

        Returns:
            Dictionary with memory metrics in GB
        """
        mem = psutil.virtual_memory()

        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
            "cached_models": len(self.cache.get_cached_models()),
        }

    def _estimate_memory_usage(self) -> float:
        """
        Estimate current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        mem = psutil.virtual_memory()
        return mem.used / (1024**2)


# ============================================================================
# Global Manager Instance
# ============================================================================

_global_manager: ModelLifecycleManager | None = None
_manager_lock = Lock()


def get_manager(
    cache_config: CacheConfig | None = None,
    telemetry_config: TelemetryConfig | None = None,
    force_new: bool = False,
) -> ModelLifecycleManager:
    """
    Get global ModelLifecycleManager instance (singleton pattern).

    Args:
        cache_config: Optional cache configuration (only used on first call)
        telemetry_config: Optional telemetry configuration (only used on first call)
        force_new: Force creation of new instance (useful for testing)

    Returns:
        Global ModelLifecycleManager instance

    Example:
        >>> manager = get_manager()
        >>> model, tokenizer = manager.load_model("mlx-community/SmolLM2-135M-Instruct")
    """
    global _global_manager

    with _manager_lock:
        if _global_manager is None or force_new:
            _global_manager = ModelLifecycleManager(
                cache_config=cache_config,
                telemetry_config=telemetry_config,
            )

        return _global_manager


__all__ = [
    "CacheConfig",
    "TelemetryConfig",
    "ModelStats",
    "ModelTelemetry",
    "ModelCache",
    "ModelLifecycleManager",
    "get_manager",
]
