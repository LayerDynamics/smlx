# Copyright © 2025 SMLX Project

"""
Key-Value cache implementations for SmolLM2-135M.

This module provides enhanced cache management using the new smlx.kv_cache module
with support for quantization, memory monitoring, and automatic configuration.

Backwards compatible with the old cache API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

# New kv_cache module (recommended)
from smlx.kv_cache import (
    CacheType,
    KVCacheManager,
    MemoryPressureGauge,
    PressureBreaker,
)

# Legacy imports for backwards compatibility
from smlx.utils.cache import (
    KVCache,
    RotatingKVCache,
)
from smlx.utils.cache import (
    make_cache as utils_make_cache,
)

if TYPE_CHECKING:
    from smlx.models.SmolLM2_135M.model import Model


def make_cache(
    model: Model,
    max_kv_size: int | None = None,
    cache_type: CacheType | Literal["auto"] = "auto",
    enable_quantization: bool = False,
    quantization_bits: int = 4,
    enable_monitoring: bool = False,
    target_memory_gb: float = 32.0,
):
    """
    Create KV cache for SmolLM2-135M model with advanced options.

    Args:
        model: The model to create cache for
        max_kv_size: Optional maximum cache size (uses RotatingKVCache if provided)
        cache_type: Type of cache ("auto", "standard", "rotating", "quantized")
        enable_quantization: Enable cache quantization (4-bit or 8-bit)
        quantization_bits: Bits for quantization (4 or 8, default: 4)
        enable_monitoring: Enable memory pressure monitoring
        target_memory_gb: Target memory usage in GB (default: 32.0)

    Returns:
        List of cache objects or KVCacheManager instance

    Examples:
        >>> # Simple usage (backwards compatible)
        >>> cache = make_cache(model)
        >>>
        >>> # With size limit (backwards compatible)
        >>> cache = make_cache(model, max_kv_size=2048)
        >>>
        >>> # Auto-configure based on available memory
        >>> cache = make_cache(model, cache_type="auto")
        >>>
        >>> # With 4-bit quantization for memory efficiency
        >>> cache = make_cache(model, enable_quantization=True, quantization_bits=4)
        >>>
        >>> # With memory monitoring for long generations
        >>> cache = make_cache(model, enable_monitoring=True)
    """
    num_layers = len(model.layers)

    # Legacy mode: if just max_kv_size is specified and cache_type is auto
    if cache_type == "auto" and not enable_quantization and not enable_monitoring:
        # Use legacy implementation for backwards compatibility
        return utils_make_cache(num_layers, max_kv_size=max_kv_size, keep=4)

    # New enhanced mode
    if cache_type == "auto":
        # Auto-select based on memory
        # Estimate model size (SmolLM2-135M is ~0.5GB)
        model_size_gb = 0.5

        manager = KVCacheManager.create_auto(
            num_layers=num_layers,
            model_size_gb=model_size_gb,
            target_memory_gb=target_memory_gb,
            num_heads=model.args.num_attention_heads,
            head_dim=model.args.head_dim or (
                model.args.hidden_size // model.args.num_attention_heads
            ),
        )
    elif cache_type == "quantized" or enable_quantization:
        # Quantized cache
        manager = KVCacheManager.create_quantized(
            num_layers=num_layers,
            bits=quantization_bits,
            max_size=max_kv_size,
            keep=4,
        )
    elif cache_type == "rotating" or max_kv_size is not None:
        # Rotating cache with max size
        manager = KVCacheManager.create_rotating(
            num_layers=num_layers,
            max_kv_size=max_kv_size or 2048,
            keep=4,
        )
    else:
        # Standard growing cache
        manager = KVCacheManager.create_standard(num_layers=num_layers)

    # Return just the cache list for backwards compatibility
    # Users can access the manager features if needed via the caches attribute
    return list(manager)


def make_cache_with_monitoring(
    model: Model,
    max_kv_size: int | None = None,
    enable_quantization: bool = False,
    target_memory_gb: float = 32.0,
):
    """
    Create KV cache with automatic memory pressure monitoring.

    This function creates a cache manager with a PressureBreaker that
    automatically intervenes to prevent OOM errors during generation.

    Args:
        model: The model to create cache for
        max_kv_size: Optional maximum cache size
        enable_quantization: Enable 4-bit quantization
        target_memory_gb: Target memory usage in GB (default: 32.0)

    Returns:
        Tuple of (cache_list, pressure_breaker)
        - cache_list: List of caches to use with model
        - pressure_breaker: PressureBreaker instance (call monitor_and_intervene()
          before each generation step)

    Example:
        >>> cache, breaker = make_cache_with_monitoring(model)
        >>>
        >>> # During generation
        >>> for step in range(max_tokens):
        ...     breaker.monitor_and_intervene()  # Auto cleanup if needed
        ...     # ... generation code ...
    """
    num_layers = len(model.layers)
    model_size_gb = 0.5  # SmolLM2-135M

    # Create manager with monitoring enabled
    if enable_quantization:
        manager = KVCacheManager.create_quantized(
            num_layers=num_layers,
            bits=4,
            max_size=max_kv_size,
            keep=4,
            enable_memory_monitoring=True,
        )
    elif max_kv_size is not None:
        manager = KVCacheManager.create_rotating(
            num_layers=num_layers,
            max_kv_size=max_kv_size,
            keep=4,
            enable_memory_monitoring=True,
        )
    else:
        manager = KVCacheManager.create_auto(
            num_layers=num_layers,
            model_size_gb=model_size_gb,
            target_memory_gb=target_memory_gb,
            num_heads=model.args.num_attention_heads,
            head_dim=model.args.head_dim or (
                model.args.hidden_size // model.args.num_attention_heads
            ),
            enable_memory_monitoring=True,
        )

    # Create pressure breaker
    gauge = MemoryPressureGauge()
    breaker = PressureBreaker(manager, gauge)

    return list(manager), breaker


# Export cache classes for backwards compatibility
__all__ = [
    "KVCache",
    "RotatingKVCache",
    "make_cache",
    "make_cache_with_monitoring",
]
