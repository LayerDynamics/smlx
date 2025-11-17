# Copyright © 2025 SMLX Project

"""
KV cache implementations for SmolVLM-256M.

This module provides enhanced KV cache with memory monitoring, quantization,
and automatic pressure relief for the SmolLM2-135M language model used in SmolVLM-256M.

Quick Start:
    >>> from smlx.models.SmolVLM_256M import load
    >>> from smlx.models.SmolVLM_256M.cache import make_cache
    >>>
    >>> model, processor = load()
    >>>
    >>> # Standard cache (enhanced API)
    >>> cache = make_cache(model.language_model)
    >>>
    >>> # Rotating cache with automatic sizing
    >>> cache = make_cache(
    ...     model.language_model,
    ...     cache_type="auto",
    ...     enable_monitoring=True
    ... )
    >>>
    >>> # With OOM prevention
    >>> cache, breaker = make_cache_with_monitoring(model.language_model)
    >>> # ... during generation ...
    >>> breaker.monitor_and_intervene()

Examples:
    Auto-configure cache based on available memory:
        >>> cache = make_cache(
        ...     language_model,
        ...     cache_type="auto",
        ...     target_memory_gb=32.0
        ... )

    Enable quantization for memory efficiency:
        >>> cache = make_cache(
        ...     language_model,
        ...     cache_type="quantized",
        ...     enable_quantization=True,
        ...     quantization_bits=4
        ... )

    Full monitoring with automatic intervention:
        >>> cache, breaker = make_cache_with_monitoring(
        ...     language_model,
        ...     target_memory_gb=32.0
        ... )
        >>> for step in range(max_steps):
        ...     breaker.monitor_and_intervene()
        ...     # ... generation code ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

# New kv_cache module imports
from smlx.kv_cache import (
    CacheType,
    KVCacheManager,
    MemoryPressureGauge,
    PressureBreaker,
)

# Legacy imports for backward compatibility
from smlx.utils.cache import KVCache, RotatingKVCache
from smlx.utils.cache import make_cache as utils_make_cache

if TYPE_CHECKING:
    from .language import LanguageModel

# Re-export cache classes for backward compatibility
__all__ = [
    "KVCache",
    "RotatingKVCache",
    "make_cache",
    "make_kv_cache",  # Alias for backward compatibility
    "make_cache_with_monitoring",
]


def make_cache(
    language_model: LanguageModel,
    max_kv_size: int | None = None,
    cache_type: CacheType | Literal["auto"] = "auto",
    enable_quantization: bool = False,
    quantization_bits: int = 4,
    enable_monitoring: bool = False,
    target_memory_gb: float = 32.0,
) -> list:
    """
    Create KV cache for SmolVLM-256M's SmolLM2-135M language model with enhanced features.

    This function provides backward compatibility while enabling new features
    like automatic cache sizing, quantization, and memory monitoring.

    Note: This is a major improvement over the previous implementation which
    returned [None] * num_layers (no actual caching).

    Args:
        language_model: SmolLM2-135M language model
        max_kv_size: Optional maximum cache size for rotating cache.
                     When None with cache_type="auto", automatically computed.
        cache_type: Cache type to use:
                   - "auto": Automatically select based on memory (default)
                   - "standard": Growing cache (no size limit)
                   - "rotating": Fixed-size rotating cache
                   - "quantized": 4-bit or 8-bit quantized cache
        enable_quantization: Enable cache quantization (default: False)
        quantization_bits: Quantization bits (4 or 8, default: 4)
        enable_monitoring: Enable memory pressure monitoring (default: False)
        target_memory_gb: Target total memory usage in GB (default: 32.0)

    Returns:
        List of cache instances (KVCache, RotatingKVCache, or QuantizedKVCache)

    Examples:
        Enhanced API with proper caching:
            >>> cache = make_cache(language_model)

        Auto-configure based on memory:
            >>> cache = make_cache(language_model, cache_type="auto", enable_monitoring=True)

        Explicit rotating cache:
            >>> cache = make_cache(language_model, cache_type="rotating", max_kv_size=2048)

        Quantized cache:
            >>> cache = make_cache(
            ...     language_model,
            ...     cache_type="quantized",
            ...     enable_quantization=True,
            ...     quantization_bits=4
            ... )
    """
    num_layers = language_model.num_hidden_layers

    # Legacy mode check - but we'll actually provide real caches instead of None!
    # This is a major improvement over the old behavior
    if (
        cache_type == "auto"
        and not enable_quantization
        and not enable_monitoring
        and max_kv_size is None
    ):
        # Return actual KVCache instances instead of [None] * num_layers
        # SmolLM2-135M uses NoPE with keep=4 pattern
        return utils_make_cache(num_layers, max_kv_size=max_kv_size, keep=4)

    # New mode: use enhanced KVCacheManager
    # SmolLM2-135M model is ~0.5GB (135M params * 2 bytes/param for fp16 ≈ 270MB, round to 0.5GB)
    model_size_gb = 0.5

    if cache_type == "auto":
        # Automatically select cache type based on available memory
        manager = KVCacheManager.create_auto(
            num_layers=num_layers,
            model_size_gb=model_size_gb,
            target_memory_gb=target_memory_gb,
            enable_monitoring=enable_monitoring,
        )
    elif cache_type == "standard":
        manager = KVCacheManager.create_standard(
            num_layers=num_layers,
            enable_monitoring=enable_monitoring,
        )
    elif cache_type == "rotating":
        if max_kv_size is None:
            # Compute safe max_kv_size based on available memory
            from smlx.kv_cache import CacheLimitManager

            limit_mgr = CacheLimitManager(
                model_size_gb=model_size_gb,
                target_memory_gb=target_memory_gb,
            )
            # SmolLM2-135M config: 30 layers, 9 heads, head_dim=64, 3 kv_heads (GQA)
            # head_dim = 576 / 9 = 64
            max_kv_size = limit_mgr.compute_max_kv_size(
                num_layers=30,
                head_dim=64,
                num_heads=9,
                num_kv_heads=3,
            )

        manager = KVCacheManager.create_rotating(
            num_layers=num_layers,
            max_kv_size=max_kv_size,
            keep=4,  # SmolLM2 uses NoPE every 4th layer
            enable_monitoring=enable_monitoring,
        )
    elif cache_type == "quantized":
        if max_kv_size is None:
            from smlx.kv_cache import CacheLimitManager

            limit_mgr = CacheLimitManager(
                model_size_gb=model_size_gb,
                target_memory_gb=target_memory_gb,
            )
            # Compute max with quantization compression
            max_kv_size_fp16 = limit_mgr.compute_max_kv_size(
                num_layers=30,
                head_dim=64,
                num_heads=9,
                num_kv_heads=3,
            )
            # Quantization provides ~4x compression for 4-bit
            compression = 16 / quantization_bits
            max_kv_size = int(max_kv_size_fp16 * compression)

        manager = KVCacheManager.create_quantized(
            num_layers=num_layers,
            bits=quantization_bits,
            max_size=max_kv_size,
            enable_monitoring=enable_monitoring,
        )
    else:
        raise ValueError(f"Unknown cache_type: {cache_type}")

    # Return as list for API compatibility
    return list(manager)


def make_cache_with_monitoring(
    language_model: LanguageModel,
    max_kv_size: int | None = None,
    cache_type: CacheType | Literal["auto"] = "auto",
    enable_quantization: bool = False,
    quantization_bits: int = 4,
    target_memory_gb: float = 32.0,
    warning_threshold: float = 0.8,
    critical_threshold: float = 0.9,
) -> tuple[list, PressureBreaker]:
    """
    Create KV cache with automatic memory pressure monitoring and intervention.

    This is a convenience function that creates a cache with PressureBreaker
    for automatic OOM prevention during generation.

    Args:
        language_model: SmolLM2-135M language model
        max_kv_size: Optional maximum cache size
        cache_type: Cache type ("auto", "standard", "rotating", "quantized")
        enable_quantization: Enable cache quantization
        quantization_bits: Quantization bits (4 or 8)
        target_memory_gb: Target total memory usage in GB
        warning_threshold: Memory threshold for warnings (0.0-1.0, default: 0.8)
        critical_threshold: Memory threshold for critical alerts (0.0-1.0, default: 0.9)

    Returns:
        Tuple of (cache_list, pressure_breaker)

    Example:
        >>> cache, breaker = make_cache_with_monitoring(
        ...     language_model,
        ...     target_memory_gb=32.0
        ... )
        >>>
        >>> # During generation loop
        >>> for step in range(max_steps):
        ...     intervention = breaker.monitor_and_intervene(current_step=step)
        ...     if intervention:
        ...         print(f"Intervention at step {step}: {intervention['action']}")
        ...     # ... generation code ...
        >>>
        >>> # Get statistics
        >>> stats = breaker.get_statistics()
        >>> print(f"Total interventions: {stats['total_interventions']}")
    """
    num_layers = language_model.num_hidden_layers
    model_size_gb = 0.5  # SmolLM2-135M is ~0.5GB

    # Create cache manager with monitoring enabled
    if cache_type == "auto":
        manager = KVCacheManager.create_auto(
            num_layers=num_layers,
            model_size_gb=model_size_gb,
            target_memory_gb=target_memory_gb,
            enable_monitoring=True,
        )
    elif cache_type == "standard":
        manager = KVCacheManager.create_standard(
            num_layers=num_layers,
            enable_monitoring=True,
        )
    elif cache_type == "rotating":
        if max_kv_size is None:
            from smlx.kv_cache import CacheLimitManager

            limit_mgr = CacheLimitManager(
                model_size_gb=model_size_gb,
                target_memory_gb=target_memory_gb,
            )
            max_kv_size = limit_mgr.compute_max_kv_size(
                num_layers=30,
                head_dim=64,
                num_heads=9,
                num_kv_heads=3,
            )

        manager = KVCacheManager.create_rotating(
            num_layers=num_layers,
            max_kv_size=max_kv_size,
            keep=4,
            enable_monitoring=True,
        )
    elif cache_type == "quantized":
        if max_kv_size is None:
            from smlx.kv_cache import CacheLimitManager

            limit_mgr = CacheLimitManager(
                model_size_gb=model_size_gb,
                target_memory_gb=target_memory_gb,
            )
            max_kv_size_fp16 = limit_mgr.compute_max_kv_size(
                num_layers=30,
                head_dim=64,
                num_heads=9,
                num_kv_heads=3,
            )
            compression = 16 / quantization_bits
            max_kv_size = int(max_kv_size_fp16 * compression)

        manager = KVCacheManager.create_quantized(
            num_layers=num_layers,
            bits=quantization_bits,
            max_size=max_kv_size,
            enable_monitoring=True,
        )
    else:
        raise ValueError(f"Unknown cache_type: {cache_type}")

    # Create pressure gauge and breaker
    gauge = MemoryPressureGauge(
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )
    breaker = PressureBreaker(manager, gauge, auto_enable=True)

    return list(manager), breaker


# Alias for backward compatibility with old generate.py function
make_kv_cache = make_cache
