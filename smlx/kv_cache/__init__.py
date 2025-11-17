# Copyright � 2025 SMLX Project

"""
KV Cache module for efficient transformer inference.

Provides comprehensive KV cache management with support for:
- Standard and rotating caches
- Quantization (4-bit, 8-bit)
- RoPE and ALiBi positional encoding
- Memory pressure monitoring and automatic intervention
- Debugging and profiling tools

Quick Start:
    >>> from smlx.kv_cache import KVCacheManager
    >>>
    >>> # Auto-configure cache based on available memory
    >>> manager = KVCacheManager.create_auto(
    ...     num_layers=24,
    ...     model_size_gb=0.5,
    ...     target_memory_gb=32.0
    ... )
    >>>
    >>> # Use in generation
    >>> for layer_idx, cache in enumerate(manager):
    ...     # ... use cache in layer ...
    ...     pass

Examples:
    Standard cache:
        >>> from smlx.kv_cache import KVCacheManager
        >>> manager = KVCacheManager.create_standard(num_layers=24)

    Rotating cache:
        >>> manager = KVCacheManager.create_rotating(
        ...     num_layers=24,
        ...     max_kv_size=2048,
        ...     keep=256
        ... )

    Quantized cache:
        >>> manager = KVCacheManager.create_quantized(
        ...     num_layers=24,
        ...     bits=4,
        ...     max_size=4096
        ... )

    RoPE-aware cache:
        >>> from smlx.kv_cache import initialize_rope_cache
        >>> caches = initialize_rope_cache(
        ...     dims=64,
        ...     num_layers=24,
        ...     max_kv_size=2048
        ... )

    ALiBi cache:
        >>> from smlx.kv_cache import initialize_alibi_cache
        >>> caches = initialize_alibi_cache(
        ...     num_heads=12,
        ...     num_layers=24
        ... )

    Memory pressure monitoring:
        >>> from smlx.kv_cache import MemoryPressureGauge, PressureBreaker
        >>>
        >>> gauge = MemoryPressureGauge()
        >>> breaker = PressureBreaker(manager, gauge)
        >>>
        >>> # During generation
        >>> for step in range(max_steps):
        ...     breaker.monitor_and_intervene()
        ...     # ... generation code ...

    Cache tracing:
        >>> from smlx.kv_cache import CacheTracer
        >>>
        >>> tracer = CacheTracer(enabled=True)
        >>> # ... run generation ...
        >>> summary = tracer.get_summary()
        >>> tracer.export_json("trace.json")
"""

# Core cache implementations
from smlx.kv_cache.mlx_kv import (
    MLXKVCache,
    MLXRotatingKVCache,
    QuantizedMLXKVCache,
)

# Cache management
from smlx.kv_cache.kv_manager import CacheType, KVCacheManager

# Positional encoding support
from smlx.kv_cache.alibi import ALiBiCache, initialize_alibi_cache
from smlx.kv_cache.rope import RoPECache, create_rope_module, initialize_rope_cache

# Cache limits and memory management
from smlx.kv_cache.cache_limits import CacheLimitManager
from smlx.kv_cache.memory_pressure_gauge import MemoryPressureGauge
from smlx.kv_cache.pressure_breaker import PressureBreaker

# Debugging and profiling
from smlx.kv_cache.cache_trace import CacheTracer, trace_cache_manager

__version__ = "0.1.0"

__all__ = [
    # Core caches
    "MLXKVCache",
    "MLXRotatingKVCache",
    "QuantizedMLXKVCache",
    # Cache management
    "KVCacheManager",
    "CacheType",
    # RoPE
    "RoPECache",
    "initialize_rope_cache",
    "create_rope_module",
    # ALiBi
    "ALiBiCache",
    "initialize_alibi_cache",
    # Memory management
    "CacheLimitManager",
    "MemoryPressureGauge",
    "PressureBreaker",
    # Debugging
    "CacheTracer",
    "trace_cache_manager",
]
