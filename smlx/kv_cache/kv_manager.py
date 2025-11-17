# Copyright � 2025 SMLX Project

"""
KV Cache Manager for model-wide cache operations.

Provides high-level management of KV caches across all transformer layers,
with automatic cache type selection based on available memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from smlx.kv_cache.mlx_kv import (
    MLXKVCache,
    MLXRotatingKVCache,
    QuantizedMLXKVCache,
)
from smlx.utils.cache import reset_cache as reset_cache_list
from smlx.utils.memory import MemoryMonitor, get_device_info

CacheType = Literal["standard", "rotating", "quantized"]
CacheList = list[MLXKVCache | MLXRotatingKVCache | QuantizedMLXKVCache]


class KVCacheManager:
    """
    Manages KV caches for all layers of a transformer model.

    Provides factory methods for creating caches, automatic type selection based
    on memory availability, and utilities for saving/loading cache states.

    Attributes:
        num_layers: Number of transformer layers
        cache_type: Type of cache ('standard', 'rotating', or 'quantized')
        caches: List of cache objects, one per layer
        memory_monitor: Optional memory monitor for tracking usage

    Example:
        >>> # Auto-configure based on memory
        >>> manager = KVCacheManager.create_auto(
        ...     num_layers=24,
        ...     model_size_gb=0.5,
        ...     target_memory_gb=32.0
        ... )
        >>>
        >>> # Or explicit cache type
        >>> manager = KVCacheManager.create_rotating(
        ...     num_layers=24,
        ...     max_kv_size=2048,
        ...     keep=256
        ... )
    """

    def __init__(
        self,
        num_layers: int,
        cache_type: CacheType,
        caches: Sequence,
        enable_memory_monitoring: bool = True,
    ):
        """
        Initialize KV cache manager.

        Args:
            num_layers: Number of transformer layers
            cache_type: Type of cache being managed
            caches: List of cache objects
            enable_memory_monitoring: Enable memory monitoring (default: True)
        """
        self.num_layers = num_layers
        self.cache_type = cache_type
        self.caches: CacheList = list(caches)

        self.memory_monitor: MemoryMonitor | None = None
        if enable_memory_monitoring:
            self.memory_monitor = MemoryMonitor()

    @classmethod
    def create_standard(
        cls,
        num_layers: int,
        step: int | None = None,
        enable_tracing: bool = False,
        enable_memory_monitoring: bool = True,
        enable_monitoring: bool | None = None,  # Alias for enable_memory_monitoring
    ) -> KVCacheManager:
        """
        Create manager with standard (growing) KV caches.

        Args:
            num_layers: Number of transformer layers
            step: Allocation step size (default: 256)
            enable_tracing: Enable trace logging (default: False)
            enable_memory_monitoring: Enable memory monitoring (default: True)
            enable_monitoring: Alias for enable_memory_monitoring (for backwards compatibility)

        Returns:
            KVCacheManager with standard caches

        Example:
            >>> manager = KVCacheManager.create_standard(num_layers=24)
        """
        # Support both enable_monitoring and enable_memory_monitoring
        if enable_monitoring is not None:
            enable_memory_monitoring = enable_monitoring

        caches = [MLXKVCache(step=step, enable_tracing=enable_tracing) for _ in range(num_layers)]
        return cls(num_layers, "standard", caches, enable_memory_monitoring)

    @classmethod
    def create_rotating(
        cls,
        num_layers: int,
        max_kv_size: int,
        keep: int = 0,
        step: int | None = None,
        enable_tracing: bool = False,
        enable_memory_monitoring: bool = True,
    ) -> KVCacheManager:
        """
        Create manager with rotating (fixed-size) KV caches.

        Args:
            num_layers: Number of transformer layers
            max_kv_size: Maximum number of tokens to cache
            keep: Number of initial tokens to preserve (default: 0)
            step: Allocation step size (default: 256)
            enable_tracing: Enable trace logging (default: False)
            enable_memory_monitoring: Enable memory monitoring (default: True)

        Returns:
            KVCacheManager with rotating caches

        Example:
            >>> manager = KVCacheManager.create_rotating(
            ...     num_layers=24,
            ...     max_kv_size=2048,
            ...     keep=256
            ... )
        """
        caches = [
            MLXRotatingKVCache(
                max_size=max_kv_size, keep=keep, step=step, enable_tracing=enable_tracing
            )
            for _ in range(num_layers)
        ]
        return cls(num_layers, "rotating", caches, enable_memory_monitoring)

    @classmethod
    def create_quantized(
        cls,
        num_layers: int,
        bits: int = 4,
        group_size: int = 64,
        quantize_threshold: int = 256,
        max_size: int | None = None,
        keep: int = 0,
        step: int | None = None,
        enable_tracing: bool = False,
        enable_memory_monitoring: bool = True,
    ) -> KVCacheManager:
        """
        Create manager with quantized KV caches.

        Args:
            num_layers: Number of transformer layers
            bits: Quantization bits (4 or 8, default: 4)
            group_size: Group size for quantization (default: 64)
            quantize_threshold: Min tokens before quantizing (default: 256)
            max_size: Optional maximum cache size (enables rotation)
            keep: Number of initial tokens to preserve (default: 0)
            step: Allocation step size (default: 256)
            enable_tracing: Enable trace logging (default: False)
            enable_memory_monitoring: Enable memory monitoring (default: True)

        Returns:
            KVCacheManager with quantized caches

        Example:
            >>> manager = KVCacheManager.create_quantized(
            ...     num_layers=24,
            ...     bits=4,
            ...     max_size=4096
            ... )
        """
        caches = [
            QuantizedMLXKVCache(
                bits=bits,
                group_size=group_size,
                quantize_threshold=quantize_threshold,
                max_size=max_size,
                keep=keep,
                step=step,
                enable_tracing=enable_tracing,
            )
            for _ in range(num_layers)
        ]
        return cls(num_layers, "quantized", caches, enable_memory_monitoring)

    @classmethod
    def create_auto(
        cls,
        num_layers: int,
        model_size_gb: float,
        target_memory_gb: float = 32.0,
        num_heads: int = 12,
        head_dim: int = 64,
        enable_tracing: bool = False,
        enable_memory_monitoring: bool = True,
    ) -> KVCacheManager:
        """
        Auto-configure cache type based on available memory.

        Analyzes available memory and model requirements to select the most
        appropriate cache type (standard, rotating, or quantized).

        Args:
            num_layers: Number of transformer layers
            model_size_gb: Model size in GB (for memory budget calculation)
            target_memory_gb: Target total memory usage (default: 32.0)
            num_heads: Number of attention heads (for cache size calculation)
            head_dim: Head dimension (for cache size calculation)
            enable_tracing: Enable trace logging (default: False)
            enable_memory_monitoring: Enable memory monitoring (default: True)

        Returns:
            KVCacheManager with automatically selected cache type

        Example:
            >>> manager = KVCacheManager.create_auto(
            ...     num_layers=24,
            ...     model_size_gb=0.5,
            ...     target_memory_gb=32.0
            ... )
        """
        # Get device info
        device_info = get_device_info()
        max_device_gb = device_info["max_recommended_working_set_size_gb"]

        # Use target or device max, whichever is smaller
        effective_target_gb = min(target_memory_gb, max_device_gb)

        # Calculate available memory for KV cache
        # Reserve space for: model weights + activations (10% of model size)
        reserved_gb = model_size_gb * 1.1
        available_gb = effective_target_gb - reserved_gb

        # Calculate bytes per token for KV cache
        # K and V: num_layers * num_heads * head_dim * 2 (for K and V) * 2 (fp16)
        bytes_per_token = num_layers * num_heads * head_dim * 2 * 2

        # Calculate max tokens we can fit
        max_tokens = int((available_gb * 1e9) / bytes_per_token) if available_gb > 0 else 256

        # Decision logic - prioritize memory-efficient caches when total memory is tight
        # Check if target memory is close to model size (indicating tight constraints)
        memory_pressure_ratio = target_memory_gb / model_size_gb

        # If target memory is at most 4x model size, prioritize memory-efficient caches
        if memory_pressure_ratio <= 4.0:
            # Tight memory constraints - use memory-efficient cache
            if available_gb > 0 and max_tokens >= 2048:
                # Enough room for rotating cache with reasonable context
                return cls.create_rotating(
                    num_layers=num_layers,
                    max_kv_size=min(max_tokens, 4096),
                    keep=256,
                    enable_tracing=enable_tracing,
                    enable_memory_monitoring=enable_memory_monitoring,
                )
            else:
                # Very tight - use quantized cache
                # Quantization reduces memory by ~4x for 4-bit
                quantized_max_tokens = max(max_tokens * 4, 1024)  # At least 1K tokens
                return cls.create_quantized(
                    num_layers=num_layers,
                    bits=4,
                    max_size=min(quantized_max_tokens, 4096),
                    keep=256,
                    enable_tracing=enable_tracing,
                    enable_memory_monitoring=enable_memory_monitoring,
                )
        elif max_tokens >= 8192:
            # Plenty of memory - use standard cache
            return cls.create_standard(
                num_layers=num_layers,
                enable_tracing=enable_tracing,
                enable_memory_monitoring=enable_memory_monitoring,
            )
        elif max_tokens >= 2048:
            # Moderate memory - use rotating cache
            return cls.create_rotating(
                num_layers=num_layers,
                max_kv_size=min(max_tokens, 4096),
                keep=256,
                enable_tracing=enable_tracing,
                enable_memory_monitoring=enable_memory_monitoring,
            )
        else:
            # Limited memory - use quantized cache
            # Quantization reduces memory by ~4x for 4-bit
            quantized_max_tokens = max(max_tokens * 4, 1024)  # At least 1K tokens
            return cls.create_quantized(
                num_layers=num_layers,
                bits=4,
                max_size=min(quantized_max_tokens, 4096),
                keep=256,
                enable_tracing=enable_tracing,
                enable_memory_monitoring=enable_memory_monitoring,
            )

    def reset_all(self) -> None:
        """
        Reset all caches to empty state.

        Example:
            >>> manager.reset_all()
        """
        reset_cache_list(self.caches)  # type: ignore

    def get_state_dict(self) -> dict:
        """
        Get state dict for checkpointing.

        Returns:
            Dictionary with cache states for all layers

        Example:
            >>> state = manager.get_state_dict()
            >>> # Save state...
        """
        return {
            "num_layers": self.num_layers,
            "cache_type": self.cache_type,
            "caches": [
                {
                    "layer_idx": i,
                    "offset": cache.offset,
                    "state": cache.state,
                }
                for i, cache in enumerate(self.caches)
            ],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load state dict from checkpoint.

        Args:
            state_dict: State dictionary to restore

        Example:
            >>> manager.load_state_dict(saved_state)
        """
        if state_dict["num_layers"] != self.num_layers:
            raise ValueError(
                f"State dict has {state_dict['num_layers']} layers, "
                f"but manager has {self.num_layers}"
            )

        if state_dict["cache_type"] != self.cache_type:
            raise ValueError(
                f"State dict is for '{state_dict['cache_type']}' cache, "
                f"but manager has '{self.cache_type}'"
            )

        for cache_state in state_dict["caches"]:
            layer_idx = cache_state["layer_idx"]
            self.caches[layer_idx].state = cache_state["state"]

    def check_memory_pressure(self) -> dict | None:
        """
        Check current memory pressure.

        Returns:
            Memory status dict from MemoryMonitor, or None if monitoring disabled

        Example:
            >>> status = manager.check_memory_pressure()
            >>> if status and status['status'] == 'critical':
            ...     print("Memory pressure detected!")
        """
        if self.memory_monitor is None:
            return None
        return self.memory_monitor.check()

    def get_memory_trend(self, last_n: int = 10) -> str | None:
        """
        Get memory usage trend.

        Args:
            last_n: Number of recent checks to analyze (default: 10)

        Returns:
            Trend string ('increasing', 'stable', 'decreasing') or None

        Example:
            >>> trend = manager.get_memory_trend()
            >>> if trend == 'increasing':
            ...     print("Memory leak detected!")
        """
        if self.memory_monitor is None:
            return None
        return self.memory_monitor.get_trend(last_n=last_n)

    def get_trace_summary(self) -> list[dict]:
        """
        Get trace summary from all caches.

        Returns:
            List of trace summaries, one per layer

        Example:
            >>> summaries = manager.get_trace_summary()
            >>> for i, summary in enumerate(summaries):
            ...     print(f"Layer {i}: {summary['total_updates']} updates")
        """
        summaries = []
        for i, cache in enumerate(self.caches):
            if hasattr(cache, "get_trace_summary"):
                summary = cache.get_trace_summary()
                summary["layer_idx"] = i
                summaries.append(summary)
        return summaries

    def clear_traces(self) -> None:
        """
        Clear trace logs from all caches.

        Example:
            >>> manager.clear_traces()
        """
        for cache in self.caches:
            if hasattr(cache, "clear_trace"):
                cache.clear_trace()

    def __getitem__(self, idx: int) -> MLXKVCache | MLXRotatingKVCache | QuantizedMLXKVCache:
        """
        Get cache for a specific layer.

        Args:
            idx: Layer index

        Returns:
            Cache object for the layer

        Example:
            >>> layer_0_cache = manager[0]
        """
        return self.caches[idx]

    def __len__(self) -> int:
        """Get number of caches (layers)."""
        return len(self.caches)

    def __iter__(self):
        """Iterate over caches."""
        return iter(self.caches)


__all__ = ["KVCacheManager", "CacheType"]
