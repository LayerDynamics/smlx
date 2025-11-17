# Copyright � 2025 SMLX Project

"""
MLX KV Cache implementations with enhanced features.

Provides enhanced KV cache classes with tracing support, quantization,
and compatibility layers for the SMLX project.
"""

from __future__ import annotations

import time

import mlx.core as mx

from smlx.utils.cache import KVCache as BaseKVCache
from smlx.utils.cache import RotatingKVCache as BaseRotatingKVCache


class MLXKVCache(BaseKVCache):
    """
    Enhanced KV cache with SMLX-specific features.

    Extends the base KVCache with optional tracing support for debugging
    and profiling cache operations.

    Attributes:
        keys: Cached keys [batch, n_heads, seq_len, head_dim]
        values: Cached values [batch, n_heads, seq_len, head_dim]
        offset: Number of tokens currently in cache
        step: Allocation step size (default: 256)
        enable_tracing: Whether to record cache operations
        trace_log: List of trace events (if tracing enabled)

    Example:
        >>> cache = MLXKVCache(enable_tracing=True)
        >>> all_keys, all_values = cache.update_and_fetch(new_keys, new_values)
        >>> print(cache.trace_log)  # View recorded operations
    """

    def __init__(self, step: int | None = None, enable_tracing: bool = False):
        """
        Initialize enhanced MLX KV cache.

        Args:
            step: Optional custom allocation step size (default: 256)
            enable_tracing: Enable trace logging (default: False)
        """
        super().__init__(step=step)
        self.enable_tracing = enable_tracing
        self.trace_log: list[dict] = [] if enable_tracing else []

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Update cache with new keys/values and return all cached keys/values.

        If tracing is enabled, records the operation details.

        Args:
            keys: New keys to add [batch, n_heads, seq_len, head_dim]
            values: New values to add [batch, n_heads, seq_len, head_dim]

        Returns:
            Tuple of (all_keys, all_values) including the new ones
        """
        if self.enable_tracing:
            old_offset = self.offset
            start_time = time.time()

        result = super().update_and_fetch(keys, values)

        if self.enable_tracing:
            elapsed = time.time() - start_time
            self.trace_log.append(
                {
                    "type": "update",
                    "old_offset": old_offset,
                    "new_offset": self.offset,
                    "keys_shape": tuple(keys.shape),
                    "values_shape": tuple(values.shape),
                    "elapsed_ms": elapsed * 1000,
                    "timestamp": time.time(),
                }
            )

        return result

    def get_trace_summary(self) -> dict:
        """
        Get summary of trace operations.

        Returns:
            Dictionary with trace statistics
        """
        if not self.enable_tracing:
            return {"enabled": False}

        if not self.trace_log:
            return {
                "enabled": True,
                "total_updates": 0,
                "total_time_ms": 0.0,
                "avg_time_ms": 0.0,
                "current_offset": self.offset,
                "events": [],
            }

        total_updates = len(self.trace_log)
        total_time_ms = sum(e.get("elapsed_ms", 0) for e in self.trace_log)

        return {
            "enabled": True,
            "total_updates": total_updates,
            "total_time_ms": total_time_ms,
            "avg_time_ms": total_time_ms / total_updates if total_updates > 0 else 0,
            "current_offset": self.offset,
            "events": self.trace_log,
        }

    def clear_trace(self) -> None:
        """Clear trace log."""
        if self.enable_tracing:
            self.trace_log = []

    def reset(self) -> None:
        """Reset cache to empty state and clear trace log."""
        super().reset()
        if self.enable_tracing:
            self.trace_log = []


class MLXRotatingKVCache(BaseRotatingKVCache):
    """
    Enhanced rotating KV cache with SMLX-specific features.

    Extends the base RotatingKVCache with optional tracing support.

    Attributes:
        max_size: Maximum number of tokens to cache
        keep: Number of initial tokens to always keep
        keys: Cached keys
        values: Cached values
        offset: Total number of tokens processed
        step: Allocation step size (default: 256)
        enable_tracing: Whether to record cache operations
        trace_log: List of trace events (if tracing enabled)

    Example:
        >>> cache = MLXRotatingKVCache(max_size=2048, keep=256, enable_tracing=True)
        >>> all_keys, all_values = cache.update_and_fetch(new_keys, new_values)
    """

    def __init__(
        self,
        max_size: int,
        keep: int = 0,
        step: int | None = None,
        enable_tracing: bool = False,
    ):
        """
        Initialize enhanced rotating cache.

        Args:
            max_size: Maximum number of tokens to keep in cache
            keep: Number of initial tokens to always preserve (default: 0)
            step: Optional custom allocation step size (default: 256)
            enable_tracing: Enable trace logging (default: False)
        """
        super().__init__(max_size=max_size, keep=keep, step=step)
        self.enable_tracing = enable_tracing
        self.trace_log: list[dict] = [] if enable_tracing else []

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Update cache and return all cached keys/values.

        If tracing is enabled, records the operation details including rotation events.

        Args:
            keys: New keys to add
            values: New values to add

        Returns:
            Tuple of (all_keys, all_values)
        """
        if self.enable_tracing:
            old_offset = self.offset
            old_idx = self._idx
            start_time = time.time()

        result = super().update_and_fetch(keys, values)

        if self.enable_tracing:
            elapsed = time.time() - start_time
            rotated = self._idx < old_idx  # Detect if rotation occurred
            self.trace_log.append(
                {
                    "type": "update",
                    "old_offset": old_offset,
                    "new_offset": self.offset,
                    "old_idx": old_idx,
                    "new_idx": self._idx,
                    "rotated": rotated,
                    "keys_shape": tuple(keys.shape),
                    "values_shape": tuple(values.shape),
                    "elapsed_ms": elapsed * 1000,
                    "timestamp": time.time(),
                }
            )

        return result

    def get_trace_summary(self) -> dict:
        """
        Get summary of trace operations.

        Returns:
            Dictionary with trace statistics including rotation count
        """
        if not self.enable_tracing or not self.trace_log:
            return {"enabled": False}

        total_updates = len(self.trace_log)
        total_time_ms = sum(e.get("elapsed_ms", 0) for e in self.trace_log)
        rotations = sum(1 for e in self.trace_log if e.get("rotated", False))

        return {
            "enabled": True,
            "total_updates": total_updates,
            "total_rotations": rotations,
            "total_time_ms": total_time_ms,
            "avg_time_ms": total_time_ms / total_updates if total_updates > 0 else 0,
            "current_offset": self.offset,
            "max_size": self.max_size,
            "keep": self.keep,
            "events": self.trace_log,
        }

    def clear_trace(self) -> None:
        """Clear trace log."""
        if self.enable_tracing:
            self.trace_log = []

    def reset(self) -> None:
        """Reset cache to empty state and clear trace log."""
        super().reset()
        if self.enable_tracing:
            self.trace_log = []


class QuantizedMLXKVCache:
    """
    Quantized KV cache for memory-efficient storage.

    Stores keys and values in quantized format (4-bit or 8-bit) to reduce
    memory usage. Uses group-based quantization for better accuracy.

    Attributes:
        cache: Underlying KVCache or RotatingKVCache
        bits: Number of bits for quantization (4 or 8)
        group_size: Group size for quantization (default: 64)
        quantize_threshold: Minimum offset before quantizing (default: 256)

    Example:
        >>> cache = QuantizedMLXKVCache(bits=4, group_size=64)
        >>> all_keys, all_values = cache.update_and_fetch(new_keys, new_values)
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 64,
        quantize_threshold: int = 256,
        max_size: int | None = None,
        keep: int = 0,
        step: int | None = None,
        enable_tracing: bool = False,
    ):
        """
        Initialize quantized KV cache.

        Args:
            bits: Number of bits for quantization (must be 4)
            group_size: Group size for quantization (default: 64)
            quantize_threshold: Min tokens before quantizing (default: 256)
            max_size: Optional maximum cache size (uses rotating cache if set)
            keep: Number of initial tokens to preserve (rotating cache only)
            step: Optional custom allocation step size (default: 256)
            enable_tracing: Enable trace logging (default: False)

        Note:
            MLX's group-based quantization currently only supports 4-bit quantization.
            Attempting to use other bit widths will raise a ValueError.
        """
        # Validate bits parameter
        if bits != 4:
            raise ValueError(
                f"MLX group-based quantization only supports 4-bit. Got bits={bits}. "
                "For other bit widths, use standard quantization methods."
            )

        self.bits = bits
        self.group_size = group_size
        self.quantize_threshold = quantize_threshold
        self.enable_tracing = enable_tracing

        # Create underlying cache
        if max_size is not None:
            self.cache: MLXKVCache | MLXRotatingKVCache = MLXRotatingKVCache(
                max_size=max_size, keep=keep, step=step, enable_tracing=enable_tracing
            )
        else:
            self.cache = MLXKVCache(step=step, enable_tracing=enable_tracing)

        self.quantized_keys: tuple | None = None
        self.quantized_values: tuple | None = None
        self.is_quantized = False

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Update cache with new keys/values.

        Automatically quantizes cache once threshold is reached.

        Args:
            keys: New keys to add
            values: New values to add

        Returns:
            Tuple of (all_keys, all_values) - dequantized if needed
        """
        # Update underlying cache
        all_keys, all_values = self.cache.update_and_fetch(keys, values)

        # Quantize if threshold reached and not already quantized
        if not self.is_quantized and self.cache.offset >= self.quantize_threshold:
            self._quantize_cache(all_keys, all_values)

        # Return dequantized if quantized, otherwise raw arrays
        if self.is_quantized:
            return self._dequantize()

        return all_keys, all_values

    def _quantize_cache(self, keys: mx.array, values: mx.array) -> None:
        """
        Quantize the current cache.

        Args:
            keys: Keys to quantize
            values: Values to quantize
        """
        # Use MLX's quantize function
        self.quantized_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        self.quantized_values = mx.quantize(
            values, group_size=self.group_size, bits=self.bits
        )
        self.is_quantized = True

        # Clear unquantized cache to save memory
        self.cache.keys = None
        self.cache.values = None

    def _dequantize(self) -> tuple[mx.array, mx.array]:
        """
        Dequantize cached keys and values.

        Returns:
            Tuple of (dequantized_keys, dequantized_values)
        """
        assert self.quantized_keys is not None and self.quantized_values is not None
        # Unpack quantized data: (w_q, scales, biases)
        # Pass group_size and bits to dequantize
        keys = mx.dequantize(
            *self.quantized_keys,
            group_size=self.group_size,
            bits=self.bits,
            dtype=mx.float32,
        )
        values = mx.dequantize(
            *self.quantized_values,
            group_size=self.group_size,
            bits=self.bits,
            dtype=mx.float32,
        )
        return keys, values

    @property
    def offset(self) -> int:
        """Get current cache offset."""
        return self.cache.offset

    @property
    def state(self) -> dict:
        """
        Get cache state for saving/loading.

        Returns:
            Dictionary with cache state including quantization info
        """
        if self.is_quantized:
            return {
                "quantized": True,
                "bits": self.bits,
                "group_size": self.group_size,
                "offset": self.cache.offset,
                "quantized_keys": self.quantized_keys,
                "quantized_values": self.quantized_values,
            }
        else:
            return {
                "quantized": False,
                "offset": self.cache.offset,
                "keys": self.cache.keys,
                "values": self.cache.values,
            }

    @state.setter
    def state(self, state_dict: dict) -> None:
        """
        Set cache state from saved data.

        Args:
            state_dict: State dictionary to restore
        """
        if state_dict.get("quantized", False):
            self.is_quantized = True
            self.bits = state_dict["bits"]
            self.group_size = state_dict["group_size"]
            self.cache.offset = state_dict["offset"]
            self.quantized_keys = state_dict["quantized_keys"]
            self.quantized_values = state_dict["quantized_values"]
        else:
            self.is_quantized = False
            self.cache.offset = state_dict["offset"]
            self.cache.keys = state_dict.get("keys")
            self.cache.values = state_dict.get("values")

    def reset(self) -> None:
        """Reset cache to empty state."""
        self.cache.reset()
        self.quantized_keys = None
        self.quantized_values = None
        self.is_quantized = False

    def get_trace_summary(self) -> dict:
        """Get trace summary from underlying cache."""
        summary = self.cache.get_trace_summary()
        summary["quantized"] = self.is_quantized
        summary["bits"] = self.bits if self.is_quantized else None
        return summary

    def clear_trace(self) -> None:
        """Clear trace log from underlying cache."""
        self.cache.clear_trace()


__all__ = [
    "MLXKVCache",
    "MLXRotatingKVCache",
    "QuantizedMLXKVCache",
]
