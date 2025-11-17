# Copyright © 2025 SMLX Project
# Adapted from MLX framework reference implementations

"""
Key-Value cache implementations for efficient autoregressive generation.

Provides KV cache classes for storing attention keys and values during text generation,
reducing computational cost by avoiding recomputation of past tokens.

Enhanced Cache Modules:
    For advanced features like automatic memory sizing, quantization, and OOM prevention,
    use model-specific cache modules:

    - smlx.models.SmolLM2_135M.cache - Enhanced cache for SmolLM2-135M
    - smlx.models.SmolLM2_360M.cache - Enhanced cache for SmolLM2-360M
    - smlx.models.Moondream2.cache - Enhanced cache for Moondream2
    - smlx.models.TinyLLaVA.cache - Enhanced cache for TinyLLaVA
    - smlx.models.SmolVLM_256M.cache - Enhanced cache for SmolVLM-256M
    - smlx.models.SmolVLM_500M_Instruct.cache - Enhanced cache for SmolVLM-500M

    These modules provide:
    - Automatic cache type selection based on available memory
    - 4-bit and 8-bit quantized caches for memory efficiency
    - Memory pressure monitoring and automatic intervention
    - OOM prevention with PressureBreaker

    See also: smlx.kv_cache module for the core enhanced cache infrastructure.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


class KVCache:
    """
    Simple Key-Value cache for efficient autoregressive generation.

    Stores keys and values from attention layers to avoid recomputing
    them for previously generated tokens. This implementation uses dynamic
    allocation with a step size to minimize memory reallocation overhead.

    The cache automatically grows as needed, allocating memory in chunks
    (default: 256 tokens at a time) to balance memory efficiency and allocation overhead.

    Attributes:
        keys: Cached keys [batch, n_heads, seq_len, head_dim]
        values: Cached values [batch, n_heads, seq_len, head_dim]
        offset: Number of tokens currently in cache
        step: Allocation step size (default: 256)

    Example:
        >>> cache = KVCache()
        >>> # During generation
        >>> all_keys, all_values = cache.update_and_fetch(new_keys, new_values)
    """

    step: int = 256  # Allocation step size
    keys: Optional[mx.array]
    values: Optional[mx.array]
    offset: int

    def __init__(self, step: Optional[int] = None):
        """
        Initialize empty cache.

        Args:
            step: Optional custom allocation step size (default: 256)
        """
        if step is not None:
            self.step = step
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Update cache with new keys/values and return all cached keys/values.

        Args:
            keys: New keys to add [batch, n_heads, seq_len, head_dim]
            values: New values to add [batch, n_heads, seq_len, head_dim]

        Returns:
            Tuple of (all_keys, all_values) including the new ones

        Example:
            >>> all_keys, all_values = cache.update_and_fetch(new_keys, new_values)
        """
        prev = self.offset

        # Allocate or expand cache if needed
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)

            if self.keys is not None:
                assert self.values is not None
                # Trim to actual size if not aligned to step
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                # Concatenate with new allocation
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        # Update offset
        self.offset += keys.shape[2]

        # Store new keys/values
        assert self.keys is not None and self.values is not None
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

        # Return all cached keys/values
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        """
        Get cache state for saving/loading.

        Returns:
            Tuple of (keys, values) trimmed to actual size.
            Returns (None, None) if cache is empty.
        """
        if self.keys is None or self.values is None:
            return None, None

        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        """
        Set cache state from saved data.

        Args:
            v: Tuple of (keys, values) to restore
        """
        self.keys, self.values = v
        assert self.keys is not None
        self.offset = self.keys.shape[2]

    def reset(self) -> None:
        """Reset cache to empty state."""
        self.keys = None
        self.values = None
        self.offset = 0


class RotatingKVCache:
    """
    Rotating Key-Value cache with maximum size limit.

    When the cache reaches max_size, old entries are removed to make room
    for new ones, keeping the most recent tokens (and optionally some initial tokens).
    This implements a circular buffer strategy for long-context generation with
    limited memory.

    The cache can preserve a fixed number of initial tokens (prompt tokens) while
    rotating through more recent tokens, which is useful for maintaining context
    in very long generation tasks.

    Attributes:
        max_size: Maximum number of tokens to cache
        keep: Number of initial tokens to always keep
        keys: Cached keys
        values: Cached values
        offset: Total number of tokens processed
        step: Allocation step size (default: 256)

    Example:
        >>> cache = RotatingKVCache(max_size=2048, keep=256)  # Keep first 256 tokens
        >>> # During generation
        >>> all_keys, all_values = cache.update_and_fetch(new_keys, new_values)
    """

    step: int = 256  # Allocation step size
    max_size: int
    keep: int
    keys: Optional[mx.array]
    values: Optional[mx.array]
    offset: int
    _idx: int

    def __init__(self, max_size: int, keep: int = 0, step: Optional[int] = None):
        """
        Initialize rotating cache.

        Args:
            max_size: Maximum number of tokens to keep in cache
            keep: Number of initial tokens to always preserve (default: 0)
            step: Optional custom allocation step size (default: 256)
        """
        if step is not None:
            self.step = step
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self._idx = 0

    def _trim(
        self, trim_size: int, v: mx.array, append: Optional[mx.array] = None
    ) -> mx.array:
        """
        Trim cache and optionally append new data.

        Args:
            trim_size: Number of tokens to trim
            v: Array to trim
            append: Optional array to append after trimming

        Returns:
            Trimmed (and optionally appended) array
        """
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v: mx.array) -> mx.array:
        """
        Rearrange cache into temporal order.

        When the cache rotates, tokens may be out of order. This method
        rearranges them back into chronological order.

        Args:
            v: Array to rearrange

        Returns:
            Array in temporal order
        """
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self.offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Update cache with concatenation (for multi-token updates).

        Args:
            keys: New keys to add
            values: New values to add

        Returns:
            Tuple of (all_keys, all_values)
        """
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put keys/values in temporal order
            assert self.values is not None
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self.keys.shape[2]

            # Trim if needed
            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)

        self.offset += keys.shape[2]
        assert self.keys is not None
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Update cache in-place (for single token updates).

        Args:
            keys: New keys to add (typically shape [B, H, 1, D])
            values: New values to add (typically shape [B, H, 1, D])

        Returns:
            Tuple of (all_keys, all_values)
        """
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self.offset

        # Allocate if needed
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)

            if self.keys is not None:
                assert self.values is not None
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if exceeding max_size
        assert self.keys is not None and self.values is not None
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate if at capacity
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign new keys/values
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self.offset += S
        self._idx += S

        # Return cached keys/values
        if self.offset < self.max_size:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )
        return self.keys, self.values

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Update cache and return all cached keys/values.

        Uses in-place updates for single tokens (faster) and concatenation
        for multiple tokens (more flexible).

        Args:
            keys: New keys to add
            values: New values to add

        Returns:
            Tuple of (all_keys, all_values)
        """
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        """Get cache state for saving/loading."""
        # Handle empty cache
        if self.keys is None or self.values is None:
            return None, None

        if self.offset < self.keys.shape[2]:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )
        else:
            return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        """Set cache state from saved data."""
        self.keys, self.values = v

    def reset(self) -> None:
        """Reset cache to empty state."""
        self.keys = None
        self.values = None
        self.offset = 0
        self._idx = 0


def make_cache(
    num_layers: int,
    max_kv_size: Optional[int] = None,
    keep: int = 4,
    step: Optional[int] = None,
) -> list[KVCache | RotatingKVCache]:
    """
    Create KV cache list for a transformer model.

    Creates one cache object per layer. If max_kv_size is specified,
    uses RotatingKVCache to limit memory usage; otherwise uses standard KVCache.

    Note: For enhanced cache features (automatic sizing, quantization, memory
    monitoring), use model-specific cache modules instead:
        - smlx.models.SmolLM2_135M.cache.make_cache()
        - smlx.models.SmolLM2_360M.cache.make_cache()
        - smlx.models.Moondream2.cache.make_cache()
        - smlx.models.TinyLLaVA.cache.make_cache()
        - smlx.models.SmolVLM_256M.cache.make_cache()
        - smlx.models.SmolVLM_500M_Instruct.cache.make_cache()

    Args:
        num_layers: Number of transformer layers in the model
        max_kv_size: Optional maximum cache size (uses RotatingKVCache if provided)
        keep: Number of initial tokens to preserve in rotating cache (default: 4)
        step: Optional custom allocation step size (default: 256)

    Returns:
        List of cache objects, one per layer

    Example:
        >>> # Standard cache (unlimited)
        >>> cache = make_cache(num_layers=30)
        >>>
        >>> # Rotating cache (limited to 2048 tokens, keep first 256)
        >>> cache = make_cache(num_layers=30, max_kv_size=2048, keep=256)
    """
    if max_kv_size is not None:
        # Use rotating cache with maximum size
        return [
            RotatingKVCache(max_size=max_kv_size, keep=keep, step=step)
            for _ in range(num_layers)
        ]
    else:
        # Use standard cache
        return [KVCache(step=step) for _ in range(num_layers)]


def reset_cache(cache: list[KVCache | RotatingKVCache]) -> None:
    """
    Reset all caches in a cache list.

    Clears all cached keys, values, and offsets for each layer's cache.
    This is useful for cleaning up between generation runs or test cases.

    Args:
        cache: List of cache objects to reset

    Example:
        >>> cache = make_cache(num_layers=30)
        >>> # ... use cache for generation ...
        >>> reset_cache(cache)  # Clean up for next generation
    """
    for c in cache:
        c.reset()


# Alias for backwards compatibility
make_kv_caches = make_cache


__all__ = [
    "KVCache",
    "RotatingKVCache",
    "make_cache",
    "make_kv_caches",
    "reset_cache",
]
