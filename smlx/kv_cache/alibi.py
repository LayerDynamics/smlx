# Copyright � 2025 SMLX Project

"""
ALiBi (Attention with Linear Biases) support for KV caching.

Provides ALiBi-aware cache management for models that use ALiBi instead of
RoPE for positional encoding. ALiBi adds distance-based biases to attention
scores rather than using explicit position embeddings.

Reference: "Train Short, Test Long: Attention with Linear Biases Enables
Input Length Extrapolation" (Press et al., 2021)
https://arxiv.org/abs/2108.12409
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from smlx.kv_cache.mlx_kv import MLXKVCache, MLXRotatingKVCache


class ALiBiCache:
    """
    Cache with ALiBi-style attention biases.

    ALiBi (Attention with Linear Biases) is an alternative to RoPE that encodes
    position information by adding linear biases to attention scores. Each head
    has a different slope, and the bias is proportional to the distance between
    query and key positions.

    The bias for position (i, j) in head h is: bias[h, i, j] = slope[h] * (i - j)

    Attributes:
        cache: Underlying KV cache
        num_heads: Number of attention heads
        slopes: Head-specific slopes for ALiBi

    Example:
        >>> cache = MLXKVCache()
        >>> alibi_cache = ALiBiCache(cache, num_heads=12)
        >>>
        >>> # During generation
        >>> keys, values, bias = alibi_cache.update_and_fetch_with_bias(k, v)
        >>> # Use bias in attention computation
    """

    def __init__(
        self,
        cache: MLXKVCache | MLXRotatingKVCache,
        num_heads: int,
    ):
        """
        Initialize ALiBi cache.

        Args:
            cache: Underlying KV cache (standard or rotating)
            num_heads: Number of attention heads
        """
        self.cache = cache
        self.num_heads = num_heads
        self.slopes = self._get_slopes(num_heads)

    def _get_slopes(self, n_heads: int) -> mx.array:
        """
        Compute head-specific slopes for ALiBi.

        The slopes are computed as: m_h = 2^(-8h/n) for head h.
        This gives each head a different receptive field.

        Args:
            n_heads: Number of attention heads

        Returns:
            Array of slopes, shape [n_heads]
        """
        # Compute slopes: 2^(-8h/n) for h in range(n_heads)
        # Start with closest heads (h=1) and go to farthest (h=n_heads)
        positions = mx.arange(1, n_heads + 1, dtype=mx.float32)
        slopes = mx.power(2.0, -8.0 * positions / n_heads)

        return slopes

    def compute_bias(self, seq_len: int, query_len: int | None = None) -> mx.array:
        """
        Compute ALiBi bias matrix.

        Generates the bias matrix for attention computation:
        bias[h, i, j] = slope[h] * (i - j)

        where i is the query position and j is the key position.

        Args:
            seq_len: Total sequence length (including cache)
            query_len: Length of query sequence (default: 1 for generation)

        Returns:
            Bias matrix of shape [num_heads, query_len, seq_len]

        Example:
            >>> bias = alibi_cache.compute_bias(seq_len=100, query_len=1)
            >>> # bias shape: [12, 1, 100]
        """
        if query_len is None:
            query_len = 1

        # Create position matrices
        # query_positions: [query_len, 1]
        # key_positions: [1, seq_len]
        query_positions = mx.arange(seq_len - query_len, seq_len, dtype=mx.float32)[:, None]
        key_positions = mx.arange(seq_len, dtype=mx.float32)[None, :]

        # Compute distance matrix: [query_len, seq_len]
        # For ALiBi, we compute (key_pos - query_pos) so past positions are negative
        # Then multiply by positive slope to get negative bias for past
        distance = key_positions - query_positions

        # Apply head-specific slopes: [num_heads, 1, 1] * [1, query_len, seq_len]
        # Result: [num_heads, query_len, seq_len]
        # ALiBi formula: bias[i,j] = slope * (j - i)
        # For past positions (j < i), this gives negative bias
        slopes_expanded = self.slopes[:, None, None]
        bias = slopes_expanded * distance[None, :, :]

        return bias

    def update_and_fetch_with_bias(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Update cache and return keys, values, and ALiBi bias.

        This is the main method to use during generation. It:
        1. Updates the cache with new keys and values
        2. Computes the appropriate ALiBi bias for the current sequence length

        Args:
            keys: New keys to add [batch, n_kv_heads, seq_len, head_dim]
            values: New values to add [batch, n_kv_heads, seq_len, head_dim]

        Returns:
            Tuple of (all_keys, all_values, bias_matrix)
            - all_keys: All cached keys
            - all_values: All cached values
            - bias_matrix: ALiBi bias [num_heads, query_len, total_seq_len]

        Example:
            >>> # In attention forward pass
            >>> all_keys, all_values, bias = alibi_cache.update_and_fetch_with_bias(k, v)
            >>> # Add bias to attention scores
            >>> scores = queries @ all_keys.transpose(0, 1, 3, 2) + bias
        """
        # Update cache
        all_keys, all_values = self.cache.update_and_fetch(keys, values)

        # Compute bias for current sequence length
        # all_keys shape: [batch, n_heads, total_seq_len, head_dim]
        total_seq_len = all_keys.shape[2]
        query_len = keys.shape[2]

        bias = self.compute_bias(seq_len=total_seq_len, query_len=query_len)

        return all_keys, all_values, bias

    @property
    def offset(self) -> int:
        """Get current cache offset."""
        return self.cache.offset

    @property
    def state(self) -> Any:
        """Get cache state for saving/loading."""
        return {
            "cache_state": self.cache.state,
            "num_heads": self.num_heads,
            "offset": self.cache.offset,
        }

    @state.setter
    def state(self, state_dict: dict) -> None:
        """Set cache state from saved data."""
        self.cache.state = state_dict["cache_state"]
        self.num_heads = state_dict["num_heads"]
        # Recompute slopes in case num_heads changed
        self.slopes = self._get_slopes(self.num_heads)

    def reset(self) -> None:
        """Reset cache to empty state."""
        self.cache.reset()

    def get_trace_summary(self) -> dict:
        """Get trace summary from underlying cache."""
        if hasattr(self.cache, "get_trace_summary"):
            return self.cache.get_trace_summary()
        return {"enabled": False}

    def clear_trace(self) -> None:
        """Clear trace log from underlying cache."""
        if hasattr(self.cache, "clear_trace"):
            self.cache.clear_trace()


def initialize_alibi_cache(
    num_heads: int,
    num_layers: int = 1,
    max_kv_size: int | None = None,
    keep: int = 0,
    step: int | None = None,
    enable_tracing: bool = False,
) -> list[ALiBiCache]:
    """
    Create ALiBi-aware cache for a transformer model.

    Factory function that creates a list of ALiBi caches, one per layer.

    Args:
        num_heads: Number of attention heads
        num_layers: Number of transformer layers (default: 1)
        max_kv_size: Optional max cache size (uses rotating cache if set)
        keep: Number of initial tokens to preserve in rotating cache
        step: Optional allocation step size (default: 256)
        enable_tracing: Enable trace logging (default: False)

    Returns:
        List of ALiBiCache objects, one per layer

    Example:
        >>> # Standard cache with ALiBi
        >>> caches = initialize_alibi_cache(
        ...     num_heads=12,
        ...     num_layers=24
        ... )
        >>>
        >>> # Rotating cache with ALiBi
        >>> caches = initialize_alibi_cache(
        ...     num_heads=12,
        ...     num_layers=24,
        ...     max_kv_size=2048,
        ...     keep=256
        ... )
    """
    caches = []
    for _ in range(num_layers):
        if max_kv_size is not None:
            # Use rotating cache
            cache: MLXKVCache | MLXRotatingKVCache = MLXRotatingKVCache(
                max_size=max_kv_size,
                keep=keep,
                step=step,
                enable_tracing=enable_tracing,
            )
        else:
            # Use standard cache
            cache = MLXKVCache(step=step, enable_tracing=enable_tracing)

        # Wrap with ALiBi
        caches.append(ALiBiCache(cache, num_heads))

    return caches


__all__ = [
    "ALiBiCache",
    "initialize_alibi_cache",
]
