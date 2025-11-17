# Copyright © 2025 SMLX Project

"""
RoPE (Rotary Position Embeddings) aware cache management.

Provides cache wrappers that automatically handle RoPE offset synchronization
to ensure correct positional encoding during autoregressive generation.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from smlx.kv_cache.mlx_kv import MLXKVCache, MLXRotatingKVCache


class RoPECache:
    """
    Cache wrapper that handles RoPE offset synchronization.

    Automatically applies rotary position embeddings with the correct offset
    based on the cache state, ensuring proper positional encoding during
    autoregressive generation.

    Attributes:
        cache: Underlying KV cache
        rope: RoPE module for applying rotary embeddings

    Example:
        >>> rope = nn.RoPE(dims=64, traditional=False, base=10000)
        >>> cache = MLXKVCache()
        >>> rope_cache = RoPECache(cache, rope)
        >>>
        >>> # During generation
        >>> queries, keys, values = rope_cache.apply_rope_and_update(q, k, v)
    """

    def __init__(
        self,
        cache: MLXKVCache | MLXRotatingKVCache,
        rope: nn.RoPE,
    ):
        """
        Initialize RoPE-aware cache.

        Args:
            cache: Underlying KV cache (standard or rotating)
            rope: RoPE module for position embeddings
        """
        self.cache = cache
        self.rope = rope

    def apply_rope_and_update(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Apply RoPE with correct offset and update cache.

        This is the main method to use during generation. It handles:
        1. Applying RoPE to queries and keys with the correct offset
        2. Updating the cache with the new keys and values
        3. Returning all cached keys and values

        Args:
            queries: Query tensor [batch, n_heads, seq_len, head_dim]
            keys: Key tensor [batch, n_kv_heads, seq_len, head_dim]
            values: Value tensor [batch, n_kv_heads, seq_len, head_dim]

        Returns:
            Tuple of (queries_with_rope, all_keys, all_values)

        Example:
            >>> # In attention forward pass
            >>> queries, all_keys, all_values = rope_cache.apply_rope_and_update(q, k, v)
            >>> output = scaled_dot_product_attention(queries, all_keys, all_values)
        """
        # Apply RoPE with offset from cache
        queries = self.rope(queries, offset=self.cache.offset)
        keys = self.rope(keys, offset=self.cache.offset)

        # Update cache and fetch all keys/values
        all_keys, all_values = self.cache.update_and_fetch(keys, values)

        return queries, all_keys, all_values

    def apply_rope_no_cache(
        self,
        queries: mx.array,
        keys: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Apply RoPE without caching (e.g., during prefill without cache).

        Args:
            queries: Query tensor
            keys: Key tensor

        Returns:
            Tuple of (queries_with_rope, keys_with_rope)

        Example:
            >>> # During prefill without cache
            >>> queries, keys = rope_cache.apply_rope_no_cache(q, k)
        """
        queries = self.rope(queries)
        keys = self.rope(keys)
        return queries, keys

    @property
    def offset(self) -> int:
        """Get current cache offset."""
        return self.cache.offset

    @property
    def state(self) -> Any:
        """Get cache state for saving/loading."""
        return self.cache.state

    @state.setter
    def state(self, v: Any) -> None:
        """Set cache state from saved data."""
        self.cache.state = v

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


def initialize_rope_cache(
    dims: int,
    base: float = 10000.0,
    traditional: bool = False,
    num_layers: int = 1,
    max_kv_size: int | None = None,
    keep: int = 0,
    step: int | None = None,
    enable_tracing: bool = False,
) -> list[RoPECache]:
    """
    Create RoPE-aware cache for a transformer model.

    Factory function that creates a list of RoPE caches, one per layer,
    with consistent RoPE configuration.

    Args:
        dims: Dimension for RoPE (typically head_dim)
        base: Base for exponential frequency scaling (default: 10000.0)
        traditional: Whether to use traditional RoPE (default: False)
        num_layers: Number of transformer layers (default: 1)
        max_kv_size: Optional max cache size (uses rotating cache if set)
        keep: Number of initial tokens to preserve in rotating cache
        step: Optional allocation step size (default: 256)
        enable_tracing: Enable trace logging (default: False)

    Returns:
        List of RoPECache objects, one per layer

    Example:
        >>> # Standard cache with RoPE
        >>> caches = initialize_rope_cache(
        ...     dims=64,
        ...     base=10000.0,
        ...     num_layers=24
        ... )
        >>>
        >>> # Rotating cache with RoPE
        >>> caches = initialize_rope_cache(
        ...     dims=64,
        ...     num_layers=24,
        ...     max_kv_size=2048,
        ...     keep=256
        ... )
    """
    # Create RoPE module (shared across all layers)
    rope = nn.RoPE(dims, traditional=traditional, base=base)

    # Create caches
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

        # Wrap with RoPE awareness
        caches.append(RoPECache(cache, rope))

    return caches


def create_rope_module(
    dims: int,
    base: float = 10000.0,
    traditional: bool = False,
    scaling_config: dict | None = None,
) -> nn.RoPE:
    """
    Create a RoPE module with optional scaling.

    Supports different RoPE variants through the scaling_config parameter.

    Args:
        dims: Dimension for RoPE (typically head_dim)
        base: Base for exponential frequency scaling (default: 10000.0)
        traditional: Whether to use traditional RoPE (default: False)
        scaling_config: Optional scaling configuration with keys:
            - type/rope_type: "default", "linear" (more types can be added)
            - factor: Scaling factor (for linear scaling)

    Returns:
        nn.RoPE module

    Example:
        >>> # Standard RoPE
        >>> rope = create_rope_module(dims=64, base=10000.0)
        >>>
        >>> # Linear scaled RoPE (for longer contexts)
        >>> rope = create_rope_module(
        ...     dims=64,
        ...     scaling_config={"type": "linear", "factor": 2.0}
        ... )
    """
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")
    else:
        rope_type = "default"

    if rope_type in ["default", "linear"]:
        scale = (
            1 / scaling_config["factor"]
            if rope_type == "linear" and scaling_config is not None
            else 1.0
        )
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)
    else:
        # For now, we only support default and linear scaling
        # Additional RoPE types (llama3, yarn, longrope) can be added later
        raise ValueError(f"Unsupported RoPE type: {rope_type}")


__all__ = [
    "RoPECache",
    "initialize_rope_cache",
    "create_rope_module",
]
