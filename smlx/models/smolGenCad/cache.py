#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Key-Value cache for smolGenCad decoder.

Provides KV cache for efficient autoregressive CAD sequence generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smlx.utils.cache import KVCache, RotatingKVCache
from smlx.utils.cache import make_cache as utils_make_cache

if TYPE_CHECKING:
    from .model import SmolGenCad


def make_cache(
    model: SmolGenCad,
    max_kv_size: int | None = None,
    keep: int = 4,
):
    """
    Create KV cache for smolGenCad decoder.

    Args:
        model: SmolGenCad model instance
        max_kv_size: Optional maximum cache size (uses RotatingKVCache if provided)
        keep: Number of recent tokens to keep when rotating (default: 4)

    Returns:
        List of cache objects (one per decoder layer)

    Examples:
        >>> # Standard cache (no size limit)
        >>> cache = make_cache(model)
        >>>
        >>> # Rotating cache with size limit
        >>> cache = make_cache(model, max_kv_size=512)
        >>>
        >>> # Use in generation
        >>> output = model.generate(
        ...     prompt="Create a cylinder",
        ...     cache=cache,
        ...     max_new_tokens=100
        ... )
    """
    num_layers = model.config.decoder.num_hidden_layers
    return utils_make_cache(num_layers, max_kv_size=max_kv_size, keep=keep)


__all__ = ["KVCache", "RotatingKVCache", "make_cache"]
