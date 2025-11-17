"""
Common attention mechanisms for SMLX models.

This module provides reusable attention layer implementations that can be shared
across different model architectures.

Implementations:
- MultiHeadAttention: Standard multi-head self-attention
- GroupedQueryAttention: Grouped-query attention (GQA) for efficient inference
- MultiQueryAttention: Multi-query attention (MQA) - special case of GQA
- CrossAttention: Cross-attention for encoder-decoder models
- SlidingWindowAttention: Local attention with sliding window
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: int | None = None,
) -> mx.array:
    """
    Create a causal attention mask.

    Args:
        N: Sequence length
        offset: Offset for the mask (used with KV cache)
        window_size: Optional sliding window size

    Returns:
        Boolean mask array of shape (N, N+offset)
    """
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    return mask


def create_attention_mask(
    hidden_states: mx.array,
    cache: Any | None = None,
    window_size: int | None = None,
) -> mx.array | str | None:
    """
    Create attention mask for the given input.

    Args:
        hidden_states: Input hidden states
        cache: Optional KV cache
        window_size: Optional sliding window size

    Returns:
        Attention mask (array, "causal" string, or None)
    """
    N = hidden_states.shape[1]  # Sequence length

    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, window_size=window_size)

    if N == 1:
        return None

    if window_size and N > window_size:
        return create_causal_mask(N, window_size=window_size)

    return "causal"


def scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
    cache: Any | None = None,
) -> mx.array:
    """
    Scaled dot-product attention with optional KV cache.

    Args:
        queries: Query tensor [batch, n_heads, seq_len, head_dim]
        keys: Key tensor [batch, n_kv_heads, seq_len, head_dim]
        values: Value tensor [batch, n_kv_heads, seq_len, head_dim]
        scale: Scaling factor (typically 1/sqrt(head_dim))
        mask: Optional attention mask
        cache: Optional KV cache

    Returns:
        Attention output [batch, n_heads, seq_len, head_dim]
    """
    # Use MLX's optimized implementation
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
    )


def initialize_rope(
    dims: int,
    base: float = 10000.0,
    traditional: bool = False,
    scaling_config: dict[str, float | str] | None = None,
    max_position_embeddings: int | None = None,
) -> nn.RoPE:
    """
    Initialize RoPE (Rotary Position Embedding) with optional scaling.

    Args:
        dims: Dimension of the embeddings
        base: Base for the exponential scaling
        traditional: Whether to use traditional RoPE
        scaling_config: Optional scaling configuration
        max_position_embeddings: Maximum sequence length

    Returns:
        RoPE module
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
        # Additional RoPE types can be added as needed
        raise ValueError(f"Unsupported RoPE type: {rope_type}")


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention.

    Implements the attention mechanism from "Attention is All You Need":
    - Projects input to Q, K, V
    - Applies scaled dot-product attention
    - Projects output back to hidden dimension

    Args:
        hidden_size: Dimension of input/output
        num_heads: Number of attention heads
        head_dim: Dimension of each head (default: hidden_size // num_heads)
        bias: Whether to use bias in linear projections
        dropout: Dropout probability (default: 0.0)
        rope_base: Base for RoPE (default: 10000.0)
        rope_traditional: Whether to use traditional RoPE
        rope_scaling: Optional RoPE scaling configuration
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, float | str] | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

        # Rotary Position Embedding
        self.rope = initialize_rope(
            self.head_dim,
            base=rope_base,
            traditional=rope_traditional,
            scaling_config=rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        """
        Apply multi-head attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            mask: Optional attention mask
            cache: Optional KV cache

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        B, L, D = x.shape

        # Project to Q, K, V
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Scaled dot-product attention
        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask, cache=cache
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-query attention (GQA) for efficient inference.

    GQA is a generalization of multi-head attention that uses fewer KV heads
    than query heads, reducing memory bandwidth during autoregressive generation.

    References:
    - "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"

    Args:
        hidden_size: Dimension of input/output
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (must divide num_heads)
        head_dim: Dimension of each head (default: hidden_size // num_heads)
        bias: Whether to use bias in linear projections
        rope_base: Base for RoPE
        rope_traditional: Whether to use traditional RoPE
        rope_scaling: Optional RoPE scaling configuration
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        bias: bool = False,
        rope_base: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, float | str] | None = None,
    ):
        super().__init__()

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        # Q, K, V projections (K and V use fewer heads)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

        # Rotary Position Embedding
        self.rope = initialize_rope(
            self.head_dim,
            base=rope_base,
            traditional=rope_traditional,
            scaling_config=rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        """
        Apply grouped-query attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            mask: Optional attention mask
            cache: Optional KV cache

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        B, L, D = x.shape

        # Project to Q, K, V
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Scaled dot-product attention
        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask, cache=cache
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MultiQueryAttention(nn.Module):
    """
    Multi-query attention (MQA) - special case of GQA with single KV head.

    MQA uses a single key/value head shared across all query heads, maximizing
    memory efficiency during inference at the cost of some quality.

    References:
    - "Fast Transformer Decoding: One Write-Head is All You Need"

    Args:
        hidden_size: Dimension of input/output
        num_heads: Number of query heads
        head_dim: Dimension of each head (default: hidden_size // num_heads)
        bias: Whether to use bias in linear projections
        rope_base: Base for RoPE
        rope_traditional: Whether to use traditional RoPE
        rope_scaling: Optional RoPE scaling configuration
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        bias: bool = False,
        rope_base: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, float | str] | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        # Q, K, V projections (K and V use single head)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

        # Rotary Position Embedding
        self.rope = initialize_rope(
            self.head_dim,
            base=rope_base,
            traditional=rope_traditional,
            scaling_config=rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        """
        Apply multi-query attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            mask: Optional attention mask
            cache: Optional KV cache

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        B, L, D = x.shape

        # Project to Q, K, V
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape for multi-head attention (single KV head)
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, 1, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, 1, -1).transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Scaled dot-product attention
        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask, cache=cache
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class CrossAttention(nn.Module):
    """
    Cross-attention for encoder-decoder models.

    Unlike self-attention, cross-attention computes queries from the decoder
    and keys/values from the encoder, enabling the decoder to attend to
    encoder outputs.

    Args:
        hidden_size: Dimension of decoder hidden states
        encoder_hidden_size: Dimension of encoder hidden states
        num_heads: Number of attention heads
        head_dim: Dimension of each head (default: hidden_size // num_heads)
        bias: Whether to use bias in linear projections
    """

    def __init__(
        self,
        hidden_size: int,
        encoder_hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        # Query projection (from decoder)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)

        # Key/Value projections (from encoder)
        self.k_proj = nn.Linear(encoder_hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(encoder_hidden_size, num_heads * self.head_dim, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

    def __call__(
        self,
        x: mx.array,
        encoder_output: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Apply cross-attention.

        Args:
            x: Decoder hidden states [batch, dec_seq_len, hidden_size]
            encoder_output: Encoder outputs [batch, enc_seq_len, encoder_hidden_size]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, dec_seq_len, hidden_size]
        """
        B, L_dec, D = x.shape
        L_enc = encoder_output.shape[1]

        # Queries from decoder
        queries = self.q_proj(x)

        # Keys and values from encoder
        keys = self.k_proj(encoder_output)
        values = self.v_proj(encoder_output)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L_dec, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L_enc, self.num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L_enc, self.num_heads, -1).transpose(0, 2, 1, 3)

        # Scaled dot-product attention (no RoPE for cross-attention)
        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L_dec, -1)
        return self.o_proj(output)


class SlidingWindowAttention(nn.Module):
    """
    Local attention with sliding window.

    Restricts attention to a local window around each token, reducing
    computational complexity from O(n^2) to O(n*w) where w is the window size.

    Args:
        hidden_size: Dimension of input/output
        num_heads: Number of attention heads
        window_size: Size of the sliding window
        head_dim: Dimension of each head (default: hidden_size // num_heads)
        bias: Whether to use bias in linear projections
        rope_base: Base for RoPE
        rope_traditional: Whether to use traditional RoPE
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int,
        head_dim: int | None = None,
        bias: bool = False,
        rope_base: float = 10000.0,
        rope_traditional: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = head_dim or hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

        # Rotary Position Embedding
        self.rope = nn.RoPE(self.head_dim, traditional=rope_traditional, base=rope_base)

    def __call__(
        self,
        x: mx.array,
        cache: Any | None = None,
    ) -> mx.array:
        """
        Apply sliding window attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            cache: Optional KV cache

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        B, L, D = x.shape

        # Project to Q, K, V
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Create sliding window mask
        mask = create_causal_mask(L, window_size=self.window_size)

        # Scaled dot-product attention with sliding window
        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask, cache=cache
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)
