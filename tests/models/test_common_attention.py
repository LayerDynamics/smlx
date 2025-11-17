"""
Tests for common attention mechanisms.
"""

import mlx.core as mx
import pytest

from smlx.models.common.attention import (
    CrossAttention,
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    SlidingWindowAttention,
    create_attention_mask,
    create_causal_mask,
    initialize_rope,
    scaled_dot_product_attention,
)


@pytest.mark.unit
def test_create_causal_mask():
    """Test causal mask creation."""
    # Basic causal mask
    mask = create_causal_mask(4)
    assert mask.shape == (4, 4)

    # With offset
    mask = create_causal_mask(4, offset=2)
    assert mask.shape == (4, 6)

    # With window size
    mask = create_causal_mask(8, window_size=4)
    assert mask.shape == (8, 8)


@pytest.mark.unit
def test_create_attention_mask():
    """Test attention mask creation."""
    # Single token - should return None
    x = mx.random.normal((1, 1, 64))
    mask = create_attention_mask(x)
    assert mask is None

    # Multiple tokens - should return "causal"
    x = mx.random.normal((1, 10, 64))
    mask = create_attention_mask(x)
    assert mask == "causal"

    # With sliding window
    x = mx.random.normal((1, 100, 64))
    mask = create_attention_mask(x, window_size=16)
    assert isinstance(mask, mx.array)


@pytest.mark.unit
def test_initialize_rope():
    """Test RoPE initialization."""
    # Default RoPE
    rope = initialize_rope(64)
    assert rope is not None

    # With linear scaling
    rope = initialize_rope(64, scaling_config={"type": "linear", "factor": 2.0})
    assert rope is not None

    # Traditional RoPE
    rope = initialize_rope(64, traditional=True)
    assert rope is not None


@pytest.mark.unit
def test_multi_head_attention():
    """Test MultiHeadAttention layer."""
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_heads = 8

    attn = MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = attn(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.unit
def test_grouped_query_attention():
    """Test GroupedQueryAttention layer."""
    batch_size = 2
    seq_len = 16
    hidden_size = 576
    num_heads = 9
    num_kv_heads = 3

    attn = GroupedQueryAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = attn(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.unit
def test_grouped_query_attention_invalid_config():
    """Test GQA with invalid configuration."""
    with pytest.raises(ValueError):
        # num_heads must be divisible by num_kv_heads
        GroupedQueryAttention(
            hidden_size=256,
            num_heads=7,
            num_kv_heads=3,
        )


@pytest.mark.unit
def test_multi_query_attention():
    """Test MultiQueryAttention layer."""
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_heads = 8

    attn = MultiQueryAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = attn(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.unit
def test_cross_attention():
    """Test CrossAttention layer."""
    batch_size = 2
    dec_seq_len = 10
    enc_seq_len = 20
    hidden_size = 256
    encoder_hidden_size = 512
    num_heads = 8

    attn = CrossAttention(
        hidden_size=hidden_size,
        encoder_hidden_size=encoder_hidden_size,
        num_heads=num_heads,
    )

    decoder_hidden = mx.random.normal((batch_size, dec_seq_len, hidden_size))
    encoder_output = mx.random.normal((batch_size, enc_seq_len, encoder_hidden_size))

    output = attn(decoder_hidden, encoder_output)

    assert output.shape == (batch_size, dec_seq_len, hidden_size)


@pytest.mark.unit
def test_sliding_window_attention():
    """Test SlidingWindowAttention layer."""
    batch_size = 2
    seq_len = 64
    hidden_size = 256
    num_heads = 8
    window_size = 16

    attn = SlidingWindowAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        window_size=window_size,
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = attn(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.unit
def test_scaled_dot_product_attention():
    """Test scaled dot-product attention."""
    batch_size = 2
    num_heads = 8
    seq_len = 16
    head_dim = 64

    queries = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

    scale = head_dim**-0.5

    output = scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask="causal"
    )

    assert output.shape == (batch_size, num_heads, seq_len, head_dim)


@pytest.mark.unit
def test_attention_with_different_head_dims():
    """Test attention with custom head dimension."""
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_heads = 8
    head_dim = 64

    attn = MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = attn(x)

    # Output should match input size
    assert output.shape == (batch_size, seq_len, hidden_size)

    # Check internal dimensions
    assert attn.head_dim == head_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
