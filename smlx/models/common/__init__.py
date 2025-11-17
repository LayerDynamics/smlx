"""
Common model components for SMLX.

This module provides reusable building blocks that can be shared across
different model architectures, reducing code duplication and ensuring
consistent implementations.

Components:
- Attention mechanisms (multi-head, GQA, MQA, cross-attention, sliding window)
- MLP/FFN layers (SwiGLU, GeGLU, standard MLP, etc.)
- Switch layers (Mixture of Experts with routing)

Usage:
    from smlx.models.common import (
        MultiHeadAttention,
        GroupedQueryAttention,
        SwiGLU,
        SwitchLinear,
    )

    # Create a GQA layer with 9 query heads and 3 KV heads
    attention = GroupedQueryAttention(
        hidden_size=576,
        num_heads=9,
        num_kv_heads=3,
    )

    # Create a SwiGLU MLP
    mlp = SwiGLU(
        hidden_size=576,
        intermediate_size=1536,
    )

    # Create a switch layer for MoE
    switch = SwitchLinear(
        input_dims=576,
        output_dims=1536,
        num_experts=8,
    )
"""

# Attention mechanisms
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

# MLP/FFN layers
from smlx.models.common.mlp import (
    ExpertMLP,
    GeGLU,
    ParallelMLP,
    ReluSquared,
    StandardMLP,
    SwiGLU,
    create_mlp,
)

# Switch layers (Mixture of Experts)
from smlx.models.common.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    SwitchMLP,
)
from smlx.models.common.switch_layers import SwiGLU as SwitchSwiGLU

__all__ = [
    # Attention utilities
    "create_attention_mask",
    "create_causal_mask",
    "scaled_dot_product_attention",
    "initialize_rope",
    # Attention mechanisms
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    "CrossAttention",
    "SlidingWindowAttention",
    # MLP/FFN layers
    "SwiGLU",
    "GeGLU",
    "StandardMLP",
    "ReluSquared",
    "ExpertMLP",
    "ParallelMLP",
    "create_mlp",
    # Switch layers
    "SwitchLinear",
    "QuantizedSwitchLinear",
    "SwitchGLU",
    "SwitchMLP",
    "SwitchSwiGLU",
]
