# Copyright © 2025 SMLX Project
# Adapted from Apple's MLX framework implementation

"""
SmolLM2-135M model implementation.

This module implements SmolLM2-135M-Instruct, which uses the SmolLM3 architecture.
Despite the naming confusion, SmolLM2-135M models use the SmolLM3 architecture with
NoPE (No Positional Encoding) - selective RoPE disabling on every 4th layer.

Architecture:
- 135M parameters
- 576 hidden size
- 30 layers
- 9 attention heads (3 KV heads)
- 49,152 vocab size
- Based on Llama with NoPE modifications
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from smlx.utils.config import BaseModelArgs

# ============================================================================
# Utilities
# ============================================================================


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: int | None = None,
):
    """Create a causal attention mask."""
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    return mask


def create_attention_mask(h, cache=None, window_size: int | None = None):
    """Create attention mask for the given input."""
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, window_size=window_size)
    if N == 1:
        return None
    if window_size and N > window_size:
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: mx.array | str | None,
) -> mx.array:
    """Scaled dot-product attention with optional KV cache quantization."""
    # Use MLX's fast scaled_dot_product_attention
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
    )


# ============================================================================
# RoPE (Rotary Position Embedding) Utilities
# ============================================================================


def initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: dict | None = None,
    max_position_embeddings: int | None = None,
):
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
        # Additional RoPE types (llama3, yarn, longrope) can be added later
        raise ValueError(f"Unsupported RoPE type {rope_type}")


# ============================================================================
# Model Configuration
# ============================================================================


@dataclass
class ModelArgs(BaseModelArgs):
    """
    Model configuration for SmolLM2-135M (SmolLM3 architecture).

    Attributes:
        model_type: Model identifier (e.g., "smollm3")
        hidden_size: Dimension of hidden layers
        num_hidden_layers: Number of transformer layers
        intermediate_size: Dimension of MLP hidden layer
        num_attention_heads: Number of attention heads
        rms_norm_eps: Epsilon for RMS normalization
        vocab_size: Size of vocabulary
        head_dim: Dimension of each attention head (computed if None)
        max_position_embeddings: Maximum sequence length
        num_key_value_heads: Number of KV heads for GQA
        attention_bias: Whether to use bias in attention projections
        mlp_bias: Whether to use bias in MLP layers
        rope_theta: Base for RoPE
        rope_traditional: Whether to use traditional RoPE
        rope_scaling: Optional RoPE scaling configuration
        tie_word_embeddings: Whether to tie input/output embeddings
        layer_types: Layer type for each layer (e.g., ["full_attention"] * num_layers)
        sliding_window: Optional sliding window size
        no_rope_layer_interval: Interval for disabling RoPE (NoPE feature)
        no_rope_layers: Binary list indicating which layers have RoPE disabled
    """

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: int | None = None
    max_position_embeddings: int | None = None
    num_key_value_heads: int | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: dict[str, float | str] | None = None
    tie_word_embeddings: bool = True
    layer_types: list[str] | None = None
    sliding_window: int | None = None
    # SmolLM3-specific: NoPE (No Positional Encoding)
    no_rope_layer_interval: int = 4
    no_rope_layers: list[int] | None = None

    def __post_init__(self):
        """Post-initialization to set default values and validate configuration."""
        # Set default num_key_value_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # Set default layer types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        # Configure NoPE layers (SmolLM3 feature)
        # Every 4th layer has RoPE disabled by default
        if self.no_rope_layers is None:
            self.no_rope_layers = [
                int((i + 1) % self.no_rope_layer_interval != 0)
                for i in range(self.num_hidden_layers)
            ]
        elif len(self.no_rope_layers) != self.num_hidden_layers:
            raise ValueError(
                f"`no_rope_layers` length ({len(self.no_rope_layers)}) "
                f"must match num_hidden_layers ({self.num_hidden_layers})"
            )


# ============================================================================
# Model Components
# ============================================================================


class Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) support.

    Uses RoPE for positional encoding and supports KV caching for efficient generation.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        # num_key_value_heads is set in __post_init__, so it's guaranteed to be not None
        self.n_kv_heads = n_kv_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads is not None
            else args.num_attention_heads
        )

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        attention_bias = args.attention_bias

        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        # Rotary Position Embedding (can be replaced with NoPE for some layers)
        self.rope: nn.RoPE | NoPE = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, D = x.shape

        # Project to Q, K, V
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """
    MLP block with SiLU activation (SwiGLU-style).

    Uses the gated linear unit pattern: down(silu(gate(x)) * up(x))
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        mlp_bias = args.mlp_bias

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """
    Transformer decoder block with pre-normalization.

    Architecture:
        x -> LayerNorm -> Attention -> Add -> LayerNorm -> MLP -> Add
    """

    def __init__(self, args: ModelArgs, use_sliding: bool = False):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.use_sliding = use_sliding
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        # Pre-norm attention with residual
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        # Pre-norm MLP with residual
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LlamaModel(nn.Module):
    """
    Base Llama model (decoder-only transformer).

    This is the core transformer without the LM head.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        # layer_types is set in __post_init__, guaranteed to be not None
        self.layer_types = (
            args.layer_types
            if args.layer_types is not None
            else ["full_attention"] * args.num_hidden_layers
        )
        self.sliding_window = args.sliding_window
        assert self.vocab_size > 0

        # Token embeddings
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        # Transformer layers
        self.layers = [
            TransformerBlock(args=args, use_sliding=layer_type == "sliding_attention")
            for layer_type in self.layer_types
        ]

        # Final layer norm
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Find indices for full attention and sliding window attention
        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for e, layer in enumerate(self.layers):
            if layer.use_sliding:
                self.swa_idx = e
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ):
        # Get input embeddings
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        # Initialize cache if not provided
        if cache is None:
            cache = [None] * len(self.layers)

        # Create attention masks
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = None
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        # Forward through transformer layers
        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.use_sliding and swa_mask is not None else fa_mask
            h = layer(h, mask, cache=c)

        # Final layer norm
        return self.norm(h)


class NoPE(nn.Module):
    """
    No-op module used to disable rotary embeddings in selected layers.

    This is a key feature of SmolLM3 architecture - selective RoPE disabling.
    """

    def __call__(self, x, offset: int = 0):
        return x


class Model(nn.Module):
    """
    SmolLM2-135M complete model (uses SmolLM3 architecture).

    This is a wrapper around LlamaModel that:
    1. Adds the language modeling head
    2. Implements NoPE (No Positional Encoding) for selected layers
    3. Handles weight tying between input embeddings and output head

    Architecture: Llama-based decoder-only transformer with:
    - 135M parameters
    - 30 layers with selective RoPE (every 4th layer has no RoPE)
    - Grouped Query Attention (9 heads, 3 KV heads)
    - SwiGLU MLP
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type: str = args.model_type

        # Core transformer model
        self.model = LlamaModel(args)

        # Language modeling head (unless tied with embeddings)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        # Disable RoPE for specified layers (NoPE feature)
        if args.no_rope_layers is not None:
            for idx, use_rope in enumerate(args.no_rope_layers):
                if not use_rope:
                    # NoPE is a no-op RoPE replacement
                    self.model.layers[idx].self_attn.rope = NoPE()  # type: ignore

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ):
        """
        Forward pass.

        Args:
            inputs: Token IDs [batch, seq_len]
            cache: Optional KV cache for generation
            input_embeddings: Optional pre-computed embeddings

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Forward through transformer
        out = self.model(inputs, cache, input_embeddings)

        # Project to vocabulary
        if self.args.tie_word_embeddings:
            # Use embedding weights as output projection
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)

        return out

    @property
    def layers(self):
        """Access to transformer layers for inspection/modification."""
        return self.model.layers

    def sanitize(self, weights: dict):
        """
        Remove unnecessary weights from checkpoint.

        This is used when loading from HuggingFace checkpoints which may
        contain unused keys like rotary_emb.inv_freq.

        Args:
            weights: Dictionary of weights

        Returns:
            Sanitized weights dictionary
        """
        # Remove precomputed rotary frequencies (not needed in MLX)
        weights = {k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k}

        # Remove LM head if using tied embeddings
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        return weights
