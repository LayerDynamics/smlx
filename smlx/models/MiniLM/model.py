#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
MiniLM sentence transformer model implementation.

Based on BERT architecture with mean pooling for sentence embeddings.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with post-normalization (BERT-style)."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: int,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.ln1 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)
        self.gelu = nn.GELU()

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)

        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(
        self,
        num_layers: int,
        dims: int,
        num_heads: int,
        mlp_dims: int,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(dims, num_heads, mlp_dims, layer_norm_eps)
            for _ in range(num_layers)
        ]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BertEmbeddings(nn.Module):
    """BERT embeddings: word + position + token_type."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self, input_ids: mx.array, token_type_ids: Optional[mx.array] = None
    ) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )

        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)

        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        return self.norm(embeddings)


class MiniLM(nn.Module):
    """MiniLM sentence transformer model.

    BERT-based encoder with mean pooling and L2 normalization
    for generating sentence embeddings.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.hidden_size,
            num_heads=config.num_attention_heads,
            mlp_dims=config.intermediate_size,
            layer_norm_eps=config.layer_norm_eps,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass returning token embeddings.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs [batch_size, seq_length]

        Returns:
            Token embeddings [batch_size, seq_length, hidden_size]
        """
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            # Convert 0's to -infs, 1's to 0's, and make it broadcastable
            attention_mask = mx.log(attention_mask.astype(mx.float32))
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        y = self.encoder(x, attention_mask)
        return y

    @staticmethod
    def sanitize(weights):
        """Sanitize weights from PyTorch/HuggingFace format to MLX format.

        Handles layer name mapping and parameter reshaping.

        HuggingFace BERT structure -> MLX MiniLM structure:
        - bert.embeddings.LayerNorm -> embeddings.norm
        - bert.encoder.layer.N -> encoder.layers.N
        - attention.self.query/key/value -> attention.query_proj/key_proj/value_proj
        - attention.output.dense -> attention.out_proj
        - attention.output.LayerNorm -> ln1
        - intermediate.dense -> linear1
        - output.dense -> linear2
        - output.LayerNorm -> ln2
        """
        sanitized_weights = {}

        for k, v in weights.items():
            # Skip pooler (not needed for sentence transformers)
            if "pooler" in k:
                continue

            # Skip position_ids (generated dynamically in MLX)
            if "position_ids" in k:
                continue

            # Remove 'bert.' prefix
            k = k.replace("bert.", "")

            # Map embeddings LayerNorm
            if "embeddings.LayerNorm" in k:
                k = k.replace("embeddings.LayerNorm", "embeddings.norm")

            # Map encoder layers
            if "encoder.layer." in k:
                k = k.replace("encoder.layer.", "encoder.layers.")

                # Map attention sublayers
                if ".attention.self.query" in k:
                    k = k.replace(".attention.self.query", ".attention.query_proj")
                elif ".attention.self.key" in k:
                    k = k.replace(".attention.self.key", ".attention.key_proj")
                elif ".attention.self.value" in k:
                    k = k.replace(".attention.self.value", ".attention.value_proj")
                elif ".attention.output.dense" in k:
                    k = k.replace(".attention.output.dense", ".attention.out_proj")
                elif ".attention.output.LayerNorm" in k:
                    k = k.replace(".attention.output.LayerNorm", ".ln1")

                # Map feed-forward sublayers
                elif ".intermediate.dense" in k:
                    k = k.replace(".intermediate.dense", ".linear1")
                elif ".output.dense" in k:
                    k = k.replace(".output.dense", ".linear2")
                elif ".output.LayerNorm" in k:
                    k = k.replace(".output.LayerNorm", ".ln2")

            sanitized_weights[k] = v

        return sanitized_weights


def mean_pooling(
    token_embeddings, attention_mask
):
    """Mean pooling over token embeddings.

    Args:
        token_embeddings: Token embeddings [batch_size, seq_length, hidden_size]
                         Can be numpy array or MLX array
        attention_mask: Attention mask [batch_size, seq_length]
                       Can be numpy array or MLX array

    Returns:
        Sentence embeddings [batch_size, hidden_size]
        Returns numpy array if inputs are numpy, MLX array otherwise
    """
    import numpy as np

    # Check if inputs are numpy arrays
    is_numpy = isinstance(token_embeddings, np.ndarray)

    # Convert to MLX arrays if needed
    if is_numpy:
        token_embeddings = mx.array(token_embeddings)
        attention_mask = mx.array(attention_mask)

    # Expand mask to match embedding dimensions
    input_mask_expanded = mx.expand_dims(attention_mask, -1).astype(
        token_embeddings.dtype
    )

    # Sum embeddings weighted by mask
    sum_embeddings = mx.sum(token_embeddings * input_mask_expanded, 1)

    # Sum of mask values
    sum_mask = mx.clip(mx.sum(input_mask_expanded, 1), a_min=1e-9, a_max=None)

    # Mean pooling
    result = sum_embeddings / sum_mask

    # Convert back to numpy if inputs were numpy
    if is_numpy:
        return np.array(result)

    return result


def normalize_embeddings(embeddings):
    """L2 normalize embeddings.

    Args:
        embeddings: Embeddings [batch_size, hidden_size]
                   Can be numpy array or MLX array

    Returns:
        Normalized embeddings [batch_size, hidden_size]
        Returns numpy array if input is numpy, MLX array otherwise
    """
    import numpy as np

    # Check if input is numpy array
    is_numpy = isinstance(embeddings, np.ndarray)

    # Convert to MLX array if needed
    if is_numpy:
        embeddings = mx.array(embeddings)

    # Compute L2 norm
    norms = mx.sqrt(mx.sum(embeddings ** 2, -1, True))
    norms = mx.clip(norms, a_min=1e-12, a_max=None)  # Avoid division by zero

    # Normalize
    result = embeddings / norms

    # Convert back to numpy if input was numpy
    if is_numpy:
        return np.array(result)

    return result


__all__ = ["MiniLM", "mean_pooling", "normalize_embeddings"]
