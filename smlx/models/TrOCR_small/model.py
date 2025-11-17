#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
TrOCR Model Architecture.

Combines BEiT vision encoder with RoBERTa text decoder
for transformer-based OCR.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import TrOCRConfig, TrOCRDecoderConfig, TrOCRVisionConfig


class Attention(nn.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dims = dims
        self.scale = (dims // num_heads) ** -0.5

        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        encoder_hidden_states: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Query tensor (batch, seq_len, dims)
            mask: Attention mask
            encoder_hidden_states: For cross-attention

        Returns:
            Output tensor
        """
        B, L, D = x.shape
        head_dim = self.dims // self.num_heads

        # Self-attention or cross-attention
        queries = self.q_proj(x)

        if encoder_hidden_states is not None:
            # Cross-attention
            keys = self.k_proj(encoder_hidden_states)
            values = self.v_proj(encoder_hidden_states)
            # Get source sequence length from encoder_hidden_states
            S = encoder_hidden_states.shape[1]
        else:
            # Self-attention
            keys = self.k_proj(x)
            values = self.v_proj(x)
            S = L

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Attention scores
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = (attn_weights @ values).transpose(0, 2, 1, 3).reshape(B, L, D)
        out = self.out_proj(out)

        return out


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer for vision encoder."""

    def __init__(self, config: TrOCRVisionConfig):
        super().__init__()
        self.attention = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_probs_dropout_prob,
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        # Self-attention with residual
        x = x + self.attention(self.ln1(x))

        # MLP with residual
        x = x + self.mlp(self.ln2(x))

        return x


class VisionEncoder(nn.Module):
    """BEiT vision encoder.

    Extracts features from images using transformer architecture.
    """

    def __init__(self, config: TrOCRVisionConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        # Position embeddings
        num_patches = config.num_patches
        self.position_embeddings = mx.zeros((1, num_patches, config.hidden_size))

        # Transformer layers
        self.layers = [
            TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

        # Final layer norm
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Forward pass.

        Args:
            pixel_values: Images (batch, height, width, channels) in MLX NHWC format

        Returns:
            Encoded features (batch, num_patches, hidden_size)
        """
        # Patch embeddings
        x = self.patch_embed(pixel_values)  # (B, H, W, hidden_size)

        # Flatten patches
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)  # (B, num_patches, hidden_size)

        # Add position embeddings
        x = x + self.position_embeddings

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = self.ln(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer for text decoder."""

    def __init__(self, config: TrOCRDecoderConfig):
        super().__init__()

        # Self-attention
        self.self_attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_probs_dropout_prob,
        )

        # Cross-attention (encoder-decoder attention)
        self.cross_attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_probs_dropout_prob,
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array,
        causal_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Decoder input
            encoder_hidden_states: Encoder outputs
            causal_mask: Causal mask for autoregressive decoding

        Returns:
            Output tensor
        """
        # Self-attention with causal mask
        x = x + self.self_attn(self.ln1(x), mask=causal_mask)

        # Cross-attention to encoder outputs
        x = x + self.cross_attn(self.ln2(x), encoder_hidden_states=encoder_hidden_states)

        # MLP
        x = x + self.mlp(self.ln3(x))

        return x


class TextDecoder(nn.Module):
    """RoBERTa text decoder.

    Generates text autoregressively from vision encoder features.
    """

    def __init__(self, config: TrOCRDecoderConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Position embeddings
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Layer norm after embeddings (RoBERTa style)
        self.ln_embedding = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Transformer decoder layers
        self.layers = [
            TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

        # Final layer norm
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            encoder_hidden_states: Encoder outputs

        Returns:
            Hidden states (batch, seq_len, hidden_size)
        """
        B, L = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Position embeddings
        positions = mx.arange(L)[None, :]
        x = x + self.position_embedding(positions)

        # Layer norm after embeddings
        x = self.ln_embedding(x)

        # Causal mask for autoregressive decoding
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        causal_mask = mx.expand_dims(causal_mask, (0, 1))  # (1, 1, L, L)

        # Decoder layers
        for layer in self.layers:
            x = layer(x, encoder_hidden_states, causal_mask)

        # Final layer norm
        x = self.ln(x)

        return x


class TrOCR(nn.Module):
    """TrOCR model for OCR.

    Combines vision encoder (BEiT) and text decoder (RoBERTa)
    for end-to-end text recognition.
    """

    def __init__(self, config: TrOCRConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.encoder = VisionEncoder(config.vision_config)

        # Text decoder
        self.decoder = TextDecoder(config.decoder_config)

        # Language model head
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.decoder_config.vocab_size)

    def encode(self, pixel_values: mx.array) -> mx.array:
        """Encode image to features.

        Args:
            pixel_values: Images (batch, height, width, channels) in MLX NHWC format

        Returns:
            Encoder hidden states
        """
        return self.encoder(pixel_values)

    def decode(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        """Decode tokens given encoder features.

        Args:
            input_ids: Token IDs
            encoder_hidden_states: Encoder outputs

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        hidden_states = self.decoder(input_ids, encoder_hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def __call__(
        self,
        pixel_values: mx.array,
        input_ids: mx.array,
    ) -> mx.array:
        """Forward pass for training.

        Args:
            pixel_values: Images
            input_ids: Token IDs (for teacher forcing)

        Returns:
            Logits
        """
        encoder_hidden_states = self.encode(pixel_values)
        logits = self.decode(input_ids, encoder_hidden_states)
        return logits


__all__ = [
    "TrOCR",
    "VisionEncoder",
    "TextDecoder",
    "Attention",
]
