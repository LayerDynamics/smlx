#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
smolGenCad: World's Smallest CAD Generation Model.

This module implements smolGenCad, a text-to-CAD generation model with
encoder-decoder transformer architecture.

Total parameters: ~158M (135M encoder + 23M decoder)

Architecture:
    - Text Encoder: SmolLM2-135M (encodes natural language descriptions)
    - CAD Decoder: 8-layer transformer (generates CAD command sequences)
    - CAD Head: Projects decoder output to CAD vocabulary

Based on Text2CAD architecture (NeurIPS 2024) but optimized for small scale.

Example:
    >>> from smlx.models.smolGenCad import SmolGenCad, SmolGenCadConfig
    >>> config = SmolGenCadConfig()
    >>> model = SmolGenCad(config)
    >>> # Generate CAD from text
    >>> output = model.generate("Create a cylinder with radius 5cm and height 10cm")
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .config import SmolGenCadConfig
from .decoder import CADDecoder
from .encoder import TextEncoder


class CADHead(nn.Module):
    """
    CAD vocabulary head.

    Projects decoder hidden states to CAD token logits.
    """

    def __init__(self, config: SmolGenCadConfig):
        """
        Initialize CAD head.

        Args:
            config: Model configuration
        """
        super().__init__()

        # Vocabulary size: commands + parameter bins + special tokens
        self.vocab_size = 1100  # Must match decoder vocab size

        # Projection layer
        self.lm_head = nn.Linear(config.decoder.hidden_size, self.vocab_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Project to vocabulary logits.

        Args:
            hidden_states: Decoder outputs [batch, seq_len, hidden_size]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        return self.lm_head(hidden_states)


class SmolGenCad(nn.Module):
    """
    smolGenCad: Text-to-CAD Generation Model.

    This is an encoder-decoder transformer that generates parametric CAD
    command sequences from natural language descriptions.

    Architecture:
        Input: "Create a cylinder with radius 5cm and height 10cm"
          ↓
        Text Encoder (SmolLM2-135M): Encodes text to embeddings
          ↓ [batch, text_len, 576]
        CAD Decoder (8-layer transformer): Generates CAD tokens autoregressively
          ↓ [batch, cad_len, 256]
        CAD Head: Projects to vocabulary
          ↓ [batch, cad_len, 1100]
        Output: CAD command sequence tokens

    Total Parameters: ~158M
        - Encoder: 135M
        - Decoder: 23M
        - Head: <1M

    Example:
        >>> config = SmolGenCadConfig()
        >>> model = SmolGenCad(config)
        >>>
        >>> # Forward pass (training)
        >>> text_ids = mx.array([[1, 123, 456, ...]]) # Tokenized text
        >>> cad_ids = mx.array([[2, 21, 100, ...]]) # Tokenized CAD sequence
        >>> logits = model(text_ids, cad_ids)
        >>>
        >>> # Generation (inference)
        >>> output = model.generate("Create a cube with side 10cm")
    """

    def __init__(self, config: SmolGenCadConfig):
        """
        Initialize smolGenCad model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Text encoder (SmolLM2-135M)
        self.encoder = TextEncoder(config.encoder)

        # CAD decoder (8-layer transformer)
        self.decoder = CADDecoder(config.decoder)

        # CAD vocabulary head
        self.cad_head = CADHead(config)

    def encode(
        self,
        input_ids: mx.array,
    ) -> mx.array:
        """
        Encode text to embeddings.

        Args:
            input_ids: Text token IDs [batch, text_len]

        Returns:
            Encoder outputs [batch, text_len, encoder_hidden_size]
        """
        return self.encoder(input_ids)

    def decode(
        self,
        cad_input_ids: mx.array,
        encoder_hidden_states: mx.array,
        cache=None,
    ) -> mx.array:
        """
        Decode CAD sequence given encoder outputs.

        Args:
            cad_input_ids: CAD token IDs [batch, cad_len]
            encoder_hidden_states: Encoder outputs [batch, text_len, encoder_hidden]
            cache: KV cache for generation

        Returns:
            Decoder hidden states [batch, cad_len, decoder_hidden_size]
        """
        return self.decoder(
            cad_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            cache=cache,
        )

    def __call__(
        self,
        input_ids: mx.array,
        cad_input_ids: mx.array,
        cache=None,
    ) -> mx.array:
        """
        Forward pass (for training).

        Args:
            input_ids: Text token IDs [batch, text_len]
            cad_input_ids: CAD token IDs [batch, cad_len]
            cache: Optional KV cache

        Returns:
            CAD token logits [batch, cad_len, vocab_size]
        """
        # Encode text
        encoder_outputs = self.encode(input_ids)

        # Decode CAD sequence
        decoder_outputs = self.decode(cad_input_ids, encoder_outputs, cache=cache)

        # Project to vocabulary
        logits = self.cad_head(decoder_outputs)

        return logits

    def generate_step(
        self,
        cad_input_ids: mx.array,
        encoder_hidden_states: mx.array,
        cache=None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
    ) -> tuple[mx.array, Any]:
        """
        Generate next CAD token.

        Args:
            cad_input_ids: Current CAD token IDs [batch, cad_len]
            encoder_hidden_states: Encoder outputs [batch, text_len, encoder_hidden]
            cache: KV cache
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling

        Returns:
            Tuple of (next_token_id, updated_cache)
        """
        # Decode current sequence
        decoder_outputs = self.decode(cad_input_ids, encoder_hidden_states, cache=cache)

        # Get logits for last token
        logits = self.cad_head(decoder_outputs[:, -1:, :])  # [batch, 1, vocab_size]
        logits = logits[:, 0, :]  # [batch, vocab_size]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Sample next token
        if top_k is not None:
            # Top-k sampling
            # Get top-k indices by sorting and taking top k
            sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]  # Descending
            top_k_indices = sorted_indices[:, :top_k]
            top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
            probs = mx.softmax(top_k_logits, axis=-1)
            # Sample from top-k
            sampled_idx = mx.random.categorical(probs[0:1])[0]
            next_token = top_k_indices[0, sampled_idx]
        elif top_p < 1.0:
            # Nucleus (top-p) sampling
            sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
            sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
            cumsum_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

            # Remove tokens with cumulative probability above threshold
            remove_mask = cumsum_probs > top_p
            sorted_logits = mx.where(remove_mask, -float("inf"), sorted_logits)

            # Sample
            probs = mx.softmax(sorted_logits, axis=-1)
            sorted_next_token = mx.random.categorical(probs[0:1])[0]
            next_token = sorted_indices[0, sorted_next_token]
        else:
            # Greedy or standard sampling
            if temperature == 0.0:
                # Greedy
                next_token = mx.argmax(logits[0], axis=-1)
            else:
                # Standard categorical sampling
                probs = mx.softmax(logits, axis=-1)
                next_token = mx.random.categorical(probs[0:1])[0]

        return next_token, cache

    @property
    def num_params(self) -> int:
        """Get total number of parameters."""
        total = 0
        for p in self.parameters().values():
            if hasattr(p, 'size'):
                total += p.size
            elif hasattr(p, 'shape'):
                import mlx.core as mx
                total += int(mx.prod(mx.array(p.shape)).item())
        return total

    @property
    def num_params_millions(self) -> float:
        """Get total parameters in millions."""
        return self.num_params / 1_000_000

    def sanitize(self, weights: dict) -> dict:
        """
        Remove unnecessary weights from checkpoint.

        Args:
            weights: Dictionary of weights

        Returns:
            Sanitized weights dictionary
        """
        # Remove rotary embedding inv_freq (not needed in MLX)
        weights = {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k
        }

        return weights
