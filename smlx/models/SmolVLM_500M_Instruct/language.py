#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Language Model for SmolVLM-256M-Instruct.

Reuses SmolLM2-135M architecture with modifications for multimodal inputs.
The language model accepts both token IDs and merged embeddings (text + image).
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# Reuse components from SmolLM2_135M
from ..SmolLM2_135M.model import (
    Attention,
    MLP,
    TransformerBlock,
    create_attention_mask,
)

from .config import TextConfig


@dataclass
class LanguageModelOutput:
    """Output from language model forward pass."""

    logits: mx.array
    hidden_states: Optional[mx.array] = None


class LanguageModel(nn.Module):
    """SmolLM2-based language model for VLM.

    Identical to SmolLM2-135M but supports multimodal inputs via
    the `inputs_embeds` parameter.

    Args:
        config: TextConfig with model hyperparameters
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        assert self.vocab_size > 0, "Vocab size must be positive"

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers (reuse SmolLM2 architecture)
        # Note: SmolLM2 uses ModelArgs, need to create compatible object
        from ..SmolLM2_135M.config import ModelArgs

        smollm_args = ModelArgs(
            model_type=config.model_type,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            num_hidden_layers=config.num_hidden_layers,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            rope_traditional=config.rope_traditional,
            max_position_embeddings=config.max_position_embeddings,
            tie_word_embeddings=config.tie_word_embeddings,
        )

        self.layers = [TransformerBlock(smollm_args, i) for i in range(config.num_hidden_layers)]

        # Final norm and projection
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> LanguageModelOutput:
        """Forward pass through language model.

        Args:
            inputs: Token IDs [B, seq_len] (for text-only)
            inputs_embeds: Pre-computed embeddings [B, seq_len, hidden_size] (for multimodal)
            mask: Attention mask
            cache: KV cache for generation

        Returns:
            LanguageModelOutput with logits
        """
        # Use either token embeddings or pre-computed embeddings
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds.astype(self.norm.weight.dtype)

        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        # Create attention mask if needed
        if mask is None:
            mask = create_attention_mask(h, cache[0] if cache else None)

        # Forward through transformer layers
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        # Final norm and projection
        logits = self.lm_head(self.norm(h))

        return LanguageModelOutput(logits=logits, hidden_states=h)

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Remove unused weights during loading.

        PyTorch models may have precomputed rotary frequencies that we don't need.
        """
        return {
            k: v
            for k, v in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in k and "position_ids" not in k
        }

    @property
    def head_dim(self):
        """Dimension of each attention head."""
        return self.config.hidden_size // self.config.num_attention_heads

    @property
    def n_kv_heads(self):
        """Number of key-value heads (for GQA)."""
        return self.config.num_key_value_heads


__all__ = ["LanguageModel", "LanguageModelOutput"]
