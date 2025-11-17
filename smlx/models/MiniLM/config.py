#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Configuration classes for MiniLM sentence embedding models.

MiniLM is a BERT-based sentence transformer for generating text embeddings.
"""

import inspect
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for MiniLM model."""

    # Architecture
    model_type: str = "bert"
    hidden_size: int = 384  # MiniLM-L6
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 1536
    hidden_act: str = "gelu"

    # Embeddings
    vocab_size: int = 30522
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    pad_token_id: int = 0

    # Regularization
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12

    # Sentence transformer specific
    pooling_mode_mean_tokens: bool = True
    pooling_mode_cls_token: bool = False
    pooling_mode_max_tokens: bool = False
    normalize_embeddings: bool = True

    @classmethod
    def from_dict(cls, params: Dict):
        """Create config from dictionary, filtering unknown params."""
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


# Default configuration for all-MiniLM-L6-v2
DEFAULT_CONFIG_L6 = ModelConfig(
    model_type="bert",
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=12,
    intermediate_size=1536,
    vocab_size=30522,
    max_position_embeddings=512,
    layer_norm_eps=1e-12,
    pooling_mode_mean_tokens=True,
    normalize_embeddings=True,
)

# Configuration for all-MiniLM-L12-v2
DEFAULT_CONFIG_L12 = ModelConfig(
    model_type="bert",
    hidden_size=384,
    num_hidden_layers=12,  # More layers
    num_attention_heads=12,
    intermediate_size=1536,
    vocab_size=30522,
    max_position_embeddings=512,
    layer_norm_eps=1e-12,
    pooling_mode_mean_tokens=True,
    normalize_embeddings=True,
)


__all__ = ["ModelConfig", "DEFAULT_CONFIG_L6", "DEFAULT_CONFIG_L12"]
