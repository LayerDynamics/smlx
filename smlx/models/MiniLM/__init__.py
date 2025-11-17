#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
MiniLM: Ultra-lightweight sentence transformer for text embeddings.

A family of BERT-based models (22-120MB) for generating semantic embeddings.

Example:
    >>> from smlx.models.MiniLM import load, encode
    >>>
    >>> # Load model
    >>> model, tokenizer = load("all-MiniLM-L6-v2")
    >>>
    >>> # Encode sentences
    >>> sentences = ["Hello world", "Hi there"]
    >>> embeddings = encode(model, tokenizer, sentences)
    >>> embeddings.shape
    (2, 384)
    >>>
    >>> # Compute similarity
    >>> from smlx.models.MiniLM import cosine_similarity
    >>> sim = cosine_similarity(embeddings[:1], embeddings[1:])
    >>> sim[0, 0]  # Similarity score
"""

from .config import ModelConfig, DEFAULT_CONFIG_L6, DEFAULT_CONFIG_L12
from .encode import encode, encode_single, cosine_similarity
from .loader import load, get_model_path
from .model import MiniLM, mean_pooling, normalize_embeddings

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "ModelConfig",
    "DEFAULT_CONFIG_L6",
    "DEFAULT_CONFIG_L12",
    # Model
    "MiniLM",
    # Loading
    "load",
    "get_model_path",
    # Encoding
    "encode",
    "encode_single",
    "cosine_similarity",
    # Utilities
    "mean_pooling",
    "normalize_embeddings",
]
