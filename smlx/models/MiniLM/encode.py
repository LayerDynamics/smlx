#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Text encoding utilities for MiniLM sentence transformers.

Provides high-level encode() function for generating sentence embeddings.
"""

from typing import List, Union

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
from transformers import PreTrainedTokenizerBase

from .model import MiniLM, mean_pooling, normalize_embeddings


def encode(
    model: MiniLM,
    tokenizer: PreTrainedTokenizerBase,
    sentences: Union[str, List[str]],
    batch_size: int = 32,
    max_length: int = 256,
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode sentences into embeddings.

    Args:
        model: MiniLM model
        tokenizer: Tokenizer
        sentences: Single sentence or list of sentences
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        normalize: Whether to L2 normalize embeddings
        show_progress: Show progress bar (requires tqdm)

    Returns:
        Embeddings as numpy array [num_sentences, embedding_dim]

    Example:
        >>> from smlx.models.MiniLM import load, encode
        >>> model, tokenizer = load("all-MiniLM-L6-v2")
        >>> sentences = ["Hello world", "Hi there"]
        >>> embeddings = encode(model, tokenizer, sentences)
        >>> embeddings.shape
        (2, 384)
    """
    # Handle single sentence
    if isinstance(sentences, str):
        sentences = [sentences]

    # Progress bar
    if show_progress:
        try:
            from tqdm import tqdm
            sentences = tqdm(sentences, desc="Encoding")
        except ImportError:
            print("Warning: tqdm not installed, progress bar disabled")

    all_embeddings = []

    # Process in batches
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        # Tokenize
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        # Convert to MLX arrays
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        # Get token embeddings from model
        token_embeddings = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Apply mean pooling
        sentence_embeddings = mean_pooling(token_embeddings, attention_mask)

        # Normalize if requested
        if normalize:
            sentence_embeddings = normalize_embeddings(sentence_embeddings)

        # Convert to numpy and store
        all_embeddings.append(np.array(sentence_embeddings))

    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)

    return embeddings


def encode_single(
    model: MiniLM,
    tokenizer: PreTrainedTokenizerBase,
    sentence: str,
    max_length: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a single sentence into an embedding.

    Args:
        model: MiniLM model
        tokenizer: Tokenizer
        sentence: Sentence to encode
        max_length: Maximum sequence length
        normalize: Whether to L2 normalize embedding

    Returns:
        Embedding as numpy array [embedding_dim]

    Example:
        >>> from smlx.models.MiniLM import load, encode_single
        >>> model, tokenizer = load()
        >>> embedding = encode_single(model, tokenizer, "Hello world")
        >>> embedding.shape
        (384,)
    """
    # Tokenize
    encoded = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )

    # Convert to MLX arrays
    input_ids = mx.array(encoded["input_ids"])
    attention_mask = mx.array(encoded["attention_mask"])

    # Get token embeddings
    token_embeddings = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Mean pooling
    sentence_embedding = mean_pooling(token_embeddings, attention_mask)

    # Normalize if requested
    if normalize:
        sentence_embedding = normalize_embeddings(sentence_embedding)

    # Convert to numpy and squeeze batch dimension
    return np.array(sentence_embedding).squeeze(0)


def cosine_similarity(
    embeddings1: Union[mx.array, NDArray],
    embeddings2: Union[mx.array, NDArray],
) -> Union[mx.array, NDArray]:
    """Compute cosine similarity between embeddings.

    Args:
        embeddings1: First set of embeddings [n, dim] (MLX or NumPy array)
        embeddings2: Second set of embeddings [m, dim] (MLX or NumPy array)

    Returns:
        Similarity matrix [n, m] (same type as input)

    Example:
        >>> from smlx.models.MiniLM import load, encode, cosine_similarity
        >>> model, tokenizer = load()
        >>> emb1 = encode(model, tokenizer, ["Hello"])
        >>> emb2 = encode(model, tokenizer, ["Hi", "Goodbye"])
        >>> sim = cosine_similarity(emb1, emb2)
        >>> sim.shape
        (1, 2)
    """
    # Check if inputs are MLX arrays
    is_mlx = isinstance(embeddings1, mx.array)

    if is_mlx:
        # Use MLX operations
        norm1 = mx.sqrt(mx.sum(embeddings1 * embeddings1, axis=-1, keepdims=True))
        norm2 = mx.sqrt(mx.sum(embeddings2 * embeddings2, axis=-1, keepdims=True))

        embeddings1 = embeddings1 / mx.maximum(norm1, 1e-12)
        embeddings2 = embeddings2 / mx.maximum(norm2, 1e-12)

        # Compute dot product
        return embeddings1 @ embeddings2.T
    else:
        # Use NumPy operations
        norm1 = np.linalg.norm(embeddings1, axis=-1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=-1, keepdims=True)

        embeddings1 = embeddings1 / np.maximum(norm1, 1e-12)
        embeddings2 = embeddings2 / np.maximum(norm2, 1e-12)

        # Compute dot product
        return embeddings1 @ embeddings2.T


__all__ = ["encode", "encode_single", "cosine_similarity"]
