#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for all-MiniLM-L6-v2 text embedding model.

Tests sentence encoding, similarity computation, semantic search.

Run with:
    python -m pytest tests/integration/test_all_minilm_l6_v2.py -v
"""

import gc

import mlx.core as mx
import pytest
import numpy as np

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
]


@pytest.fixture(scope="module")
def minilm_l6_v2_model():
    """
    Load all-MiniLM-L6-v2 model once for all tests.

    Memory Requirements:
    - Model size: ~100MB
    - Peak memory: ~200MB with activations
    """
    from smlx.models.all_MiniLM_L6_v2 import load

    model, tokenizer = load()

    yield model, tokenizer

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up all-MiniLM-L6-v2 model...")
    del model
    del tokenizer
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print("Cleanup complete. Memory freed.")


def test_model_loading(minilm_l6_v2_model):
    """Test that all-MiniLM-L6-v2 model loads successfully."""
    model, tokenizer = minilm_l6_v2_model

    assert model is not None, "Model should not be None"
    assert tokenizer is not None, "Tokenizer should not be None"


def test_single_sentence_encoding(minilm_l6_v2_model):
    """Test encoding a single sentence."""
    from smlx.models.all_MiniLM_L6_v2 import encode

    model, tokenizer = minilm_l6_v2_model

    # Encode single sentence
    sentence = "Hello world"
    embedding = encode(model, tokenizer, [sentence])

    assert embedding is not None, "Embedding should not be None"
    assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
    assert embedding.shape == (1, 384), "Should have shape (1, 384)"


def test_multiple_sentences(minilm_l6_v2_model):
    """Test encoding multiple sentences."""
    from smlx.models.all_MiniLM_L6_v2 import encode

    model, tokenizer = minilm_l6_v2_model

    sentences = [
        "The cat sits on the mat",
        "A feline rests on a rug",
        "Python is a programming language",
    ]
    embeddings = encode(model, tokenizer, sentences)

    assert embeddings.shape == (3, 384), "Should have shape (3, 384)"


def test_encode_single(minilm_l6_v2_model):
    """Test single sentence encoding."""
    from smlx.models.all_MiniLM_L6_v2 import encode_single

    model, tokenizer = minilm_l6_v2_model

    sentence = "Test sentence"
    embedding = encode_single(model, tokenizer, sentence)

    assert embedding.shape == (384,), "Should have shape (384,)"


def test_cosine_similarity(minilm_l6_v2_model):
    """Test cosine similarity computation."""
    from smlx.models.all_MiniLM_L6_v2 import encode, cosine_similarity

    model, tokenizer = minilm_l6_v2_model

    sentences = [
        "Machine learning is AI",
        "Artificial intelligence and ML",
    ]
    embeddings = encode(model, tokenizer, sentences)

    sim = cosine_similarity(embeddings[:1], embeddings[1:])

    assert -1.0 <= sim[0, 0] <= 1.0, "Similarity should be between -1 and 1"
    # Similar sentences should have higher similarity
    assert sim[0, 0] > 0.5, "Similar sentences should have high similarity"


def test_semantic_search(minilm_l6_v2_model):
    """Test semantic search use case."""
    from smlx.models.all_MiniLM_L6_v2 import encode, cosine_similarity

    model, tokenizer = minilm_l6_v2_model

    documents = [
        "Machine learning is a subset of AI",
        "Python is a programming language",
        "The weather is nice today",
    ]
    query = "What is artificial intelligence?"

    doc_embeddings = encode(model, tokenizer, documents)
    query_embedding = encode(model, tokenizer, [query])

    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    best_idx = np.argmax(similarities)
    assert best_idx == 0, "AI document should match AI query"


def test_batch_encoding(minilm_l6_v2_model):
    """Test batch encoding."""
    from smlx.models.all_MiniLM_L6_v2 import encode

    model, tokenizer = minilm_l6_v2_model

    sentences = [f"Sentence {i}" for i in range(10)]

    embeddings = encode(model, tokenizer, sentences, batch_size=4)

    assert embeddings.shape == (10, 384), "Should encode all sentences"


def test_normalized_embeddings(minilm_l6_v2_model):
    """Test that embeddings are normalized."""
    from smlx.models.all_MiniLM_L6_v2 import encode

    model, tokenizer = minilm_l6_v2_model

    embeddings = encode(model, tokenizer, ["Test"], normalize=True)

    norm = np.linalg.norm(embeddings[0])
    assert np.abs(norm - 1.0) < 1e-5, "Embedding should be normalized"


def test_identical_sentences(minilm_l6_v2_model):
    """Test identical sentences have high similarity."""
    from smlx.models.all_MiniLM_L6_v2 import encode, cosine_similarity

    model, tokenizer = minilm_l6_v2_model

    sentences = [
        "Python programming",
        "Python programming",
    ]
    embeddings = encode(model, tokenizer, sentences)

    sim = cosine_similarity(embeddings[:1], embeddings[1:])[0, 0]

    assert sim > 0.99, "Identical sentences should have very high similarity"


def test_different_topics(minilm_l6_v2_model):
    """Test sentences on different topics."""
    from smlx.models.all_MiniLM_L6_v2 import encode, cosine_similarity

    model, tokenizer = minilm_l6_v2_model

    sentences = [
        "The weather is sunny",
        "Machine learning algorithms",
    ]
    embeddings = encode(model, tokenizer, sentences)

    sim = cosine_similarity(embeddings[:1], embeddings[1:])[0, 0]

    # Different topics should have lower similarity
    assert sim < 0.7, "Different topics should have lower similarity"


def test_mean_pooling():
    """Test mean pooling utility."""
    from smlx.models.all_MiniLM_L6_v2 import mean_pooling

    token_embeddings = np.random.randn(1, 10, 384).astype(np.float32)
    attention_mask = np.ones((1, 10), dtype=np.float32)

    pooled = mean_pooling(token_embeddings, attention_mask)

    assert pooled.shape == (1, 384), "Pooled should have shape (1, 384)"


def test_normalize_embeddings():
    """Test normalization utility."""
    from smlx.models.all_MiniLM_L6_v2 import normalize_embeddings

    embeddings = np.random.randn(3, 384).astype(np.float32)

    normalized = normalize_embeddings(embeddings)

    for i in range(len(normalized)):
        norm = np.linalg.norm(normalized[i])
        assert np.abs(norm - 1.0) < 1e-5, f"Embedding {i} should be normalized"


def test_max_length(minilm_l6_v2_model):
    """Test max length parameter."""
    from smlx.models.all_MiniLM_L6_v2 import encode

    model, tokenizer = minilm_l6_v2_model

    long_text = "This is a long sentence. " * 100

    embeddings = encode(model, tokenizer, [long_text], max_length=256)

    assert embeddings is not None, "Should handle long text with truncation"


def test_special_characters(minilm_l6_v2_model):
    """Test text with special characters."""
    from smlx.models.all_MiniLM_L6_v2 import encode

    model, tokenizer = minilm_l6_v2_model

    texts = [
        "Email: test@example.com",
        "Price: $99.99",
        "Hello, world!",
    ]

    embeddings = encode(model, tokenizer, texts)

    assert embeddings.shape[0] == len(texts), "Should encode all texts"


def test_duplicate_detection(minilm_l6_v2_model):
    """Test duplicate detection."""
    from smlx.models.all_MiniLM_L6_v2 import encode, cosine_similarity

    model, tokenizer = minilm_l6_v2_model

    texts = [
        "Hello world",
        "Hi there",
        "Hello world",
    ]

    embeddings = encode(model, tokenizer, texts)
    similarities = cosine_similarity(embeddings, embeddings)

    dup_sim = similarities[0, 2]
    assert dup_sim > 0.95, "Duplicates should have very high similarity"


def test_model_config(minilm_l6_v2_model):
    """Test model configuration."""
    from smlx.models.all_MiniLM_L6_v2 import ModelConfig, DEFAULT_CONFIG_L6

    model, tokenizer = minilm_l6_v2_model

    assert hasattr(model, "config"), "Model should have config"
    assert DEFAULT_CONFIG_L6 is not None, "DEFAULT_CONFIG_L6 should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
