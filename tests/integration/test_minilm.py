#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for MiniLM text embedding model.

Tests sentence encoding, similarity computation, batch processing.

Run with:
    python -m pytest tests/integration/test_minilm.py -v
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
def minilm_model():
    """
    Load MiniLM model once for all tests.

    Memory Requirements:
    - Model size: ~100MB
    - Peak memory: ~200MB with activations
    """
    from smlx.models.MiniLM import load

    model, tokenizer = load("all-MiniLM-L6-v2")

    yield model, tokenizer

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up MiniLM model...")
    del model
    del tokenizer
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print("Cleanup complete. Memory freed.")


def test_model_loading(minilm_model):
    """Test that MiniLM model loads successfully."""
    model, tokenizer = minilm_model

    assert model is not None, "Model should not be None"
    assert tokenizer is not None, "Tokenizer should not be None"


def test_single_sentence_encoding(minilm_model):
    """Test encoding a single sentence."""
    from smlx.models.MiniLM import encode

    model, tokenizer = minilm_model

    # Encode single sentence
    sentence = "Hello world"
    embedding = encode(model, tokenizer, [sentence])

    assert embedding is not None, "Embedding should not be None"
    assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
    assert len(embedding.shape) == 2, "Embedding should be 2D"
    assert embedding.shape[0] == 1, "Should have 1 embedding"
    assert embedding.shape[1] == 384, "Should have 384 dimensions"


def test_multiple_sentence_encoding(minilm_model):
    """Test encoding multiple sentences."""
    from smlx.models.MiniLM import encode

    model, tokenizer = minilm_model

    # Encode multiple sentences
    sentences = [
        "Hello world",
        "How are you?",
        "This is a test",
    ]
    embeddings = encode(model, tokenizer, sentences)

    assert embeddings is not None, "Embeddings should not be None"
    assert embeddings.shape[0] == len(sentences), "Should have embedding for each sentence"
    assert embeddings.shape[1] == 384, "Should have 384 dimensions"


def test_encode_single(minilm_model):
    """Test dedicated single sentence encoding."""
    from smlx.models.MiniLM import encode_single

    model, tokenizer = minilm_model

    # Encode single sentence
    sentence = "Test sentence"
    embedding = encode_single(model, tokenizer, sentence)

    assert embedding is not None, "Embedding should not be None"
    assert isinstance(embedding, np.ndarray), "Should be numpy array"
    assert len(embedding.shape) == 1, "Should be 1D for single sentence"
    assert embedding.shape[0] == 384, "Should have 384 dimensions"


def test_cosine_similarity(minilm_model):
    """Test cosine similarity computation."""
    from smlx.models.MiniLM import encode, cosine_similarity

    model, tokenizer = minilm_model

    # Encode two sentences
    sentences = [
        "The cat sits on the mat",
        "A feline rests on a rug",
    ]
    embeddings = encode(model, tokenizer, sentences)

    # Compute similarity
    sim = cosine_similarity(embeddings[:1], embeddings[1:])

    assert sim is not None, "Similarity should not be None"
    assert isinstance(sim, np.ndarray), "Should be numpy array"
    assert -1.0 <= sim[0, 0] <= 1.0, "Similarity should be between -1 and 1"


def test_similar_sentences_high_similarity(minilm_model):
    """Test that similar sentences have high similarity."""
    from smlx.models.MiniLM import encode, cosine_similarity

    model, tokenizer = minilm_model

    # Very similar sentences
    sentences = [
        "Python is a programming language",
        "Python is a programming language",
    ]
    embeddings = encode(model, tokenizer, sentences)

    # Compute similarity
    sim = cosine_similarity(embeddings[:1], embeddings[1:])[0, 0]

    # Should be very high (close to 1.0)
    assert sim > 0.95, "Identical sentences should have very high similarity"


def test_dissimilar_sentences_low_similarity(minilm_model):
    """Test that dissimilar sentences have low similarity."""
    from smlx.models.MiniLM import encode, cosine_similarity

    model, tokenizer = minilm_model

    # Completely different sentences
    sentences = [
        "The weather is nice today",
        "Machine learning algorithms process data",
    ]
    embeddings = encode(model, tokenizer, sentences)

    # Compute similarity
    sim = cosine_similarity(embeddings[:1], embeddings[1:])[0, 0]

    # Should be lower (different topics)
    assert sim < 0.8, "Dissimilar sentences should have lower similarity"


def test_mean_pooling():
    """Test mean pooling function."""
    import mlx.core as mx
    from smlx.models.MiniLM import mean_pooling

    # Create dummy token embeddings and attention mask
    token_embeddings = np.random.randn(1, 10, 384).astype(np.float32)
    attention_mask = np.ones((1, 10), dtype=np.float32)

    # Convert to MLX arrays
    token_embeddings_mx = mx.array(token_embeddings)
    attention_mask_mx = mx.array(attention_mask)

    # Pool
    pooled = mean_pooling(token_embeddings_mx, attention_mask_mx)

    assert pooled is not None, "Pooled embeddings should not be None"
    assert pooled.shape == (1, 384), "Should have shape (1, 384)"


def test_normalize_embeddings():
    """Test embedding normalization."""
    import mlx.core as mx
    from smlx.models.MiniLM import normalize_embeddings

    # Create dummy embeddings
    embeddings = np.random.randn(3, 384).astype(np.float32)

    # Convert to MLX array
    embeddings_mx = mx.array(embeddings)

    # Normalize
    normalized = normalize_embeddings(embeddings_mx)

    assert normalized is not None, "Normalized embeddings should not be None"
    assert normalized.shape == embeddings_mx.shape, "Shape should be preserved"

    # Convert back to numpy for checking
    normalized_np = np.array(normalized)

    # Check L2 norm is 1.0 for each embedding
    for i in range(len(normalized_np)):
        norm = np.linalg.norm(normalized_np[i])
        assert np.abs(norm - 1.0) < 1e-5, f"Embedding {i} should have L2 norm of 1.0"


def test_batch_encoding(minilm_model):
    """Test batch encoding with different batch sizes."""
    from smlx.models.MiniLM import encode

    model, tokenizer = minilm_model

    # Create many sentences
    sentences = [f"This is sentence number {i}" for i in range(10)]

    # Encode with different batch sizes
    for batch_size in [1, 4, 8]:
        embeddings = encode(model, tokenizer, sentences, batch_size=batch_size)

        assert embeddings.shape[0] == len(sentences), "Should encode all sentences"
        assert embeddings.shape[1] == 384, "Should have 384 dimensions"


def test_max_length_parameter(minilm_model):
    """Test max_length parameter."""
    from smlx.models.MiniLM import encode

    model, tokenizer = minilm_model

    # Long sentence
    long_sentence = "This is a very long sentence. " * 50

    # Encode with different max lengths
    for max_length in [128, 256]:
        embeddings = encode(model, tokenizer, [long_sentence], max_length=max_length)

        assert embeddings is not None, f"Should work with max_length {max_length}"


def test_empty_string(minilm_model):
    """Test encoding empty string."""
    from smlx.models.MiniLM import encode

    model, tokenizer = minilm_model

    # Empty string should be handled
    try:
        embeddings = encode(model, tokenizer, [""])
        assert embeddings is not None, "Should handle empty string"
    except (ValueError, RuntimeError):
        # If it raises an error, that's also acceptable
        assert True


def test_special_characters(minilm_model):
    """Test encoding text with special characters."""
    from smlx.models.MiniLM import encode

    model, tokenizer = minilm_model

    # Text with special characters
    texts = [
        "Hello, world!",
        "What's happening?",
        "Email: test@example.com",
        "Price: $99.99",
    ]

    embeddings = encode(model, tokenizer, texts)

    assert embeddings is not None, "Should handle special characters"
    assert embeddings.shape[0] == len(texts), "Should encode all texts"


def test_multilingual_text(minilm_model):
    """Test encoding multilingual text."""
    from smlx.models.MiniLM import encode

    model, tokenizer = minilm_model

    # Different languages (model may not perform well on all)
    texts = [
        "Hello world",
        "Bonjour le monde",
        "Hola mundo",
    ]

    embeddings = encode(model, tokenizer, texts)

    assert embeddings is not None, "Should handle multilingual text"


def test_semantic_search(minilm_model):
    """Test semantic search use case."""
    from smlx.models.MiniLM import encode, cosine_similarity

    model, tokenizer = minilm_model

    # Corpus
    documents = [
        "Machine learning is a subset of AI",
        "Python is a programming language",
        "The weather is nice today",
    ]

    # Query
    query = "What is artificial intelligence?"

    # Encode
    doc_embeddings = encode(model, tokenizer, documents)
    query_embedding = encode(model, tokenizer, [query])

    # Find most similar
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # First document should be most similar
    best_idx = np.argmax(similarities)
    assert best_idx == 0, "AI-related document should be most similar to AI query"


def test_model_config(minilm_model):
    """Test model configuration."""
    from smlx.models.MiniLM import ModelConfig

    model, tokenizer = minilm_model

    # Model should have config
    assert hasattr(model, "config"), "Model should have config"


def test_default_configs():
    """Test default configurations."""
    from smlx.models.MiniLM import DEFAULT_CONFIG_L6, DEFAULT_CONFIG_L12

    assert DEFAULT_CONFIG_L6 is not None, "L6 config should exist"
    assert DEFAULT_CONFIG_L12 is not None, "L12 config should exist"


def test_embedding_normalization(minilm_model):
    """Test that embeddings are normalized by default."""
    from smlx.models.MiniLM import encode

    model, tokenizer = minilm_model

    # Encode with normalization (default)
    embeddings = encode(model, tokenizer, ["Test"], normalize=True)

    # Check L2 norm is 1.0
    norm = np.linalg.norm(embeddings[0])
    assert np.abs(norm - 1.0) < 1e-5, "Embedding should be normalized"


def test_embedding_without_normalization(minilm_model):
    """Test encoding without normalization."""
    from smlx.models.MiniLM import encode

    model, tokenizer = minilm_model

    # Encode without normalization
    embeddings = encode(model, tokenizer, ["Test"], normalize=False)

    assert embeddings is not None, "Should work without normalization"


def test_duplicate_detection(minilm_model):
    """Test duplicate detection use case."""
    from smlx.models.MiniLM import encode, cosine_similarity

    model, tokenizer = minilm_model

    # Texts with duplicates
    texts = [
        "Hello world",
        "Hi there",
        "Hello world",  # Duplicate of first
    ]

    embeddings = encode(model, tokenizer, texts)

    # Compute all similarities
    similarities = cosine_similarity(embeddings, embeddings)

    # Similarity between first and third should be very high
    dup_sim = similarities[0, 2]
    assert dup_sim > 0.95, "Duplicate texts should have very high similarity"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
