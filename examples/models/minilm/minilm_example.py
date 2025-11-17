#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
MiniLM Embedding Examples

Demonstrates text embedding capabilities:
- Basic sentence encoding
- Semantic search
- Text similarity comparison
- Document clustering
- Duplicate detection
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from smlx.models.MiniLM import load, encode, cosine_similarity
import numpy as np


def example_1_basic_encoding():
    """Example 1: Basic sentence encoding."""
    print("=" * 70)
    print("Example 1: Basic Sentence Encoding")
    print("=" * 70)
    print()

    # Load model
    print("Loading all-MiniLM-L6-v2 model...")
    model, tokenizer = load("all-MiniLM-L6-v2")

    # Encode sentences
    sentences = [
        "The cat sits on the mat",
        "A feline rests on a rug",
        "Python is a programming language",
    ]

    print("Encoding sentences...")
    embeddings = encode(model, tokenizer, sentences)

    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print()


def example_2_semantic_search():
    """Example 2: Semantic search over documents."""
    print("=" * 70)
    print("Example 2: Semantic Search")
    print("=" * 70)
    print()

    from smlx.models.MiniLM import load, encode, cosine_similarity

    model, tokenizer = load("all-MiniLM-L6-v2")

    # Corpus of documents
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "The weather is nice today with clear skies",
        "Deep learning uses neural networks with multiple layers",
        "JavaScript is commonly used for web development",
    ]

    # Query
    query = "What is AI and machine learning?"

    print(f"Query: {query}")
    print()
    print("Documents:")
    for i, doc in enumerate(documents):
        print(f"  {i + 1}. {doc}")
    print()

    # Encode
    print("Encoding documents and query...")
    doc_embeddings = encode(model, tokenizer, documents)
    query_embedding = encode(model, tokenizer, [query])

    # Find most similar
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Sort by similarity
    ranked_indices = np.argsort(similarities)[::-1]

    print("Results (ranked by similarity):")
    for rank, idx in enumerate(ranked_indices[:3], 1):
        print(f"  {rank}. [{similarities[idx]:.3f}] {documents[idx]}")
    print()


def example_3_text_similarity():
    """Example 3: Compare text similarity."""
    print("=" * 70)
    print("Example 3: Text Similarity Comparison")
    print("=" * 70)
    print()

    from smlx.models.MiniLM import load, encode, cosine_similarity

    model, tokenizer = load("all-MiniLM-L6-v2")

    # Text pairs to compare
    pairs = [
        ("The cat sits on the mat", "A feline rests on a rug"),
        ("I love programming", "I hate coding"),
        ("The weather is nice", "It's a beautiful day"),
    ]

    print("Comparing text pairs:")
    for text1, text2 in pairs:
        emb1 = encode(model, tokenizer, [text1])
        emb2 = encode(model, tokenizer, [text2])

        sim = cosine_similarity(emb1, emb2)[0, 0]

        print(f"\nText 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Similarity: {sim:.3f}")
    print()


def example_4_clustering():
    """Example 4: Document clustering."""
    print("=" * 70)
    print("Example 4: Document Clustering")
    print("=" * 70)
    print()

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("Note: This example requires scikit-learn")
        print("Install with: pip install scikit-learn")
        print()
        return

    from smlx.models.MiniLM import load, encode

    model, tokenizer = load("all-MiniLM-L6-v2")

    # Documents to cluster
    documents = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "The cat sat on the mat",
        "JavaScript is used for web development",
        "Deep learning is a subset of machine learning",
        "The dog played in the park",
        "Ruby is another programming language",
    ]

    print("Documents to cluster:")
    for i, doc in enumerate(documents):
        print(f"  {i + 1}. {doc}")
    print()

    # Encode
    print("Encoding documents...")
    embeddings = encode(model, tokenizer, documents)

    # Cluster
    print("Clustering into 3 groups...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    print("\nCluster assignments:")
    for cluster_id in range(3):
        print(f"\nCluster {cluster_id + 1}:")
        for i, cluster in enumerate(clusters):
            if cluster == cluster_id:
                print(f"  - {documents[i]}")
    print()


def example_5_duplicate_detection():
    """Example 5: Find duplicate or near-duplicate texts."""
    print("=" * 70)
    print("Example 5: Duplicate Detection")
    print("=" * 70)
    print()

    from smlx.models.MiniLM import load, encode, cosine_similarity

    model, tokenizer = load("all-MiniLM-L6-v2")

    # Texts with some duplicates
    texts = [
        "Hello world",
        "Hi there",
        "Hello world!",  # Near duplicate of first
        "Goodbye everyone",
        "Hi there",  # Exact duplicate of second
        "Good morning",
    ]

    print("Texts:")
    for i, text in enumerate(texts):
        print(f"  {i + 1}. {text}")
    print()

    # Encode
    print("Encoding texts...")
    embeddings = encode(model, tokenizer, texts)

    # Find duplicates
    print("Finding duplicates (threshold = 0.9)...")
    similarities = cosine_similarity(embeddings, embeddings)

    duplicates = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarities[i, j] > 0.9:
                duplicates.append((i, j, similarities[i, j]))

    print(f"\nFound {len(duplicates)} duplicate pair(s):")
    for i, j, sim in duplicates:
        print(f"  [{sim:.3f}] '{texts[i]}' ≈ '{texts[j]}'")
    print()


def main():
    """Run all examples."""
    print()
    print("=" * 70)
    print("MiniLM Embedding Examples")
    print("=" * 70)
    print()
    print("Sentence Transformer Model:")
    print("- all-MiniLM-L6-v2 (22.7MB)")
    print("- 384-dimensional embeddings")
    print("- BERT-based encoder with mean pooling")
    print()

    try:
        # Run examples
        example_1_basic_encoding()
        example_2_semantic_search()
        example_3_text_similarity()
        example_4_clustering()
        example_5_duplicate_detection()

        print("=" * 70)
        print("✓ All examples completed!")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"Error running examples: {e}")
        print()
        print("Make sure you have:")
        print("1. Installed required dependencies (pip install smlx transformers)")
        print("2. Internet connection for downloading the model")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
