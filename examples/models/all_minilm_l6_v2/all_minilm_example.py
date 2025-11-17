#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
all-MiniLM-L6-v2 Sentence Embedding Examples

This script demonstrates various use cases for the all-MiniLM-L6-v2 model:
1. Basic encoding and similarity
2. Semantic search
3. Text clustering
4. Duplicate detection
5. FAQ matching
6. Batch processing
7. Performance benchmarking

The all-MiniLM-L6-v2 model is an ultra-lightweight (22MB) sentence transformer
that generates 384-dimensional embeddings for semantic search and similarity tasks.

Usage:
    python all_minilm_example.py
"""

import time

import numpy as np


def example_1_basic_encoding():
    """
    Example 1: Basic encoding and similarity.

    Demonstrates:
    - Loading the model
    - Encoding sentences
    - Computing cosine similarity
    """
    print("=" * 70)
    print("Example 1: Basic Encoding and Similarity")
    print("=" * 70)

    from smlx.models.all_MiniLM_L6_v2 import load, encode, cosine_similarity

    # Load model
    print("\n1. Loading all-MiniLM-L6-v2 model...")
    model, tokenizer = load()

    # Encode sentences
    print("\n2. Encoding sentences...")
    sentences = [
        "The cat sits on the mat",
        "A feline rests on a rug",
        "Python is a programming language",
    ]

    embeddings = encode(model, tokenizer, sentences)
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")

    # Compute similarities
    print("\n3. Computing pairwise similarities...")
    similarities = cosine_similarity(embeddings, embeddings)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            print(f"\n   '{sentences[i]}'")
            print(f"   vs")
            print(f"   '{sentences[j]}'")
            print(f"   Similarity: {similarities[i][j]:.4f}")


def example_2_semantic_search():
    """
    Example 2: Semantic search.

    Demonstrates:
    - Building a document corpus
    - Searching with natural language queries
    - Ranking results by relevance
    """
    print("\n" + "=" * 70)
    print("Example 2: Semantic Search")
    print("=" * 70)

    from smlx.models.all_MiniLM_L6_v2 import load, encode, cosine_similarity

    # Load model
    print("\n1. Loading model...")
    model, tokenizer = load()

    # Document corpus
    print("\n2. Creating document corpus...")
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "The weather forecast predicts rain tomorrow",
        "Deep learning uses neural networks with multiple layers",
        "JavaScript is commonly used for web development",
        "Natural language processing helps computers understand human language",
        "The stock market experienced significant volatility today",
        "Computer vision enables machines to interpret visual information",
    ]

    print(f"   Corpus size: {len(documents)} documents")

    # Encode documents
    print("\n3. Encoding documents...")
    doc_embeddings = encode(model, tokenizer, documents)

    # Search queries
    queries = [
        "What is AI?",
        "Programming languages",
        "Weather information",
    ]

    print("\n4. Searching with queries...")
    for query in queries:
        print(f"\n   Query: '{query}'")

        # Encode query
        query_embedding = encode(model, tokenizer, [query])

        # Compute similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Get top 3 results
        top_k = 3
        top_indices = np.argsort(similarities)[::-1][:top_k]

        print(f"   Top {top_k} results:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"      {rank}. [{similarities[idx]:.4f}] {documents[idx]}")


def example_3_text_clustering():
    """
    Example 3: Text clustering.

    Demonstrates:
    - Encoding a collection of texts
    - Clustering with K-means
    - Analyzing cluster assignments
    """
    print("\n" + "=" * 70)
    print("Example 3: Text Clustering")
    print("=" * 70)

    from sklearn.cluster import KMeans
    from smlx.models.all_MiniLM_L6_v2 import load, encode

    # Load model
    print("\n1. Loading model...")
    model, tokenizer = load()

    # Sample texts (3 topics: AI, weather, food)
    print("\n2. Preparing texts from 3 different topics...")
    texts = [
        # AI/ML topic
        "Machine learning algorithms learn from data",
        "Neural networks are powerful AI models",
        "Deep learning achieves state-of-the-art results",
        # Weather topic
        "The weather is sunny and warm today",
        "Tomorrow will be cloudy with rain",
        "Temperature will drop below freezing tonight",
        # Food topic
        "Pizza is a popular Italian dish",
        "Sushi is traditional Japanese cuisine",
        "Tacos are a beloved Mexican food",
    ]

    # Encode
    print("\n3. Encoding texts...")
    embeddings = encode(model, tokenizer, texts)

    # Cluster
    print("\n4. Clustering with K-means (k=3)...")
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # Show results
    print("\n5. Cluster assignments:")
    for cluster_id in range(n_clusters):
        print(f"\n   Cluster {cluster_id}:")
        cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
        for text in cluster_texts:
            print(f"      - {text}")


def example_4_duplicate_detection():
    """
    Example 4: Duplicate detection.

    Demonstrates:
    - Finding near-duplicate texts
    - Using similarity threshold
    - Identifying paraphrases
    """
    print("\n" + "=" * 70)
    print("Example 4: Duplicate Detection")
    print("=" * 70)

    from smlx.models.all_MiniLM_L6_v2 import load, encode, cosine_similarity

    # Load model
    print("\n1. Loading model...")
    model, tokenizer = load()

    # Texts with some duplicates
    print("\n2. Preparing text collection with duplicates...")
    texts = [
        "Hello world",
        "Hi there",
        "Hello world",  # Exact duplicate
        "Greetings everyone",
        "Hello there world",  # Near duplicate
        "Goodbye everyone",
        "Hi there",  # Exact duplicate
        "See you later",
    ]

    # Encode
    print("\n3. Encoding texts...")
    embeddings = encode(model, tokenizer, texts)

    # Find duplicates
    print("\n4. Finding duplicates (threshold=0.85)...")
    threshold = 0.85
    similarities = cosine_similarity(embeddings, embeddings)

    duplicates = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarities[i][j] > threshold:
                duplicates.append((i, j, similarities[i][j]))

    print(f"\n   Found {len(duplicates)} duplicate pairs:")
    for i, j, sim in duplicates:
        print(f"\n   [{sim:.4f}] Duplicate pair:")
        print(f"      Text {i}: '{texts[i]}'")
        print(f"      Text {j}: '{texts[j]}'")


def example_5_faq_matching():
    """
    Example 5: FAQ matching.

    Demonstrates:
    - Building a FAQ database
    - Matching user questions to FAQs
    - Finding the most relevant answer
    """
    print("\n" + "=" * 70)
    print("Example 5: FAQ Matching")
    print("=" * 70)

    from smlx.models.all_MiniLM_L6_v2 import load, encode, cosine_similarity

    # Load model
    print("\n1. Loading model...")
    model, tokenizer = load()

    # FAQ database
    print("\n2. Creating FAQ database...")
    faqs = [
        {
            "question": "How do I reset my password?",
            "answer": "Click 'Forgot Password' on the login page and follow the instructions.",
        },
        {
            "question": "What are your business hours?",
            "answer": "We're open Monday-Friday, 9 AM to 5 PM EST.",
        },
        {
            "question": "How can I contact customer support?",
            "answer": "Email support@example.com or call 1-800-SUPPORT.",
        },
        {
            "question": "What is your return policy?",
            "answer": "We accept returns within 30 days of purchase with receipt.",
        },
        {
            "question": "Do you offer international shipping?",
            "answer": "Yes, we ship to over 50 countries worldwide.",
        },
    ]

    faq_questions = [faq["question"] for faq in faqs]
    print(f"   FAQ database size: {len(faqs)} entries")

    # Encode FAQ questions
    print("\n3. Encoding FAQ questions...")
    faq_embeddings = encode(model, tokenizer, faq_questions)

    # User queries
    user_queries = [
        "I forgot my password",
        "When are you open?",
        "Can you ship to Canada?",
    ]

    print("\n4. Matching user queries to FAQs...")
    for query in user_queries:
        print(f"\n   User query: '{query}'")

        # Encode query
        query_embedding = encode(model, tokenizer, [query])

        # Find most similar FAQ
        similarities = cosine_similarity(query_embedding, faq_embeddings)[0]
        best_idx = np.argmax(similarities)

        print(f"   Best match [{similarities[best_idx]:.4f}]:")
        print(f"      Q: {faqs[best_idx]['question']}")
        print(f"      A: {faqs[best_idx]['answer']}")


def example_6_batch_processing():
    """
    Example 6: Batch processing.

    Demonstrates:
    - Encoding large batches of texts
    - Different batch sizes
    - Performance comparison
    """
    print("\n" + "=" * 70)
    print("Example 6: Batch Processing")
    print("=" * 70)

    from smlx.models.all_MiniLM_L6_v2 import load, encode

    # Load model
    print("\n1. Loading model...")
    model, tokenizer = load()

    # Generate sample texts
    print("\n2. Generating sample texts...")
    num_texts = 100
    texts = [f"This is sample sentence number {i} for testing." for i in range(num_texts)]

    # Test different batch sizes
    batch_sizes = [1, 8, 16, 32, 64]

    print(f"\n3. Encoding {num_texts} texts with different batch sizes...")
    for batch_size in batch_sizes:
        start = time.time()
        embeddings = encode(model, tokenizer, texts, batch_size=batch_size)
        elapsed = time.time() - start

        sentences_per_sec = num_texts / elapsed
        print(
            f"   Batch size {batch_size:2d}: {elapsed:.2f}s ({sentences_per_sec:.0f} sentences/sec)"
        )


def example_7_performance_benchmark():
    """
    Example 7: Performance benchmarking.

    Demonstrates:
    - Single sentence encoding speed
    - Batch encoding throughput
    - Memory efficiency
    """
    print("\n" + "=" * 70)
    print("Example 7: Performance Benchmark")
    print("=" * 70)

    from smlx.models.all_MiniLM_L6_v2 import load, encode, encode_single

    # Load model
    print("\n1. Loading model...")
    model, tokenizer = load()

    # Single sentence benchmark
    print("\n2. Benchmarking single sentence encoding...")
    sentence = "This is a test sentence for benchmarking."

    num_runs = 100
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = encode_single(model, tokenizer, sentence)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"   Runs: {num_runs}")
    print(f"   Average time: {avg_time*1000:.2f}ms")
    print(f"   Min time: {min_time*1000:.2f}ms")
    print(f"   Max time: {max_time*1000:.2f}ms")

    # Batch benchmark
    print("\n3. Benchmarking batch encoding...")
    batch_sizes = [10, 50, 100]

    for batch_size in batch_sizes:
        sentences = [f"Test sentence {i}" for i in range(batch_size)]

        num_runs = 10
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = encode(model, tokenizer, sentences, batch_size=32)
            times.append(time.time() - start)

        avg_time = np.mean(times)
        throughput = batch_size / avg_time

        print(f"\n   Batch size {batch_size}:")
        print(f"      Average time: {avg_time*1000:.1f}ms")
        print(f"      Throughput: {throughput:.0f} sentences/sec")
        print(f"      Per-sentence: {avg_time/batch_size*1000:.2f}ms")

    # Model size
    print("\n4. Model information:")
    print(f"   Model: all-MiniLM-L6-v2")
    print(f"   Parameters: ~22M")
    print(f"   Size: ~22.7MB")
    print(f"   Embedding dimension: 384")
    print(f"   Max sequence length: 256 tokens")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("all-MiniLM-L6-v2 Sentence Embedding Examples")
    print("=" * 70)
    print("\nDemonstrating the ultra-lightweight 22MB sentence embedding model")
    print("Optimized for Apple M4 chipsets with MLX framework\n")

    examples = [
        ("1. Basic Encoding", example_1_basic_encoding),
        ("2. Semantic Search", example_2_semantic_search),
        ("3. Text Clustering", example_3_text_clustering),
        ("4. Duplicate Detection", example_4_duplicate_detection),
        ("5. FAQ Matching", example_5_faq_matching),
        ("6. Batch Processing", example_6_batch_processing),
        ("7. Performance Benchmark", example_7_performance_benchmark),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. all-MiniLM-L6-v2 is ultra-lightweight (22MB) and fast")
    print("2. Generates high-quality 384-dimensional embeddings")
    print("3. Perfect for semantic search, clustering, and similarity tasks")
    print("4. Excellent performance on Apple M4 with MLX (~500 sentences/sec)")
    print("5. Easy to use with simple load() and encode() API")
    print("\nUse Cases:")
    print("- Semantic search and document retrieval")
    print("- Text clustering and categorization")
    print("- Duplicate and paraphrase detection")
    print("- FAQ matching and question answering")
    print("- Recommendation systems")
    print("- Zero-shot classification")
    print("\nFor more information:")
    print("- HuggingFace: sentence-transformers/all-MiniLM-L6-v2")
    print("- Paper: https://arxiv.org/abs/1908.10084")
    print("- Docs: smlx/models/all-MiniLM-L6-v2/README.md")


if __name__ == "__main__":
    main()
