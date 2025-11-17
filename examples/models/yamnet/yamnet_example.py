#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
YAMNet Examples.

Demonstrates audio event classification capabilities:
1. Basic audio classification
2. Batch classification
3. Audio embedding extraction
4. Event detection
5. Audio similarity
6. Real-time monitoring
7. Audio search engine
"""

import sys
from pathlib import Path

import numpy as np

# Add smlx to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from smlx.models.YAMNet import (
    classify,
    classify_batch,
    compute_audio_similarity,
    detect_events,
    extract_embeddings,
    load,
)


def create_synthetic_audio(duration_seconds=3, sample_rate=16000, frequencies=None):
    """Create synthetic audio for testing.

    Args:
        duration_seconds: Duration in seconds
        sample_rate: Sample rate
        frequencies: List of frequencies to include

    Returns:
        Audio array
    """
    if frequencies is None:
        frequencies = [440, 880]  # A4 and A5

    num_samples = int(duration_seconds * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    # Add sine waves at different frequencies
    t = np.linspace(0, duration_seconds, num_samples)
    for freq in frequencies:
        audio += 0.2 * np.sin(2 * np.pi * freq * t)

    # Add some noise
    audio += 0.01 * np.random.randn(num_samples).astype(np.float32)

    # Normalize
    audio = audio / np.max(np.abs(audio))

    return audio


def example_1_basic_classification():
    """Example 1: Basic audio event classification."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Audio Classification")
    print("=" * 80)

    # Load model
    print("\nLoading YAMNet model...")
    model = load()

    # Create synthetic audio
    print("Creating synthetic audio...")
    audio = create_synthetic_audio(duration_seconds=3)

    # Classify
    print("\nClassifying audio...")
    predictions = classify(model, audio, top_k=5)

    print("\nTop 5 predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred.label}: {pred.score:.3f} (class {pred.class_id})")


def example_2_batch_classification():
    """Example 2: Batch classification of multiple audio files."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Classification")
    print("=" * 80)

    # Load model
    print("\nLoading YAMNet model...")
    model = load()

    # Create multiple synthetic audio samples
    print("\nCreating synthetic audio samples...")
    audio_samples = [
        create_synthetic_audio(3, frequencies=[440]),  # Music-like
        create_synthetic_audio(3, frequencies=[200, 300]),  # Speech-like
        create_synthetic_audio(3, frequencies=[1000, 2000]),  # High-pitched
    ]

    # Batch classify
    print("\nClassifying batch...")
    predictions_batch = classify_batch(model, audio_samples, top_k=3)

    print("\nResults:")
    for i, predictions in enumerate(predictions_batch, 1):
        print(f"\n  Audio {i}:")
        for pred in predictions:
            print(f"    {pred.label}: {pred.score:.3f}")


def example_3_extract_embeddings():
    """Example 3: Extract audio embeddings."""
    print("\n" + "=" * 80)
    print("Example 3: Audio Embedding Extraction")
    print("=" * 80)

    # Load model
    print("\nLoading YAMNet model...")
    model = load()

    # Create audio
    audio = create_synthetic_audio(duration_seconds=5)

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(model, audio)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"  Patches: {embeddings.shape[0]}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")

    # Average over time for single vector
    import mlx.core as mx

    audio_embedding = mx.mean(embeddings, axis=0)
    print(f"\nTime-averaged embedding shape: {audio_embedding.shape}")
    print("\nUse cases:")
    print("  - Audio similarity search")
    print("  - Clustering similar sounds")
    print("  - Audio retrieval")
    print("  - Downstream ML tasks")


def example_4_detect_events():
    """Example 4: Detect specific audio events."""
    print("\n" + "=" * 80)
    print("Example 4: Specific Event Detection")
    print("=" * 80)

    # Load model
    print("\nLoading YAMNet model...")
    model = load()

    # Create audio
    audio = create_synthetic_audio(duration_seconds=4)

    # Detect specific events
    print("\nDetecting music events...")
    music_events = detect_events(
        model, audio, event_classes=["music", "musical instrument"], threshold=0.1
    )

    print(f"\nFound {len(music_events)} music-related events:")
    for event in music_events:
        print(f"  {event.label}: {event.score:.3f}")

    print("\nDetecting speech events...")
    speech_events = detect_events(
        model, audio, event_classes=["speech", "conversation"], threshold=0.1
    )

    print(f"\nFound {len(speech_events)} speech-related events:")
    for event in speech_events:
        print(f"  {event.label}: {event.score:.3f}")


def example_5_audio_similarity():
    """Example 5: Compute audio similarity."""
    print("\n" + "=" * 80)
    print("Example 5: Audio Similarity")
    print("=" * 80)

    # Load model
    print("\nLoading YAMNet model...")
    model = load()

    # Create similar and dissimilar audio
    print("\nCreating audio samples...")
    audio1 = create_synthetic_audio(3, frequencies=[440, 880])
    audio2 = create_synthetic_audio(3, frequencies=[440, 880])  # Similar
    audio3 = create_synthetic_audio(3, frequencies=[200, 300])  # Different

    # Compute similarities
    print("\nComputing similarities...")
    similarity_12 = compute_audio_similarity(model, audio1, audio2)
    similarity_13 = compute_audio_similarity(model, audio1, audio3)

    print("\nResults:")
    print(f"  Audio 1 vs Audio 2 (similar): {similarity_12:.3f}")
    print(f"  Audio 1 vs Audio 3 (different): {similarity_13:.3f}")

    if similarity_12 > similarity_13:
        print("\n✓ Correctly identified similar audio has higher similarity")
    else:
        print("\n⚠ Unexpected similarity results (synthetic audio may not be ideal test case)")


def example_6_realtime_monitoring():
    """Example 6: Real-time audio monitoring simulation."""
    print("\n" + "=" * 80)
    print("Example 6: Real-time Audio Monitoring")
    print("=" * 80)

    # Load model
    print("\nLoading YAMNet model...")
    model = load()

    # Simulate audio stream
    print("\nSimulating audio stream (10 seconds)...")
    sample_rate = 16000
    total_duration = 10
    chunk_duration = 2  # 2-second chunks

    audio_stream = create_synthetic_audio(total_duration, sample_rate)

    chunk_size = sample_rate * chunk_duration
    num_chunks = len(audio_stream) // chunk_size

    print(f"Processing {num_chunks} chunks of {chunk_duration}s each...\n")

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = audio_stream[start_idx:end_idx]

        # Classify chunk
        predictions = classify(model, chunk, top_k=1)
        top_event = predictions[0]

        # Alert if confidence is high
        time_s = i * chunk_duration
        if top_event.score > 0.5:
            print(
                f"  [{time_s:02d}s] ⚠️  Detected: {top_event.label} (conf: {top_event.score:.2f})"
            )
        else:
            print(f"  [{time_s:02d}s] ... {top_event.label} (conf: {top_event.score:.2f})")


def example_7_audio_search_engine():
    """Example 7: Audio search engine using embeddings."""
    print("\n" + "=" * 80)
    print("Example 7: Audio Search Engine")
    print("=" * 80)

    # Load model
    print("\nLoading YAMNet model...")
    model = load()

    # Create audio database
    print("\nCreating audio database...")
    audio_db = {
        "sample1.wav": create_synthetic_audio(3, frequencies=[440]),
        "sample2.wav": create_synthetic_audio(3, frequencies=[880]),
        "sample3.wav": create_synthetic_audio(3, frequencies=[440, 880]),
        "sample4.wav": create_synthetic_audio(3, frequencies=[200, 300]),
    }

    # Extract embeddings for all audio
    print("Indexing audio files...")
    import mlx.core as mx

    embeddings_db = {}
    for filename, audio in audio_db.items():
        emb = extract_embeddings(model, audio)
        # Average over time
        embeddings_db[filename] = mx.mean(emb, axis=0)
        print(f"  Indexed: {filename}")

    # Search query
    print("\nSearching for similar audio...")
    query_audio = create_synthetic_audio(3, frequencies=[440, 880])
    query_emb = extract_embeddings(model, query_audio)
    query_emb_avg = mx.mean(query_emb, axis=0)

    # Compute similarities
    similarities = {}
    for filename, emb in embeddings_db.items():
        # Cosine similarity
        dot_product = mx.sum(query_emb_avg * emb)
        norm1 = mx.sqrt(mx.sum(query_emb_avg * query_emb_avg))
        norm2 = mx.sqrt(mx.sum(emb * emb))
        similarity = float(dot_product / (norm1 * norm2))
        similarities[filename] = similarity

    # Sort by similarity
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    print("\nSearch results (most similar first):")
    for i, (filename, similarity) in enumerate(sorted_results, 1):
        print(f"  {i}. {filename}: {similarity:.3f}")


def example_8_class_exploration():
    """Example 8: Explore AudioSet classes."""
    print("\n" + "=" * 80)
    print("Example 8: AudioSet Class Exploration")
    print("=" * 80)

    from smlx.models.YAMNet import load_class_names

    # Load class names
    class_names = load_class_names()

    print(f"\nYAMNet recognizes {len(class_names)} AudioSet classes")

    # Show some categories
    speech_classes = [c for c in class_names if "speech" in c.lower()]
    music_classes = [c for c in class_names if "music" in c.lower()]
    animal_classes = [c for c in class_names if any(
        animal in c.lower() for animal in ["dog", "cat", "bird", "animal"]
    )]

    print(f"\nSpeech-related classes ({len(speech_classes)}):")
    for cls in speech_classes[:5]:
        print(f"  - {cls}")
    if len(speech_classes) > 5:
        print(f"  ... and {len(speech_classes) - 5} more")

    print(f"\nMusic-related classes ({len(music_classes)}):")
    for cls in music_classes[:5]:
        print(f"  - {cls}")
    if len(music_classes) > 5:
        print(f"  ... and {len(music_classes) - 5} more")

    print(f"\nAnimal-related classes ({len(animal_classes)}):")
    for cls in animal_classes[:5]:
        print(f"  - {cls}")
    if len(animal_classes) > 5:
        print(f"  ... and {len(animal_classes) - 5} more")


def main():
    """Run all examples."""
    print("=" * 80)
    print("YAMNet - Audio Event Classification Examples")
    print("=" * 80)
    print("\nNote: These examples use synthetic audio (sine waves) for demonstration.")
    print("For real audio files, provide file paths to classify().")
    print("\nOn first run, YAMNet will download and convert pre-trained weights (~15MB).")
    print("This requires PyTorch. Install with: pip install torch")
    print("Subsequent runs will use cached weights.")

    examples = [
        ("Basic Classification", example_1_basic_classification),
        ("Batch Classification", example_2_batch_classification),
        ("Extract Embeddings", example_3_extract_embeddings),
        ("Detect Events", example_4_detect_events),
        ("Audio Similarity", example_5_audio_similarity),
        ("Real-time Monitoring", example_6_realtime_monitoring),
        ("Audio Search Engine", example_7_audio_search_engine),
        ("Class Exploration", example_8_class_exploration),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"\n\nError in {name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Audio event classification")
    print("  ✓ Batch processing")
    print("  ✓ Embedding extraction")
    print("  ✓ Specific event detection")
    print("  ✓ Audio similarity computation")
    print("  ✓ Real-time monitoring")
    print("  ✓ Audio search/retrieval")

    print("\nModel Advantages:")
    print("  - Ultra-tiny (3.7M parameters, ~15MB)")
    print("  - 521 AudioSet event classes")
    print("  - Fast inference")
    print("  - Low memory usage")
    print("  - Real-time capable")
    print("  - No GPU required")

    print("\nCommon Applications:")
    print("  - Smart home monitoring")
    print("  - Audio content tagging")
    print("  - Sound effect detection")
    print("  - Environmental sound analysis")
    print("  - Audio similarity search")
    print("  - Preprocessing for audio pipelines")


if __name__ == "__main__":
    main()
