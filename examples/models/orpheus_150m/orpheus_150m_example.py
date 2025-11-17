#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Orpheus-150M Text-to-Speech Examples

This script demonstrates various TTS tasks using Orpheus-150M:
1. Basic speech synthesis
2. Batch synthesis
3. Streaming synthesis (low latency)
4. Speed control
5. Duration estimation
6. Mel-spectrogram extraction
7. Performance benchmarking

IMPORTANT NOTE:
    These examples use a reference implementation with placeholder outputs.
    For production use, search HuggingFace for TTS models:
    - facebook/fastspeech2-en-ljspeech
    - facebook/tts_transformer-en-ljspeech
    - suno/bark (larger, high quality)

Usage:
    python orpheus_150m_example.py
"""

import time
from pathlib import Path

import numpy as np


def example_1_basic_synthesis():
    """
    Example 1: Basic speech synthesis.

    Demonstrates:
    - Loading Orpheus-150M model
    - Synthesizing speech from text
    - Saving to WAV file
    """
    print("=" * 70)
    print("Example 1: Basic Speech Synthesis")
    print("=" * 70)

    from smlx.models.Orpheus_150M import load, save_audio, synthesize

    # Load model
    print("\n1. Loading Orpheus-150M model...")
    model, processor = load()

    # Synthesize speech
    print("\n2. Synthesizing speech...")
    text = "Hello, this is a test of text to speech synthesis using Orpheus."
    audio = synthesize(model=model, processor=processor, text=text, sample_rate=24000)

    print(f"\nGenerated audio: {len(audio)} samples")
    print(f"Duration: {len(audio)/24000:.2f} seconds")

    # Save to file
    output_path = "orpheus_basic.wav"
    save_audio(audio, output_path, sample_rate=24000)

    print(
        "\nNote: This is placeholder audio. Load pre-trained weights for actual synthesis."
    )


def example_2_batch_synthesis():
    """
    Example 2: Batch synthesis.

    Demonstrates:
    - Synthesizing multiple texts
    - Efficient batch processing
    - Saving multiple files
    """
    print("\n" + "=" * 70)
    print("Example 2: Batch Synthesis")
    print("=" * 70)

    from smlx.models.Orpheus_150M import load, save_audio, synthesize_batch

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    # Prepare batch of texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Text to speech synthesis enables many applications.",
        "Natural language processing is advancing rapidly.",
        "Voice assistants use TTS for communication.",
    ]

    # Synthesize batch
    print(f"\n2. Synthesizing {len(texts)} texts...")
    audios = synthesize_batch(model=model, processor=processor, texts=texts)

    # Save files
    print("\n3. Saving audio files...")
    for i, audio in enumerate(audios):
        output_path = f"orpheus_batch_{i+1}.wav"
        save_audio(audio, output_path, sample_rate=24000)
        print(f"   Saved: {output_path}")

    print(
        "\nNote: Placeholder audio generated. Use pre-trained model for real synthesis."
    )


def example_3_streaming_synthesis():
    """
    Example 3: Streaming synthesis for low latency.

    Demonstrates:
    - Streaming synthesis in chunks
    - Lower latency for real-time applications
    - Progressive audio generation
    """
    print("\n" + "=" * 70)
    print("Example 3: Streaming Synthesis")
    print("=" * 70)

    from smlx.models.Orpheus_150M import load, stream_synthesize

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    # Long text for streaming
    long_text = """
    Text to speech synthesis is the artificial production of human speech.
    A computer system used for this purpose is called a speech synthesizer.
    Modern TTS systems use deep learning to produce natural-sounding speech.
    Streaming synthesis allows for lower latency by generating audio in chunks.
    This is especially useful for real-time applications like voice assistants.
    """

    # Stream synthesis
    print("\n2. Streaming synthesis...")
    print("Generating audio chunks:")

    all_chunks = []
    for i, audio_chunk in enumerate(
        stream_synthesize(model, processor, long_text, chunk_size=20)
    ):
        print(f"   Chunk {i+1}: {len(audio_chunk)} samples")
        all_chunks.append(audio_chunk)

        # In a real application, you would play audio_chunk immediately
        # play_audio(audio_chunk)

    # Combine all chunks
    full_audio = np.concatenate(all_chunks)
    print(f"\nTotal audio: {len(full_audio)} samples ({len(full_audio)/24000:.2f}s)")

    print(
        "\nNote: In production, each chunk would be played immediately for low latency."
    )


def example_4_speed_control():
    """
    Example 4: Speech speed control.

    Demonstrates:
    - Controlling speech speed
    - Slower speech (0.75x)
    - Normal speed (1.0x)
    - Faster speech (1.5x)
    """
    print("\n" + "=" * 70)
    print("Example 4: Speed Control")
    print("=" * 70)

    from smlx.models.Orpheus_150M import load, save_audio, synthesize_with_speed

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    text = "This is a test of speech speed control."

    # Different speeds
    speeds = [0.75, 1.0, 1.5]
    speed_names = ["slow", "normal", "fast"]

    print("\n2. Generating speech at different speeds...")
    for speed, name in zip(speeds, speed_names):
        print(f"\n   Speed: {speed}x ({name})")
        audio = synthesize_with_speed(
            model=model, processor=processor, text=text, speed=speed
        )

        output_path = f"orpheus_speed_{name}.wav"
        save_audio(audio, output_path, sample_rate=24000)
        print(f"   Duration: {len(audio)/24000:.2f}s")
        print(f"   Saved: {output_path}")

    print(
        "\nNote: Speed control modifies duration predictions. Use pre-trained model for actual synthesis."
    )


def example_5_duration_estimation():
    """
    Example 5: Estimate speech duration without synthesis.

    Demonstrates:
    - Estimating speech duration
    - Planning audio generation
    - Resource allocation
    """
    print("\n" + "=" * 70)
    print("Example 5: Duration Estimation")
    print("=" * 70)

    from smlx.models.Orpheus_150M import estimate_duration, load

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    # Test texts
    texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence are transforming technology in profound ways.",
        "This is a very long sentence that will take much more time to synthesize because it contains many more words and characters than a short sentence.",
    ]

    print("\n2. Estimating durations...")
    for text in texts:
        duration = estimate_duration(
            model=model, processor=processor, text=text, sample_rate=24000
        )

        print(f"\nText: {text[:50]}...")
        print(f"Estimated duration: {duration:.2f}s")


def example_6_mel_spectrogram():
    """
    Example 6: Extract mel-spectrogram.

    Demonstrates:
    - Getting mel-spectrogram without vocoding
    - Visualization preparation
    - External vocoder usage
    """
    print("\n" + "=" * 70)
    print("Example 6: Mel-Spectrogram Extraction")
    print("=" * 70)

    from smlx.models.Orpheus_150M import get_mel_spectrogram, load

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    # Generate mel-spectrogram
    text = "Hello, this is a mel-spectrogram test."
    print(f"\n2. Generating mel-spectrogram for: '{text}'")

    mel = get_mel_spectrogram(model=model, processor=processor, text=text)

    print(f"\nMel-spectrogram shape: {mel.shape}")
    print(f"Time frames: {mel.shape[0]}")
    print(f"Mel bins: {mel.shape[1]}")

    # Visualize (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.imshow(mel.T, aspect="auto", origin="lower", cmap="viridis")
        plt.ylabel("Mel Bins")
        plt.xlabel("Time Frames")
        plt.title("Mel-Spectrogram")
        plt.colorbar(label="Magnitude")

        output_path = "orpheus_mel_spectrogram.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nMel-spectrogram visualization saved: {output_path}")
    except ImportError:
        print("\nMatplotlib not available. Install with: pip install matplotlib")

    print("\nNote: Mel-spectrogram can be used with external vocoders like HiFi-GAN.")


def example_7_performance_benchmark():
    """
    Example 7: Performance benchmarking.

    Demonstrates:
    - Measuring synthesis speed
    - Real-time factor calculation
    - Memory efficiency
    """
    print("\n" + "=" * 70)
    print("Example 7: Performance Benchmark")
    print("=" * 70)

    from smlx.models.Orpheus_150M import load, print_model_info, synthesize

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    # Print model info
    print_model_info(model)

    # Benchmark text
    text = "The quick brown fox jumps over the lazy dog. " * 5  # Repeat for longer audio

    # Warmup
    print("\n2. Warmup run...")
    _ = synthesize(model=model, processor=processor, text=text, sample_rate=24000)

    # Benchmark runs
    print("\n3. Running benchmark (10 iterations)...")
    num_runs = 10
    times = []

    for i in range(num_runs):
        start = time.time()
        audio = synthesize(model=model, processor=processor, text=text, sample_rate=24000)
        elapsed = time.time() - start
        times.append(elapsed)

        audio_duration = len(audio) / 24000
        rtf = elapsed / audio_duration  # Real-Time Factor

        print(f"   Run {i+1}: {elapsed*1000:.1f}ms (RTF: {rtf:.2f}x)")

    # Statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)

    audio_duration = len(audio) / 24000
    avg_rtf = avg_time / audio_duration

    print(f"\nBenchmark Results:")
    print(f"  Average time: {avg_time*1000:.1f}ms")
    print(f"  Min time: {min_time*1000:.1f}ms")
    print(f"  Max time: {max_time*1000:.1f}ms")
    print(f"  Std dev: {std_time*1000:.1f}ms")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  Real-Time Factor: {avg_rtf:.2f}x")

    if avg_rtf < 1.0:
        speedup = 1.0 / avg_rtf
        print(f"  Speed: {speedup:.1f}x faster than real-time")
    else:
        print(f"  Speed: {avg_rtf:.2f}x slower than real-time")

    print(
        "\nNote: Performance will be significantly better with actual implementation and GPU."
    )


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Orpheus-150M Text-to-Speech Examples")
    print("=" * 70)
    print("\nIMPORTANT: These examples use a reference implementation.")
    print("For production use, search HuggingFace for TTS models:")
    print("  - facebook/fastspeech2-en-ljspeech")
    print("  - facebook/tts_transformer-en-ljspeech")
    print("  - suno/bark (larger, high quality)")
    print("\nRunning 7 examples:\n")

    examples = [
        ("1. Basic Synthesis", example_1_basic_synthesis),
        ("2. Batch Synthesis", example_2_batch_synthesis),
        ("3. Streaming Synthesis", example_3_streaming_synthesis),
        ("4. Speed Control", example_4_speed_control),
        ("5. Duration Estimation", example_5_duration_estimation),
        ("6. Mel-Spectrogram", example_6_mel_spectrogram),
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
    print("\nNext steps:")
    print("1. Search HuggingFace for TTS models")
    print("2. Load pre-trained weights")
    print("3. Integrate TTS into your application")
    print("4. Fine-tune on custom voice data if needed")
    print("5. See docs/ModelImplementations.md for details")


if __name__ == "__main__":
    main()
