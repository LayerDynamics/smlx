#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Silero VAD Examples.

Demonstrates voice activity detection capabilities:
1. Basic speech detection
2. Speech segmentation
3. Streaming detection
4. Filtering speech from audio
5. Custom thresholds
6. Different sample rates
"""

import sys
from pathlib import Path
import numpy as np

# Add smlx to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smlx.models.SileroVAD import (
    load,
    detect_speech,
    create_streaming_vad,
    filter_audio_by_speech,
)


def create_synthetic_audio(duration_seconds=5, sample_rate=16000):
    """Create synthetic audio with speech and silence patterns."""
    num_samples = int(duration_seconds * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    # Add "speech" segments (sine waves of different frequencies)
    speech_segments = [
        (0.5, 1.5, 440),   # 1 second at 440Hz
        (2.0, 3.0, 880),   # 1 second at 880Hz
        (3.5, 4.5, 660),   # 1 second at 660Hz
    ]

    for start, end, freq in speech_segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        t = np.linspace(0, end - start, end_idx - start_idx)
        audio[start_idx:end_idx] = 0.3 * np.sin(2 * np.pi * freq * t)

    # Add some noise
    audio += 0.01 * np.random.randn(num_samples)

    return audio


def example_1_basic_detection():
    """Example 1: Basic speech detection."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Speech Detection")
    print("=" * 80)

    # Load model
    print("\nLoading Silero VAD model...")
    vad = load(sample_rate=16000)

    # Create synthetic audio
    print("Creating synthetic audio with speech segments...")
    audio = create_synthetic_audio(duration_seconds=5)

    # Detect speech (simple boolean)
    print("\nDetecting speech (boolean)...")
    has_speech = detect_speech(vad, audio, return_timestamps=False)
    print(f"Speech detected: {has_speech}")

    # Detect with timestamps
    print("\nDetecting speech segments...")
    segments = detect_speech(vad, audio, return_timestamps=True)

    print(f"\nFound {len(segments)} speech segments:")
    for i, seg in enumerate(segments, 1):
        print(f"  {i}. {seg.start:.2f}s - {seg.end:.2f}s (duration: {seg.duration:.2f}s, confidence: {seg.confidence:.3f})")


def example_2_streaming_detection():
    """Example 2: Streaming voice activity detection."""
    print("\n" + "=" * 80)
    print("Example 2: Streaming Detection")
    print("=" * 80)

    # Load model
    print("\nLoading Silero VAD model...")
    vad = load(sample_rate=16000)

    # Create streaming VAD
    streaming = create_streaming_vad(vad)

    # Create audio stream (simulate real-time chunks)
    print("\nSimulating real-time audio stream...")
    audio = create_synthetic_audio(duration_seconds=5)

    chunk_size = 512  # Process 512 samples at a time
    num_chunks = len(audio) // chunk_size

    print(f"Processing {num_chunks} chunks of {chunk_size} samples each...")

    speech_chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = audio[start_idx:end_idx]

        # Process chunk
        probs = streaming.process_chunk(chunk)

        if probs.size > 0:
            prob = probs[0]
            time_s = i * chunk_size / 16000

            if prob > 0.5:
                speech_chunks.append(i)
                print(f"  Chunk {i} ({time_s:.2f}s): Speech detected (prob={prob:.3f})")

    print(f"\nDetected speech in {len(speech_chunks)} chunks")


def example_3_custom_thresholds():
    """Example 3: Using custom detection thresholds."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Thresholds")
    print("=" * 80)

    # Load model
    print("\nLoading Silero VAD model...")
    vad = load(sample_rate=16000)

    # Create audio
    audio = create_synthetic_audio(duration_seconds=5)

    # Try different thresholds
    thresholds = [0.3, 0.5, 0.7]

    print("\nComparing different thresholds:")
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        segments = detect_speech(vad, audio, threshold=threshold)
        print(f"  Segments detected: {len(segments)}")
        for seg in segments:
            print(f"    {seg.start:.2f}s - {seg.end:.2f}s (conf: {seg.confidence:.3f})")


def example_4_filter_speech():
    """Example 4: Extract only speech portions."""
    print("\n" + "=" * 80)
    print("Example 4: Filter Speech from Audio")
    print("=" * 80)

    # Load model
    print("\nLoading Silero VAD model...")
    vad = load(sample_rate=16000)

    # Create audio with silence
    audio = create_synthetic_audio(duration_seconds=5)

    print(f"Original audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")

    # Detect segments
    segments = detect_speech(vad, audio)

    # Filter to keep only speech
    speech_only = filter_audio_by_speech(audio, segments, sample_rate=16000)

    print(f"Speech-only audio: {len(speech_only)} samples ({len(speech_only)/16000:.2f}s)")
    print(f"Removed {len(audio) - len(speech_only)} samples of silence")


def example_5_different_sample_rates():
    """Example 5: Using different sample rates."""
    print("\n" + "=" * 80)
    print("Example 5: Different Sample Rates")
    print("=" * 80)

    sample_rates = [8000, 16000]

    for sr in sample_rates:
        print(f"\n--- Sample Rate: {sr}Hz ---")

        # Load model
        vad = load(sample_rate=sr)

        # Create audio at this sample rate
        audio = create_synthetic_audio(duration_seconds=3, sample_rate=sr)

        # Detect speech
        segments = detect_speech(vad, audio)

        print(f"Detected {len(segments)} segments:")
        for seg in segments:
            print(f"  {seg}")


def example_6_speech_statistics():
    """Example 6: Compute speech statistics."""
    print("\n" + "=" * 80)
    print("Example 6: Speech Statistics")
    print("=" * 80)

    # Load model
    print("\nLoading Silero VAD model...")
    vad = load(sample_rate=16000)

    # Create audio
    duration_s = 10
    audio = create_synthetic_audio(duration_seconds=duration_s)

    # Detect segments
    segments = detect_speech(vad, audio)

    # Compute statistics
    total_speech_time = sum(seg.duration for seg in segments)
    total_silence_time = duration_s - total_speech_time
    speech_ratio = total_speech_time / duration_s

    print("\nAudio Statistics:")
    print(f"  Total duration: {duration_s:.2f}s")
    print(f"  Speech time: {total_speech_time:.2f}s ({speech_ratio*100:.1f}%)")
    print(f"  Silence time: {total_silence_time:.2f}s ({(1-speech_ratio)*100:.1f}%)")
    print(f"  Number of segments: {len(segments)}")

    if segments:
        avg_segment_duration = total_speech_time / len(segments)
        avg_confidence = sum(seg.confidence for seg in segments) / len(segments)
        print(f"  Average segment duration: {avg_segment_duration:.2f}s")
        print(f"  Average confidence: {avg_confidence:.3f}")


def example_7_real_world_usage():
    """Example 7: Real-world usage pattern."""
    print("\n" + "=" * 80)
    print("Example 7: Real-World Usage Pattern")
    print("=" * 80)

    print("\nReal-world VAD workflow:")
    print("1. Load model once")
    print("2. Process audio files or streams")
    print("3. Use segments for downstream tasks")

    # Load model
    vad = load(sample_rate=16000)

    # Simulate processing multiple files
    files = ["file1", "file2", "file3"]

    print(f"\nProcessing {len(files)} audio files:")

    all_segments = {}
    for filename in files:
        # Create synthetic audio (in practice, load from file)
        audio = create_synthetic_audio(duration_seconds=3)

        # Detect speech
        segments = detect_speech(vad, audio)

        all_segments[filename] = segments
        print(f"  {filename}: {len(segments)} segments")

    print("\nUse cases for detected segments:")
    print("  - Extract speech for transcription")
    print("  - Remove silence for compression")
    print("  - Trigger recording when speech detected")
    print("  - Measure speaking time in conversations")
    print("  - Detect voice commands in real-time")


def main():
    """Run all examples."""
    print("=" * 80)
    print("Silero VAD - Voice Activity Detection Examples")
    print("=" * 80)
    print("\nNote: These examples use synthetic audio (sine waves).")
    print("For real audio files, provide file paths to detect_speech().")

    examples = [
        ("Basic Detection", example_1_basic_detection),
        ("Streaming Detection", example_2_streaming_detection),
        ("Custom Thresholds", example_3_custom_thresholds),
        ("Filter Speech", example_4_filter_speech),
        ("Different Sample Rates", example_5_different_sample_rates),
        ("Speech Statistics", example_6_speech_statistics),
        ("Real-World Usage", example_7_real_world_usage),
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
    print("  ✓ Basic speech detection")
    print("  ✓ Speech segmentation with timestamps")
    print("  ✓ Real-time streaming detection")
    print("  ✓ Speech filtering (remove silence)")
    print("  ✓ Custom detection thresholds")
    print("  ✓ Multiple sample rates (8kHz, 16kHz)")
    print("  ✓ Speech statistics and analysis")

    print("\nModel Advantages:")
    print("  - Very small (~1MB)")
    print("  - Fast inference")
    print("  - Low memory usage")
    print("  - Real-time capable")
    print("  - No GPU required")

    print("\nCommon Applications:")
    print("  - Voice assistants")
    print("  - Speech recognition preprocessing")
    print("  - Audio compression")
    print("  - Meeting analysis")
    print("  - Voice command detection")


if __name__ == "__main__":
    main()
