"""
Streaming transcription with partial results example.

Demonstrates real-time transcription with both partial and final results,
showing how to configure and use the streaming partial results feature.

This example shows:
- Configuring partial result detection
- Processing audio chunks in real-time
- Distinguishing between partial and final results
- VAD-based and punctuation-based detection strategies
"""

import time
from pathlib import Path

import numpy as np

from smlx.models.Whisper_tiny import load
from smlx.models.Whisper_tiny.audio import SAMPLE_RATE, load_audio
from smlx.models.Whisper_tiny.streaming import StreamingConfig, StreamingTranscriber


def simulate_streaming_from_file(audio_path: str, enable_vad: bool = False):
    """
    Simulate streaming transcription from a pre-recorded file.

    This simulates real-time streaming by processing the file in chunks,
    showing both partial and final results as they would appear in
    a live transcription scenario.

    Args:
        audio_path: Path to audio file
        enable_vad: Enable Voice Activity Detection for better partial/final detection
    """
    print("=" * 80)
    print("Streaming Transcription with Partial Results")
    print("=" * 80)

    # Load model
    print("\nLoading Whisper model...")
    model, tokenizer = load()
    print("Model loaded!")

    # Configure streaming with partial results
    config = StreamingConfig(
        chunk_duration=3.0,  # Process 3-second chunks
        overlap_duration=0.5,  # 0.5s overlap between chunks
        min_chunk_duration=1.0,  # Minimum 1s to process
        enable_partial_results=True,  # Enable partial results
        partial_result_interval=0.3,  # Emit partials every 0.3s
        enable_vad=enable_vad,  # VAD for better detection
        vad_threshold=0.5 if enable_vad else None,
        temperature=0.0,  # Greedy decoding for consistency
    )

    # Create transcriber
    transcriber = StreamingTranscriber(model, tokenizer, config=config)

    # Load audio file
    print(f"\nLoading audio from: {audio_path}")
    audio = load_audio(audio_path)
    print(f"Audio duration: {len(audio) / SAMPLE_RATE:.2f}s")

    # Simulate streaming by processing in chunks
    print("\n" + "=" * 80)
    print("Transcription Results (P=partial, F=final):")
    print("=" * 80 + "\n")

    chunk_size = int(config.chunk_duration * SAMPLE_RATE)
    partial_count = 0
    final_count = 0

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]

        # Simulate real-time delay
        time.sleep(0.1)

        # Process chunk
        result = transcriber.process_chunk(chunk)

        if result:
            result_type = "F" if result.is_final else "P"
            print(f"[{result.start_time:6.2f}s - {result.end_time:6.2f}s] [{result_type}] {result.text}")
            print(f"  Language: {result.language}, Confidence: {result.confidence:.4f}")

            if result.is_final:
                final_count += 1
            else:
                partial_count += 1

    # Process any remaining buffered audio
    print("\nProcessing remaining buffer...")
    while transcriber.has_enough_audio():
        result = transcriber.process_chunk()
        if result:
            result_type = "F" if result.is_final else "P"
            print(f"[{result.start_time:6.2f}s - {result.end_time:6.2f}s] [{result_type}] {result.text}")

            if result.is_final:
                final_count += 1
            else:
                partial_count += 1

    print("\n" + "=" * 80)
    print(f"Summary: {partial_count} partial results, {final_count} final results")
    print("=" * 80)


def demonstrate_partial_detection_strategies():
    """
    Demonstrate different partial result detection strategies.

    Shows how punctuation, buffer state, and VAD affect final/partial classification.
    """
    print("=" * 80)
    print("Partial Result Detection Strategies")
    print("=" * 80)

    # Load model
    model, tokenizer = load()

    # Create transcriber with VAD disabled (punctuation-based)
    config = StreamingConfig(
        enable_partial_results=True,
        enable_vad=False,
    )
    transcriber = StreamingTranscriber(model, tokenizer, config=config)

    print("\n1. Punctuation-based Detection")
    print("-" * 80)

    test_cases = [
        ("Hello world.", "Should be FINAL (ends with period)"),
        ("Hello world!", "Should be FINAL (ends with exclamation)"),
        ("Hello world?", "Should be FINAL (ends with question mark)"),
        ("Hello world", "Should be PARTIAL (no punctuation)"),
        ("This is a long sentence with multiple words and clauses,", "Should be FINAL (long with comma)"),
        ("Short,", "Should be PARTIAL (too short with comma)"),
        ("First part; second part", "Should be FINAL (semicolon)"),
        ("Introduction: details", "Should be FINAL (colon)"),
    ]

    for text, expected in test_cases:
        is_final = transcriber._is_final_result(text)
        result = "FINAL" if is_final else "PARTIAL"
        status = "✓" if (is_final and "FINAL" in expected) or (not is_final and "PARTIAL" in expected) else "✗"
        print(f"{status} '{text}' → {result}")
        print(f"  {expected}")

    print("\n2. Buffer Exhaustion Detection")
    print("-" * 80)

    # Empty buffer case
    transcriber.buffer.clear()
    is_final = transcriber._is_final_result("No punctuation")
    print(f"Empty buffer, no punctuation → {'FINAL' if is_final else 'PARTIAL'}")
    print(f"  Expected: FINAL (buffer exhausted)")

    # Full buffer case
    audio = np.random.randn(int(5.0 * SAMPLE_RATE))
    transcriber.add_audio(audio)
    is_final = transcriber._is_final_result("No punctuation")
    print(f"Full buffer, no punctuation → {'FINAL' if is_final else 'PARTIAL'}")
    print(f"  Expected: PARTIAL (buffer has audio)")

    print("\n" + "=" * 80)


def demonstrate_throttling():
    """
    Demonstrate partial result throttling.

    Shows how partial results are throttled to avoid excessive updates
    while final results always get through.
    """
    print("=" * 80)
    print("Partial Result Throttling")
    print("=" * 80)

    model, tokenizer = load()

    # Configure with throttling
    config = StreamingConfig(
        enable_partial_results=True,
        partial_result_interval=1.0,  # 1 second throttle
    )
    transcriber = StreamingTranscriber(model, tokenizer, config=config)

    print("\nThrottle interval: 1.0 seconds")
    print("-" * 80)

    # Simulate rapid partial results
    print("\nSimulating rapid partial results (0.2s apart):")

    for i in range(5):
        # Mock is_final to return False (partial)
        is_final = False
        print(f"  Attempt {i+1}: is_final={is_final}, time_since_last={time.time() - transcriber.last_partial_time:.2f}s")

        if i == 0:
            # First one should go through
            transcriber.last_partial_time = time.time()
            print("    → Would emit (first partial)")
        else:
            # Others should be throttled
            time_since = time.time() - transcriber.last_partial_time
            if time_since < config.partial_result_interval:
                print("    → Would be THROTTLED")
            else:
                transcriber.last_partial_time = time.time()
                print("    → Would emit (throttle expired)")

        time.sleep(0.2)

    print("\nSimulating final results (always emit):")
    for i in range(3):
        is_final = True
        print(f"  Attempt {i+1}: is_final={is_final}")
        print("    → Would emit (final results bypass throttle)")
        time.sleep(0.2)

    print("\n" + "=" * 80)


def main():
    """Run all streaming partial results examples."""
    # Example 1: Detection strategies
    demonstrate_partial_detection_strategies()
    print("\n\n")

    # Example 2: Throttling
    demonstrate_throttling()
    print("\n\n")

    # Example 3: Real streaming (if audio file exists)
    # You can provide your own audio file path here
    audio_files = [
        "data/audio/speech/jfk.flac",
        "data/audio/speech/sample.wav",
        "data/audio/speech/sample.mp3",
    ]

    for audio_path in audio_files:
        if Path(audio_path).exists():
            print("\n\n")
            simulate_streaming_from_file(audio_path, enable_vad=False)
            break
    else:
        print("\n" + "=" * 80)
        print("NOTE: No audio files found for streaming demo.")
        print("To test streaming with real audio, place an audio file at:")
        print("  data/audio/speech/sample.wav")
        print("=" * 80)


if __name__ == "__main__":
    main()
