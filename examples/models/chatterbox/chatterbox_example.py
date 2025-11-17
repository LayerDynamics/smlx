#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Chatterbox Voice Cloning TTS Examples

This script demonstrates voice cloning and expressive TTS with Chatterbox:
1. Basic synthesis with emotions
2. Voice cloning from reference audio
3. Expressiveness control
4. Emotion demonstration (all 8 emotions)
5. Batch synthesis with same voice
6. Multi-emotion synthesis
7. Performance benchmarking

IMPORTANT NOTE:
    These examples use Chatterbox with a working HiFi-GAN vocoder.
    The model generates real audio waveforms from mel-spectrograms.

    For production-quality output, you need:
    1. Pre-trained HiFi-GAN vocoder weights (download from HuggingFace)
    2. Pre-trained TTS model weights for voice cloning

    Recommended models to try:
    - facebook/tts_transformer
    - suno/bark
    - coqui/XTTS-v2

Usage:
    python chatterbox_example.py
"""

import time

import numpy as np


def create_sample_audio(duration: float = 5.0, sample_rate: int = 24000) -> np.ndarray:
    """
    Create sample reference audio for voice cloning demo.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate

    Returns:
        Sample audio waveform
    """
    # Generate simple sine wave as placeholder
    t = np.linspace(0, duration, int(duration * sample_rate))
    freq = 440.0  # A4 note
    audio = 0.3 * np.sin(2 * np.pi * freq * t)

    return audio.astype(np.float32)


def example_1_basic_synthesis():
    """
    Example 1: Basic synthesis with emotions.

    Demonstrates:
    - Loading Chatterbox model
    - Synthesizing with different emotions
    - Saving audio files
    """
    print("=" * 70)
    print("Example 1: Basic Synthesis with Emotions")
    print("=" * 70)

    from smlx.models.Chatterbox import load, save_audio, synthesize

    # Load model
    print("\n1. Loading Chatterbox model...")
    model, processor = load()

    # Test different emotions
    emotions = ["neutral", "happy", "excited"]
    texts = [
        "This is a neutral statement.",
        "This is happy and cheerful!",
        "This is very exciting news!!!",
    ]

    print("\n2. Synthesizing with different emotions...")
    for emotion, text in zip(emotions, texts):
        print(f"\n   Emotion: {emotion}")
        print(f"   Text: {text}")

        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            emotion=emotion,
            expressiveness=0.7,
        )

        output_path = f"chatterbox_{emotion}.wav"
        save_audio(audio, output_path, sample_rate=24000)

    print(
        "\nNote: Placeholder audio generated. Load pre-trained weights for actual synthesis."
    )


def example_2_voice_cloning():
    """
    Example 2: Voice cloning from reference audio.

    Demonstrates:
    - Creating/loading reference audio
    - Cloning voice
    - Synthesizing with cloned voice
    """
    print("\n" + "=" * 70)
    print("Example 2: Voice Cloning")
    print("=" * 70)

    from smlx.models.Chatterbox import clone_voice, load, save_audio, synthesize

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    # Create sample reference audio
    # In production, load real audio: ref_audio, sr = sf.read("my_voice.wav")
    print("\n2. Creating sample reference audio...")
    ref_audio = create_sample_audio(duration=5.0, sample_rate=24000)
    print(f"   Reference audio: {len(ref_audio)/24000:.1f}s")

    # Clone voice
    print("\n3. Cloning voice from reference...")
    voice_embedding = clone_voice(
        model=model, processor=processor, reference_audio=ref_audio, sample_rate=24000
    )

    print(f"   Voice embedding shape: {voice_embedding.shape}")

    # Synthesize with cloned voice
    print("\n4. Synthesizing with cloned voice...")
    texts = [
        "Hello, this is my cloned voice.",
        "I can speak anything with this voice.",
        "Voice cloning is amazing!",
    ]

    for i, text in enumerate(texts):
        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            voice_embedding=voice_embedding,
            emotion="neutral",
            expressiveness=0.6,
        )

        output_path = f"chatterbox_cloned_{i+1}.wav"
        save_audio(audio, output_path, sample_rate=24000)
        print(f"   Saved: {output_path}")

    print(
        "\nNote: Use real reference audio (3-10s) with pre-trained model for actual voice cloning."
    )


def example_3_expressiveness_control():
    """
    Example 3: Expressiveness control.

    Demonstrates:
    - Different expressiveness levels (0.0 to 1.0)
    - Impact on speech naturalness
    """
    print("\n" + "=" * 70)
    print("Example 3: Expressiveness Control")
    print("=" * 70)

    from smlx.models.Chatterbox import load, save_audio, synthesize

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    text = "This is a test of expressiveness control."

    # Different expressiveness levels
    expressiveness_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    level_names = ["monotone", "low", "medium", "high", "very_high"]

    print(f"\n2. Generating speech with different expressiveness...")
    for exp_level, name in zip(expressiveness_levels, level_names):
        print(f"\n   Expressiveness: {exp_level:.2f} ({name})")

        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            emotion="neutral",
            expressiveness=exp_level,
        )

        output_path = f"chatterbox_exp_{name}.wav"
        save_audio(audio, output_path, sample_rate=24000)

    print("\nNote: Expressiveness ranges from 0 (monotone) to 1 (very expressive)")


def example_4_all_emotions():
    """
    Example 4: Demonstrate all available emotions.

    Demonstrates:
    - All 8 emotion categories
    - Emotion-appropriate text
    """
    print("\n" + "=" * 70)
    print("Example 4: All Emotions Demo")
    print("=" * 70)

    from smlx.models.Chatterbox import demo_emotions, get_available_emotions, load

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    # Get available emotions
    emotions = get_available_emotions()
    print(f"\n2. Available emotions: {', '.join(emotions)}")

    # Generate speech for all emotions
    text = "This is a test of emotion control in speech synthesis."
    print(f"\n3. Generating '{text}' with all emotions...")

    emotion_audios = demo_emotions(model=model, processor=processor, text=text)

    # Save all emotion audio files
    print("\n4. Saving audio files...")
    from smlx.models.Chatterbox import save_audio

    for emotion, audio in emotion_audios.items():
        output_path = f"chatterbox_emotion_{emotion}.wav"
        save_audio(audio, output_path, sample_rate=24000)
        print(f"   {emotion}: {output_path}")


def example_5_batch_synthesis():
    """
    Example 5: Batch synthesis with same voice.

    Demonstrates:
    - Cloning voice once
    - Using for multiple texts
    - Efficient batch processing
    """
    print("\n" + "=" * 70)
    print("Example 5: Batch Synthesis")
    print("=" * 70)

    from smlx.models.Chatterbox import (
        clone_voice,
        load,
        save_audio,
        synthesize_batch,
    )

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    # Clone voice
    print("\n2. Cloning voice...")
    ref_audio = create_sample_audio(duration=5.0, sample_rate=24000)
    voice_embedding = clone_voice(
        model=model, processor=processor, reference_audio=ref_audio, sample_rate=24000
    )

    # Batch texts
    texts = [
        "Welcome to the batch synthesis demo.",
        "This uses the same cloned voice for all sentences.",
        "Batch processing is more efficient than individual synthesis.",
        "All sentences maintain the same voice characteristics.",
        "This is perfect for audiobook generation.",
    ]

    # Synthesize batch
    print(f"\n3. Synthesizing {len(texts)} texts...")
    audios = synthesize_batch(
        model=model,
        processor=processor,
        texts=texts,
        voice_embedding=voice_embedding,
        emotion="neutral",
        expressiveness=0.6,
    )

    # Save files
    print("\n4. Saving batch audio files...")
    for i, audio in enumerate(audios):
        output_path = f"chatterbox_batch_{i+1}.wav"
        save_audio(audio, output_path, sample_rate=24000)
        print(f"   Sentence {i+1}: {output_path}")


def example_6_multi_emotion_synthesis():
    """
    Example 6: Synthesize with different emotions.

    Demonstrates:
    - Different emotions for different texts
    - Emotion-appropriate expressiveness
    """
    print("\n" + "=" * 70)
    print("Example 6: Multi-Emotion Synthesis")
    print("=" * 70)

    from smlx.models.Chatterbox import load, save_audio, synthesize_with_emotions

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    # Emotion-appropriate texts
    texts = [
        "This is a neutral statement.",
        "I'm so happy and excited!",
        "This is very sad news.",
        "I'm very angry about this!",
    ]

    emotions = ["neutral", "excited", "sad", "angry"]

    print(f"\n2. Synthesizing {len(texts)} texts with matching emotions...")
    audios = synthesize_with_emotions(
        model=model,
        processor=processor,
        texts=texts,
        emotions=emotions,
        expressiveness=0.8,
    )

    # Save files
    print("\n3. Saving audio files...")
    for i, (audio, emotion, text) in enumerate(zip(audios, emotions, texts)):
        output_path = f"chatterbox_multi_{i+1}_{emotion}.wav"
        save_audio(audio, output_path, sample_rate=24000)
        print(f"   {emotion}: '{text[:40]}...'")
        print(f"   Saved: {output_path}")


def example_7_performance_benchmark():
    """
    Example 7: Performance benchmarking.

    Demonstrates:
    - Measuring synthesis speed
    - Real-time factor calculation
    - Voice cloning overhead
    """
    print("\n" + "=" * 70)
    print("Example 7: Performance Benchmark")
    print("=" * 70)

    from smlx.models.Chatterbox import clone_voice, load, print_model_info, synthesize

    # Load model
    print("\n1. Loading model...")
    model, processor = load()

    # Print model info
    print_model_info(model)

    # Benchmark voice cloning
    print("\n2. Benchmarking voice cloning...")
    ref_audio = create_sample_audio(duration=5.0, sample_rate=24000)

    num_runs = 5
    clone_times = []

    for i in range(num_runs):
        start = time.time()
        voice_embedding = clone_voice(
            model=model,
            processor=processor,
            reference_audio=ref_audio,
            sample_rate=24000,
        )
        elapsed = time.time() - start
        clone_times.append(elapsed)
        print(f"   Run {i+1}: {elapsed*1000:.1f}ms")

    avg_clone_time = np.mean(clone_times)
    print(f"\n   Average voice cloning time: {avg_clone_time*1000:.1f}ms")

    # Benchmark synthesis
    print("\n3. Benchmarking synthesis...")
    text = "This is a benchmark test for speech synthesis. " * 3

    num_runs = 10
    synth_times = []

    for i in range(num_runs):
        start = time.time()
        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            voice_embedding=voice_embedding,
            emotion="neutral",
            expressiveness=0.6,
        )
        elapsed = time.time() - start
        synth_times.append(elapsed)

        audio_duration = len(audio) / 24000
        rtf = elapsed / audio_duration

        print(f"   Run {i+1}: {elapsed*1000:.1f}ms (RTF: {rtf:.2f}x)")

    # Statistics
    avg_time = np.mean(synth_times)
    min_time = np.min(synth_times)
    max_time = np.max(synth_times)

    audio_duration = len(audio) / 24000
    avg_rtf = avg_time / audio_duration

    print(f"\nBenchmark Results:")
    print(f"  Average time: {avg_time*1000:.1f}ms")
    print(f"  Min time: {min_time*1000:.1f}ms")
    print(f"  Max time: {max_time*1000:.1f}ms")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  Real-Time Factor: {avg_rtf:.2f}x")

    if avg_rtf < 1.0:
        speedup = 1.0 / avg_rtf
        print(f"  Speed: {speedup:.1f}x faster than real-time")

    print("\nNote: Performance will improve with actual implementation and GPU.")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Chatterbox Voice Cloning TTS Examples")
    print("=" * 70)
    print("\nIMPORTANT: These examples use a reference implementation.")
    print("For production use, search HuggingFace for voice cloning TTS:")
    print("  - facebook/tts_transformer")
    print("  - suno/bark")
    print("  - coqui/XTTS-v2")
    print("\nRunning 7 examples:\n")

    examples = [
        ("1. Basic Synthesis", example_1_basic_synthesis),
        ("2. Voice Cloning", example_2_voice_cloning),
        ("3. Expressiveness Control", example_3_expressiveness_control),
        ("4. All Emotions", example_4_all_emotions),
        ("5. Batch Synthesis", example_5_batch_synthesis),
        ("6. Multi-Emotion Synthesis", example_6_multi_emotion_synthesis),
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
    print("1. Search HuggingFace for voice cloning TTS models")
    print("2. Load pre-trained weights")
    print("3. Integrate voice cloning into your application")
    print("4. Fine-tune on your voice data if needed")
    print("5. See docs/ModelImplementations.md for details")


if __name__ == "__main__":
    main()
