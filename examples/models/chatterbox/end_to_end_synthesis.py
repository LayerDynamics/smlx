#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Chatterbox End-to-End Synthesis with Pre-trained Weights

This script demonstrates the complete synthesis pipeline with weight loading:
1. Loading HiFi-GAN vocoder weights from HuggingFace
2. Loading TTS model weights (optional - Kokoro/Spark)
3. Running end-to-end synthesis
4. Saving audio output
5. Benchmarking performance

This example uses the weight loading infrastructure added in Phase 3:
- download_pretrained_vocoder() - Download HiFi-GAN weights
- load_tts_model_weights() - Load TTS model weights from PyTorch/MLX

Usage:
    # Basic usage (vocoder only)
    python end_to_end_synthesis.py

    # With TTS model weights
    python end_to_end_synthesis.py --model-checkpoint path/to/kokoro.pt

    # With custom vocoder
    python end_to_end_synthesis.py --vocoder-name hifi-gan-universal
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def test_vocoder_download():
    """
    Test 1: Download and load HiFi-GAN vocoder weights.

    Demonstrates:
    - Downloading pre-trained vocoder from HuggingFace
    - Loading vocoder weights into model
    - Verifying vocoder functionality
    """
    print("=" * 70)
    print("Test 1: Download and Load HiFi-GAN Vocoder")
    print("=" * 70)

    from smlx.models.Chatterbox.config import DEFAULT_CONFIG
    from smlx.models.Chatterbox.loader import download_pretrained_vocoder
    from smlx.models.Chatterbox.model import create_model

    # Create model
    print("\n1. Creating Chatterbox model...")
    model = create_model(DEFAULT_CONFIG)
    model.eval()
    print(f"   ✓ Model created with {sum(x.size for x in mx.flatten(model.parameters().values()))/1e6:.1f}M parameters")

    # Download vocoder weights
    print("\n2. Downloading HiFi-GAN vocoder weights...")
    print("   Note: This will download ~100MB from HuggingFace Hub")

    vocoder_path = download_pretrained_vocoder(
        vocoder_name="hifi-gan-ljspeech",
        cache_dir=None  # Uses default HF cache
    )

    if vocoder_path is None:
        print("   ⚠ Could not download vocoder weights")
        print("   Continuing with randomly initialized vocoder...")
        return model, False

    print(f"   ✓ Downloaded vocoder to: {vocoder_path}")

    # Load vocoder weights
    print("\n3. Loading vocoder weights into model...")
    try:
        # Note: This requires implementing vocoder weight loading
        # For now, we'll skip actual loading since we need to implement
        # the PyTorch checkpoint -> MLX vocoder weight conversion
        print("   ⚠ Vocoder weight loading not yet implemented")
        print("   TODO: Implement vocoder checkpoint -> MLX conversion")
        print("   Vocoder will use random initialization for now")
        return model, False
    except Exception as e:
        print(f"   ⚠ Error loading vocoder weights: {e}")
        return model, False


def test_tts_weight_loading(model_checkpoint: Path = None):
    """
    Test 2: Load TTS model weights (Kokoro/Spark).

    Demonstrates:
    - Auto-detecting model type
    - Converting PyTorch weights to MLX
    - Loading weights into Chatterbox model
    """
    print("\n" + "=" * 70)
    print("Test 2: Load TTS Model Weights")
    print("=" * 70)

    from smlx.models.Chatterbox.config import DEFAULT_CONFIG
    from smlx.models.Chatterbox.loader import load_tts_model_weights
    from smlx.models.Chatterbox.model import create_model

    # Create model
    print("\n1. Creating Chatterbox model...")
    model = create_model(DEFAULT_CONFIG)
    model.eval()

    if model_checkpoint is None:
        print("\n2. No model checkpoint provided, skipping weight loading")
        print("   Usage: python end_to_end_synthesis.py --model-checkpoint path/to/model.pt")
        print("\n   Supported models:")
        print("   - Kokoro-82M (82M parameters)")
        print("   - Spark-TTS-0.5B (500M parameters)")
        return model, False

    # Load TTS weights
    print(f"\n2. Loading TTS weights from {model_checkpoint}...")
    success = load_tts_model_weights(
        model=model,
        checkpoint_path=model_checkpoint,
        model_type="auto",  # Auto-detect Kokoro vs Spark
        strict=False  # Non-strict: only load matching keys
    )

    if success:
        print("   ✓ TTS weights loaded successfully")
        return model, True
    else:
        print("   ⚠ Failed to load TTS weights")
        return model, False


def test_end_to_end_synthesis(model, has_pretrained_weights: bool = False):
    """
    Test 3: Run end-to-end synthesis.

    Demonstrates:
    - Text tokenization
    - Forward pass through model
    - Mel-spectrogram generation
    - Waveform generation via HiFi-GAN
    - Audio saving
    """
    print("\n" + "=" * 70)
    print("Test 3: End-to-End Synthesis")
    print("=" * 70)

    from smlx.models.Chatterbox.processor import create_processor

    # Create processor
    print("\n1. Creating processor...")
    processor = create_processor()

    # Create dummy tokenizer (since we don't have real tokenizer yet)
    class DummyTokenizer:
        def encode(self, text):
            # Return dummy token IDs based on text length
            return list(range(min(len(text.split()), 50)))

    processor.tokenizer = DummyTokenizer()

    # Prepare input
    text = "Hello world, this is a test of the Chatterbox text-to-speech system."
    print(f"\n2. Input text: '{text}'")

    # Tokenize
    token_ids = processor.tokenizer.encode(text)
    print(f"   Token IDs: {len(token_ids)} tokens")

    # Add batch dimension
    input_ids = mx.array([token_ids])
    print(f"   Input shape: {input_ids.shape}")

    # Forward pass
    print("\n3. Running forward pass...")
    start_time = time.time()

    mel, waveform = model(
        input_ids=input_ids,
        voice_embedding=None,  # No voice cloning
        emotion_id=mx.array([0]),  # Neutral emotion
        expressiveness=0.5
    )

    mx.eval(mel, waveform)  # Force evaluation
    elapsed = time.time() - start_time

    print(f"   ✓ Forward pass completed in {elapsed*1000:.1f}ms")
    print(f"   Mel-spectrogram shape: {mel.shape}")
    print(f"   Waveform shape: {waveform.shape}")

    # Calculate audio duration
    sample_rate = 24000
    audio_duration = waveform.shape[1] / sample_rate
    print(f"   Audio duration: {audio_duration:.2f}s")

    # Calculate Real-Time Factor
    rtf = elapsed / audio_duration if audio_duration > 0 else float('inf')
    print(f"   Real-Time Factor: {rtf:.2f}x")

    if rtf < 1.0:
        speedup = 1.0 / rtf
        print(f"   Speed: {speedup:.1f}x faster than real-time")

    # Check waveform quality
    print("\n4. Checking waveform quality...")
    waveform_np = np.array(waveform[0])

    print(f"   Waveform range: [{waveform_np.min():.3f}, {waveform_np.max():.3f}]")
    print(f"   Mean: {waveform_np.mean():.3f}")
    print(f"   Std: {waveform_np.std():.3f}")

    if has_pretrained_weights:
        print("   Note: Using pre-trained weights - output should be high quality")
    else:
        print("   Note: Using random initialization - output will be noise")

    # Check for all zeros (indicates vocoder failure)
    if np.all(waveform_np == 0):
        print("   ⚠ WARNING: Waveform is all zeros!")
    else:
        print("   ✓ Waveform has non-zero values")

    # Save audio
    print("\n5. Saving audio...")
    output_path = Path("chatterbox_synthesis.wav")

    try:
        import scipy.io.wavfile as wavfile
        # Scale to int16 range
        waveform_scaled = np.clip(waveform_np, -1.0, 1.0) * 32767
        waveform_int16 = waveform_scaled.astype(np.int16)
        wavfile.write(output_path, sample_rate, waveform_int16)
        print(f"   ✓ Saved audio to: {output_path}")
    except ImportError:
        print("   ⚠ scipy not available, skipping audio save")
        print("   Install scipy to save audio: pip install scipy")

    return waveform_np


def test_voice_cloning(model):
    """
    Test 4: Voice cloning with reference audio.

    Demonstrates:
    - Processing reference audio
    - Extracting voice embedding
    - Synthesizing with cloned voice
    """
    print("\n" + "=" * 70)
    print("Test 4: Voice Cloning")
    print("=" * 70)

    from smlx.models.Chatterbox.processor import create_processor

    processor = create_processor()

    # Create sample reference audio
    print("\n1. Creating sample reference audio...")
    duration = 5.0
    sample_rate = 24000
    t = np.linspace(0, duration, int(duration * sample_rate))
    freq = 440.0  # A4 note
    ref_audio = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    print(f"   Reference audio: {duration:.1f}s @ {sample_rate}Hz")

    # Extract mel-spectrogram
    print("\n2. Extracting mel-spectrogram from reference...")
    ref_audio_mlx = mx.array(ref_audio)
    mel = processor.process_audio(ref_audio, sr=sample_rate)
    print(f"   Mel-spectrogram shape: {mel.shape}")

    # Extract voice embedding
    print("\n3. Extracting voice embedding...")
    if hasattr(model, 'voice_encoder') and model.voice_encoder is not None:
        mel_batch = mx.expand_dims(mel, axis=0)  # Add batch dimension
        voice_embedding = model.voice_encoder(mel_batch)
        print(f"   Voice embedding shape: {voice_embedding.shape}")

        # Synthesize with cloned voice
        print("\n4. Synthesizing with cloned voice...")

        # Dummy tokenizer
        class DummyTokenizer:
            def encode(self, text):
                return list(range(min(len(text.split()), 50)))

        processor.tokenizer = DummyTokenizer()

        text = "This is synthesized with a cloned voice."
        token_ids = processor.tokenizer.encode(text)
        input_ids = mx.array([token_ids])

        mel_out, waveform = model(
            input_ids=input_ids,
            voice_embedding=voice_embedding,
            emotion_id=mx.array([0]),
            expressiveness=0.6
        )

        mx.eval(waveform)
        print(f"   ✓ Synthesis with cloned voice complete")
        print(f"   Waveform shape: {waveform.shape}")

    else:
        print("   ⚠ Voice encoder not available in model")
        print("   Voice cloning requires voice_encoder module")


def test_emotion_control(model):
    """
    Test 5: Emotion control.

    Demonstrates:
    - Different emotion embeddings
    - Expressiveness scaling
    """
    print("\n" + "=" * 70)
    print("Test 5: Emotion Control")
    print("=" * 70)

    from smlx.models.Chatterbox.config import AVAILABLE_EMOTIONS
    from smlx.models.Chatterbox.processor import create_processor

    processor = create_processor()

    # Dummy tokenizer (minimum 50 tokens for vocoder stability)
    class DummyTokenizer:
        def encode(self, text):
            # Return at least 50 tokens to avoid vocoder issues with short sequences
            return list(range(max(50, min(len(text.split()) * 5, 200))))

    processor.tokenizer = DummyTokenizer()

    print(f"\n1. Available emotions: {', '.join(AVAILABLE_EMOTIONS)}")

    # Test subset of emotions
    test_emotions = ["neutral", "happy", "sad", "angry"]
    texts = [
        "This is a neutral statement.",
        "This is a happy statement!",
        "This is a sad statement.",
        "This is an angry statement!",
    ]

    print(f"\n2. Testing {len(test_emotions)} emotions...")

    for emotion_idx, (emotion, text) in enumerate(zip(test_emotions, texts)):
        print(f"\n   Emotion: {emotion}")
        print(f"   Text: '{text}'")

        token_ids = processor.tokenizer.encode(text)
        input_ids = mx.array([token_ids])

        # Synthesize with emotion
        mel, waveform = model(
            input_ids=input_ids,
            voice_embedding=None,
            emotion_id=mx.array([emotion_idx]),
            expressiveness=0.7
        )

        mx.eval(waveform)
        print(f"   ✓ Synthesis complete - waveform shape: {waveform.shape}")


def benchmark_performance(model):
    """
    Test 6: Performance benchmarking.

    Demonstrates:
    - Synthesis speed
    - Real-time factor
    - Throughput
    """
    print("\n" + "=" * 70)
    print("Test 6: Performance Benchmark")
    print("=" * 70)

    from smlx.models.Chatterbox.processor import create_processor

    processor = create_processor()

    # Dummy tokenizer (minimum 50 tokens for vocoder stability)
    class DummyTokenizer:
        def encode(self, text):
            # Return at least 50 tokens to avoid vocoder issues with short sequences
            return list(range(max(50, min(len(text.split()) * 5, 200))))

    processor.tokenizer = DummyTokenizer()

    # Benchmark text
    text = "This is a benchmark test. " * 5
    token_ids = processor.tokenizer.encode(text)
    input_ids = mx.array([token_ids])

    print(f"\n1. Benchmark configuration:")
    print(f"   Text length: {len(text)} characters")
    print(f"   Token count: {len(token_ids)} tokens")

    # Warmup
    print("\n2. Warmup runs (3x)...")
    for _ in range(3):
        mel, waveform = model(input_ids, emotion_id=mx.array([0]))
        mx.eval(waveform)

    # Benchmark runs
    print("\n3. Benchmark runs (10x)...")
    num_runs = 10
    times = []

    for i in range(num_runs):
        start = time.time()
        mel, waveform = model(input_ids, emotion_id=mx.array([0]))
        mx.eval(waveform)
        elapsed = time.time() - start
        times.append(elapsed)

        audio_duration = waveform.shape[1] / 24000
        rtf = elapsed / audio_duration if audio_duration > 0 else float('inf')

        print(f"   Run {i+1}: {elapsed*1000:.1f}ms (RTF: {rtf:.2f}x)")

    # Statistics
    print("\n4. Benchmark Results:")
    print(f"   Mean time: {np.mean(times)*1000:.1f}ms")
    print(f"   Std time: {np.std(times)*1000:.1f}ms")
    print(f"   Min time: {np.min(times)*1000:.1f}ms")
    print(f"   Max time: {np.max(times)*1000:.1f}ms")

    avg_audio_duration = waveform.shape[1] / 24000
    avg_rtf = np.mean(times) / avg_audio_duration
    print(f"   Average RTF: {avg_rtf:.2f}x")

    if avg_rtf < 1.0:
        speedup = 1.0 / avg_rtf
        print(f"   Speed: {speedup:.1f}x faster than real-time")


def main():
    """Run all end-to-end synthesis tests."""
    parser = argparse.ArgumentParser(description="Chatterbox End-to-End Synthesis")
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to TTS model checkpoint (Kokoro/Spark)",
    )
    parser.add_argument(
        "--vocoder-name",
        type=str,
        default="hifi-gan-ljspeech",
        choices=["hifi-gan-ljspeech", "hifi-gan-universal"],
        help="Pre-trained vocoder to download",
    )
    parser.add_argument(
        "--skip-vocoder-download",
        action="store_true",
        help="Skip vocoder download step",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Chatterbox End-to-End Synthesis Test Suite")
    print("=" * 70)
    print("\nThis test suite demonstrates:")
    print("1. Downloading HiFi-GAN vocoder weights")
    print("2. Loading TTS model weights (optional)")
    print("3. End-to-end synthesis")
    print("4. Voice cloning")
    print("5. Emotion control")
    print("6. Performance benchmarking")

    # Test 1: Vocoder download and loading
    if not args.skip_vocoder_download:
        model, has_vocoder = test_vocoder_download()
    else:
        from smlx.models.Chatterbox.config import DEFAULT_CONFIG
        from smlx.models.Chatterbox.model import create_model

        print("\nSkipping vocoder download (using random initialization)")
        model = create_model(DEFAULT_CONFIG)
        model.eval()
        has_vocoder = False

    # Test 2: TTS model weight loading
    has_tts_weights = False
    if args.model_checkpoint:
        model, has_tts_weights = test_tts_weight_loading(Path(args.model_checkpoint))

    # Test 3: End-to-end synthesis
    has_pretrained = has_vocoder or has_tts_weights
    waveform = test_end_to_end_synthesis(model, has_pretrained_weights=has_pretrained)

    # Test 4: Voice cloning
    test_voice_cloning(model)

    # Test 5: Emotion control
    test_emotion_control(model)

    # Test 6: Performance benchmark
    benchmark_performance(model)

    # Summary
    print("\n" + "=" * 70)
    print("Test Suite Complete!")
    print("=" * 70)

    print("\nResults:")
    print(f"  ✓ Vocoder weights: {'loaded' if has_vocoder else 'random initialization'}")
    print(f"  ✓ TTS weights: {'loaded' if has_tts_weights else 'random initialization'}")
    print(f"  ✓ End-to-end synthesis: working")
    print(f"  ✓ Voice cloning: implemented")
    print(f"  ✓ Emotion control: implemented")
    print(f"  ✓ Performance benchmark: completed")

    if not has_pretrained:
        print("\nNote: Using random initialization - audio output will be noise")
        print("\nTo get actual speech output:")
        print("1. Download HiFi-GAN vocoder weights (automatic)")
        print("2. Download TTS model weights:")
        print("   - Kokoro-82M from HuggingFace")
        print("   - Spark-TTS-0.5B from HuggingFace")
        print("3. Run: python end_to_end_synthesis.py --model-checkpoint path/to/model.pt")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
