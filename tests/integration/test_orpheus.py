#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for Orpheus-150M TTS model.

Tests speech synthesis, batch processing, streaming, and speed control.

Run with:
    python -m pytest tests/integration/test_orpheus.py -v
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
def orpheus_model():
    """
    Load Orpheus model once for all tests.

    Memory Requirements:
    - Model size: ~300MB (150M parameters in FP16)
    - Peak memory: ~600MB with activations
    """
    from smlx.models.Orpheus_150M import load

    model, processor = load()

    yield model, processor

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up Orpheus model...")
    del model
    del processor
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print("Cleanup complete. Memory freed.")


def test_model_loading(orpheus_model):
    """Test that Orpheus model loads successfully."""
    model, processor = orpheus_model

    assert model is not None, "Model should not be None"
    assert processor is not None, "Processor should not be None"
    assert hasattr(model, "text_encoder"), "Model should have text_encoder"
    assert hasattr(model, "duration_predictor"), "Model should have duration_predictor"
    assert hasattr(model, "decoder"), "Model should have decoder"
    assert hasattr(model, "vocoder"), "Model should have vocoder"


def test_basic_synthesis(orpheus_model):
    """Test basic text-to-speech synthesis."""
    from smlx.models.Orpheus_150M import synthesize

    model, processor = orpheus_model

    text = "Hello, this is a test of text to speech."
    audio = synthesize(
        model=model,
        processor=processor,
        text=text,
        sample_rate=24000,
    )

    assert audio is not None, "Audio output should not be None"
    assert isinstance(audio, np.ndarray), "Audio should be numpy array"
    assert len(audio) > 0, "Audio should have non-zero length"
    assert audio.dtype in [np.float32, np.float64], "Audio should be float type"


def test_batch_synthesis(orpheus_model):
    """Test batch synthesis of multiple texts."""
    from smlx.models.Orpheus_150M import synthesize_batch

    model, processor = orpheus_model

    texts = [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]

    audios = synthesize_batch(
        model=model,
        processor=processor,
        texts=texts,
    )

    assert len(audios) == len(texts), "Should have same number of audios as texts"

    for i, audio in enumerate(audios):
        assert audio is not None, f"Audio {i} should not be None"
        assert isinstance(audio, np.ndarray), f"Audio {i} should be numpy array"
        assert len(audio) > 0, f"Audio {i} should have content"


def test_streaming_synthesis(orpheus_model):
    """Test streaming synthesis functionality."""
    from smlx.models.Orpheus_150M import stream_synthesize

    model, processor = orpheus_model

    text = "This is a longer text for streaming synthesis testing. " * 3

    # Collect all chunks
    chunks = list(
        stream_synthesize(
            model=model,
            processor=processor,
            text=text,
            chunk_size=20,
        )
    )

    assert len(chunks) > 0, "Should generate at least one chunk"

    for i, chunk in enumerate(chunks):
        assert chunk is not None, f"Chunk {i} should not be None"
        assert isinstance(chunk, np.ndarray), f"Chunk {i} should be numpy array"
        assert len(chunk) > 0, f"Chunk {i} should have content"

    # Concatenate all chunks
    full_audio = np.concatenate(chunks)
    assert len(full_audio) > 0, "Concatenated audio should have content"


def test_speed_control(orpheus_model):
    """Test speech speed control."""
    from smlx.models.Orpheus_150M import synthesize_with_speed

    model, processor = orpheus_model

    text = "Testing speech speed control."

    # Test different speeds
    speeds = [0.75, 1.0, 1.5]

    for speed in speeds:
        audio = synthesize_with_speed(
            model=model,
            processor=processor,
            text=text,
            speed=speed,
        )

        assert audio is not None, f"Audio at speed {speed} should not be None"
        assert len(audio) > 0, f"Audio at speed {speed} should have content"


def test_duration_estimation(orpheus_model):
    """Test duration estimation functionality."""
    from smlx.models.Orpheus_150M import estimate_duration

    model, processor = orpheus_model

    texts = [
        "Short text.",
        "This is a medium length text for testing duration estimation.",
        "This is a much longer text that should have a longer estimated duration because it contains many more words and characters.",
    ]

    for text in texts:
        duration = estimate_duration(
            model=model,
            processor=processor,
            text=text,
            sample_rate=24000,
        )

        assert duration is not None, f"Duration for '{text[:30]}...' should not be None"
        assert duration > 0, f"Duration for '{text[:30]}...' should be positive"


def test_mel_spectrogram_extraction(orpheus_model):
    """Test mel-spectrogram extraction."""
    from smlx.models.Orpheus_150M import get_mel_spectrogram

    model, processor = orpheus_model

    text = "Test mel-spectrogram extraction."
    mel = get_mel_spectrogram(
        model=model,
        processor=processor,
        text=text,
    )

    assert mel is not None, "Mel-spectrogram should not be None"
    assert len(mel.shape) == 2, "Mel-spectrogram should be 2D (time, mel_bins)"
    assert mel.shape[0] > 0, "Should have time frames"
    assert mel.shape[1] > 0, "Should have mel bins"


def test_audio_normalization(orpheus_model):
    """Test that output audio is properly normalized."""
    from smlx.models.Orpheus_150M import synthesize

    model, processor = orpheus_model

    audio = synthesize(
        model=model,
        processor=processor,
        text="Normalization test.",
        sample_rate=24000,
    )

    # Check audio is in reasonable range (typically [-1, 1] for normalized audio)
    assert audio.min() >= -2.0, "Audio minimum should be reasonable"
    assert audio.max() <= 2.0, "Audio maximum should be reasonable"


def test_empty_text(orpheus_model):
    """Test handling of empty text."""
    from smlx.models.Orpheus_150M import synthesize

    model, processor = orpheus_model

    # Should handle empty text gracefully
    audio = synthesize(
        model=model,
        processor=processor,
        text="",
        sample_rate=24000,
    )

    # Should not crash (might return zeros or very short audio)
    assert audio is not None, "Should handle empty text"


def test_long_text(orpheus_model):
    """Test handling of long text."""
    from smlx.models.Orpheus_150M import synthesize

    model, processor = orpheus_model

    # Create long text
    long_text = "This is a test sentence. " * 20

    audio = synthesize(
        model=model,
        processor=processor,
        text=long_text,
        sample_rate=24000,
    )

    assert audio is not None, "Should handle long text"
    assert len(audio) > 0, "Should generate audio for long text"


def test_special_characters(orpheus_model):
    """Test handling of special characters and punctuation."""
    from smlx.models.Orpheus_150M import synthesize

    model, processor = orpheus_model

    texts = [
        "Hello, world!",
        "What's this? A test!",
        "Numbers: 1, 2, 3.",
        "Email: test@example.com",
    ]

    for text in texts:
        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            sample_rate=24000,
        )

        assert audio is not None, f"Should handle text: '{text}'"
        assert len(audio) > 0, f"Should generate audio for: '{text}'"


def test_different_sample_rates(orpheus_model):
    """Test synthesis with different sample rates."""
    from smlx.models.Orpheus_150M import synthesize

    model, processor = orpheus_model

    text = "Testing different sample rates."
    sample_rates = [16000, 22050, 24000]

    for sr in sample_rates:
        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            sample_rate=sr,
        )

        assert audio is not None, f"Should work with sample rate {sr}"
        assert len(audio) > 0, f"Should generate audio at {sr}Hz"


def test_speed_edge_cases(orpheus_model):
    """Test speed control with edge cases."""
    from smlx.models.Orpheus_150M import synthesize_with_speed

    model, processor = orpheus_model

    text = "Speed edge case testing."

    # Test extreme speeds
    speeds = [0.5, 2.0]

    for speed in speeds:
        audio = synthesize_with_speed(
            model=model,
            processor=processor,
            text=text,
            speed=speed,
        )

        assert audio is not None, f"Should handle speed {speed}"
        assert len(audio) > 0, f"Should generate audio at speed {speed}"


def test_model_info():
    """Test model info functions."""
    from smlx.models.Orpheus_150M import get_model_info, load

    model, processor = load()

    # Get model info
    info = get_model_info(model)

    assert info is not None, "Model info should not be None"
    assert isinstance(info, dict), "Model info should be a dictionary"


def test_configuration():
    """Test configuration loading."""
    from smlx.models.Orpheus_150M import DEFAULT_CONFIG, Orpheus150MConfig

    # Test DEFAULT_CONFIG exists
    assert DEFAULT_CONFIG is not None, "DEFAULT_CONFIG should exist"
    assert isinstance(DEFAULT_CONFIG, Orpheus150MConfig), "Should be Orpheus150MConfig"

    # Test config attributes
    assert hasattr(DEFAULT_CONFIG, "text_encoder_config"), "Should have text_encoder_config"
    assert hasattr(DEFAULT_CONFIG, "duration_config"), "Should have duration_config"
    assert hasattr(DEFAULT_CONFIG, "decoder_config"), "Should have decoder_config"
    assert hasattr(DEFAULT_CONFIG, "vocoder_config"), "Should have vocoder_config"


def test_streaming_chunk_sizes(orpheus_model):
    """Test streaming with different chunk sizes."""
    from smlx.models.Orpheus_150M import stream_synthesize

    model, processor = orpheus_model

    text = "Testing different chunk sizes for streaming synthesis."

    # Test different chunk sizes
    chunk_sizes = [10, 30, 50]

    for chunk_size in chunk_sizes:
        chunks = list(
            stream_synthesize(
                model=model,
                processor=processor,
                text=text,
                chunk_size=chunk_size,
            )
        )

        assert len(chunks) > 0, f"Should generate chunks with size {chunk_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
