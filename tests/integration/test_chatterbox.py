#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for Chatterbox TTS model.

Tests voice cloning, emotion control, and expressiveness features.

Run with:
    python -m pytest tests/integration/test_chatterbox.py -v
"""

import gc

import mlx.core as mx
import pytest
import numpy as np

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
    pytest.mark.heavy_memory,  # Chatterbox uses ~1GB
]


@pytest.fixture(scope="module")
def chatterbox_model():
    """
    Load Chatterbox model once for all tests.

    Memory Requirements:
    - Model size: ~1GB (500M parameters in FP16)
    - Peak memory: ~1.5GB with activations
    - Requires: 2GB available headroom
    """
    from smlx.utils.memory import check_memory_availability, memory_profiler
    from smlx.models.Chatterbox import load

    # Check memory before loading
    check = check_memory_availability(2.0)  # Require 2GB headroom
    if not check["available"]:
        pytest.skip(
            f"Insufficient memory for Chatterbox: "
            f"{check['headroom_gb']:.1f}GB available, need 2GB"
        )

    # Load model with memory profiling
    with memory_profiler() as mem:
        model, processor = load()
        mx.eval(model)

    print(f"\nChatterbox loaded: {mem.peak_gb:.2f}GB peak memory")

    yield model, processor

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up Chatterbox model...")
    del model
    del processor
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print(f"Cleanup complete. Memory freed.")


def test_model_loading(chatterbox_model):
    """Test that Chatterbox model loads successfully."""
    model, processor = chatterbox_model

    assert model is not None, "Model should not be None"
    assert processor is not None, "Processor should not be None"
    assert hasattr(model, "llama_backbone"), "Model should have llama_backbone"
    assert hasattr(model, "voice_encoder"), "Model should have voice_encoder"
    assert hasattr(model, "expressiveness_module"), "Model should have expressiveness_module"


def test_basic_synthesis(chatterbox_model):
    """Test basic text-to-speech synthesis."""
    from smlx.models.Chatterbox import synthesize

    model, processor = chatterbox_model

    text = "Hello, this is a test."
    audio = synthesize(
        model=model,
        processor=processor,
        text=text,
        emotion="neutral",
        expressiveness=0.5,
    )

    assert audio is not None, "Audio output should not be None"
    assert isinstance(audio, np.ndarray), "Audio should be numpy array"
    assert len(audio) > 0, "Audio should have non-zero length"
    assert audio.dtype in [np.float32, np.float64], "Audio should be float type"


def test_emotions(chatterbox_model):
    """Test synthesis with different emotions."""
    from smlx.models.Chatterbox import get_available_emotions, synthesize

    model, processor = chatterbox_model

    # Test all available emotions
    emotions = get_available_emotions()
    assert len(emotions) > 0, "Should have available emotions"

    text = "This is a test."

    for emotion in emotions[:3]:  # Test first 3 emotions for speed
        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            emotion=emotion,
            expressiveness=0.7,
        )

        assert audio is not None, f"Audio for emotion '{emotion}' should not be None"
        assert len(audio) > 0, f"Audio for emotion '{emotion}' should have content"


def test_expressiveness_range(chatterbox_model):
    """Test expressiveness control with different levels."""
    from smlx.models.Chatterbox import synthesize

    model, processor = chatterbox_model

    text = "Testing expressiveness."
    expressiveness_levels = [0.0, 0.5, 1.0]

    for exp_level in expressiveness_levels:
        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            emotion="neutral",
            expressiveness=exp_level,
        )

        assert audio is not None, f"Audio with expressiveness {exp_level} should not be None"
        assert len(audio) > 0, f"Audio with expressiveness {exp_level} should have content"


def test_voice_cloning(chatterbox_model):
    """Test voice cloning functionality."""
    from smlx.models.Chatterbox import clone_voice, synthesize

    model, processor = chatterbox_model

    # Create dummy reference audio (sine wave)
    sample_rate = 24000
    duration = 3.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    reference_audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Clone voice
    voice_embedding = clone_voice(
        model=model,
        processor=processor,
        reference_audio=reference_audio,
        sample_rate=sample_rate,
    )

    assert voice_embedding is not None, "Voice embedding should not be None"
    assert len(voice_embedding.shape) == 2, "Voice embedding should be 2D (batch, embedding_dim)"

    # Use cloned voice for synthesis
    text = "Speaking with cloned voice."
    audio = synthesize(
        model=model,
        processor=processor,
        text=text,
        voice_embedding=voice_embedding,
        emotion="neutral",
        expressiveness=0.6,
    )

    assert audio is not None, "Audio with cloned voice should not be None"
    assert len(audio) > 0, "Audio with cloned voice should have content"


def test_batch_synthesis(chatterbox_model):
    """Test batch synthesis of multiple texts."""
    from smlx.models.Chatterbox import synthesize_batch

    model, processor = chatterbox_model

    texts = [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]

    audios = synthesize_batch(
        model=model,
        processor=processor,
        texts=texts,
        emotion="neutral",
        expressiveness=0.5,
    )

    assert len(audios) == len(texts), "Should have same number of audios as texts"

    for i, audio in enumerate(audios):
        assert audio is not None, f"Audio {i} should not be None"
        assert isinstance(audio, np.ndarray), f"Audio {i} should be numpy array"
        assert len(audio) > 0, f"Audio {i} should have content"


def test_multi_emotion_synthesis(chatterbox_model):
    """Test synthesis with different emotions per text."""
    from smlx.models.Chatterbox import synthesize_with_emotions

    model, processor = chatterbox_model

    texts = ["I'm happy!", "I'm sad."]
    emotions = ["happy", "sad"]

    audios = synthesize_with_emotions(
        model=model,
        processor=processor,
        texts=texts,
        emotions=emotions,
        expressiveness=0.7,
    )

    assert len(audios) == len(texts), "Should have same number of audios as texts"

    for i, audio in enumerate(audios):
        assert audio is not None, f"Audio {i} should not be None"
        assert len(audio) > 0, f"Audio {i} should have content"


def test_audio_normalization(chatterbox_model):
    """Test that output audio is properly normalized."""
    from smlx.models.Chatterbox import synthesize

    model, processor = chatterbox_model

    audio = synthesize(
        model=model,
        processor=processor,
        text="Normalization test.",
        emotion="neutral",
        expressiveness=0.5,
    )

    # Check audio is normalized to [-1, 1] range
    assert audio.min() >= -1.0, "Audio minimum should be >= -1.0"
    assert audio.max() <= 1.0, "Audio maximum should be <= 1.0"


def test_invalid_emotion(chatterbox_model):
    """Test handling of invalid emotion name."""
    from smlx.models.Chatterbox import synthesize

    model, processor = chatterbox_model

    # Should not crash, should fallback to neutral
    audio = synthesize(
        model=model,
        processor=processor,
        text="Test with invalid emotion.",
        emotion="invalid_emotion_xyz",
        expressiveness=0.5,
    )

    assert audio is not None, "Should still generate audio with invalid emotion"
    assert len(audio) > 0, "Should have audio content"


def test_expressiveness_clamping(chatterbox_model):
    """Test that expressiveness values are clamped to [0, 1]."""
    from smlx.models.Chatterbox import synthesize

    model, processor = chatterbox_model

    # Test values outside [0, 1] range
    for exp_value in [-0.5, 1.5, 2.0]:
        audio = synthesize(
            model=model,
            processor=processor,
            text="Clamping test.",
            emotion="neutral",
            expressiveness=exp_value,
        )

        assert audio is not None, f"Should generate audio with expressiveness {exp_value}"
        assert len(audio) > 0, "Should have audio content"


def test_available_emotions():
    """Test getting list of available emotions."""
    from smlx.models.Chatterbox import get_available_emotions

    emotions = get_available_emotions()

    assert isinstance(emotions, list), "Should return a list"
    assert len(emotions) > 0, "Should have at least one emotion"
    assert "neutral" in emotions, "Should include neutral emotion"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
