#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for YAMNet audio event classification.

Tests audio classification, embeddings, event detection, batch processing.

Run with:
    python -m pytest tests/integration/test_yamnet.py -v
"""

import gc

import mlx.core as mx
import pytest
import numpy as np

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
]


def create_test_audio(duration=1.0, sample_rate=16000, frequency=440):
    """Create a simple sine wave test audio."""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    # Generate sine wave
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture(scope="module")
def yamnet_model():
    """
    Load YAMNet model once for all tests.

    Memory Requirements:
    - Model size: ~100MB
    - Peak memory: ~200MB with activations
    """
    from smlx.models.YAMNet import load

    model = load()

    yield model

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up YAMNet model...")
    del model
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print("Cleanup complete. Memory freed.")


def test_model_loading(yamnet_model):
    """Test that YAMNet model loads successfully."""
    model = yamnet_model

    assert model is not None, "Model should not be None"
    # Check for actual model structure (MobileNet-v1 layers)
    assert hasattr(model, "conv1"), "Model should have initial conv layer"
    assert hasattr(model, "conv_blocks"), "Model should have depthwise separable conv blocks"
    assert hasattr(model, "embedding"), "Model should have embedding layer"
    assert hasattr(model, "classifier"), "Model should have classifier"


def test_basic_classification(yamnet_model):
    """Test basic audio classification."""
    from smlx.models.YAMNet import classify

    model = yamnet_model

    # Create test audio
    test_audio = create_test_audio(duration=1.0)

    # Classify
    predictions = classify(model, test_audio, sample_rate=16000, top_k=5)

    assert predictions is not None, "Predictions should not be None"
    assert len(predictions) > 0, "Should have predictions"
    assert len(predictions) <= 5, "Should respect top_k=5"

    # Check prediction format
    for pred in predictions:
        assert hasattr(pred, "label"), "Prediction should have label"
        assert hasattr(pred, "score"), "Prediction should have score"
        assert 0.0 <= pred.score <= 1.0, "Score should be between 0 and 1"


def test_classify_with_top_k(yamnet_model):
    """Test classification with different top_k values."""
    from smlx.models.YAMNet import classify

    model = yamnet_model

    # Create test audio
    test_audio = create_test_audio()

    # Test different top_k values
    for k in [1, 3, 10]:
        predictions = classify(model, test_audio, sample_rate=16000, top_k=k)

        assert len(predictions) <= k, f"Should have at most {k} predictions"


def test_batch_classification(yamnet_model):
    """Test batch classification."""
    from smlx.models.YAMNet import classify_batch

    model = yamnet_model

    # Create multiple test audio samples
    audios = [
        create_test_audio(frequency=220),
        create_test_audio(frequency=440),
        create_test_audio(frequency=880),
    ]

    # Classify batch
    batch_predictions = classify_batch(model, audios, sample_rate=16000, top_k=3)

    assert len(batch_predictions) == len(audios), "Should have predictions for each audio"

    for predictions in batch_predictions:
        assert predictions is not None, "Each prediction should not be None"
        assert len(predictions) > 0, "Should have predictions"


def test_extract_embeddings(yamnet_model):
    """Test audio embedding extraction."""
    from smlx.models.YAMNet import extract_embeddings
    import mlx.core as mx

    model = yamnet_model

    # Create test audio
    test_audio = create_test_audio()

    # Extract embeddings
    embeddings = extract_embeddings(model, test_audio, sample_rate=16000)

    assert embeddings is not None, "Embeddings should not be None"
    assert isinstance(embeddings, mx.array), "Embeddings should be MLX array"
    assert len(embeddings.shape) == 2, "Embeddings should be 2D (frames, features)"
    assert embeddings.shape[1] == 1024, "Should have 1024-dimensional embeddings"


def test_detect_events(yamnet_model):
    """Test event detection."""
    from smlx.models.YAMNet import detect_events

    model = yamnet_model

    # Create test audio
    test_audio = create_test_audio(duration=3.0)

    # Detect events
    events = detect_events(model, test_audio, sample_rate=16000, threshold=0.3)

    assert events is not None, "Events should not be None"
    assert isinstance(events, list), "Events should be a list"


def test_audio_similarity(yamnet_model):
    """Test audio similarity computation."""
    from smlx.models.YAMNet import compute_audio_similarity

    model = yamnet_model

    # Create two test audios
    audio1 = create_test_audio(frequency=440)
    audio2 = create_test_audio(frequency=440)  # Same frequency

    # Compute similarity
    similarity = compute_audio_similarity(
        model, audio1, audio2, sample_rate=16000
    )

    assert similarity is not None, "Similarity should not be None"
    # Allow small floating point tolerance
    assert (
        -1e-6 <= similarity <= 1.0 + 1e-6
    ), f"Similarity should be between 0 and 1, got {similarity}"


def test_audio_loading():
    """Test audio loading utilities."""
    from smlx.models.YAMNet import load_audio
    import mlx.core as mx

    # Create test audio
    test_audio = create_test_audio()

    # Should handle numpy arrays
    loaded = load_audio(test_audio, target_sr=16000)

    assert isinstance(loaded, mx.array), "Should return MLX array"
    assert loaded.size > 0, "Should have samples"


def test_preprocess_audio():
    """Test audio preprocessing."""
    from smlx.models.YAMNet import preprocess_audio

    # Create test audio
    test_audio = create_test_audio()

    # Preprocess
    processed = preprocess_audio(test_audio, sample_rate=16000)

    assert processed is not None, "Processed audio should not be None"


def test_mel_spectrogram():
    """Test mel spectrogram computation."""
    from smlx.models.YAMNet import compute_mel_spectrogram

    # Create test audio
    test_audio = create_test_audio()

    # Compute mel spectrogram
    mel_spec = compute_mel_spectrogram(test_audio, sample_rate=16000)

    assert mel_spec is not None, "Mel spectrogram should not be None"
    assert len(mel_spec.shape) == 2, "Mel spec should be 2D"


def test_extract_patches():
    """Test patch extraction from spectrograms."""
    from smlx.models.YAMNet import extract_patches, compute_mel_spectrogram
    import mlx.core as mx

    # Create test audio
    test_audio = create_test_audio(duration=2.0)

    # Compute mel spectrogram
    mel_spec = compute_mel_spectrogram(test_audio, sample_rate=16000)

    # Extract patches
    patches = extract_patches(mel_spec)

    assert patches is not None, "Patches should not be None"
    assert isinstance(patches, mx.array), "Patches should be MLX array"


def test_class_names():
    """Test AudioSet class names."""
    from smlx.models.YAMNet import AUDIOSET_CLASSES, load_class_names

    # Check AUDIOSET_CLASSES constant
    assert AUDIOSET_CLASSES is not None, "AUDIOSET_CLASSES should exist"
    assert len(AUDIOSET_CLASSES) == 521, "Should have 521 AudioSet classes"

    # Load class names
    class_names = load_class_names()
    assert class_names is not None, "Class names should not be None"
    assert len(class_names) == 521, "Should have 521 class names"


def test_different_audio_lengths(yamnet_model):
    """Test with different audio lengths."""
    from smlx.models.YAMNet import classify

    model = yamnet_model

    # Test different durations
    durations = [0.5, 1.0, 2.0, 5.0]

    for duration in durations:
        test_audio = create_test_audio(duration=duration)
        predictions = classify(model, test_audio, sample_rate=16000, top_k=3)

        assert predictions is not None, f"Should work with {duration}s audio"


def test_different_frequencies(yamnet_model):
    """Test with different audio frequencies."""
    from smlx.models.YAMNet import classify

    model = yamnet_model

    # Test different frequencies
    frequencies = [220, 440, 880, 1760]

    for freq in frequencies:
        test_audio = create_test_audio(frequency=freq)
        predictions = classify(model, test_audio, sample_rate=16000, top_k=3)

        assert predictions is not None, f"Should work with {freq}Hz audio"


def test_silent_audio(yamnet_model):
    """Test with silent audio."""
    from smlx.models.YAMNet import classify

    model = yamnet_model

    # Create silent audio
    silent_audio = np.zeros(16000, dtype=np.float32)

    # Should handle gracefully
    predictions = classify(model, silent_audio, sample_rate=16000, top_k=5)

    assert predictions is not None, "Should handle silent audio"


def test_noisy_audio(yamnet_model):
    """Test with noisy audio."""
    from smlx.models.YAMNet import classify

    model = yamnet_model

    # Create noisy audio
    noisy_audio = np.random.randn(16000).astype(np.float32) * 0.1

    # Should handle noisy audio
    predictions = classify(model, noisy_audio, sample_rate=16000, top_k=3)

    assert predictions is not None, "Should handle noisy audio"


def test_model_config(yamnet_model):
    """Test model configuration."""
    from smlx.models.YAMNet import YAMNetConfig

    model = yamnet_model

    # Model should have config
    assert hasattr(model, "config"), "Model should have config"

    config = model.config
    assert config is not None, "Config should not be None"


def test_prediction_format(yamnet_model):
    """Test prediction output format."""
    from smlx.models.YAMNet import classify, Prediction

    model = yamnet_model

    # Create test audio
    test_audio = create_test_audio()

    # Classify
    predictions = classify(model, test_audio, sample_rate=16000, top_k=3)

    # Check prediction structure
    for pred in predictions:
        assert isinstance(pred, Prediction), "Should be Prediction object"
        assert isinstance(pred.label, str), "Label should be string"
        assert isinstance(pred.score, (float, np.floating)), "Score should be float"


def test_embedding_dimensions(yamnet_model):
    """Test embedding dimensions."""
    from smlx.models.YAMNet import extract_embeddings

    model = yamnet_model

    # Create test audio
    test_audio = create_test_audio(duration=2.0)

    # Extract embeddings
    embeddings = extract_embeddings(model, test_audio, sample_rate=16000)

    # Check dimensions
    assert embeddings.shape[1] == 1024, "Should have 1024-dim embeddings"
    assert embeddings.shape[0] > 0, "Should have at least one frame"


def test_multiple_classifications(yamnet_model):
    """Test multiple sequential classifications."""
    from smlx.models.YAMNet import classify

    model = yamnet_model

    # Perform multiple classifications
    for i in range(3):
        test_audio = create_test_audio(frequency=440 * (i + 1))
        predictions = classify(model, test_audio, sample_rate=16000, top_k=3)

        assert predictions is not None, f"Classification {i} should work"


def test_threshold_parameter(yamnet_model):
    """Test threshold parameter in event detection."""
    from smlx.models.YAMNet import detect_events

    model = yamnet_model

    # Create test audio
    test_audio = create_test_audio(duration=2.0)

    # Test different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7]

    for threshold in thresholds:
        events = detect_events(
            model, test_audio, sample_rate=16000, threshold=threshold
        )

        assert events is not None, f"Should work with threshold {threshold}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
