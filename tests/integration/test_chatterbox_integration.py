#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for Chatterbox TTS model.

Tests the complete synthesis pipeline with the HiFi-GAN vocoder.
"""

import pytest
import mlx.core as mx
import numpy as np


@pytest.mark.integration
@pytest.mark.requires_model
def test_chatterbox_model_creation():
    """Test Chatterbox model can be created."""
    from smlx.models.Chatterbox.model import create_model

    model = create_model()
    assert model is not None
    assert hasattr(model, 'vocoder')
    assert hasattr(model, 'llama_backbone')
    assert hasattr(model, 'acoustic_head')


@pytest.mark.integration
@pytest.mark.requires_model
def test_chatterbox_forward_pass():
    """Test complete forward pass through model."""
    from smlx.models.Chatterbox.model import create_model
    from smlx.models.Chatterbox.config import DEFAULT_CONFIG

    model = create_model(DEFAULT_CONFIG)
    model.eval()

    # Create sample input
    batch_size = 2
    seq_len = 20
    input_ids = mx.random.randint(0, 1000, (batch_size, seq_len))

    # Forward pass
    mel, waveform = model(input_ids)

    # Check mel-spectrogram shape
    assert mel.ndim == 3
    assert mel.shape[0] == batch_size
    assert mel.shape[2] == 80  # n_mels

    # Check waveform shape
    assert waveform.ndim == 2
    assert waveform.shape[0] == batch_size

    # Waveform should not be all zeros (real vocoder!)
    assert not mx.all(waveform == 0)

    # Waveform should be in tanh range [-1, 1]
    assert float(waveform.min()) >= -1.0
    assert float(waveform.max()) <= 1.0


@pytest.mark.integration
def test_processor_audio_loading():
    """Test processor can handle audio processing."""
    from smlx.models.Chatterbox.processor import create_processor
    import numpy as np

    processor = create_processor(sample_rate=24000)

    # Create synthetic audio
    audio = np.random.randn(24000).astype(np.float32)  # 1 second

    # Process audio
    mel = processor.process_audio(audio, sr=24000)

    # Check mel-spectrogram
    assert mel.ndim == 2
    assert mel.shape[1] == 80  # n_mels
    assert mel.shape[0] > 0  # Has time frames


@pytest.mark.integration
def test_vocoder_mel_to_audio():
    """Test vocoder converts mel-spectrogram to audio."""
    from smlx.models.Chatterbox.vocoder import create_vocoder

    vocoder = create_vocoder()

    # Create random mel-spectrogram
    batch_size = 1
    time_frames = 100
    n_mels = 80
    mel = mx.random.normal((batch_size, time_frames, n_mels))

    # Generate waveform
    waveform = vocoder(mel)

    # Check shape
    assert waveform.ndim == 2
    assert waveform.shape[0] == batch_size

    # Check approximate length (100 frames × 256 hop = 25600 samples)
    expected_samples = time_frames * 256
    assert abs(waveform.shape[1] - expected_samples) < 1000

    # Check range
    assert float(waveform.min()) >= -1.0
    assert float(waveform.max()) <= 1.0


@pytest.mark.integration
def test_audio_utils_pipeline():
    """Test complete audio processing pipeline."""
    from smlx.models.Chatterbox import audio_utils
    import numpy as np

    # Create synthetic audio
    audio = np.random.randn(24000).astype(np.float32)
    audio_mlx = mx.array(audio)

    # Extract mel-spectrogram
    mel = audio_utils.log_mel_spectrogram(audio_mlx, n_mels=80, sample_rate=24000)

    # Check output
    assert mel.ndim == 2
    assert mel.shape[1] == 80  # n_mels
    assert mel.shape[0] > 0  # time frames


@pytest.mark.integration
@pytest.mark.requires_model
def test_end_to_end_synthesis():
    """Test end-to-end text-to-speech synthesis."""
    from smlx.models.Chatterbox.model import create_model
    from smlx.models.Chatterbox.processor import create_processor

    # Create model and processor
    model = create_model()
    processor = create_processor()
    model.eval()

    # Create dummy tokenizer behavior
    class DummyTokenizer:
        def encode(self, text):
            # Return dummy token IDs
            return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    processor.tokenizer = DummyTokenizer()

    # Process text
    text = "Hello world, this is a test."
    token_ids = processor(text)

    # Add batch dimension
    input_ids = mx.expand_dims(token_ids, axis=0)

    # Synthesize
    mel, waveform = model(input_ids)

    # Verify output
    assert mel.shape[0] == 1  # batch size
    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0  # has samples

    # Waveform should not be zeros
    assert not mx.all(waveform == 0)

    print(f"✓ End-to-end synthesis successful!")
    print(f"  Mel shape: {mel.shape}")
    print(f"  Waveform shape: {waveform.shape}")
    print(f"  Audio duration: {waveform.shape[1]/24000:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
