#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Unit tests for Chatterbox HiFi-GAN vocoder.

Tests the vocoder architecture, forward pass, and weight loading.
"""

import pytest
import mlx.core as mx
import numpy as np

from smlx.models.Chatterbox.vocoder import (
    ResBlock,
    MRFBlock,
    HiFiGANGenerator,
    HiFiGANConfig,
    create_vocoder,
)


@pytest.mark.unit
class TestResBlock:
    """Test ResBlock module."""

    def test_resblock_creation(self):
        """Test ResBlock can be created."""
        block = ResBlock(channels=512, kernel_size=3, dilation_rates=[1, 3, 5])
        assert block is not None
        assert block.channels == 512
        assert block.kernel_size == 3

    def test_resblock_forward(self):
        """Test ResBlock forward pass."""
        block = ResBlock(channels=256, kernel_size=3, dilation_rates=[1, 3, 5])

        # Input: (batch, length, channels)
        x = mx.random.normal((2, 100, 256))
        output = block(x)

        # Output should have same shape as input
        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_resblock_different_dilations(self):
        """Test ResBlock with different dilation rates."""
        block = ResBlock(channels=128, kernel_size=7, dilation_rates=[1, 3, 5, 7])

        x = mx.random.normal((1, 50, 128))
        output = block(x)

        assert output.shape == (1, 50, 128)


@pytest.mark.unit
class TestMRFBlock:
    """Test Multi-Receptive Field block."""

    def test_mrf_creation(self):
        """Test MRFBlock can be created."""
        mrf = MRFBlock(
            channels=512,
            kernel_sizes=[3, 7, 11],
            dilation_rates=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        )
        assert mrf is not None
        assert len(mrf.resblocks) == 3

    def test_mrf_forward(self):
        """Test MRFBlock forward pass."""
        mrf = MRFBlock(channels=256, kernel_sizes=[3, 7, 11])

        x = mx.random.normal((2, 100, 256))
        output = mrf(x)

        # Output should have same shape as input
        assert output.shape == x.shape

    def test_mrf_single_kernel(self):
        """Test MRFBlock with single kernel size."""
        mrf = MRFBlock(channels=128, kernel_sizes=[3], dilation_rates=[[1, 3, 5]])

        x = mx.random.normal((1, 50, 128))
        output = mrf(x)

        assert output.shape == (1, 50, 128)


@pytest.mark.unit
class TestHiFiGANGenerator:
    """Test HiFi-GAN Generator."""

    def test_generator_creation(self):
        """Test HiFiGANGenerator can be created."""
        generator = HiFiGANGenerator(
            n_mels=80,
            upsample_rates=[8, 8, 2, 2],
            upsample_kernel_sizes=[16, 16, 4, 4],
        )
        assert generator is not None
        assert generator.n_mels == 80
        assert generator.num_upsamples == 4

    def test_generator_forward_shape(self):
        """Test generator produces correct output shape."""
        generator = HiFiGANGenerator(
            n_mels=80,
            upsample_rates=[8, 8, 2, 2],
        )

        # Input mel: (batch, time, n_mels)
        mel = mx.random.normal((1, 100, 80))
        waveform = generator(mel)

        # Output waveform: (batch, samples)
        # 100 mel frames × 256 hop_length = 25,600 samples (approximately)
        assert waveform.ndim == 2
        assert waveform.shape[0] == 1
        # Check output is roughly correct length (allow some variation due to padding)
        expected_length = 100 * 256
        assert abs(waveform.shape[1] - expected_length) < 1000

    def test_generator_batch_processing(self):
        """Test generator handles batch processing."""
        generator = HiFiGANGenerator(n_mels=80)

        # Batch of 4 mel-spectrograms
        mel = mx.random.normal((4, 50, 80))
        waveform = generator(mel)

        assert waveform.shape[0] == 4
        assert waveform.ndim == 2

    def test_generator_different_lengths(self):
        """Test generator handles variable-length inputs."""
        generator = HiFiGANGenerator(n_mels=80)

        for time_frames in [50, 100, 200]:
            mel = mx.random.normal((1, time_frames, 80))
            waveform = generator(mel)

            assert waveform.ndim == 2
            # Check approximate output length
            expected = time_frames * 256
            assert abs(waveform.shape[1] - expected) < 1000

    def test_generator_output_range(self):
        """Test generator output is in tanh range [-1, 1]."""
        generator = HiFiGANGenerator(n_mels=80)

        mel = mx.random.normal((1, 100, 80))
        waveform = generator(mel)

        # Tanh activation ensures output is in [-1, 1]
        assert float(waveform.min()) >= -1.0
        assert float(waveform.max()) <= 1.0

    def test_generator_custom_config(self):
        """Test generator with custom configuration."""
        generator = HiFiGANGenerator(
            n_mels=128,
            upsample_rates=[4, 4, 4, 4],  # 256x upsampling
            upsample_kernel_sizes=[8, 8, 8, 8],
            upsample_initial_channel=256,
        )

        mel = mx.random.normal((1, 100, 128))
        waveform = generator(mel)

        assert waveform.ndim == 2
        assert waveform.shape[0] == 1


@pytest.mark.unit
class TestHiFiGANConfig:
    """Test HiFi-GAN configuration."""

    def test_config_creation(self):
        """Test config can be created with defaults."""
        config = HiFiGANConfig()
        assert config.n_mels == 80
        assert config.sample_rate == 24000
        assert config.hop_length == 256
        assert config.upsample_rates == [8, 8, 2, 2]

    def test_config_upsampling_validation(self):
        """Test config validates upsampling product."""
        # Valid config (8*8*2*2 = 256)
        config = HiFiGANConfig(
            hop_length=256,
            upsample_rates=[8, 8, 2, 2],
        )
        assert config.hop_length == 256

        # Invalid config (8*8 = 64, not 256)
        with pytest.raises(ValueError, match="must equal hop_length"):
            HiFiGANConfig(
                hop_length=256,
                upsample_rates=[8, 8],  # Product = 64, not 256
            )

    def test_config_custom_sample_rate(self):
        """Test config with custom sample rate."""
        config = HiFiGANConfig(
            sample_rate=16000,
            hop_length=200,
            upsample_rates=[8, 5, 5],  # 8*5*5 = 200
        )
        assert config.sample_rate == 16000
        assert config.hop_length == 200


@pytest.mark.unit
class TestCreateVocoder:
    """Test vocoder creation helper."""

    def test_create_vocoder_default(self):
        """Test creating vocoder with default config."""
        vocoder = create_vocoder()
        assert isinstance(vocoder, HiFiGANGenerator)
        assert vocoder.n_mels == 80

    def test_create_vocoder_custom_config(self):
        """Test creating vocoder with custom config."""
        config = HiFiGANConfig(
            n_mels=128,
            sample_rate=16000,
            hop_length=200,
            upsample_rates=[8, 5, 5],
        )
        vocoder = create_vocoder(config)
        assert vocoder.n_mels == 128

    def test_created_vocoder_works(self):
        """Test that created vocoder can process mel-spectrograms."""
        vocoder = create_vocoder()

        mel = mx.random.normal((1, 100, 80))
        waveform = vocoder(mel)

        assert waveform.shape[0] == 1
        assert waveform.ndim == 2


@pytest.mark.unit
class TestVocoderNumericalStability:
    """Test vocoder numerical stability."""

    def test_zero_input(self):
        """Test vocoder handles zero input."""
        vocoder = HiFiGANGenerator(n_mels=80)

        mel = mx.zeros((1, 100, 80))
        waveform = vocoder(mel)

        assert not mx.any(mx.isnan(waveform))
        assert not mx.any(mx.isinf(waveform))

    def test_large_input(self):
        """Test vocoder handles large input values."""
        vocoder = HiFiGANGenerator(n_mels=80)

        mel = mx.ones((1, 100, 80)) * 10.0
        waveform = vocoder(mel)

        # Should still produce valid output due to tanh
        assert not mx.any(mx.isnan(waveform))
        assert not mx.any(mx.isinf(waveform))
        assert float(waveform.min()) >= -1.0
        assert float(waveform.max()) <= 1.0

    def test_negative_input(self):
        """Test vocoder handles negative input values."""
        vocoder = HiFiGANGenerator(n_mels=80)

        mel = mx.ones((1, 100, 80)) * -5.0
        waveform = vocoder(mel)

        assert not mx.any(mx.isnan(waveform))
        assert not mx.any(mx.isinf(waveform))


@pytest.mark.unit
class TestVocoderConsistency:
    """Test vocoder consistency and determinism."""

    def test_same_input_same_output(self):
        """Test same input produces same output."""
        vocoder = HiFiGANGenerator(n_mels=80)

        # Create deterministic input using key
        key = mx.random.key(42)
        mel = mx.random.normal((1, 50, 80), key=key)

        # Run twice with same input
        out1 = vocoder(mel)
        out2 = vocoder(mel)

        # Should produce identical output
        assert mx.allclose(out1, out2)

    def test_different_instances_same_output(self):
        """Test different vocoder instances produce same output (with same weights)."""
        # Create deterministic input using key
        key = mx.random.key(42)
        mel = mx.random.normal((1, 50, 80), key=key)

        # Create two vocoders with same config
        vocoder1 = HiFiGANGenerator(n_mels=80, upsample_rates=[8, 8, 2, 2])
        vocoder2 = HiFiGANGenerator(n_mels=80, upsample_rates=[8, 8, 2, 2])

        # Copy weights from vocoder1 to vocoder2
        vocoder2.update(dict(vocoder1.parameters()))

        out1 = vocoder1(mel)
        out2 = vocoder2(mel)

        # Should produce identical output with same weights
        assert mx.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
