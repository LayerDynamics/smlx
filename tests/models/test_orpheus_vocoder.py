#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Tests for Orpheus-150M HiFi-GAN vocoder.
"""

import pytest
import mlx.core as mx

from smlx.models.Orpheus_150M.vocoder import (
    HiFiGANConfig,
    HiFiGANVocoder,
    ResBlock,
    UpsampleBlock,
    Generator,
    create_hifigan_v3,
    create_hifigan_v1,
)


class TestHiFiGANConfig:
    """Test HiFi-GAN configuration."""

    def test_default_config(self):
        """Test default V3 configuration."""
        config = HiFiGANConfig()

        assert config.mel_channels == 80
        assert config.upsample_rates == [8, 8, 4]
        assert config.upsample_kernel_sizes == [16, 16, 8]
        assert config.upsample_initial_channel == 256
        assert config.resblock_kernel_sizes == [3, 5, 7]
        assert config.resblock_dilation_sizes == [[1, 2], [2, 6], [3, 12]]

    def test_total_upsample_factor(self):
        """Test total upsampling factor calculation."""
        config = HiFiGANConfig()
        # 8 * 8 * 4 = 256
        assert config.total_upsample_factor == 256

    def test_custom_config(self):
        """Test custom configuration."""
        config = HiFiGANConfig(
            mel_channels=100,
            upsample_rates=[4, 4, 4],
            upsample_initial_channel=512,
        )

        assert config.mel_channels == 100
        assert config.upsample_rates == [4, 4, 4]
        assert config.total_upsample_factor == 64


class TestResBlock:
    """Test residual block."""

    def test_resblock_forward(self):
        """Test ResBlock forward pass."""
        resblock = ResBlock(
            channels=256, kernel_size=3, dilations=[1, 2], leaky_relu_slope=0.1
        )

        # Input: (batch, length, channels)
        x = mx.random.normal((2, 100, 256))
        output = resblock(x)

        # Output should have same shape
        assert output.shape == x.shape

    def test_resblock_shapes(self):
        """Test ResBlock with different shapes."""
        resblock = ResBlock(channels=128, kernel_size=5, dilations=[1, 3, 5])

        x = mx.random.normal((1, 50, 128))
        output = resblock(x)

        assert output.shape == (1, 50, 128)


class TestUpsampleBlock:
    """Test upsampling block."""

    def test_upsample_forward(self):
        """Test UpsampleBlock forward pass."""
        upsample = UpsampleBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=16,
            stride=8,
        )

        # Input: (batch, length, channels)
        x = mx.random.normal((2, 50, 256))
        output = upsample(x)

        # Length should be upsampled by stride (8)
        assert output.shape == (2, 50 * 8, 128)

    def test_upsample_different_strides(self):
        """Test different upsampling strides."""
        for stride in [2, 4, 8]:
            upsample = UpsampleBlock(
                in_channels=128, out_channels=64, kernel_size=stride * 2, stride=stride
            )

            x = mx.random.normal((1, 32, 128))
            output = upsample(x)

            assert output.shape == (1, 32 * stride, 64)


class TestGenerator:
    """Test HiFi-GAN generator."""

    def test_generator_v3_forward(self):
        """Test V3 generator forward pass."""
        config = HiFiGANConfig()  # Default V3 config
        generator = Generator(config)

        # Input: mel-spectrogram (batch, time, mel_channels)
        mel = mx.random.normal((2, 100, 80))
        waveform = generator(mel)

        # Output: waveform (batch, time * upsample_factor)
        # 100 * 256 = 25600
        assert waveform.shape == (2, 25600)

        # Output should be in [-1, 1] due to tanh
        assert waveform.min() >= -1.0
        assert waveform.max() <= 1.0

    def test_generator_v1_forward(self):
        """Test V1 generator forward pass."""
        config = HiFiGANConfig(
            upsample_rates=[8, 8, 2, 2],
            upsample_kernel_sizes=[16, 16, 4, 4],
            upsample_initial_channel=512,
        )
        generator = Generator(config)

        mel = mx.random.normal((1, 50, 80))
        waveform = generator(mel)

        # 50 * 256 = 12800
        assert waveform.shape == (1, 12800)


class TestHiFiGANVocoder:
    """Test complete HiFi-GAN vocoder."""

    def test_vocoder_forward(self):
        """Test vocoder forward pass."""
        vocoder = HiFiGANVocoder()

        mel = mx.random.normal((2, 100, 80))
        waveform = vocoder(mel)

        assert waveform.shape == (2, 25600)

    def test_vocoder_inference(self):
        """Test vocoder inference method."""
        vocoder = HiFiGANVocoder()

        mel = mx.random.normal((1, 50, 80))
        waveform = vocoder.inference(mel)

        assert waveform.shape == (1, 12800)

    def test_vocoder_single_sample(self):
        """Test vocoder with single sample."""
        vocoder = HiFiGANVocoder()

        # Single sample (batch=1)
        mel = mx.random.normal((1, 10, 80))
        waveform = vocoder(mel)

        # 10 * 256 = 2560
        assert waveform.shape == (1, 2560)

    def test_vocoder_normalization(self):
        """Test vocoder with normalization options."""
        vocoder = HiFiGANVocoder()
        mel = mx.random.uniform(0, 1, (1, 20, 80))  # Pre-normalized

        # Without normalization
        waveform1 = vocoder(mel, normalize=False)

        # With normalization
        waveform2 = vocoder(mel, normalize=True)

        # Both should produce valid waveforms
        assert waveform1.shape == (1, 5120)
        assert waveform2.shape == (1, 5120)


class TestVocoderFactories:
    """Test vocoder factory functions."""

    def test_create_hifigan_v3(self):
        """Test V3 factory function."""
        vocoder = create_hifigan_v3(mel_channels=80)

        assert isinstance(vocoder, HiFiGANVocoder)
        assert vocoder.config.mel_channels == 80
        assert vocoder.config.upsample_rates == [8, 8, 4]

    def test_create_hifigan_v1(self):
        """Test V1 factory function."""
        vocoder = create_hifigan_v1(mel_channels=80)

        assert isinstance(vocoder, HiFiGANVocoder)
        assert vocoder.config.mel_channels == 80
        assert vocoder.config.upsample_rates == [8, 8, 2, 2]
        assert vocoder.config.upsample_initial_channel == 512


@pytest.mark.benchmark
class TestVocoderPerformance:
    """Performance benchmarks for vocoder."""

    def test_vocoder_speed(self):
        """Test vocoder inference speed."""
        import time

        vocoder = create_hifigan_v3()

        # Warm up
        mel = mx.random.normal((1, 100, 80))
        _ = vocoder(mel)
        mx.eval(_)

        # Benchmark
        iterations = 10
        start_time = time.time()

        for _ in range(iterations):
            mel = mx.random.normal((1, 100, 80))
            waveform = vocoder(mel)
            mx.eval(waveform)

        elapsed = time.time() - start_time
        avg_time = elapsed / iterations

        print(f"\nAverage vocoder inference time: {avg_time * 1000:.2f} ms")
        print(f"Throughput: {100 * 256 / 24000 / avg_time:.1f}x real-time")

        # Should be reasonably fast (adjust threshold based on M4 performance)
        assert avg_time < 1.0  # Less than 1 second for 100 frames

    def test_vocoder_memory(self):
        """Test vocoder memory usage."""
        vocoder = create_hifigan_v3()

        # Count parameters
        total_params = 0
        for param in vocoder.parameters().values():
            if hasattr(param, "size"):
                total_params += param.size

        print(f"\nVocoder parameters: {total_params:,}")

        # V3 should be under 1M parameters
        assert total_params < 1_500_000  # 1.5M params max for V3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
