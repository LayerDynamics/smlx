#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
HiFi-GAN Vocoder for mel-spectrogram to waveform conversion.

Based on: "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"
Paper: https://arxiv.org/abs/2010.05646

This implementation provides the generator network for converting mel-spectrograms
to audio waveforms with high quality and fast inference.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional


class ResBlock(nn.Module):
    """
    Residual block with dilated convolutions.

    Uses multiple dilated convolutions with different dilation rates
    to capture different temporal patterns.

    Args:
        channels: Number of channels
        kernel_size: Convolution kernel size
        dilation_rates: List of dilation rates for each layer
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation_rates: List[int] = [1, 3, 5],
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # Create conv layers for each dilation rate
        self.convs1 = []
        self.convs2 = []

        for dilation in dilation_rates:
            # Calculate padding to maintain sequence length
            # For same-padding: padding = dilation * (kernel_size - 1) / 2
            # MLX Conv1d needs integer padding
            padding = (kernel_size - 1) // 2

            # First conv in pair
            self.convs1.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )

            # Second conv in pair
            self.convs2.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through residual block.

        Args:
            x: Input tensor (batch, length, channels)

        Returns:
            Output tensor (batch, length, channels)
        """
        for conv1, conv2 in zip(self.convs1, self.convs2):
            # Store input for residual connection
            residual = x

            # First conv + activation
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = conv1(x)

            # Trim or pad to match residual shape
            if x.shape[1] > residual.shape[1]:
                # Trim excess
                x = x[:, : residual.shape[1], :]
            elif x.shape[1] < residual.shape[1]:
                # Pad to match
                pad_amount = residual.shape[1] - x.shape[1]
                x = mx.pad(x, [(0, 0), (0, pad_amount), (0, 0)])

            # Second conv + activation
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = conv2(x)

            # Trim or pad to match residual shape again
            if x.shape[1] > residual.shape[1]:
                x = x[:, : residual.shape[1], :]
            elif x.shape[1] < residual.shape[1]:
                pad_amount = residual.shape[1] - x.shape[1]
                x = mx.pad(x, [(0, 0), (0, pad_amount), (0, 0)])

            # Add residual
            x = x + residual

        return x


class MRFBlock(nn.Module):
    """
    Multi-Receptive Field (MRF) block.

    Applies multiple residual blocks with different kernel sizes in parallel
    and sums their outputs. This allows the model to capture patterns at
    multiple temporal scales.

    Args:
        channels: Number of channels
        kernel_sizes: List of kernel sizes for parallel ResBlocks
        dilation_rates: Dilation rates for each ResBlock
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int] = [3, 7, 11],
        dilation_rates: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()

        # Create ResBlock for each kernel size
        self.resblocks = []
        for kernel_size, dilations in zip(kernel_sizes, dilation_rates):
            self.resblocks.append(
                ResBlock(channels, kernel_size, dilations)
            )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through MRF block.

        Args:
            x: Input tensor (batch, length, channels)

        Returns:
            Output tensor (batch, length, channels)
        """
        # Apply all ResBlocks in parallel and sum
        output = None
        for resblock in self.resblocks:
            if output is None:
                output = resblock(x)
            else:
                output = output + resblock(x)

        # Average the outputs
        return output / len(self.resblocks)


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator network.

    Converts mel-spectrogram to waveform using transposed convolutions
    for upsampling and MRF blocks for high-quality synthesis.

    Architecture:
        Input: Mel-spectrogram (batch, time, n_mels)
        ↓
        Initial Conv1d
        ↓
        Upsample Block 1 (×8) + MRF
        ↓
        Upsample Block 2 (×8) + MRF
        ↓
        Upsample Block 3 (×2) + MRF
        ↓
        Upsample Block 4 (×2) + MRF
        ↓
        Final Conv1d → Tanh
        ↓
        Output: Waveform (batch, time×256, 1)

    Args:
        n_mels: Number of mel-frequency bins (default: 80)
        upsample_rates: Upsampling factors for each block (default: [8, 8, 2, 2])
        upsample_kernel_sizes: Kernel sizes for upsampling convs
        upsample_initial_channel: Initial channel count (default: 512)
        mrf_kernel_sizes: Kernel sizes for MRF blocks
        mrf_dilation_rates: Dilation rates for MRF blocks
    """

    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        mrf_kernel_sizes: List[int] = [3, 7, 11],
        mrf_dilation_rates: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()

        self.n_mels = n_mels
        self.num_upsamples = len(upsample_rates)

        # Initial conv to expand mel features
        self.conv_pre = nn.Conv1d(
            n_mels,
            upsample_initial_channel,
            kernel_size=7,
            padding=3,
        )

        # Upsampling blocks
        self.ups = []
        self.mrf_blocks = []

        for i, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Calculate channels for this layer (halve channels as we go deeper)
            channels = upsample_initial_channel // (2 ** i)

            # Transposed convolution for upsampling
            # Padding calculation for ConvTranspose1d
            padding = (kernel_size - rate) // 2

            self.ups.append(
                nn.ConvTranspose1d(
                    channels,
                    channels // 2,
                    kernel_size=kernel_size,
                    stride=rate,
                    padding=padding,
                )
            )

            # MRF block after upsampling
            self.mrf_blocks.append(
                MRFBlock(
                    channels // 2,
                    kernel_sizes=mrf_kernel_sizes,
                    dilation_rates=mrf_dilation_rates,
                )
            )

        # Final conv to produce waveform (single channel)
        final_channels = upsample_initial_channel // (2 ** self.num_upsamples)
        self.conv_post = nn.Conv1d(
            final_channels,
            1,
            kernel_size=7,
            padding=3,
        )

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Generate waveform from mel-spectrogram.

        Args:
            mel: Mel-spectrogram (batch, time, n_mels)
                 MLX Conv1d expects NLC format

        Returns:
            Waveform (batch, samples)
        """
        # Initial convolution
        x = self.conv_pre(mel)

        # Apply upsampling blocks and MRF
        for up, mrf in zip(self.ups, self.mrf_blocks):
            # LeakyReLU activation
            x = nn.leaky_relu(x, negative_slope=0.1)

            # Upsample
            x = up(x)

            # Apply MRF
            x = mrf(x)

        # Final activation and conv
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = self.conv_post(x)
        x = mx.tanh(x)

        # Remove channel dimension: (batch, samples, 1) → (batch, samples)
        x = mx.squeeze(x, axis=-1)

        return x

    def remove_weight_norm(self):
        """
        Remove weight normalization from all conv layers.

        Note: MLX doesn't have built-in weight normalization yet,
        so this is a placeholder for future compatibility.
        """
        pass


class HiFiGANConfig:
    """
    Configuration for HiFi-GAN vocoder.

    Default configuration is optimized for 24kHz audio with hop_length=256.
    This gives 256x upsampling: [8, 8, 2, 2] = 256

    Args:
        n_mels: Number of mel bins
        sample_rate: Target sample rate
        hop_length: STFT hop length (upsampling factor)
        upsample_rates: Upsampling factors (product should equal hop_length)
        upsample_kernel_sizes: Kernel sizes for upsampling
        upsample_initial_channel: Initial channel count
        mrf_kernel_sizes: MRF kernel sizes
        mrf_dilation_rates: MRF dilation rates
    """

    def __init__(
        self,
        n_mels: int = 80,
        sample_rate: int = 24000,
        hop_length: int = 256,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        mrf_kernel_sizes: List[int] = [3, 7, 11],
        mrf_dilation_rates: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_initial_channel = upsample_initial_channel
        self.mrf_kernel_sizes = mrf_kernel_sizes
        self.mrf_dilation_rates = mrf_dilation_rates

        # Verify upsampling rates match hop length
        upsample_product = 1
        for rate in upsample_rates:
            upsample_product *= rate

        if upsample_product != hop_length:
            raise ValueError(
                f"Product of upsample_rates ({upsample_product}) must equal "
                f"hop_length ({hop_length})"
            )


def create_vocoder(config: Optional[HiFiGANConfig] = None) -> HiFiGANGenerator:
    """
    Create HiFi-GAN vocoder.

    Args:
        config: Optional vocoder configuration

    Returns:
        HiFiGANGenerator instance

    Example:
        >>> vocoder = create_vocoder()
        >>> mel = mx.random.normal((1, 100, 80))  # (batch, time, n_mels)
        >>> waveform = vocoder(mel)
        >>> waveform.shape
        (1, 25600)  # 100 mel frames × 256 hop_length = 25,600 samples
    """
    if config is None:
        config = HiFiGANConfig()

    return HiFiGANGenerator(
        n_mels=config.n_mels,
        upsample_rates=config.upsample_rates,
        upsample_kernel_sizes=config.upsample_kernel_sizes,
        upsample_initial_channel=config.upsample_initial_channel,
        mrf_kernel_sizes=config.mrf_kernel_sizes,
        mrf_dilation_rates=config.mrf_dilation_rates,
    )
