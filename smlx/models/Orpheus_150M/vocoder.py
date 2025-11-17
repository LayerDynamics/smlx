"""
HiFi-GAN V3 Vocoder for Orpheus TTS.

Implements lightweight neural vocoder for converting mel-spectrograms to audio waveforms.
Based on HiFi-GAN V3 architecture (0.92M parameters, CPU-optimized).

Reference:
    Kong et al., "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"
    https://arxiv.org/abs/2010.05646
"""

from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class HiFiGANConfig:
    """Configuration for HiFi-GAN V3 vocoder."""

    # Input/output
    mel_channels: int = 80
    """Number of mel-spectrogram channels"""

    # Upsampling configuration
    upsample_rates: List[int] = None
    """Upsampling rates for each layer. Default: [8, 8, 4] (total 256x)"""

    upsample_kernel_sizes: List[int] = None
    """Kernel sizes for upsampling layers. Default: [16, 16, 8]"""

    upsample_initial_channel: int = 256
    """Number of channels after initial convolution"""

    # Residual block configuration
    resblock_kernel_sizes: List[int] = None
    """Kernel sizes for residual blocks. Default: [3, 5, 7]"""

    resblock_dilation_sizes: List[List[int]] = None
    """Dilation sizes for each residual block. Default: [[1,2], [2,6], [3,12]]"""

    # Activation
    leaky_relu_slope: float = 0.1
    """Slope for LeakyReLU activation"""

    def __post_init__(self):
        """Set default values for list fields."""
        if self.upsample_rates is None:
            self.upsample_rates = [8, 8, 4]

        if self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [16, 16, 8]

        if self.resblock_kernel_sizes is None:
            self.resblock_kernel_sizes = [3, 5, 7]

        if self.resblock_dilation_sizes is None:
            self.resblock_dilation_sizes = [[1, 2], [2, 6], [3, 12]]

    @property
    def total_upsample_factor(self) -> int:
        """Total upsampling factor (product of all rates)."""
        factor = 1
        for rate in self.upsample_rates:
            factor *= rate
        return factor


class ResBlock(nn.Module):
    """Residual block with dilated convolutions (Type 2 for V3).

    Uses multi-receptive field fusion with different kernel sizes and dilations.
    """

    def __init__(
        self, channels: int, kernel_size: int, dilations: List[int], leaky_relu_slope: float = 0.1
    ):
        """Initialize residual block.

        Args:
            channels: Number of channels
            kernel_size: Kernel size for convolutions
            dilations: List of dilation values
            leaky_relu_slope: Slope for LeakyReLU
        """
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        # Type 2: 1 conv layer per dilation, 2 stacks
        self.convs = []
        for dilation in dilations:
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size * dilation - dilation) // 2,
                dilation=dilation,
            )
            self.convs.append(conv)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, length, channels)

        Returns:
            Output tensor (batch, length, channels)
        """
        # Store input for residual connection
        residual = x

        # Apply convolutions with LeakyReLU
        for conv in self.convs:
            # LeakyReLU
            x = mx.maximum(self.leaky_relu_slope * x, x)
            # Convolution
            x = conv(x)

        # Residual connection
        return x + residual


class MultiReceptiveFieldFusion(nn.Module):
    """Multi-receptive field fusion module.

    Combines outputs from residual blocks with different kernel sizes.
    """

    def __init__(
        self,
        channels: int,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        leaky_relu_slope: float = 0.1,
    ):
        """Initialize MRF fusion module.

        Args:
            channels: Number of channels
            resblock_kernel_sizes: List of kernel sizes for residual blocks
            resblock_dilation_sizes: List of dilation lists for each kernel size
            leaky_relu_slope: Slope for LeakyReLU
        """
        super().__init__()

        self.resblocks = []
        for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
            block = ResBlock(channels, kernel_size, dilations, leaky_relu_slope)
            self.resblocks.append(block)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, length, channels)

        Returns:
            Fused output tensor (batch, length, channels)
        """
        # Sum outputs from all residual blocks
        output = None
        for resblock in self.resblocks:
            block_out = resblock(x)
            if output is None:
                output = block_out
            else:
                output = output + block_out

        # Average the outputs
        return output / len(self.resblocks)


class UpsampleBlock(nn.Module):
    """Upsampling block with transposed convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        leaky_relu_slope: float = 0.1,
    ):
        """Initialize upsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for transposed convolution
            stride: Stride for upsampling
            leaky_relu_slope: Slope for LeakyReLU
        """
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        # Note: MLX doesn't have ConvTranspose1d, so we'll use manual upsampling
        # Upsample then convolve (equivalent to transposed convolution)
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - stride) // 2,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, length, channels)

        Returns:
            Upsampled output tensor (batch, length * stride, channels)
        """
        # LeakyReLU activation
        x = mx.maximum(self.leaky_relu_slope * x, x)

        # Manual upsampling (nearest neighbor interpolation)
        # Repeat each frame 'stride' times
        batch, length, channels = x.shape
        x_expanded = mx.repeat(x, self.stride, axis=1)  # (batch, length * stride, channels)

        # Apply convolution for smoothing
        x = self.conv(x_expanded)

        return x


class Generator(nn.Module):
    """HiFi-GAN V3 Generator.

    Converts mel-spectrogram to audio waveform using:
    1. Initial convolution (mel → hidden dim)
    2. Series of upsample + MRF fusion blocks
    3. Final convolution (hidden → waveform)
    """

    def __init__(self, config: HiFiGANConfig):
        """Initialize generator.

        Args:
            config: HiFi-GAN configuration
        """
        super().__init__()
        self.config = config

        # Initial convolution
        self.conv_pre = nn.Conv1d(
            in_channels=config.mel_channels,
            out_channels=config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # Upsampling blocks
        self.upsample_blocks = []
        self.mrf_blocks = []

        channels = config.upsample_initial_channel

        for i, (rate, kernel) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            # Channel reduction after each upsample
            out_channels = channels // 2

            # Upsampling block
            upsample = UpsampleBlock(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=rate,
                leaky_relu_slope=config.leaky_relu_slope,
            )
            self.upsample_blocks.append(upsample)

            # MRF fusion block
            mrf = MultiReceptiveFieldFusion(
                channels=out_channels,
                resblock_kernel_sizes=config.resblock_kernel_sizes,
                resblock_dilation_sizes=config.resblock_dilation_sizes,
                leaky_relu_slope=config.leaky_relu_slope,
            )
            self.mrf_blocks.append(mrf)

            channels = out_channels

        # Final convolution to waveform
        self.conv_post = nn.Conv1d(
            in_channels=channels, out_channels=1, kernel_size=7, stride=1, padding=3
        )

    def __call__(self, mel: mx.array) -> mx.array:
        """Generate waveform from mel-spectrogram.

        Args:
            mel: Mel-spectrogram (batch, time, mel_channels)

        Returns:
            Waveform (batch, time * upsample_factor)
        """
        # Initial convolution
        x = self.conv_pre(mel)

        # Upsampling + MRF fusion
        for upsample, mrf in zip(self.upsample_blocks, self.mrf_blocks):
            x = upsample(x)
            x = mrf(x)

        # LeakyReLU before final conv
        x = mx.maximum(self.config.leaky_relu_slope * x, x)

        # Final convolution to waveform
        x = self.conv_post(x)

        # Remove channel dimension and apply tanh to bound output
        x = x.squeeze(-1)  # (batch, time, 1) → (batch, time)
        x = mx.tanh(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization (placeholder for compatibility)."""
        # MLX doesn't use weight norm like PyTorch
        # This is here for API compatibility
        pass


class HiFiGANVocoder(nn.Module):
    """Complete HiFi-GAN vocoder with pre/post processing."""

    def __init__(self, config: Optional[HiFiGANConfig] = None):
        """Initialize vocoder.

        Args:
            config: HiFi-GAN configuration (default: V3 config)
        """
        super().__init__()
        self.config = config if config is not None else HiFiGANConfig()
        self.generator = Generator(self.config)

    def __call__(
        self, mel: mx.array, normalize: bool = True, denormalize: bool = True
    ) -> mx.array:
        """Convert mel-spectrogram to waveform.

        Args:
            mel: Mel-spectrogram (batch, time, mel_channels)
                 Can be normalized [0, 1] or unnormalized
            normalize: Whether to normalize input mel (if not already normalized)
            denormalize: Whether to denormalize output waveform

        Returns:
            Audio waveform (batch, time * upsample_factor)
            Normalized to [-1, 1] if denormalize=True
        """
        # Input normalization (if needed)
        if normalize:
            # Assume input is in linear scale, convert to normalized
            # This is a simple normalization; adjust if needed
            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

        # Generate waveform
        waveform = self.generator(mel)

        # Output is already in [-1, 1] due to tanh in generator
        return waveform

    def inference(self, mel: mx.array) -> mx.array:
        """Inference mode (alias for __call__).

        Args:
            mel: Mel-spectrogram (batch, time, mel_channels)

        Returns:
            Audio waveform (batch, time * upsample_factor)
        """
        return self(mel, normalize=False, denormalize=True)

    def remove_weight_norm(self):
        """Remove weight normalization from generator."""
        self.generator.remove_weight_norm()


def create_hifigan_v3(mel_channels: int = 80) -> HiFiGANVocoder:
    """Create HiFi-GAN V3 vocoder with default configuration.

    Args:
        mel_channels: Number of mel-spectrogram channels

    Returns:
        HiFi-GAN V3 vocoder instance

    Example:
        >>> vocoder = create_hifigan_v3(mel_channels=80)
        >>> mel = mx.random.normal((1, 100, 80))  # 1 batch, 100 frames, 80 mels
        >>> waveform = vocoder(mel)
        >>> waveform.shape
        (1, 25600)  # 100 * 256 = 25600 samples
    """
    config = HiFiGANConfig(mel_channels=mel_channels)
    return HiFiGANVocoder(config)


def create_hifigan_v1(mel_channels: int = 80) -> HiFiGANVocoder:
    """Create HiFi-GAN V1 vocoder (higher quality, more parameters).

    Args:
        mel_channels: Number of mel-spectrogram channels

    Returns:
        HiFi-GAN V1 vocoder instance
    """
    config = HiFiGANConfig(
        mel_channels=mel_channels,
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    )
    return HiFiGANVocoder(config)
