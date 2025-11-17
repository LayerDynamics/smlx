#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Configuration for Orpheus-150M TTS model.

Defines the text encoder, duration predictor, decoder, and vocoder configurations.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TextEncoderConfig:
    """
    Text encoder configuration.

    Args:
        vocab_size: Size of vocabulary (phonemes + special tokens)
        embedding_dim: Dimension of text embeddings
        num_layers: Number of encoder layers
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension in feedforward
        dropout: Dropout probability
    """

    vocab_size: int = 1024  # Phoneme vocabulary
    embedding_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    hidden_dim: int = 1536
    dropout: float = 0.1
    max_seq_len: int = 512


@dataclass
class DurationPredictorConfig:
    """
    Duration predictor configuration.

    Predicts how long each phoneme should be pronounced.

    Args:
        input_dim: Input dimension (from text encoder)
        hidden_dim: Hidden dimension
        num_layers: Number of convolutional layers
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    input_dim: int = 384
    hidden_dim: int = 256
    num_layers: int = 4
    kernel_size: int = 3
    dropout: float = 0.1


@dataclass
class DecoderConfig:
    """
    Acoustic decoder configuration.

    Generates mel-spectrogram or acoustic features.

    Args:
        input_dim: Input dimension (from text encoder)
        num_mels: Number of mel-frequency bins
        num_layers: Number of decoder layers
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension in feedforward
        dropout: Dropout probability
    """

    input_dim: int = 384
    num_mels: int = 80  # Standard mel-spectrogram
    num_layers: int = 6
    num_heads: int = 6
    hidden_dim: int = 1536
    dropout: float = 0.1


@dataclass
class VocoderConfig:
    """
    Vocoder configuration.

    Converts mel-spectrogram to waveform.

    Args:
        num_mels: Number of mel-frequency bins (input)
        upsample_rates: Upsampling factors for each layer
        hidden_dim: Hidden dimension
        num_layers: Number of vocoder layers
    """

    num_mels: int = 80
    upsample_rates: tuple = (8, 8, 2, 2)  # Total upsample: 256 (hop length)
    hidden_dim: int = 512
    num_layers: int = 4
    kernel_size: int = 7


@dataclass
class Orpheus150MConfig:
    """
    Orpheus-150M TTS model configuration.

    Total parameters: ~150M
    - Text encoder: ~40M
    - Duration predictor: ~10M
    - Decoder: ~50M
    - Vocoder: ~50M

    Args:
        text_encoder_config: Text encoder configuration
        duration_config: Duration predictor configuration
        decoder_config: Acoustic decoder configuration
        vocoder_config: Vocoder configuration
        sample_rate: Audio sample rate (16000 or 24000)
        hop_length: STFT hop length (samples between frames)
        win_length: STFT window length
        n_fft: FFT size
    """

    text_encoder_config: TextEncoderConfig = None
    duration_config: DurationPredictorConfig = None
    decoder_config: DecoderConfig = None
    vocoder_config: VocoderConfig = None

    # Audio parameters
    sample_rate: int = 24000  # 24kHz for better quality
    hop_length: int = 256  # ~10.7ms at 24kHz
    win_length: int = 1024
    n_fft: int = 1024
    num_mels: int = 80

    # Generation parameters
    max_duration: int = 100  # Max duration per phoneme (frames)
    min_duration: int = 1

    def __post_init__(self):
        if self.text_encoder_config is None:
            self.text_encoder_config = TextEncoderConfig()
        if self.duration_config is None:
            self.duration_config = DurationPredictorConfig()
        if self.decoder_config is None:
            self.decoder_config = DecoderConfig()
        if self.vocoder_config is None:
            self.vocoder_config = VocoderConfig()

    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)


# Default configuration
DEFAULT_CONFIG = Orpheus150MConfig()


def load_config(model_path: str) -> Orpheus150MConfig:
    """
    Load configuration from model directory.

    Args:
        model_path: Path to model directory

    Returns:
        Orpheus150MConfig instance
    """
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        print(f"Config not found at {config_path}, using default config")
        return DEFAULT_CONFIG

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Parse nested configs
    text_encoder_config = TextEncoderConfig(**config_dict.get("text_encoder_config", {}))
    duration_config = DurationPredictorConfig(**config_dict.get("duration_config", {}))
    decoder_config = DecoderConfig(**config_dict.get("decoder_config", {}))
    vocoder_config = VocoderConfig(**config_dict.get("vocoder_config", {}))

    # Create main config
    config = Orpheus150MConfig(
        text_encoder_config=text_encoder_config,
        duration_config=duration_config,
        decoder_config=decoder_config,
        vocoder_config=vocoder_config,
        sample_rate=config_dict.get("sample_rate", 24000),
        hop_length=config_dict.get("hop_length", 256),
        win_length=config_dict.get("win_length", 1024),
        n_fft=config_dict.get("n_fft", 1024),
        num_mels=config_dict.get("num_mels", 80),
    )

    return config


def save_config(config: Orpheus150MConfig, output_path: str):
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        output_path: Output directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print(f"✓ Config saved to {config_path}")
