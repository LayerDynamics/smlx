#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Configuration for Silero VAD model.

Silero VAD is a lightweight voice activity detection model
based on LSTM architecture, designed for real-time speech detection.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VADConfig:
    """Configuration for Silero VAD model.

    The Silero VAD model is a compact LSTM-based model for
    voice activity detection. It's designed to be fast and
    accurate for real-time applications.

    Model variants:
        - v3.1: Latest version, best accuracy (~1MB)
        - v4.0: Improved false positive rate
    """

    # Model architecture
    model_type: str = "silero_vad"
    sample_rate: int = 16000  # Supported: 8000, 16000
    num_channels: int = 1  # Mono audio only

    # LSTM architecture
    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = False

    # Processing
    context_size: int = 64  # Context window for LSTM
    min_speech_duration_ms: int = 250  # Minimum speech segment duration
    min_silence_duration_ms: int = 100  # Minimum silence to split segments
    speech_pad_ms: int = 30  # Padding around speech segments

    # Thresholds
    threshold: float = 0.5  # Speech probability threshold
    neg_threshold: Optional[float] = None  # Negative threshold (for hysteresis)

    # Window settings
    window_size_samples: int = 512  # Window size for processing

    def __post_init__(self):
        """Validate configuration."""
        if self.sample_rate not in (8000, 16000):
            raise ValueError(f"Sample rate must be 8000 or 16000, got {self.sample_rate}")

        if self.num_channels != 1:
            raise ValueError(f"Only mono audio (1 channel) is supported, got {self.num_channels}")

        if not 0 <= self.threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {self.threshold}")

        # Set neg_threshold if not provided
        if self.neg_threshold is None:
            self.neg_threshold = self.threshold - 0.15

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VADConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            VADConfig instance
        """
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "model_type": self.model_type,
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "context_size": self.context_size,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
            "threshold": self.threshold,
            "neg_threshold": self.neg_threshold,
            "window_size_samples": self.window_size_samples,
        }


# Default configurations
DEFAULT_CONFIG = VADConfig()

DEFAULT_CONFIG_8K = VADConfig(
    sample_rate=8000,
    window_size_samples=256,
)

DEFAULT_CONFIG_16K = VADConfig(
    sample_rate=16000,
    window_size_samples=512,
)


__all__ = [
    "VADConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_8K",
    "DEFAULT_CONFIG_16K",
]
