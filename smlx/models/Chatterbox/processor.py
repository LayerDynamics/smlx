#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Processor for Chatterbox TTS.

Handles text processing and audio preprocessing for voice cloning.
"""

from typing import Optional, Union
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import audio_utils


class ChatterboxProcessor:
    """
    Processor for Chatterbox TTS.

    Handles:
    - Text tokenization
    - Audio preprocessing for voice cloning
    - Mel-spectrogram extraction

    Args:
        tokenizer: HuggingFace tokenizer
        sample_rate: Audio sample rate
        n_mels: Number of mel bins
    """

    def __init__(
        self,
        tokenizer=None,
        sample_rate: int = 24000,
        n_mels: int = 80,
    ):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.num_mels = n_mels  # Alias for compatibility

    def __call__(self, text: str, **kwargs):
        """
        Process text to tokens.

        Args:
            text: Input text

        Returns:
            Token IDs as MLX array (1D, no batch dimension)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        # Tokenize (returns list of token IDs)
        tokens = self.tokenizer.encode(text)

        # Convert to MLX array (1D)
        return mx.array(tokens)

    def process_audio(
        self, audio: Union[str, Path, np.ndarray, mx.array], sr: Optional[int] = None
    ) -> mx.array:
        """
        Process reference audio for voice cloning.

        Uses pure MLX implementation - no librosa dependency.

        Args:
            audio: Audio file path, numpy array, or MLX array
            sr: Sample rate of input audio (only used if audio is array)

        Returns:
            Mel-spectrogram as MLX array
            Shape: (time, n_mels)

        Example:
            >>> # From file
            >>> mel = processor.process_audio("reference.wav")
            >>> mel.shape
            (375, 80)

            >>> # From array
            >>> audio_array = np.random.randn(24000)
            >>> mel = processor.process_audio(audio_array, sr=24000)
        """
        # Load audio if it's a file path
        if isinstance(audio, (str, Path)):
            return audio_utils.log_mel_spectrogram(
                audio,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
            )

        # Convert to MLX array if needed
        if not isinstance(audio, mx.array):
            audio = mx.array(audio)

        # Resample if needed
        if sr is not None and sr != self.sample_rate:
            audio = audio_utils.resample_audio(audio, sr, self.sample_rate)

        # Extract mel-spectrogram using pure MLX
        mel = audio_utils.log_mel_spectrogram(
            audio,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
        )

        return mel

    def load_audio(self, file_path: Union[str, Path]) -> mx.array:
        """
        Load audio from file.

        Uses FFmpeg for audio loading and resampling.

        Args:
            file_path: Path to audio file

        Returns:
            Audio waveform as MLX array, normalized to [-1, 1]

        Example:
            >>> audio = processor.load_audio("speech.wav")
            >>> audio.shape
            (48000,)  # 2 seconds at 24kHz
        """
        return audio_utils.load_audio(file_path, sr=self.sample_rate)

    def decode(self, token_ids: list) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def create_processor(tokenizer=None, sample_rate: int = 24000) -> ChatterboxProcessor:
    """
    Create Chatterbox processor.

    Args:
        tokenizer: HuggingFace tokenizer
        sample_rate: Audio sample rate

    Returns:
        ChatterboxProcessor instance
    """
    return ChatterboxProcessor(tokenizer=tokenizer, sample_rate=sample_rate)
