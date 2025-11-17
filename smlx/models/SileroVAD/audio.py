#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Audio preprocessing for Silero VAD.

Handles loading and preprocessing audio for voice activity detection.
"""

from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np


def load_audio(
    audio_source: Union[str, Path, np.ndarray, mx.array],
    sample_rate: int = 16000,
) -> mx.array:
    """Load and preprocess audio for VAD.

    Args:
        audio_source: Audio file path, numpy array, or MLX array
        sample_rate: Target sample rate (8000 or 16000)

    Returns:
        Audio waveform as MLX array [num_samples]
    """
    if isinstance(audio_source, (str, Path)):
        # Load from file
        audio = load_audio_file(audio_source, sample_rate)
    elif isinstance(audio_source, np.ndarray):
        # Convert from numpy
        audio = mx.array(audio_source.astype(np.float32))
    elif isinstance(audio_source, mx.array):
        # Already MLX array
        audio = audio_source
    else:
        raise ValueError(
            f"Unsupported audio source type: {type(audio_source)}"
        )

    # Ensure mono
    if audio.ndim > 1:
        audio = mx.mean(audio, axis=-1)

    # Normalize to [-1, 1]
    audio = normalize_audio(audio)

    return audio


def load_audio_file(
    file_path: Union[str, Path],
    sample_rate: int = 16000,
) -> mx.array:
    """Load audio from file.

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate

    Returns:
        Audio waveform as MLX array
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for loading audio files. "
            "Install with: pip install soundfile"
        )

    # Load audio
    audio, sr = sf.read(str(file_path), dtype="float32")

    # Resample if needed
    if sr != sample_rate:
        audio = resample_audio(audio, sr, sample_rate)

    return mx.array(audio)


def resample_audio(
    audio: Union[np.ndarray, mx.array],
    orig_sr: int,
    target_sr: int,
) -> mx.array:
    """Resample audio to target sample rate.

    Args:
        audio: Audio waveform
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        if isinstance(audio, np.ndarray):
            return mx.array(audio)
        return audio

    try:
        import librosa
    except ImportError:
        raise ImportError(
            "librosa is required for audio resampling. "
            "Install with: pip install librosa"
        )

    # Convert to numpy if needed
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Resample
    audio_resampled = librosa.resample(
        audio, orig_sr=orig_sr, target_sr=target_sr
    )

    return mx.array(audio_resampled)


def normalize_audio(audio: mx.array) -> mx.array:
    """Normalize audio to [-1, 1] range.

    Args:
        audio: Audio waveform

    Returns:
        Normalized audio
    """
    # Get max absolute value
    max_val = mx.max(mx.abs(audio))

    if max_val > 0:
        audio = audio / max_val

    return audio


def split_audio_chunks(
    audio: mx.array,
    chunk_size: int,
    overlap: int = 0,
) -> list[mx.array]:
    """Split audio into chunks with optional overlap.

    Args:
        audio: Audio waveform [num_samples]
        chunk_size: Size of each chunk in samples
        overlap: Overlap between chunks in samples

    Returns:
        List of audio chunks
    """
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(audio), step):
        end = start + chunk_size
        if end > len(audio):
            # Pad last chunk if needed
            chunk = audio[start:]
            padding = chunk_size - len(chunk)
            if padding > 0:
                chunk = mx.pad(chunk, [(0, padding)])
        else:
            chunk = audio[start:end]

        chunks.append(chunk)

        # Stop if we've covered the audio
        if end >= len(audio):
            break

    return chunks


def create_audio_buffer(sample_rate: int = 16000, duration_seconds: float = 10.0) -> mx.array:
    """Create an empty audio buffer.

    Args:
        sample_rate: Sample rate
        duration_seconds: Buffer duration in seconds

    Returns:
        Empty audio buffer
    """
    buffer_size = int(sample_rate * duration_seconds)
    return mx.zeros(buffer_size)


__all__ = [
    "load_audio",
    "load_audio_file",
    "resample_audio",
    "normalize_audio",
    "split_audio_chunks",
    "create_audio_buffer",
]
