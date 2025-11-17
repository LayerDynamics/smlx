#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Pure MLX audio processing utilities for Chatterbox TTS.

Adapted from Whisper audio processing with modifications for 24kHz audio.
No librosa dependency - all operations in pure MLX.
"""

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np


# Chatterbox audio hyperparameters (24 kHz)
SAMPLE_RATE = 24000
"""Target sample rate for Chatterbox (24 kHz)"""

N_FFT = 1024
"""FFT window size"""

HOP_LENGTH = 256
"""Number of samples between successive frames"""

WIN_LENGTH = 1024
"""Window length for STFT"""

N_MELS = 80
"""Number of mel-frequency bins"""


def load_audio(file_path: Union[str, Path], sr: int = SAMPLE_RATE) -> mx.array:
    """Load audio file and convert to mono waveform.

    Uses ffmpeg to decode audio and resample to target sample rate.

    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 24000 Hz)

    Returns:
        Audio waveform as float32 array, normalized to [-1, 1]

    Raises:
        RuntimeError: If ffmpeg fails to load audio
        FileNotFoundError: If ffmpeg is not installed

    Example:
        >>> audio = load_audio("speech.wav")
        >>> audio.shape
        (48000,)  # 2 seconds at 24 kHz
    """
    file_path = str(file_path)

    # Check if ffmpeg is available
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise FileNotFoundError(
            "ffmpeg not found. Please install ffmpeg: "
            "https://ffmpeg.org/download.html"
        )

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        pcm_data = result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Failed to load audio: {error_msg}") from e

    # Convert PCM bytes to float array
    audio = np.frombuffer(pcm_data, dtype=np.int16)
    audio = mx.array(audio).astype(mx.float32) / 32768.0  # Normalize to [-1, 1]

    return audio


@lru_cache(maxsize=2)
def mel_filters(
    n_mels: int = 80,
    n_fft: int = N_FFT,
    sample_rate: int = SAMPLE_RATE,
) -> mx.array:
    """Create mel filterbank matrix.

    This matrix projects STFT spectrogram to mel-frequency scale.

    Args:
        n_mels: Number of mel bins
        n_fft: FFT size
        sample_rate: Sample rate

    Returns:
        Mel filterbank of shape (n_mels, n_fft//2 + 1)
    """
    # Mel scale parameters
    f_min = 0.0
    f_max = sample_rate / 2.0

    # Convert Hz to mel
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    # Convert mel to Hz
    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    # Create mel points
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert Hz points to FFT bin numbers
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create filterbank
    filters = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        for j in range(left, center):
            filters[i, j] = (j - left) / (center - left)

        # Falling slope
        for j in range(center, right):
            filters[i, j] = (right - j) / (right - center)

    # Normalize to unit area
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    filters *= enorm[:, np.newaxis]

    return mx.array(filters)


@lru_cache(maxsize=1)
def hanning_window(size: int) -> mx.array:
    """Create Hanning window.

    Args:
        size: Window size

    Returns:
        Hanning window of shape (size,)
    """
    return mx.array(np.hanning(size + 1)[:-1])


def stft(
    x: mx.array,
    window: mx.array,
    nperseg: int = N_FFT,
    noverlap: int = None,
    nfft: int = None,
    pad_mode: str = "reflect",
) -> mx.array:
    """Compute Short-Time Fourier Transform.

    Args:
        x: Input signal
        window: Window function
        nperseg: Length of each segment
        noverlap: Number of samples to overlap
        nfft: FFT length (default: nperseg)
        pad_mode: Padding mode ('constant' or 'reflect')

    Returns:
        STFT of shape (n_frames, nfft // 2 + 1)
    """
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nperseg - HOP_LENGTH

    def _pad(x: mx.array, padding: int, mode: str = "constant") -> mx.array:
        if mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode: {mode}")

    # Pad signal
    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)

    # Calculate hop length from noverlap
    hop = nperseg - noverlap

    # Create strided view for windowing
    strides = [hop, 1]
    t = (x.size - nperseg + hop) // hop
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)

    # Apply window and compute FFT
    return mx.fft.rfft(x * window)


def log_mel_spectrogram(
    audio: Union[str, Path, np.ndarray, mx.array],
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> mx.array:
    """Compute log-mel spectrogram.

    Pure MLX implementation - no librosa dependency.

    Args:
        audio: Audio file path, numpy array, or MLX array
               If path: will load audio with ffmpeg
               If array: expects mono waveform at target sample rate
        n_mels: Number of mel bins (default: 80)
        n_fft: FFT size (default: 1024)
        hop_length: Hop length (default: 256)
        sample_rate: Sample rate (default: 24000)

    Returns:
        Log-mel spectrogram of shape (time_frames, n_mels)
        Format matches MLX Conv1d NLC format
        Values are normalized log-magnitude

    Example:
        >>> # From file
        >>> mel = log_mel_spectrogram("speech.wav")
        >>> mel.shape
        (375, 80)  # 2 seconds at 24kHz with hop=256

        >>> # From array
        >>> audio = load_audio("speech.wav")
        >>> mel = log_mel_spectrogram(audio)
    """
    # Temporarily switch to CPU for audio processing
    device = mx.default_device()
    mx.set_default_device(mx.cpu)

    try:
        # Load audio if needed
        if isinstance(audio, (str, Path)):
            audio = load_audio(audio, sr=sample_rate)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        # Compute STFT
        window = hanning_window(n_fft)
        freqs = stft(
            audio,
            window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
        )

        # Compute magnitude spectrogram
        magnitudes = freqs[:-1, :].abs().square()

        # Apply mel filterbank
        filters = mel_filters(n_mels, n_fft, sample_rate)
        # filters is (n_mels, n_freqs), magnitudes is (n_frames, n_freqs)
        # magnitudes @ filters.T gives (n_frames, n_mels)
        mel_spec = magnitudes @ filters.T

        # Convert to log scale
        log_spec = mx.maximum(mel_spec, 1e-10).log10()

        # Normalize to reasonable range
        # Clip to dynamic range of 80 dB
        log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)

        # Shift to [0, 1] approximately
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    finally:
        # Restore original device
        mx.set_default_device(device)


def resample_audio(
    audio: mx.array,
    orig_sr: int,
    target_sr: int,
) -> mx.array:
    """Resample audio to target sample rate.

    Simple linear interpolation resampling.
    For higher quality, use librosa.resample if available.

    Args:
        audio: Input audio (MLX array)
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio

    Example:
        >>> audio_16k = mx.random.normal((16000,))  # 1 second at 16kHz
        >>> audio_24k = resample_audio(audio_16k, 16000, 24000)
        >>> audio_24k.shape
        (24000,)  # 1 second at 24kHz
    """
    if orig_sr == target_sr:
        return audio

    # Calculate target length
    duration = audio.shape[0] / orig_sr
    target_length = int(duration * target_sr)

    # Simple linear interpolation
    # Convert to numpy for interpolation
    audio_np = np.array(audio)
    indices = np.linspace(0, len(audio_np) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(audio_np)), audio_np)

    return mx.array(resampled)


def pad_or_trim(array: mx.array, length: int, axis: int = -1) -> mx.array:
    """Pad or trim array to target length.

    Args:
        array: Input array
        length: Target length
        axis: Axis to pad/trim (default: -1)

    Returns:
        Array with shape[axis] == length

    Example:
        >>> audio = mx.random.normal((16000,))
        >>> padded = pad_or_trim(audio, 24000)
        >>> padded.shape
        (24000,)
    """
    if array.shape[axis] > length:
        # Trim to length
        indices = [slice(None)] * array.ndim
        indices[axis] = slice(0, length)
        array = array[tuple(indices)]

    if array.shape[axis] < length:
        # Pad to length
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = mx.pad(array, pad_widths)

    return array
