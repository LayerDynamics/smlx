"""
Audio processing utilities for Whisper.

Handles audio loading, preprocessing, and mel-spectrogram computation.
"""

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np

# Audio hyperparameters (from Whisper paper)
SAMPLE_RATE = 16000
"""Target sample rate for audio (16 kHz)"""

N_FFT = 400
"""FFT window size"""

HOP_LENGTH = 160
"""Number of samples between successive frames"""

CHUNK_LENGTH = 30
"""Audio chunk length in seconds"""

N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
"""Number of samples in a chunk (480,000)"""

N_FRAMES = N_SAMPLES // HOP_LENGTH
"""Number of frames in a chunk (3,000)"""

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
"""Samples per audio token (320) - due to stride 2 in encoder"""

FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH
"""Frames per second (100)"""

TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN
"""Audio tokens per second (50)"""


def load_audio(file_path: Union[str, Path], sr: int = SAMPLE_RATE) -> mx.array:
    """Load audio file and convert to mono waveform.

    Uses ffmpeg to decode audio and resample to target sample rate.

    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 16000 Hz)

    Returns:
        Audio waveform as float32 array, normalized to [-1, 1]

    Raises:
        RuntimeError: If ffmpeg fails to load audio
        FileNotFoundError: If ffmpeg is not installed

    Example:
        >>> audio = load_audio("speech.wav")
        >>> audio.shape
        (48000,)  # 3 seconds at 16 kHz
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
    # This decodes audio, converts to mono, resamples, and outputs as raw PCM
    cmd = [
        "ffmpeg",
        "-nostdin",  # Don't expect stdin input
        "-threads",
        "0",  # Use optimal number of threads
        "-i",
        file_path,  # Input file
        "-f",
        "s16le",  # Output format: signed 16-bit little-endian
        "-ac",
        "1",  # Convert to mono
        "-acodec",
        "pcm_s16le",  # PCM codec
        "-ar",
        str(sr),  # Resample to target rate
        "-",  # Output to stdout
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


def pad_or_trim(array: mx.array, length: int = N_SAMPLES, axis: int = -1) -> mx.array:
    """Pad or trim audio array to target length.

    Args:
        array: Audio array
        length: Target length (default: N_SAMPLES = 480,000)
        axis: Axis to pad/trim (default: -1)

    Returns:
        Array with shape[axis] == length

    Example:
        >>> audio = mx.random.normal((16000,))  # 1 second
        >>> padded = pad_or_trim(audio, N_SAMPLES)
        >>> padded.shape
        (480000,)  # 30 seconds
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


@lru_cache(maxsize=2)
def mel_filters(n_mels: int = 80) -> mx.array:
    """Load mel filterbank matrix.

    This matrix projects STFT spectrogram to mel-frequency scale.

    Args:
        n_mels: Number of mel bins (80 or 128)

    Returns:
        Mel filterbank of shape (n_mels, n_fft//2 + 1)

    Raises:
        AssertionError: If n_mels not in {80, 128}
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}, must be 80 or 128"

    # Get path to mel filters
    assets_dir = Path(__file__).parent / "assets"
    mel_filters_path = assets_dir / "mel_filters.npz"

    if not mel_filters_path.exists():
        raise FileNotFoundError(
            f"Mel filters not found at {mel_filters_path}. "
            "Please ensure the assets directory is properly set up."
        )

    # Load filters
    filters = mx.load(str(mel_filters_path))
    return filters[f"mel_{n_mels}"]


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
    nperseg: int = 256,
    noverlap: int = None,
    nfft: int = None,
    pad_mode: str = "reflect",
) -> mx.array:
    """Compute Short-Time Fourier Transform.

    Args:
        x: Input signal
        window: Window function
        nperseg: Length of each segment
        noverlap: Number of samples to overlap (default: nperseg // 4)
        nfft: FFT length (default: nperseg)
        pad_mode: Padding mode ('constant' or 'reflect')

    Returns:
        STFT of shape (n_frames, nfft // 2 + 1)
    """
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

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

    # Create strided view for windowing
    strides = [noverlap, 1]
    t = (x.size - nperseg + noverlap) // noverlap
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)

    # Apply window and compute FFT
    return mx.fft.rfft(x * window)


def log_mel_spectrogram(
    audio: Union[str, Path, np.ndarray, mx.array],
    n_mels: int = 80,
    padding: int = 0,
) -> mx.array:
    """Compute log-mel spectrogram.

    This is the main audio preprocessing function for Whisper.

    Args:
        audio: Audio file path, numpy array, or MLX array
               If path: will load audio with ffmpeg
               If array: expects mono waveform at 16 kHz
        n_mels: Number of mel bins (80 or 128)
        padding: Number of zero samples to pad on the right

    Returns:
        Log-mel spectrogram of shape (n_frames, n_mels)
        Values are in range [0, 1]
        Format matches MLX Conv1d NLC format

    Example:
        >>> # From file
        >>> mel = log_mel_spectrogram("speech.wav")
        >>> mel.shape
        (3000, 80)  # 3000 frames (30 seconds), 80 mel bins

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
            audio = load_audio(audio)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        # Pad if requested
        if padding > 0:
            audio = mx.pad(audio, (0, padding))

        # Compute STFT
        window = hanning_window(N_FFT)
        freqs = stft(audio, window, nperseg=N_FFT, noverlap=HOP_LENGTH)

        # Compute magnitude spectrogram
        magnitudes = freqs[:-1, :].abs().square()

        # Apply mel filterbank
        filters = mel_filters(n_mels)
        # filters is (n_mels, n_freqs), magnitudes is (n_frames, n_freqs)
        # magnitudes @ filters.T gives (n_frames, n_mels) - matches MLX Conv1d NLC format
        mel_spec = magnitudes @ filters.T

        # Convert to log scale
        log_spec = mx.maximum(mel_spec, 1e-10).log10()

        # Normalize to [0, 1]
        # Clip to dynamic range of 80 dB
        log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
        # Shift to [0, 1]
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    finally:
        # Restore original device
        mx.set_default_device(device)


def prepare_audio(
    audio_path: Union[str, Path],
    n_mels: int = 80,
) -> mx.array:
    """Prepare audio for Whisper inference.

    Convenience function that loads audio, pads/trims to 30s,
    and computes log-mel spectrogram.

    Args:
        audio_path: Path to audio file
        n_mels: Number of mel bins

    Returns:
        Log-mel spectrogram ready for model input
        Shape: (n_mels, n_frames)

    Example:
        >>> mel = prepare_audio("speech.wav")
        >>> mel.shape
        (80, 3000)
        >>> # Ready for model:
        >>> mel = mel[None, ...]  # Add batch dimension
        >>> audio_features = model.encode_audio(mel)
    """
    # Load and normalize audio
    audio = load_audio(audio_path)

    # Pad or trim to 30 seconds
    audio = pad_or_trim(audio, N_SAMPLES)

    # Compute log-mel spectrogram
    mel = log_mel_spectrogram(audio, n_mels=n_mels)

    return mel


def get_audio_duration(audio_path: Union[str, Path]) -> float:
    """Get duration of audio file in seconds.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds

    Example:
        >>> duration = get_audio_duration("speech.wav")
        >>> print(f"Audio is {duration:.2f} seconds long")
        Audio is 12.34 seconds long
    """
    audio = load_audio(audio_path)
    return float(audio.shape[0]) / SAMPLE_RATE


def split_audio_chunks(
    audio: mx.array,
    chunk_length: int = N_SAMPLES,
    overlap: int = 0,
) -> list[mx.array]:
    """Split long audio into overlapping chunks.

    Useful for processing audio longer than 30 seconds.

    Args:
        audio: Audio waveform
        chunk_length: Length of each chunk in samples
        overlap: Number of samples to overlap between chunks

    Returns:
        List of audio chunks

    Example:
        >>> audio = load_audio("long_speech.wav")  # 2 minutes
        >>> chunks = split_audio_chunks(audio, overlap=SAMPLE_RATE)  # 1s overlap
        >>> len(chunks)
        4  # Four 30-second chunks with 1s overlap
    """
    chunks = []
    stride = chunk_length - overlap

    for start in range(0, audio.shape[0], stride):
        end = min(start + chunk_length, audio.shape[0])
        chunk = audio[start:end]

        # Pad last chunk if needed
        if chunk.shape[0] < chunk_length:
            chunk = pad_or_trim(chunk, chunk_length)

        chunks.append(chunk)

        # Stop if we've reached the end
        if end >= audio.shape[0]:
            break

    return chunks
