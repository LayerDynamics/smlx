"""
Audio processing utilities for TTS models.

Handles mel-spectrogram computation, normalization, and audio I/O
optimized for text-to-speech applications.
"""

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np

# TTS audio hyperparameters
SAMPLE_RATE = 24000
"""Target sample rate for TTS (24 kHz)"""

N_FFT = 1024
"""FFT window size"""

HOP_LENGTH = 256
"""Number of samples between successive frames (12.5 ms at 24kHz)"""

WIN_LENGTH = 1024
"""Window length for STFT"""

N_MELS = 80
"""Number of mel frequency bins"""

FMIN = 0.0
"""Minimum frequency for mel filterbank (Hz)"""

FMAX = 8000.0
"""Maximum frequency for mel filterbank (Hz) - typically 8kHz for TTS"""

MEL_MIN_DB = -100.0
"""Minimum dB for mel-spectrogram dynamic range"""

MEL_MAX_DB = 0.0
"""Maximum dB for mel-spectrogram"""


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
        (72000,)  # 3 seconds at 24 kHz
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
            "ffmpeg not found. Please install ffmpeg: " "https://ffmpeg.org/download.html"
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
    audio = mx.array(audio).astype(mx.float32) / 32768.0

    return audio


def save_audio(
    file_path: Union[str, Path], audio: mx.array, sr: int = SAMPLE_RATE
) -> None:
    """Save audio waveform to file.

    Args:
        file_path: Output file path
        audio: Audio waveform array (normalized to [-1, 1])
        sr: Sample rate (default: 24000 Hz)

    Example:
        >>> audio = mx.random.normal((24000,))
        >>> save_audio("output.wav", audio)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to int16 PCM
    audio_np = np.array(audio)
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    # Use ffmpeg to write audio file
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-f",
        "s16le",  # Input format
        "-ar",
        str(sr),  # Sample rate
        "-ac",
        "1",  # Mono
        "-i",
        "pipe:0",  # Read from stdin
        str(file_path),
    ]

    subprocess.run(cmd, input=audio_int16.tobytes(), check=True, capture_output=True)


@lru_cache(maxsize=4)
def mel_filters_matrix(
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    sample_rate: int = SAMPLE_RATE,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> mx.array:
    """Create mel filterbank matrix.

    This matrix projects STFT magnitude spectrogram to mel-frequency scale.

    Args:
        n_mels: Number of mel bins
        n_fft: FFT size
        sample_rate: Audio sample rate
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)

    Returns:
        Mel filterbank of shape (n_mels, n_fft // 2 + 1)

    Note:
        Uses the HTK formula for mel scale conversion.
    """
    # Helper functions for mel scale
    def hz_to_mel(frequencies):
        """Convert Hz to mels (HTK formula)"""
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    def mel_to_hz(mels):
        """Convert mels to Hz (HTK formula)"""
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Frequency bins
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2, n_freqs)

    # Mel scale bins
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_bins = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_bins = mel_to_hz(mel_bins)

    # Create triangular filterbank
    filters = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        # Left, center, right points for triangular filter
        left = hz_bins[i]
        center = hz_bins[i + 1]
        right = hz_bins[i + 2]

        # Rising slope
        rising_mask = (freqs >= left) & (freqs < center)
        filters[i, rising_mask] = (freqs[rising_mask] - left) / (center - left)

        # Falling slope
        falling_mask = (freqs >= center) & (freqs < right)
        filters[i, falling_mask] = (right - freqs[falling_mask]) / (right - center)

    return mx.array(filters)


@lru_cache(maxsize=2)
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
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: Optional[int] = None,
    window: Optional[mx.array] = None,
    center: bool = True,
    pad_mode: str = "reflect",
) -> mx.array:
    """Compute Short-Time Fourier Transform.

    Args:
        x: Input signal (waveform)
        n_fft: FFT size
        hop_length: Number of samples between frames
        win_length: Window length (default: n_fft)
        window: Window function (default: Hanning window)
        center: Whether to center frames (pad signal)
        pad_mode: Padding mode ('constant' or 'reflect')

    Returns:
        STFT of shape (n_frames, n_fft // 2 + 1) - complex values
    """
    if win_length is None:
        win_length = n_fft

    if window is None:
        window = hanning_window(win_length)

    # Pad window if needed
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = mx.pad(window, [(pad_left, pad_right)])

    # Center-pad signal
    if center:
        padding = n_fft // 2
        if pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            x = mx.concatenate([prefix, x, suffix])
        else:  # constant
            x = mx.pad(x, [(padding, padding)])

    # Create frame indices
    n_frames = 1 + (x.shape[0] - n_fft) // hop_length

    # Extract frames using strided indexing
    strides = [hop_length, 1]
    shape = [n_frames, n_fft]
    frames = mx.as_strided(x, shape=shape, strides=strides)

    # Apply window and compute FFT
    windowed = frames * window
    return mx.fft.rfft(windowed)


def compute_mel_spectrogram(
    waveform: Union[mx.array, np.ndarray],
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: Optional[int] = None,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> mx.array:
    """Compute mel-spectrogram from waveform.

    Args:
        waveform: Audio waveform array
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop length between frames
        win_length: Window length (default: n_fft)
        n_mels: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel-spectrogram of shape (n_frames, n_mels)
        Values in linear scale (not dB)

    Example:
        >>> audio = load_audio("speech.wav")
        >>> mel = compute_mel_spectrogram(audio)
        >>> mel.shape
        (300, 80)  # ~3 seconds at 24kHz
    """
    if not isinstance(waveform, mx.array):
        waveform = mx.array(waveform)

    # Compute STFT
    stft_matrix = stft(
        waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )

    # Compute power spectrogram
    power_spec = stft_matrix.abs().square()

    # Apply mel filterbank
    mel_basis = mel_filters_matrix(n_mels, n_fft, sample_rate, fmin, fmax)
    # mel_basis: (n_mels, n_freqs), power_spec: (n_frames, n_freqs)
    # Result: (n_frames, n_mels)
    mel_spec = power_spec @ mel_basis.T

    return mel_spec


def mel_to_db(mel_spec: mx.array, min_db: float = MEL_MIN_DB) -> mx.array:
    """Convert mel-spectrogram to dB scale.

    Args:
        mel_spec: Mel-spectrogram in linear scale
        min_db: Minimum dB value (floor)

    Returns:
        Mel-spectrogram in dB scale
    """
    # Convert to log scale
    log_spec = 10.0 * mx.maximum(mel_spec, 1e-10).log10()

    # Clip to minimum dB
    return mx.maximum(log_spec, min_db)


def db_to_mel(mel_db: mx.array) -> mx.array:
    """Convert dB-scale mel-spectrogram to linear scale.

    Args:
        mel_db: Mel-spectrogram in dB scale

    Returns:
        Mel-spectrogram in linear scale
    """
    return mx.power(10.0, mel_db / 10.0)


def normalize_mel(
    mel_spec: mx.array, min_db: float = MEL_MIN_DB, max_db: float = MEL_MAX_DB
) -> mx.array:
    """Normalize mel-spectrogram to [0, 1] range.

    Converts to dB scale and normalizes.

    Args:
        mel_spec: Mel-spectrogram in linear scale
        min_db: Minimum dB value
        max_db: Maximum dB value

    Returns:
        Normalized mel-spectrogram in [0, 1]
    """
    # Convert to dB
    mel_db = mel_to_db(mel_spec, min_db)

    # Normalize to [0, 1]
    return (mel_db - min_db) / (max_db - min_db)


def denormalize_mel(
    mel_norm: mx.array, min_db: float = MEL_MIN_DB, max_db: float = MEL_MAX_DB
) -> mx.array:
    """Denormalize mel-spectrogram from [0, 1] to linear scale.

    Args:
        mel_norm: Normalized mel-spectrogram in [0, 1]
        min_db: Minimum dB value used in normalization
        max_db: Maximum dB value used in normalization

    Returns:
        Mel-spectrogram in linear scale
    """
    # Denormalize from [0, 1]
    mel_db = mel_norm * (max_db - min_db) + min_db

    # Convert from dB to linear
    return db_to_mel(mel_db)


def griffin_lim(
    mel_spec: mx.array,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: Optional[int] = None,
    n_iter: int = 32,
) -> mx.array:
    """Reconstruct audio from mel-spectrogram using Griffin-Lim algorithm.

    This is a fallback method for mel-to-waveform conversion when no
    neural vocoder is available. Quality is lower than neural vocoders.

    Args:
        mel_spec: Mel-spectrogram (n_frames, n_mels)
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_iter: Number of Griffin-Lim iterations

    Returns:
        Reconstructed audio waveform

    Note:
        This is primarily for debugging/testing. Use neural vocoder
        (HiFi-GAN) for production quality synthesis.
    """
    if win_length is None:
        win_length = n_fft

    # Convert mel back to linear spectrogram (approximate)
    mel_basis = mel_filters_matrix(N_MELS, n_fft)
    # Pseudo-inverse to convert mel → linear spec
    mel_basis_inv = mx.linalg.pinv(mel_basis)
    linear_spec = mel_spec @ mel_basis_inv

    # Initialize with random phase
    angles = mx.random.uniform(0, 2 * np.pi, linear_spec.shape)
    complex_spec = linear_spec * mx.exp(1j * angles)

    window = hanning_window(win_length)

    # Griffin-Lim iterations
    for _ in range(n_iter):
        # Inverse STFT
        waveform = _istft(complex_spec, hop_length, win_length, window)

        # Forward STFT
        stft_matrix = stft(waveform, n_fft, hop_length, win_length, window)

        # Update magnitude, keep phase
        angles = mx.angle(stft_matrix)
        complex_spec = linear_spec * mx.exp(1j * angles)

    # Final inverse STFT
    waveform = _istft(complex_spec, hop_length, win_length, window)

    return waveform


def _istft(
    stft_matrix: mx.array,
    hop_length: int,
    win_length: int,
    window: mx.array,
    center: bool = True,
) -> mx.array:
    """Inverse Short-Time Fourier Transform.

    Args:
        stft_matrix: STFT matrix (n_frames, n_fft // 2 + 1)
        hop_length: Hop length
        win_length: Window length
        window: Window function
        center: Whether frames were centered

    Returns:
        Reconstructed waveform
    """
    n_frames = stft_matrix.shape[0]
    expected_signal_len = n_frames * hop_length + win_length

    # Inverse FFT
    frames = mx.fft.irfft(stft_matrix, n=win_length)

    # Apply window
    frames = frames * window

    # Overlap-add
    waveform = mx.zeros(expected_signal_len)
    window_sum = mx.zeros(expected_signal_len)

    for i in range(n_frames):
        start = i * hop_length
        end = start + win_length
        waveform[start:end] = waveform[start:end] + frames[i]
        window_sum[start:end] = window_sum[start:end] + window

    # Normalize by window sum
    nonzero_mask = window_sum > 1e-8
    waveform = mx.where(nonzero_mask, waveform / window_sum, waveform)

    # Remove padding if centered
    if center:
        padding = win_length // 2
        waveform = waveform[padding:-padding]

    return waveform


def get_audio_duration(audio_path: Union[str, Path], sr: int = SAMPLE_RATE) -> float:
    """Get duration of audio file in seconds.

    Args:
        audio_path: Path to audio file
        sr: Sample rate

    Returns:
        Duration in seconds
    """
    audio = load_audio(audio_path, sr=sr)
    return float(audio.shape[0]) / sr
