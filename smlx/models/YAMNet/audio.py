#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Audio preprocessing for YAMNet.

Handles loading, resampling, and mel spectrogram extraction
for audio classification.
"""

from pathlib import Path
from typing import Union, Optional

import mlx.core as mx
import numpy as np

from .config import YAMNetConfig, DEFAULT_CONFIG


def load_audio(
    audio_source: Union[str, Path, np.ndarray, mx.array],
    sample_rate: int = 16000,
    target_sr: Optional[int] = None,
) -> mx.array:
    """Load audio from various sources.

    Args:
        audio_source: Audio file path, numpy array, or MLX array
        sample_rate: Target sample rate (YAMNet uses 16kHz)
        target_sr: Alias for sample_rate (for compatibility)

    Returns:
        Audio as MLX array (mono, float32)
    """
    # Support both parameter names
    if target_sr is not None:
        sample_rate = target_sr
    if isinstance(audio_source, (str, Path)):
        audio = load_audio_file(audio_source, sample_rate)
    elif isinstance(audio_source, np.ndarray):
        audio = mx.array(audio_source.astype(np.float32))
    elif isinstance(audio_source, mx.array):
        audio = audio_source
    else:
        raise TypeError(f"Unsupported audio source type: {type(audio_source)}")

    # Convert to mono if needed
    if audio.ndim > 1:
        audio = mx.mean(audio, axis=-1)

    return normalize_audio(audio)


def load_audio_file(file_path: Union[str, Path], sample_rate: int = 16000) -> mx.array:
    """Load audio file and resample to target rate.

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate

    Returns:
        Audio as MLX array
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for audio loading. Install with: pip install soundfile"
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Load audio
    audio, sr = sf.read(str(file_path), dtype="float32")

    # Resample if needed
    if sr != sample_rate:
        audio = resample_audio(audio, sr, sample_rate)

    return mx.array(audio)


def resample_audio(
    audio: Union[np.ndarray, mx.array], orig_sr: int, target_sr: int
) -> mx.array:
    """Resample audio to target sample rate.

    Args:
        audio: Audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return mx.array(audio) if isinstance(audio, np.ndarray) else audio

    try:
        import librosa
    except ImportError:
        raise ImportError(
            "librosa is required for resampling. Install with: pip install librosa"
        )

    # Convert to numpy for librosa
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    audio_resampled = librosa.resample(
        audio, orig_sr=orig_sr, target_sr=target_sr
    )

    return mx.array(audio_resampled)


def normalize_audio(audio: mx.array) -> mx.array:
    """Normalize audio to [-1, 1] range.

    Args:
        audio: Audio array

    Returns:
        Normalized audio
    """
    max_val = mx.max(mx.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def compute_mel_spectrogram(
    audio: mx.array,
    sample_rate: Optional[int] = None,
    config: Optional[YAMNetConfig] = None,
) -> mx.array:
    """Compute mel spectrogram for YAMNet.

    YAMNet uses mel spectrograms as input features. This computes
    the log-mel spectrogram with the parameters expected by YAMNet.

    Args:
        audio: Audio waveform (mono, 16kHz)
        sample_rate: Sample rate of audio (overrides config if provided)
        config: YAMNet configuration

    Returns:
        Mel spectrogram (num_frames, num_mel_bins)
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Override sample rate if provided
    if sample_rate is not None and sample_rate != config.sample_rate:
        # Create a modified config with the new sample rate
        import copy
        config = copy.copy(config)
        object.__setattr__(config, 'sample_rate', sample_rate)
    # Use librosa for mel spectrogram computation
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "librosa is required for mel spectrogram. Install with: pip install librosa"
        )

    # Convert to numpy for librosa
    audio_np = np.array(audio)

    # Compute STFT
    n_fft = int(2 ** np.ceil(np.log2(config.stft_window_length_samples)))

    mel_spec = librosa.feature.melspectrogram(
        y=audio_np,
        sr=config.sample_rate,
        n_fft=n_fft,
        hop_length=config.stft_hop_length_samples,
        win_length=config.stft_window_length_samples,
        n_mels=config.num_mel_bins,
        fmin=config.mel_min_hz,
        fmax=config.mel_max_hz,
        power=2.0,  # Power spectrogram
    )

    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Convert to MLX and transpose to (time, freq)
    log_mel_spec = mx.array(log_mel_spec.T)

    return log_mel_spec


def extract_patches(
    mel_spectrogram: mx.array,
    sample_rate: Optional[int] = None,
    config: Optional[YAMNetConfig] = None,
) -> mx.array:
    """Extract overlapping patches from mel spectrogram.

    YAMNet processes audio in overlapping patches. Each patch
    is a fixed-size window of mel spectrogram frames.

    Args:
        mel_spectrogram: Mel spectrogram (num_frames, num_mel_bins)
        sample_rate: Sample rate (overrides config if provided, but not used in this function)
        config: YAMNet configuration

    Returns:
        Patches (num_patches, patch_frames, patch_bands)
    """
    if config is None:
        config = DEFAULT_CONFIG
    num_frames = mel_spectrogram.shape[0]
    patch_frames = config.patch_frames

    # Calculate hop in frames
    hop_frames = int(
        round(config.patch_hop_seconds / config.stft_hop_seconds)
    )

    patches = []
    for start_frame in range(0, num_frames - patch_frames + 1, hop_frames):
        end_frame = start_frame + patch_frames
        patch = mel_spectrogram[start_frame:end_frame, :]
        patches.append(patch)

    if not patches:
        # Audio too short, pad the mel spectrogram
        if num_frames < patch_frames:
            padding = patch_frames - num_frames
            mel_spectrogram = mx.pad(
                mel_spectrogram,
                ((0, padding), (0, 0)),
                constant_values=0,
            )
            patches = [mel_spectrogram]

    return mx.stack(patches)


def preprocess_audio(
    audio: Union[str, Path, np.ndarray, mx.array],
    sample_rate: Optional[int] = None,
    config: Optional[YAMNetConfig] = None,
) -> mx.array:
    """Complete preprocessing pipeline for YAMNet.

    Loads audio, computes mel spectrogram, and extracts patches
    ready for model inference.

    Args:
        audio: Audio source (file path, numpy array, or MLX array)
        sample_rate: Sample rate of audio (overrides config if provided)
        config: YAMNet configuration

    Returns:
        Patches (num_patches, patch_frames, patch_bands)
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Use provided sample rate or config default
    sr = sample_rate if sample_rate is not None else config.sample_rate

    # Load and normalize audio
    audio_array = load_audio(audio, sr)

    # Compute mel spectrogram
    mel_spec = compute_mel_spectrogram(audio_array, sample_rate=sr, config=config)

    # Extract patches
    patches = extract_patches(mel_spec, config=config)

    return patches


def postprocess_predictions(
    predictions: mx.array,
    aggregate: str = "max",
) -> mx.array:
    """Aggregate predictions across time frames.

    YAMNet outputs predictions for each patch. This aggregates
    them into a single prediction.

    Args:
        predictions: Predictions per patch (num_patches, num_classes)
        aggregate: Aggregation method ("max", "mean", or "median")

    Returns:
        Aggregated predictions (num_classes,)
    """
    if aggregate == "max":
        return mx.max(predictions, axis=0)
    elif aggregate == "mean":
        return mx.mean(predictions, axis=0)
    elif aggregate == "median":
        # MLX doesn't have median, use numpy
        preds_np = np.array(predictions)
        median = np.median(preds_np, axis=0)
        return mx.array(median)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregate}")


__all__ = [
    "load_audio",
    "load_audio_file",
    "resample_audio",
    "normalize_audio",
    "compute_mel_spectrogram",
    "extract_patches",
    "preprocess_audio",
    "postprocess_predictions",
]
