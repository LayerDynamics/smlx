#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Speech synthesis functions for Orpheus-150M.

Provides functions for generating speech from text.
"""

from typing import Generator, Optional

import mlx.core as mx
import numpy as np

from .model import Orpheus150M
from .processor import TextProcessor


def synthesize(
    model: Orpheus150M,
    processor: TextProcessor,
    text: str,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Synthesize speech from text.

    Args:
        model: Orpheus-150M model
        processor: Text processor
        text: Input text to synthesize
        sample_rate: Output sample rate (16000 or 24000)

    Returns:
        Audio waveform as numpy array
        Shape: (samples,)
        Range: [-1.0, 1.0]

    Example:
        >>> model, processor = load()
        >>> audio = synthesize(model, processor, "Hello world", sample_rate=24000)
        >>> print(audio.shape)  # (n_samples,)
        >>> import soundfile as sf
        >>> sf.write("output.wav", audio, 24000)

    Note:
        This model uses HiFi-GAN V3 neural vocoder. For best quality:
        - Load pre-trained model weights from HuggingFace Hub
        - Load pre-trained vocoder weights (nvidia/tts_hifigan)
    """
    # Process text
    input_ids = processor(text, padding=False)

    # Add batch dimension
    input_ids = mx.expand_dims(input_ids, axis=0)

    # Generate audio
    waveform, mel, durations = model(input_ids)

    # Remove batch dimension
    waveform = waveform[0]

    # Convert to numpy
    audio = np.array(waveform)

    # Normalize to [-1, 1]
    if audio.max() > 0:
        audio = audio / np.abs(audio).max()

    duration_s = len(audio) / sample_rate
    if getattr(model, "weights_loaded", False):
        print(
            f"\n✓ Generated {duration_s:.2f}s of audio (HiFi-GAN V3 vocoder)"
        )
    else:
        # Honesty: with random-init weights the forward pass produces noise,
        # not intelligible speech. Do not present it as a successful synthesis.
        print(
            f"\n⚠ Produced {duration_s:.2f}s of audio from UNINITIALIZED (random) "
            "weights — this is noise, not speech."
        )
        print("  Load pre-trained weights to synthesize real audio "
              "(see load_weights / load_vocoder_weights).")

    return audio


def synthesize_batch(
    model: Orpheus150M,
    processor: TextProcessor,
    texts: list[str],
    sample_rate: int = 24000,
) -> list[np.ndarray]:
    """
    Synthesize speech for multiple texts.

    Args:
        model: Orpheus-150M model
        processor: Text processor
        texts: List of input texts
        sample_rate: Output sample rate

    Returns:
        List of audio waveforms

    Example:
        >>> texts = ["Hello", "World", "How are you?"]
        >>> audios = synthesize_batch(model, processor, texts)
        >>> for i, audio in enumerate(audios):
        ...     sf.write(f"output_{i}.wav", audio, 24000)
    """
    audios = []

    for text in texts:
        audio = synthesize(model, processor, text, sample_rate)
        audios.append(audio)

    return audios


def stream_synthesize(
    model: Orpheus150M,
    processor: TextProcessor,
    text: str,
    sample_rate: int = 24000,
    chunk_size: int = 50,
) -> Generator[np.ndarray, None, None]:
    """
    Stream synthesis for lower latency.

    Generates audio in chunks for real-time playback.

    Args:
        model: Orpheus-150M model
        processor: Text processor
        text: Input text
        sample_rate: Output sample rate
        chunk_size: Number of words per chunk

    Yields:
        Audio chunks as numpy arrays

    Example:
        >>> for audio_chunk in stream_synthesize(model, processor, long_text):
        ...     # Play audio_chunk immediately
        ...     play_audio(audio_chunk)
    """
    # Split text into chunks
    words = text.split()

    for i in range(0, len(words), chunk_size):
        # Get chunk
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)

        # Synthesize chunk
        audio_chunk = synthesize(model, processor, chunk_text, sample_rate)

        yield audio_chunk


def estimate_duration(
    model: Orpheus150M,
    processor: TextProcessor,
    text: str,
    sample_rate: int = 24000,
) -> float:
    """
    Estimate speech duration without generating audio.

    Args:
        model: Orpheus-150M model
        processor: Text processor
        text: Input text
        sample_rate: Sample rate

    Returns:
        Estimated duration in seconds

    Example:
        >>> duration = estimate_duration(model, processor, "Hello world")
        >>> print(f"Estimated duration: {duration:.2f}s")
    """
    # Process text
    input_ids = processor(text, padding=False)
    input_ids = mx.expand_dims(input_ids, axis=0)

    # Encode text
    encoder_output = model.text_encoder(input_ids)

    # Predict durations
    durations = model.duration_predictor(encoder_output)

    # Sum durations (in frames)
    total_frames = mx.sum(durations)

    # Convert to seconds
    # hop_length frames per second
    hop_length = model.config.hop_length
    duration_seconds = float(total_frames * hop_length / sample_rate)

    return duration_seconds


def synthesize_with_speed(
    model: Orpheus150M,
    processor: TextProcessor,
    text: str,
    speed: float = 1.0,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Synthesize speech with speed control.

    Args:
        model: Orpheus-150M model
        processor: Text processor
        text: Input text
        speed: Speed factor (0.5 = slower, 2.0 = faster)
        sample_rate: Output sample rate

    Returns:
        Audio waveform

    Example:
        >>> # Slower speech
        >>> audio_slow = synthesize_with_speed(model, processor, "Hello", speed=0.75)
        >>>
        >>> # Faster speech
        >>> audio_fast = synthesize_with_speed(model, processor, "Hello", speed=1.5)
    """
    # Process text
    input_ids = processor(text, padding=False)
    input_ids = mx.expand_dims(input_ids, axis=0)

    # Generate with speed modification
    # Modify duration predictions
    encoder_output = model.text_encoder(input_ids)
    durations = model.duration_predictor(encoder_output)

    # Scale durations by speed factor
    durations = durations / speed

    # Generate audio with modified durations
    waveform, mel, _ = model(input_ids, durations=durations)

    # Convert to numpy
    audio = np.array(waveform[0])

    # Normalize
    if audio.max() > 0:
        audio = audio / np.abs(audio).max()

    return audio


def get_mel_spectrogram(
    model: Orpheus150M,
    processor: TextProcessor,
    text: str,
) -> np.ndarray:
    """
    Get mel-spectrogram without vocoding.

    Useful for visualization or external vocoder.

    Args:
        model: Orpheus-150M model
        processor: Text processor
        text: Input text

    Returns:
        Mel-spectrogram as numpy array
        Shape: (time, num_mels)

    Example:
        >>> mel = get_mel_spectrogram(model, processor, "Hello")
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(mel.T, aspect='auto', origin='lower')
        >>> plt.show()
    """
    # Process text
    input_ids = processor(text, padding=False)
    input_ids = mx.expand_dims(input_ids, axis=0)

    # Generate mel-spectrogram
    waveform, mel, durations = model(input_ids)

    # Remove batch dimension
    mel = mel[0]

    # Convert to numpy
    mel_np = np.array(mel)

    return mel_np


def save_audio(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 24000,
):
    """
    Save audio to file.

    Args:
        audio: Audio waveform
        output_path: Output file path (.wav)
        sample_rate: Sample rate

    Example:
        >>> audio = synthesize(model, processor, "Hello")
        >>> save_audio(audio, "output.wav", sample_rate=24000)
    """
    try:
        import soundfile as sf

        sf.write(output_path, audio, sample_rate)
        print(f"✓ Audio saved to {output_path}")
    except ImportError:
        print("soundfile not installed. Install with: pip install soundfile")
        print("Falling back to scipy...")

        try:
            from scipy.io import wavfile

            # scipy expects int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, audio_int16)
            print(f"✓ Audio saved to {output_path}")
        except ImportError:
            print("Neither soundfile nor scipy available")
            print("Install soundfile: pip install soundfile")
