#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Voice Activity Detection interface.

Provides high-level API for detecting speech in audio using Silero VAD.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
import numpy as np

from .audio import load_audio, split_audio_chunks
from .config import VADConfig
from .model import SileroVAD, StreamingVAD


@dataclass
class SpeechSegment:
    """A detected speech segment.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        confidence: Average speech probability
    """

    start: float
    end: float
    confidence: float

    @property
    def duration(self) -> float:
        """Duration of segment in seconds."""
        return self.end - self.start

    def __repr__(self) -> str:
        return f"SpeechSegment(start={self.start:.2f}s, end={self.end:.2f}s, confidence={self.confidence:.3f})"


def detect_speech(
    model: SileroVAD,
    audio: Union[str, Path, np.ndarray, mx.array],
    threshold: Optional[float] = None,
    return_timestamps: bool = True,
) -> Union[bool, List[SpeechSegment]]:
    """Detect speech in audio.

    Args:
        model: Silero VAD model
        audio: Audio file path or waveform
        threshold: Speech detection threshold (uses model config if None)
        return_timestamps: Return speech segments with timestamps

    Returns:
        If return_timestamps=False: Boolean indicating speech detected
        If return_timestamps=True: List of SpeechSegment objects

    Example:
        >>> from smlx.models.SileroVAD import load, detect_speech
        >>> vad = load()
        >>> segments = detect_speech(vad, "audio.wav")
        >>> for seg in segments:
        ...     print(f"Speech from {seg.start:.1f}s to {seg.end:.1f}s")
    """
    # Load audio
    audio_array = load_audio(audio, model.config.sample_rate)

    # Get threshold
    if threshold is None:
        threshold = model.config.threshold

    # Predict speech probabilities
    model.reset_state()
    probs = model.predict(audio_array, reset_state=False)

    if not return_timestamps:
        # Just return if speech detected
        return mx.max(probs).item() >= threshold

    # Extract speech segments
    segments = extract_speech_segments(
        probs,
        sample_rate=model.config.sample_rate,
        threshold=threshold,
        neg_threshold=model.config.neg_threshold,
        min_speech_duration_ms=model.config.min_speech_duration_ms,
        min_silence_duration_ms=model.config.min_silence_duration_ms,
        speech_pad_ms=model.config.speech_pad_ms,
        window_size_samples=model.config.context_size,
    )

    return segments


def extract_speech_segments(
    speech_probs: mx.array,
    sample_rate: int,
    threshold: float = 0.5,
    neg_threshold: Optional[float] = None,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    window_size_samples: int = 64,
) -> List[SpeechSegment]:
    """Extract speech segments from speech probabilities.

    Args:
        speech_probs: Speech probability sequence
        sample_rate: Audio sample rate
        threshold: Positive threshold for speech detection
        neg_threshold: Negative threshold (for hysteresis)
        min_speech_duration_ms: Minimum speech segment duration
        min_silence_duration_ms: Minimum silence to split segments
        speech_pad_ms: Padding around speech segments
        window_size_samples: Window size in samples (default: 64, Silero VAD context_size)

    Returns:
        List of detected speech segments
    """
    if neg_threshold is None:
        neg_threshold = threshold - 0.15

    # Convert to numpy for easier processing
    probs = np.array(speech_probs)

    # Apply thresholds
    triggered = False
    speeches = []
    current_speech = {}

    for i, prob in enumerate(probs):
        if not triggered:
            # Looking for speech start
            if prob >= threshold:
                triggered = True
                current_speech["start"] = i
                current_speech["probs"] = [prob]
        else:
            # In speech segment
            current_speech["probs"].append(prob)

            if prob < neg_threshold:
                # End of speech
                triggered = False
                current_speech["end"] = i
                speeches.append(current_speech.copy())
                current_speech = {}

    # Handle case where speech continues to end
    if triggered:
        current_speech["end"] = len(probs)
        speeches.append(current_speech)

    # Convert indices to timestamps
    # Each probability corresponds to one window of audio
    # window_duration_ms = (window_size_samples / sample_rate) * 1000
    window_duration_ms = (window_size_samples / sample_rate) * 1000.0

    segments = []
    for speech in speeches:
        start_ms = speech["start"] * window_duration_ms
        end_ms = speech["end"] * window_duration_ms
        duration_ms = end_ms - start_ms

        # Filter by minimum duration
        if duration_ms >= min_speech_duration_ms:
            # Add padding
            start_ms = max(0, start_ms - speech_pad_ms)
            end_ms = end_ms + speech_pad_ms

            # Calculate confidence
            confidence = float(np.mean(speech["probs"]))

            segments.append(
                SpeechSegment(
                    start=start_ms / 1000.0,
                    end=end_ms / 1000.0,
                    confidence=confidence,
                )
            )

    # Merge close segments
    if len(segments) > 1:
        segments = merge_segments(segments, min_silence_duration_ms / 1000.0)

    return segments


def merge_segments(
    segments: List[SpeechSegment],
    min_gap: float,
) -> List[SpeechSegment]:
    """Merge speech segments that are close together.

    Args:
        segments: List of speech segments
        min_gap: Minimum gap between segments (in seconds)

    Returns:
        List of merged segments
    """
    if not segments:
        return segments

    merged = [segments[0]]

    for current in segments[1:]:
        previous = merged[-1]

        # Check if segments should be merged
        gap = current.start - previous.end

        if gap < min_gap:
            # Merge segments
            merged[-1] = SpeechSegment(
                start=previous.start,
                end=current.end,
                confidence=(previous.confidence + current.confidence) / 2,
            )
        else:
            merged.append(current)

    return merged


def filter_audio_by_speech(
    audio: Union[str, Path, np.ndarray, mx.array],
    segments: List[SpeechSegment],
    sample_rate: int = 16000,
) -> mx.array:
    """Extract only speech portions from audio.

    Args:
        audio: Audio file path or waveform
        segments: Speech segments to extract
        sample_rate: Audio sample rate

    Returns:
        Concatenated speech audio
    """
    # Load audio
    audio_array = load_audio(audio, sample_rate)

    # Extract speech segments
    speech_chunks = []

    for segment in segments:
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)

        # Clip to audio bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_array), end_sample)

        if start_sample < end_sample:
            chunk = audio_array[start_sample:end_sample]
            speech_chunks.append(chunk)

    # Concatenate all speech segments
    if speech_chunks:
        speech_audio = mx.concatenate(speech_chunks)
    else:
        speech_audio = mx.array([])

    return speech_audio


def create_streaming_vad(model: SileroVAD) -> StreamingVAD:
    """Create streaming VAD processor.

    Args:
        model: Silero VAD model

    Returns:
        StreamingVAD instance

    Example:
        >>> vad = load()
        >>> streaming = create_streaming_vad(vad)
        >>> # Process audio chunks
        >>> for chunk in audio_stream:
        ...     probs = streaming.process_chunk(chunk)
        ...     if probs and probs[0] > 0.5:
        ...         print("Speech detected!")
    """
    return StreamingVAD(model, model.config)


__all__ = [
    "SpeechSegment",
    "detect_speech",
    "extract_speech_segments",
    "merge_segments",
    "filter_audio_by_speech",
    "create_streaming_vad",
]
