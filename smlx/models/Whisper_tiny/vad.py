"""
Voice Activity Detection (VAD) integration for Whisper.

Provides utilities to detect speech segments in audio before transcription,
reducing processing time and improving accuracy by skipping silent regions.

This module supports integration with external VAD models like:
- Silero VAD (recommended)
- WebRTC VAD
- Custom VAD models

Usage:
    from smlx.models.Whisper_tiny import load, transcribe
    from smlx.models.Whisper_tiny.vad import detect_speech_segments, transcribe_with_vad

    model, tokenizer = load()

    # Detect speech segments first
    segments = detect_speech_segments("audio.wav", threshold=0.5)
    print(f"Found {len(segments)} speech segments")

    # Transcribe with VAD pre-segmentation
    result = transcribe_with_vad("audio.wav", model, tokenizer, vad_threshold=0.5)

Example with Silero VAD:
    # Install: pip install silero-vad
    from smlx.models.Whisper_tiny.vad import SileroVAD

    vad = SileroVAD()
    segments = vad.detect_segments("audio.wav")
    for seg in segments:
        print(f"Speech from {seg['start']:.2f}s to {seg['end']:.2f}s")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .audio import SAMPLE_RATE, load_audio


@dataclass
class SpeechSegment:
    """Speech segment detected by VAD.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        confidence: VAD confidence score (0-1)
    """

    start: float
    end: float
    confidence: float = 1.0

    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


def merge_segments(
    segments: List[SpeechSegment],
    *,
    min_gap: float = 0.3,
    max_duration: float = 30.0,
) -> List[SpeechSegment]:
    """Merge nearby speech segments.

    Combines segments that are close together to reduce fragmentation,
    while respecting maximum duration constraints.

    Args:
        segments: List of speech segments
        min_gap: Minimum gap between segments to keep separate (seconds)
        max_duration: Maximum duration of merged segment (seconds)

    Returns:
        List of merged segments

    Example:
        >>> seg1 = SpeechSegment(0.0, 1.0, 0.9)
        >>> seg2 = SpeechSegment(1.2, 2.5, 0.95)  # 0.2s gap
        >>> merged = merge_segments([seg1, seg2], min_gap=0.3)
        >>> len(merged)
        1  # Merged into single segment
    """
    if not segments:
        return []

    # Sort by start time
    segments = sorted(segments, key=lambda s: s.start)

    merged = []
    current = segments[0]

    for next_seg in segments[1:]:
        gap = next_seg.start - current.end
        merged_duration = next_seg.end - current.start

        # Merge if gap is small and duration constraint satisfied
        if gap < min_gap and merged_duration <= max_duration:
            # Extend current segment
            current = SpeechSegment(
                start=current.start,
                end=next_seg.end,
                confidence=min(current.confidence, next_seg.confidence),
            )
        else:
            # Start new segment
            merged.append(current)
            current = next_seg

    # Add last segment
    merged.append(current)

    return merged


def filter_segments(
    segments: List[SpeechSegment],
    *,
    min_duration: float = 0.1,
    min_confidence: float = 0.5,
) -> List[SpeechSegment]:
    """Filter out short or low-confidence segments.

    Args:
        segments: List of speech segments
        min_duration: Minimum segment duration (seconds)
        min_confidence: Minimum VAD confidence score (0-1)

    Returns:
        Filtered list of segments

    Example:
        >>> segments = [
        ...     SpeechSegment(0.0, 0.05, 0.3),  # Too short and low confidence
        ...     SpeechSegment(1.0, 2.0, 0.9),   # Good segment
        ... ]
        >>> filtered = filter_segments(segments, min_duration=0.1, min_confidence=0.5)
        >>> len(filtered)
        1
    """
    return [
        seg
        for seg in segments
        if seg.duration() >= min_duration and seg.confidence >= min_confidence
    ]


class SileroVAD:
    """Silero VAD model for speech detection.

    A pre-trained voice activity detection model that can detect speech
    segments in audio with high accuracy.

    Requires silero-vad package:
        pip install silero-vad

    Attributes:
        model: Silero VAD model
        sampling_rate: Expected sampling rate (8000 or 16000)
        threshold: Detection threshold (0-1)
    """

    def __init__(
        self,
        *,
        sampling_rate: int = 16000,
        threshold: float = 0.5,
    ):
        """Initialize Silero VAD.

        Args:
            sampling_rate: Audio sampling rate (8000 or 16000)
            threshold: Speech detection threshold (0-1, higher = more conservative)

        Raises:
            ImportError: If silero-vad is not installed
        """
        try:
            import torch
            from silero_vad import load_silero_vad, get_speech_timestamps
        except ImportError:
            raise ImportError(
                "silero-vad is required for SileroVAD. Install with: pip install silero-vad"
            )

        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.model = load_silero_vad()
        self.get_speech_timestamps = get_speech_timestamps

    def detect_segments(
        self,
        audio: Union[str, np.ndarray, mx.array],
        *,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
        min_speech_duration_ms: int = 100,
    ) -> List[SpeechSegment]:
        """Detect speech segments in audio.

        Args:
            audio: Audio file path, numpy array, or MLX array
            min_silence_duration_ms: Minimum silence duration to split segments (ms)
            speech_pad_ms: Padding to add to detected segments (ms)
            min_speech_duration_ms: Minimum speech segment duration (ms)

        Returns:
            List of detected speech segments

        Example:
            >>> vad = SileroVAD()
            >>> segments = vad.detect_segments("speech.wav")
            >>> for seg in segments:
            ...     print(f"Speech: {seg.start:.2f}s - {seg.end:.2f}s")
        """
        import torch

        # Load audio
        if isinstance(audio, (str, Path)):
            audio = load_audio(audio)

        # Convert to numpy if MLX array
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Resample if needed
        if self.sampling_rate != SAMPLE_RATE:
            from scipy import signal
            num_samples = int(len(audio) * self.sampling_rate / SAMPLE_RATE)
            audio = signal.resample(audio, num_samples)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio.astype(np.float32))

        # Detect speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            min_speech_duration_ms=min_speech_duration_ms,
        )

        # Convert to SpeechSegment objects
        segments = []
        for ts in speech_timestamps:
            start_sec = ts["start"] / self.sampling_rate
            end_sec = ts["end"] / self.sampling_rate
            segments.append(SpeechSegment(start_sec, end_sec, confidence=1.0))

        return segments


def detect_speech_segments(
    audio: Union[str, np.ndarray, mx.array],
    *,
    vad_model: Optional[SileroVAD] = None,
    threshold: float = 0.5,
    min_gap: float = 0.3,
    max_duration: float = 30.0,
    min_duration: float = 0.1,
) -> List[SpeechSegment]:
    """Detect and merge speech segments in audio.

    Convenience function that combines VAD detection, merging, and filtering.

    Args:
        audio: Audio file path, numpy array, or MLX array
        vad_model: VAD model to use (SileroVAD created if None)
        threshold: VAD threshold (0-1)
        min_gap: Minimum gap between segments (seconds)
        max_duration: Maximum merged segment duration (seconds)
        min_duration: Minimum segment duration (seconds)

    Returns:
        List of speech segments

    Example:
        >>> segments = detect_speech_segments("audio.wav", threshold=0.5)
        >>> print(f"Found {len(segments)} speech segments")
    """
    # Create VAD model if not provided
    if vad_model is None:
        vad_model = SileroVAD(threshold=threshold)

    # Detect segments
    segments = vad_model.detect_segments(audio)

    # Merge nearby segments
    segments = merge_segments(segments, min_gap=min_gap, max_duration=max_duration)

    # Filter short segments
    segments = filter_segments(segments, min_duration=min_duration, min_confidence=threshold)

    return segments


def transcribe_with_vad(
    audio: Union[str, np.ndarray, mx.array],
    model,
    tokenizer,
    *,
    vad_threshold: float = 0.5,
    min_gap: float = 0.3,
    max_duration: float = 30.0,
    min_duration: float = 0.1,
    verbose: bool = True,
    **transcribe_kwargs,
) -> Dict:
    """Transcribe audio with VAD pre-segmentation.

    Uses VAD to detect speech segments first, then transcribes each segment
    separately. This is more efficient than transcribing the entire audio,
    especially for files with long silent periods.

    Args:
        audio: Audio file path, numpy array, or MLX array
        model: Whisper model
        tokenizer: Whisper tokenizer
        vad_threshold: VAD detection threshold (0-1)
        min_gap: Minimum gap between segments (seconds)
        max_duration: Maximum segment duration (seconds)
        min_duration: Minimum segment duration (seconds)
        verbose: Print progress
        **transcribe_kwargs: Additional arguments for transcribe()

    Returns:
        Dictionary with:
        - text: Full transcription
        - segments: List of segments with timestamps
        - language: Detected language
        - vad_segments: List of VAD segments

    Example:
        >>> from smlx.models.Whisper_tiny import load
        >>> from smlx.models.Whisper_tiny.vad import transcribe_with_vad
        >>>
        >>> model, tokenizer = load()
        >>> result = transcribe_with_vad("audio.wav", model, tokenizer, vad_threshold=0.5)
        >>> print(result["text"])
    """
    from .transcribe import transcribe

    # Load audio
    if isinstance(audio, (str, Path)):
        audio_path = audio
        audio_array = load_audio(audio)
    else:
        audio_path = None
        audio_array = audio

    # Detect speech segments
    if verbose:
        print("Detecting speech segments...")

    vad_segments = detect_speech_segments(
        audio_array,
        threshold=vad_threshold,
        min_gap=min_gap,
        max_duration=max_duration,
        min_duration=min_duration,
    )

    if verbose:
        print(f"Found {len(vad_segments)} speech segments")

    # If no segments found, return empty result
    if not vad_segments:
        return {
            "text": "",
            "segments": [],
            "language": transcribe_kwargs.get("language", "en"),
            "vad_segments": [],
        }

    # Transcribe each VAD segment
    all_segments = []
    all_text = []

    for i, vad_seg in enumerate(vad_segments):
        if verbose:
            print(
                f"Transcribing segment {i + 1}/{len(vad_segments)} "
                f"({vad_seg.start:.2f}s - {vad_seg.end:.2f}s)"
            )

        # Extract audio for this segment
        start_sample = int(vad_seg.start * SAMPLE_RATE)
        end_sample = int(vad_seg.end * SAMPLE_RATE)

        # Convert to numpy if needed
        if isinstance(audio_array, mx.array):
            segment_audio = np.array(audio_array[start_sample:end_sample])
        else:
            segment_audio = audio_array[start_sample:end_sample]

        # Transcribe segment
        result = transcribe(
            segment_audio,
            model,
            tokenizer,
            verbose=False,
            **transcribe_kwargs,
        )

        # Adjust timestamps to absolute positions
        for seg in result["segments"]:
            seg["start"] += vad_seg.start
            seg["end"] += vad_seg.start
            all_segments.append(seg)

        all_text.append(result["text"])

    # Combine results
    return {
        "text": " ".join(all_text).strip(),
        "segments": all_segments,
        "language": result["language"] if all_segments else transcribe_kwargs.get("language", "en"),
        "vad_segments": [seg.to_dict() for seg in vad_segments],
    }
