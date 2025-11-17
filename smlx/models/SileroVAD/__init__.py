#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Silero VAD - Voice Activity Detection.

Lightweight LSTM-based model for detecting speech in audio streams.
Optimized for real-time applications with ~1MB model size.

Features:
- Fast voice activity detection
- Streaming support
- Speech segmentation
- Low memory footprint
- Works with 8kHz and 16kHz audio

Example:
    >>> from smlx.models.SileroVAD import load, detect_speech
    >>>
    >>> # Load model
    >>> vad = load(sample_rate=16000)
    >>>
    >>> # Detect speech in audio file
    >>> segments = detect_speech(vad, "audio.wav")
    >>> for seg in segments:
    ...     print(f"Speech: {seg.start:.1f}s - {seg.end:.1f}s")

Example (Streaming):
    >>> from smlx.models.SileroVAD import load, create_streaming_vad
    >>>
    >>> vad = load()
    >>> streaming = create_streaming_vad(vad)
    >>>
    >>> # Process audio chunks
    >>> for chunk in audio_stream:
    ...     probs = streaming.process_chunk(chunk)
    ...     if probs and probs[0] > 0.5:
    ...         print("Speech detected!")

Model Details:
    Architecture: LSTM-based
    Model Size: ~1MB
    Sample Rates: 8kHz, 16kHz
    Input: Raw audio waveform
    Output: Speech probability (0-1)

Use Cases:
    - Voice activity detection
    - Speech/non-speech segmentation
    - Preprocessing for speech recognition
    - Real-time voice detection
    - Audio compression (remove silence)

Performance:
    - Very fast inference (~1ms per chunk on M4)
    - Low memory usage
    - Suitable for real-time streaming
    - Works well on mobile/edge devices

Reference:
    Silero VAD: https://github.com/snakers4/silero-vad
    Paper: "Silero VAD: pre-trained enterprise-grade Voice Activity Detector"
"""

from .audio import (
    load_audio,
    load_audio_file,
    normalize_audio,
    resample_audio,
    split_audio_chunks,
)
from .config import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_16K,
    DEFAULT_CONFIG_8K,
    VADConfig,
)
from .loader import load, get_model_path
from .model import SileroVAD, StreamingVAD
from .vad import (
    SpeechSegment,
    create_streaming_vad,
    detect_speech,
    extract_speech_segments,
    filter_audio_by_speech,
)

__version__ = "0.1.0"

__all__ = [
    # Main API
    "load",
    "detect_speech",
    "create_streaming_vad",
    # Model
    "SileroVAD",
    "StreamingVAD",
    # Configuration
    "VADConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_8K",
    "DEFAULT_CONFIG_16K",
    # VAD utilities
    "SpeechSegment",
    "extract_speech_segments",
    "filter_audio_by_speech",
    # Audio utilities
    "load_audio",
    "load_audio_file",
    "resample_audio",
    "normalize_audio",
    "split_audio_chunks",
    # Loader
    "get_model_path",
]
