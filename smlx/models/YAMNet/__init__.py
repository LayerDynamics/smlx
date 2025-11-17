#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
YAMNet - Ultra-Tiny Audio Event Classifier.

YAMNet is a 3.7M parameter MobileNet-v1 based audio classifier
trained on AudioSet with 521 event classes. It's designed for
efficient audio event detection and embedding extraction.

This implementation uses real pre-trained weights automatically downloaded
and converted from PyTorch (torch_audioset) on first load.

Quick Start:
    >>> from smlx.models.YAMNet import load, classify
    >>> import soundfile as sf
    >>>
    >>> # Load model (downloads weights on first run)
    >>> model = load()  # Requires: pip install torch
    >>>
    >>> # Load and classify audio
    >>> audio, sr = sf.read("sound.wav")
    >>> predictions = classify(model, audio, sample_rate=sr, top_k=5)
    >>>
    >>> # Show results
    >>> for pred in predictions:
    ...     print(f"{pred.label}: {pred.score:.3f}")

Features:
    - ✓ Real pre-trained weights (not random initialization)
    - ✓ 521 AudioSet event classes (speech, music, animals, etc.)
    - ✓ 1,024-dimensional audio embeddings
    - ✓ ~15MB model size
    - ✓ Fast inference on M4
    - ✓ Real-time capable
    - ✓ Automatic weight caching

Example Applications:
    - Audio event detection
    - Sound effect classification
    - Audio tagging and indexing
    - Audio similarity search
    - Smart home monitoring
    - Environmental sound analysis

Architecture:
    - MobileNet-v1 with depthwise-separable convolutions
    - Mel spectrogram input (96 frames x 64 bands)
    - Global average pooling
    - Dense classification head

Model Details:
    - Parameters: 3.7M
    - Input: Mel spectrograms (16kHz audio)
    - Output: 521 class probabilities
    - Embeddings: 1,024-dim features
    - License: Apache 2.0

Weight Sources:
    - Original: Google TensorFlow Hub (Apache 2.0)
    - PyTorch: w-hc/torch_audioset (auto-downloaded and converted)
    - MLX: Converted on first load, cached for reuse

Dependencies:
    - First-time load: PyTorch (pip install torch)
    - Audio loading: soundfile, librosa
    - Optional: huggingface_hub (for pre-converted MLX weights)

References:
    - Original TensorFlow: https://tfhub.dev/google/yamnet/1
    - PyTorch Source: https://github.com/w-hc/torch_audioset
    - AudioSet: https://research.google.com/audioset/
"""

from .audio import (
    compute_mel_spectrogram,
    extract_patches,
    load_audio,
    preprocess_audio,
)
from .classify import (
    Prediction,
    classify,
    classify_batch,
    compute_audio_similarity,
    detect_events,
    extract_embeddings,
)
from .config import AUDIOSET_CLASSES, YAMNetConfig
from .loader import load, load_class_names, save_model
from .model import YAMNet

__version__ = "0.1.0"

__all__ = [
    # Model
    "YAMNet",
    "YAMNetConfig",
    # Loading
    "load",
    "save_model",
    "load_class_names",
    # Classification
    "classify",
    "classify_batch",
    "extract_embeddings",
    "detect_events",
    "compute_audio_similarity",
    "Prediction",
    # Audio processing
    "load_audio",
    "preprocess_audio",
    "compute_mel_spectrogram",
    "extract_patches",
    # Constants
    "AUDIOSET_CLASSES",
]
