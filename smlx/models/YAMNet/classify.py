#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
YAMNet Classification Interface.

High-level API for audio event classification using YAMNet.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

import mlx.core as mx
import numpy as np

from .audio import preprocess_audio, postprocess_predictions
from .config import YAMNetConfig, AUDIOSET_CLASSES
from .loader import load_class_names
from .model import YAMNet


@dataclass
class Prediction:
    """Audio event prediction."""

    label: str
    score: float
    class_id: int


def classify(
    model: YAMNet,
    audio: Union[str, Path, np.ndarray, mx.array],
    sample_rate: Optional[int] = None,
    top_k: int = 5,
    aggregate: str = "max",
) -> list[Prediction]:
    """Classify audio events.

    Args:
        model: YAMNet model
        audio: Audio source (file path, numpy array, or MLX array)
        sample_rate: Sample rate of audio (if not a file)
        top_k: Number of top predictions to return
        aggregate: How to aggregate predictions across patches ("max", "mean", "median")

    Returns:
        List of top-k predictions

    Example:
        >>> from smlx.models.YAMNet import load, classify
        >>> model = load()
        >>> predictions = classify(model, "audio.wav", top_k=3)
        >>> for pred in predictions:
        ...     print(f"{pred.label}: {pred.score:.3f}")
    """
    # Preprocess audio to patches
    patches = preprocess_audio(audio, sample_rate=sample_rate, config=model.config)

    # Add channel dimension: (num_patches, height, width, 1)
    patches = mx.expand_dims(patches, axis=-1)

    # Get predictions for each patch
    logits = model(patches)
    probs = mx.softmax(logits, axis=-1)

    # Aggregate across patches
    aggregated_probs = postprocess_predictions(probs, aggregate=aggregate)

    # Get top-k predictions
    class_names = load_class_names()
    predictions = get_top_k_predictions(aggregated_probs, class_names, top_k)

    return predictions


def classify_batch(
    model: YAMNet,
    audio_list: list[Union[str, Path, np.ndarray, mx.array]],
    sample_rate: Optional[int] = None,
    top_k: int = 5,
    aggregate: str = "max",
) -> list[list[Prediction]]:
    """Classify multiple audio files in batch.

    Args:
        model: YAMNet model
        audio_list: List of audio sources
        sample_rate: Sample rate of audio (if not a file)
        top_k: Number of top predictions per audio
        aggregate: Aggregation method

    Returns:
        List of predictions for each audio

    Example:
        >>> predictions_batch = classify_batch(model, ["audio1.wav", "audio2.wav"])
        >>> for i, predictions in enumerate(predictions_batch):
        ...     print(f"Audio {i}:")
        ...     for pred in predictions:
        ...         print(f"  {pred.label}: {pred.score:.3f}")
    """
    results = []
    for audio in audio_list:
        predictions = classify(model, audio, sample_rate=sample_rate, top_k=top_k, aggregate=aggregate)
        results.append(predictions)
    return results


def extract_embeddings(
    model: YAMNet,
    audio: Union[str, Path, np.ndarray, mx.array],
    sample_rate: Optional[int] = None,
) -> mx.array:
    """Extract audio embeddings from YAMNet.

    YAMNet can be used as a feature extractor, producing 1,024-dimensional
    embeddings for each audio patch. These can be used for:
    - Audio similarity search
    - Clustering
    - Downstream tasks

    Args:
        model: YAMNet model
        audio: Audio source
        sample_rate: Sample rate of audio (if not a file)

    Returns:
        Embeddings (num_patches, embedding_size)

    Example:
        >>> embeddings = extract_embeddings(model, "audio.wav")
        >>> print(f"Shape: {embeddings.shape}")  # (num_patches, 1024)
        >>> # Average over time for a single vector per audio
        >>> audio_embedding = embeddings.mean(axis=0)
    """
    # Preprocess audio to patches
    patches = preprocess_audio(audio, sample_rate=sample_rate, config=model.config)

    # Add channel dimension
    patches = mx.expand_dims(patches, axis=-1)

    # Extract embeddings
    embeddings = model.extract_embeddings(patches)

    return embeddings


def get_top_k_predictions(
    probs: mx.array,
    class_names: list[str],
    top_k: int = 5,
) -> list[Prediction]:
    """Get top-k predictions from probabilities.

    Args:
        probs: Class probabilities (num_classes,)
        class_names: List of class names
        top_k: Number of top predictions

    Returns:
        List of top-k predictions
    """
    # Get top-k indices
    top_k_indices = mx.argsort(probs)[-top_k:][::-1]
    top_k_indices = top_k_indices.tolist()

    predictions = []
    for idx in top_k_indices:
        pred = Prediction(
            label=class_names[idx] if idx < len(class_names) else f"Class_{idx}",
            score=float(probs[idx]),
            class_id=idx,
        )
        predictions.append(pred)

    return predictions


def detect_events(
    model: YAMNet,
    audio: Union[str, Path, np.ndarray, mx.array],
    sample_rate: Optional[int] = None,
    event_classes: Optional[list[str]] = None,
    threshold: float = 0.3,
    top_k: int = 20,
) -> list[Prediction]:
    """Detect specific audio events in recording.

    Useful for finding specific sounds (e.g., "dog bark", "music", "speech")
    in longer audio files.

    Args:
        model: YAMNet model
        audio: Audio source
        sample_rate: Sample rate of audio (if not a file)
        event_classes: List of event keywords to look for (if None, returns all above threshold)
        threshold: Minimum confidence threshold
        top_k: Number of predictions to consider

    Returns:
        Detected events matching criteria

    Example:
        >>> # Detect speech or music
        >>> events = detect_events(model, "podcast.wav", event_classes=["speech", "music"], threshold=0.5)
        >>> for event in events:
        ...     print(f"{event.label}: {event.score:.2f}")
    """
    # Get predictions
    predictions = classify(model, audio, sample_rate=sample_rate, top_k=top_k)

    # Filter by threshold and event classes
    detected = []
    for pred in predictions:
        if pred.score >= threshold:
            # If no event classes specified, include all above threshold
            if event_classes is None:
                detected.append(pred)
            else:
                # Check if any event keyword matches
                label_lower = pred.label.lower()
                if any(event.lower() in label_lower for event in event_classes):
                    detected.append(pred)

    return detected


def compute_audio_similarity(
    model: YAMNet,
    audio1: Union[str, Path, np.ndarray, mx.array],
    audio2: Union[str, Path, np.ndarray, mx.array],
    sample_rate: Optional[int] = None,
    metric: str = "cosine",
) -> float:
    """Compute similarity between two audio clips.

    Uses YAMNet embeddings to compute audio similarity.

    Args:
        model: YAMNet model
        audio1: First audio source
        audio2: Second audio source
        sample_rate: Sample rate of audio (if not a file)
        metric: Similarity metric ("cosine" or "euclidean")

    Returns:
        Similarity score (higher = more similar for cosine)

    Example:
        >>> similarity = compute_audio_similarity(model, "audio1.wav", "audio2.wav")
        >>> print(f"Similarity: {similarity:.3f}")
    """
    # Extract embeddings
    emb1 = extract_embeddings(model, audio1, sample_rate=sample_rate)
    emb2 = extract_embeddings(model, audio2, sample_rate=sample_rate)

    # Average over time
    emb1_avg = mx.mean(emb1, axis=0)
    emb2_avg = mx.mean(emb2, axis=0)

    if metric == "cosine":
        # Cosine similarity
        dot_product = mx.sum(emb1_avg * emb2_avg)
        norm1 = mx.sqrt(mx.sum(emb1_avg * emb1_avg))
        norm2 = mx.sqrt(mx.sum(emb2_avg * emb2_avg))
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    elif metric == "euclidean":
        # Negative euclidean distance (higher = more similar)
        distance = mx.sqrt(mx.sum((emb1_avg - emb2_avg) ** 2))
        return float(-distance)

    else:
        raise ValueError(f"Unknown metric: {metric}")


__all__ = [
    "classify",
    "classify_batch",
    "extract_embeddings",
    "detect_events",
    "compute_audio_similarity",
    "Prediction",
]
