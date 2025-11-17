"""
SMLX Data Module - Centralized data handling for all modalities.

This module provides comprehensive data loading, preprocessing, batching,
and augmentation utilities for text, images, audio, and multimodal data.

Quick Start:
    >>> from smlx.data import load_image, load_audio, TextDataset, DataLoader
    >>>
    >>> # Load data
    >>> image = load_image("photo.jpg")
    >>> audio = load_audio("speech.wav", sr=16000)
    >>>
    >>> # Create dataset
    >>> dataset = TextDataset(data, tokenizer)
    >>>
    >>> # Batch processing
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for batch in dataloader:
    ...     outputs = model(batch)

Modules:
    - loaders: Load images, audio, text, video from various sources
    - datasets: Dataset classes for different tasks (text, chat, VLM, audio)
    - preprocessing: Preprocessors for images, audio, text
    - batch: DataLoader and collation utilities
    - hf: HuggingFace datasets integration
    - augmentation: Data augmentation transforms

Adapted from MLX-LM, MLX-VLM, and other MLX ecosystem patterns.
"""

# Loaders
from .loaders import (
    load_audio,
    load_image,
    load_text,
    load_video,
    resample_audio,
)

# Datasets
from .datasets import (
    AudioDataset,
    CacheDataset,
    ChatDataset,
    CompletionsDataset,
    ConcatenatedDataset,
    SubsetDataset,
    TextDataset,
    VisionLanguageDataset,
)

# Preprocessing
from .preprocessing import (
    AudioPreprocessor,
    ImagePreprocessor,
    MultimodalPreprocessor,
    TextPreprocessor,
)

# Batching
from .batch import (
    DataLoader,
    batch_images,
    collate_audio,
    collate_images,
    collate_text,
    collate_vlm,
    create_batches,
    dynamic_batching,
    pad_sequences,
)

# HuggingFace integration
from .hf import (
    create_dataset,
    download_from_hub,
    load_dataset_splits,
    load_hf_dataset,
    load_local_dataset,
    save_dataset_to_jsonl,
)

# Augmentation
from .augmentation import (
    AudioAugmentation,
    Compose,
    ImageAugmentation,
    RandomApply,
    RandomChoice,
)

__all__ = [
    # Loaders
    "load_image",
    "load_audio",
    "load_text",
    "load_video",
    "resample_audio",
    # Datasets
    "TextDataset",
    "ChatDataset",
    "CompletionsDataset",
    "VisionLanguageDataset",
    "AudioDataset",
    "ConcatenatedDataset",
    "CacheDataset",
    "SubsetDataset",
    # Preprocessing
    "ImagePreprocessor",
    "AudioPreprocessor",
    "TextPreprocessor",
    "MultimodalPreprocessor",
    # Batching
    "DataLoader",
    "collate_text",
    "collate_images",
    "collate_audio",
    "collate_vlm",
    "pad_sequences",
    "batch_images",
    "create_batches",
    "dynamic_batching",
    # HuggingFace
    "load_hf_dataset",
    "load_local_dataset",
    "load_dataset_splits",
    "create_dataset",
    "download_from_hub",
    "save_dataset_to_jsonl",
    # Augmentation
    "ImageAugmentation",
    "AudioAugmentation",
    "Compose",
    "RandomApply",
    "RandomChoice",
]
