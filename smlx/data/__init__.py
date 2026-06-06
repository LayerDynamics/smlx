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
# Local bundled-dataset access (repo-root data/ tree).
# The generic verbs (load/get/registry) are intentionally accessed via the
# submodule -- ``from smlx.data import local; local.load("coco8")`` -- to avoid
# polluting the package namespace; the well-named symbols are re-exported here.
from . import local

# Augmentation
from .augmentation import (
    AudioAugmentation,
    Compose,
    ImageAugmentation,
    RandomApply,
    RandomChoice,
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

# HuggingFace integration
from .hf import (
    create_dataset,
    download_from_hub,
    load_dataset_splits,
    load_hf_dataset,
    load_local_dataset,
    save_dataset_to_jsonl,
)
from .loaders import (
    load_audio,
    load_image,
    load_text,
    load_video,
    resample_audio,
)
from .local import (
    DatasetEntry,
    ImageTree,
    JsonIndex,
    Layout,
    ProbeResult,
    available_splits,
    detect_layout,
    find_orphans,
    inventory,
    is_available,
    iter_samples,
    local_path,
    probe,
)

# Preprocessing
from .preprocessing import (
    AudioPreprocessor,
    ImagePreprocessor,
    MultimodalPreprocessor,
    TextPreprocessor,
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
    # Local bundled-dataset access
    "local",
    "DatasetEntry",
    "ImageTree",
    "JsonIndex",
    "Layout",
    "ProbeResult",
    "available_splits",
    "detect_layout",
    "find_orphans",
    "inventory",
    "is_available",
    "iter_samples",
    "local_path",
    "probe",
]
