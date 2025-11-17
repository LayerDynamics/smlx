#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Moondream2: Production-Ready Vision-Language Model

A well-established 1.8B parameter multimodal model combining:
- Custom vision encoder with crop-based tiling
- Phi-based language model (2048 hidden size, 24 layers)
- Region modules for object detection and pointing

Capabilities:
- Image captioning
- Visual question answering (VQA)
- Object detection with bounding boxes
- Spatial localization (pointing)
- Multi-turn conversations

Model Variants:
- moondream2 (2B): Full model with best performance
- moondream-0_5b (0.5B): Smaller variant for efficiency

Example:
    >>> from smlx.models.Moondream2 import load, caption, query
    >>> from PIL import Image
    >>>
    >>> # Load model
    >>> model, tokenizer = load("vikhyatk/moondream2")
    >>>
    >>> # Caption an image
    >>> image = Image.open("photo.jpg")
    >>> description = caption(model, tokenizer, image)
    >>> print(description)
    >>>
    >>> # Answer questions
    >>> answer = query(model, tokenizer, image, "What objects are visible?")
    >>> print(answer)
"""

from .config import (
    ModelConfig,
    VisionConfig,
    TextConfig,
    RegionConfig,
    DEFAULT_CONFIG_2B,
    DEFAULT_CONFIG_05B,
)
from .model import Moondream2, VisionProjector
from .vision import VisionEncoder, prepare_crops, reconstruct_from_crops
from .language import PhiModel, PhiAttention, PhiMLP
from .region import (
    DetectionHead,
    CoordinateEncoder,
    CoordinateDecoder,
    BoxEncoder,
    BoxDecoder,
    parse_coordinates_from_text,
    parse_boxes_from_text,
)
from .loader import load, get_model_path
from .generate import (
    generate,
    stream_generate,
    caption,
    query,
    detect,
    point,
    preprocess_image,
)
from .cache import (
    KVCache,
    RotatingKVCache,
    make_cache,
    make_kv_caches,
    make_cache_with_monitoring,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "ModelConfig",
    "VisionConfig",
    "TextConfig",
    "RegionConfig",
    "DEFAULT_CONFIG_2B",
    "DEFAULT_CONFIG_05B",
    # Model
    "Moondream2",
    "VisionProjector",
    # Vision
    "VisionEncoder",
    "prepare_crops",
    "reconstruct_from_crops",
    # Language
    "PhiModel",
    "PhiAttention",
    "PhiMLP",
    # Region modules
    "DetectionHead",
    "CoordinateEncoder",
    "CoordinateDecoder",
    "BoxEncoder",
    "BoxDecoder",
    "parse_coordinates_from_text",
    "parse_boxes_from_text",
    # Loading
    "load",
    "get_model_path",
    # Generation
    "generate",
    "stream_generate",
    "caption",
    "query",
    "detect",
    "point",
    "preprocess_image",
    # Cache
    "KVCache",
    "RotatingKVCache",
    "make_cache",
    "make_kv_caches",
    "make_cache_with_monitoring",
]
