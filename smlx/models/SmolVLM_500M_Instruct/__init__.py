#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
SmolVLM-500M-Instruct: Vision-Language Model

A small multimodal model combining:
- SigLIP 93M vision encoder (768 hidden size, 12 layers)
- SmolLM2-360M language model (960 hidden size, 32 layers)
- Idefics3 connector with pixel shuffle

Total parameters: ~500M

Example:
    >>> from smlx.models.SmolVLM_500M_Instruct import load, generate
    >>>
    >>> # Load model and processor
    >>> model, processor = load("HuggingFaceTB/SmolVLM-500M-Instruct")
    >>>
    >>> # Generate with image
    >>> output = generate(
    ...     model=model,
    ...     processor=processor,
    ...     prompt="Describe this image:",
    ...     image="https://example.com/photo.jpg",
    ...     max_tokens=100
    ... )
    >>> print(output)
    >>>
    >>> # Stream generation
    >>> for text in stream_generate(
    ...     model=model,
    ...     processor=processor,
    ...     prompt="What do you see?",
    ...     image="photo.jpg"
    ... ):
    ...     print(text, end="", flush=True)
    >>>
    >>> # Chat interface
    >>> messages = [
    ...     {"role": "user", "content": "What's in this image?"}
    ... ]
    >>> response = chat(
    ...     model=model,
    ...     processor=processor,
    ...     messages=messages,
    ...     image="photo.jpg"
    ... )
"""

from .cache import (
    KVCache,
    RotatingKVCache,
    make_cache,
    make_cache_with_monitoring,
    make_kv_cache,
)
from .config import ModelConfig, TextConfig, VisionConfig, DEFAULT_CONFIG
from .connector import Idefics3Connector
from .generate import chat, generate, prepare_inputs, stream_generate
from .image_processor import ImageProcessor, load_image
from .language import LanguageModel, LanguageModelOutput
from .loader import Processor, load, save_model
from .model import Model
from .vision import VisionModel

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "DEFAULT_CONFIG",
    # Model components
    "Model",
    "VisionModel",
    "LanguageModel",
    "Idefics3Connector",
    # Loading/saving
    "load",
    "save_model",
    "Processor",
    # Generation
    "generate",
    "stream_generate",
    "chat",
    "prepare_inputs",
    # Cache
    "KVCache",
    "RotatingKVCache",
    "make_cache",
    "make_kv_cache",
    "make_cache_with_monitoring",
    # Image processing
    "ImageProcessor",
    "load_image",
    # Outputs
    "LanguageModelOutput",
]
