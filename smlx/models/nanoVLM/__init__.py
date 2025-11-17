#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
nanoVLM: Minimal Vision-Language Model

A lightweight 222M parameter multimodal model for learning and experimentation.

Architecture:
    - SigLIP-base vision encoder (85M parameters, 224x224 images)
    - MLP projection layer (~2M parameters)
    - SmolLM2-135M language model (135M parameters)

Total: ~222M parameters, ~750 lines of implementation code

Example - Basic Usage:
    >>> from smlx.models.nanoVLM import load, generate
    >>> from PIL import Image
    >>>
    >>> # Load model
    >>> model, processor = load("lusxvr/nanoVLM-222M")
    >>>
    >>> # Generate from image
    >>> image = Image.open("photo.jpg")
    >>> response = generate(
    ...     model=model,
    ...     processor=processor,
    ...     prompt="Describe this image:",
    ...     image=image,
    ...     max_tokens=100
    ... )
    >>> print(response)

Example - Image Captioning:
    >>> from smlx.models.nanoVLM import load, caption
    >>>
    >>> model, processor = load("lusxvr/nanoVLM-222M")
    >>> caption_text = caption(model, processor, "photo.jpg")
    >>> print(caption_text)

Example - Visual Question Answering:
    >>> from smlx.models.nanoVLM import load, query
    >>>
    >>> model, processor = load("lusxvr/nanoVLM-222M")
    >>> answer = query(
    ...     model=model,
    ...     processor=processor,
    ...     image="photo.jpg",
    ...     question="What objects are in this image?"
    ... )
    >>> print(answer)

Example - Streaming Generation:
    >>> from smlx.models.nanoVLM import load, stream_generate
    >>>
    >>> model, processor = load("lusxvr/nanoVLM-222M")
    >>> for chunk in stream_generate(
    ...     model=model,
    ...     processor=processor,
    ...     prompt="Describe:",
    ...     image="photo.jpg"
    ... ):
    ...     print(chunk, end="", flush=True)

Features:
    - Minimal implementation (~750 lines total)
    - Fast to train (6 hours on H100)
    - Easy to customize and understand
    - Perfect for learning VLM architecture
    - Runs on <1GB RAM
    - Apache 2.0 license

Model Details:
    - Parameters: 222M (135M language + 85M vision + 2M projection)
    - Image Size: 224x224
    - Context Length: 2048 tokens
    - Vision Tokens: 196 (14x14 patches)
    - Memory (FP16): ~900MB
    - Memory (4-bit): ~220MB

Performance (M4 Pro):
    - Tokens/sec: ~52 (FP16)
    - Tokens/sec: ~62 (4-bit)
    - Image encoding: ~35ms
    - First token latency: ~45ms

Use Cases:
    - Learning VLM architecture from scratch
    - Rapid prototyping of vision-language apps
    - Custom domain-specific VLMs (medical, e-commerce, etc.)
    - Ultra-lightweight deployments
    - Educational projects
    - Research experiments
    - Basic image captioning and VQA
    - Devices with <1GB RAM constraint

Limitations:
    - Quality: Lower than larger VLMs (SmolVLM-500M, Moondream2)
    - Resolution: Limited to 224×224 images
    - Reasoning: Minimal reasoning capabilities
    - Specialized Tasks: Requires fine-tuning for specific domains

Why nanoVLM?:
    - ✅ Fastest to train from scratch (~6 hours)
    - ✅ Easiest to understand (~750 lines)
    - ✅ Smallest memory footprint (<1GB)
    - ✅ Perfect for learning and experimentation
    - ✅ Ideal for rapid prototyping
    - ✅ Apache 2.0 license (fully open)
"""

# Configuration
from .config import (
    DEFAULT_CONFIG,
    LanguageConfig,
    NanoVLMConfig,
    ProjectionConfig,
    VisionConfig,
    load_config,
    save_config,
)

# Generation
from .generate import caption, generate, prepare_inputs, query, stream_generate

# Image processing
from .image_processor import ImageProcessor, create_image_processor, load_image

# Loading and saving
from .loader import Processor, get_model_path, load, load_weights, save_model

# Model components
from .model import NanoVLM, create_model
from .projection import MLPProjection, create_projection
from .vision import VisionModel

__version__ = "0.1.0"

__all__ = [
    # Main API (most commonly used)
    "load",
    "generate",
    "stream_generate",
    "caption",
    "query",
    # Model
    "NanoVLM",
    "create_model",
    # Components
    "VisionModel",
    "MLPProjection",
    # Processor
    "Processor",
    "ImageProcessor",
    # Configuration
    "NanoVLMConfig",
    "VisionConfig",
    "LanguageConfig",
    "ProjectionConfig",
    "DEFAULT_CONFIG",
    # Utilities
    "prepare_inputs",
    "load_image",
    "create_image_processor",
    "create_projection",
    # Loading/Saving
    "get_model_path",
    "load_weights",
    "save_model",
    "load_config",
    "save_config",
]
