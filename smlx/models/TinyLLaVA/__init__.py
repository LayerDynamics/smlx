#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
TinyLLaVA Vision-Language Models.

TinyLLaVA combines SigLIP vision encoder with efficient language models
for multimodal understanding. Supports three variants:

- TinyLLaVA-1.5B: TinyLlama language model (1.5B parameters)
- TinyLLaVA-2.0B: StableLM-2 language model (2.0B parameters)
- TinyLLaVA-3.1B: Phi-2 language model (3.1B parameters)

All variants use SigLIP-so400m vision encoder and simple MLP connector.

Example:
    >>> from smlx.models.TinyLLaVA import load, generate, query
    >>> from PIL import Image
    >>>
    >>> # Load model
    >>> model, processor = load("bczhou/TinyLLaVA-1.5B", variant="1.5b")
    >>>
    >>> # Generate caption
    >>> image = Image.open("photo.jpg")
    >>> caption = generate(
    ...     model=model,
    ...     processor=processor,
    ...     prompt="Describe this image:",
    ...     image=image
    ... )
    >>>
    >>> # Visual question answering
    >>> answer = query(
    ...     model=model,
    ...     processor=processor,
    ...     image=image,
    ...     question="What color is the car?"
    ... )

Architecture:
    Vision Encoder: SigLIP-so400m (1152 hidden, 27 layers)
    - Same as SmolVLM-256M
    - Image size: 384x384
    - Patch size: 14x14

    Language Model (1.5B variant):
    - TinyLlama (2048 hidden, 22 layers)
    - Grouped Query Attention (32 heads, 4 KV heads)
    - RMSNorm and RoPE
    - SwiGLU activation

    Connector:
    - 2-layer MLP with GELU activation
    - Projects vision features (1152) to language space (2048)

Variants:
    1.5B: bczhou/TinyLLaVA-1.5B (TinyLlama)
    2.0B: bczhou/TinyLLaVA-2.0B (StableLM-2)
    3.1B: tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B (Phi-2)

Model Size:
    - 1.5B variant: ~3.0 GB (FP16)
    - 2.0B variant: ~4.0 GB (FP16)
    - 3.1B variant: ~6.2 GB (FP16)

Features:
    - Visual question answering
    - Image captioning
    - Multi-turn conversation
    - Streaming generation
    - Temperature and top-p sampling
"""

from .cache import (
    KVCache,
    RotatingKVCache,
    make_cache,
    make_cache_with_monitoring,
    make_kv_cache,
)
from .config import (
    DEFAULT_CONFIG_1_5B,
    DEFAULT_CONFIG_2_0B,
    DEFAULT_CONFIG_3_1B,
    ConnectorConfig,
    ModelConfig,
    TextConfig,
    VisionConfig,
)
from .generate import (
    caption,
    generate,
    prepare_inputs,
    query,
    stream_generate,
)
from .image_processor import ImageProcessor, load_image
from .loader import Processor, get_model_path, load
from .model import TinyLLaVA

__version__ = "0.1.0"

__all__ = [
    # Main API
    "load",
    "generate",
    "stream_generate",
    "caption",
    "query",
    # Model
    "TinyLLaVA",
    # Processor
    "Processor",
    "ImageProcessor",
    # Configuration
    "ModelConfig",
    "VisionConfig",
    "TextConfig",
    "ConnectorConfig",
    "DEFAULT_CONFIG_1_5B",
    "DEFAULT_CONFIG_2_0B",
    "DEFAULT_CONFIG_3_1B",
    # Cache
    "KVCache",
    "RotatingKVCache",
    "make_cache",
    "make_kv_cache",
    "make_cache_with_monitoring",
    # Utilities
    "load_image",
    "prepare_inputs",
    "get_model_path",
]
