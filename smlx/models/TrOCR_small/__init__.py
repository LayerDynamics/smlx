#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
TrOCR-small - Transformer-based OCR.

Microsoft's lightweight Transformer-based optical character recognition model
combining BEiT vision encoder with RoBERTa text decoder for both printed
and handwritten text recognition.

Quick Start:
    >>> from smlx.models.TrOCR_small import load, recognize
    >>> from PIL import Image
    >>>
    >>> # Load model (printed text variant)
    >>> model, processor = load("printed")
    >>>
    >>> # Load image with text
    >>> image = Image.open("document.jpg")
    >>>
    >>> # Recognize text
    >>> text = recognize(model, processor, image)
    >>> print(f"Text: {text}")

Features:
    - ~60M parameters (lightweight)
    - Transformer-based (no external OCR engine)
    - Two variants: printed and handwritten text
    - Single-line text recognition
    - End-to-end trainable
    - MIT license

Model Variants:
    - microsoft/trocr-small-printed - For printed/typed text
    - microsoft/trocr-small-handwritten - For handwriting

Architecture:
    - Vision Encoder: BEiT (BERT Pre-Training of Image Transformers)
    - Text Decoder: RoBERTa (autoregressive)
    - Input: 384x384 RGB images
    - Output: Text sequences

Use Cases:
    - Document digitization
    - Receipt parsing
    - Form field extraction
    - Handwritten note recognition
    - On-device OCR
    - Privacy-sensitive document processing

Performance:
    - Model Size: ~250MB (FP16), ~65MB (4-bit quantized)
    - Memory Usage: ~250MB
    - Latency: Fast on M4
    - Accuracy: High for printed, medium-high for handwritten

References:
    - HuggingFace: microsoft/trocr-small-printed
    - Paper: "TrOCR: Transformer-based Optical Character Recognition"
    - ArXiv: https://arxiv.org/abs/2109.10282
"""

from .config import (
    DEFAULT_CONFIG_HANDWRITTEN,
    DEFAULT_CONFIG_PRINTED,
    TrOCRConfig,
    TrOCRDecoderConfig,
    TrOCRVisionConfig,
)
from .loader import load, save_model
from .model import TextDecoder, TrOCR, VisionEncoder
from .processor import TrOCRImageProcessor, TrOCRProcessor, TrOCRTokenizer
from .recognize import (
    preprocess_image,
    recognize,
    recognize_batch,
    recognize_with_confidence,
)

__version__ = "0.1.0"

__all__ = [
    # Model
    "TrOCR",
    "VisionEncoder",
    "TextDecoder",
    # Config
    "TrOCRConfig",
    "TrOCRVisionConfig",
    "TrOCRDecoderConfig",
    "DEFAULT_CONFIG_PRINTED",
    "DEFAULT_CONFIG_HANDWRITTEN",
    # Loading
    "load",
    "save_model",
    # Processing
    "TrOCRProcessor",
    "TrOCRImageProcessor",
    "TrOCRTokenizer",
    # Recognition
    "recognize",
    "recognize_batch",
    "recognize_with_confidence",
    "preprocess_image",
]
