#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
TrOCR Text Recognition Interface.

High-level API for optical character recognition using TrOCR.
"""

from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np
from PIL import Image

from .model import TrOCR
from .processor import TrOCRProcessor


def _decoder_start_token_id(model: TrOCR, processor: TrOCRProcessor) -> int:
    """Return the token id the decoder starts generation from.

    TrOCR (like other VisionEncoderDecoder models) begins decoding from
    ``decoder_start_token_id``, which is the </s>/eos token (id 2), not the bos
    token (id 0). Prefer an explicit config value, then the decoder's eos.
    """
    cfg = getattr(model, "config", None)
    explicit = getattr(cfg, "decoder_start_token_id", None)
    if explicit is not None:
        return int(explicit)
    dec = getattr(cfg, "decoder", None)
    dec_start = getattr(dec, "decoder_start_token_id", None) if dec is not None else None
    if dec_start is not None:
        return int(dec_start)
    eos = getattr(processor.tokenizer, "eos_token_id", None)
    return int(eos) if eos is not None else 2


def generate_text(
    model: TrOCR,
    processor: TrOCRProcessor,
    pixel_values: mx.array,
    max_length: int = 128,
    num_beams: int = 1,
    temperature: float = 1.0,
) -> str:
    """Generate text from image features.

    Args:
        model: TrOCR model
        processor: TrOCR processor
        pixel_values: Processed image
        max_length: Maximum generation length
        num_beams: Number of beams for beam search (1 = greedy)
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    # Encode image
    encoder_hidden_states = model.encode(pixel_values)

    # TrOCR autoregressive decoding starts from the decoder_start_token_id, which
    # for TrOCR is the </s>/eos token (id 2) — NOT the bos token (id 0). Starting
    # from bos produces garbage (the model was trained to begin after </s>).
    start_token_id = _decoder_start_token_id(model, processor)
    eos_token_id = processor.tokenizer.eos_token_id

    input_ids = mx.array([[start_token_id]])

    # Autoregressive generation
    for _ in range(max_length):
        # Decode
        logits = model.decode(input_ids, encoder_hidden_states)

        # Get next token logits
        next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Greedy decoding (take argmax)
        if num_beams == 1:
            next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)
        else:
            # Simple beam search (sample from top-k)
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = mx.random.categorical(probs, num_samples=1)

        # Check for EOS
        if int(next_token[0, 0]) == eos_token_id:
            break

        # Append to input_ids
        input_ids = mx.concatenate([input_ids, next_token], axis=1)

    # Decode to text
    # tolist() can return various types, but decode handles them all
    generated_ids = input_ids[0]  # Keep as array, decode handles conversion
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    return text


def recognize(
    model: TrOCR,
    processor: TrOCRProcessor,
    image: Union[str, Path, Image.Image, np.ndarray, mx.array],
    max_length: int = 128,
    num_beams: int = 1,
    temperature: float = 1.0,
) -> str:
    """Recognize text from image.

    Args:
        model: TrOCR model
        processor: TrOCR processor
        image: Image source
        max_length: Maximum text length
        num_beams: Number of beams for beam search
        temperature: Sampling temperature

    Returns:
        Recognized text

    Example:
        >>> from smlx.models.TrOCR_small import load, recognize
        >>> from PIL import Image
        >>>
        >>> model, processor = load("printed")
        >>> image = Image.open("document.jpg")
        >>> text = recognize(model, processor, image)
        >>> print(text)
    """
    # Process image
    pixel_values = processor.process_image(image)

    # Generate text
    text = generate_text(model, processor, pixel_values, max_length, num_beams, temperature)

    return text


def recognize_batch(
    model: TrOCR,
    processor: TrOCRProcessor,
    images: list[Union[str, Path, Image.Image, np.ndarray, mx.array]],
    max_length: int = 128,
    num_beams: int = 1,
) -> list[str]:
    """Recognize text from multiple images.

    Args:
        model: TrOCR model
        processor: TrOCR processor
        images: List of image sources
        max_length: Maximum text length
        num_beams: Number of beams

    Returns:
        List of recognized texts

    Example:
        >>> images = ["doc1.jpg", "doc2.jpg", "doc3.jpg"]
        >>> texts = recognize_batch(model, processor, images)
        >>> for img, text in zip(images, texts):
        ...     print(f"{img}: {text}")
    """
    results = []
    for image in images:
        text = recognize(model, processor, image, max_length, num_beams)
        results.append(text)
    return results


def recognize_with_confidence(
    model: TrOCR,
    processor: TrOCRProcessor,
    image: Union[str, Path, Image.Image, np.ndarray, mx.array],
    max_length: int = 128,
) -> tuple[str, float]:
    """Recognize text with confidence score.

    Args:
        model: TrOCR model
        processor: TrOCR processor
        image: Image source
        max_length: Maximum text length

    Returns:
        Tuple of (text, confidence)
    """
    # Process image
    pixel_values = processor.process_image(image)

    # Encode image
    encoder_hidden_states = model.encode(pixel_values)

    # Generate with confidence tracking (start from the decoder_start token, id 2)
    start_token_id = _decoder_start_token_id(model, processor)
    eos_token_id = processor.tokenizer.eos_token_id

    input_ids = mx.array([[start_token_id]])
    confidences = []

    for _ in range(max_length):
        logits = model.decode(input_ids, encoder_hidden_states)
        next_token_logits = logits[:, -1, :]

        # Get probabilities
        probs = mx.softmax(next_token_logits, axis=-1)

        # Greedy decoding
        next_token = mx.argmax(probs, axis=-1, keepdims=True)

        # Store confidence (max probability)
        confidence = float(mx.max(probs))
        confidences.append(confidence)

        # Check for EOS
        if int(next_token[0, 0]) == eos_token_id:
            break

        input_ids = mx.concatenate([input_ids, next_token], axis=1)

    # Decode text
    # Keep as array, decode handles conversion
    generated_ids = input_ids[0]
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Average confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return text, avg_confidence


def preprocess_image(image: Image.Image, enhance: bool = True) -> Image.Image:
    """Preprocess image to improve OCR accuracy.

    Args:
        image: PIL Image
        enhance: Whether to apply enhancements

    Returns:
        Preprocessed image
    """
    if not enhance:
        return image

    from PIL import ImageEnhance

    # Convert to grayscale
    image = image.convert("L").convert("RGB")

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)

    return image


__all__ = [
    "recognize",
    "recognize_batch",
    "recognize_with_confidence",
    "generate_text",
    "preprocess_image",
]
