#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Text generation with TinyLLaVA.

Supports:
- Text generation with image inputs
- Streaming generation
- Visual question answering
- Image captioning
- Temperature and top-p sampling
"""

from typing import Generator, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .image_processor import load_image
from .loader import Processor
from .model import TinyLLaVA

# Import cache from local module - enhanced with monitoring & quantization
from .cache import make_kv_cache

# Import sampling utilities from SmolLM2 (reuse where possible)
try:
    from smlx.utils.sampling import sample_token
except ImportError:
    # Fallback implementation if utils not available yet
    def sample_token(logits: mx.array, temperature: float = 0.0, top_p: float = 1.0) -> int:
        """Sample next token from logits."""
        if temperature == 0:
            return int(mx.argmax(logits, axis=-1).item())

        logits = logits / temperature

        if top_p < 1.0:
            # Top-p (nucleus) sampling
            sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

            # Find cutoff
            cutoff_index = mx.argmax(cumsum_probs >= top_p, axis=-1).item()
            cutoff_logit = sorted_logits[0, cutoff_index].item()

            # Mask out low-probability tokens
            logits = mx.where(logits < cutoff_logit, float("-inf"), logits)

        # Sample from remaining distribution
        probs = mx.softmax(logits, axis=-1)
        token = mx.random.categorical(mx.log(probs))
        return int(token.item())


def prepare_inputs(
    processor: Processor,
    prompt: str,
    image: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None,
    image_token: str = "<image>",
) -> dict:
    """Prepare inputs for generation.

    Args:
        processor: Combined tokenizer + image processor
        prompt: Text prompt
        image: Optional image(s) (URL, path, or PIL Image)
        image_token: Token to use for image placeholders

    Returns:
        Dictionary with input_ids and pixel_values
    """
    # Load and preprocess images if provided
    pixel_values = None
    if image is not None:
        if not isinstance(image, list):
            images = [image]
        else:
            images = image

        # Load images
        loaded_images = [
            load_image(img) if isinstance(img, str) else img for img in images
        ]

        # Preprocess images
        processed_images = processor.image_processor(loaded_images)
        pixel_values = mx.array(np.stack(processed_images))

        # Add <image> tokens to prompt if not already present
        if image_token not in prompt:
            # Prepend image tokens (one per image)
            prompt = (image_token + "\n") * len(images) + prompt

    # Tokenize text
    inputs = processor.tokenizer(
        prompt,
        return_tensors="np",
        padding=False,
        truncation=False,
    )

    input_ids = mx.array(inputs["input_ids"])

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }


def generate(
    model: TinyLLaVA,
    processor: Processor,
    prompt: str,
    image: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = False,
) -> str:
    """Generate text response from prompt and optional image.

    Args:
        model: TinyLLaVA model
        processor: Combined tokenizer + image processor
        prompt: Text prompt
        image: Optional image(s) (URL, path, or PIL Image)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        verbose: Print generation statistics

    Returns:
        Generated text

    Example:
        >>> from smlx.models.TinyLLaVA import load, generate
        >>> model, processor = load()
        >>> image = "https://example.com/photo.jpg"
        >>> output = generate(
        ...     model=model,
        ...     processor=processor,
        ...     prompt="Describe this image:",
        ...     image=image,
        ...     max_tokens=300
        ... )
    """
    # Prepare inputs
    inputs = prepare_inputs(processor, prompt, image)
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]

    # Encode image if provided (only once at the start)
    image_features = None
    if pixel_values is not None:
        image_features = model.encode_images(pixel_values)
        # Evaluate image features to prevent graph accumulation
        mx.eval(image_features)

    # Create KV cache
    cache = make_kv_cache(model.language_model)

    # First forward pass (process prompt + image)
    # Model handles combining image and text embeddings
    logits, _ = model(input_ids, pixel_values=None, image_features=image_features, cache=cache)
    logits = logits[:, -1, :]
    # Evaluate to prevent computation graph accumulation
    mx.eval(logits)

    # Sample first token
    token = sample_token(logits, temperature, top_p)
    tokens = [token]

    # Generate remaining tokens
    for _ in range(max_tokens - 1):
        # Forward pass with just the new token (no image features after first pass)
        y = mx.array([[token]])
        # Evaluate input array
        mx.eval(y)
        logits, _ = model(y, pixel_values=None, image_features=None, cache=cache)
        logits = logits[:, -1, :]
        # Evaluate to prevent computation graph accumulation
        mx.eval(logits)

        # Sample next token
        token = sample_token(logits, temperature, top_p)

        # Check for EOS
        if processor.tokenizer.eos_token_id and token == processor.tokenizer.eos_token_id:
            break

        tokens.append(token)

    # Decode tokens
    output_text = processor.tokenizer.decode(tokens, skip_special_tokens=True)

    if verbose:
        print(f"Generated {len(tokens)} tokens")

    return output_text


def stream_generate(
    model: TinyLLaVA,
    processor: Processor,
    prompt: str,
    image: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """Generate text with streaming output.

    Args:
        model: TinyLLaVA model
        processor: Combined tokenizer + image processor
        prompt: Text prompt
        image: Optional image(s) (URL, path, or PIL Image)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Yields:
        Generated text tokens as they are produced

    Example:
        >>> from smlx.models.TinyLLaVA import load, stream_generate
        >>> model, processor = load()
        >>> for text in stream_generate(
        ...     model=model,
        ...     processor=processor,
        ...     prompt="What is in this image?",
        ...     image="photo.jpg",
        ...     max_tokens=300
        ... ):
        ...     print(text, end="", flush=True)
    """
    # Prepare inputs
    inputs = prepare_inputs(processor, prompt, image)
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]

    # Encode image if provided (only once at the start)
    image_features = None
    if pixel_values is not None:
        image_features = model.encode_images(pixel_values)
        # Evaluate image features to prevent graph accumulation
        mx.eval(image_features)

    # Create KV cache
    cache = make_kv_cache(model.language_model)

    # First forward pass (process prompt + image)
    # Model handles combining image and text embeddings
    logits, _ = model(input_ids, pixel_values=None, image_features=image_features, cache=cache)
    logits = logits[:, -1, :]
    # Evaluate to prevent computation graph accumulation
    mx.eval(logits)

    # Sample first token
    token = sample_token(logits, temperature, top_p)

    # Decode and yield first token
    text = processor.tokenizer.decode([token], skip_special_tokens=True)
    yield text

    # Generate remaining tokens
    for _ in range(max_tokens - 1):
        # Forward pass with just the new token (no image features after first pass)
        y = mx.array([[token]])
        # Evaluate input array
        mx.eval(y)
        logits, _ = model(y, pixel_values=None, image_features=None, cache=cache)
        logits = logits[:, -1, :]
        # Evaluate to prevent computation graph accumulation
        mx.eval(logits)

        token = sample_token(logits, temperature, top_p)

        # Check for EOS
        if processor.tokenizer.eos_token_id and token == processor.tokenizer.eos_token_id:
            break

        # Decode and yield
        text = processor.tokenizer.decode([token], skip_special_tokens=True)
        yield text


def caption(
    model: TinyLLaVA,
    processor: Processor,
    image: Union[str, Image.Image],
    prompt: str = "Describe this image in detail.",
    max_tokens: int = 300,
    **kwargs,
) -> str:
    """Generate image caption.

    Args:
        model: TinyLLaVA model
        processor: Combined tokenizer + image processor
        image: Image (URL, path, or PIL Image)
        prompt: Caption prompt
        max_tokens: Maximum tokens to generate
        **kwargs: Additional arguments for generate()

    Returns:
        Image caption text

    Example:
        >>> from smlx.models.TinyLLaVA import load, caption
        >>> model, processor = load()
        >>> caption_text = caption(
        ...     model=model,
        ...     processor=processor,
        ...     image="photo.jpg"
        ... )
    """
    return generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=image,
        max_tokens=max_tokens,
        **kwargs,
    )


def query(
    model: TinyLLaVA,
    processor: Processor,
    image: Union[str, Image.Image],
    question: str,
    max_tokens: int = 300,
    **kwargs,
) -> str:
    """Visual question answering.

    Args:
        model: TinyLLaVA model
        processor: Combined tokenizer + image processor
        image: Image (URL, path, or PIL Image)
        question: Question about the image
        max_tokens: Maximum tokens to generate
        **kwargs: Additional arguments for generate()

    Returns:
        Answer to the question

    Example:
        >>> from smlx.models.TinyLLaVA import load, query
        >>> model, processor = load()
        >>> answer = query(
        ...     model=model,
        ...     processor=processor,
        ...     image="photo.jpg",
        ...     question="How many people are in this image?"
        ... )
    """
    return generate(
        model=model,
        processor=processor,
        prompt=question,
        image=image,
        max_tokens=max_tokens,
        **kwargs,
    )


__all__ = ["generate", "stream_generate", "caption", "query", "prepare_inputs"]
