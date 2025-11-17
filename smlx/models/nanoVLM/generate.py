#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Text generation for nanoVLM.

Handles vision-language generation with streaming support.
"""

from typing import Generator, Optional, Union

import mlx.core as mx
from PIL import Image

from .loader import Processor
from .model import NanoVLM


def prepare_inputs(
    processor: Processor,
    prompt: str,
    image: Optional[Union[str, Image.Image]] = None,
) -> dict:
    """
    Prepare inputs for nanoVLM.

    Args:
        processor: Processor instance
        prompt: Text prompt
        image: Optional image

    Returns:
        Dictionary with input_ids, pixel_values, and image_token_mask
    """
    # Process text
    input_ids = processor.tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)

    # Add batch dimension if needed
    if len(input_ids.shape) == 1:
        input_ids = mx.expand_dims(input_ids, axis=0)

    inputs = {"input_ids": input_ids}

    # Process image if provided
    if image is not None:
        pixel_values = processor.image_processor(image)
        inputs["pixel_values"] = pixel_values

        # Create image token mask
        # For simplicity, assume image tokens are at the beginning
        batch_size, seq_len = input_ids.shape
        num_image_tokens = 196  # 14x14 patches

        # Create mask: first num_image_tokens positions are 1, rest are 0
        image_token_mask = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        if seq_len >= num_image_tokens:
            image_token_mask[:, :num_image_tokens] = 1

        inputs["image_token_mask"] = image_token_mask

    return inputs


def sample(logits: mx.array, temperature: float = 1.0, top_p: float = 0.95) -> int:
    """
    Sample next token from logits.

    Args:
        logits: Logits for next token
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Sampled token ID
    """
    if temperature == 0:
        # Greedy sampling
        return int(mx.argmax(logits, axis=-1))

    # Apply temperature
    logits = logits / temperature

    # Softmax to get probabilities
    probs = mx.softmax(logits, axis=-1)

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        # Sort probabilities
        sorted_indices = mx.argsort(probs, axis=-1)[::-1]
        sorted_probs = probs[sorted_indices]

        # Cumulative probabilities
        cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

        # Find cutoff
        cutoff_idx = mx.argmax(cumsum_probs > top_p)
        cutoff_idx = int(cutoff_idx)  # Convert MLX array to Python int
        if cutoff_idx == 0:
            cutoff_idx = len(sorted_probs)

        # Keep only top-p tokens
        top_indices = sorted_indices[:cutoff_idx]
        top_probs = probs[top_indices]

        # Renormalize
        top_probs = top_probs / mx.sum(top_probs)

        # Sample from top-p
        token_idx = mx.random.categorical(mx.log(top_probs))
        token = int(top_indices[token_idx])
    else:
        # Sample from full distribution
        token = int(mx.random.categorical(mx.log(probs)))

    return token


def generate(
    model: NanoVLM,
    processor: Processor,
    prompt: str,
    image: Optional[Union[str, Image.Image]] = None,
    max_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> str:
    """
    Generate text from nanoVLM.

    Args:
        model: NanoVLM model
        processor: Processor instance
        prompt: Text prompt
        image: Optional image
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Generated text

    Example:
        >>> model, processor = load("lusxvr/nanoVLM-222M")
        >>> image = Image.open("photo.jpg")
        >>> text = generate(
        ...     model, processor,
        ...     prompt="Describe this image:",
        ...     image=image,
        ...     max_tokens=100
        ... )
        >>> print(text)
    """
    # Prepare inputs
    inputs = prepare_inputs(processor, prompt, image)

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    image_token_mask = inputs.get("image_token_mask")

    # Generate tokens
    generated_tokens = []

    for _ in range(max_tokens):
        # Forward pass
        logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_mask=image_token_mask,
        )

        # Get logits for next token (last position)
        next_token_logits = logits[0, -1, :]
        # Evaluate to prevent computation graph accumulation
        mx.eval(next_token_logits)

        # Sample next token
        next_token = sample(next_token_logits, temperature, top_p)

        # Check for EOS
        if next_token == processor.tokenizer.eos_token_id:
            break

        # Add to generated tokens
        generated_tokens.append(next_token)

        # Update input_ids for next iteration
        next_token_array = mx.array([[next_token]])
        input_ids = mx.concatenate([input_ids, next_token_array], axis=1)
        # Evaluate to prevent graph accumulation from concatenation
        mx.eval(input_ids)

        # Clear image inputs after first token (only use once)
        pixel_values = None
        image_token_mask = None

    # Decode generated tokens
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )

    return generated_text


def stream_generate(
    model: NanoVLM,
    processor: Processor,
    prompt: str,
    image: Optional[Union[str, Image.Image]] = None,
    max_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> Generator[str, None, None]:
    """
    Stream generated text token by token.

    Args:
        model: NanoVLM model
        processor: Processor instance
        prompt: Text prompt
        image: Optional image
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Yields:
        Generated text chunks

    Example:
        >>> model, processor = load("lusxvr/nanoVLM-222M")
        >>> image = Image.open("photo.jpg")
        >>> for chunk in stream_generate(
        ...     model, processor,
        ...     prompt="Describe:",
        ...     image=image
        ... ):
        ...     print(chunk, end="", flush=True)
    """
    # Prepare inputs
    inputs = prepare_inputs(processor, prompt, image)

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    image_token_mask = inputs.get("image_token_mask")

    # Generate tokens
    for _ in range(max_tokens):
        # Forward pass
        logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_mask=image_token_mask,
        )

        # Get logits for next token
        next_token_logits = logits[0, -1, :]
        # Evaluate to prevent computation graph accumulation
        mx.eval(next_token_logits)

        # Sample
        next_token = sample(next_token_logits, temperature, top_p)

        # Check for EOS
        if next_token == processor.tokenizer.eos_token_id:
            break

        # Decode and yield
        token_text = processor.tokenizer.decode([next_token], skip_special_tokens=True)
        yield token_text

        # Update input_ids
        next_token_array = mx.array([[next_token]])
        input_ids = mx.concatenate([input_ids, next_token_array], axis=1)
        # Evaluate to prevent graph accumulation from concatenation
        mx.eval(input_ids)

        # Clear image after first token
        pixel_values = None
        image_token_mask = None


def caption(
    model: NanoVLM,
    processor: Processor,
    image: Union[str, Image.Image],
    max_tokens: int = 100,
) -> str:
    """
    Generate image caption.

    Args:
        model: NanoVLM model
        processor: Processor instance
        image: Image to caption
        max_tokens: Maximum tokens to generate

    Returns:
        Image caption

    Example:
        >>> model, processor = load("lusxvr/nanoVLM-222M")
        >>> caption_text = caption(model, processor, "photo.jpg")
        >>> print(caption_text)
    """
    prompt = "Describe this image:"
    return generate(model, processor, prompt, image, max_tokens=max_tokens)


def query(
    model: NanoVLM,
    processor: Processor,
    image: Union[str, Image.Image],
    question: str,
    max_tokens: int = 80,
) -> str:
    """
    Visual question answering.

    Args:
        model: NanoVLM model
        processor: Processor instance
        image: Image to query
        question: Question about image
        max_tokens: Maximum tokens for answer

    Returns:
        Answer to question

    Example:
        >>> model, processor = load("lusxvr/nanoVLM-222M")
        >>> answer = query(
        ...     model, processor,
        ...     image="photo.jpg",
        ...     question="What color is the car?"
        ... )
        >>> print(answer)
    """
    prompt = f"Question: {question}\nAnswer:"
    return generate(model, processor, prompt, image, max_tokens=max_tokens)
