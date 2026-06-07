#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Text generation with SmolVLM-256M-Instruct.

Supports:
- Text generation with image inputs
- Streaming generation
- Chat-style interactions
- Temperature and top-p sampling
"""

from collections.abc import Generator
from typing import Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image

# Import cache from local module - enhanced with monitoring & quantization
from .cache import make_kv_cache
from .image_processor import load_image
from .loader import Processor
from .model import Model

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
    image: Optional[Union[str, Image.Image, list[Union[str, Image.Image]]]] = None,
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
    # Prefer the HuggingFace AutoProcessor, which expands each <image> placeholder
    # into exactly the number of vision tokens the encoder produces (required for the
    # 1:1 image-token replacement in the model). Detect it by its HF-specific attrs.
    is_hf_processor = (
        hasattr(processor, "image_seq_len")
        and hasattr(processor, "image_processor")
        and hasattr(processor, "tokenizer")
    )

    if is_hf_processor and image is not None:
        images = image if isinstance(image, list) else [image]
        loaded_images = [load_image(img) if isinstance(img, str) else img for img in images]
        if image_token not in prompt:
            prompt = (image_token + "\n") * len(images) + prompt
        inputs = processor(text=prompt, images=loaded_images, return_tensors="np")
        return {
            "input_ids": mx.array(inputs["input_ids"]),
            "pixel_values": (
                mx.array(inputs["pixel_values"]) if "pixel_values" in inputs else None
            ),
        }

    # Fallback: custom processor path.
    pixel_values = None
    if image is not None:
        images = image if isinstance(image, list) else [image]
        loaded_images = [load_image(img) if isinstance(img, str) else img for img in images]
        processed_images = processor.image_processor(loaded_images)
        pixel_values = mx.array(np.stack(processed_images))
        if image_token not in prompt:
            prompt = (image_token + "\n") * len(images) + prompt

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
    model: Model,
    processor: Processor,
    prompt: str,
    image: Optional[Union[str, Image.Image, list[Union[str, Image.Image]]]] = None,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = False,
) -> str:
    """Generate text response from prompt and optional image.

    Args:
        model: SmolVLM model
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
        >>> from smlx.models.SmolVLM_256M import load, generate
        >>> model, processor = load()
        >>> image = "https://example.com/photo.jpg"
        >>> output = generate(
        ...     model=model,
        ...     processor=processor,
        ...     prompt="Describe this image:",
        ...     image=image,
        ...     max_tokens=100
        ... )
    """
    # Prepare inputs
    inputs = prepare_inputs(processor, prompt, image)
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]

    # Create KV cache
    cache = make_kv_cache(model.language_model)

    # First forward pass (process prompt + image)
    outputs = model(input_ids, pixel_values, cache=cache)
    logits = outputs.logits[:, -1, :]
    # Evaluate to prevent computation graph accumulation
    mx.eval(logits)

    # Sample first token
    token = sample_token(logits, temperature, top_p)
    tokens = [token]

    # Generate remaining tokens
    for _ in range(max_tokens - 1):
        # Forward pass with just the new token
        y = mx.array([[token]])
        # Evaluate input array
        mx.eval(y)
        outputs = model.language_model(y, cache=cache)
        logits = outputs.logits[:, -1, :]
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
    model: Model,
    processor: Processor,
    prompt: str,
    image: Optional[Union[str, Image.Image, list[Union[str, Image.Image]]]] = None,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """Generate text with streaming output.

    Args:
        model: SmolVLM model
        processor: Combined tokenizer + image processor
        prompt: Text prompt
        image: Optional image(s) (URL, path, or PIL Image)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Yields:
        Generated text tokens as they are produced

    Example:
        >>> from smlx.models.SmolVLM_256M import load, stream_generate
        >>> model, processor = load()
        >>> for text in stream_generate(
        ...     model=model,
        ...     processor=processor,
        ...     prompt="What is in this image?",
        ...     image="photo.jpg",
        ...     max_tokens=50
        ... ):
        ...     print(text, end="", flush=True)
    """
    # Prepare inputs
    inputs = prepare_inputs(processor, prompt, image)
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]

    # Create KV cache
    cache = make_kv_cache(model.language_model)

    # First forward pass
    outputs = model(input_ids, pixel_values, cache=cache)
    logits = outputs.logits[:, -1, :]
    # Evaluate to prevent computation graph accumulation
    mx.eval(logits)

    # Sample first token
    token = sample_token(logits, temperature, top_p)

    # Decode and yield first token
    text = processor.tokenizer.decode([token], skip_special_tokens=True)
    yield text

    # Generate remaining tokens
    for _ in range(max_tokens - 1):
        y = mx.array([[token]])
        # Evaluate input array
        mx.eval(y)
        outputs = model.language_model(y, cache=cache)
        logits = outputs.logits[:, -1, :]
        # Evaluate to prevent computation graph accumulation
        mx.eval(logits)

        token = sample_token(logits, temperature, top_p)

        # Check for EOS
        if processor.tokenizer.eos_token_id and token == processor.tokenizer.eos_token_id:
            break

        # Decode and yield
        text = processor.tokenizer.decode([token], skip_special_tokens=True)
        yield text


def chat(
    model: Model,
    processor: Processor,
    messages: list[dict],
    image: Optional[Union[str, Image.Image, list[Union[str, Image.Image]]]] = None,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Chat-style interaction with conversation history.

    Args:
        model: SmolVLM model
        processor: Combined tokenizer + image processor
        messages: List of message dicts with 'role' and 'content'
        image: Optional image(s)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Generated response text

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "What do you see in this image?"}
        ... ]
        >>> response = chat(
        ...     model=model,
        ...     processor=processor,
        ...     messages=messages,
        ...     image="photo.jpg"
        ... )
    """
    # Apply chat template if available
    if hasattr(processor.tokenizer, "apply_chat_template"):
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Simple fallback formatting
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        prompt += "assistant: "

    return generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=image,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


__all__ = ["generate", "stream_generate", "chat", "prepare_inputs"]
