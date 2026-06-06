#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Text generation for nanoVLM.

Handles vision-language generation with streaming support.
Includes output validation to detect and prevent gibberish outputs.
"""

import logging
from collections.abc import Generator
from typing import Optional, Union

import mlx.core as mx
from PIL import Image

from ...utils.cache import make_cache
from ...utils.validation import validate_text_output
from .loader import Processor
from .model import NanoVLM

logger = logging.getLogger(__name__)


def prepare_inputs(
    processor: Processor,
    prompt: str,
    image: Optional[Union[str, Image.Image]] = None,
    image_token_id: int = 49150,
    num_image_tokens: int = 49,
) -> dict:
    """
    Prepare inputs for nanoVLM with proper image token insertion.

    Args:
        processor: Processor instance
        prompt: Text prompt (should contain <image> placeholder)
        image: Optional image
        image_token_id: Token ID for image placeholder (default: 49150)
        num_image_tokens: Number of image tokens after pixel shuffle (default: 49 for 7x7)

    Returns:
        Dictionary with input_ids and pixel_values

    Note:
        Follows mlx-vlm pattern:
        1. Splits prompt on "<image>" placeholder
        2. Tokenizes each chunk separately
        3. Inserts num_image_tokens (49) instances of image_token_id between chunks
        4. Model will replace these tokens with actual vision embeddings
    """
    # Process image if provided
    if image is not None:
        # Process image
        pixel_values = processor.image_processor(image)

        # Split prompt on <image> placeholder (mlx-vlm pattern)
        if "<image>" in prompt:
            chunks = prompt.split("<image>")
            # Tokenize each chunk
            chunk_ids = [
                processor.tokenizer.encode(chunk, return_tensors="np")[0].tolist()
                for chunk in chunks
            ]

            # Insert num_image_tokens instances of image_token_id between chunks
            # Example: "Describe <image>" -> [tokens("Describe ")] + [49150]*49 + [tokens("")]
            input_ids = chunk_ids[0] + [image_token_id] * num_image_tokens + chunk_ids[1]
        else:
            # If no <image> placeholder, prepend image tokens before text (fallback)
            text_ids = processor.tokenizer.encode(prompt, return_tensors="np")[0].tolist()
            input_ids = [image_token_id] * num_image_tokens + text_ids

        # Convert to MLX array with batch dimension
        input_ids = mx.array([input_ids])

        inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
    else:
        # Text-only mode
        text_ids = processor.tokenizer.encode(prompt, return_tensors="np")
        text_ids = mx.array(text_ids)

        # Add batch dimension if needed
        if len(text_ids.shape) == 1:
            text_ids = mx.expand_dims(text_ids, axis=0)

        inputs = {"input_ids": text_ids}

    return inputs


def sample(
    logits: mx.array,
    temperature: float = 0.5,
    top_p: float = 1.0,
    previous_tokens: Optional[list] = None,
    repetition_penalty: float = 1.0,
) -> int:
    """
    Sample next token from logits with optional repetition penalty.

    Args:
        logits: Logits for next token
        temperature: Sampling temperature (default 0.5 per mlx-vlm)
        top_p: Nucleus sampling threshold
        previous_tokens: List of previously generated token IDs
        repetition_penalty: Penalty for repeated tokens (>1.0 discourages repetition)

    Returns:
        Sampled token ID
    """
    # CRITICAL: bfloat16 workaround for quantized models
    # bfloat16 can cause kernel loading issues with cumsum operations
    # Reference: mlx-vlm/mlx_vlm/sample_utils.py:15-18
    if logits.dtype == mx.bfloat16:
        logits = logits.astype(mx.float32)

    # Apply repetition penalty
    if previous_tokens and repetition_penalty != 1.0:
        # Convert to list for indexing (MLX doesn't support item assignment)
        logits_array = logits.tolist()
        unique_previous = set(previous_tokens)

        # Debug: print penalty info
        import os
        if os.getenv("SMLX_DEBUG") == "1":
            print(f"  [PENALTY] Applying penalty={repetition_penalty} to {len(unique_previous)} unique tokens")

        for token in unique_previous:
            original_logit = logits_array[token]
            if logits_array[token] > 0:
                logits_array[token] /= repetition_penalty
            else:
                logits_array[token] *= repetition_penalty

            # Debug: show first few penalties
            if os.getenv("SMLX_DEBUG") == "1" and len(unique_previous) <= 5:
                print(f"  [PENALTY] Token {token}: {original_logit:.4f} → {logits_array[token]:.4f}")

        logits = mx.array(logits_array)

    if temperature == 0:
        # Greedy sampling
        return int(mx.argmax(logits, axis=-1))

    # Convert to log probabilities (critical for numerical stability!)
    # This matches mlx-lm's approach and avoids precision loss from softmax → log
    logprobs = logits - mx.logsumexp(logits, keepdims=True)

    # Apply temperature
    logprobs = logprobs / temperature

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        # Convert to probabilities for top-p filtering
        probs = mx.exp(logprobs)

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

        # Keep only top-p tokens (in logprob space)
        top_indices = sorted_indices[:cutoff_idx]
        top_logprobs = logprobs[top_indices]

        # Renormalize (in log space)
        top_logprobs = top_logprobs - mx.logsumexp(top_logprobs)

        # Sample from top-p
        token_idx = mx.random.categorical(top_logprobs)
        token = int(top_indices[token_idx])
    else:
        # Sample from full distribution
        token = int(mx.random.categorical(logprobs))

    return token


def generate(
    model: NanoVLM,
    processor: Processor,
    prompt: str,
    image: Optional[Union[str, Image.Image]] = None,
    max_tokens: int = 128,
    temperature: float = 0.5,
    top_p: float = 1.0,
    validate_output: bool = False,
    max_repetition_ratio: float = 0.6,
    check_gibberish: bool = True,
    retry_on_failure: bool = False,
    max_retries: int = 2,
    min_length: int = 5,
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
        validate_output: Enable output validation (gibberish/repetition detection)
        max_repetition_ratio: Maximum allowed repetition ratio
        check_gibberish: Check for gibberish patterns
        retry_on_failure: Retry generation if validation fails
        max_retries: Maximum number of retries
        min_length: Minimum output length

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
        >>>
        >>> # With validation
        >>> text = generate(
        ...     model, processor,
        ...     prompt="Describe this image:",
        ...     image=image,
        ...     validate_output=True,
        ...     retry_on_failure=True
        ... )
    """
    # Internal generation function for retry logic
    def _generate_internal(current_temperature: float) -> str:
        # Prepare inputs
        inputs = prepare_inputs(processor, prompt, image)

        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        image_token_mask = inputs.get("image_token_mask")

        # One KV cache per language-model layer. The cache lets us run the full
        # prompt once (prefill) and then feed a single new token per decode step,
        # turning generation from O(N^3) (full re-forward each step) into O(N^2).
        # It also fixes image conditioning: the vision-augmented keys/values for
        # the image-token positions are computed during prefill and reused on
        # every decode step, so the image conditions ALL generated tokens. (The
        # previous full-re-forward loop cleared pixel_values after the first
        # token while re-feeding the image markers as plain text, silently
        # dropping the image after token 0.)
        cache = make_cache(len(model.language_model.model.layers))

        # Prefill: encode the entire prompt (and image) in a single forward.
        logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_mask=image_token_mask,
            cache=cache,
        )
        next_token_logits = logits[0, -1, :]
        # Evaluate to prevent computation graph accumulation
        mx.eval(next_token_logits)

        # Decode: one forward per new token, reusing the cache.
        generated_tokens = []
        for _ in range(max_tokens):
            # Sample next token from the most recent logits
            next_token = sample(next_token_logits, current_temperature, top_p)

            # Check for EOS
            if next_token == processor.tokenizer.eos_token_id:
                break

            # Add to generated tokens
            generated_tokens.append(next_token)

            # Stop before an unnecessary final decode once the cap is reached.
            if len(generated_tokens) >= max_tokens:
                break

            # Decode step: feed only the new token; the cache holds all past
            # keys/values (text + vision). pixel_values is omitted because the
            # image is already encoded in the cache.
            next_token_array = mx.array([[next_token]])
            logits = model(input_ids=next_token_array, cache=cache)
            next_token_logits = logits[0, -1, :]
            # Evaluate to prevent compute graph accumulation
            mx.eval(next_token_logits)

        # Decode generated tokens
        generated_text = processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return generated_text

    # Retry loop with validation
    current_temperature = temperature
    for attempt in range(max(1, max_retries + 1 if retry_on_failure else 1)):
        generated_text = _generate_internal(current_temperature)

        # Validate output if enabled
        if validate_output:
            is_valid, reason = validate_text_output(
                generated_text,
                min_length=min_length,
                max_repetition_ratio=max_repetition_ratio,
                check_gibberish=check_gibberish,
            )

            if not is_valid:
                logger.warning(f"nanoVLM output validation failed: {reason}")

                if retry_on_failure and attempt < max_retries:
                    logger.info(f"Retrying generation (attempt {attempt + 2}/{max_retries + 1})...")
                    # Adjust temperature for retry (decrease for more deterministic output)
                    current_temperature = max(0.1, current_temperature * 0.8)
                    continue

        # Success or max retries reached
        break

    return generated_text


def stream_generate(
    model: NanoVLM,
    processor: Processor,
    prompt: str,
    image: Optional[Union[str, Image.Image]] = None,
    max_tokens: int = 128,
    temperature: float = 0.5,
    top_p: float = 1.0,
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

    # One KV cache per language-model layer (see generate() for the full
    # rationale): prefill the prompt once, then decode single tokens reusing the
    # cache so generation is linear and stays image-conditioned for every token.
    cache = make_cache(len(model.language_model.model.layers))

    # Prefill: encode the entire prompt (and image) in a single forward.
    logits = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_token_mask=image_token_mask,
        cache=cache,
    )
    next_token_logits = logits[0, -1, :]
    # Evaluate to prevent computation graph accumulation
    mx.eval(next_token_logits)

    # Decode: one forward per new token, reusing the cache.
    generated_count = 0
    for _ in range(max_tokens):
        # Sample
        next_token = sample(next_token_logits, temperature, top_p)

        # Check for EOS
        if next_token == processor.tokenizer.eos_token_id:
            break

        # Decode and yield
        token_text = processor.tokenizer.decode([next_token], skip_special_tokens=True)
        yield token_text

        generated_count += 1
        # Stop before an unnecessary final decode once the cap is reached.
        if generated_count >= max_tokens:
            break

        # Decode step: feed only the new token; the cache holds all past
        # keys/values (text + vision). pixel_values is omitted because the
        # image is already encoded in the cache.
        next_token_array = mx.array([[next_token]])
        logits = model(input_ids=next_token_array, cache=cache)
        next_token_logits = logits[0, -1, :]
        # Evaluate to prevent compute graph accumulation
        mx.eval(next_token_logits)


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
