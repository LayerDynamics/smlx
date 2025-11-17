# Copyright © 2025 SMLX Project
# Adapted from MLX framework reference implementations

"""
Text generation utilities for language models.

Provides high-level functions for generating text from language models,
including basic generation, streaming, and chat interfaces. These utilities
work with any MLX-based language model in SMLX.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .cache import make_cache
from .sampling import make_logits_processors, make_sampler


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    This dataclass holds all parameters for controlling text generation,
    including sampling strategies, stopping criteria, and output formatting.

    Args:
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0 = greedy, higher = more random)
        top_p: Nucleus sampling threshold (0-1, 1 = disabled)
        top_k: Top-k sampling threshold (>0 to enable, 0 = disabled)
        min_p: Minimum probability threshold scaled by top token
        min_tokens_to_keep: Minimum tokens to keep for min_p sampling
        repetition_penalty: Penalty for repeating tokens (>1 = penalize)
        repetition_context_size: Number of recent tokens for repetition penalty
        logit_bias: Dictionary mapping token IDs to bias values
        min_tokens: Minimum tokens to generate before stopping
        stop_token_ids: List of token IDs that stop generation
        stop_strings: List of strings that stop generation

    Example:
        >>> config = GenerationConfig(
        ...     max_tokens=200,
        ...     temperature=0.8,
        ...     top_p=0.95,
        ...     repetition_penalty=1.1
        ... )
    """

    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20
    logit_bias: Optional[dict[int, float]] = None
    min_tokens: int = 0
    stop_token_ids: Optional[list[int]] = None
    stop_strings: Optional[list[str]] = None


def generate_step(
    model: nn.Module,
    prompt_tokens: mx.array,
    temp: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    repetition_context_size: int = 20,
    logit_bias: Optional[dict[int, float]] = None,
    cache: Optional[list[Any]] = None,
    max_kv_size: Optional[int] = None,
) -> Generator[tuple[mx.array, mx.array], None, None]:
    """
    Generate tokens one at a time (generator function).

    This is the core generation loop that yields tokens and logits as they
    are generated. It handles KV caching, sampling, and logits processing.

    Args:
        model: The language model (must have forward method accepting cache)
        prompt_tokens: Input token IDs [seq_len]
        temp: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        min_p: Minimum probability threshold
        repetition_penalty: Penalty for repeating tokens
        repetition_context_size: Context size for repetition penalty
        logit_bias: Bias for specific tokens
        cache: Optional KV cache (will be created if None)
        max_kv_size: Optional maximum KV cache size

    Yields:
        Tuple of (token_id, logits) for each generated token

    Example:
        >>> for token, logits in generate_step(model, prompt_tokens, temp=0.7):
        ...     print(tokenizer.decode([int(token.item())]))
    """
    # Create sampler
    sampler = make_sampler(
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        min_tokens_to_keep=1,
    )

    # Create logits processors
    logits_processors = make_logits_processors(
        logit_bias=logit_bias,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
    )

    # Initialize cache if not provided
    if cache is None:
        # Assume model has num_layers attribute or use make_cache with model
        if hasattr(model, "layers"):
            num_layers = len(model.layers)
            cache = make_cache(num_layers, max_kv_size=max_kv_size)
        else:
            # Fallback: no cache
            cache = None

    # Process prompt
    y = prompt_tokens[None]  # Add batch dimension
    cache_input = cache if cache is not None else []

    # Forward pass through prompt
    if cache is not None:
        logits = model(y, cache=cache_input)
    else:
        logits = model(y)

    # Get last token logits (handle both 2D and 3D outputs)
    if logits.ndim == 3:
        logits = logits[:, -1, :]  # [batch, seq_len, vocab_size] -> [batch, vocab_size]
    # else: logits is already [batch, vocab_size]
    mx.eval(logits)  # Evaluate to prevent compute graph accumulation

    # Track generated tokens for repetition penalty
    y = prompt_tokens

    # Generation loop
    while True:
        # Apply logits processors
        for processor in logits_processors:
            logits = processor(y, logits)

        # Sample next token
        next_token = sampler(logits)
        mx.eval(next_token)  # Evaluate sampled token

        # Yield token and logits
        yield next_token, logits

        # Update token history
        y = mx.concatenate([y, next_token.reshape(1)])
        mx.eval(y)  # Evaluate concatenated token history

        # Forward pass with next token
        next_token_input = next_token.reshape(1, 1)
        if cache is not None:
            logits = model(next_token_input, cache=cache_input)
        else:
            logits = model(next_token_input)

        # Get last token logits (handle both 2D and 3D outputs)
        if logits.ndim == 3:
            logits = logits[:, -1, :]  # [batch, seq_len, vocab_size] -> [batch, vocab_size]
        # else: logits is already [batch, vocab_size]
        mx.eval(logits)  # Evaluate to prevent compute graph accumulation


def generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    stop_token_ids: Optional[list[int]] = None,
    stop_strings: Optional[list[str]] = None,
    verbose: bool = False,
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: The language model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        min_p: Minimum probability threshold
        repetition_penalty: Penalty for repeating tokens
        stop_token_ids: Token IDs that stop generation
        stop_strings: Strings that stop generation
        verbose: Print generation progress

    Returns:
        Generated text (excluding prompt)

    Example:
        >>> model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        >>> text = generate(
        ...     model, tokenizer,
        ...     prompt="Write a Python function to",
        ...     max_tokens=100,
        ...     temperature=0.7
        ... )
    """
    # Tokenize prompt
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    if verbose:
        print(f"Prompt tokens: {len(prompt_tokens)}")
        print(f"Generating up to {max_tokens} tokens...")
        print("-" * 50)

    # Default stop tokens
    if stop_token_ids is None:
        stop_token_ids = []

    # Add EOS token if available
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        stop_token_ids = stop_token_ids + [tokenizer.eos_token_id]

    # Generate tokens
    generated_tokens = []
    generated_text = ""

    for i, (token, _) in enumerate(
        generate_step(
            model=model,
            prompt_tokens=prompt_tokens,
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )
    ):
        if i >= max_tokens:
            break

        token_id = int(token.item())
        generated_tokens.append(token_id)

        # Decode incrementally
        generated_text = tokenizer.decode(generated_tokens)

        if verbose:
            # Print new text
            print(tokenizer.decode([token_id]), end="", flush=True)

        # Check for stop strings
        if stop_strings:
            for stop in stop_strings:
                if stop in generated_text:
                    generated_text = generated_text.split(stop)[0]
                    if verbose:
                        print()
                    return generated_text

        # Check for stop tokens
        if token_id in stop_token_ids:
            break

    if verbose:
        print()
        print("-" * 50)
        print(f"Generated {len(generated_tokens)} tokens")

    return generated_text


def stream_generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    stop_token_ids: Optional[list[int]] = None,
    stop_strings: Optional[list[str]] = None,
) -> Generator[str, None, None]:
    """
    Stream generated text token by token.

    This is useful for applications that need to show generation progress
    in real-time, such as chat interfaces.

    Args:
        model: The language model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        min_p: Minimum probability threshold
        repetition_penalty: Penalty for repeating tokens
        stop_token_ids: Token IDs that stop generation
        stop_strings: Strings that stop generation

    Yields:
        Generated text segments (delta from previous state)

    Example:
        >>> model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        >>> for text in stream_generate(model, tokenizer, "Hello"):
        ...     print(text, end="", flush=True)
    """
    # Tokenize prompt
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    # Default stop tokens
    if stop_token_ids is None:
        stop_token_ids = []

    # Add EOS token if available
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        stop_token_ids = stop_token_ids + [tokenizer.eos_token_id]

    # Generate tokens
    generated_tokens = []
    prev_text = ""

    for i, (token, _) in enumerate(
        generate_step(
            model=model,
            prompt_tokens=prompt_tokens,
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )
    ):
        if i >= max_tokens:
            break

        token_id = int(token.item())
        generated_tokens.append(token_id)

        # Decode and yield delta
        current_text = tokenizer.decode(generated_tokens)
        new_text = current_text[len(prev_text) :]
        prev_text = current_text

        yield new_text

        # Check for stop strings
        if stop_strings:
            for stop in stop_strings:
                if stop in current_text:
                    return

        # Check for stop tokens
        if token_id in stop_token_ids:
            return


def chat(
    model: nn.Module,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    verbose: bool = False,
    stream: bool = False,
):
    """
    Generate a chat response from a list of messages.

    Uses the tokenizer's chat template to format the conversation properly
    for instruction-tuned models.

    Args:
        model: The language model
        tokenizer: Tokenizer with chat template support
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        repetition_penalty: Penalty for repeating tokens
        verbose: Print generation progress
        stream: If True, return a generator that yields tokens. If False, return complete string.

    Returns:
        Generated response text (str) if stream=False, or generator yielding tokens if stream=True

    Example:
        >>> model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        >>> messages = [
        ...     {"role": "user", "content": "What is Python?"}
        ... ]
        >>> response = chat(model, tokenizer, messages)
        >>> # Or streaming:
        >>> for token in chat(model, tokenizer, messages, stream=True):
        ...     print(token, end="", flush=True)
    """
    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: simple formatting
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "

    # Generate response
    if stream:
        # Return streaming generator
        return stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
        )
    else:
        # Return complete string
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
        )

        return response.strip()


def complete(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    verbose: bool = False,
) -> str:
    """
    Complete a prompt using a GenerationConfig object.

    This provides a convenient way to use all generation parameters
    through a single config object.

    Args:
        model: The language model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input text prompt
        config: GenerationConfig instance (uses defaults if None)
        verbose: Print generation progress

    Returns:
        Generated text

    Example:
        >>> config = GenerationConfig(
        ...     max_tokens=200,
        ...     temperature=0.8,
        ...     top_p=0.95,
        ...     repetition_penalty=1.1,
        ...     stop_strings=["###"]
        ... )
        >>> text = complete(model, tokenizer, "Write a story", config)
    """
    if config is None:
        config = GenerationConfig()

    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=config.min_p,
        repetition_penalty=config.repetition_penalty,
        stop_token_ids=config.stop_token_ids,
        stop_strings=config.stop_strings,
        verbose=verbose,
    )


__all__ = [
    "GenerationConfig",
    "generate_step",
    "generate",
    "stream_generate",
    "chat",
    "complete",
]
