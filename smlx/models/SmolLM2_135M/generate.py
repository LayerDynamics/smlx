# Copyright © 2025 SMLX Project

"""
Text generation utilities for SmolLM2-135M-Instruct.

This module now uses the shared generation implementations from smlx.utils.
It provides convenient wrappers that maintain backward compatibility.
"""

from collections.abc import Generator
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# Import generation utilities from utils
from smlx.utils.generation import (
    GenerationConfig,
    chat as utils_chat,
    complete as utils_complete,
    generate as utils_generate,
    generate_step as utils_generate_step,
    stream_generate as utils_stream_generate,
)
from smlx.utils.sampling import sample as utils_sample

# Re-export GenerationConfig for backward compatibility
__all__ = [
    "GenerationConfig",
    "sample",
    "generate_step",
    "generate",
    "stream_generate",
    "chat",
    "complete",
]


def sample(
    logits: mx.array,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> mx.array:
    """
    Sample a token from logits using temperature, top-p, and top-k sampling.

    Wrapper around smlx.utils.sampling.sample for backward compatibility.

    Args:
        logits: Logits from the model [vocab_size]
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold

    Returns:
        Sampled token ID

    Example:
        >>> logits = model(tokens)
        >>> next_token = sample(logits, temperature=0.7, top_p=0.9)
    """
    return utils_sample(logits, temperature, top_p, top_k)


def generate_step(
    model: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    cache: Optional[list] = None,
    max_kv_size: Optional[int] = None,
) -> Generator[tuple[mx.array, mx.array], None, None]:
    """
    Generate tokens one at a time (generator).

    Wrapper around smlx.utils.generation.generate_step for backward compatibility.

    Args:
        model: The model to generate from
        prompt_tokens: Input token IDs [seq_len]
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        cache: Optional KV cache (will be created if None)
        max_kv_size: Optional maximum KV cache size

    Yields:
        Tuple of (token_id, logits)

    Example:
        >>> for token_id, logits in generate_step(model, prompt_tokens):
        ...     print(tokenizer.decode([int(token_id.item())]))
    """
    # Use utils generate_step
    for token, logits in utils_generate_step(
        model=model,
        prompt_tokens=prompt_tokens,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        cache=cache,
        max_kv_size=max_kv_size,
    ):
        yield token, logits


def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    stop_strings: Optional[list[str]] = None,
    verbose: bool = False,
) -> str:
    """
    Generate text from a prompt.

    Wrapper around smlx.utils.generation.generate for backward compatibility.

    Args:
        model: The model to generate from
        tokenizer: HuggingFace tokenizer
        prompt: Input text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        stop_strings: Optional list of strings that stop generation
        verbose: Print generation progress

    Returns:
        Generated text (excluding prompt)

    Example:
        >>> model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        >>> response = generate(
        ...     model, tokenizer,
        ...     prompt="Write a Python function to",
        ...     max_tokens=100,
        ...     temperature=0.7
        ... )
    """
    return utils_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_strings=stop_strings,
        verbose=verbose,
    )


def stream_generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    stop_strings: Optional[list[str]] = None,
) -> Generator[str, None, None]:
    """
    Stream generated text token by token.

    Wrapper around smlx.utils.generation.stream_generate for backward compatibility.

    Args:
        model: The model to generate from
        tokenizer: HuggingFace tokenizer
        prompt: Input text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        stop_strings: Optional list of strings that stop generation

    Yields:
        Generated text segments

    Example:
        >>> model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        >>> for text in stream_generate(model, tokenizer, "Hello"):
        ...     print(text, end="", flush=True)
    """
    yield from utils_stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_strings=stop_strings,
    )


def chat(
    model: nn.Module,
    tokenizer,
    messages: list[dict],
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    verbose: bool = False,
) -> str:
    """
    Generate a chat response from a list of messages.

    Wrapper around smlx.utils.generation.chat for backward compatibility.

    Args:
        model: The model to generate from
        tokenizer: HuggingFace tokenizer with chat template
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        verbose: Print generation progress

    Returns:
        Generated response text

    Example:
        >>> model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        >>> messages = [
        ...     {"role": "user", "content": "Write a Python function."}
        ... ]
        >>> response = chat(model, tokenizer, messages)
    """
    return utils_chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        verbose=verbose,
    )


def complete(
    model: nn.Module,
    tokenizer,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    verbose: bool = False,
) -> str:
    """
    Complete a prompt with custom generation configuration.

    Wrapper around smlx.utils.generation.complete for backward compatibility.

    Args:
        model: The model to generate from
        tokenizer: HuggingFace tokenizer
        prompt: Input text prompt
        config: GenerationConfig instance
        verbose: Print generation progress

    Returns:
        Generated text

    Example:
        >>> config = GenerationConfig(
        ...     max_tokens=200,
        ...     temperature=0.8,
        ...     top_p=0.95,
        ...     stop_strings=["###"]
        ... )
        >>> response = complete(model, tokenizer, prompt, config)
    """
    return utils_complete(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
        verbose=verbose,
    )
