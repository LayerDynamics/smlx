# Copyright © 2025 SMLX Project
# Adapted from MLX framework reference implementations

"""
Token sampling utilities for text generation.

Provides various sampling strategies including temperature, top-p (nucleus),
top-k, min-p, and repetition penalty. These utilities are used by all
language models in SMLX for controlling generation randomness and quality.
"""

from __future__ import annotations

import math
import os
from functools import partial
from typing import Callable

import mlx.core as mx

# Conditional compilation for testing
_TESTING = os.getenv("SMLX_TESTING", "0") == "1"


def _maybe_compile(func):
    """Conditionally compile function, skipping during tests."""
    if _TESTING:
        return func
    return partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)(func)


def sample(
    logits: mx.array,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> mx.array:
    """
    Sample a token from logits using temperature, top-p, and top-k sampling.

    This is a simple, easy-to-use sampling function for basic use cases.
    For more advanced control, use `make_sampler()` instead.

    Args:
        logits: Logits from the model [vocab_size]
        temperature: Sampling temperature (0 = greedy, higher = more random)
        top_p: Nucleus sampling threshold (0-1)
        top_k: Top-k sampling threshold (>0 to enable)

    Returns:
        Sampled token ID

    Example:
        >>> logits = model(tokens)
        >>> next_token = sample(logits, temperature=0.7, top_p=0.9)
    """
    # CRITICAL: bfloat16 workaround for quantized models
    # bfloat16 can cause kernel loading issues with cumsum operations
    # Reference: mlx-vlm/mlx_vlm/sample_utils.py:15-18
    if logits.dtype == mx.bfloat16:
        logits = logits.astype(mx.float32)

    # Greedy sampling (deterministic)
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)

    # Convert to log probabilities for numerical stability
    # This matches mlx-lm's approach and avoids precision loss
    logprobs = logits - mx.logsumexp(logits, keepdims=True)

    # Apply temperature
    logprobs = logprobs / temperature

    # Top-k sampling
    if top_k > 0:
        logprobs = top_k_sampling(logprobs, top_k)

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        logprobs = top_p_sampling(logprobs, top_p)

    # Sample from filtered distribution
    return mx.random.categorical(logprobs)


def sample_token(
    logits: mx.array,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> int:
    """
    Sample a token and return as Python int.

    Convenience wrapper around `sample()` that returns a Python int instead
    of an MLX array. Commonly used in generation loops.

    Args:
        logits: Logits from the model [vocab_size] or [batch, vocab_size]
        temperature: Sampling temperature (0 = greedy, higher = more random)
        top_p: Nucleus sampling threshold (0-1)

    Returns:
        Sampled token ID as Python int

    Example:
        >>> logits = model(tokens)
        >>> next_token = sample_token(logits, temperature=0.7, top_p=0.9)
        >>> # next_token is a Python int, ready to use
    """
    # Handle batch dimension if present
    if logits.ndim > 1:
        logits = logits[0]  # Take first batch

    token = sample(logits, temperature=temperature, top_p=top_p, top_k=0)
    return int(token.item())


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
) -> Callable[[mx.array], mx.array]:
    """
    Create a sampler function with chained sampling techniques.

    This allows building complex sampling strategies by combining multiple
    filtering techniques. The filters are applied in order: top-p, min-p, top-k.

    Args:
        temp: Temperature for sampling (0 = greedy/argmax)
        top_p: Nucleus sampling threshold (0-1, 0 = disabled)
        top_k: Top-k sampling threshold (>0 to enable)
        min_p: Minimum probability threshold scaled by top token (0-1)
        min_tokens_to_keep: Minimum tokens to keep in min_p sampling

    Returns:
        Sampling function that takes log probabilities and returns token IDs

    Example:
        >>> sampler = make_sampler(temp=0.7, top_p=0.9, min_p=0.05)
        >>> for step in generation_loop:
        ...     logprobs = logits - mx.logsumexp(logits, keepdims=True)
        ...     token = sampler(logprobs)
    """
    # Greedy sampling
    if temp == 0:
        return lambda x: mx.argmax(x, axis=-1)

    # Build sampling chain
    sampling_methods = []

    if top_p > 0 and top_p < 1.0:
        sampling_methods.append(lambda x: top_p_sampling(x, top_p))

    if min_p != 0.0:
        sampling_methods.append(lambda x: min_p_sampling(x, min_p, min_tokens_to_keep))

    if top_k > 0:
        sampling_methods.append(lambda x: top_k_sampling(x, top_k))

    # Apply sampling methods in chain
    def sampler(logprobs):
        for method in sampling_methods:
            logprobs = method(logprobs)
        return categorical_sampling(logprobs, temp)

    return sampler


@_maybe_compile
def top_k_sampling(logprobs: mx.array, top_k: int) -> mx.array:
    """
    Apply top-k sampling by masking all but the k highest probability tokens.

    Args:
        logprobs: Log probabilities [vocab_size]
        top_k: Number of top tokens to keep

    Returns:
        Filtered log probabilities with -inf for tokens outside top-k

    Example:
        >>> filtered_logprobs = top_k_sampling(logprobs, top_k=50)
        >>> token = mx.random.categorical(filtered_logprobs)
    """
    vocab_size = logprobs.shape[-1]

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"`top_k` must be a positive integer, got {top_k}")

    # If top_k >= vocab_size, keep all tokens (no filtering)
    if top_k >= vocab_size:
        return logprobs

    # Find the k-th largest value
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]

    # Mask all tokens outside top-k with -inf
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )

    return masked_logprobs


@_maybe_compile
def top_p_sampling(logprobs: mx.array, top_p: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling by keeping tokens with cumulative probability <= top_p.

    This dynamically adjusts the number of tokens kept based on their probability
    distribution, which is more adaptive than fixed top-k sampling.

    Args:
        logprobs: Log probabilities [vocab_size]
        top_p: Cumulative probability threshold (0-1)

    Returns:
        Filtered log probabilities with -inf for tokens outside nucleus

    Example:
        >>> filtered_logprobs = top_p_sampling(logprobs, top_p=0.9)
        >>> token = mx.random.categorical(filtered_logprobs)
    """
    # CRITICAL: bfloat16 workaround for quantized models
    # Prevents "unable to load kernel contiguous_scan_inclusive_sum_bfloat16_bfloat16" error
    # Reference: mlx-vlm/mlx_vlm/sample_utils.py:15-18
    if logprobs.dtype == mx.bfloat16:
        logprobs = logprobs.astype(mx.float32)

    # Convert log probabilities to probabilities
    probs = mx.exp(logprobs)

    # Sort in ascending order to compute cumulative sum
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Compute cumulative probabilities
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Create inverse mapping to restore original order
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )

    # Rearrange cumulative probs back to original order
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)

    # Mask tokens with cumulative probability > threshold
    return mx.where(
        cumulative_probs > 1 - top_p,
        logprobs,
        -float("inf"),
    )


@_maybe_compile
def min_p_sampling(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> mx.array:
    """
    Apply min-p sampling: keep tokens with probability >= min_p * max_probability.

    Min-p is more adaptive than top-p as it scales the threshold based on the
    confidence of the top token. When the model is very confident, the filter
    is more aggressive.

    Args:
        logprobs: Log probabilities [vocab_size]
        min_p: Minimum probability threshold scaled by top token (0-1)
        min_tokens_to_keep: Minimum number of tokens to keep

    Returns:
        Filtered log probabilities with -inf for tokens below threshold

    Example:
        >>> # Keep tokens with prob >= 0.05 * top_token_prob
        >>> filtered_logprobs = min_p_sampling(logprobs, min_p=0.05)
        >>> token = mx.random.categorical(filtered_logprobs)
    """
    if not (0 <= min_p <= 1.0):
        raise ValueError(f"`min_p` must be in [0, 1], got {min_p}")

    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(f"`min_tokens_to_keep` must be positive int, got {min_tokens_to_keep}")

    # Sort by probability (descending)
    sorted_indices = mx.argsort(-logprobs, axis=-1)
    sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)

    # Get top probability (in log space)
    top_logprob = sorted_logprobs[:, 0:1]

    # Calculate min_p threshold in log space: log(min_p * top_prob) = log(top_prob) + log(min_p)
    scaled_min_p = top_logprob + math.log(min_p)

    # Mask tokens below threshold (but keep minimum number)
    tokens_to_remove = sorted_logprobs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False

    # Apply mask
    selected_logprobs = mx.where(tokens_to_remove, -float("inf"), sorted_logprobs)

    # Rearrange back to original order
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )

    original_order_logprobs = mx.take_along_axis(selected_logprobs, inverse_indices, axis=-1)

    return original_order_logprobs


def categorical_sampling(logits: mx.array, temp: float) -> mx.array:
    """
    Sample from categorical distribution with temperature.

    Args:
        logits: Log probabilities or logits
        temp: Temperature (higher = more random)

    Returns:
        Sampled token ID
    """
    return mx.random.categorical(logits * (1 / temp))


def make_repetition_penalty(
    penalty: float,
    context_size: int = 20,
) -> Callable[[mx.array, mx.array], mx.array]:
    """
    Create repetition penalty processor for reducing repeated tokens.

    Applies a penalty to tokens that appear in recent context, making the
    model less likely to repeat itself. Based on the paper:
    https://arxiv.org/abs/1909.05858

    Args:
        penalty: Penalty factor (>1 = penalize, <1 = encourage repetition)
        context_size: Number of recent tokens to consider

    Returns:
        Processor function that takes (tokens, logits) and returns penalized logits

    Example:
        >>> penalty_fn = make_repetition_penalty(penalty=1.2, context_size=20)
        >>> logits = penalty_fn(previous_tokens, logits)
    """
    if penalty < 0 or not isinstance(penalty, (int, float)):
        raise ValueError(f"penalty must be non-negative float, got {penalty}")

    def repetition_penalty_processor(tokens: mx.array, logits: mx.array) -> mx.array:
        """
        Apply repetition penalty to logits based on previous tokens.

        Args:
            tokens: Previous token IDs [seq_len]
            logits: Current logits [batch, vocab_size]

        Returns:
            Penalized logits
        """
        if len(tokens) > 0:
            # Use only recent context
            tokens = tokens[-context_size:]

            # Get logits for tokens in context
            selected_logits = logits[:, tokens]

            # Apply penalty: increase negative logits, decrease positive logits
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
            )

            # Update logits (create new array to avoid in-place modification)
            logits = mx.array(logits)
            logits[:, tokens] = selected_logits

        return logits

    return repetition_penalty_processor


def make_logits_processors(
    logit_bias: dict[int, float] | None = None,
    repetition_penalty: float | None = None,
    repetition_context_size: int = 20,
) -> list[Callable[[mx.array, mx.array], mx.array]]:
    """
    Create a list of logits processors for generation.

    Logits processors modify the logits before sampling, allowing control
    over which tokens are more or less likely to be generated.

    Args:
        logit_bias: Dictionary mapping token IDs to bias values
        repetition_penalty: Penalty for repeating tokens (>1 = penalize)
        repetition_context_size: Number of recent tokens for repetition penalty

    Returns:
        List of processor functions, each taking (tokens, logits) -> logits

    Example:
        >>> processors = make_logits_processors(
        ...     logit_bias={50256: -100},  # Ban EOS token
        ...     repetition_penalty=1.2
        ... )
        >>> for processor in processors:
        ...     logits = processor(tokens, logits)
    """
    logits_processors = []

    # Add logit bias processor
    if logit_bias:
        indices = mx.array(list(logit_bias.keys()))
        values = mx.array(list(logit_bias.values()))

        def logit_bias_processor(_, logits):
            # Create a copy to avoid modifying the input
            logits = mx.array(logits)
            logits[:, indices] += values
            return logits

        logits_processors.append(logit_bias_processor)

    # Add repetition penalty processor
    if repetition_penalty and repetition_penalty != 0.0:
        logits_processors.append(
            make_repetition_penalty(repetition_penalty, repetition_context_size)
        )

    return logits_processors


# Alias for backwards compatibility
sample_with_temperature = sample


__all__ = [
    "sample",
    "sample_token",
    "sample_with_temperature",
    "make_sampler",
    "top_k_sampling",
    "top_p_sampling",
    "min_p_sampling",
    "categorical_sampling",
    "make_repetition_penalty",
    "make_logits_processors",
]
