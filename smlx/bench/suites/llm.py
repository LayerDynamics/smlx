"""
Benchmark suite for language models (LLMs).

Provides functions for benchmarking text generation performance,
following patterns from MLX-LM. Works universally with different model APIs.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, cast

import mlx.core as mx

from smlx.utils.memory import clear_cache, reset_peak_memory

from ..stats import ModelBenchmarkStats, create_model_stats


@dataclass
class LLMBenchmarkConfig:
    """Configuration for LLM benchmarks."""

    prompt_tokens: int = 100
    """Number of tokens in the prompt"""

    generation_tokens: int = 100
    """Number of tokens to generate"""

    batch_size: int = 1
    """Batch size"""

    num_trials: int = 1
    """Number of trials to run (default 1 for single benchmark)"""

    temperature: float = 0.0
    """Generation temperature (0.0 for greedy)"""

    top_p: float = 1.0
    """Top-p sampling parameter"""

    seed: int = 0
    """Random seed for reproducibility"""

    warmup_tokens: int = 10
    """Number of tokens to generate during warmup"""

    measure_ttft: bool = True
    """Whether to measure time to first token separately"""

    cache_type: Optional[str] = None
    """Cache type for enhanced models: 'auto', 'standard', 'rotating', 'quantized'"""

    max_kv_size: Optional[int] = None
    """Maximum KV cache size for rotating/quantized caches"""

    enable_monitoring: bool = False
    """Enable memory pressure monitoring (for enhanced cache models)"""

    quantization_bits: int = 4
    """Quantization bits for quantized cache (4 or 8)"""


def benchmark_llm(
    model: Any,
    tokenizer: Any = None,
    prompt: Union[str, list[int]] = "The quick brown fox",
    config: Optional[LLMBenchmarkConfig] = None,
    generate_fn: Optional[Callable] = None,
) -> ModelBenchmarkStats:
    """
    Universal benchmark for language models.

    This function works with various model APIs:
    1. Models with a generate() method
    2. Models with __call__ for logits + manual sampling
    3. Custom generation function via generate_fn parameter

    Args:
        model: Model to benchmark (any MLX-compatible model)
        tokenizer: Tokenizer with encode/decode methods (optional if prompt is tokens)
        prompt: Input prompt (string or token IDs)
        config: Benchmark configuration (includes cache options for enhanced models)
        generate_fn: Custom generation function (model, tokens, max_tokens, **kwargs) -> generated_tokens

    Returns:
        ModelBenchmarkStats with performance metrics

    Example:
        >>> # With mlx_lm style model
        >>> from mlx_lm import load, generate
        >>> model, tokenizer = load("mlx-community/SmolLM2-135M")
        >>> stats = benchmark_llm(model, tokenizer, generate_fn=generate)

        >>> # With custom model
        >>> stats = benchmark_llm(my_model, my_tokenizer)

        >>> # With pre-tokenized input
        >>> tokens = [1, 2, 3, 4, 5]
        >>> stats = benchmark_llm(model, prompt=tokens)

        >>> # With enhanced cache configuration
        >>> config = LLMBenchmarkConfig(
        ...     generation_tokens=200,
        ...     cache_type="quantized",
        ...     quantization_bits=4,
        ...     enable_monitoring=True
        ... )
        >>> stats = benchmark_llm(model, tokenizer, config=config)

    Note:
        Cache configuration (cache_type, max_kv_size, etc.) is intended for models
        using enhanced cache modules (SmolLM2, Moondream2, TinyLLaVA, SmolVLM).
        Standard models will ignore these parameters.
    """
    if config is None:
        config = LLMBenchmarkConfig()

    # Set seed for reproducibility
    mx.random.seed(config.seed)

    # Tokenize prompt if string
    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("Tokenizer required when prompt is a string")
        prompt_tokens = _encode(tokenizer, prompt)
    else:
        prompt_tokens = list(prompt)

    num_prompt_tokens = len(prompt_tokens)

    # Determine generation function
    if generate_fn is None:
        generate_fn = _auto_detect_generate_fn(model)

    # Warmup
    _run_warmup(model, prompt_tokens, config, generate_fn)

    # Clear cache and reset memory tracking
    clear_cache()
    reset_peak_memory()

    # Run benchmark
    prompt_time, generation_time, num_generated_tokens = _run_benchmark(
        model, prompt_tokens, config, generate_fn
    )

    # Get peak memory
    peak_memory_gb = mx.metal.get_peak_memory() / 1e9 if mx.metal.is_available() else 0.0

    # Get model name
    model_name = _get_model_name(model)

    # Get quantization info if available
    quantization = _detect_quantization(model)

    # Create statistics
    stats = create_model_stats(
        model_name=model_name,
        prompt_tokens=num_prompt_tokens,
        prompt_time=prompt_time,
        generation_tokens=num_generated_tokens,
        generation_time=generation_time,
        peak_memory_gb=peak_memory_gb,
        quantization=quantization,
        batch_size=config.batch_size,
        temperature=config.temperature,
        seed=config.seed,
    )

    return stats


def _run_warmup(
    model: Any,
    prompt_tokens: list[int],
    config: LLMBenchmarkConfig,
    generate_fn: Callable,
):
    """Run warmup generation."""
    try:
        _ = generate_fn(
            model,
            prompt_tokens,
            max_tokens=config.warmup_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        # Ensure evaluation
        mx.eval([])
    except Exception:
        # Warmup failure is not critical
        pass


def _run_benchmark(
    model: Any,
    prompt_tokens: list[int],
    config: LLMBenchmarkConfig,
    generate_fn: Callable,
) -> tuple[float, float, int]:
    """
    Run the actual benchmark.

    Returns:
        Tuple of (prompt_time, generation_time, num_generated_tokens)
    """
    # Convert to MLX array if needed
    if not isinstance(prompt_tokens, mx.array):
        tokens = mx.array(prompt_tokens)
    else:
        tokens = prompt_tokens

    # Measure prompt processing (prefill)
    prompt_start = time.perf_counter()

    # First forward pass to process prompt
    if callable(model):
        try:
            # Run model on prompt to fill KV cache
            logits = model(tokens[:-1])
            mx.eval(logits)
        except Exception:
            # Some models may not support this, skip prompt timing
            pass

    prompt_end = time.perf_counter()
    prompt_time = prompt_end - prompt_start

    # If we couldn't measure prompt time, we'll estimate it from total time
    measure_prompt_separately = prompt_time > 0.0

    # Measure generation
    gen_start = time.perf_counter()

    generated_tokens = generate_fn(
        model,
        prompt_tokens,
        max_tokens=config.generation_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )

    # Ensure evaluation
    if isinstance(generated_tokens, mx.array):
        mx.eval(generated_tokens)
    elif isinstance(generated_tokens, list):
        for token in generated_tokens:
            if isinstance(token, mx.array):
                mx.eval(token)

    gen_end = time.perf_counter()
    total_time = gen_end - gen_start

    # Calculate actual generation time
    if measure_prompt_separately:
        generation_time = total_time  # Already measured prompt separately
    else:
        # Estimate: assume prompt is proportional to token count
        # This is a rough approximation
        total_tokens = (
            len(generated_tokens)
            if isinstance(generated_tokens, list)
            else int(generated_tokens.size)
        )
        estimated_prompt_tokens = len(prompt_tokens)
        estimated_gen_tokens = total_tokens - estimated_prompt_tokens

        if estimated_gen_tokens > 0:
            # Estimate prompt took proportional time
            prompt_time = total_time * (estimated_prompt_tokens / total_tokens)
            generation_time = total_time - prompt_time
        else:
            prompt_time = total_time
            generation_time = 0.0

    # Count generated tokens
    if isinstance(generated_tokens, list):
        num_generated = len(generated_tokens) - len(prompt_tokens)
    elif isinstance(generated_tokens, mx.array):
        num_generated = int(generated_tokens.size) - len(prompt_tokens)
    else:
        num_generated = config.generation_tokens  # Fallback estimate

    return prompt_time, generation_time, max(num_generated, 1)


def _auto_detect_generate_fn(model: Any) -> Callable:
    """
    Auto-detect the appropriate generation function for the model.

    Returns:
        Generation function that matches signature:
        fn(model, tokens, max_tokens, **kwargs) -> generated_tokens
    """
    # Check if model has a generate method
    if hasattr(model, "generate"):
        return _use_model_generate

    # Check if model is callable (for manual generation loop)
    if callable(model):
        return _manual_generate_loop

    # Fallback: assume model has generate
    return _use_model_generate


def _use_model_generate(
    model: Any,
    prompt_tokens: Union[list[int], mx.array],
    max_tokens: int = 100,
    **kwargs,
) -> Union[list[int], mx.array]:
    """Use model's generate() method."""
    if not isinstance(prompt_tokens, mx.array):
        prompt_tokens = mx.array(prompt_tokens)

    return model.generate(prompt_tokens, max_tokens=max_tokens, **kwargs)


def _manual_generate_loop(
    model: Any,
    prompt_tokens: list[int],
    max_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
    **kwargs,
) -> list[int]:
    """
    Manual generation loop for models without generate() method.

    This implements autoregressive generation manually.
    """
    if not isinstance(prompt_tokens, list):
        if isinstance(prompt_tokens, mx.array):
            prompt_tokens = prompt_tokens.tolist()
        else:
            prompt_tokens = list(prompt_tokens)

    tokens = prompt_tokens.copy()

    for _ in range(max_tokens):
        # Get logits from model
        input_tokens = mx.array(tokens)
        logits = model(input_tokens)

        # Get logits for last token
        if len(logits.shape) == 3:  # (batch, seq, vocab)
            logits = logits[0, -1, :]
        elif len(logits.shape) == 2:  # (seq, vocab)
            logits = logits[-1, :]

        # Sample next token
        if temperature == 0.0:
            # Greedy sampling
            next_token = mx.argmax(logits).item()
        else:
            # Temperature sampling
            logits = logits / temperature

            # Top-p sampling if specified
            if top_p < 1.0:
                # Sort logits
                sorted_indices = mx.argsort(logits)[::-1]
                sorted_logits = logits[sorted_indices]

                # Compute cumulative probabilities
                probs = mx.softmax(sorted_logits)
                cumsum = mx.cumsum(probs)

                # Find cutoff
                cutoff_idx = mx.argmax(cumsum >= top_p).item()
                cutoff_idx = max(1, cutoff_idx)  # Keep at least one token

                # Zero out logits beyond cutoff
                top_indices = sorted_indices[:cutoff_idx]
                filtered_logits = mx.full(logits.shape, -float("inf"))
                filtered_logits[top_indices] = logits[top_indices]
                logits = filtered_logits

            # Sample
            probs = mx.softmax(logits)
            next_token = mx.random.categorical(mx.log(probs)).item()

        tokens.append(next_token)

        # Check for EOS token (common values: 0, 1, 2)
        # In practice, you'd get this from the tokenizer
        # For now, we'll just generate max_tokens
        # if next_token in [0, 1, 2]:  # potential EOS tokens
        #     break

    return tokens


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode text using tokenizer (supports various APIs)."""
    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(text)
        # Handle different return types
        if isinstance(result, list):
            return result
        elif isinstance(result, mx.array):
            list_or_scalar = result.tolist()
            # tolist() can return list or scalar; we handle both cases
            if isinstance(list_or_scalar, list):
                return cast(list[int], list_or_scalar)
            else:
                return [int(list_or_scalar)]
        elif hasattr(result, "tolist"):
            list_or_scalar = result.tolist()  # type: ignore[attr-defined]
            # tolist() can return list or scalar; we handle both cases
            if isinstance(list_or_scalar, list):
                return cast(list[int], list_or_scalar)
            else:
                return [int(list_or_scalar)]
        else:
            return list(result)  # type: ignore[arg-type]
    elif callable(tokenizer):
        result = tokenizer(text)
        # Ensure we return a list
        if isinstance(result, list):
            return result
        elif isinstance(result, mx.array):
            list_or_scalar = result.tolist()
            # tolist() can return list or scalar; we handle both cases
            if isinstance(list_or_scalar, list):
                return cast(list[int], list_or_scalar)
            else:
                return [int(list_or_scalar)]
        elif hasattr(result, "tolist"):
            list_or_scalar = result.tolist()  # type: ignore[attr-defined]
            # tolist() can return list or scalar; we handle both cases
            if isinstance(list_or_scalar, list):
                return cast(list[int], list_or_scalar)
            else:
                return [int(list_or_scalar)]
        else:
            return list(result)  # type: ignore[arg-type]
    else:
        raise ValueError("Tokenizer must have encode() method or be callable")


def _get_model_name(model: Any) -> str:
    """Get model name from various model types."""
    if hasattr(model, "name"):
        return model.name
    elif hasattr(model, "model_type"):
        return model.model_type
    elif hasattr(model, "config") and hasattr(model.config, "model_type"):
        return model.config.model_type
    elif hasattr(model, "__class__"):
        return model.__class__.__name__
    else:
        return "unknown"


def _detect_quantization(model: Any) -> Optional[str]:
    """Detect quantization method from model."""
    # Check for common quantization indicators
    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "quantization"):
            return str(config.quantization)
        if hasattr(config, "bits"):
            return f"{config.bits}bit"

    # Check model attributes
    if hasattr(model, "quantization"):
        return str(model.quantization)

    # Check for quantized layers
    if hasattr(model, "layers") or hasattr(model, "model"):
        layers = getattr(model, "layers", None) or getattr(model, "model", None)
        if layers is not None:
            # Check first layer for quantization
            if hasattr(layers, "__iter__"):
                try:
                    first_layer = next(iter(layers))
                    if hasattr(first_layer, "bits"):
                        return f"{first_layer.bits}bit"
                    if "quantized" in str(type(first_layer)).lower():
                        return "quantized"
                except (StopIteration, TypeError):
                    pass

    return None


def benchmark_llm_batch(
    model: Any,
    tokenizer: Any = None,
    prompts: Optional[list[Union[str, list[int]]]] = None,
    config: Optional[LLMBenchmarkConfig] = None,
    generate_fn: Optional[Callable] = None,
) -> list[ModelBenchmarkStats]:
    """
    Benchmark a language model with multiple prompts.

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer (optional if prompts are pre-tokenized)
        prompts: list of input prompts (strings or token lists)
        config: Benchmark configuration
        generate_fn: Custom generation function

    Returns:
        list of ModelBenchmarkStats for each prompt

    Example:
        >>> prompts = ["Hello", "The quick brown fox", "Once upon a time"]
        >>> stats_list = benchmark_llm_batch(model, tokenizer, prompts)
        >>> avg_tps = sum(s.generation_tps for s in stats_list) / len(stats_list)
    """
    if prompts is None:
        prompts = ["The quick brown fox", "Once upon a time", "In a galaxy far, far away"]

    results = []
    for prompt in prompts:
        stats = benchmark_llm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            generate_fn=generate_fn,
        )
        results.append(stats)

    return results


def benchmark_llm_streaming(
    model: Any,
    tokenizer: Any = None,
    prompt: Union[str, list[int]] = "The quick brown fox",
    config: Optional[LLMBenchmarkConfig] = None,
    generate_fn: Optional[Callable] = None,
) -> tuple[ModelBenchmarkStats, list[float]]:
    """
    Benchmark with per-token timing (streaming).

    Returns:
        Tuple of (ModelBenchmarkStats, per_token_times)
        per_token_times is a list of time (ms) for each generated token

    Example:
        >>> stats, token_times = benchmark_llm_streaming(model, tokenizer)
        >>> print(f"TTFT: {token_times[0]:.2f}ms")
        >>> print(f"Avg per-token: {sum(token_times[1:]) / len(token_times[1:]):.2f}ms")
    """
    # This would require a streaming-aware implementation
    # For now, return regular stats with empty timing list
    stats = benchmark_llm(model, tokenizer, prompt, config, generate_fn)
    return stats, []
