"""
Comprehensive text generation benchmark suite.

Provides functions for benchmarking various aspects of text generation:
- Context scaling (performance vs prompt length)
- Generation length (performance vs generation length)
- Temperature effects (sampling vs greedy decoding)
- Batch size effects

These benchmarks help understand model performance characteristics
across different usage patterns.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..stats import ModelBenchmarkStats
from .llm import LLMBenchmarkConfig, benchmark_llm


@dataclass
class TextGenerationBenchmarkResult:
    """Results from a text generation benchmark suite."""

    name: str
    """Name of the benchmark suite"""

    benchmarks: list[ModelBenchmarkStats] = field(default_factory=list)
    """List of individual benchmark results"""

    parameter_name: str = ""
    """Name of the parameter being varied (e.g., 'context_length', 'generation_length')"""

    parameter_values: list[Any] = field(default_factory=list)
    """Values of the parameter for each benchmark"""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "parameter_name": self.parameter_name,
            "parameter_values": self.parameter_values,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
        }

    def print_summary(self):
        """Print a summary of the benchmark results."""
        print(f"\n{'=' * 80}")
        print(f"Benchmark Suite: {self.name}")
        print(f"{'=' * 80}")

        if self.parameter_name and self.parameter_values:
            print(f"\nParameter: {self.parameter_name}")
            print(f"Values: {self.parameter_values}")

        print(f"\nResults ({len(self.benchmarks)} benchmarks):")
        print(f"{'─' * 80}")

        for i, (bench, param_val) in enumerate(zip(self.benchmarks, self.parameter_values)):
            print(f"\n{i + 1}. {self.parameter_name}={param_val}")
            print(f"   Prompt TPS: {bench.prompt_tps:.2f} tok/s")
            print(f"   Generation TPS: {bench.generation_tps:.2f} tok/s")
            print(f"   Peak Memory: {bench.peak_memory_gb:.2f} GB")
            print(f"   Total Time: {bench.total_time:.3f}s")

        print(f"\n{'=' * 80}\n")


def benchmark_single_generation(
    model: Any,
    tokenizer: Any = None,
    prompt: str = "The quick brown fox",
    config: Optional[LLMBenchmarkConfig] = None,
    generate_fn: Optional[Callable] = None,
) -> ModelBenchmarkStats:
    """
    Benchmark a single text generation run.

    This is a convenience wrapper around benchmark_llm for a single benchmark run.

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input prompt
        config: Benchmark configuration
        generate_fn: Custom generation function

    Returns:
        ModelBenchmarkStats with performance metrics

    Example:
        >>> stats = benchmark_single_generation(model, tokenizer, "Hello world")
        >>> print(f"Generation speed: {stats.generation_tps:.2f} tok/s")
    """
    return benchmark_llm(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
        generate_fn=generate_fn,
    )


def benchmark_context_scaling(
    model: Any,
    tokenizer: Any = None,
    base_prompt: str = "The quick brown fox jumps over the lazy dog. ",
    context_lengths: list[int] = None,
    generation_tokens: int = 100,
    generate_fn: Optional[Callable] = None,
) -> TextGenerationBenchmarkResult:
    """
    Benchmark how performance scales with context length (prompt tokens).

    Tests model performance with varying prompt lengths to understand
    how prefill time and memory scale with context.

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer with encode/decode methods
        base_prompt: Base prompt string (will be repeated to reach target lengths)
        context_lengths: List of context lengths to test (default: [128, 256, 512, 1024, 2048])
        generation_tokens: Number of tokens to generate for each test
        generate_fn: Custom generation function

    Returns:
        TextGenerationBenchmarkResult with scaling analysis

    Example:
        >>> result = benchmark_context_scaling(
        ...     model, tokenizer,
        ...     context_lengths=[128, 512, 2048]
        ... )
        >>> result.print_summary()
    """
    if context_lengths is None:
        context_lengths = [128, 256, 512, 1024, 2048]

    # Create prompts of target lengths
    prompts = []
    for target_length in context_lengths:
        # Encode base prompt to get token length
        base_tokens = tokenizer.encode(base_prompt)
        base_length = len(base_tokens)

        # Calculate how many repetitions needed
        num_repeats = max(1, target_length // base_length)
        repeated_prompt = base_prompt * num_repeats

        # Truncate to exact length if needed
        tokens = tokenizer.encode(repeated_prompt)[:target_length]
        prompts.append(tokens)

    # Run benchmarks
    benchmarks = []
    for prompt_tokens in prompts:
        config = LLMBenchmarkConfig(
            generation_tokens=generation_tokens,
            warmup_tokens=10,
        )

        stats = benchmark_llm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_tokens,
            config=config,
            generate_fn=generate_fn,
        )
        benchmarks.append(stats)

    return TextGenerationBenchmarkResult(
        name="Context Scaling Benchmark",
        benchmarks=benchmarks,
        parameter_name="context_length",
        parameter_values=context_lengths,
    )


def benchmark_generation_length(
    model: Any,
    tokenizer: Any = None,
    prompt: str = "The quick brown fox",
    generation_lengths: list[int] = None,
    generate_fn: Optional[Callable] = None,
) -> TextGenerationBenchmarkResult:
    """
    Benchmark how performance scales with generation length.

    Tests model throughput with varying generation lengths to understand
    autoregressive decoding performance.

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input prompt (fixed across tests)
        generation_lengths: List of generation lengths to test (default: [50, 100, 200, 500])
        generate_fn: Custom generation function

    Returns:
        TextGenerationBenchmarkResult with scaling analysis

    Example:
        >>> result = benchmark_generation_length(
        ...     model, tokenizer,
        ...     generation_lengths=[50, 200, 500]
        ... )
        >>> result.print_summary()
    """
    if generation_lengths is None:
        generation_lengths = [50, 100, 200, 500]

    benchmarks = []
    for gen_length in generation_lengths:
        config = LLMBenchmarkConfig(
            generation_tokens=gen_length,
            warmup_tokens=10,
        )

        stats = benchmark_llm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            generate_fn=generate_fn,
        )
        benchmarks.append(stats)

    return TextGenerationBenchmarkResult(
        name="Generation Length Benchmark",
        benchmarks=benchmarks,
        parameter_name="generation_length",
        parameter_values=generation_lengths,
    )


def benchmark_temperature_effects(
    model: Any,
    tokenizer: Any = None,
    prompt: str = "The quick brown fox",
    temperatures: list[float] = None,
    generation_tokens: int = 100,
    generate_fn: Optional[Callable] = None,
) -> TextGenerationBenchmarkResult:
    """
    Benchmark impact of temperature on generation performance.

    Tests how sampling (temperature > 0) affects throughput compared to
    greedy decoding (temperature = 0).

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input prompt
        temperatures: List of temperature values to test (default: [0.0, 0.3, 0.7, 1.0])
        generation_tokens: Number of tokens to generate
        generate_fn: Custom generation function

    Returns:
        TextGenerationBenchmarkResult with temperature comparison

    Example:
        >>> result = benchmark_temperature_effects(
        ...     model, tokenizer,
        ...     temperatures=[0.0, 0.5, 1.0]
        ... )
        >>> result.print_summary()
    """
    if temperatures is None:
        temperatures = [0.0, 0.3, 0.7, 1.0]

    benchmarks = []
    for temp in temperatures:
        config = LLMBenchmarkConfig(
            generation_tokens=generation_tokens,
            temperature=temp,
            warmup_tokens=10,
        )

        stats = benchmark_llm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            generate_fn=generate_fn,
        )
        benchmarks.append(stats)

    return TextGenerationBenchmarkResult(
        name="Temperature Effects Benchmark",
        benchmarks=benchmarks,
        parameter_name="temperature",
        parameter_values=temperatures,
    )


def benchmark_batch_size_scaling(
    model: Any,
    tokenizer: Any = None,
    prompt: str = "The quick brown fox",
    batch_sizes: list[int] = None,
    generation_tokens: int = 100,
    generate_fn: Optional[Callable] = None,
) -> TextGenerationBenchmarkResult:
    """
    Benchmark how performance scales with batch size.

    Tests throughput with varying batch sizes to understand batching efficiency.
    Note: Not all models/generation functions support batching.

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input prompt (same for all batch items)
        batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8])
        generation_tokens: Number of tokens to generate per item
        generate_fn: Custom generation function (must support batching)

    Returns:
        TextGenerationBenchmarkResult with batch scaling analysis

    Example:
        >>> result = benchmark_batch_size_scaling(
        ...     model, tokenizer,
        ...     batch_sizes=[1, 4, 8]
        ... )
        >>> result.print_summary()
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    benchmarks = []
    for batch_size in batch_sizes:
        config = LLMBenchmarkConfig(
            generation_tokens=generation_tokens,
            batch_size=batch_size,
            warmup_tokens=10,
        )

        stats = benchmark_llm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            generate_fn=generate_fn,
        )
        benchmarks.append(stats)

    return TextGenerationBenchmarkResult(
        name="Batch Size Scaling Benchmark",
        benchmarks=benchmarks,
        parameter_name="batch_size",
        parameter_values=batch_sizes,
    )


def run_comprehensive_suite(
    model: Any,
    tokenizer: Any = None,
    prompt: str = "The quick brown fox",
    generate_fn: Optional[Callable] = None,
    include_context_scaling: bool = True,
    include_generation_length: bool = True,
    include_temperature_effects: bool = True,
    include_batch_scaling: bool = False,
) -> dict[str, TextGenerationBenchmarkResult]:
    """
    Run a comprehensive suite of text generation benchmarks.

    This function runs multiple benchmark suites to provide a complete
    performance profile of the model.

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer with encode/decode methods
        prompt: Base input prompt
        generate_fn: Custom generation function
        include_context_scaling: Run context scaling benchmark
        include_generation_length: Run generation length benchmark
        include_temperature_effects: Run temperature benchmark
        include_batch_scaling: Run batch size scaling benchmark (requires batching support)

    Returns:
        Dictionary mapping benchmark names to results

    Example:
        >>> results = run_comprehensive_suite(model, tokenizer)
        >>> for name, result in results.items():
        ...     result.print_summary()
        >>> # Or access specific results:
        >>> ctx_result = results['context_scaling']
        >>> print(f"Max context TPS: {max(b.generation_tps for b in ctx_result.benchmarks):.2f}")
    """
    print("=" * 80)
    print("Running Comprehensive Text Generation Benchmark Suite")
    print("=" * 80)

    results = {}

    if include_context_scaling:
        print("\n1. Context Scaling Benchmark...")
        results["context_scaling"] = benchmark_context_scaling(
            model=model,
            tokenizer=tokenizer,
            generate_fn=generate_fn,
        )
        results["context_scaling"].print_summary()

    if include_generation_length:
        print("\n2. Generation Length Benchmark...")
        results["generation_length"] = benchmark_generation_length(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generate_fn=generate_fn,
        )
        results["generation_length"].print_summary()

    if include_temperature_effects:
        print("\n3. Temperature Effects Benchmark...")
        results["temperature_effects"] = benchmark_temperature_effects(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generate_fn=generate_fn,
        )
        results["temperature_effects"].print_summary()

    if include_batch_scaling:
        print("\n4. Batch Size Scaling Benchmark...")
        try:
            results["batch_scaling"] = benchmark_batch_size_scaling(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                generate_fn=generate_fn,
            )
            results["batch_scaling"].print_summary()
        except Exception as e:
            print(f"   Batch scaling benchmark failed: {e}")
            print("   (This is normal if the model doesn't support batching)")

    print("\n" + "=" * 80)
    print("Comprehensive Suite Complete")
    print("=" * 80)

    return results
