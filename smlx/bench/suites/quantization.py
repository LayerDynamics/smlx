"""
Quantization Comparison Benchmark Suite

Compare different quantization methods for MLX models:
- FP16 (baseline)
- 8-bit quantization (int8, mxfp8)
- 4-bit quantization (4bit, mxfp4)
- GPTQ, AWQ, DWQ methods
- FP8 formats (mxfp8 recommended, fp8 deprecated)

Measures:
- Speed (tokens/sec) - prompt and generation
- Memory usage (peak GB)
- Model size (GB)
- Accuracy (perplexity or eval scores)
- Hardware acceleration (M4-specific)

Note: This benchmarks model weight quantization. For KV cache quantization
(which reduces memory usage during generation), see the cache configuration
options in LLMBenchmarkConfig (cache_type="quantized", quantization_bits=4/8).

Usage:
    from smlx.bench.suites.quantization import compare_quantization_methods, compare_mxfp8_vs_int8

    # Compare multiple methods
    results = compare_quantization_methods(
        model_path="mlx-community/SmolLM2-135M-Instruct",
        quantization_methods=["fp16", "mxfp8", "int8", "4bit"]
    )

    # Detailed MXFP8 vs INT8 comparison for M4
    comparison = compare_mxfp8_vs_int8(
        model_path="mlx-community/SmolLM2-135M-Instruct",
        verbose=True
    )

CLI Usage:
    # General comparison
    python -m smlx.bench.suites.quantization \\
        --model mlx-community/SmolLM2-135M-Instruct \\
        --methods fp16 mxfp8 int8 4bit \\
        --output quantization_comparison.json

    # MXFP8 vs INT8 (recommended for M4)
    python -m smlx.bench.suites.quantization \\
        --model mlx-community/SmolLM2-135M-Instruct \\
        --methods fp16 mxfp8 int8
"""

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from smlx.utils import (
    clear_cache,
    format_memory,
    memory_profiler,
    reset_peak_memory,
    save_json,
)
from smlx.utils.quantization import apply_quantization


@dataclass
class QuantizationBenchmarkResult:
    """Results from quantization benchmark."""

    quantization_method: str
    model_name: str

    # Size metrics
    model_size_gb: float
    memory_reduction_percent: float  # vs FP16

    # Speed metrics (tokens/sec)
    prompt_tps: float
    generation_tps: float

    # Speed improvement (vs FP16)
    prompt_speedup: float
    generation_speedup: float

    # Memory metrics
    peak_memory_gb: float
    memory_savings_percent: float  # vs FP16

    # Quality metrics (optional)
    perplexity: Optional[float] = None
    perplexity_degradation: Optional[float] = None  # vs FP16

    # Benchmark configuration
    prompt_tokens: int = 0
    generation_tokens: int = 0
    test_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def estimate_model_size(model, quantization: str = "fp16") -> float:
    """
    Estimate model size in GB based on parameter count and quantization.

    Args:
        model: Model to estimate
        quantization: Quantization method

    Returns:
        Estimated size in GB
    """
    # Count parameters robustly over the (possibly nested) parameter tree.
    # tree_flatten yields every leaf array regardless of dict/list nesting.
    from mlx.utils import tree_flatten

    try:
        total_params = sum(int(v.size) for _, v in tree_flatten(model.parameters()))
    except Exception as e:
        # Do NOT fabricate a model size here — a wrong parameter count silently
        # corrupts every downstream memory metric. Fail loudly instead.
        raise ValueError(
            f"Could not count model parameters for size estimation: {e}"
        ) from e

    # Bytes per parameter based on quantization
    bytes_per_param = {
        "fp16": 2,      # 16-bit = 2 bytes
        "8bit": 1,      # 8-bit = 1 byte
        "4bit": 0.5,    # 4-bit = 0.5 bytes
        "gptq": 0.5,    # GPTQ typically 4-bit
        "awq": 0.5,     # AWQ typically 4-bit
        "dwq": 1,       # DWQ typically 8-bit
        "mxfp8": 1.03,  # MXFP8: 1 byte/element + 1 byte scale/32 elements
        "mxfp4": 0.53,  # MXFP4: 0.5 byte/element + 1 byte scale/32 elements
        "int8": 1.03,   # INT8: 1 byte/element + 4 byte scale/group
        "fp8": 2,       # Simulated FP8 (deprecated) - stored as float16
    }

    bytes_per = bytes_per_param.get(quantization, 2)
    size_bytes = total_params * bytes_per
    size_gb = size_bytes / (1024**3)

    return size_gb


def benchmark_quantized_model(
    model,
    tokenizer,
    quantization_method: str,
    test_prompt: str = "The quick brown fox jumps over the lazy dog",
    generation_tokens: int = 100,
    verbose: bool = False,
) -> QuantizationBenchmarkResult:
    """
    Benchmark a single quantization configuration.

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        quantization_method: Name of quantization method
        test_prompt: Test prompt
        generation_tokens: Number of tokens to generate
        verbose: Print progress

    Returns:
        QuantizationBenchmarkResult with metrics
    """
    from mlx_lm import generate as lm_generate
    from mlx_lm import stream_generate as lm_stream_generate
    from mlx_lm.sample_utils import make_sampler

    model_name = getattr(model, "model_type", "unknown")

    if verbose:
        print(f"\nBenchmarking {quantization_method}...")

    # Estimate model size
    model_size = estimate_model_size(model, quantization_method)

    # Clear cache
    clear_cache()
    reset_peak_memory()

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(test_prompt)
    num_prompt_tokens = len(prompt_tokens)

    # Warmup
    _ = lm_generate(
        model,
        tokenizer,
        test_prompt,
        max_tokens=10,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )

    clear_cache()
    reset_peak_memory()

    # Benchmark generation with a real prefill/decode split.
    #
    # Stream tokens one at a time and timestamp each. The first token's arrival
    # marks the end of prompt prefill (prefill processes all num_prompt_tokens),
    # so:
    #   prompt_time     = time from start to first token  (prefill)
    #   generation_time = time spent emitting the remaining tokens (decode)
    # No hardcoded ratios — both phases are measured directly.
    token_pieces: list[str] = []
    token_times: list[float] = []
    with memory_profiler() as mem:
        start_time = time.perf_counter()
        for resp in lm_stream_generate(
            model,
            tokenizer,
            test_prompt,
            max_tokens=generation_tokens,
            sampler=make_sampler(temp=0.0),
        ):
            token_times.append(time.perf_counter())
            token_pieces.append(resp.text)
        end_time = time.perf_counter()

    total_time = end_time - start_time
    num_generated = len(token_pieces)

    if num_generated >= 1:
        # Prefill = start -> first token.
        prompt_time = token_times[0] - start_time
        # Decode = first token -> last token (time for the remaining tokens).
        generation_time = token_times[-1] - token_times[0]
        decode_tokens = num_generated - 1
    else:
        # No tokens produced (e.g. immediate stop): attribute all time to prefill.
        prompt_time = total_time
        generation_time = 0.0
        decode_tokens = 0

    # Calculate throughput from the measured phases.
    prompt_tps = num_prompt_tokens / prompt_time if prompt_time > 0 else 0
    generation_tps = decode_tokens / generation_time if generation_time > 0 else 0

    result = QuantizationBenchmarkResult(
        quantization_method=quantization_method,
        model_name=model_name,
        model_size_gb=model_size,
        memory_reduction_percent=0.0,  # Will be calculated later
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        prompt_speedup=1.0,  # Will be calculated later
        generation_speedup=1.0,  # Will be calculated later
        peak_memory_gb=mem.peak_gb,
        memory_savings_percent=0.0,  # Will be calculated later
        prompt_tokens=num_prompt_tokens,
        generation_tokens=num_generated,
        test_prompt=test_prompt,
    )

    if verbose:
        print(f"  Model size: {format_memory(model_size * 1e9)}")
        print(f"  Generation speed: {generation_tps:.0f} tokens/sec")
        print(f"  Peak memory: {format_memory(mem.peak_gb * 1e9)}")

    return result


def compare_quantization_methods(
    model_path: str = "mlx-community/SmolLM2-135M-Instruct",
    quantization_methods: list[str] | None = None,
    test_prompt: str = "The quick brown fox jumps over the lazy dog",
    generation_tokens: int = 100,
    verbose: bool = True,
) -> dict[str, QuantizationBenchmarkResult]:
    """
    Compare multiple quantization methods.

    Args:
        model_path: Path to model or HuggingFace ID
        quantization_methods: List of methods to compare
        test_prompt: Test prompt
        generation_tokens: Tokens to generate
        verbose: Print progress

    Returns:
        Dictionary mapping method names to results
    """
    if quantization_methods is None:
        quantization_methods = ["fp16"]

    from smlx.models import mlx_backend

    if verbose:
        print("=" * 70)
        print("QUANTIZATION COMPARISON BENCHMARK")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Methods: {', '.join(quantization_methods)}")

    results = {}
    baseline_result = None

    for method in quantization_methods:
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Testing: {method.upper()}")
            print('=' * 70)

        # Load model
        if verbose:
            print("Loading model...")

        bm = mlx_backend.load(model_path)
        model, tokenizer = bm.model, bm.processor

        # Apply quantization
        if verbose:
            print(f"Applying {method} quantization...")

        model = apply_quantization(model, method=method, verbose=verbose)

        # Benchmark
        result = benchmark_quantized_model(
            model=model,
            tokenizer=tokenizer,
            quantization_method=method,
            test_prompt=test_prompt,
            generation_tokens=generation_tokens,
            verbose=verbose,
        )

        results[method] = result

        # Use first method as baseline (typically fp16)
        if baseline_result is None:
            baseline_result = result

    # Calculate relative metrics
    if baseline_result is not None:
        for result in results.values():
            # Memory reduction
            result.memory_reduction_percent = (
                (1 - result.model_size_gb / baseline_result.model_size_gb) * 100
            )

            # Speed improvements
            result.prompt_speedup = result.prompt_tps / baseline_result.prompt_tps
            result.generation_speedup = result.generation_tps / baseline_result.generation_tps

            # Memory savings
            result.memory_savings_percent = (
                (1 - result.peak_memory_gb / baseline_result.peak_memory_gb) * 100
            )

        if verbose:
            print_comparison_summary(results, baseline_result)

    return results


def print_comparison_summary(
    results: dict[str, QuantizationBenchmarkResult],
    baseline: QuantizationBenchmarkResult,
):
    """Print comparison summary table."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    # Create comparison table
    print(f"\n{'Method':<12} {'Size (GB)':<12} {'Reduction':<12} {'Speed (tok/s)':<15} {'Speedup':<10} {'Memory (GB)':<12}")
    print("-" * 70)

    for method, result in results.items():
        reduction = f"{result.memory_reduction_percent:+.1f}%"
        speedup = f"{result.generation_speedup:.2f}x"

        print(
            f"{method:<12} "
            f"{result.model_size_gb:<12.2f} "
            f"{reduction:<12} "
            f"{result.generation_tps:<15.0f} "
            f"{speedup:<10} "
            f"{result.peak_memory_gb:<12.2f}"
        )

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find best options
    best_speed = max(results.values(), key=lambda r: r.generation_tps)
    best_memory = min(results.values(), key=lambda r: r.peak_memory_gb)
    best_size = min(results.values(), key=lambda r: r.model_size_gb)

    print(f"  Fastest generation: {best_speed.quantization_method} ({best_speed.generation_tps:.0f} tok/s)")
    print(f"  Lowest memory: {best_memory.quantization_method} ({format_memory(best_memory.peak_memory_gb * 1e9)})")
    print(f"  Smallest size: {best_size.quantization_method} ({format_memory(best_size.model_size_gb * 1e9)})")

    # Recommendations
    print("\n  Use cases:")
    print("    - Production deployment: 4bit (best size/memory tradeoff)")
    print(f"    - Maximum speed: {best_speed.quantization_method}")
    print("    - Maximum quality: fp16 (baseline)")
    print("=" * 70)


def compare_mxfp8_vs_int8(
    model_path: str = "mlx-community/SmolLM2-135M-Instruct",
    test_prompt: str = "The quick brown fox jumps over the lazy dog",
    generation_tokens: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Detailed comparison of MXFP8 vs INT8 quantization on Apple M4.

    This function benchmarks both methods and provides a recommendation
    based on M4 hardware characteristics.

    Args:
        model_path: Path to model or HuggingFace ID
        test_prompt: Test prompt
        generation_tokens: Tokens to generate
        verbose: Print detailed comparison

    Returns:
        Dictionary with comparison results and recommendation
    """
    if verbose:
        print("=" * 70)
        print("MXFP8 vs INT8 Comparison (Apple M4)")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Test: {test_prompt[:50]}...")
        print()

    # Compare MXFP8, INT8, and FP16 (baseline)
    results = compare_quantization_methods(
        model_path=model_path,
        quantization_methods=["fp16", "mxfp8", "int8"],
        test_prompt=test_prompt,
        generation_tokens=generation_tokens,
        verbose=verbose,
    )

    mxfp8_result = results.get("mxfp8")
    int8_result = results.get("int8")
    fp16_result = results.get("fp16")

    if not mxfp8_result or not int8_result or not fp16_result:
        raise ValueError("Failed to benchmark all methods")

    # Calculate detailed comparison
    comparison = {
        # MXFP8 metrics
        "mxfp8_tokens_per_sec": mxfp8_result.generation_tps,
        "mxfp8_memory_mb": mxfp8_result.peak_memory_gb * 1024,
        "mxfp8_size_mb": mxfp8_result.model_size_gb * 1024,
        "mxfp8_speedup_vs_fp16": mxfp8_result.generation_speedup,

        # INT8 metrics
        "int8_tokens_per_sec": int8_result.generation_tps,
        "int8_memory_mb": int8_result.peak_memory_gb * 1024,
        "int8_size_mb": int8_result.model_size_gb * 1024,
        "int8_speedup_vs_fp16": int8_result.generation_speedup,

        # FP16 baseline
        "fp16_tokens_per_sec": fp16_result.generation_tps,
        "fp16_memory_mb": fp16_result.peak_memory_gb * 1024,
        "fp16_size_mb": fp16_result.model_size_gb * 1024,

        # Direct comparison
        "int8_speedup_vs_mxfp8": int8_result.generation_tps / mxfp8_result.generation_tps,
        "mxfp8_memory_advantage": (int8_result.peak_memory_gb - mxfp8_result.peak_memory_gb) * 1024,
        "mxfp8_size_advantage": (int8_result.model_size_gb - mxfp8_result.model_size_gb) * 1024,

        # Format info
        "mxfp8_format": "E4M3 (4-bit exp, 3-bit mantissa)",
        "mxfp8_block_size": 32,
        "mxfp8_hardware": "Software emulated (Metal shaders)",
        "int8_format": "Symmetric INT8",
        "int8_hardware": "Native AMX + GPU acceleration",
    }

    # Generate recommendation
    if int8_result.generation_tps > mxfp8_result.generation_tps * 1.1:
        recommendation = (
            "Use INT8 for speed-critical inference on M4 (native acceleration)\n"
            f"    INT8 is {comparison['int8_speedup_vs_mxfp8']:.1f}x faster than MXFP8 on M4"
        )
    elif mxfp8_result.peak_memory_gb < int8_result.peak_memory_gb * 0.9:
        recommendation = (
            "Use MXFP8 for memory-constrained scenarios\n"
            f"    MXFP8 saves {comparison['mxfp8_memory_advantage']:.0f} MB vs INT8"
        )
    else:
        recommendation = (
            "INT8 and MXFP8 perform similarly on M4\n"
            "    Use INT8 for speed, MXFP8 for OCP standard compatibility"
        )

    comparison["recommendation"] = recommendation

    if verbose:
        print("\n" + "=" * 70)
        print("DETAILED COMPARISON")
        print("=" * 70)

        print("\nMXFP8 (OCP Microscaling):")
        print(f"  Format: {comparison['mxfp8_format']}")
        print(f"  Storage: uint8 (true 8-bit)")
        print(f"  Block size: {comparison['mxfp8_block_size']}")
        print(f"  Scale storage: uint8 (8-bit)")
        print(f"\n  Model size: {comparison['mxfp8_size_mb']:.1f} MB")
        print(f"  Memory usage: {comparison['mxfp8_memory_mb']:.1f} MB")
        print(f"  Tokens/sec: {comparison['mxfp8_tokens_per_sec']:.1f}")
        print(f"  Speedup vs FP16: {comparison['mxfp8_speedup_vs_fp16']:.2f}x")
        print(f"\n  Hardware: {comparison['mxfp8_hardware']}")

        print("\nINT8 (GPTQ/AWQ):")
        print(f"  Format: {comparison['int8_format']}")
        print(f"  Storage: uint8 (true 8-bit)")
        print(f"  Scale storage: float32 (32-bit)")
        print(f"\n  Model size: {comparison['int8_size_mb']:.1f} MB")
        print(f"  Memory usage: {comparison['int8_memory_mb']:.1f} MB")
        print(f"  Tokens/sec: {comparison['int8_tokens_per_sec']:.1f}")
        print(f"  Speedup vs FP16: {comparison['int8_speedup_vs_fp16']:.2f}x")
        print(f"\n  Hardware: {comparison['int8_hardware']}")

        print("\nDirect Comparison:")
        print(f"  INT8 speedup vs MXFP8: {comparison['int8_speedup_vs_mxfp8']:.2f}x")
        if comparison['mxfp8_memory_advantage'] > 0:
            print(f"  MXFP8 memory advantage: {comparison['mxfp8_memory_advantage']:.0f} MB")
        else:
            print(f"  INT8 memory advantage: {-comparison['mxfp8_memory_advantage']:.0f} MB")

        if comparison['mxfp8_size_advantage'] > 0:
            print(f"  MXFP8 size advantage: {comparison['mxfp8_size_advantage']:.0f} MB")
        else:
            print(f"  INT8 size advantage: {-comparison['mxfp8_size_advantage']:.0f} MB")

        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        print(f"\n{comparison['recommendation']}\n")
        print("=" * 70)

    return comparison


def main():
    """CLI entry point for quantization benchmarks."""
    parser = argparse.ArgumentParser(
        description="Compare quantization methods for MLX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare FP16, 8-bit, and 4-bit
  python -m smlx.bench.suites.quantization \\
      --model mlx-community/SmolLM2-135M-Instruct \\
      --methods fp16 8bit 4bit

  # Save results to JSON
  python -m smlx.bench.suites.quantization \\
      --model mlx-community/SmolLM2-135M-Instruct \\
      --methods fp16 8bit 4bit \\
      --output quantization_results.json

  # Custom test configuration
  python -m smlx.bench.suites.quantization \\
      --model mlx-community/SmolLM2-135M-Instruct \\
      --methods fp16 4bit \\
      --prompt "Once upon a time" \\
      --generation-tokens 200
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/SmolLM2-135M-Instruct",
        help="Model path or HuggingFace ID",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fp16"],
        help="Quantization methods to compare (fp16, 8bit, 4bit, int8, mxfp8, mxfp4, fp8, gptq, awq, dwq)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog",
        help="Test prompt",
    )

    parser.add_argument(
        "--generation-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )

    args = parser.parse_args()

    # Run comparison
    results = compare_quantization_methods(
        model_path=args.model,
        quantization_methods=args.methods,
        test_prompt=args.prompt,
        generation_tokens=args.generation_tokens,
        verbose=args.verbose,
    )

    # Save results
    if args.output:
        output_data = {
            method: result.to_dict()
            for method, result in results.items()
        }

        save_json(output_data, args.output)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
