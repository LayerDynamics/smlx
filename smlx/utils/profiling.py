#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Performance profiling utilities for M4 optimization.

Provides tools for measuring and optimizing model performance on Apple Silicon.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx
import numpy as np


@dataclass
class ProfileResult:
    """
    Result of performance profiling.

    Attributes:
        operation: Name of operation
        elapsed_time: Time in seconds
        memory_used: Memory used in MB
        tokens_per_second: Tokens/sec (if applicable)
        throughput: Items/sec (generic throughput)
    """

    operation: str
    elapsed_time: float
    memory_used: Optional[float] = None
    tokens_per_second: Optional[float] = None
    throughput: Optional[float] = None

    def __str__(self):
        result = f"{self.operation}:\n"
        result += f"  Time: {self.elapsed_time*1000:.2f}ms\n"
        if self.memory_used is not None:
            result += f"  Memory: {self.memory_used:.1f}MB\n"
        if self.tokens_per_second is not None:
            result += f"  Speed: {self.tokens_per_second:.1f} tokens/sec\n"
        if self.throughput is not None:
            result += f"  Throughput: {self.throughput:.1f} items/sec\n"
        return result


@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager for timing operations.

    Args:
        name: Name of operation

    Example:
        >>> with timer("Generation"):
        ...     output = model.generate(prompt)
        Generation: 123.45ms
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name}: {elapsed*1000:.2f}ms")


@contextmanager
def profile_memory(name: str = "Operation"):
    """
    Context manager for profiling memory usage.

    Args:
        name: Name of operation

    Example:
        >>> with profile_memory("Model Loading"):
        ...     model = load_model()
        Model Loading: 512.3MB
    """
    # Get memory before
    mx.metal.clear_cache()
    mem_before = mx.metal.get_active_memory() / (1024 * 1024)

    yield

    # Get memory after
    mx.eval(mx.zeros(1))  # Force evaluation
    mem_after = mx.metal.get_active_memory() / (1024 * 1024)

    memory_used = mem_after - mem_before
    print(f"{name}: {memory_used:.1f}MB")


def benchmark_function(
    func: Callable,
    num_runs: int = 10,
    warmup_runs: int = 3,
    name: str = "Function",
) -> dict[str, float]:
    """
    Benchmark a function with multiple runs.

    Args:
        func: Function to benchmark
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        name: Name for display

    Returns:
        Dictionary with timing statistics

    Example:
        >>> def my_function():
        ...     return model.generate(prompt)
        >>> stats = benchmark_function(my_function, num_runs=20)
        >>> print(f"Average: {stats['mean']:.2f}ms")
    """
    # Warmup
    print(f"Warming up {name}...")
    for _ in range(warmup_runs):
        func()

    # Benchmark
    print(f"Benchmarking {name} ({num_runs} runs)...")
    times = []

    for _ in range(num_runs):
        start = time.time()
        func()
        elapsed = time.time() - start
        times.append(elapsed)

    # Statistics
    times_np = np.array(times)
    stats = {
        "mean": np.mean(times_np) * 1000,  # ms
        "median": np.median(times_np) * 1000,
        "min": np.min(times_np) * 1000,
        "max": np.max(times_np) * 1000,
        "std": np.std(times_np) * 1000,
        "p95": np.percentile(times_np, 95) * 1000,
        "p99": np.percentile(times_np, 99) * 1000,
    }

    # Print results
    print(f"\n{name} Benchmark Results:")
    print(f"  Mean: {stats['mean']:.2f}ms")
    print(f"  Median: {stats['median']:.2f}ms")
    print(f"  Min: {stats['min']:.2f}ms")
    print(f"  Max: {stats['max']:.2f}ms")
    print(f"  Std: {stats['std']:.2f}ms")
    print(f"  P95: {stats['p95']:.2f}ms")
    print(f"  P99: {stats['p99']:.2f}ms")

    return stats


def profile_generation(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    num_runs: int = 5,
) -> ProfileResult:
    """
    Profile text generation performance.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        num_runs: Number of runs for averaging

    Returns:
        ProfileResult with generation statistics

    Example:
        >>> result = profile_generation(model, tokenizer, "Hello", max_tokens=50)
        >>> print(result)
    """
    from ..models.SmolLM2_135M import generate

    # Warmup
    generate(model, tokenizer, prompt, max_tokens=10)

    # Benchmark
    times = []
    total_tokens = 0

    for _ in range(num_runs):
        start = time.time()
        output = generate(model, tokenizer, prompt, max_tokens=max_tokens)
        elapsed = time.time() - start

        times.append(elapsed)
        total_tokens += len(tokenizer.encode(output))

    # Calculate metrics
    avg_time = float(np.mean(times))
    avg_tokens = total_tokens / num_runs
    tokens_per_second = float(avg_tokens / avg_time)

    # Memory usage
    mx.metal.clear_cache()
    mem_before = mx.metal.get_active_memory() / (1024 * 1024)
    generate(model, tokenizer, prompt, max_tokens=max_tokens)
    mx.eval(mx.zeros(1))
    mem_after = mx.metal.get_active_memory() / (1024 * 1024)
    memory_used = float(mem_after - mem_before)

    result = ProfileResult(
        operation="Text Generation",
        elapsed_time=avg_time,
        memory_used=memory_used,
        tokens_per_second=tokens_per_second,
    )

    return result


def get_system_info() -> dict[str, Any]:
    """
    Get system information for performance benchmarking.

    Returns:
        Dictionary with system info

    Example:
        >>> info = get_system_info()
        >>> print(f"Device: {info['device']}")
    """
    import platform

    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "mlx_device": "Apple Silicon (Metal)",
        "metal_available": mx.metal.is_available(),
    }

    # Try to get memory info
    try:
        total_memory = mx.metal.get_peak_memory() / (1024 * 1024 * 1024)
        info["total_memory_gb"] = total_memory
    except Exception:
        info["total_memory_gb"] = "Unknown"

    return info


def print_system_info():
    """
    Print system information.

    Example:
        >>> print_system_info()
    """
    info = get_system_info()

    print("\n" + "=" * 70)
    print("System Information")
    print("=" * 70)
    print(f"Platform: {info['platform']} {info['platform_version']}")
    print(f"Processor: {info['processor']}")
    print(f"Python: {info['python_version']}")
    print(f"Device: {info['mlx_device']}")
    print(f"Metal: {'Available' if info['metal_available'] else 'Not Available'}")
    if info["total_memory_gb"] != "Unknown":
        print(f"Memory: {info['total_memory_gb']:.1f}GB")
    print("=" * 70)


def compare_implementations(
    implementations: dict[str, Callable],
    num_runs: int = 10,
) -> dict[str, dict[str, float]]:
    """
    Compare performance of different implementations.

    Args:
        implementations: Dict mapping names to functions
        num_runs: Number of runs per implementation

    Returns:
        Dict mapping names to timing statistics

    Example:
        >>> implementations = {
        ...     "naive": naive_impl,
        ...     "optimized": optimized_impl,
        ... }
        >>> results = compare_implementations(implementations)
        >>> # Shows which is faster
    """
    results = {}

    print("\n" + "=" * 70)
    print("Implementation Comparison")
    print("=" * 70)

    for name, func in implementations.items():
        stats = benchmark_function(func, num_runs=num_runs, name=name)
        results[name] = stats

    # Print comparison
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)

    # Find fastest
    fastest = min(results.items(), key=lambda x: x[1]["mean"])
    print(f"Fastest: {fastest[0]} ({fastest[1]['mean']:.2f}ms)")

    # Compare to fastest
    print("\nRelative to fastest:")
    for name, stats in results.items():
        speedup = fastest[1]["mean"] / stats["mean"]
        if name == fastest[0]:
            print(f"  {name}: 1.00x (baseline)")
        else:
            slowdown = stats["mean"] / fastest[1]["mean"]
            print(f"  {name}: {slowdown:.2f}x slower ({speedup:.2f}x speedup needed)")

    return results
