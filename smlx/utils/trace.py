#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Performance tracing and profiling utilities for MLX models.

Provides context managers and decorators for tracking:
- Execution time
- Memory usage
- MLX graph evaluation
- Token generation performance
"""

import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

import mlx.core as mx


@contextmanager
def trace_time(name: str = "Operation", verbose: bool = True):
    """Context manager for timing code execution.

    Args:
        name: Name of the operation being timed
        verbose: Whether to print timing information

    Yields:
        Dictionary that will contain timing results

    Example:
        >>> with trace_time("Model inference") as timing:
        ...     result = model(input)
        >>> print(f"Took {timing['elapsed']:.3f}s")
    """
    timing = {}
    start = time.perf_counter()

    try:
        yield timing
    finally:
        end = time.perf_counter()
        elapsed = end - start
        timing["start"] = start
        timing["end"] = end
        timing["elapsed"] = elapsed

        if verbose:
            print(f"ñ  {name}: {elapsed:.3f}s")


@contextmanager
def trace_memory(name: str = "Operation", verbose: bool = True):
    """Context manager for tracking memory usage.

    Args:
        name: Name of the operation being tracked
        verbose: Whether to print memory information

    Yields:
        Dictionary that will contain memory results

    Example:
        >>> with trace_memory("Model loading") as mem:
        ...     model = load_model()
        >>> print(f"Used {mem['delta_mb']:.1f} MB")
    """
    memory = {}

    # Get initial memory stats
    try:
        import psutil

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        memory["available"] = psutil.virtual_memory().available / 1024 / 1024  # MB
    except ImportError:
        mem_before = 0
        memory["available"] = 0

    # MLX memory (Metal GPU cache)
    mx_before = mx.metal.get_cache_memory() / 1024 / 1024  # MB

    try:
        yield memory
    finally:
        # Get final memory stats
        try:
            mem_after = process.memory_info().rss / 1024 / 1024
        except (NameError, ImportError):
            mem_after = 0

        mx_after = mx.metal.get_cache_memory() / 1024 / 1024

        memory["mem_before_mb"] = mem_before
        memory["mem_after_mb"] = mem_after
        memory["delta_mb"] = mem_after - mem_before
        memory["mlx_before_mb"] = mx_before
        memory["mlx_after_mb"] = mx_after
        memory["mlx_delta_mb"] = mx_after - mx_before

        if verbose:
            print(f"=¾ {name}:")
            print(
                f"   RAM: {mem_before:.1f} MB ’ {mem_after:.1f} MB "
                f"(” {memory['delta_mb']:+.1f} MB)"
            )
            print(
                f"   MLX: {mx_before:.1f} MB ’ {mx_after:.1f} MB "
                f"(” {memory['mlx_delta_mb']:+.1f} MB)"
            )


@contextmanager
def trace_performance(name: str = "Operation", verbose: bool = True):
    """Combined timing and memory tracking.

    Args:
        name: Name of the operation
        verbose: Whether to print performance information

    Yields:
        Dictionary containing timing and memory results

    Example:
        >>> with trace_performance("Full pipeline") as perf:
        ...     output = pipeline(input)
        >>> print(perf)
        {'elapsed': 1.234, 'delta_mb': 123.4, ...}
    """
    perf = {}

    with trace_time(name, verbose=False) as timing:
        with trace_memory(name, verbose=False) as memory:
            yield perf

    # Combine results
    perf.update(timing)
    perf.update(memory)

    if verbose:
        print(f"=Ê {name}:")
        print(f"   Time: {timing['elapsed']:.3f}s")
        print(f"   RAM:  ” {memory['delta_mb']:+.1f} MB")
        print(f"   MLX:  ” {memory['mlx_delta_mb']:+.1f} MB")


def trace_function(name: Optional[str] = None, verbose: bool = True):
    """Decorator for tracing function performance.

    Args:
        name: Name for the operation (defaults to function name)
        verbose: Whether to print performance information

    Example:
        >>> @trace_function(verbose=True)
        ... def slow_operation():
        ...     time.sleep(1)
        >>> slow_operation()
        ñ  slow_operation: 1.001s
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            op_name = name or func.__name__

            with trace_performance(op_name, verbose=verbose):
                result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator


@contextmanager
def trace_generation(name: str = "Generation", verbose: bool = True):
    """Specialized tracing for text generation.

    Tracks tokens per second and time to first token.

    Args:
        name: Name of the generation operation
        verbose: Whether to print generation statistics

    Yields:
        Dictionary to track generation metrics

    Example:
        >>> with trace_generation() as gen:
        ...     for token in generate_tokens():
        ...         gen["tokens"] += 1
        ...         if gen["tokens"] == 1:
        ...             gen["ttft"] = time.time() - gen["start_time"]
    """
    gen = {
        "tokens": 0,
        "ttft": None,  # Time to first token
        "start_time": time.perf_counter(),
    }

    try:
        yield gen
    finally:
        end_time = time.perf_counter()
        total_time = end_time - gen["start_time"]

        gen["total_time"] = total_time
        gen["end_time"] = end_time

        if gen["tokens"] > 0:
            gen["tokens_per_sec"] = gen["tokens"] / total_time
        else:
            gen["tokens_per_sec"] = 0

        if verbose:
            print(f"<¯ {name}:")
            print(f"   Tokens: {gen['tokens']}")
            if gen["ttft"] is not None:
                print(f"   TTFT: {gen['ttft']:.3f}s")
            print(f"   Total: {total_time:.3f}s")
            print(f"   Speed: {gen['tokens_per_sec']:.1f} tokens/s")


@contextmanager
def trace_mlx_eval(name: str = "MLX Eval", verbose: bool = True):
    """Context manager that ensures MLX operations are evaluated and timed.

    Forces evaluation of lazy MLX computations and measures actual execution time.

    Args:
        name: Name of the operation
        verbose: Whether to print timing information

    Yields:
        Dictionary with timing results

    Example:
        >>> with trace_mlx_eval("Matrix multiply") as timing:
        ...     result = mx.matmul(a, b)
        >>> # Result is evaluated before exiting context
    """
    timing = {}
    start = time.perf_counter()

    try:
        yield timing
    finally:
        # Force evaluation of any pending MLX operations
        mx.eval(mx.metal.get_active_memory())

        end = time.perf_counter()
        elapsed = end - start

        timing["start"] = start
        timing["end"] = end
        timing["elapsed"] = elapsed

        if verbose:
            print(f"¡ {name}: {elapsed:.3f}s (evaluated)")


class PerformanceProfiler:
    """Performance profiler for tracking multiple operations.

    Collects timing and memory statistics for multiple named operations.

    Example:
        >>> profiler = PerformanceProfiler()
        >>> with profiler.trace("load_model"):
        ...     model = load()
        >>> with profiler.trace("inference"):
        ...     output = model(input)
        >>> profiler.print_summary()
    """

    def __init__(self):
        """Initialize profiler."""
        self.traces = []

    @contextmanager
    def trace(self, name: str):
        """Trace an operation.

        Args:
            name: Name of the operation

        Yields:
            Trace dictionary
        """
        trace_data = {"name": name}

        with trace_performance(name, verbose=False) as perf:
            yield trace_data

        trace_data.update(perf)
        self.traces.append(trace_data)

    def print_summary(self):
        """Print summary of all traced operations."""
        if not self.traces:
            print("No traces recorded.")
            return

        print("\n=Ê Performance Summary")
        print("=" * 70)
        print(f"{'Operation':<30} {'Time (s)':<12} {'RAM (MB)':<12} {'MLX (MB)':<12}")
        print("-" * 70)

        total_time = 0
        total_ram = 0
        total_mlx = 0

        for trace in self.traces:
            name = trace["name"]
            elapsed = trace.get("elapsed", 0)
            delta_ram = trace.get("delta_mb", 0)
            delta_mlx = trace.get("mlx_delta_mb", 0)

            total_time += elapsed
            total_ram += delta_ram
            total_mlx += delta_mlx

            print(f"{name:<30} {elapsed:>10.3f}  {delta_ram:>+10.1f}  {delta_mlx:>+10.1f}")

        print("-" * 70)
        print(f"{'TOTAL':<30} {total_time:>10.3f}  {total_ram:>+10.1f}  {total_mlx:>+10.1f}")
        print("=" * 70)

    def get_trace(self, name: str) -> Optional[dict]:
        """Get trace by name.

        Args:
            name: Operation name

        Returns:
            Trace dictionary or None if not found
        """
        for trace in self.traces:
            if trace["name"] == name:
                return trace
        return None

    def clear(self):
        """Clear all traces."""
        self.traces.clear()


__all__ = [
    "trace_time",
    "trace_memory",
    "trace_performance",
    "trace_function",
    "trace_generation",
    "trace_mlx_eval",
    "PerformanceProfiler",
]
