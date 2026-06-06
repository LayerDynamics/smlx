"""
Timing utilities for benchmarking.

Provides high-resolution timing with warmup and statistical aggregation,
following MLX benchmarking best practices.
"""

import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import mlx.core as mx

T = TypeVar("T")


def timer(
    fn: Callable[..., T],
    *args,
    num_warmup: int = 5,
    num_iterations: int = 10,
    force_eval: bool = True,
    **kwargs,
) -> tuple[T, list[float]]:
    """
    Time a function with warmup and multiple iterations.

    Args:
        fn: Function to benchmark
        *args: Positional arguments to pass to fn
        num_warmup: Number of warmup iterations (default: 5)
        num_iterations: Number of timed iterations (default: 10)
        force_eval: Whether to call mx.eval() on result (default: True)
        **kwargs: Keyword arguments to pass to fn

    Returns:
        Tuple of (last_result, list_of_times_in_seconds)

    Example:
        >>> def matmul(a, b):
        ...     return a @ b
        >>> result, times = timer(matmul, a, b, num_iterations=100)
        >>> print(f"Mean time: {sum(times)/len(times)*1000:.3f}ms")
    """
    # Warmup phase
    result: Optional[T] = None
    for _ in range(num_warmup):
        result = fn(*args, **kwargs)
        if force_eval and hasattr(result, "__mlx_array__"):
            mx.eval(result)
        elif force_eval and isinstance(result, (list, tuple)):
            # Handle multiple return values
            for r in result:
                if hasattr(r, "__mlx_array__"):
                    mx.eval(r)

    # Timed iterations
    times = []
    for _ in range(num_iterations):
        tic = time.perf_counter()
        result = fn(*args, **kwargs)
        if force_eval and hasattr(result, "__mlx_array__"):
            mx.eval(result)
        elif force_eval and isinstance(result, (list, tuple)):
            for r in result:
                if hasattr(r, "__mlx_array__"):
                    mx.eval(r)
        toc = time.perf_counter()
        times.append(toc - tic)

    # When iterations were requested, the function must have run and produced a
    # result. Zero iterations is a valid request (e.g. a no-op timing probe):
    # return whatever the warmup produced (possibly None) with empty `times`.
    if num_iterations > 0:
        assert result is not None, "At least one iteration must be run"
    return result, times


def benchmark(
    num_warmup: int = 5,
    num_iterations: int = 10,
    force_eval: bool = True,
):
    """
    Decorator for benchmarking functions.

    Args:
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        force_eval: Whether to call mx.eval() on MLX arrays

    Returns:
        Decorated function that returns (result, times)

    Example:
        >>> @benchmark(num_iterations=100)
        ... def my_function(x):
        ...     return x @ x.T
        >>> result, times = my_function(x)
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., tuple[T, list[float]]]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> tuple[T, list[float]]:
            return timer(
                fn,
                *args,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
                force_eval=force_eval,
                **kwargs,
            )

        return wrapper

    return decorator


class Timer:
    """
    Context manager for timing code blocks.

    Example:
        >>> with Timer() as t:
        ...     result = expensive_operation()
        >>> print(f"Elapsed: {t.elapsed_ms:.3f}ms")
    """

    def __init__(self, force_eval: bool = True):
        """
        Initialize timer.

        Args:
            force_eval: Whether to call mx.eval() before stopping timer
        """
        self.force_eval = force_eval
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.force_eval:
            # Force evaluation of any pending MLX operations
            mx.eval([])
        self.end_time = time.perf_counter()
        return False

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000.0


def measure_runtime(
    fn: Callable[..., Any],
    *args,
    num_warmup: int = 5,
    num_iterations: int = 100,
    **kwargs,
) -> float:
    """
    Measure average runtime of a function in milliseconds.

    Similar to MLX core benchmarks - focuses on mean time per iteration.

    Args:
        fn: Function to measure
        *args: Positional arguments
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        **kwargs: Keyword arguments

    Returns:
        Mean time per iteration in milliseconds

    Example:
        >>> ms_per_iter = measure_runtime(matmul, a, b)
        >>> print(f"Time: {ms_per_iter:.3f}ms")
    """
    _, times = timer(fn, *args, num_warmup=num_warmup, num_iterations=num_iterations, **kwargs)
    mean_time_sec = sum(times) / len(times)
    return mean_time_sec * 1000.0  # Convert to milliseconds
