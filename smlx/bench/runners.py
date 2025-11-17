"""
Benchmark runners for executing and managing benchmarks.

Provides classes and functions for running benchmarks with proper
setup, teardown, and result collection.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import mlx.core as mx

from smlx.utils.memory import clear_cache, memory_profiler, reset_peak_memory
from smlx.utils.timing import timer

from .stats import BenchmarkStats, OperationBenchmarkStats


class BenchmarkRunner(ABC):
    """
    Abstract base class for benchmark runners.

    Subclass this to create custom benchmark runners for specific
    model types or operations.
    """

    def __init__(
        self,
        name: str,
        num_warmup: int = 5,
        num_iterations: int = 10,
        clear_cache_between_runs: bool = True,
    ):
        """
        Initialize benchmark runner.

        Args:
            name: Benchmark name
            num_warmup: Number of warmup iterations
            num_iterations: Number of timed iterations
            clear_cache_between_runs: Whether to clear cache between runs
        """
        self.name = name
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.clear_cache_between_runs = clear_cache_between_runs

    @abstractmethod
    def run(self) -> BenchmarkStats:
        """
        Run the benchmark and return statistics.

        Returns:
            BenchmarkStats or subclass
        """
        pass

    def setup(self):
        """Setup before benchmark (override if needed)."""
        if self.clear_cache_between_runs:
            clear_cache()
        reset_peak_memory()

    def teardown(self):  # noqa: B027
        """Teardown after benchmark (override if needed)."""
        pass


class FunctionBenchmarkRunner(BenchmarkRunner):
    """
    Benchmark runner for arbitrary functions.

    Example:
        >>> def my_op(x):
        ...     return mx.matmul(x, x.T)
        >>> runner = FunctionBenchmarkRunner(
        ...     name="matmul",
        ...     fn=my_op,
        ...     args=(mx.random.normal((1000, 1000)),)
        ... )
        >>> stats = runner.run()
    """

    def __init__(
        self,
        name: str,
        fn: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        num_warmup: int = 5,
        num_iterations: int = 10,
        clear_cache_between_runs: bool = True,
    ):
        """
        Initialize function benchmark runner.

        Args:
            name: Benchmark name
            fn: Function to benchmark
            args: Positional arguments for fn
            kwargs: Keyword arguments for fn
            num_warmup: Number of warmup iterations
            num_iterations: Number of timed iterations
            clear_cache_between_runs: Whether to clear cache between runs
        """
        super().__init__(name, num_warmup, num_iterations, clear_cache_between_runs)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs or {}

    def run(self) -> BenchmarkStats:
        """Run the benchmark."""
        self.setup()

        try:
            # Run benchmark with memory profiling
            with memory_profiler() as mem:
                result, times = timer(
                    self.fn,
                    *self.args,
                    num_warmup=self.num_warmup,
                    num_iterations=self.num_iterations,
                    force_eval=True,
                    **self.kwargs,
                )

            # Calculate statistics
            mean_time_ms = (sum(times) / len(times)) * 1000

            stats = BenchmarkStats(
                name=self.name,
                duration_ms=mean_time_ms,
                iterations=self.num_iterations,
                peak_memory_gb=mem.peak_gb,
            )

            return stats

        finally:
            self.teardown()


class OperationBenchmarkRunner(BenchmarkRunner):
    """
    Benchmark runner for MLX operations.

    Example:
        >>> runner = OperationBenchmarkRunner(
        ...     name="matmul_benchmark",
        ...     operation="matmul",
        ...     fn=lambda a, b: mx.matmul(a, b),
        ...     args=(mx.random.normal((1000, 1000)), mx.random.normal((1000, 1000))),
        ...     input_shapes=[(1000, 1000), (1000, 1000)],
        ... )
        >>> stats = runner.run()
    """

    def __init__(
        self,
        name: str,
        operation: str,
        fn: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        input_shapes: Optional[list[tuple]] = None,
        output_shape: Optional[tuple] = None,
        dtype: str = "float32",
        num_warmup: int = 5,
        num_iterations: int = 100,  # More iterations for ops
    ):
        """
        Initialize operation benchmark runner.

        Args:
            name: Benchmark name
            operation: Operation name (e.g., 'matmul', 'attention')
            fn: Function implementing the operation
            args: Positional arguments for fn
            kwargs: Keyword arguments for fn
            input_shapes: List of input tensor shapes
            output_shape: Output tensor shape
            dtype: Data type
            num_warmup: Number of warmup iterations
            num_iterations: Number of timed iterations
        """
        super().__init__(name, num_warmup, num_iterations)
        self.operation = operation
        self.fn = fn
        self.args = args
        self.kwargs = kwargs or {}
        self.input_shapes = input_shapes or []
        self.output_shape = output_shape
        self.dtype = dtype

    def run(self) -> OperationBenchmarkStats:
        """Run the benchmark."""
        self.setup()

        try:
            with memory_profiler() as mem:
                result, times = timer(
                    self.fn,
                    *self.args,
                    num_warmup=self.num_warmup,
                    num_iterations=self.num_iterations,
                    force_eval=True,
                    **self.kwargs,
                )

            mean_time_ms = (sum(times) / len(times)) * 1000

            # Determine output shape from result
            output_shape = self.output_shape
            if output_shape is None and hasattr(result, "shape"):
                output_shape = tuple(result.shape)

            stats = OperationBenchmarkStats(
                name=self.name,
                operation=self.operation,
                duration_ms=mean_time_ms,
                iterations=self.num_iterations,
                peak_memory_gb=mem.peak_gb,
                input_shapes=self.input_shapes,
                output_shape=output_shape,
                dtype=self.dtype,
                device="gpu" if mx.metal.is_available() else "cpu",
            )

            return stats

        finally:
            self.teardown()


def quick_benchmark(
    fn: Callable,
    *args,
    name: str = "benchmark",
    num_warmup: int = 5,
    num_iterations: int = 10,
    **kwargs,
) -> BenchmarkStats:
    """
    Quick benchmark for a function.

    Convenience function for one-off benchmarks.

    Args:
        fn: Function to benchmark
        *args: Positional arguments for fn
        name: Benchmark name
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        **kwargs: Keyword arguments for fn

    Returns:
        BenchmarkStats

    Example:
        >>> stats = quick_benchmark(
        ...     lambda x: x @ x.T,
        ...     mx.random.normal((1000, 1000)),
        ...     name="matmul_1000x1000"
        ... )
        >>> print(f"{stats.name}: {stats.duration_ms:.2f}ms")
    """
    runner = FunctionBenchmarkRunner(
        name=name,
        fn=fn,
        args=args,
        kwargs=kwargs,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )
    return runner.run()


def compare_implementations(
    baseline_fn: Callable,
    comparison_fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    baseline_name: str = "baseline",
    comparison_name: str = "comparison",
    num_warmup: int = 5,
    num_iterations: int = 10,
) -> dict:
    """
    Compare two implementations.

    Args:
        baseline_fn: Baseline function
        comparison_fn: Comparison function
        args: Arguments to pass to both functions
        kwargs: Keyword arguments to pass to both functions
        baseline_name: Name for baseline
        comparison_name: Name for comparison
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations

    Returns:
        Dictionary with:
        - baseline: BenchmarkStats for baseline
        - comparison: BenchmarkStats for comparison
        - speedup: Speedup factor
        - memory_reduction_gb: Memory reduction in GB

    Example:
        >>> def baseline(x):
        ...     return x @ x.T
        >>> def optimized(x):
        ...     return mx.compile(lambda a: a @ a.T)(x)
        >>> results = compare_implementations(baseline, optimized, args=(x,))
        >>> print(f"Speedup: {results['speedup']:.2f}x")
    """
    kwargs = kwargs or {}

    # Run baseline
    baseline_stats = quick_benchmark(
        baseline_fn,
        *args,
        name=baseline_name,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        **kwargs,
    )

    # Run comparison
    comparison_stats = quick_benchmark(
        comparison_fn,
        *args,
        name=comparison_name,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        **kwargs,
    )

    # Calculate metrics
    speedup = (
        baseline_stats.duration_ms / comparison_stats.duration_ms
        if comparison_stats.duration_ms > 0
        else 0.0
    )
    memory_reduction = baseline_stats.peak_memory_gb - comparison_stats.peak_memory_gb

    return {
        "baseline": baseline_stats,
        "comparison": comparison_stats,
        "speedup": speedup,
        "memory_reduction_gb": memory_reduction,
    }
