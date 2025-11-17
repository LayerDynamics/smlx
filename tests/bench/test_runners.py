"""
Tests for benchmark runners.

Tests the BenchmarkRunner classes used for executing benchmarks.
"""

import pytest
import mlx.core as mx

from smlx.bench.runners import (
    BenchmarkRunner,
    FunctionBenchmarkRunner,
    OperationBenchmarkRunner,
    compare_implementations,
    quick_benchmark,
)
from smlx.bench.stats import BenchmarkStats, OperationBenchmarkStats


class DummyRunner(BenchmarkRunner):
    """Dummy runner for testing abstract base class."""

    def __init__(self, **kwargs):
        super().__init__(name="dummy", **kwargs)
        self.setup_called = False
        self.teardown_called = False

    def run(self) -> BenchmarkStats:
        """Run dummy benchmark."""
        return BenchmarkStats(
            name=self.name,
            duration_ms=100.0,
            iterations=self.num_iterations,
        )

    def setup(self):
        """Track setup call."""
        super().setup()
        self.setup_called = True

    def teardown(self):
        """Track teardown call."""
        super().teardown()
        self.teardown_called = True


class TestBenchmarkRunner:
    """Test abstract BenchmarkRunner base class."""

    def test_initialization(self):
        """Test runner initialization."""
        runner = DummyRunner(
            num_warmup=3,
            num_iterations=5,
            clear_cache_between_runs=False,
        )

        assert runner.name == "dummy"
        assert runner.num_warmup == 3
        assert runner.num_iterations == 5
        assert runner.clear_cache_between_runs is False

    def test_defaults(self):
        """Test default values."""
        runner = DummyRunner()

        assert runner.num_warmup == 5
        assert runner.num_iterations == 10
        assert runner.clear_cache_between_runs is True

    def test_setup_teardown(self):
        """Test setup and teardown are called."""
        runner = DummyRunner()

        assert not runner.setup_called
        assert not runner.teardown_called

        runner.setup()
        assert runner.setup_called

        runner.teardown()
        assert runner.teardown_called

    def test_run_returns_stats(self):
        """Test run() returns BenchmarkStats."""
        runner = DummyRunner()
        stats = runner.run()

        assert isinstance(stats, BenchmarkStats)
        assert stats.name == "dummy"


@pytest.mark.unit
class TestFunctionBenchmarkRunner:
    """Test FunctionBenchmarkRunner for arbitrary functions."""

    def test_simple_function(self):
        """Test benchmarking a simple function."""
        def simple_add(a, b):
            return a + b

        runner = FunctionBenchmarkRunner(
            name="add_test",
            fn=simple_add,
            args=(5, 3),
            num_warmup=1,
            num_iterations=2,
        )

        stats = runner.run()

        assert isinstance(stats, BenchmarkStats)
        assert stats.name == "add_test"
        assert stats.iterations == 2
        assert stats.duration_ms > 0

    def test_with_kwargs(self):
        """Test benchmarking function with keyword arguments."""
        def func_with_kwargs(a, b=10, c=20):
            return a + b + c

        runner = FunctionBenchmarkRunner(
            name="kwargs_test",
            fn=func_with_kwargs,
            args=(5,),
            kwargs={"b": 15, "c": 25},
            num_warmup=1,
            num_iterations=2,
        )

        stats = runner.run()

        assert isinstance(stats, BenchmarkStats)
        assert stats.name == "kwargs_test"

    @pytest.mark.gpu
    def test_mlx_operation(self):
        """Test benchmarking MLX operation."""
        def matmul_op(x):
            return mx.matmul(x, x.T)

        x = mx.random.normal((100, 100))

        runner = FunctionBenchmarkRunner(
            name="matmul_100x100",
            fn=matmul_op,
            args=(x,),
            num_warmup=2,
            num_iterations=5,
        )

        stats = runner.run()

        assert stats.name == "matmul_100x100"
        assert stats.iterations == 5
        assert stats.duration_ms > 0
        assert stats.peak_memory_gb >= 0

    @pytest.mark.gpu
    def test_memory_tracking(self):
        """Test that memory usage is tracked."""
        def allocate_array():
            # Allocate a reasonably large array
            x = mx.random.normal((1000, 1000))
            mx.eval(x)
            return x

        runner = FunctionBenchmarkRunner(
            name="memory_test",
            fn=allocate_array,
            num_warmup=1,
            num_iterations=2,
        )

        stats = runner.run()

        # Should track some memory usage
        if mx.metal.is_available():
            assert stats.peak_memory_gb > 0


@pytest.mark.unit
class TestOperationBenchmarkRunner:
    """Test OperationBenchmarkRunner for MLX operations."""

    @pytest.mark.gpu
    def test_matmul_benchmark(self):
        """Test benchmarking matmul operation."""
        def matmul_fn(a, b):
            return mx.matmul(a, b)

        a = mx.random.normal((100, 100))
        b = mx.random.normal((100, 100))

        runner = OperationBenchmarkRunner(
            name="matmul_100x100",
            operation="matmul",
            fn=matmul_fn,
            args=(a, b),
            input_shapes=[(100, 100), (100, 100)],
            dtype="float32",
            num_warmup=2,
            num_iterations=10,
        )

        stats = runner.run()

        assert isinstance(stats, OperationBenchmarkStats)
        assert stats.name == "matmul_100x100"
        assert stats.operation == "matmul"
        assert stats.input_shapes == [(100, 100), (100, 100)]
        assert stats.dtype == "float32"
        assert stats.iterations == 10
        assert stats.duration_ms > 0

    @pytest.mark.gpu
    def test_output_shape_detection(self):
        """Test automatic output shape detection."""
        def matmul_fn(a, b):
            return mx.matmul(a, b)

        a = mx.random.normal((50, 100))
        b = mx.random.normal((100, 75))

        runner = OperationBenchmarkRunner(
            name="matmul_50x75",
            operation="matmul",
            fn=matmul_fn,
            args=(a, b),
            input_shapes=[(50, 100), (100, 75)],
            num_warmup=1,
            num_iterations=2,
        )

        stats = runner.run()

        # Output shape should be (50, 75)
        assert stats.output_shape == (50, 75)

    @pytest.mark.gpu
    def test_explicit_output_shape(self):
        """Test with explicit output shape."""
        def matmul_fn(a, b):
            return mx.matmul(a, b)

        a = mx.random.normal((100, 100))
        b = mx.random.normal((100, 100))

        runner = OperationBenchmarkRunner(
            name="matmul_100x100",
            operation="matmul",
            fn=matmul_fn,
            args=(a, b),
            input_shapes=[(100, 100), (100, 100)],
            output_shape=(100, 100),
            num_warmup=1,
            num_iterations=2,
        )

        stats = runner.run()

        assert stats.output_shape == (100, 100)

    @pytest.mark.gpu
    def test_device_detection(self):
        """Test device detection."""
        def simple_op(x):
            return x * 2

        x = mx.random.normal((10, 10))

        runner = OperationBenchmarkRunner(
            name="device_test",
            operation="multiply",
            fn=simple_op,
            args=(x,),
            num_warmup=1,
            num_iterations=2,
        )

        stats = runner.run()

        if mx.metal.is_available():
            assert stats.device == "gpu"
        else:
            assert stats.device == "cpu"


@pytest.mark.unit
class TestQuickBenchmark:
    """Test quick_benchmark convenience function."""

    def test_simple_benchmark(self):
        """Test quick benchmark with simple function."""
        def simple_fn(x):
            return x * 2

        stats = quick_benchmark(
            simple_fn,
            5,
            name="multiply_test",
            num_warmup=1,
            num_iterations=2,
        )

        assert isinstance(stats, BenchmarkStats)
        assert stats.name == "multiply_test"
        assert stats.iterations == 2

    @pytest.mark.gpu
    def test_mlx_lambda(self):
        """Test quick benchmark with MLX lambda."""
        x = mx.random.normal((100, 100))

        stats = quick_benchmark(
            lambda arr: arr @ arr.T,
            x,
            name="matmul_lambda",
            num_warmup=1,
            num_iterations=3,
        )

        assert stats.name == "matmul_lambda"
        assert stats.iterations == 3
        assert stats.duration_ms > 0

    def test_with_kwargs(self):
        """Test quick benchmark with keyword arguments."""
        def func(a, b, c=10):
            return a + b + c

        stats = quick_benchmark(
            func,
            1, 2,
            c=20,
            name="kwargs_bench",
            num_warmup=1,
            num_iterations=2,
        )

        assert stats.name == "kwargs_bench"


@pytest.mark.unit
class TestCompareImplementations:
    """Test compare_implementations function."""

    def test_compare_two_functions(self):
        """Test comparing two implementations."""
        def baseline(x):
            # Slower implementation
            result = x
            for _ in range(10):
                result = result + 1
            return result

        def optimized(x):
            # Faster implementation
            return x + 10

        results = compare_implementations(
            baseline_fn=baseline,
            comparison_fn=optimized,
            args=(5,),
            num_warmup=1,
            num_iterations=5,
        )

        assert "baseline" in results
        assert "comparison" in results
        assert "speedup" in results
        assert "memory_reduction_gb" in results

        assert isinstance(results["baseline"], BenchmarkStats)
        assert isinstance(results["comparison"], BenchmarkStats)
        assert results["baseline"].name == "baseline"
        assert results["comparison"].name == "comparison"

    def test_speedup_calculation(self):
        """Test speedup is calculated correctly."""
        import time

        def slow_fn(x):
            time.sleep(0.001)  # 1ms delay
            return x

        def fast_fn(x):
            return x

        results = compare_implementations(
            baseline_fn=slow_fn,
            comparison_fn=fast_fn,
            args=(5,),
            num_warmup=0,
            num_iterations=2,
        )

        # Speedup should be > 1 (baseline slower than comparison)
        assert results["speedup"] > 1.0

    @pytest.mark.gpu
    def test_compare_mlx_implementations(self):
        """Test comparing MLX implementations."""
        def baseline(x):
            # Slower: multiple operations
            return mx.exp(mx.log(x + 1))

        def optimized(x):
            # Faster: direct operation
            return x + 1

        x = mx.random.normal((100, 100))

        results = compare_implementations(
            baseline_fn=baseline,
            comparison_fn=optimized,
            args=(x,),
            num_warmup=2,
            num_iterations=5,
        )

        # Both should complete
        assert results["baseline"].duration_ms > 0
        assert results["comparison"].duration_ms > 0
        assert results["speedup"] > 0

    def test_custom_names(self):
        """Test with custom names."""
        def fn1(x):
            return x

        def fn2(x):
            return x * 2

        results = compare_implementations(
            baseline_fn=fn1,
            comparison_fn=fn2,
            args=(5,),
            baseline_name="identity",
            comparison_name="double",
            num_warmup=1,
            num_iterations=2,
        )

        assert results["baseline"].name == "identity"
        assert results["comparison"].name == "double"

    def test_with_kwargs(self):
        """Test comparison with keyword arguments."""
        def fn(a, b=10):
            return a + b

        results = compare_implementations(
            baseline_fn=fn,
            comparison_fn=fn,
            args=(5,),
            kwargs={"b": 20},
            num_warmup=1,
            num_iterations=2,
        )

        # Both should run with same kwargs
        assert results["baseline"].iterations == 2
        assert results["comparison"].iterations == 2


@pytest.mark.integration
@pytest.mark.gpu
class TestBenchmarkRunnerIntegration:
    """Integration tests for benchmark runners."""

    def test_end_to_end_function_benchmark(self):
        """Test complete function benchmark workflow."""
        def complex_operation(x):
            """Simulate a complex MLX operation."""
            result = x
            for _ in range(5):
                result = mx.matmul(result, result.T)
                if result.shape[0] > 10:
                    result = result[:10, :10]
            return result

        x = mx.random.normal((10, 10))

        runner = FunctionBenchmarkRunner(
            name="complex_op",
            fn=complex_operation,
            args=(x,),
            num_warmup=2,
            num_iterations=5,
            clear_cache_between_runs=True,
        )

        stats = runner.run()

        assert stats.name == "complex_op"
        assert stats.iterations == 5
        assert stats.duration_ms > 0
        assert stats.peak_memory_gb >= 0

    def test_multiple_benchmarks(self):
        """Test running multiple benchmarks sequentially."""
        def op1(x):
            return x @ x.T

        def op2(x):
            return mx.sum(x)

        def op3(x):
            return mx.softmax(x)

        x = mx.random.normal((100, 100))

        results = []
        for name, fn in [("matmul", op1), ("sum", op2), ("softmax", op3)]:
            stats = quick_benchmark(
                fn, x,
                name=name,
                num_warmup=1,
                num_iterations=3,
            )
            results.append(stats)

        assert len(results) == 3
        assert all(s.iterations == 3 for s in results)
        assert all(s.duration_ms > 0 for s in results)

    def test_benchmark_with_large_arrays(self):
        """Test benchmarking with larger arrays."""
        def large_matmul(a, b):
            return mx.matmul(a, b)

        # Use reasonably large arrays
        a = mx.random.normal((500, 500))
        b = mx.random.normal((500, 500))

        runner = OperationBenchmarkRunner(
            name="large_matmul",
            operation="matmul",
            fn=large_matmul,
            args=(a, b),
            input_shapes=[(500, 500), (500, 500)],
            num_warmup=1,
            num_iterations=3,
        )

        stats = runner.run()

        assert stats.duration_ms > 0
        if mx.metal.is_available():
            # Memory tracking should return a value >= 0
            # (may be 0 for very small operations or if tracking fails)
            assert stats.peak_memory_gb >= 0
