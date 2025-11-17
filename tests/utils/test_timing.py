"""Tests for smlx.utils.timing module."""

import time

import mlx.core as mx
import pytest

from smlx.utils.timing import (
    Timer,
    benchmark,
    measure_runtime,
    timer,
)


class TestTimer:
    """Test timer function."""

    def test_timer_basic(self):
        """Test basic timing with warmup and iterations."""

        def simple_fn(x):
            return x + 1

        result, times = timer(simple_fn, 5, num_warmup=2, num_iterations=5)

        assert result == 6
        assert len(times) == 5
        assert all(isinstance(t, float) for t in times)
        assert all(t >= 0 for t in times)

    def test_timer_with_mlx_arrays(self):
        """Test timing with MLX arrays (forces evaluation)."""

        def mlx_fn(a, b):
            return a @ b

        a = mx.random.normal((10, 10))
        b = mx.random.normal((10, 10))

        result, times = timer(mlx_fn, a, b, num_warmup=2, num_iterations=3)

        assert result.shape == (10, 10)
        assert len(times) == 3
        assert all(t >= 0 for t in times)

    def test_timer_multiple_returns(self):
        """Test timing with multiple return values."""

        def multi_return_fn(x):
            return mx.array([x, x + 1, x + 2])

        result, times = timer(multi_return_fn, 5, num_warmup=1, num_iterations=2)

        assert isinstance(result, mx.array)
        assert len(times) == 2

    def test_timer_no_eval(self):
        """Test timing without forcing evaluation."""

        def mlx_fn(a):
            return a * 2

        a = mx.array([1, 2, 3])
        result, times = timer(mlx_fn, a, num_iterations=3, force_eval=False)

        assert isinstance(result, mx.array)
        assert len(times) == 3


class TestBenchmarkDecorator:
    """Test benchmark decorator."""

    def test_benchmark_decorator_basic(self):
        """Test basic benchmark decorator."""

        @benchmark(num_warmup=2, num_iterations=3)
        def decorated_fn(x):
            return x * 2

        result, times = decorated_fn(5)

        assert result == 10
        assert len(times) == 3

    def test_benchmark_decorator_with_mlx(self):
        """Test benchmark decorator with MLX operations."""

        @benchmark(num_warmup=1, num_iterations=2, force_eval=True)
        def mlx_operation(x):
            return mx.sum(x)

        arr = mx.array([1, 2, 3, 4, 5])
        result, times = mlx_operation(arr)

        assert result.item() == 15
        assert len(times) == 2


class TestTimerContextManager:
    """Test Timer context manager."""

    def test_timer_context_basic(self):
        """Test basic Timer context manager."""
        with Timer(force_eval=False) as t:
            time.sleep(0.01)

        assert t.elapsed >= 0.01
        assert t.elapsed_ms >= 10
        assert t.start_time is not None
        assert t.end_time is not None

    def test_timer_context_with_mlx(self):
        """Test Timer with MLX operations."""
        with Timer(force_eval=True) as t:
            a = mx.random.normal((100, 100))
            b = mx.random.normal((100, 100))
            c = a @ b

        assert t.elapsed >= 0
        assert t.elapsed_ms >= 0

    def test_timer_elapsed_during_execution(self):
        """Test that elapsed can be checked during execution."""
        with Timer(force_eval=False) as t:
            time.sleep(0.01)
            mid_elapsed = t.elapsed
            time.sleep(0.01)

        assert mid_elapsed >= 0.01
        assert t.elapsed >= mid_elapsed


class TestMeasureRuntime:
    """Test measure_runtime function."""

    def test_measure_runtime_basic(self):
        """Test basic runtime measurement."""

        def simple_fn(x, y):
            return x + y

        runtime_ms = measure_runtime(simple_fn, 5, 10, num_warmup=2, num_iterations=5)

        assert isinstance(runtime_ms, float)
        assert runtime_ms >= 0

    def test_measure_runtime_with_mlx(self):
        """Test runtime measurement with MLX operations."""

        def matmul(a, b):
            return a @ b

        a = mx.random.normal((50, 50))
        b = mx.random.normal((50, 50))

        runtime_ms = measure_runtime(matmul, a, b, num_warmup=2, num_iterations=10)

        assert isinstance(runtime_ms, float)
        assert runtime_ms > 0

    def test_measure_runtime_consistency(self):
        """Test that measure_runtime gives consistent results."""

        def constant_fn():
            arr = mx.ones((10, 10))
            return mx.sum(arr)

        runtime1 = measure_runtime(constant_fn, num_warmup=5, num_iterations=20)
        runtime2 = measure_runtime(constant_fn, num_warmup=5, num_iterations=20)

        # Should be in the same order of magnitude
        assert 0.1 < (runtime1 / runtime2) < 10


class TestTimingEdgeCases:
    """Test edge cases and error conditions."""

    def test_timer_with_zero_iterations(self):
        """Test timer with zero iterations (should still return valid structure)."""

        def simple_fn():
            return 42

        result, times = timer(simple_fn, num_warmup=0, num_iterations=0)

        # After warmup but no iterations, should have empty times
        assert len(times) == 0

    def test_timer_with_exception(self):
        """Test that timer properly handles exceptions."""

        def error_fn():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            timer(error_fn, num_iterations=1)

    def test_timer_context_with_exception(self):
        """Test Timer context manager with exceptions."""
        try:
            with Timer() as t:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Timer should still record end time
        assert t.end_time is not None
        assert t.elapsed >= 0
