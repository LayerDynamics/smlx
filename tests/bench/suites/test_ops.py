"""
Tests for operation benchmark suite.

Tests the low-level MLX operation benchmarking functions.
"""

import pytest
import mlx.core as mx

from smlx.bench.suites.ops import (
    benchmark_attention,
    benchmark_gelu,
    benchmark_layernorm,
    benchmark_matmul,
    benchmark_operation,
    run_ops_suite,
)
from smlx.bench.stats import BenchmarkSuite, OperationBenchmarkStats


@pytest.mark.unit
class TestBenchmarkOperation:
    """Test benchmark_operation function."""

    @pytest.mark.gpu
    def test_simple_operation(self):
        """Test benchmarking a simple operation."""
        def add_op(x, y):
            return x + y

        x = mx.random.normal((10, 10))
        y = mx.random.normal((10, 10))

        stats = benchmark_operation(
            "add_test",
            add_op,
            x,
            y,
            input_shapes=[(10, 10), (10, 10)],
            num_warmup=1,
            num_iterations=2,
        )

        assert isinstance(stats, OperationBenchmarkStats)
        assert stats.name == "add_test"
        assert stats.operation == "add_test"
        assert stats.input_shapes == [(10, 10), (10, 10)]
        assert stats.iterations == 2
        assert stats.duration_ms > 0

    @pytest.mark.gpu
    def test_with_auto_shape_detection(self):
        """Test automatic input shape detection."""
        def multiply_op(x):
            return x * 2

        x = mx.random.normal((50, 50))

        stats = benchmark_operation(
            "multiply",
            multiply_op,
            x,
            num_warmup=1,
            num_iterations=2,
        )

        # Should auto-detect shape
        assert stats.input_shapes == [(50, 50)]

    @pytest.mark.gpu
    def test_with_kwargs(self):
        """Test operation with keyword arguments."""
        def scale_op(x, scale=1.0):
            return x * scale

        x = mx.random.normal((10, 10))

        stats = benchmark_operation(
            "scale",
            scale_op,
            x,
            scale=2.5,
            num_warmup=1,
            num_iterations=2,
        )

        assert isinstance(stats, OperationBenchmarkStats)


@pytest.mark.unit
@pytest.mark.gpu
class TestBenchmarkMatmul:
    """Test benchmark_matmul function."""

    def test_square_matrices(self):
        """Test matmul with square matrices."""
        stats = benchmark_matmul(
            shape_a=(100, 100),
            shape_b=(100, 100),
            num_iterations=5,
        )

        assert isinstance(stats, OperationBenchmarkStats)
        assert stats.operation == "matmul_(100, 100)x(100, 100)"
        assert stats.input_shapes == [(100, 100), (100, 100)]
        assert stats.iterations == 5
        assert stats.duration_ms > 0
        assert stats.output_shape == (100, 100)

    def test_rectangular_matrices(self):
        """Test matmul with rectangular matrices."""
        stats = benchmark_matmul(
            shape_a=(50, 100),
            shape_b=(100, 75),
            num_iterations=5,
        )

        assert stats.input_shapes == [(50, 100), (100, 75)]
        assert stats.output_shape == (50, 75)

    def test_different_dtypes(self):
        """Test matmul with different data types."""
        for dtype in [mx.float32, mx.float16]:
            stats = benchmark_matmul(
                shape_a=(50, 50),
                shape_b=(50, 50),
                dtype=dtype,
                num_iterations=2,
            )

            assert stats.dtype == str(dtype)


@pytest.mark.unit
@pytest.mark.gpu
class TestBenchmarkAttention:
    """Test benchmark_attention function."""

    def test_default_parameters(self):
        """Test attention with default parameters."""
        stats = benchmark_attention(num_iterations=2)

        assert isinstance(stats, OperationBenchmarkStats)
        assert "attention" in stats.operation
        assert stats.iterations == 2
        assert stats.duration_ms > 0

    def test_custom_dimensions(self):
        """Test attention with custom dimensions."""
        stats = benchmark_attention(
            batch_size=2,
            seq_len=256,
            num_heads=4,
            head_dim=32,
            num_iterations=2,
        )

        assert "attention_b2_l256_h4" in stats.operation
        # Shape: (batch, seq_len, num_heads, head_dim)
        expected_shape = (2, 256, 4, 32)
        assert expected_shape in stats.input_shapes

    def test_different_dtypes(self):
        """Test attention with different data types."""
        for dtype in [mx.float16, mx.float32]:
            stats = benchmark_attention(
                batch_size=1,
                seq_len=128,
                dtype=dtype,
                num_iterations=2,
            )

            assert stats.dtype == str(dtype)


@pytest.mark.unit
@pytest.mark.gpu
class TestBenchmarkLayernorm:
    """Test benchmark_layernorm function."""

    def test_basic_layernorm(self):
        """Test basic layer normalization."""
        shape = (32, 512, 768)
        stats = benchmark_layernorm(shape=shape, num_iterations=5)

        assert isinstance(stats, OperationBenchmarkStats)
        assert f"layernorm_{shape}" in stats.operation
        assert stats.input_shapes == [shape]
        assert stats.iterations == 5
        assert stats.duration_ms > 0

    def test_different_shapes(self):
        """Test layernorm with different shapes."""
        for shape in [(16, 256), (32, 512, 768), (8, 128, 1024)]:
            stats = benchmark_layernorm(shape=shape, num_iterations=2)

            assert stats.input_shapes == [shape]

    def test_different_dtypes(self):
        """Test layernorm with different data types."""
        for dtype in [mx.float32, mx.float16]:
            stats = benchmark_layernorm(
                shape=(16, 128),
                dtype=dtype,
                num_iterations=2,
            )

            assert stats.dtype == str(dtype)


@pytest.mark.unit
@pytest.mark.gpu
class TestBenchmarkGelu:
    """Test benchmark_gelu function."""

    def test_basic_gelu(self):
        """Test basic GELU activation."""
        shape = (1000, 1000)
        stats = benchmark_gelu(shape=shape, num_iterations=10)

        assert isinstance(stats, OperationBenchmarkStats)
        assert f"gelu_{shape}" in stats.operation
        assert stats.input_shapes == [shape]
        assert stats.iterations == 10
        assert stats.duration_ms > 0

    def test_different_shapes(self):
        """Test GELU with different shapes."""
        for shape in [(100,), (100, 100), (10, 100, 100)]:
            stats = benchmark_gelu(shape=shape, num_iterations=2)

            assert stats.input_shapes == [shape]


@pytest.mark.unit
@pytest.mark.gpu
class TestRunOpsSuite:
    """Test run_ops_suite function."""

    def test_matmul_only(self):
        """Test running only matmul benchmark."""
        suite = run_ops_suite(operation="matmul", num_iterations=2)

        assert isinstance(suite, BenchmarkSuite)
        assert suite.name == "MLX Operations"
        assert len(suite.benchmarks) == 1
        assert "matmul" in suite.benchmarks[0].operation

    def test_attention_only(self):
        """Test running only attention benchmark."""
        suite = run_ops_suite(operation="attention", num_iterations=2)

        assert len(suite.benchmarks) == 1
        assert "attention" in suite.benchmarks[0].operation

    def test_layernorm_only(self):
        """Test running only layernorm benchmark."""
        suite = run_ops_suite(operation="layernorm", num_iterations=2)

        assert len(suite.benchmarks) == 1
        assert "layernorm" in suite.benchmarks[0].operation

    def test_all_operations(self):
        """Test running all operations."""
        suite = run_ops_suite(operation="all", num_iterations=2)

        # Should have matmul, attention, and layernorm
        assert len(suite.benchmarks) >= 3
        operation_names = [b.operation for b in suite.benchmarks]
        assert any("matmul" in name for name in operation_names)
        assert any("attention" in name for name in operation_names)
        assert any("layernorm" in name for name in operation_names)

    def test_custom_shape(self):
        """Test with custom matmul shape."""
        suite = run_ops_suite(
            operation="matmul",
            shape="500,500",
            num_iterations=2,
        )

        assert len(suite.benchmarks) == 1
        # Shape should be (500, 500)
        assert (500, 500) in suite.benchmarks[0].input_shapes

    def test_invalid_shape_format(self):
        """Test with invalid shape format (should use defaults)."""
        suite = run_ops_suite(
            operation="matmul",
            shape="500",  # Invalid - only one dimension
            num_iterations=2,
        )

        # Should fall back to default shape
        assert len(suite.benchmarks) == 1


@pytest.mark.integration
@pytest.mark.gpu
class TestOpsIntegration:
    """Integration tests for operation benchmarks."""

    def test_complete_ops_workflow(self):
        """Test complete operation benchmark workflow."""
        # Run full suite
        suite = run_ops_suite(operation="all", num_iterations=3)

        # Verify all benchmarks completed
        assert len(suite.benchmarks) >= 3

        # Check statistics
        for benchmark in suite.benchmarks:
            assert benchmark.duration_ms > 0
            assert benchmark.iterations == 3
            assert len(benchmark.input_shapes) > 0

        # Calculate summary stats
        total_duration = suite.total_duration_ms
        mean_duration = suite.mean_duration_ms

        assert total_duration > 0
        assert mean_duration > 0

    def test_matmul_different_sizes(self):
        """Test matmul with various sizes."""
        sizes = [(100, 100), (500, 500), (1000, 1000)]

        results = []
        for shape_a in sizes:
            shape_b = shape_a  # Square matrices
            stats = benchmark_matmul(
                shape_a=shape_a,
                shape_b=shape_b,
                num_iterations=3,
            )
            results.append(stats)

        # Larger matrices should generally take longer
        assert len(results) == 3
        for stats in results:
            assert stats.duration_ms > 0

    def test_attention_scaling(self):
        """Test how attention performance scales."""
        configs = [
            (1, 128, 8, 64),  # Small
            (1, 256, 8, 64),  # Medium
            (1, 512, 8, 64),  # Large
        ]

        results = []
        for batch, seq_len, heads, head_dim in configs:
            stats = benchmark_attention(
                batch_size=batch,
                seq_len=seq_len,
                num_heads=heads,
                head_dim=head_dim,
                num_iterations=2,
            )
            results.append(stats)

        # All should complete successfully
        assert len(results) == 3
        for stats in results:
            assert stats.duration_ms > 0


@pytest.mark.benchmark
@pytest.mark.gpu
class TestOpsPerformance:
    """Performance tests for operation benchmarks."""

    def test_benchmark_overhead(self):
        """Test that benchmark overhead is minimal."""
        import time

        # Create small operation
        def simple_op(x):
            return x + 1

        x = mx.random.normal((10, 10))

        # Measure total time
        start = time.perf_counter()
        stats = benchmark_operation(
            "add",
            simple_op,
            x,
            num_warmup=1,
            num_iterations=5,
        )
        elapsed = time.perf_counter() - start

        # Overhead should be reasonable
        # For very fast operations (sub-millisecond), overhead can be proportionally large
        # due to timing infrastructure, memory tracking, etc.
        # Allow more generous overhead: 100x for sub-ms operations, 10x otherwise
        overhead_factor = 100 if stats.duration_ms < 1.0 else 10
        assert elapsed < (stats.duration_ms / 1000) * overhead_factor

    def test_consistent_results(self):
        """Test that results are consistent across runs."""
        shape = (100, 100)

        results = []
        for _ in range(3):
            stats = benchmark_matmul(
                shape_a=shape,
                shape_b=shape,
                num_iterations=5,
            )
            results.append(stats)

        # All should have same iteration count
        assert all(s.iterations == 5 for s in results)

        # Durations should be similar (within 50% variance for safety)
        mean_duration = sum(s.duration_ms for s in results) / len(results)
        for stats in results:
            variance = abs(stats.duration_ms - mean_duration) / mean_duration
            assert variance < 0.5  # 50% variance
