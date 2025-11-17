"""
Tests for benchmark statistics dataclasses.

Tests the core BenchmarkStats classes used throughout the benchmark suite.
"""

import pytest
from datetime import datetime

from smlx.bench.stats import (
    BenchmarkStats,
    BenchmarkSuite,
    ComparisonStats,
    ModelBenchmarkStats,
    OperationBenchmarkStats,
    create_model_stats,
)


class TestBenchmarkStats:
    """Test BenchmarkStats base class."""

    def test_creation(self):
        """Test creating basic benchmark stats."""
        stats = BenchmarkStats(
            name="test_benchmark",
            duration_ms=100.0,
            iterations=10,
            peak_memory_gb=2.5,
        )

        assert stats.name == "test_benchmark"
        assert stats.duration_ms == 100.0
        assert stats.iterations == 10
        assert stats.peak_memory_gb == 2.5
        assert stats.timestamp  # Should have timestamp
        assert isinstance(stats.metadata, dict)

    def test_duration_per_iter(self):
        """Test duration per iteration calculation."""
        stats = BenchmarkStats(
            name="test",
            duration_ms=100.0,
            iterations=10,
        )
        assert stats.duration_per_iter_ms == 10.0

    def test_duration_per_iter_zero_iterations(self):
        """Test duration per iteration with zero iterations."""
        stats = BenchmarkStats(
            name="test",
            duration_ms=100.0,
            iterations=0,
        )
        assert stats.duration_per_iter_ms == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = BenchmarkStats(
            name="test",
            duration_ms=100.0,
            iterations=10,
            peak_memory_gb=2.5,
        )
        data = stats.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["duration_ms"] == 100.0
        assert data["iterations"] == 10
        assert data["peak_memory_gb"] == 2.5

    def test_metadata(self):
        """Test metadata field."""
        stats = BenchmarkStats(
            name="test",
            metadata={"foo": "bar", "num": 42},
        )
        assert stats.metadata["foo"] == "bar"
        assert stats.metadata["num"] == 42


class TestModelBenchmarkStats:
    """Test ModelBenchmarkStats for model inference."""

    def test_creation(self):
        """Test creating model benchmark stats."""
        stats = ModelBenchmarkStats(
            name="llm_benchmark",
            model_name="SmolLM2-135M",
            prompt_tokens=100,
            generation_tokens=50,
            prompt_time=0.5,
            generation_time=2.0,
            prompt_tps=200.0,
            generation_tps=25.0,
            peak_memory_gb=2.3,
            quantization="4bit",
            batch_size=1,
        )

        assert stats.model_name == "SmolLM2-135M"
        assert stats.prompt_tokens == 100
        assert stats.generation_tokens == 50
        assert stats.quantization == "4bit"

    def test_total_tokens(self):
        """Test total tokens calculation."""
        stats = ModelBenchmarkStats(
            name="test",
            prompt_tokens=100,
            generation_tokens=50,
        )
        assert stats.total_tokens == 150

    def test_total_time(self):
        """Test total time calculation."""
        stats = ModelBenchmarkStats(
            name="test",
            prompt_time=0.5,
            generation_time=2.0,
        )
        assert stats.total_time == 2.5

    def test_time_to_first_token(self):
        """Test TTFT calculation."""
        stats = ModelBenchmarkStats(
            name="test",
            prompt_time=0.5,
        )
        assert stats.time_to_first_token == 0.5

    def test_overall_tps(self):
        """Test overall TPS calculation."""
        stats = ModelBenchmarkStats(
            name="test",
            prompt_tokens=100,
            generation_tokens=50,
            prompt_time=0.5,
            generation_time=2.0,
        )
        # Total: 150 tokens / 2.5 seconds = 60 tok/s
        assert stats.overall_tps == pytest.approx(60.0)

    def test_overall_tps_zero_time(self):
        """Test overall TPS with zero time."""
        stats = ModelBenchmarkStats(
            name="test",
            prompt_tokens=100,
            generation_tokens=50,
            prompt_time=0.0,
            generation_time=0.0,
        )
        assert stats.overall_tps == 0.0


class TestOperationBenchmarkStats:
    """Test OperationBenchmarkStats for low-level operations."""

    def test_creation(self):
        """Test creating operation benchmark stats."""
        stats = OperationBenchmarkStats(
            name="matmul_1000x1000",
            operation="matmul",
            duration_ms=10.5,
            iterations=100,
            input_shapes=[(1000, 1000), (1000, 1000)],
            output_shape=(1000, 1000),
            dtype="float32",
            device="gpu",
        )

        assert stats.operation == "matmul"
        assert stats.input_shapes == [(1000, 1000), (1000, 1000)]
        assert stats.output_shape == (1000, 1000)
        assert stats.dtype == "float32"
        assert stats.device == "gpu"

    def test_gflops(self):
        """Test GFLOPS calculation."""
        stats = OperationBenchmarkStats(
            name="test",
            duration_ms=10.0,  # 0.01 seconds
            flops=1e9,  # 1 billion FLOPs
        )
        # 1e9 FLOPs / 0.01s = 100 GFLOPS
        assert stats.gflops == pytest.approx(100.0)

    def test_gflops_zero_duration(self):
        """Test GFLOPS with zero duration."""
        stats = OperationBenchmarkStats(
            name="test",
            duration_ms=0.0,
            flops=1e9,
        )
        assert stats.gflops == 0.0

    def test_gflops_none(self):
        """Test GFLOPS when flops not specified."""
        stats = OperationBenchmarkStats(
            name="test",
            duration_ms=10.0,
        )
        assert stats.gflops == 0.0

    def test_throughput(self):
        """Test throughput calculation."""
        stats = OperationBenchmarkStats(
            name="test",
            duration_ms=1000.0,  # 1 second
            iterations=100,
        )
        # 100 iterations / 1 second = 100 ops/s
        assert stats.throughput == 100.0


class TestComparisonStats:
    """Test ComparisonStats for comparing benchmarks."""

    def test_creation(self):
        """Test creating comparison stats."""
        baseline = BenchmarkStats(
            name="baseline",
            duration_ms=100.0,
            peak_memory_gb=4.0,
        )
        comparison = BenchmarkStats(
            name="optimized",
            duration_ms=50.0,
            peak_memory_gb=2.0,
        )
        comp = ComparisonStats(baseline=baseline, comparison=comparison)

        assert comp.baseline.name == "baseline"
        assert comp.comparison.name == "optimized"

    def test_speedup(self):
        """Test speedup calculation."""
        baseline = BenchmarkStats(
            name="baseline",
            duration_ms=100.0,
        )
        comparison = BenchmarkStats(
            name="optimized",
            duration_ms=50.0,
        )
        comp = ComparisonStats(baseline=baseline, comparison=comparison)

        # 100ms / 50ms = 2x speedup
        assert comp.speedup == 2.0

    def test_speedup_zero_comparison(self):
        """Test speedup with zero comparison time."""
        baseline = BenchmarkStats(
            name="baseline",
            duration_ms=100.0,
        )
        comparison = BenchmarkStats(
            name="optimized",
            duration_ms=0.0,
        )
        comp = ComparisonStats(baseline=baseline, comparison=comparison)

        assert comp.speedup == 0.0

    def test_memory_reduction(self):
        """Test memory reduction calculation."""
        baseline = BenchmarkStats(
            name="baseline",
            peak_memory_gb=4.0,
        )
        comparison = BenchmarkStats(
            name="optimized",
            peak_memory_gb=2.0,
        )
        comp = ComparisonStats(baseline=baseline, comparison=comparison)

        # 4GB - 2GB = 2GB reduction
        assert comp.memory_reduction == 2.0

    def test_memory_reduction_percent(self):
        """Test memory reduction percentage."""
        baseline = BenchmarkStats(
            name="baseline",
            peak_memory_gb=4.0,
        )
        comparison = BenchmarkStats(
            name="optimized",
            peak_memory_gb=2.0,
        )
        comp = ComparisonStats(baseline=baseline, comparison=comparison)

        # (2GB / 4GB) * 100 = 50%
        assert comp.memory_reduction_percent == 50.0

    def test_memory_reduction_percent_zero_baseline(self):
        """Test memory reduction percentage with zero baseline."""
        baseline = BenchmarkStats(
            name="baseline",
            peak_memory_gb=0.0,
        )
        comparison = BenchmarkStats(
            name="optimized",
            peak_memory_gb=2.0,
        )
        comp = ComparisonStats(baseline=baseline, comparison=comparison)

        assert comp.memory_reduction_percent == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        baseline = BenchmarkStats(name="baseline", duration_ms=100.0)
        comparison = BenchmarkStats(name="optimized", duration_ms=50.0)
        comp = ComparisonStats(baseline=baseline, comparison=comparison)

        data = comp.to_dict()

        assert isinstance(data, dict)
        assert "baseline" in data
        assert "comparison" in data
        assert "speedup" in data
        assert "memory_reduction_gb" in data
        assert "memory_reduction_percent" in data


class TestBenchmarkSuite:
    """Test BenchmarkSuite for collections of benchmarks."""

    def test_creation(self):
        """Test creating benchmark suite."""
        suite = BenchmarkSuite(name="test_suite")

        assert suite.name == "test_suite"
        assert len(suite.benchmarks) == 0
        assert suite.timestamp

    def test_add_benchmark(self):
        """Test adding benchmarks to suite."""
        suite = BenchmarkSuite(name="test_suite")
        stats1 = BenchmarkStats(name="test1", duration_ms=100.0)
        stats2 = BenchmarkStats(name="test2", duration_ms=200.0)

        suite.add(stats1)
        suite.add(stats2)

        assert len(suite.benchmarks) == 2

    def test_total_duration(self):
        """Test total duration calculation."""
        suite = BenchmarkSuite(name="test_suite")
        suite.add(BenchmarkStats(name="test1", duration_ms=100.0))
        suite.add(BenchmarkStats(name="test2", duration_ms=200.0))

        assert suite.total_duration_ms == 300.0

    def test_mean_duration(self):
        """Test mean duration calculation."""
        suite = BenchmarkSuite(name="test_suite")
        suite.add(BenchmarkStats(name="test1", duration_ms=100.0))
        suite.add(BenchmarkStats(name="test2", duration_ms=200.0))

        assert suite.mean_duration_ms == 150.0

    def test_mean_duration_empty(self):
        """Test mean duration with empty suite."""
        suite = BenchmarkSuite(name="test_suite")
        assert suite.mean_duration_ms == 0.0

    def test_peak_memory(self):
        """Test peak memory calculation."""
        suite = BenchmarkSuite(name="test_suite")
        suite.add(BenchmarkStats(name="test1", peak_memory_gb=2.0))
        suite.add(BenchmarkStats(name="test2", peak_memory_gb=3.5))
        suite.add(BenchmarkStats(name="test3", peak_memory_gb=1.0))

        assert suite.peak_memory_gb == 3.5

    def test_peak_memory_empty(self):
        """Test peak memory with empty suite."""
        suite = BenchmarkSuite(name="test_suite")
        assert suite.peak_memory_gb == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        suite = BenchmarkSuite(name="test_suite")
        suite.add(BenchmarkStats(name="test1", duration_ms=100.0))
        suite.add(BenchmarkStats(name="test2", duration_ms=200.0))

        data = suite.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "test_suite"
        assert "timestamp" in data
        assert "benchmarks" in data
        assert len(data["benchmarks"]) == 2
        assert "summary" in data
        assert data["summary"]["count"] == 2


class TestCreateModelStats:
    """Test create_model_stats helper function."""

    def test_basic_creation(self):
        """Test creating model stats with basic parameters."""
        stats = create_model_stats(
            model_name="SmolLM2-135M",
            prompt_tokens=100,
            prompt_time=0.5,
            generation_tokens=50,
            generation_time=2.0,
            peak_memory_gb=2.3,
        )

        assert stats.model_name == "SmolLM2-135M"
        assert stats.prompt_tokens == 100
        assert stats.generation_tokens == 50
        assert stats.peak_memory_gb == 2.3

    def test_tps_calculation(self):
        """Test TPS is automatically calculated."""
        stats = create_model_stats(
            model_name="test",
            prompt_tokens=100,
            prompt_time=0.5,
            generation_tokens=50,
            generation_time=2.0,
            peak_memory_gb=2.0,
        )

        # Prompt TPS: 100 / 0.5 = 200
        assert stats.prompt_tps == pytest.approx(200.0)
        # Generation TPS: 50 / 2.0 = 25
        assert stats.generation_tps == pytest.approx(25.0)

    def test_tps_zero_time(self):
        """Test TPS with zero time."""
        stats = create_model_stats(
            model_name="test",
            prompt_tokens=100,
            prompt_time=0.0,
            generation_tokens=50,
            generation_time=0.0,
            peak_memory_gb=2.0,
        )

        assert stats.prompt_tps == 0.0
        assert stats.generation_tps == 0.0

    def test_duration_ms(self):
        """Test duration_ms is automatically calculated."""
        stats = create_model_stats(
            model_name="test",
            prompt_tokens=100,
            prompt_time=0.5,
            generation_tokens=50,
            generation_time=2.0,
            peak_memory_gb=2.0,
        )

        # Total time: 0.5 + 2.0 = 2.5 seconds = 2500ms
        assert stats.duration_ms == pytest.approx(2500.0)

    def test_with_quantization(self):
        """Test creating stats with quantization info."""
        stats = create_model_stats(
            model_name="test",
            prompt_tokens=100,
            prompt_time=0.5,
            generation_tokens=50,
            generation_time=2.0,
            peak_memory_gb=2.0,
            quantization="4bit",
        )

        assert stats.quantization == "4bit"

    def test_with_batch_size(self):
        """Test creating stats with batch size."""
        stats = create_model_stats(
            model_name="test",
            prompt_tokens=100,
            prompt_time=0.5,
            generation_tokens=50,
            generation_time=2.0,
            peak_memory_gb=2.0,
            batch_size=8,
        )

        assert stats.batch_size == 8

    def test_with_metadata(self):
        """Test creating stats with additional metadata."""
        stats = create_model_stats(
            model_name="test",
            prompt_tokens=100,
            prompt_time=0.5,
            generation_tokens=50,
            generation_time=2.0,
            peak_memory_gb=2.0,
            custom_field="custom_value",
            temperature=0.7,
        )

        assert stats.metadata["custom_field"] == "custom_value"
        assert stats.metadata["temperature"] == 0.7


@pytest.mark.unit
class TestBenchmarkStatsUnit:
    """Unit tests for benchmark stats classes."""

    def test_benchmark_stats_defaults(self):
        """Test default values for BenchmarkStats."""
        stats = BenchmarkStats(name="test")

        assert stats.duration_ms == 0.0
        assert stats.iterations == 1
        assert stats.peak_memory_gb == 0.0
        assert isinstance(stats.metadata, dict)
        assert len(stats.metadata) == 0

    def test_model_stats_defaults(self):
        """Test default values for ModelBenchmarkStats."""
        stats = ModelBenchmarkStats(name="test")

        assert stats.prompt_tokens == 0
        assert stats.generation_tokens == 0
        assert stats.prompt_time == 0.0
        assert stats.generation_time == 0.0
        assert stats.prompt_tps == 0.0
        assert stats.generation_tps == 0.0
        assert stats.model_name == ""
        assert stats.quantization is None
        assert stats.batch_size == 1

    def test_operation_stats_defaults(self):
        """Test default values for OperationBenchmarkStats."""
        stats = OperationBenchmarkStats(name="test")

        assert stats.operation == ""
        assert stats.input_shapes == []
        assert stats.output_shape is None
        assert stats.dtype == "float32"
        assert stats.device == "gpu"
        assert stats.flops is None
