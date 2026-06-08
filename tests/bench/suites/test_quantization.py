"""
Tests for quantization comparison benchmark suite.

Tests the quantization benchmarking and comparison functions.
"""

import pytest

from smlx.bench.suites.quantization import (
    QuantizationBenchmarkResult,
    benchmark_quantized_model,
    compare_quantization_methods,
    estimate_model_size,
    print_comparison_summary,
)


# Mock classes for testing


class MockModel:
    """Mock model for testing."""

    def __init__(self, name="MockModel"):
        self.model_type = name
        self._params = {}

    def parameters(self):
        """Return mock parameters."""

        class MockParam:
            def __init__(self, size):
                self.size = size

        return {
            "layer1": MockParam(1000000),
            "layer2": MockParam(500000),
        }


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text):
        """Return mock tokens."""
        # Simple mock: 1 token per word
        return list(range(len(text.split())))

    def decode(self, tokens):
        """Return mock text."""
        return f"Generated {len(tokens)} tokens"


@pytest.mark.unit
class TestQuantizationBenchmarkResult:
    """Test QuantizationBenchmarkResult dataclass."""

    def test_creation(self):
        """Test creating benchmark result."""
        result = QuantizationBenchmarkResult(
            quantization_method="4bit",
            model_name="SmolLM2-135M",
            model_size_gb=0.5,
            memory_reduction_percent=75.0,
            prompt_tps=200.0,
            generation_tps=50.0,
            prompt_speedup=1.2,
            generation_speedup=1.5,
            peak_memory_gb=1.0,
            memory_savings_percent=50.0,
            prompt_tokens=100,
            generation_tokens=50,
            test_prompt="Test",
        )

        assert result.quantization_method == "4bit"
        assert result.model_name == "SmolLM2-135M"
        assert result.model_size_gb == 0.5
        assert result.memory_reduction_percent == 75.0
        assert result.prompt_tps == 200.0
        assert result.generation_tps == 50.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = QuantizationBenchmarkResult(
            quantization_method="8bit",
            model_name="TestModel",
            model_size_gb=1.0,
            memory_reduction_percent=50.0,
            prompt_tps=150.0,
            generation_tps=40.0,
            prompt_speedup=1.0,
            generation_speedup=1.0,
            peak_memory_gb=2.0,
            memory_savings_percent=25.0,
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["quantization_method"] == "8bit"
        assert data["model_name"] == "TestModel"
        assert data["model_size_gb"] == 1.0

    def test_optional_fields(self):
        """Test optional fields."""
        result = QuantizationBenchmarkResult(
            quantization_method="4bit",
            model_name="Test",
            model_size_gb=0.5,
            memory_reduction_percent=0.0,
            prompt_tps=100.0,
            generation_tps=25.0,
            prompt_speedup=1.0,
            generation_speedup=1.0,
            peak_memory_gb=1.0,
            memory_savings_percent=0.0,
            perplexity=10.5,
            perplexity_degradation=2.0,
        )

        assert result.perplexity == 10.5
        assert result.perplexity_degradation == 2.0


@pytest.mark.unit
class TestEstimateModelSize:
    """Test estimate_model_size function."""

    def test_with_mock_model(self):
        """Test model size estimation with mock model."""
        model = MockModel()

        # FP16: 1.5M params * 2 bytes = 3MB
        size_fp16 = estimate_model_size(model, "fp16")
        assert size_fp16 > 0

        # 8bit should be half of FP16
        size_8bit = estimate_model_size(model, "8bit")
        assert size_8bit == pytest.approx(size_fp16 / 2)

        # 4bit should be quarter of FP16
        size_4bit = estimate_model_size(model, "4bit")
        assert size_4bit == pytest.approx(size_fp16 / 4)

    def test_different_quantization_methods(self):
        """Test size estimates for different methods."""
        model = MockModel()

        sizes = {}
        for method in ["fp16", "8bit", "4bit", "gptq", "awq", "dwq"]:
            sizes[method] = estimate_model_size(model, method)

        # FP16 should be largest
        assert sizes["fp16"] >= sizes["8bit"]
        assert sizes["fp16"] >= sizes["4bit"]

        # 8bit methods should be similar
        assert sizes["8bit"] == pytest.approx(sizes["dwq"])

        # 4bit methods should be similar
        assert sizes["4bit"] == pytest.approx(sizes["gptq"])
        assert sizes["4bit"] == pytest.approx(sizes["awq"])

    def test_unknown_quantization(self):
        """Test with unknown quantization method."""
        model = MockModel()

        # Should default to FP16
        size_unknown = estimate_model_size(model, "unknown")
        size_fp16 = estimate_model_size(model, "fp16")

        assert size_unknown == size_fp16


@pytest.mark.unit
class TestBenchmarkQuantizedModel:
    """Test benchmark_quantized_model function."""

    @pytest.mark.requires_model
    def test_basic_benchmark(self):
        """Test basic quantized model benchmark end-to-end with a real model.

        benchmark_quantized_model runs real generation (generate/stream_generate),
        so it needs an actual model — a mock cannot drive it. Gated by
        requires_model.
        """
        from smlx.models import mlx_backend

        bm = mlx_backend.load("mlx-community/SmolLM2-135M-Instruct")
        model, tokenizer = bm.model, bm.processor

        result = benchmark_quantized_model(
            model=model,
            tokenizer=tokenizer,
            quantization_method="fp16",
            test_prompt="The capital of France is",
            generation_tokens=10,
            verbose=False,
        )

        assert isinstance(result, QuantizationBenchmarkResult)
        assert result.quantization_method == "fp16"
        assert isinstance(result.model_name, str) and result.model_name
        assert result.prompt_tps > 0
        assert result.generation_tps > 0


@pytest.mark.unit
class TestCompareQuantizationMethods:
    """Test compare_quantization_methods function."""

    @pytest.mark.skip(reason="Requires SmolLM2_135M model implementation")
    @pytest.mark.requires_model
    def test_single_method(self):
        """Test comparison with single method."""
        results = compare_quantization_methods(
            model_path="test_model",
            quantization_methods=["fp16"],
            test_prompt="Test",
            generation_tokens=10,
            verbose=False,
        )

        assert isinstance(results, dict)
        assert "fp16" in results
        assert isinstance(results["fp16"], QuantizationBenchmarkResult)

    @pytest.mark.skip(reason="Requires SmolLM2_135M model implementation")
    @pytest.mark.requires_model
    def test_multiple_methods(self):
        """Test comparison with multiple methods."""
        results = compare_quantization_methods(
            model_path="test_model",
            quantization_methods=["fp16", "8bit", "4bit"],
            test_prompt="Test",
            generation_tokens=10,
            verbose=False,
        )

        assert len(results) == 3
        assert "fp16" in results
        assert "8bit" in results
        assert "4bit" in results

    @pytest.mark.skip(reason="Requires SmolLM2_135M model implementation")
    @pytest.mark.requires_model
    def test_relative_metrics_calculated(self):
        """Test that relative metrics are calculated."""
        results = compare_quantization_methods(
            model_path="test_model",
            quantization_methods=["fp16", "4bit"],
            test_prompt="Test",
            generation_tokens=10,
            verbose=False,
        )

        # Baseline (fp16) should have 0% reduction
        assert results["fp16"].memory_reduction_percent == 0.0

        # 4bit should have positive reduction
        assert results["4bit"].memory_reduction_percent > 0


@pytest.mark.unit
class TestPrintComparisonSummary:
    """Test print_comparison_summary function."""

    def test_print_summary(self, capsys):
        """Test printing comparison summary."""
        baseline = QuantizationBenchmarkResult(
            quantization_method="fp16",
            model_name="Test",
            model_size_gb=2.0,
            memory_reduction_percent=0.0,
            prompt_tps=200.0,
            generation_tps=50.0,
            prompt_speedup=1.0,
            generation_speedup=1.0,
            peak_memory_gb=4.0,
            memory_savings_percent=0.0,
        )

        optimized = QuantizationBenchmarkResult(
            quantization_method="4bit",
            model_name="Test",
            model_size_gb=0.5,
            memory_reduction_percent=75.0,
            prompt_tps=240.0,
            generation_tps=60.0,
            prompt_speedup=1.2,
            generation_speedup=1.2,
            peak_memory_gb=2.0,
            memory_savings_percent=50.0,
        )

        results = {"fp16": baseline, "4bit": optimized}

        print_comparison_summary(results, baseline)

        captured = capsys.readouterr()

        assert "COMPARISON SUMMARY" in captured.out
        assert "fp16" in captured.out
        assert "4bit" in captured.out
        assert "RECOMMENDATIONS" in captured.out

    def test_summary_with_single_result(self, capsys):
        """Test summary with single result."""
        result = QuantizationBenchmarkResult(
            quantization_method="fp16",
            model_name="Test",
            model_size_gb=2.0,
            memory_reduction_percent=0.0,
            prompt_tps=200.0,
            generation_tps=50.0,
            prompt_speedup=1.0,
            generation_speedup=1.0,
            peak_memory_gb=4.0,
            memory_savings_percent=0.0,
        )

        results = {"fp16": result}

        print_comparison_summary(results, result)

        captured = capsys.readouterr()

        assert "COMPARISON SUMMARY" in captured.out
        assert "RECOMMENDATIONS" in captured.out


@pytest.mark.integration
@pytest.mark.requires_model
@pytest.mark.slow
class TestQuantizationIntegration:
    """Integration tests for quantization benchmarks."""

    def test_end_to_end_comparison(self):
        """Test complete quantization comparison workflow."""
        results = compare_quantization_methods(
            model_path="mlx-community/SmolLM2-135M-Instruct",
            quantization_methods=["fp16", "4bit"],
            test_prompt="The quick brown fox",
            generation_tokens=50,
            verbose=True,
        )

        # Verify results
        assert len(results) == 2
        assert "fp16" in results
        assert "4bit" in results

        # Check that metrics are populated
        for method, result in results.items():
            assert result.model_size_gb > 0
            assert result.prompt_tps > 0
            assert result.generation_tps > 0
            assert result.peak_memory_gb > 0

        # Check relative metrics
        assert results["4bit"].memory_reduction_percent > 0
        assert results["fp16"].memory_reduction_percent == 0.0


@pytest.mark.benchmark
class TestQuantizationPerformance:
    """Performance tests for quantization benchmarks."""

    def test_size_calculation_accuracy(self):
        """Test that size calculations are accurate."""
        model = MockModel()

        # Create a model with known parameter count
        expected_params = 1_500_000  # 1.5M params

        # FP16: 2 bytes per param
        size_fp16 = estimate_model_size(model, "fp16")
        expected_fp16_gb = (expected_params * 2) / (1024**3)

        # Should be close (within 10% due to mock approximation)
        assert abs(size_fp16 - expected_fp16_gb) / expected_fp16_gb < 0.5

    def test_multiple_benchmarks_dont_interfere(self):
        """Test that multiple benchmarks don't interfere with each other."""
        model = MockModel()

        # Run multiple size estimates
        sizes = []
        for _ in range(5):
            size = estimate_model_size(model, "fp16")
            sizes.append(size)

        # All should be identical
        assert all(s == sizes[0] for s in sizes)


@pytest.mark.unit
class TestQuantizationHelpers:
    """Test helper functions."""

    def test_result_serialization(self):
        """Test that results can be serialized."""
        result = QuantizationBenchmarkResult(
            quantization_method="4bit",
            model_name="Test",
            model_size_gb=0.5,
            memory_reduction_percent=75.0,
            prompt_tps=200.0,
            generation_tps=50.0,
            prompt_speedup=1.2,
            generation_speedup=1.5,
            peak_memory_gb=1.0,
            memory_savings_percent=50.0,
        )

        # Convert to dict
        data = result.to_dict()

        # Should be JSON-serializable
        import json

        json_str = json.dumps(data)
        loaded = json.loads(json_str)

        assert loaded["quantization_method"] == "4bit"
        assert loaded["model_size_gb"] == 0.5

    def test_bytes_per_param_mapping(self):
        """Test that bytes per parameter mapping is correct."""
        model = MockModel()

        # Get sizes for all methods
        fp16_size = estimate_model_size(model, "fp16")
        bit8_size = estimate_model_size(model, "8bit")
        bit4_size = estimate_model_size(model, "4bit")

        # Ratios should be correct
        assert bit8_size == pytest.approx(fp16_size / 2)
        assert bit4_size == pytest.approx(fp16_size / 4)
        assert bit4_size == pytest.approx(bit8_size / 2)
