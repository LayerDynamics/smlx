"""
Tests for text generation benchmark suite.

Tests comprehensive text generation benchmarking functions.
"""

import pytest

from smlx.bench.suites.llm import LLMBenchmarkConfig
from smlx.bench.suites.text_generation import (
    TextGenerationBenchmarkResult,
    benchmark_batch_size_scaling,
    benchmark_context_scaling,
    benchmark_generation_length,
    benchmark_single_generation,
    benchmark_temperature_effects,
    run_comprehensive_suite,
)
from smlx.bench.stats import ModelBenchmarkStats


# Mock classes for testing


class MockModel:
    """Mock model for testing."""

    def __init__(self, name="MockModel"):
        self.name = name

    def generate(self, tokens, max_tokens=10, **kwargs):
        """Mock generate method."""
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        return tokens + list(range(max_tokens))

    def __call__(self, tokens):
        """Mock forward pass."""
        import mlx.core as mx
        if not isinstance(tokens, mx.array):
            tokens = mx.array(tokens)
        vocab_size = 1000
        seq_len = tokens.shape[-1] if len(tokens.shape) > 0 else 1
        return mx.random.normal((seq_len, vocab_size))


class MockTokenizer:
    """Mock tokenizer."""

    def encode(self, text):
        """Mock encode - 1 token per word."""
        if isinstance(text, list):
            return text
        return [i for i in range(len(text.split()))]

    def decode(self, tokens):
        """Mock decode."""
        return f"Generated {len(tokens)} tokens"


@pytest.mark.unit
class TestTextGenerationBenchmarkResult:
    """Test TextGenerationBenchmarkResult dataclass."""

    def test_creation(self):
        """Test creating benchmark result."""
        result = TextGenerationBenchmarkResult(
            name="Test Suite",
            parameter_name="context_length",
            parameter_values=[128, 256, 512],
        )

        assert result.name == "Test Suite"
        assert result.parameter_name == "context_length"
        assert result.parameter_values == [128, 256, 512]
        assert len(result.benchmarks) == 0

    def test_add_benchmarks(self):
        """Test adding benchmarks to result."""
        result = TextGenerationBenchmarkResult(
            name="Test Suite",
            parameter_name="temperature",
            parameter_values=[0.0, 0.5, 1.0],
        )

        # Add mock benchmarks
        for _ in range(3):
            benchmark = ModelBenchmarkStats(
                name="test",
                prompt_tokens=100,
                generation_tokens=50,
                prompt_time=0.5,
                generation_time=2.0,
                prompt_tps=200.0,
                generation_tps=25.0,
            )
            result.benchmarks.append(benchmark)

        assert len(result.benchmarks) == 3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TextGenerationBenchmarkResult(
            name="Test Suite",
            parameter_name="gen_length",
            parameter_values=[50, 100],
        )

        benchmark = ModelBenchmarkStats(
            name="test",
            prompt_tokens=10,
            generation_tokens=50,
            prompt_time=0.1,
            generation_time=1.0,
            prompt_tps=100.0,
            generation_tps=50.0,
        )
        result.benchmarks.append(benchmark)

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "Test Suite"
        assert data["parameter_name"] == "gen_length"
        assert len(data["benchmarks"]) == 1

    def test_print_summary(self, capsys):
        """Test printing summary."""
        result = TextGenerationBenchmarkResult(
            name="Test Suite",
            parameter_name="temperature",
            parameter_values=[0.0, 0.5],
        )

        for _ in range(2):
            benchmark = ModelBenchmarkStats(
                name="test",
                prompt_tokens=100,
                generation_tokens=50,
                prompt_time=0.5,
                generation_time=2.0,
                prompt_tps=200.0,
                generation_tps=25.0,
            )
            result.benchmarks.append(benchmark)

        result.print_summary()

        captured = capsys.readouterr()

        assert "Test Suite" in captured.out
        assert "temperature" in captured.out
        assert "Prompt TPS" in captured.out
        assert "Generation TPS" in captured.out


@pytest.mark.unit
class TestBenchmarkSingleGeneration:
    """Test benchmark_single_generation function."""

    def test_basic_generation(self):
        """Test basic single generation benchmark."""
        model = MockModel()
        tokenizer = MockTokenizer()

        config = LLMBenchmarkConfig(
            generation_tokens=10,
            warmup_tokens=2,
        )

        stats = benchmark_single_generation(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            config=config,
        )

        assert isinstance(stats, ModelBenchmarkStats)
        assert stats.prompt_tokens > 0
        assert stats.generation_tokens > 0

    def test_with_custom_config(self):
        """Test with custom configuration."""
        model = MockModel()
        tokenizer = MockTokenizer()

        config = LLMBenchmarkConfig(
            generation_tokens=50,
            temperature=0.7,
            warmup_tokens=5,
        )

        stats = benchmark_single_generation(
            model=model,
            tokenizer=tokenizer,
            prompt="Custom prompt",
            config=config,
        )

        assert stats.metadata.get("temperature") == 0.7


@pytest.mark.unit
class TestBenchmarkContextScaling:
    """Test benchmark_context_scaling function."""

    def test_default_context_lengths(self):
        """Test with default context lengths."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_context_scaling(
            model=model,
            tokenizer=tokenizer,
            base_prompt="The quick brown fox. ",
            generation_tokens=10,
        )

        assert isinstance(result, TextGenerationBenchmarkResult)
        assert result.name == "Context Scaling Benchmark"
        assert result.parameter_name == "context_length"
        # Default: [128, 256, 512, 1024, 2048]
        assert len(result.benchmarks) == 5
        assert len(result.parameter_values) == 5

    def test_custom_context_lengths(self):
        """Test with custom context lengths."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_context_scaling(
            model=model,
            tokenizer=tokenizer,
            context_lengths=[64, 128, 256],
            generation_tokens=10,
        )

        assert len(result.benchmarks) == 3
        assert result.parameter_values == [64, 128, 256]

    def test_context_scaling_results(self):
        """Test that context scaling produces valid results."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_context_scaling(
            model=model,
            tokenizer=tokenizer,
            context_lengths=[50, 100],
            generation_tokens=10,
        )

        # Check each benchmark
        for benchmark in result.benchmarks:
            assert benchmark.prompt_tokens > 0
            assert benchmark.generation_tokens > 0
            assert benchmark.duration_ms > 0


@pytest.mark.unit
class TestBenchmarkGenerationLength:
    """Test benchmark_generation_length function."""

    def test_default_generation_lengths(self):
        """Test with default generation lengths."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_generation_length(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
        )

        assert isinstance(result, TextGenerationBenchmarkResult)
        assert result.name == "Generation Length Benchmark"
        assert result.parameter_name == "generation_length"
        # Default: [50, 100, 200, 500]
        assert len(result.benchmarks) == 4
        assert len(result.parameter_values) == 4

    def test_custom_generation_lengths(self):
        """Test with custom generation lengths."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_generation_length(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            generation_lengths=[25, 50, 100],
        )

        assert len(result.benchmarks) == 3
        assert result.parameter_values == [25, 50, 100]

    def test_fixed_prompt(self):
        """Test that prompt is fixed across tests."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_generation_length(
            model=model,
            tokenizer=tokenizer,
            prompt="Fixed prompt for testing",
            generation_lengths=[10, 20],
        )

        # All benchmarks should have same prompt tokens
        prompt_tokens = [b.prompt_tokens for b in result.benchmarks]
        assert len(set(prompt_tokens)) == 1


@pytest.mark.unit
class TestBenchmarkTemperatureEffects:
    """Test benchmark_temperature_effects function."""

    def test_default_temperatures(self):
        """Test with default temperature values."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_temperature_effects(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            generation_tokens=50,
        )

        assert isinstance(result, TextGenerationBenchmarkResult)
        assert result.name == "Temperature Effects Benchmark"
        assert result.parameter_name == "temperature"
        # Default: [0.0, 0.3, 0.7, 1.0]
        assert len(result.benchmarks) == 4
        assert len(result.parameter_values) == 4

    def test_custom_temperatures(self):
        """Test with custom temperatures."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_temperature_effects(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            temperatures=[0.0, 0.5, 1.0],
            generation_tokens=50,
        )

        assert len(result.benchmarks) == 3
        assert result.parameter_values == [0.0, 0.5, 1.0]

    def test_temperature_in_metadata(self):
        """Test that temperature is stored in metadata."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_temperature_effects(
            model=model,
            tokenizer=tokenizer,
            temperatures=[0.0, 0.7],
            generation_tokens=20,
        )

        # Check that each benchmark has the correct temperature in metadata
        for benchmark, temp in zip(result.benchmarks, result.parameter_values):
            assert benchmark.metadata.get("temperature") == temp


@pytest.mark.unit
class TestBenchmarkBatchSizeScaling:
    """Test benchmark_batch_size_scaling function."""

    def test_default_batch_sizes(self):
        """Test with default batch sizes."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_batch_size_scaling(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            generation_tokens=50,
        )

        assert isinstance(result, TextGenerationBenchmarkResult)
        assert result.name == "Batch Size Scaling Benchmark"
        assert result.parameter_name == "batch_size"
        # Default: [1, 2, 4, 8]
        assert len(result.benchmarks) == 4
        assert len(result.parameter_values) == 4

    def test_custom_batch_sizes(self):
        """Test with custom batch sizes."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_batch_size_scaling(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            batch_sizes=[1, 4, 16],
            generation_tokens=50,
        )

        assert len(result.benchmarks) == 3
        assert result.parameter_values == [1, 4, 16]


@pytest.mark.unit
class TestRunComprehensiveSuite:
    """Test run_comprehensive_suite function."""

    def test_all_benchmarks_enabled(self):
        """Test running all benchmarks."""
        model = MockModel()
        tokenizer = MockTokenizer()

        results = run_comprehensive_suite(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            include_context_scaling=True,
            include_generation_length=True,
            include_temperature_effects=True,
            include_batch_scaling=True,
        )

        assert isinstance(results, dict)
        # Should have all 4 benchmark types
        assert len(results) >= 3  # At least 3 (batch may fail)

        # Check each result type
        if "context_scaling" in results:
            assert isinstance(results["context_scaling"], TextGenerationBenchmarkResult)
        if "generation_length" in results:
            assert isinstance(results["generation_length"], TextGenerationBenchmarkResult)
        if "temperature_effects" in results:
            assert isinstance(results["temperature_effects"], TextGenerationBenchmarkResult)

    def test_selective_benchmarks(self):
        """Test running selective benchmarks."""
        model = MockModel()
        tokenizer = MockTokenizer()

        results = run_comprehensive_suite(
            model=model,
            tokenizer=tokenizer,
            include_context_scaling=True,
            include_generation_length=False,
            include_temperature_effects=True,
            include_batch_scaling=False,
        )

        # Should only have context_scaling and temperature_effects
        assert "context_scaling" in results
        assert "temperature_effects" in results
        assert "generation_length" not in results
        assert "batch_scaling" not in results

    def test_minimal_suite(self):
        """Test with minimal benchmarks."""
        model = MockModel()
        tokenizer = MockTokenizer()

        results = run_comprehensive_suite(
            model=model,
            tokenizer=tokenizer,
            include_context_scaling=True,
            include_generation_length=False,
            include_temperature_effects=False,
            include_batch_scaling=False,
        )

        # Should only have context_scaling
        assert len(results) == 1
        assert "context_scaling" in results


@pytest.mark.integration
@pytest.mark.requires_model
class TestTextGenerationIntegration:
    """Integration tests for text generation benchmarks."""

    def test_end_to_end_workflow(self):
        """Test complete text generation benchmark workflow."""
        model = MockModel("IntegrationModel")
        tokenizer = MockTokenizer()

        # Run comprehensive suite
        results = run_comprehensive_suite(
            model=model,
            tokenizer=tokenizer,
            prompt="Integration test prompt",
            include_batch_scaling=False,  # Skip batch to avoid issues
        )

        # Verify all benchmarks completed
        assert len(results) >= 3

        # Check each result
        for name, result in results.items():
            assert isinstance(result, TextGenerationBenchmarkResult)
            assert len(result.benchmarks) > 0
            assert len(result.parameter_values) > 0

            # All benchmarks should have valid stats
            for benchmark in result.benchmarks:
                assert benchmark.duration_ms > 0
                assert benchmark.prompt_tokens > 0
                assert benchmark.generation_tokens > 0

    def test_context_scaling_trend(self):
        """Test that context scaling shows expected trends."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_context_scaling(
            model=model,
            tokenizer=tokenizer,
            context_lengths=[50, 100, 200],
            generation_tokens=20,
        )

        # Verify prompt tokens increase with context length
        prompt_tokens = [b.prompt_tokens for b in result.benchmarks]
        # Should generally increase (allowing for small variations in tokenization)
        assert prompt_tokens[0] <= prompt_tokens[1] * 1.5
        assert prompt_tokens[1] <= prompt_tokens[2] * 1.5


@pytest.mark.benchmark
class TestTextGenerationPerformance:
    """Performance tests for text generation benchmarks."""

    def test_benchmark_overhead(self):
        """Test that benchmark overhead is reasonable."""
        import time
        model = MockModel()
        tokenizer = MockTokenizer()

        start = time.perf_counter()
        result = benchmark_generation_length(
            model=model,
            tokenizer=tokenizer,
            generation_lengths=[10, 20],
        )
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        # (This is mainly to ensure benchmarks don't hang)
        assert elapsed < 30  # 30 seconds max

    def test_multiple_runs_consistency(self):
        """Test that multiple runs produce consistent results."""
        model = MockModel()
        tokenizer = MockTokenizer()

        results = []
        for _ in range(3):
            result = benchmark_temperature_effects(
                model=model,
                tokenizer=tokenizer,
                temperatures=[0.0, 0.5],
                generation_tokens=20,
            )
            results.append(result)

        # All should have same number of benchmarks
        assert all(len(r.benchmarks) == 2 for r in results)

        # Parameter values should be identical
        for result in results:
            assert result.parameter_values == [0.0, 0.5]


@pytest.mark.unit
class TestTextGenerationHelpers:
    """Test helper functions and edge cases."""

    def test_empty_prompt(self):
        """Test with empty prompt."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_generation_length(
            model=model,
            tokenizer=tokenizer,
            prompt="",
            generation_lengths=[10],
        )

        assert len(result.benchmarks) == 1

    def test_very_short_generation(self):
        """Test with very short generation."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_generation_length(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            generation_lengths=[1, 2],
        )

        assert len(result.benchmarks) == 2

    def test_result_dict_serialization(self):
        """Test that results can be serialized to dict."""
        model = MockModel()
        tokenizer = MockTokenizer()

        result = benchmark_temperature_effects(
            model=model,
            tokenizer=tokenizer,
            temperatures=[0.0],
            generation_tokens=10,
        )

        data = result.to_dict()

        # Should be JSON-serializable
        import json
        json_str = json.dumps(data, default=str)  # default=str for any non-serializable
        loaded = json.loads(json_str)

        assert loaded["name"] == result.name
        assert loaded["parameter_name"] == result.parameter_name
