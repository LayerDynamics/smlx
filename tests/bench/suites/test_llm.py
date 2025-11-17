"""
Tests for LLM benchmark suite.

Tests the language model benchmarking functions.
"""

import pytest
import mlx.core as mx

from smlx.bench.suites.llm import (
    LLMBenchmarkConfig,
    benchmark_llm,
    benchmark_llm_batch,
    benchmark_llm_streaming,
)
from smlx.bench.stats import ModelBenchmarkStats


# Mock model classes for testing


class MockModel:
    """Mock model with generate method."""

    def __init__(self, name="MockModel"):
        self.name = name
        self.generate_called = False

    def generate(self, tokens, max_tokens=10, **kwargs):
        """Mock generate method."""
        self.generate_called = True
        # Return input tokens + generated tokens
        if isinstance(tokens, mx.array):
            tokens = tokens.tolist()
        return tokens + list(range(max_tokens))

    def __call__(self, tokens):
        """Mock forward pass."""
        # Return fake logits
        if not isinstance(tokens, mx.array):
            tokens = mx.array(tokens)
        vocab_size = 1000
        batch_size = 1 if len(tokens.shape) == 1 else tokens.shape[0]
        seq_len = tokens.shape[-1]
        return mx.random.normal((batch_size, seq_len, vocab_size))


class MockTokenizer:
    """Mock tokenizer."""

    def encode(self, text):
        """Mock encode method."""
        # Simple mock: return word count as tokens
        return [1, 2, 3, 4, 5]  # Fixed 5 tokens for simplicity

    def decode(self, tokens):
        """Mock decode method."""
        return f"Generated {len(tokens)} tokens"


class CallableModel:
    """Mock model that is callable but has no generate method."""

    def __call__(self, tokens):
        """Mock forward pass."""
        if not isinstance(tokens, mx.array):
            tokens = mx.array(tokens)
        vocab_size = 1000
        seq_len = tokens.shape[-1] if len(tokens.shape) > 0 else 1
        return mx.random.normal((seq_len, vocab_size))


@pytest.mark.unit
class TestLLMBenchmarkConfig:
    """Test LLMBenchmarkConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = LLMBenchmarkConfig()

        assert config.prompt_tokens == 100
        assert config.generation_tokens == 100
        assert config.batch_size == 1
        assert config.num_trials == 1
        assert config.temperature == 0.0
        assert config.top_p == 1.0
        assert config.seed == 0
        assert config.warmup_tokens == 10
        assert config.measure_ttft is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMBenchmarkConfig(
            prompt_tokens=50,
            generation_tokens=200,
            temperature=0.7,
            seed=42,
        )

        assert config.prompt_tokens == 50
        assert config.generation_tokens == 200
        assert config.temperature == 0.7
        assert config.seed == 42


@pytest.mark.unit
class TestBenchmarkLLM:
    """Test benchmark_llm function."""

    def test_with_mock_model_and_tokenizer(self):
        """Test benchmarking with mock model and tokenizer."""
        model = MockModel()
        tokenizer = MockTokenizer()

        config = LLMBenchmarkConfig(
            generation_tokens=10,
            warmup_tokens=2,
        )

        stats = benchmark_llm(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            config=config,
        )

        assert isinstance(stats, ModelBenchmarkStats)
        assert stats.model_name == "MockModel"
        assert stats.prompt_tokens == 5  # MockTokenizer returns 5 tokens
        assert stats.generation_tokens >= 1
        assert stats.duration_ms > 0
        assert model.generate_called

    def test_with_pretokenized_input(self):
        """Test benchmarking with pre-tokenized input."""
        model = MockModel()
        tokens = [1, 2, 3, 4, 5]

        config = LLMBenchmarkConfig(
            generation_tokens=10,
            warmup_tokens=2,
        )

        stats = benchmark_llm(
            model=model,
            tokenizer=None,
            prompt=tokens,
            config=config,
        )

        assert isinstance(stats, ModelBenchmarkStats)
        assert stats.prompt_tokens == 5

    def test_without_tokenizer_raises_error(self):
        """Test that string prompt without tokenizer raises error."""
        model = MockModel()

        with pytest.raises(ValueError, match="Tokenizer required"):
            benchmark_llm(
                model=model,
                tokenizer=None,
                prompt="Test prompt",  # String prompt without tokenizer
            )

    def test_with_callable_model(self):
        """Test benchmarking with callable model (no generate method)."""
        model = CallableModel()
        tokens = [1, 2, 3, 4, 5]

        config = LLMBenchmarkConfig(
            generation_tokens=10,
            warmup_tokens=2,
        )

        stats = benchmark_llm(
            model=model,
            tokenizer=None,
            prompt=tokens,
            config=config,
        )

        assert isinstance(stats, ModelBenchmarkStats)
        assert stats.prompt_tokens == 5

    def test_custom_generate_fn(self):
        """Test benchmarking with custom generate function."""
        model = MockModel()
        tokens = [1, 2, 3]

        custom_generate_called = False

        def custom_generate(model, tokens, max_tokens=10, **kwargs):
            nonlocal custom_generate_called
            custom_generate_called = True
            if isinstance(tokens, mx.array):
                tokens = tokens.tolist()
            return tokens + list(range(max_tokens))

        config = LLMBenchmarkConfig(
            generation_tokens=5,
            warmup_tokens=1,
        )

        stats = benchmark_llm(
            model=model,
            tokenizer=None,
            prompt=tokens,
            config=config,
            generate_fn=custom_generate,
        )

        assert custom_generate_called
        assert isinstance(stats, ModelBenchmarkStats)

    def test_statistics_calculation(self):
        """Test that statistics are calculated correctly."""
        model = MockModel()
        tokens = [1, 2, 3, 4, 5]

        config = LLMBenchmarkConfig(
            generation_tokens=10,
            warmup_tokens=1,
        )

        stats = benchmark_llm(
            model=model,
            tokenizer=None,
            prompt=tokens,
            config=config,
        )

        # Check that TPS calculations are present
        assert stats.prompt_time >= 0
        assert stats.generation_time >= 0
        assert stats.prompt_tps >= 0
        assert stats.generation_tps >= 0

        # Total tokens should be prompt + generation
        assert stats.total_tokens == stats.prompt_tokens + stats.generation_tokens

    def test_with_different_temperatures(self):
        """Test benchmarking with different temperature settings."""
        model = MockModel()
        tokens = [1, 2, 3]

        for temp in [0.0, 0.5, 1.0]:
            config = LLMBenchmarkConfig(
                generation_tokens=5,
                temperature=temp,
                warmup_tokens=1,
            )

            stats = benchmark_llm(
                model=model,
                tokenizer=None,
                prompt=tokens,
                config=config,
            )

            assert isinstance(stats, ModelBenchmarkStats)
            # Temperature should be stored in metadata
            assert stats.metadata.get("temperature") == temp

    @pytest.mark.gpu
    def test_memory_tracking(self):
        """Test that memory usage is tracked."""
        model = MockModel()
        tokens = [1, 2, 3, 4, 5]

        config = LLMBenchmarkConfig(
            generation_tokens=10,
            warmup_tokens=1,
        )

        stats = benchmark_llm(
            model=model,
            tokenizer=None,
            prompt=tokens,
            config=config,
        )

        # Should track memory (will be 0 if MLX not available)
        assert stats.peak_memory_gb >= 0


@pytest.mark.unit
class TestBenchmarkLLMBatch:
    """Test benchmark_llm_batch function."""

    def test_with_multiple_prompts(self):
        """Test benchmarking with multiple prompts."""
        model = MockModel()
        tokenizer = MockTokenizer()

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        config = LLMBenchmarkConfig(
            generation_tokens=5,
            warmup_tokens=1,
        )

        results = benchmark_llm_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            config=config,
        )

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(s, ModelBenchmarkStats) for s in results)

    def test_with_pretokenized_prompts(self):
        """Test with pre-tokenized prompts."""
        model = MockModel()

        prompts = [
            [1, 2, 3],
            [4, 5, 6, 7],
            [8, 9],
        ]

        config = LLMBenchmarkConfig(
            generation_tokens=5,
            warmup_tokens=1,
        )

        results = benchmark_llm_batch(
            model=model,
            tokenizer=None,
            prompts=prompts,
            config=config,
        )

        assert len(results) == 3
        assert results[0].prompt_tokens == 3
        assert results[1].prompt_tokens == 4
        assert results[2].prompt_tokens == 2

    def test_default_prompts(self):
        """Test with default prompts."""
        model = MockModel()
        tokenizer = MockTokenizer()

        config = LLMBenchmarkConfig(
            generation_tokens=5,
            warmup_tokens=1,
        )

        # Call without prompts argument
        results = benchmark_llm_batch(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

        # Should use default prompts
        assert isinstance(results, list)
        assert len(results) > 0


@pytest.mark.unit
class TestBenchmarkLLMStreaming:
    """Test benchmark_llm_streaming function."""

    def test_streaming_benchmark(self):
        """Test streaming benchmark."""
        model = MockModel()
        tokenizer = MockTokenizer()

        config = LLMBenchmarkConfig(
            generation_tokens=10,
            warmup_tokens=1,
        )

        stats, token_times = benchmark_llm_streaming(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            config=config,
        )

        assert isinstance(stats, ModelBenchmarkStats)
        assert isinstance(token_times, list)
        # Currently returns empty list as per implementation
        # This is a placeholder for future streaming implementation


@pytest.mark.integration
@pytest.mark.requires_model
class TestLLMBenchmarkIntegration:
    """Integration tests for LLM benchmarking."""

    def test_end_to_end_benchmark(self):
        """Test complete benchmark workflow."""
        model = MockModel(name="TestLLM")
        tokenizer = MockTokenizer()

        # Run comprehensive benchmark
        config = LLMBenchmarkConfig(
            prompt_tokens=50,
            generation_tokens=50,
            warmup_tokens=5,
            temperature=0.0,
            seed=42,
        )

        stats = benchmark_llm(
            model=model,
            tokenizer=tokenizer,
            prompt="This is a test prompt for benchmarking",
            config=config,
        )

        # Verify all statistics are populated
        assert stats.model_name == "TestLLM"
        assert stats.prompt_tokens > 0
        assert stats.generation_tokens > 0
        assert stats.duration_ms > 0
        assert stats.prompt_time > 0
        assert stats.generation_time > 0
        assert stats.prompt_tps > 0 or stats.prompt_time == 0
        assert stats.generation_tps > 0 or stats.generation_time == 0

    def test_batch_benchmark_workflow(self):
        """Test batch benchmark workflow."""
        model = MockModel()
        tokenizer = MockTokenizer()

        prompts = [
            "Short prompt",
            "A longer prompt with more words",
            "Another test prompt",
        ]

        config = LLMBenchmarkConfig(
            generation_tokens=20,
            warmup_tokens=3,
        )

        results = benchmark_llm_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            config=config,
        )

        # Verify all benchmarks completed
        assert len(results) == 3

        # Calculate average TPS
        avg_gen_tps = sum(s.generation_tps for s in results) / len(results)
        assert avg_gen_tps > 0 or all(s.generation_time == 0 for s in results)

        # All should have same model
        assert all(s.model_name == model.name for s in results)

    def test_comparison_across_configs(self):
        """Test comparing different configurations."""
        model = MockModel()
        tokens = [1, 2, 3, 4, 5]

        configs = [
            LLMBenchmarkConfig(generation_tokens=10, temperature=0.0),
            LLMBenchmarkConfig(generation_tokens=50, temperature=0.0),
            LLMBenchmarkConfig(generation_tokens=100, temperature=0.0),
        ]

        results = []
        for config in configs:
            stats = benchmark_llm(
                model=model,
                tokenizer=None,
                prompt=tokens,
                config=config,
            )
            results.append(stats)

        # Longer generation should take more time (generally)
        assert all(s.generation_tokens > 0 for s in results)


@pytest.mark.benchmark
class TestLLMBenchmarkPerformance:
    """Performance tests for LLM benchmarking."""

    def test_benchmark_overhead_is_minimal(self):
        """Test that benchmark overhead is minimal."""
        import time

        model = MockModel()
        tokens = [1, 2, 3]

        config = LLMBenchmarkConfig(
            generation_tokens=5,
            warmup_tokens=1,
        )

        # Measure total benchmark time
        start = time.perf_counter()
        stats = benchmark_llm(
            model=model,
            tokenizer=None,
            prompt=tokens,
            config=config,
        )
        elapsed = time.perf_counter() - start

        # Benchmark overhead should be reasonable
        # (actual time should be close to measured time)
        measured_total_time = (stats.prompt_time + stats.generation_time)

        # Allow for some overhead (< 2x)
        assert elapsed < measured_total_time * 2 + 0.5  # Add 500ms buffer

    def test_multiple_runs_consistency(self):
        """Test that multiple runs produce consistent results."""
        model = MockModel()
        tokens = [1, 2, 3, 4, 5]

        config = LLMBenchmarkConfig(
            generation_tokens=10,
            warmup_tokens=2,
            seed=42,  # Fixed seed for reproducibility
        )

        # Run multiple times
        results = []
        for _ in range(3):
            stats = benchmark_llm(
                model=model,
                tokenizer=None,
                prompt=tokens,
                config=config,
            )
            results.append(stats)

        # Token counts should be identical
        assert all(s.prompt_tokens == results[0].prompt_tokens for s in results)
        assert all(s.generation_tokens == results[0].generation_tokens for s in results)

        # Times should be similar (within reasonable variance)
        mean_time = sum(s.duration_ms for s in results) / len(results)
        for s in results:
            # Allow 100% variance for very small operations (sub-millisecond timing)
            # where system noise dominates
            assert abs(s.duration_ms - mean_time) < mean_time * 1.0


# Helper function tests


@pytest.mark.unit
class TestLLMHelperFunctions:
    """Test helper functions in llm.py."""

    def test_encode_with_different_tokenizers(self):
        """Test _encode helper with various tokenizer APIs."""
        from smlx.bench.suites.llm import _encode

        # Mock tokenizer that returns list
        class ListTokenizer:
            def encode(self, text):
                return [1, 2, 3]

        tokenizer = ListTokenizer()
        tokens = _encode(tokenizer, "test")
        assert tokens == [1, 2, 3]

    def test_get_model_name(self):
        """Test _get_model_name helper."""
        from smlx.bench.suites.llm import _get_model_name

        # Model with name attribute
        class NamedModel:
            name = "TestModel"

        assert _get_model_name(NamedModel()) == "TestModel"

        # Model with __class__.__name__
        class UnnamedModel:
            pass

        assert _get_model_name(UnnamedModel()) == "UnnamedModel"

    def test_detect_quantization(self):
        """Test _detect_quantization helper."""
        from smlx.bench.suites.llm import _detect_quantization

        # Model with quantization attribute
        class QuantizedModel:
            quantization = "4bit"

        assert _detect_quantization(QuantizedModel()) == "4bit"

        # Model without quantization
        class UnquantizedModel:
            pass

        assert _detect_quantization(UnquantizedModel()) is None
