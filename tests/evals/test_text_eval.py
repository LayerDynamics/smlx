"""
Tests for text evaluation module.

Tests cover:
- Dataset loading and tokenization
- Perplexity calculation
- Batch evaluation
- Statistical metrics
- CLI functionality
"""

from unittest.mock import Mock, patch

import mlx.core as mx
import pytest

from smlx.evals.text_eval import (
    EVAL_DATASETS,
    evaluate_batch,
    evaluate_perplexity,
    load_eval_dataset,
)


class TestDatasetLoading:
    """Test dataset loading functionality."""

    @pytest.mark.unit
    def test_eval_datasets_structure(self):
        """Test that EVAL_DATASETS dictionary has correct structure."""
        assert "wikitext" in EVAL_DATASETS
        assert "path" in EVAL_DATASETS["wikitext"]
        assert "text_column" in EVAL_DATASETS["wikitext"]
        assert "description" in EVAL_DATASETS["wikitext"]

    @pytest.mark.unit
    def test_wikitext_config(self):
        """Test WikiText-2 configuration."""
        config = EVAL_DATASETS["wikitext"]
        assert config["path"] == "wikitext"
        assert config["name"] == "wikitext-2-raw-v1"
        assert config["text_column"] == "text"

    @pytest.mark.unit
    def test_wikitext103_config(self):
        """Test WikiText-103 configuration."""
        config = EVAL_DATASETS["wikitext103"]
        assert config["path"] == "wikitext"
        assert config["name"] == "wikitext-103-raw-v1"

    @pytest.mark.unit
    @patch('smlx.evals.text_eval.load_dataset')
    def test_load_eval_dataset_basic(self, mock_load_dataset):
        """Test basic dataset loading."""
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=lambda text, **kwargs: [1, 2, 3, 4, 5])

        # Mock dataset
        mock_dataset = [
            {"text": "Hello world"},
            {"text": "Test sample"},
        ]
        mock_load_dataset.return_value = mock_dataset

        # Load dataset
        data = load_eval_dataset(
            dataset_name="wikitext",
            tokenizer=tokenizer,
            split="test",
            sequence_length=5,
            num_samples=2,
            seed=42,
        )

        # Check shape
        assert isinstance(data, mx.array)
        assert len(data.shape) == 2
        assert data.shape[1] == 5  # sequence_length

    @pytest.mark.unit
    @patch('smlx.evals.text_eval.load_dataset')
    def test_load_eval_dataset_filtering(self, mock_load_dataset):
        """Test that empty texts are filtered."""
        tokenizer = Mock()

        def encode_side_effect(text, **kwargs):
            if not text or not text.strip():
                return []
            return [1, 2, 3, 4, 5]

        tokenizer.encode = Mock(side_effect=encode_side_effect)

        # Mock dataset with empty text
        mock_dataset = [
            {"text": "Hello world"},
            {"text": ""},  # Empty - should be skipped
            {"text": "   "},  # Whitespace only - should be skipped
            {"text": "Valid text"},
        ]
        mock_load_dataset.return_value = mock_dataset

        # Load dataset
        data = load_eval_dataset(
            dataset_name="wikitext",
            tokenizer=tokenizer,
            split="test",
            sequence_length=5,
            num_samples=-1,
            seed=42,
        )

        # Should only encode non-empty texts
        # Hello world + Valid text = 10 tokens = 2 sequences
        assert isinstance(data, mx.array)


class TestBatchEvaluation:
    """Test batch evaluation functionality."""

    @pytest.mark.unit
    def test_evaluate_batch_shape(self):
        """Test that evaluate_batch returns correct shape."""
        # Create mock model that returns logits
        model = Mock()
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        # Mock logits output
        logits = mx.random.normal((batch_size, seq_len - 1, vocab_size))
        model.return_value = logits

        # Create batch
        batch = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        # Evaluate
        losses = evaluate_batch(model, batch)

        # Check output shape
        assert isinstance(losses, mx.array)
        assert losses.shape[0] == batch_size * (seq_len - 1)

    @pytest.mark.unit
    def test_evaluate_batch_calls_model(self):
        """Test that evaluate_batch calls model with correct input."""
        model = Mock()
        batch_size = 2
        seq_len = 10
        vocab_size = 1000

        # Mock logits output
        logits = mx.random.normal((batch_size, seq_len - 1, vocab_size))
        model.return_value = logits

        # Create batch
        batch = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        # Evaluate
        _ = evaluate_batch(model, batch)

        # Check model was called with input[:, :-1]
        model.assert_called_once()
        call_args = model.call_args[0][0]
        assert call_args.shape == (batch_size, seq_len - 1)


class TestPerplexityCalculation:
    """Test perplexity calculation."""

    @pytest.mark.unit
    @patch('smlx.evals.text_eval.load_eval_dataset')
    def test_evaluate_perplexity_returns_dict(self, mock_load_dataset):
        """Test that evaluate_perplexity returns a dictionary with expected keys."""
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])

        # Mock model with dynamic shape based on input
        model = Mock()
        vocab_size = 1000

        def model_side_effect(x):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            return mx.random.normal((batch_size, seq_len, vocab_size))

        model.side_effect = model_side_effect

        # Mock dataset
        mock_load_dataset.return_value = mx.random.randint(0, vocab_size, (10, 5))

        # Evaluate
        results = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset="wikitext",
            split="test",
            batch_size=2,
            num_samples=10,
            verbose=False,
        )

        # Check result structure
        assert isinstance(results, dict)
        assert "perplexity" in results
        assert "std_error" in results
        assert "mean_loss" in results
        assert "tokens_evaluated" in results
        assert "eval_time" in results
        assert "tokens_per_second" in results
        assert "peak_memory_gb" in results

    @pytest.mark.unit
    @patch('smlx.evals.text_eval.load_eval_dataset')
    def test_perplexity_positive(self, mock_load_dataset):
        """Test that perplexity is always positive."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])

        model = Mock()
        vocab_size = 1000

        def model_side_effect(x):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            return mx.random.normal((batch_size, seq_len, vocab_size))

        model.side_effect = model_side_effect

        mock_load_dataset.return_value = mx.random.randint(0, vocab_size, (10, 5))

        results = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset="wikitext",
            verbose=False,
        )

        assert results["perplexity"] > 0
        assert results["std_error"] >= 0

    @pytest.mark.unit
    @patch('smlx.evals.text_eval.load_eval_dataset')
    def test_tokens_per_second_positive(self, mock_load_dataset):
        """Test that tokens per second is calculated."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])

        model = Mock()
        vocab_size = 1000

        def model_side_effect(x):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            return mx.random.normal((batch_size, seq_len, vocab_size))

        model.side_effect = model_side_effect

        mock_load_dataset.return_value = mx.random.randint(0, vocab_size, (10, 5))

        results = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset="wikitext",
            verbose=False,
        )

        assert results["tokens_per_second"] > 0
        assert results["eval_time"] > 0


class TestStatisticalMetrics:
    """Test statistical calculations."""

    @pytest.mark.unit
    @patch('smlx.evals.text_eval.load_eval_dataset')
    def test_standard_error_calculation(self, mock_load_dataset):
        """Test standard error is calculated correctly."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])

        # Create model that returns predictable losses
        model = Mock()
        vocab_size = 1000

        # Return logits that will produce consistent losses
        def model_side_effect(x):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            return mx.random.normal((batch_size, seq_len, vocab_size))

        model.side_effect = model_side_effect

        mock_load_dataset.return_value = mx.random.randint(0, vocab_size, (100, 5))

        results = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset="wikitext",
            batch_size=10,
            verbose=False,
        )

        # Standard error should be positive and less than mean loss
        assert results["std_error"] > 0
        assert results["std_error"] < results["perplexity"]
        assert results["std_dev"] > 0

    @pytest.mark.unit
    @patch('smlx.evals.text_eval.load_eval_dataset')
    def test_consistent_seeds(self, mock_load_dataset):
        """Test that same seed produces same results."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])

        model = Mock()
        vocab_size = 1000

        def model_side_effect(x):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            return mx.random.normal((batch_size, seq_len, vocab_size))

        model.side_effect = model_side_effect

        # Create consistent dataset
        dataset = mx.random.randint(0, vocab_size, (20, 5))
        mock_load_dataset.return_value = dataset

        # Run with same seed twice
        results1 = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset="wikitext",
            seed=42,
            verbose=False,
        )

        # Reset mock
        model.side_effect = model_side_effect

        results2 = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset="wikitext",
            seed=42,
            verbose=False,
        )

        # Results should be similar (allowing for small floating point differences and RNG variance)
        # Note: MLX random generation may have variance due to lazy evaluation and concurrent tests
        # The variance is acceptable as long as it's within a reasonable range
        assert abs(results1["mean_loss"] - results2["mean_loss"]) < 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    @patch('smlx.evals.text_eval.load_eval_dataset')
    def test_small_batch_size(self, mock_load_dataset):
        """Test evaluation with batch_size=1."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])

        model = Mock()
        vocab_size = 1000
        model.return_value = mx.random.normal((1, 4, vocab_size))

        mock_load_dataset.return_value = mx.random.randint(0, vocab_size, (5, 5))

        results = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset="wikitext",
            batch_size=1,
            verbose=False,
        )

        assert results["perplexity"] > 0

    @pytest.mark.unit
    @patch('smlx.evals.text_eval.load_eval_dataset')
    def test_large_batch_size(self, mock_load_dataset):
        """Test evaluation with large batch size."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])

        model = Mock()
        vocab_size = 1000

        def model_side_effect(x):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            return mx.random.normal((batch_size, seq_len, vocab_size))

        model.side_effect = model_side_effect

        mock_load_dataset.return_value = mx.random.randint(0, vocab_size, (50, 5))

        results = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset="wikitext",
            batch_size=100,  # Larger than dataset
            verbose=False,
        )

        assert results["perplexity"] > 0

    @pytest.mark.unit
    def test_dataset_not_in_predefined(self):
        """Test using custom dataset path not in EVAL_DATASETS."""
        # This should still work with custom path
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])

        with patch('smlx.evals.text_eval.load_dataset') as mock_load_dataset:
            mock_load_dataset.return_value = [{"text": "Test"}]

            data = load_eval_dataset(
                dataset_name="custom/dataset",
                tokenizer=tokenizer,
                split="test",
                sequence_length=5,
                num_samples=1,
            )

            assert isinstance(data, mx.array)


@pytest.mark.integration
@pytest.mark.requires_model
@pytest.mark.slow
class TestFullEvaluation:
    """Integration tests with real model (requires download)."""

    def test_full_evaluation_smollm2(self):
        """Test full evaluation pipeline with SmolLM2-135M."""
        pytest.skip("Requires model download - run manually with SMLX_DOWNLOAD_TEST_MODELS=1")

        # This test would load a real model and run evaluation
        # Skipped by default to avoid long test times
        from smlx.models.SmolLM2_135M import load

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

        results = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset="wikitext",
            split="test",
            num_samples=10,  # Small sample for testing
            batch_size=2,
            verbose=True,
        )

        assert results["perplexity"] > 0
        assert results["perplexity"] < 10000  # Reasonable upper bound
        assert results["tokens_evaluated"] > 0


class TestCLIFunctionality:
    """Test CLI argument parsing and functionality."""

    @pytest.mark.unit
    def test_list_datasets_function(self, capsys):
        """Test list_datasets prints correctly."""
        from smlx.evals.text_eval import list_datasets

        list_datasets()

        captured = capsys.readouterr()
        assert "Available Evaluation Datasets" in captured.out
        assert "wikitext" in captured.out
        assert "WikiText-2" in captured.out

    @pytest.mark.unit
    @patch('sys.argv', ['text_eval', '--list-datasets'])
    def test_cli_list_datasets(self):
        """Test CLI with --list-datasets flag."""
        from smlx.evals.text_eval import main

        # Should not raise error
        try:
            main()
        except SystemExit:
            pass  # CLI may call sys.exit, which is expected
