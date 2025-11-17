"""
Tests for LM evaluation harness integration.

Tests cover:
- Helper functions (padding, stop sequences)
- SMLXLM wrapper initialization
- Tokenization
- Loglikelihood computation
- Rolling loglikelihood (perplexity)
- Text generation
- CLI functionality
- Integration with lm-eval tasks
"""

from unittest.mock import Mock, patch

import mlx.core as mx
import pytest

# Skip entire module if lm_eval is not installed
pytest.importorskip("lm_eval", reason="lm-eval not installed. Install with: pip install 'lm-eval>=0.4.0'")

from smlx.evals.lm_harness import (
    POPULAR_TASKS,
    _pad_inputs,
    _rstrip_until,
    list_tasks,
    run_evaluation,
)


class TestHelperFunctions:
    """Test helper functions."""

    @pytest.mark.unit
    def test_pad_inputs_basic(self):
        """Test basic input padding."""
        inputs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        padded, lengths = _pad_inputs(inputs)

        assert padded.shape == (3, 4)
        assert lengths.shape == (3,)
        assert lengths[0].item() == 3
        assert lengths[1].item() == 2
        assert lengths[2].item() == 4

    @pytest.mark.unit
    def test_pad_inputs_single(self):
        """Test padding with single input."""
        inputs = [[1, 2, 3]]
        padded, lengths = _pad_inputs(inputs)

        assert padded.shape == (1, 3)
        assert lengths[0].item() == 3

    @pytest.mark.unit
    def test_pad_inputs_all_same_length(self):
        """Test padding when all inputs are same length."""
        inputs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        padded, lengths = _pad_inputs(inputs)

        assert padded.shape == (3, 3)
        assert all(length.item() == 3 for length in lengths)

    @pytest.mark.unit
    def test_rstrip_until_basic(self):
        """Test basic stop sequence truncation."""
        text = "Hello world<|endoftext|>More text"
        result = _rstrip_until(text, ["<|endoftext|>"])
        assert result == "Hello world"

    @pytest.mark.unit
    def test_rstrip_until_multiple_stops(self):
        """Test with multiple stop sequences."""
        text = "Hello world\n\nMore text###End"
        result = _rstrip_until(text, ["\n\n", "###"])
        assert result == "Hello world"

    @pytest.mark.unit
    def test_rstrip_until_no_match(self):
        """Test when no stop sequence found."""
        text = "Hello world"
        result = _rstrip_until(text, ["###", "\n\n"])
        assert result == "Hello world"

    @pytest.mark.unit
    def test_rstrip_until_empty(self):
        """Test with empty string."""
        result = _rstrip_until("", ["###"])
        assert result == ""


class TestSMLXLMInitialization:
    """Test SMLXLM wrapper initialization."""

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    def test_init_basic(self, mock_load):
        """Test basic initialization."""
        from smlx.evals.lm_harness import SMLXLM

        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Initialize
        lm = SMLXLM(model_path="mlx-community/SmolLM2-135M-Instruct")

        # Check initialization
        assert lm._model == mock_model
        assert lm.tokenizer == mock_tokenizer
        assert lm._batch_size == 8  # Default
        mock_load.assert_called_once_with("mlx-community/SmolLM2-135M-Instruct")

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    def test_init_custom_batch_size(self, mock_load):
        """Test initialization with custom batch size."""
        from smlx.evals.lm_harness import SMLXLM

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        lm = SMLXLM(
            model_path="mlx-community/SmolLM2-135M-Instruct",
            batch_size=16,
        )

        assert lm._batch_size == 16

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    def test_init_custom_max_tokens(self, mock_load):
        """Test initialization with custom max tokens."""
        from smlx.evals.lm_harness import SMLXLM

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        lm = SMLXLM(
            model_path="mlx-community/SmolLM2-135M-Instruct",
            max_tokens=4096,
        )

        assert lm._max_tokens == 4096


class TestTokenization:
    """Test tokenization functionality."""

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    def test_tok_encode(self, mock_load):
        """Test token encoding."""
        from smlx.evals.lm_harness import SMLXLM

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        mock_load.return_value = (mock_model, mock_tokenizer)

        lm = SMLXLM(model_path="mlx-community/SmolLM2-135M-Instruct")

        tokens = lm.tok_encode("Hello world")
        assert tokens == [1, 2, 3, 4, 5]
        mock_tokenizer.encode.assert_called_once_with("Hello world")

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    def test_tok_decode(self, mock_load):
        """Test token decoding."""
        from smlx.evals.lm_harness import SMLXLM

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value="Hello world")
        mock_load.return_value = (mock_model, mock_tokenizer)

        lm = SMLXLM(model_path="mlx-community/SmolLM2-135M-Instruct")

        text = lm.tok_decode([1, 2, 3, 4, 5])
        assert text == "Hello world"
        mock_tokenizer.decode.assert_called_once_with([1, 2, 3, 4, 5])

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    def test_eot_token_id(self, mock_load):
        """Test EOT token ID retrieval."""
        from smlx.evals.lm_harness import SMLXLM

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_load.return_value = (mock_model, mock_tokenizer)

        lm = SMLXLM(model_path="mlx-community/SmolLM2-135M-Instruct")

        assert lm.eot_token_id == 2


class TestScoring:
    """Test batch scoring functionality."""

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    def test_batch_score(self, mock_load):
        """Test batch scoring."""
        from smlx.evals.lm_harness import SMLXLM

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Mock model output
        batch_size = 2
        seq_len = 5
        vocab_size = 1000

        # Create logits
        logits = mx.random.normal((batch_size, seq_len, vocab_size))
        mock_model.return_value = logits

        lm = SMLXLM(model_path="mlx-community/SmolLM2-135M-Instruct")

        # Test batch
        tokens = mx.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        lengths = mx.array([6, 6])

        scores = lm._batch_score(tokens, lengths)

        # Check output shape
        assert isinstance(scores, list)
        assert len(scores) == batch_size

        # Check model was called
        mock_model.assert_called_once()


class TestLoglikelihood:
    """Test loglikelihood computation."""

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    def test_loglikelihood_basic(self, mock_load):
        """Test basic loglikelihood computation."""
        from smlx.evals.lm_harness import SMLXLM

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(side_effect=lambda x: list(range(len(x))))
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Mock model to return predictable logits
        vocab_size = 1000

        def model_side_effect(x):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            return mx.random.normal((batch_size, seq_len, vocab_size))

        mock_model.side_effect = model_side_effect

        lm = SMLXLM(model_path="mlx-community/SmolLM2-135M-Instruct")

        # Test requests
        requests = [
            ("Hello", " world"),
            ("The quick", " brown fox"),
        ]

        results = lm.loglikelihood(requests)

        # Check results structure
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)
        assert all(isinstance(r[0], float) for r in results)
        assert all(isinstance(r[1], bool) for r in results)

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    def test_loglikelihood_truncation(self, mock_load):
        """Test loglikelihood with truncation."""
        from smlx.evals.lm_harness import SMLXLM

        mock_model = Mock()
        mock_tokenizer = Mock()

        # Create long sequence that exceeds max_tokens
        def encode_side_effect(x):
            return list(range(len(x) * 100))  # Long sequence

        mock_tokenizer.encode = Mock(side_effect=encode_side_effect)
        mock_load.return_value = (mock_model, mock_tokenizer)

        lm = SMLXLM(
            model_path="mlx-community/SmolLM2-135M-Instruct",
            max_tokens=100,
        )

        # Request with long context
        requests = [("Very long context", " short continuation")]

        results = lm.loglikelihood(requests)

        # Should handle truncation gracefully
        assert len(results) == 1
        assert isinstance(results[0][0], float)


class TestRollingLoglikelihood:
    """Test rolling loglikelihood (perplexity) computation."""

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    def test_loglikelihood_rolling_basic(self, mock_load):
        """Test rolling loglikelihood computation."""
        from smlx.evals.lm_harness import SMLXLM

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(side_effect=lambda x: list(range(len(x))))
        mock_load.return_value = (mock_model, mock_tokenizer)

        vocab_size = 1000

        def model_side_effect(x):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            return mx.random.normal((batch_size, seq_len, vocab_size))

        mock_model.side_effect = model_side_effect

        lm = SMLXLM(model_path="mlx-community/SmolLM2-135M-Instruct")

        # Test requests (full strings, not context/continuation)
        requests = [
            ("Hello world this is a test",),
            ("The quick brown fox jumps over the lazy dog",),
        ]

        results = lm.loglikelihood_rolling(requests)

        # Check results
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, float) for r in results)


class TestGeneration:
    """Test text generation."""

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    @patch('smlx.evals.lm_harness.generate')
    def test_generate_until_basic(self, mock_generate, mock_load):
        """Test basic text generation."""
        from smlx.evals.lm_harness import SMLXLM

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(side_effect=lambda x: list(range(len(x))))
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Mock generation
        mock_generate.return_value = "Generated text"

        lm = SMLXLM(model_path="mlx-community/SmolLM2-135M-Instruct")

        # Test requests
        requests = [
            ("Hello", {"until": ["\n"], "max_gen_toks": 50}),
            ("The quick", {"until": [".", "\n"], "max_gen_toks": 100}),
        ]

        results = lm.generate_until(requests)

        # Check results
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.unit
    @patch('smlx.evals.lm_harness.load')
    @patch('smlx.evals.lm_harness.generate')
    def test_generate_until_with_stop_sequences(self, mock_generate, mock_load):
        """Test generation with stop sequences."""
        from smlx.evals.lm_harness import SMLXLM

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(side_effect=lambda x: list(range(len(x))))
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Mock generation with stop sequence
        mock_generate.return_value = "Generated text\n\nMore text"

        lm = SMLXLM(model_path="mlx-community/SmolLM2-135M-Instruct")

        requests = [("Hello", {"until": ["\n\n"], "max_gen_toks": 50})]

        results = lm.generate_until(requests)

        # Should truncate at stop sequence
        assert len(results) == 1
        assert "\n\n" not in results[0] or results[0].endswith("\n\n")


class TestCLIFunctions:
    """Test CLI functionality."""

    @pytest.mark.unit
    def test_list_tasks_output(self, capsys):
        """Test list_tasks prints correctly."""
        list_tasks()

        captured = capsys.readouterr()
        assert "Popular LM Evaluation Tasks" in captured.out
        assert "hellaswag" in captured.out
        assert "Common Sense Reasoning" in captured.out

    @pytest.mark.unit
    def test_popular_tasks_structure(self):
        """Test POPULAR_TASKS dictionary structure."""
        assert isinstance(POPULAR_TASKS, dict)
        assert "hellaswag" in POPULAR_TASKS
        assert "description" in POPULAR_TASKS["hellaswag"]
        assert "category" in POPULAR_TASKS["hellaswag"]
        assert "num_fewshot" in POPULAR_TASKS["hellaswag"]

    @pytest.mark.unit
    @patch('lm_eval.simple_evaluate')
    @patch('smlx.evals.lm_harness.SMLXLM')
    def test_run_evaluation_basic(self, mock_smlxlm, mock_evaluate):
        """Test run_evaluation function."""
        # Mock evaluation results
        mock_evaluate.return_value = {
            "results": {
                "hellaswag": {
                    "acc": 0.456,
                    "acc_stderr": 0.012,
                    "acc_norm": 0.478,
                    "acc_norm_stderr": 0.011,
                }
            },
            "config": {"model": "smlx"},
        }

        # Run evaluation
        results = run_evaluation(
            model_path="mlx-community/SmolLM2-135M-Instruct",
            tasks=["hellaswag"],
            batch_size=4,
            limit=10,
            verbose=False,
        )

        # Check results
        assert isinstance(results, dict)
        assert "results" in results
        assert "hellaswag" in results["results"]

        # Check that evaluate was called
        mock_evaluate.assert_called_once()


@pytest.mark.integration
@pytest.mark.requires_model
@pytest.mark.slow
class TestFullEvaluation:
    """Integration tests with real model (requires download)."""

    def test_full_evaluation_smollm2(self):
        """Test full evaluation pipeline with SmolLM2-135M."""
        pytest.skip(
            "Requires model download and lm-eval install - run manually with SMLX_DOWNLOAD_TEST_MODELS=1"
        )

        # This test would run a real evaluation
        from smlx.evals.lm_harness import run_evaluation

        results = run_evaluation(
            model_path="mlx-community/SmolLM2-135M-Instruct",
            tasks=["hellaswag"],
            limit=10,  # Small sample for testing
            batch_size=2,
            verbose=True,
        )

        assert "results" in results
        assert "hellaswag" in results["results"]
        assert "acc" in results["results"]["hellaswag"]

    def test_smlxlm_with_real_model(self):
        """Test SMLXLM wrapper with real model."""
        pytest.skip("Requires model download - run manually with SMLX_DOWNLOAD_TEST_MODELS=1")

        from smlx.evals.lm_harness import SMLXLM

        lm = SMLXLM(model_path="mlx-community/SmolLM2-135M-Instruct")

        # Test loglikelihood
        requests = [("Hello", " world")]
        results = lm.loglikelihood(requests)

        assert len(results) == 1
        assert isinstance(results[0][0], float)
        assert isinstance(results[0][1], bool)

        # Test generation
        gen_requests = [("Once upon a time", {"until": ["\n"], "max_gen_toks": 20})]
        gen_results = lm.generate_until(gen_requests)

        assert len(gen_results) == 1
        assert isinstance(gen_results[0], str)
        assert len(gen_results[0]) > 0
