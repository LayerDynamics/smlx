"""
Tests for MathVista evaluation module.

Tests cover:
- Question formatting for multiple choice
- Answer normalization with various patterns
- Evaluation logic and correctness checking
- Mock integration tests
"""

import pytest

from smlx.evals.math_vista import (
    evaluate_answer,
    normalize_answer,
    process_question,
)


class TestQuestionFormatting:
    """Test question formatting with choices."""

    @pytest.mark.unit
    def test_process_question_multiple_choice(self):
        """Test formatting multiple choice questions with (A), (B), (C) format."""
        sample = {
            "query": "What is 2+2?",
            "question_type": "multi_choice",
            "choices": ["3", "4", "5"],
        }
        result = process_question(sample)
        assert "What is 2+2?" in result
        assert "(A) 3" in result
        assert "(B) 4" in result
        assert "(C) 5" in result

    @pytest.mark.unit
    def test_process_question_free_form(self):
        """Test formatting free-form questions (no choices)."""
        sample = {
            "query": "What is the area?",
            "question_type": "free_form",
            "choices": None,
        }
        result = process_question(sample)
        assert result == "What is the area?"
        assert "(A)" not in result

    @pytest.mark.unit
    def test_process_question_empty_choices(self):
        """Test handling empty choices list."""
        sample = {
            "query": "Solve this problem.",
            "question_type": "multi_choice",
            "choices": [],
        }
        result = process_question(sample)
        assert result == "Solve this problem."


class TestAnswerNormalization:
    """Test answer extraction and normalization."""

    @pytest.mark.unit
    def test_normalize_boxed_multiple_choice(self):
        """Test extracting boxed answer like \\boxed{A}."""
        problem = {
            "question_type": "multi_choice",
            "answer_type": "text",
            "choices": ["Red", "Blue", "Green"],
        }
        response = "The answer is \\boxed{B}"
        result = normalize_answer(response, problem)
        assert result == "Blue"

    @pytest.mark.unit
    def test_normalize_explicit_answer_pattern(self):
        """Test 'the answer is X' pattern."""
        problem = {
            "question_type": "multi_choice",
            "answer_type": "text",
            "choices": ["Option 1", "Option 2", "Option 3"],
        }
        response = "After careful analysis, the answer is C"
        result = normalize_answer(response, problem)
        assert result == "Option 3"

    @pytest.mark.unit
    def test_normalize_isolated_letter(self):
        """Test extracting isolated letter from end of response."""
        problem = {
            "question_type": "multi_choice",
            "answer_type": "text",
            "choices": ["First", "Second", "Third"],
        }
        response = "This is complicated. The correct choice is A."
        result = normalize_answer(response, problem)
        assert result == "First"

    @pytest.mark.unit
    def test_normalize_integer_boxed(self):
        """Test extracting boxed integer answer."""
        problem = {
            "question_type": "free_form",
            "answer_type": "integer",
            "choices": [],
        }
        response = "The calculation yields \\boxed{42}"
        result = normalize_answer(response, problem)
        assert result == "42"

    @pytest.mark.unit
    def test_normalize_integer_with_commas(self):
        """Test extracting integer with comma formatting."""
        problem = {
            "question_type": "free_form",
            "answer_type": "integer",
            "choices": [],
        }
        response = "The total is 7,518 units"
        result = normalize_answer(response, problem)
        assert result == "7518"

    @pytest.mark.unit
    def test_normalize_integer_scientific_notation(self):
        """Test extracting integer from scientific notation."""
        problem = {
            "question_type": "free_form",
            "answer_type": "integer",
            "choices": [],
        }
        response = "The result is 1.5e3"
        result = normalize_answer(response, problem)
        assert result == "1500"

    @pytest.mark.unit
    def test_normalize_float_with_precision(self):
        """Test extracting float answer with precision."""
        problem = {
            "question_type": "free_form",
            "answer_type": "float",
            "precision": 2,
            "choices": [],
        }
        response = "The answer is 3.14159"
        result = normalize_answer(response, problem)
        assert result == "3.14"

    @pytest.mark.unit
    def test_normalize_float_boxed(self):
        """Test extracting boxed float answer."""
        problem = {
            "question_type": "free_form",
            "answer_type": "float",
            "precision": 1,
            "choices": [],
        }
        response = "Therefore, \\boxed{2.718}"
        result = normalize_answer(response, problem)
        assert result == "2.7"

    @pytest.mark.unit
    def test_normalize_chinese_pattern(self):
        """Test Chinese answer pattern '故选：A'."""
        problem = {
            "question_type": "multi_choice",
            "answer_type": "text",
            "choices": ["选项1", "选项2", "选项3"],
        }
        response = "经过分析，故选：B"
        result = normalize_answer(response, problem)
        assert result == "选项2"

    @pytest.mark.unit
    def test_normalize_empty_response(self):
        """Test handling empty response."""
        problem = {
            "question_type": "multi_choice",
            "answer_type": "text",
            "choices": ["A", "B", "C"],
        }
        response = ""
        result = normalize_answer(response, problem)
        assert result is None

    @pytest.mark.unit
    def test_normalize_edit_distance_fallback(self):
        """Test fuzzy matching with edit distance for multiple choice."""
        problem = {
            "question_type": "multi_choice",
            "answer_type": "text",
            "choices": ["Red", "Blue", "Green"],
        }
        # Model responds with just the choice (with typo)
        # "Blu" has no letter indicators (not B, not isolated letter)
        response = "Blu"  # Typo - should match Blue via edit distance
        result = normalize_answer(response, problem)
        # Should match "Blue" due to small edit distance
        # Red: 3, Blue: 1, Green: 4
        assert result == "Blue"


class TestEvaluateAnswer:
    """Test answer evaluation and comparison logic."""

    @pytest.mark.unit
    def test_evaluate_exact_match(self):
        """Test exact string match."""
        assert evaluate_answer("42", "42") is True
        assert evaluate_answer("Blue", "Blue") is True

    @pytest.mark.unit
    def test_evaluate_mismatch(self):
        """Test mismatched answers."""
        assert evaluate_answer("42", "43") is False
        assert evaluate_answer("Red", "Blue") is False

    @pytest.mark.unit
    def test_evaluate_none_prediction(self):
        """Test handling None prediction."""
        assert evaluate_answer(None, "42") is False

    @pytest.mark.unit
    def test_evaluate_numeric_words(self):
        """Test numeric word matching (e.g., 'one' == '1')."""
        assert evaluate_answer("one", "1") is True
        assert evaluate_answer("1", "one") is True
        assert evaluate_answer("five", "5") is True
        assert evaluate_answer("twelve", "12") is True

    @pytest.mark.unit
    def test_evaluate_case_insensitive(self):
        """Test case-insensitive comparison after normalization."""
        # After word-to-num conversion, these should match
        assert evaluate_answer("FIVE", "5") is True
        assert evaluate_answer("Ten", "10") is True

    @pytest.mark.unit
    def test_evaluate_whitespace_handling(self):
        """Test whitespace trimming."""
        assert evaluate_answer("  42  ", "42") is True
        assert evaluate_answer("answer", "  answer  ") is True


class TestIntegrationMock:
    """Mock integration tests for the full evaluation pipeline."""

    @pytest.mark.unit
    def test_multiple_choice_workflow(self):
        """Test complete workflow for multiple choice question."""
        # Sample from dataset
        sample = {
            "query": "What color is the sky?",
            "question_type": "multi_choice",
            "answer_type": "text",
            "choices": ["Red", "Blue", "Green"],
        }

        # Format question
        question = process_question(sample)
        assert "(A) Red" in question
        assert "(B) Blue" in question

        # Simulate model response
        model_response = "Based on the image, the answer is B"

        # Normalize
        prediction = normalize_answer(model_response, sample)
        assert prediction == "Blue"

        # Evaluate
        ground_truth = "Blue"
        is_correct = evaluate_answer(prediction, ground_truth)
        assert is_correct is True

    @pytest.mark.unit
    def test_integer_workflow(self):
        """Test complete workflow for integer answer question."""
        sample = {
            "query": "How many objects are in the image?",
            "question_type": "free_form",
            "answer_type": "integer",
            "choices": [],
        }

        # Format question
        question = process_question(sample)
        assert question == sample["query"]

        # Simulate model response
        model_response = "I can count 15 objects in total."

        # Normalize
        prediction = normalize_answer(model_response, sample)
        assert prediction == "15"

        # Evaluate
        ground_truth = "15"
        is_correct = evaluate_answer(prediction, ground_truth)
        assert is_correct is True

    @pytest.mark.unit
    def test_float_workflow(self):
        """Test complete workflow for float answer question."""
        sample = {
            "query": "What is the area of the shape?",
            "question_type": "free_form",
            "answer_type": "float",
            "precision": 2,
            "choices": [],
        }

        # Format question
        question = process_question(sample)
        assert "area" in question.lower()

        # Simulate model response
        model_response = "The area is 12.566 square units"

        # Normalize
        prediction = normalize_answer(model_response, sample)
        assert prediction == "12.57"  # Rounded to 2 decimal places

        # Evaluate
        ground_truth = "12.57"
        is_correct = evaluate_answer(prediction, ground_truth)
        assert is_correct is True


# Slow integration test that would require actual model and dataset
@pytest.mark.slow
@pytest.mark.eval
@pytest.mark.requires_model
@pytest.mark.gpu
class TestFullEvaluation:
    """Full evaluation tests requiring model downloads and GPU."""

    def test_mathvista_small_sample(self, smolvlm_256m_model, mathvista_dataset):
        """
        Test full evaluation on a small sample.

        This test requires:
        - mlx-vlm installed
        - HuggingFace datasets installed
        - A vision-language model downloaded
        - MLX/GPU support

        Run with: SMLX_DOWNLOAD_TEST_MODELS=1 pytest -m "eval and slow" tests/evals/test_math_vista.py
        """
        from smlx.evals.utils import inference

        model = smolvlm_256m_model["model"]
        processor = smolvlm_256m_model["processor"]
        dataset = mathvista_dataset["dataset"]

        # Test on first 3 samples only
        max_samples = 3
        results = []

        for idx, sample in enumerate(dataset):
            if idx >= max_samples:
                break

            # Skip samples without images
            if "decoded_image" not in sample or sample["decoded_image"] is None:
                continue

            # Get the image
            image = sample["decoded_image"]
            if hasattr(image, "convert"):
                image = image.convert("RGB")

            # Format question
            question = process_question(sample)

            # Run inference
            response = inference(
                model,
                processor,
                question,
                image=image,
                max_tokens=512,
                temperature=0.0,
                verbose=False,
            )

            # Normalize answer
            prediction = normalize_answer(response, sample)

            # Evaluate
            ground_truth = sample.get("answer", "")
            is_correct = evaluate_answer(prediction, ground_truth) if ground_truth else None

            results.append(
                {
                    "question": sample.get("query", ""),
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "correct": is_correct,
                }
            )

        # Verify we got at least one result
        assert len(results) > 0, "Should have processed at least one sample"

        # Print results for debugging
        print(f"\n\nProcessed {len(results)} samples from MathVista:")
        for i, result in enumerate(results):
            print(f"\nSample {i+1}:")
            print(f"  Question: {result['question'][:100]}...")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Ground Truth: {result['ground_truth']}")
            print(f"  Correct: {result['correct']}")


# Edge cases and error handling
class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_normalize_very_long_response(self):
        """Test handling very long model responses."""
        problem = {
            "question_type": "multi_choice",
            "answer_type": "text",
            "choices": ["A", "B", "C"],
        }
        # Very long response with answer at the end
        response = "x" * 10000 + " the answer is B"
        result = normalize_answer(response, problem)
        assert result == "B"

    @pytest.mark.unit
    def test_normalize_multiple_boxed_answers(self):
        """Test when response has multiple boxed patterns (use first)."""
        problem = {
            "question_type": "free_form",
            "answer_type": "integer",
            "choices": [],
        }
        response = "First \\boxed{10} then \\boxed{20}"
        result = normalize_answer(response, problem)
        assert result == "10"

    @pytest.mark.unit
    def test_process_question_unicode(self):
        """Test handling unicode characters in questions and choices."""
        sample = {
            "query": "What is π approximately?",
            "question_type": "multi_choice",
            "choices": ["3.14", "2.71", "1.41"],
        }
        result = process_question(sample)
        assert "π" in result
        assert "(A) 3.14" in result

    @pytest.mark.unit
    def test_evaluate_special_characters(self):
        """Test evaluation with special characters."""
        assert evaluate_answer("$100", "$100") is True
        assert evaluate_answer("50%", "50%") is True

    @pytest.mark.unit
    def test_normalize_negative_numbers(self):
        """Test handling negative numbers."""
        problem = {
            "question_type": "free_form",
            "answer_type": "integer",
            "choices": [],
        }
        response = "The answer is -42"
        result = normalize_answer(response, problem)
        assert result == "-42"

    @pytest.mark.unit
    def test_normalize_answer_with_units(self):
        """Test extracting numbers from answers with units."""
        problem = {
            "question_type": "free_form",
            "answer_type": "integer",
            "choices": [],
        }
        response = "The distance is 100 meters"
        result = normalize_answer(response, problem)
        assert result == "100"
