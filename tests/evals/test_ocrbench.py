"""
Tests for OCRBench evaluation module.

Tests cover:
- Question formatting (minimal processing)
- Answer normalization (strip whitespace)
- Evaluation logic with substring matching
- Multiple ground truth handling
- Mock integration tests
"""

import pytest

from smlx.evals.ocrbench import evaluate_answer, normalize_answer, process_question


class TestQuestionFormatting:
    """Test question formatting for OCRBench."""

    @pytest.mark.unit
    def test_process_question_simple(self):
        """Test that questions are returned as-is."""
        sample = {"question": "What is the text in this image?"}
        result = process_question(sample)
        assert result == "What is the text in this image?"

    @pytest.mark.unit
    def test_process_question_complex(self):
        """Test processing complex OCR questions."""
        sample = {
            "question": "Extract all the text from the document and list the key information."
        }
        result = process_question(sample)
        assert result == "Extract all the text from the document and list the key information."

    @pytest.mark.unit
    def test_process_question_with_special_chars(self):
        """Test questions with special characters."""
        sample = {"question": "What is written in the image? (Include punctuation!)"}
        result = process_question(sample)
        assert result == "What is written in the image? (Include punctuation!)"


class TestAnswerNormalization:
    """Test answer normalization for OCRBench."""

    @pytest.mark.unit
    def test_normalize_simple_text(self):
        """Test normalizing simple text responses."""
        response = "The text is Hello World"
        result = normalize_answer(response, {})
        assert result == "The text is Hello World"

    @pytest.mark.unit
    def test_normalize_with_whitespace(self):
        """Test stripping leading/trailing whitespace."""
        response = "  Some text with spaces  "
        result = normalize_answer(response, {})
        assert result == "Some text with spaces"

    @pytest.mark.unit
    def test_normalize_empty_response(self):
        """Test handling empty responses."""
        response = ""
        result = normalize_answer(response, {})
        assert result is None

    @pytest.mark.unit
    def test_normalize_none_response(self):
        """Test handling None responses."""
        response = None
        result = normalize_answer(response, {})
        assert result is None

    @pytest.mark.unit
    def test_normalize_multiline_text(self):
        """Test normalizing multiline text (preserves internal formatting)."""
        response = "  Line 1\nLine 2\nLine 3  "
        result = normalize_answer(response, {})
        assert result == "Line 1\nLine 2\nLine 3"


class TestEvaluateAnswer:
    """Test answer evaluation logic for OCRBench."""

    @pytest.mark.unit
    def test_evaluate_single_exact_match(self):
        """Test exact match with single ground truth."""
        prediction = "42"
        ground_truth = ["42"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_evaluate_substring_match(self):
        """Test substring matching (key feature of OCRBench)."""
        prediction = "The answer is 42 degrees"
        ground_truth = ["42"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_evaluate_multiple_ground_truths(self):
        """Test matching against multiple acceptable answers."""
        prediction = "The result is forty-two"
        ground_truth = ["42", "forty-two", "fourty-two"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_evaluate_case_insensitive(self):
        """Test case-insensitive matching."""
        prediction = "The ANSWER is HELLO WORLD"
        ground_truth = ["hello world"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_evaluate_no_match(self):
        """Test when prediction doesn't match any ground truth."""
        prediction = "The answer is 43"
        ground_truth = ["42", "forty-two"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is False

    @pytest.mark.unit
    def test_evaluate_none_prediction(self):
        """Test handling None prediction."""
        prediction = None
        ground_truth = ["42"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is False

    @pytest.mark.unit
    def test_evaluate_empty_prediction(self):
        """Test handling empty prediction."""
        prediction = ""
        ground_truth = ["42"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is False

    @pytest.mark.unit
    def test_evaluate_partial_word_match(self):
        """Test substring matching within words."""
        prediction = "The category is categorical"
        ground_truth = ["category"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_evaluate_multiple_truths_first_matches(self):
        """Test when first ground truth matches."""
        prediction = "The answer is apple pie"
        ground_truth = ["apple", "banana", "cherry"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_evaluate_multiple_truths_last_matches(self):
        """Test when last ground truth matches."""
        prediction = "The answer is cherry pie"
        ground_truth = ["apple", "banana", "cherry"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_evaluate_with_punctuation(self):
        """Test matching with punctuation in ground truth."""
        prediction = "The text reads: Hello, World!"
        ground_truth = ["hello, world!"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_evaluate_numeric_strings(self):
        """Test matching numeric strings."""
        prediction = "The number is 123.45"
        ground_truth = ["123.45"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_evaluate_whitespace_in_ground_truth(self):
        """Test handling whitespace in ground truth."""
        prediction = "The answer is hello world"
        ground_truth = ["  hello world  "]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_empty_ground_truth_list(self):
        """Test evaluation with empty ground truth list."""
        prediction = "some answer"
        ground_truth = []
        result = evaluate_answer(prediction, ground_truth)
        assert result is False

    @pytest.mark.unit
    def test_special_characters_in_prediction(self):
        """Test handling special characters."""
        prediction = "The answer is: café ñoño™"
        ground_truth = ["café"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_unicode_characters(self):
        """Test Unicode character handling."""
        prediction = "答案是 42"
        ground_truth = ["42"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_very_long_prediction(self):
        """Test with very long prediction text."""
        prediction = "Lorem ipsum " * 100 + "the answer is 42 " + "dolor sit amet " * 100
        ground_truth = ["42"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True

    @pytest.mark.unit
    def test_ground_truth_with_newlines(self):
        """Test ground truth containing newlines."""
        prediction = "Line 1\nLine 2\nLine 3"
        ground_truth = ["line 2"]
        result = evaluate_answer(prediction, ground_truth)
        assert result is True


class TestIntegrationMock:
    """Mock integration tests for the full pipeline."""

    @pytest.mark.unit
    def test_full_pipeline_correct_answer(self):
        """Test full pipeline with correct answer."""
        sample = {"question": "What text is in the image?"}
        question = process_question(sample)
        assert question == "What text is in the image?"

        # Simulate model response
        model_response = "The text in the image is 'STOP'"
        prediction = normalize_answer(model_response, sample)

        ground_truth = ["STOP", "stop"]
        is_correct = evaluate_answer(prediction, ground_truth)

        assert is_correct is True

    @pytest.mark.unit
    def test_full_pipeline_incorrect_answer(self):
        """Test full pipeline with incorrect answer."""
        sample = {"question": "What number appears in the image?"}
        question = process_question(sample)
        assert question == "What number appears in the image?"

        # Simulate model response
        model_response = "The number shown is 99"
        prediction = normalize_answer(model_response, sample)

        ground_truth = ["42", "forty-two"]
        is_correct = evaluate_answer(prediction, ground_truth)

        assert is_correct is False

    @pytest.mark.unit
    def test_full_pipeline_multiple_acceptable_answers(self):
        """Test pipeline with multiple acceptable answers."""
        sample = {"question": "Extract the date from the document"}
        question = process_question(sample)
        assert question == "Extract the date from the document"

        # Simulate model response
        model_response = "The date is 2024-01-15"
        prediction = normalize_answer(model_response, sample)

        ground_truth = ["2024-01-15", "January 15, 2024", "01/15/2024"]
        is_correct = evaluate_answer(prediction, ground_truth)

        assert is_correct is True

    @pytest.mark.unit
    def test_full_pipeline_empty_response(self):
        """Test pipeline handling empty model response."""
        sample = {"question": "What is written here?"}
        question = process_question(sample)
        assert question == "What is written here?"

        # Simulate empty model response
        model_response = ""
        prediction = normalize_answer(model_response, sample)

        ground_truth = ["some text"]
        is_correct = evaluate_answer(prediction, ground_truth)

        assert is_correct is False

    @pytest.mark.unit
    def test_full_pipeline_ocr_number_extraction(self):
        """Test OCR number extraction scenario."""
        sample = {"question": "What is the total amount shown?"}
        question = process_question(sample)
        assert question == "What is the total amount shown?"

        # Simulate model response
        model_response = "Total Amount: $1,234.56"
        prediction = normalize_answer(model_response, sample)

        ground_truth = ["1,234.56", "$1,234.56", "1234.56"]
        is_correct = evaluate_answer(prediction, ground_truth)

        assert is_correct is True


class TestDatasetIntegration:
    """Test dataset-specific scenarios."""

    @pytest.mark.unit
    def test_semicolon_separated_ground_truth(self):
        """Test handling semicolon-separated ground truth (CSV format)."""
        # This tests the format used when loading from CSV files
        ground_truth_str = "answer1; answer2; answer3"
        ground_truth_list = [a.strip() for a in ground_truth_str.split(";")]

        prediction = "The result is answer2"
        result = evaluate_answer(prediction, ground_truth_list)
        assert result is True

    @pytest.mark.unit
    def test_regular_text_recognition(self):
        """Test Regular Text Recognition task type."""
        sample = {"question": "Recognize all the text in the image", "type": "Regular Text Recognition"}
        question = process_question(sample)
        assert question == "Recognize all the text in the image"

        model_response = "HELLO WORLD 2024"
        prediction = normalize_answer(model_response, sample)

        ground_truth = ["HELLO WORLD 2024"]
        is_correct = evaluate_answer(prediction, ground_truth)
        assert is_correct is True

    @pytest.mark.unit
    def test_irregular_text_recognition(self):
        """Test Irregular Text Recognition task type."""
        sample = {"question": "What text appears in the curved banner?", "type": "Irregular Text Recognition"}
        question = process_question(sample)
        assert question == "What text appears in the curved banner?"

        model_response = "The curved text says GRAND OPENING"
        prediction = normalize_answer(model_response, sample)

        ground_truth = ["GRAND OPENING", "Grand Opening"]
        is_correct = evaluate_answer(prediction, ground_truth)
        assert is_correct is True


@pytest.mark.slow
@pytest.mark.eval
@pytest.mark.requires_model
class TestFullEvaluation:
    """Full evaluation tests requiring model and dataset downloads."""

    def test_ocrbench_predictions_evaluation_logic(self):
        """Test evaluation logic with predictions."""
        # Test that we can handle predictions correctly using evaluate_answer
        test_cases = [
            {
                "prediction": "The text says HELLO",
                "ground_truth": ["HELLO"],
                "expected": True,
            },
            {
                "prediction": "The number is 42",
                "ground_truth": ["42", "forty-two"],
                "expected": True,
            },
            {
                "prediction": "Some random text",
                "ground_truth": ["specific answer"],
                "expected": False,
            },
        ]

        for tc in test_cases:
            result = evaluate_answer(tc["prediction"], tc["ground_truth"])
            assert result == tc["expected"], f"Failed for: {tc}"

        # Calculate accuracy
        correct = sum(1 for tc in test_cases if tc["expected"])
        accuracy = correct / len(test_cases) if test_cases else 0.0

        assert accuracy == 2.0 / 3.0  # 2 out of 3 correct

    @pytest.mark.unit
    def test_ocrbench_small_sample(self, smolvlm_256m_model, ocrbench_dataset):
        """Test OCRBench evaluation on a small sample."""
        from smlx.evals.utils import inference

        model = smolvlm_256m_model["model"]
        processor = smolvlm_256m_model["processor"]
        dataset = ocrbench_dataset["dataset"]

        # Test on first 2 samples only
        max_samples = 2
        results = []

        for idx, sample in enumerate(dataset):
            if idx >= max_samples:
                break

            # Get question and image
            question = process_question(sample)
            image = sample.get("image")
            ground_truth = sample.get("answer", [])

            # Skip samples without images
            if image is None:
                continue

            # Run inference (inference() applies chat template automatically)
            response = inference(
                model,
                processor,
                question,
                image=image,
                max_tokens=512,
                temperature=0.0,
                verbose=False,
            )

            # Normalize and evaluate
            prediction = normalize_answer(response, sample)
            is_correct = evaluate_answer(prediction, ground_truth)

            results.append(
                {
                    "question": question[:100],
                    "response": response[:100],
                    "prediction": prediction[:100] if prediction else "",
                    "ground_truth": ground_truth,
                    "correct": is_correct,
                }
            )

        # Verify we got results
        assert len(results) > 0, "Should have processed at least one sample"
        # Note: We may process fewer than max_samples if some don't have images

        print(f"\n\nProcessed {len(results)} samples from OCRBench (checked {idx + 1} total)")
        for i, result in enumerate(results):
            print(f"\nSample {i + 1}:")
            print(f"  Question: {result['question']}")
            print(f"  Response: {result['response']}")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Ground Truth: {result['ground_truth']}")
            print(f"  Correct: {result['correct']}")
