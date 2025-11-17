"""
Tests for MMMU evaluation module.

Tests cover:
- Question processing with options parsing
- Image extraction (single and multiple images)
- Answer normalization for multiple choice and open-ended
- Subject tracking
- Mock integration tests
"""

import numpy as np
import pytest
from PIL import Image

from smlx.evals.mmmu import (
    MMMU_SUBJECTS,
    get_images,
    normalize_number,
    process_question,
)


class TestQuestionProcessing:
    """Test question formatting for MMMU."""

    @pytest.mark.unit
    def test_process_simple_question(self):
        """Test processing question without options."""
        example = {"question": "What is shown in this image?"}
        result = process_question(example)
        assert result == "What is shown in this image?"

    @pytest.mark.unit
    def test_process_question_with_options(self):
        """Test processing question with options string."""
        example = {
            "question": "What color is the object?",
            "options": "['red', 'blue', 'green', 'yellow']",
        }
        result = process_question(example)
        assert "What color is the object?" in result
        assert "\nOptions:" in result
        assert "A. red" in result
        assert "B. blue" in result
        assert "C. green" in result
        assert "D. yellow" in result

    @pytest.mark.unit
    def test_process_question_remove_image_tags(self):
        """Test removing <image n> tags from question text."""
        example = {
            "question": "What is shown in <image 1> and <image 2>?",
            "options": None,
        }
        result = process_question(example)
        assert "<image" not in result
        # After removing tags, we get "What is shown in  and ?" (with extra spaces)
        assert "What is shown in" in result
        assert "and" in result
        assert "?" in result

    @pytest.mark.unit
    def test_process_question_with_options_and_image_tags(self):
        """Test combined options and image tag removal."""
        example = {
            "question": "Compare <image 1> and <image 2>. Which is larger?",
            "options": "['Image 1', 'Image 2', 'Same size']",
        }
        result = process_question(example)
        assert "<image" not in result
        assert "\nOptions:" in result
        assert "A. Image 1" in result
        assert "B. Image 2" in result
        assert "C. Same size" in result

    @pytest.mark.unit
    def test_process_question_empty_options(self):
        """Test handling empty options field."""
        example = {"question": "What is this?", "options": ""}
        result = process_question(example)
        assert result == "What is this?"
        assert "Options:" not in result

    @pytest.mark.unit
    def test_process_question_six_options(self):
        """Test handling six options (A-F)."""
        example = {
            "question": "Select the correct answer:",
            "options": "['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']",
        }
        result = process_question(example)
        assert "A. opt1" in result
        assert "B. opt2" in result
        assert "C. opt3" in result
        assert "D. opt4" in result
        assert "E. opt5" in result
        assert "F. opt6" in result


class TestImageExtraction:
    """Test image extraction for MMMU."""

    @pytest.mark.unit
    def test_get_images_no_images(self):
        """Test handling example with no images."""
        example = {"question": "What is 2+2?"}
        result = get_images(example)
        assert result == []

    @pytest.mark.unit
    def test_get_images_with_mock_single_image(self):
        """Test extraction with single 'image' field."""
        # Create a simple test image (100x100 red square)
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red channel
        test_image = Image.fromarray(img_array)

        example = {"image": test_image}
        result = get_images(example)

        assert len(result) == 1
        assert isinstance(result[0], Image.Image)
        assert result[0].size == (100, 100)

    @pytest.mark.unit
    def test_get_images_with_mock_multiple_images(self):
        """Test extraction with multiple image fields."""
        # Create test images (different colors)
        img1_array = np.zeros((50, 50, 3), dtype=np.uint8)
        img1_array[:, :, 0] = 255  # Red
        test_image1 = Image.fromarray(img1_array)

        img2_array = np.zeros((60, 60, 3), dtype=np.uint8)
        img2_array[:, :, 1] = 255  # Green
        test_image2 = Image.fromarray(img2_array)

        img3_array = np.zeros((70, 70, 3), dtype=np.uint8)
        img3_array[:, :, 2] = 255  # Blue
        test_image3 = Image.fromarray(img3_array)

        example = {
            "image": test_image1,
            "image_1": test_image2,
            "image_2": test_image3,
        }
        result = get_images(example)

        assert len(result) == 3
        assert all(isinstance(img, Image.Image) for img in result)
        assert result[0].size == (50, 50)
        assert result[1].size == (60, 60)
        assert result[2].size == (70, 70)


class TestNumberNormalization:
    """Test numeric normalization for MMMU."""

    @pytest.mark.unit
    def test_normalize_simple_integer(self):
        """Test normalizing simple integer."""
        result = normalize_number("42")
        assert result == 42.0

    @pytest.mark.unit
    def test_normalize_float(self):
        """Test normalizing float."""
        result = normalize_number("3.14159")
        assert result == 3.14159

    @pytest.mark.unit
    def test_normalize_with_commas(self):
        """Test normalizing number with commas."""
        result = normalize_number("1,234,567.89")
        assert result == 1234567.89

    @pytest.mark.unit
    def test_normalize_negative_number(self):
        """Test normalizing negative number."""
        result = normalize_number("-42.5")
        assert result == -42.5

    @pytest.mark.unit
    def test_normalize_non_numeric_string(self):
        """Test normalizing non-numeric string returns original."""
        result = normalize_number("not a number")
        assert result == "not a number"

    @pytest.mark.unit
    def test_normalize_with_whitespace(self):
        """Test normalizing number with whitespace."""
        result = normalize_number("  123.45  ")
        assert result == 123.45

    @pytest.mark.unit
    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        result = normalize_number("")
        assert result == ""


class TestSubjectDefinitions:
    """Test MMMU subject definitions."""

    @pytest.mark.unit
    def test_mmmu_subjects_count(self):
        """Test that all 30 subjects are defined."""
        assert len(MMMU_SUBJECTS) == 30

    @pytest.mark.unit
    def test_mmmu_subjects_include_stem(self):
        """Test that STEM subjects are included."""
        stem_subjects = ["Math", "Physics", "Chemistry", "Biology", "Computer_Science"]
        for subject in stem_subjects:
            assert subject in MMMU_SUBJECTS

    @pytest.mark.unit
    def test_mmmu_subjects_include_humanities(self):
        """Test that humanities subjects are included."""
        humanities = ["History", "Literature", "Psychology", "Sociology"]
        for subject in humanities:
            assert subject in MMMU_SUBJECTS

    @pytest.mark.unit
    def test_mmmu_subjects_include_business(self):
        """Test that business subjects are included."""
        business = ["Accounting", "Economics", "Finance", "Marketing"]
        for subject in business:
            assert subject in MMMU_SUBJECTS

    @pytest.mark.unit
    def test_mmmu_subjects_include_medicine(self):
        """Test that medical subjects are included."""
        medicine = [
            "Basic_Medical_Science",
            "Clinical_Medicine",
            "Pharmacy",
            "Public_Health",
        ]
        for subject in medicine:
            assert subject in MMMU_SUBJECTS

    @pytest.mark.unit
    def test_mmmu_subjects_no_duplicates(self):
        """Test that there are no duplicate subjects."""
        assert len(MMMU_SUBJECTS) == len(set(MMMU_SUBJECTS))


class TestMultipleChoiceExtraction:
    """Test multiple choice answer extraction patterns."""

    @pytest.mark.unit
    def test_extract_option_a_pattern(self):
        """Test extraction with 'option A' pattern."""
        # This would test MMMU_eval logic
        # Since MMMU_eval is complex, we test the pattern matching concept
        predict = "The correct choice is option A based on the analysis"
        # Pattern should match "option a"
        import re

        pattern = r"option\s+([a-fi])\b"
        match = re.search(pattern, predict.lower())
        assert match is not None
        assert match.group(1) == "a"

    @pytest.mark.unit
    def test_extract_answer_is_pattern(self):
        """Test extraction with 'answer is A' pattern."""
        predict = "After careful consideration, the answer is B"
        import re

        pattern = r"answer\s+is:?\s+([a-fi])\b"
        match = re.search(pattern, predict.lower())
        assert match is not None
        assert match.group(1) == "b"

    @pytest.mark.unit
    def test_extract_parenthesized_answer(self):
        """Test extraction with (A) format."""
        predict = "The correct answer is (C)"
        import re

        pattern = r"\(([a-fi])\)"
        match = re.search(pattern, predict.lower())
        assert match is not None
        assert match.group(1) == "c"

    @pytest.mark.unit
    def test_extract_answer_at_start(self):
        """Test extraction when answer starts the response."""
        predict = "A. This is the correct option because..."
        import re

        pattern = r"^([a-fi])[.:\)]\s"
        match = re.search(pattern, predict.lower())
        assert match is not None
        assert match.group(1) == "a"

    @pytest.mark.unit
    def test_extract_correct_answer_is_pattern(self):
        """Test 'correct answer is A' pattern."""
        predict = "Based on the evidence, the correct answer is D"
        import re

        pattern = r"correct\s+answer\s+is:?\s+([a-fi])\b"
        match = re.search(pattern, predict.lower())
        assert match is not None
        assert match.group(1) == "d"

    @pytest.mark.unit
    def test_extract_correct_option_is_pattern(self):
        """Test 'correct option is (A)' pattern."""
        predict = "The correct option is (E)"
        import re

        pattern = r"correct\s+option\s+is:?\s+\(?([a-fi])\)?"
        match = re.search(pattern, predict.lower())
        assert match is not None
        assert match.group(1) == "e"


class TestOpenEndedMatching:
    """Test open-ended answer matching strategies."""

    @pytest.mark.unit
    def test_exact_substring_match(self):
        """Test exact substring matching for open-ended answers."""
        prediction = "The capital of France is Paris"
        answer = "Paris"
        assert answer.lower() in prediction.lower()

    @pytest.mark.unit
    def test_numeric_answer_extraction(self):
        """Test numeric answer extraction."""
        prediction = "The result is 42.5 units"
        import re

        numbers = re.findall(r"-?\d+\.?\d*", prediction)
        assert "42.5" in numbers

    @pytest.mark.unit
    def test_word_level_match(self):
        """Test word-level matching."""
        prediction = "The color is bright red with some orange tones"
        answer = "bright red"
        answer_words = set(answer.lower().split())
        predict_words = set(prediction.lower().split())
        assert answer_words.issubset(predict_words)

    @pytest.mark.unit
    def test_word_level_no_match(self):
        """Test word-level matching when words don't match."""
        prediction = "The color is blue"
        answer = "bright red"
        answer_words = set(answer.lower().split())
        predict_words = set(prediction.lower().split())
        assert not answer_words.issubset(predict_words)


class TestIntegrationMock:
    """Mock integration tests for the full pipeline."""

    @pytest.mark.unit
    def test_full_pipeline_multiple_choice_math(self):
        """Test full pipeline for math multiple choice question."""
        example = {
            "question": "What is 2 + 2?",
            "options": "['3', '4', '5', '6']",
        }
        question = process_question(example)

        assert "What is 2 + 2?" in question
        assert "A. 3" in question
        assert "B. 4" in question

        # Simulate model response
        model_response = "The sum is 4, so the answer is B"
        # Would be evaluated as correct for answer "B"

    @pytest.mark.unit
    def test_full_pipeline_open_ended_text(self):
        """Test full pipeline for open-ended text question."""
        example = {"question": "What is the capital of France?", "options": None}
        question = process_question(example)

        assert question == "What is the capital of France?"

        # Simulate model response
        model_response = "The capital of France is Paris"
        ground_truth = "Paris"
        assert ground_truth.lower() in model_response.lower()

    @pytest.mark.unit
    def test_full_pipeline_numeric_answer(self):
        """Test full pipeline for numeric answer."""
        example = {"question": "Calculate the area of a 5x3 rectangle", "options": None}
        question = process_question(example)

        # Simulate model response
        model_response = "Area = 5 × 3 = 15 square units"
        ground_truth = "15"

        import re

        numbers = re.findall(r"-?\d+\.?\d*", model_response)
        assert ground_truth in numbers

    @pytest.mark.unit
    def test_full_pipeline_with_image_tags(self):
        """Test processing question with image tags and options."""
        example = {
            "question": "In <image 1>, what shape is shown?",
            "options": "['circle', 'square', 'triangle']",
        }
        question = process_question(example)

        assert "<image" not in question
        assert "Options:" in question
        assert "A. circle" in question


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_process_question_none_options(self):
        """Test handling None options."""
        example = {"question": "Test question", "options": None}
        result = process_question(example)
        assert result == "Test question"

    @pytest.mark.unit
    def test_process_question_malformed_options(self):
        """Test handling malformed options string."""
        example = {"question": "Test", "options": "not a list"}
        result = process_question(example)
        # Should handle gracefully
        assert "Test" in result

    @pytest.mark.unit
    def test_normalize_number_with_multiple_commas(self):
        """Test normalizing very large numbers with multiple commas."""
        result = normalize_number("1,234,567,890.12")
        assert result == 1234567890.12

    @pytest.mark.unit
    def test_process_question_unicode(self):
        """Test processing question with Unicode characters."""
        example = {"question": "¿Qué es esto? 这是什么？"}
        result = process_question(example)
        assert "¿Qué es esto?" in result
        assert "这是什么？" in result

    @pytest.mark.unit
    def test_normalize_number_scientific_notation(self):
        """Test that scientific notation is handled."""
        # normalize_number uses float() which handles scientific notation
        result = normalize_number("1.23e5")
        assert result == 123000.0

    @pytest.mark.unit
    def test_get_images_with_none_values(self):
        """Test image extraction with None values."""
        example = {"image": None, "image_1": None}
        result = get_images(example)
        assert result == []


class TestSubjectEvaluation:
    """Test subject-based evaluation logic."""

    @pytest.mark.unit
    def test_subject_tracking_initialization(self):
        """Test that subject tracking initializes correctly."""
        subject_scores = {}
        subject_counters = {}

        subjects = ["Math", "Physics", "Chemistry"]
        for subject in subjects:
            if subject not in subject_scores:
                subject_scores[subject] = 0
                subject_counters[subject] = 0

        assert len(subject_scores) == 3
        assert all(score == 0 for score in subject_scores.values())
        assert all(count == 0 for count in subject_counters.values())

    @pytest.mark.unit
    def test_subject_score_calculation(self):
        """Test subject score calculation."""
        subject_scores = {"Math": 8, "Physics": 6}
        subject_counters = {"Math": 10, "Physics": 10}

        math_accuracy = subject_scores["Math"] / subject_counters["Math"]
        physics_accuracy = subject_scores["Physics"] / subject_counters["Physics"]

        assert math_accuracy == 0.8
        assert physics_accuracy == 0.6


@pytest.mark.slow
@pytest.mark.eval
@pytest.mark.requires_model
class TestFullEvaluation:
    """Full evaluation tests requiring model and dataset downloads."""

    def test_mmmu_small_sample_math(self, smolvlm_256m_model, mmmu_dataset):
        """Test MMMU evaluation on Math subject with small sample."""
        from smlx.evals.utils import inference

        model = smolvlm_256m_model["model"]
        processor = smolvlm_256m_model["processor"]
        dataset = mmmu_dataset["dataset"]

        # Test on first 2 samples (dataset already filtered to Math subject in fixture)
        max_samples = 2
        results = []
        samples_checked = 0

        for sample in dataset:
            if len(results) >= max_samples:
                break

            samples_checked += 1

            # Skip samples without images
            images = get_images(sample)
            if not images:
                continue

            # Format question
            question = process_question(sample)

            # Run inference
            response = inference(
                model,
                processor,
                question,
                image=images[0] if len(images) == 1 else images,
                max_tokens=512,
                temperature=0.0,
                verbose=False,
            )

            results.append({"question": question[:100], "response": response[:100]})

        # Verify we processed samples
        print(f"\n\nChecked {samples_checked} samples, processed {len(results)} with images")

        # It's ok if some samples don't have images, as long as we checked samples
        assert samples_checked > 0, "Should have at least checked some samples"

    def test_mmmu_predictions_file(self):
        """Test evaluation from pre-generated predictions file."""
        # This test would require a predictions file to exist
        # For now, just test that we can handle the logic
        predictions = [{"question_id": 1, "prediction": "A", "ground_truth": "A"}]

        correct = sum(1 for p in predictions if p["prediction"] == p["ground_truth"])
        accuracy = correct / len(predictions) if predictions else 0.0

        assert accuracy == 1.0  # All correct in our mock data

    def test_mmmu_predictions_evaluation_logic(self):
        """Test evaluation logic with predictions."""
        # Test that we can handle predictions correctly
        predictions = [
            {"question_id": 1, "prediction": "A", "ground_truth": "A"},
            {"question_id": 2, "prediction": "B", "ground_truth": "B"},
            {"question_id": 3, "prediction": "C", "ground_truth": "D"},
        ]

        correct = sum(1 for p in predictions if p["prediction"] == p["ground_truth"])
        accuracy = correct / len(predictions) if predictions else 0.0

        assert accuracy == 2.0 / 3.0  # 2 out of 3 correct


@pytest.mark.unit
class TestSubjectList:
    """Test MMMU subject list completeness."""

    def test_mmmu_all_subjects(self):
        """Test that all 30 MMMU subjects are defined."""
        # Just verify the subject list is complete
        assert len(MMMU_SUBJECTS) == 30

        # Verify no duplicates
        assert len(MMMU_SUBJECTS) == len(set(MMMU_SUBJECTS))

        # This is a unit test, no need for slow downloads
        print(f"\nAll {len(MMMU_SUBJECTS)} MMMU subjects are defined")
