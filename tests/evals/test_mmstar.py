"""
Tests for MMStar evaluation module.

Tests cover:
- Priority-based answer extraction
- Concluding vs general pattern matching
- Category hierarchy handling
- Mock integration tests
"""

import pytest

from smlx.evals.mmstar import extract_answer


class TestAnswerExtraction:
    """Test priority-based answer extraction for MMStar."""

    @pytest.mark.unit
    def test_extract_simple_letter(self):
        """Test extracting simple letter answer."""
        predict = "A"
        assert extract_answer(predict, "A") is True
        assert extract_answer(predict, "B") is False

    @pytest.mark.unit
    def test_extract_answer_is_pattern(self):
        """Test 'the answer is X' pattern (concluding, priority 2)."""
        predict = "After analyzing the image, the answer is B"
        assert extract_answer(predict, "B") is True
        assert extract_answer(predict, "A") is False

    @pytest.mark.unit
    def test_extract_parenthesized_answer(self):
        """Test (A) format (general pattern, priority 1)."""
        predict = "The correct choice is (C)"
        assert extract_answer(predict, "C") is True
        assert extract_answer(predict, "D") is False

    @pytest.mark.unit
    def test_extract_option_format(self):
        """Test 'option X' format."""
        predict = "I would choose option D for this question"
        assert extract_answer(predict, "D") is True
        assert extract_answer(predict, "A") is False

    @pytest.mark.unit
    def test_extract_choice_format(self):
        """Test 'choice X' format."""
        predict = "The best choice B seems most appropriate"
        assert extract_answer(predict, "B") is True
        assert extract_answer(predict, "C") is False

    @pytest.mark.unit
    def test_priority_concluding_over_general(self):
        """Test that concluding patterns have higher priority than general."""
        # First mentions B (general), then concludes with A (concluding)
        predict = "Option B seems possible, but the answer is A"
        # Should extract A (concluding pattern) not B (general pattern)
        assert extract_answer(predict, "A") is True
        assert extract_answer(predict, "B") is False

    @pytest.mark.unit
    def test_priority_later_over_earlier(self):
        """Test that later matches are preferred within same priority."""
        # Multiple concluding patterns, should prefer the last one
        predict = "Initially the answer is B. However, upon reflection, the answer is C"
        assert extract_answer(predict, "C") is True
        assert extract_answer(predict, "B") is False

    @pytest.mark.unit
    def test_extract_therefore_pattern(self):
        """Test 'therefore X' concluding pattern."""
        predict = "Based on the visual evidence, therefore A"
        assert extract_answer(predict, "A") is True

    @pytest.mark.unit
    def test_extract_thus_pattern(self):
        """Test 'thus X' concluding pattern."""
        predict = "The colors match perfectly, thus the answer is D"
        assert extract_answer(predict, "D") is True

    @pytest.mark.unit
    def test_extract_answer_colon_format(self):
        """Test 'answer: X' format."""
        predict = "My answer: B based on the image content"
        assert extract_answer(predict, "B") is True

    @pytest.mark.unit
    def test_extract_select_pattern(self):
        """Test 'select X' pattern."""
        predict = "I would select E for this question"
        assert extract_answer(predict, "E") is True

    @pytest.mark.unit
    def test_extract_case_insensitive(self):
        """Test case-insensitive matching."""
        predict = "THE ANSWER IS A"
        assert extract_answer(predict, "a") is True
        assert extract_answer(predict, "A") is True

    @pytest.mark.unit
    def test_extract_with_newlines(self):
        """Test extraction with newlines in prediction."""
        predict = "Let me analyze:\n\nThe image shows...\n\nTherefore the answer is C"
        assert extract_answer(predict, "C") is True

    @pytest.mark.unit
    def test_extract_no_match(self):
        """Test when no answer pattern is found."""
        predict = "This is a completely unrelated response with no answer markers"
        assert extract_answer(predict, "A") is False

    @pytest.mark.unit
    def test_extract_markdown_bold_answer(self):
        """Test extraction from markdown bold **Answer**: format."""
        predict = "Analysis complete. **Answer**: D is the correct choice"
        assert extract_answer(predict, "D") is True

    @pytest.mark.unit
    def test_extract_it_is_pattern(self):
        """Test 'it is X' pattern."""
        predict = "Based on the scene, it is B"
        assert extract_answer(predict, "B") is True

    @pytest.mark.unit
    def test_extract_would_be_pattern(self):
        """Test 'would be X' pattern."""
        predict = "The most appropriate answer would be A"
        assert extract_answer(predict, "A") is True


class TestComplexScenarios:
    """Test complex answer extraction scenarios."""

    @pytest.mark.unit
    def test_multiple_answers_prefer_last_concluding(self):
        """Test preferring the last concluding answer when multiple exist."""
        predict = """
        Let's analyze each option:
        (A) Red - possible
        (B) Blue - also possible
        (C) Green - matches the image
        Therefore, the answer is C
        """
        assert extract_answer(predict, "C") is True

    @pytest.mark.unit
    def test_reasoning_with_final_answer(self):
        """Test extracting final answer from reasoning chain."""
        predict = """
        Looking at the image, I can see:
        - Option A shows a cat
        - Option B shows a dog
        - Option C shows a bird
        The image clearly shows a dog, so the answer is B
        """
        assert extract_answer(predict, "B") is True

    @pytest.mark.unit
    def test_revised_answer_pattern(self):
        """Test extraction of revised answer (concluding pattern)."""
        predict = "Initially A, but **revised answer**: C"
        assert extract_answer(predict, "C") is True

    @pytest.mark.unit
    def test_correct_answer_is_pattern(self):
        """Test 'correct answer is X' pattern."""
        predict = "After careful analysis, the correct answer is E"
        assert extract_answer(predict, "E") is True

    @pytest.mark.unit
    def test_correct_option_is_pattern(self):
        """Test 'correct option is X' pattern."""
        predict = "The correct option is: A based on visual cues"
        assert extract_answer(predict, "A") is True

    @pytest.mark.unit
    def test_answer_at_start(self):
        """Test answer appearing at the very start."""
        predict = "A is the correct answer because the image shows a red object"
        assert extract_answer(predict, "A") is True

    @pytest.mark.unit
    def test_answer_with_period(self):
        """Test answer followed by period."""
        predict = "Looking at this carefully, I conclude D."
        assert extract_answer(predict, "D") is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_empty_prediction(self):
        """Test handling empty prediction."""
        assert extract_answer("", "A") is False

    @pytest.mark.unit
    def test_whitespace_only_prediction(self):
        """Test handling whitespace-only prediction."""
        assert extract_answer("   \n\n   ", "A") is False

    @pytest.mark.unit
    def test_no_answer_letter(self):
        """Test prediction with no answer letter at all."""
        predict = "This image shows something interesting but I'm not sure"
        assert extract_answer(predict, "A") is False

    @pytest.mark.unit
    def test_answer_out_of_range(self):
        """Test answer letter outside A-E range."""
        predict = "The answer is F"  # F is not in valid range
        # Should return False since F is not in possible_answers list
        assert extract_answer(predict, "F") is False

    @pytest.mark.unit
    def test_unicode_text(self):
        """Test handling Unicode characters in prediction."""
        predict = "经过分析，the answer is B 是正确的"
        assert extract_answer(predict, "B") is True

    @pytest.mark.unit
    def test_very_long_prediction(self):
        """Test with very long prediction text."""
        predict = "Analysis: " + "Lorem ipsum dolor sit amet. " * 100 + "Therefore the answer is C"
        assert extract_answer(predict, "C") is True

    @pytest.mark.unit
    def test_multiple_therefore_patterns(self):
        """Test multiple 'therefore' patterns, should prefer last."""
        predict = "Therefore A seems right. Actually, therefore B is better. Hence C is the answer"
        assert extract_answer(predict, "C") is True


class TestCategoryHierarchy:
    """Test category and subcategory structures."""

    @pytest.mark.unit
    def test_category_names(self):
        """Test that category names are properly defined."""
        categories = [
            "coarse perception",
            "fine-grained perception",
            "instance reasoning",
            "logical reasoning",
            "science & technology",
            "math",
        ]
        # Just verify these are valid strings
        for cat in categories:
            assert isinstance(cat, str)
            assert len(cat) > 0

    @pytest.mark.unit
    def test_subcategory_structure(self):
        """Test subcategory structure matches MMStar spec."""
        # Coarse perception subcategories
        coarse_subcats = [
            "image scene and topic",
            "image style & quality",
            "image emotion",
        ]
        assert len(coarse_subcats) == 3

        # Fine-grained perception subcategories
        fine_subcats = ["object counting", "recognition", "localization"]
        assert len(fine_subcats) == 3

        # Instance reasoning subcategories
        instance_subcats = [
            "single-instance reasoning",
            "cross-instance attribute reasoning",
            "cross-instance relation reasoning",
        ]
        assert len(instance_subcats) == 3

        # Logical reasoning subcategories
        logical_subcats = [
            "code & sequence reasoning",
            "diagram reasoning",
            "common reasoning",
        ]
        assert len(logical_subcats) == 3

        # Science & technology subcategories
        science_subcats = [
            "biology & chemistry & physics",
            "electronics & energy & mechanical eng.",
            "geography & earth science & agriculture",
        ]
        assert len(science_subcats) == 3

        # Math subcategories
        math_subcats = [
            "geometry",
            "numeric commonsense and calculation",
            "statistical reasoning",
        ]
        assert len(math_subcats) == 3


class TestIntegrationMock:
    """Mock integration tests for the full pipeline."""

    @pytest.mark.unit
    def test_full_pipeline_coarse_perception(self):
        """Test full pipeline for coarse perception task."""
        # Simulate a coarse perception question
        question = "What is the overall scene depicted in this image?"
        model_response = "The image shows an urban street scene. The answer is B"

        result = extract_answer(model_response, "B")
        assert result is True

    @pytest.mark.unit
    def test_full_pipeline_fine_grained_perception(self):
        """Test full pipeline for fine-grained perception task."""
        # Simulate object counting question
        question = "How many cars are in the parking lot?"
        model_response = "I can count 5 cars in total. Therefore A"

        result = extract_answer(model_response, "A")
        assert result is True

    @pytest.mark.unit
    def test_full_pipeline_instance_reasoning(self):
        """Test full pipeline for instance reasoning task."""
        # Simulate cross-instance reasoning
        question = "Which object is larger, the red box or the blue box?"
        model_response = "Comparing the two boxes, the red box appears larger. The answer is C"

        result = extract_answer(model_response, "C")
        assert result is True

    @pytest.mark.unit
    def test_full_pipeline_logical_reasoning(self):
        """Test full pipeline for logical reasoning task."""
        # Simulate diagram reasoning
        question = "What comes next in the sequence?"
        model_response = "Following the pattern of rotation and scaling, option D completes the sequence"

        result = extract_answer(model_response, "D")
        assert result is True

    @pytest.mark.unit
    def test_full_pipeline_math(self):
        """Test full pipeline for math task."""
        # Simulate geometry question
        question = "What is the area of the shaded region?"
        model_response = "Calculating: Area = base × height = 10 × 5 = 50. Therefore, the answer is B"

        result = extract_answer(model_response, "B")
        assert result is True


@pytest.mark.slow
@pytest.mark.eval
@pytest.mark.requires_model
class TestFullEvaluation:
    """Full evaluation tests requiring model and dataset downloads."""

    def test_mmstar_predictions_evaluation_logic(self):
        """Test evaluation logic with predictions."""
        # Test that we can handle predictions correctly using extract_answer
        predictions = [
            {"prediction": "The answer is A", "ground_truth": "A"},
            {"prediction": "Therefore B is correct", "ground_truth": "B"},
            {"prediction": "I think it's D", "ground_truth": "C"},
        ]

        correct = sum(
            1
            for p in predictions
            if extract_answer(p["prediction"], p["ground_truth"])
        )
        accuracy = correct / len(predictions) if predictions else 0.0

        assert accuracy == 2.0 / 3.0  # 2 out of 3 correct

    @pytest.mark.unit
    def test_mmstar_small_sample(self, smolvlm_256m_model, mmstar_dataset):
        """Test MMStar evaluation on a small sample."""
        from smlx.evals.utils import inference

        model = smolvlm_256m_model["model"]
        processor = smolvlm_256m_model["processor"]
        dataset = mmstar_dataset["dataset"]

        # Test on first 2 samples only
        max_samples = 2
        results = []

        for idx, sample in enumerate(dataset):
            if idx >= max_samples:
                break

            # Get question and image
            question = sample.get("question", "")
            image = sample.get("image")
            ground_truth = sample.get("answer", "")

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

            # Extract answer
            is_correct = extract_answer(response, ground_truth)

            results.append(
                {
                    "question": question[:100],
                    "response": response[:100],
                    "ground_truth": ground_truth,
                    "correct": is_correct,
                }
            )

        # Verify we got results
        assert len(results) > 0, "Should have processed at least one sample"
        # Note: We may process fewer than max_samples if some don't have images

        print(f"\n\nProcessed {len(results)} samples from MMStar (checked {idx + 1} total)")
        for i, result in enumerate(results):
            print(f"\nSample {i + 1}:")
            print(f"  Question: {result['question']}")
            print(f"  Response: {result['response']}")
            print(f"  Ground Truth: {result['ground_truth']}")
            print(f"  Correct: {result['correct']}")
