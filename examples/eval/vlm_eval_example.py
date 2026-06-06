#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Vision-Language Model Evaluation Example

Demonstrates how to evaluate VLMs on multimodal benchmarks:
- Math-Vista: Math reasoning with vision
- MMMU: Multimodal understanding
- MMStar: Multimodal reasoning
- OCRBench: OCR capabilities
"""

from smlx.evals.math_vista import process_question, normalize_answer
from smlx.evals.mmmu import process_question as mmmu_process_question, normalize_number, MMMU_SUBJECTS
from smlx.evals.mmstar import extract_answer, CATEGORIES
from smlx.evals.ocrbench import evaluate_answer as ocrbench_evaluate_answer, normalize_answer as ocrbench_normalize_answer


def demo_math_vista():
    """Demonstrate Math-Vista evaluation utilities."""
    print("\n" + "=" * 70)
    print("Math-Vista Evaluation")
    print("=" * 70)

    # Example question (multiple choice)
    question = {
        "question": "What is 2 + 2?",
        "choices": ["A. 3", "B. 4", "C. 5", "D. 6"],
        "answer": "B",
    }

    # Process question
    formatted = process_question(question)
    print(f"\nFormatted question:\n{formatted}")

    # Simulate model predictions
    predictions = [
        "The answer is B. 4",
        "B",
        "\\boxed{B}",
        "I think the correct answer is (B) 4",
    ]

    print("\nAnswer normalization:")
    for pred in predictions:
        normalized = normalize_answer(pred, question["choices"])
        print(f"  '{pred}' → '{normalized}'")

    print("\n✅ Math-Vista handles multiple answer formats!")


def demo_mmmu():
    """Demonstrate MMMU evaluation utilities."""
    print("\n" + "=" * 70)
    print("MMMU Evaluation")
    print("=" * 70)

    print(f"\nMMMU covers {len(MMMU_SUBJECTS)} subjects:")
    for subject in MMMU_SUBJECTS[:5]:
        print(f"  - {subject}")
    print(f"  ... and {len(MMMU_SUBJECTS) - 5} more")

    # Example: number normalization
    numbers = ["1,234", "1234.56", "-42", "1.5e3"]

    print("\nNumber normalization:")
    for num in numbers:
        normalized = normalize_number(num)
        print(f"  '{num}' → '{normalized}'")


def demo_mmstar():
    """Demonstrate MMStar evaluation utilities."""
    print("\n" + "=" * 70)
    print("MMStar Evaluation")
    print("=" * 70)

    print("\nMMStar evaluates across 6 categories:")
    for category, subcats in CATEGORIES.items():
        print(f"  {category}:")
        for subcat in subcats:
            print(f"    - {subcat}")

    # Example: answer extraction
    responses = [
        "Therefore, the answer is C.",
        "The correct option is (B)",
        "Answer: D",
        "I would select choice A because...",
    ]

    print("\nAnswer extraction:")
    for response in responses:
        extracted = extract_answer(response, num_options=4)
        print(f"  '{response[:40]}...' → '{extracted}'")


def demo_ocrbench():
    """Demonstrate OCRBench evaluation utilities."""
    print("\n" + "=" * 70)
    print("OCRBench Evaluation")
    print("=" * 70)

    # Ground truths (semicolon-separated alternatives)
    ground_truths = "hello;Hello;HELLO"

    predictions = [
        "Hello world",
        "HELLO",
        "hi there",
        "The text says hello",
    ]

    print("\nOCR answer matching:")
    for pred in predictions:
        result = ocrbench_evaluate_answer(pred, ground_truths)
        print(f"  '{pred}' → {'✓ Match' if result else '✗ No match'}")


def main():
    print("=" * 70)
    print("Vision-Language Model Evaluation Examples")
    print("=" * 70)

    print("\nThis example demonstrates VLM evaluation utilities.")
    print("Full VLM evaluation requires vision models (SmolVLM, etc.)")

    # Demo each benchmark
    demo_math_vista()
    demo_mmmu()
    demo_mmstar()
    demo_ocrbench()

    print("\n" + "=" * 70)
    print("VLM Evaluation Utilities Overview")
    print("=" * 70)

    print("\n📊 Available Benchmarks:")
    print("  1. Math-Vista: 6,141 examples, 28 datasets")
    print("     - Multiple choice and free-form math problems")
    print("     - Requires vision + language reasoning")

    print("\n  2. MMMU: Massive multimodal understanding")
    print("     - 30+ subjects (STEM, humanities, business)")
    print("     - Expert-level questions")

    print("\n  3. MMStar: Multimodal reasoning")
    print("     - 6 main categories, 18 subcategories")
    print("     - Coarse and fine-grained perception")

    print("\n  4. OCRBench: OCR capabilities")
    print("     - Text recognition (regular & irregular)")
    print("     - Multiple acceptable answers")

    print("\n💡 Usage with VLMs:")
    print("   When SmolVLM is implemented, you can use these evaluation")
    print("   utilities to assess VLM performance on multimodal benchmarks.")
    print("   >>> from smlx.models.SmolVLM_256M import load")
    print("   >>> from smlx.evals.math_vista import process_question")
    print("   >>> model, processor = load('SmolVLM-256M')")
    print("   >>> # Use process_question() to format prompts for evaluation")

    print("\n" + "=" * 70)
    print("✅ VLM Eval Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
