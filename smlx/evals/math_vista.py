"""
MathVista evaluation benchmark for SMLX.

MathVista is a benchmark combining mathematical and visual reasoning tasks,
consisting of 6,141 examples derived from 28 existing multimodal datasets.
This implementation supports evaluation of vision-language models on Apple Silicon using MLX.

Dataset: AI4Math/MathVista
Splits: testmini (1,000 samples with ground truth), test (5,141 samples)
Tasks: Multiple choice, free-form questions with integer/float answers
Metrics: Overall accuracy, category-wise performance breakdown

Example usage:
    python -m smlx.evals.math_vista \\
        --model mlx-community/SmolVLM-256M-Instruct \\
        --split testmini \\
        --max-samples 10
"""

import argparse
import csv
import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from smlx.evals.utils import inference, load_model


def process_question(sample: dict) -> str:
    """
    Format the question with choices if it's multiple choice.

    Args:
        sample: Dataset sample containing 'query', 'question_type', and 'choices'

    Returns:
        str: Formatted question with choices (A), (B), (C), etc. for multiple choice

    Example:
        >>> sample = {
        ...     "query": "What is 2+2?",
        ...     "question_type": "multi_choice",
        ...     "choices": ["3", "4", "5"]
        ... }
        >>> process_question(sample)
        'What is 2+2?\\n(A) 3\\n(B) 4\\n(C) 5'
    """
    question = sample["query"]

    if sample["question_type"] == "multi_choice" and sample.get("choices"):
        choices_text = "\n".join(
            [f"({chr(65+i)}) {choice}" for i, choice in enumerate(sample["choices"])]
        )
        question = f"{question}\n{choices_text}"

    return question


def normalize_answer(response: str, problem: dict) -> Optional[str]:
    """
    Normalize the model's response to extract the answer using regex patterns.

    This function handles various answer formats including:
    - Boxed answers: \\boxed{A} or \\boxed{42}
    - Chinese patterns: "E	A"
    - English patterns: "the answer is A", "answer: A"
    - Numeric extraction with scientific notation
    - Fuzzy matching for multiple choice

    Args:
        response: Raw model output text
        problem: Dataset sample with 'question_type', 'answer_type', 'choices', etc.

    Returns:
        Optional[str]: Normalized answer or None if extraction failed
    """
    response = response.strip()

    if not response:
        return None

    question_type = problem["question_type"]
    answer_type = problem["answer_type"]
    choices = problem.get("choices", [])

    # === MULTIPLE CHOICE EXTRACTION ===
    if question_type == "multi_choice":
        # First, try to find boxed answers
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed_match:
            boxed_content = boxed_match.group(1)
            # Check if it's a choice letter
            letter_match = re.match(r"^\(?([A-Z])\)?\.?$", boxed_content.strip().upper())
            if letter_match:
                letter = letter_match.group(1)
                idx = ord(letter) - ord("A")
                if 0 <= idx < len(choices):
                    return choices[idx]
            # Check if it's directly one of the choices
            if boxed_content.strip() in choices:
                return boxed_content.strip()

        # Try to find Chinese answer pattern "E	X" or "E	X"
        chinese_match = re.search(r"E	[:]\s*([A-Z])", response.upper())
        if not chinese_match:
            chinese_match = re.search(r"E	\s*([A-Z])", response.upper())
        if chinese_match:
            letter = chinese_match.group(1)
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(choices):
                return choices[idx]

        # Try to find "the answer is X" or "answer: X" patterns near the end
        answer_patterns = [
            r"(?:the\s+)?answer\s+is\s+\(?([A-Z])\)?",
            r"answer:\s*\(?([A-Z])\)?",
            r"choose\s+\(?([A-Z])\)?",
            r"option\s+\(?([A-Z])\)?",
        ]

        # Search from the end of the response (last 500 chars)
        end_section = response[-500:] if len(response) > 500 else response
        for pattern in answer_patterns:
            matches = list(re.finditer(pattern, end_section, re.IGNORECASE))
            if matches:
                # Take the last match
                letter = matches[-1].group(1).upper()
                idx = ord(letter) - ord("A")
                if 0 <= idx < len(choices):
                    return choices[idx]

        # Look for patterns like "(A)", "A)", "A.", "A" - prioritize from the end
        matches = list(re.finditer(r"\(?([A-Z])\)?\.?", response.upper()))
        if matches:
            # Try the last few matches first
            for match in reversed(matches[-5:]):
                letter = match.group(1)
                idx = ord(letter) - ord("A")
                if 0 <= idx < len(choices):
                    return choices[idx]

        # If the response is exactly one of the choices
        if response in choices:
            return response

        # Try to find the most similar choice using edit distance (Levenshtein)
        def edit_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        if choices:
            distances = [edit_distance(response.lower(), choice.lower()) for choice in choices]
            return choices[distances.index(min(distances))]

    # === INTEGER EXTRACTION ===
    elif answer_type == "integer":
        # First try to find boxed answer
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed_match:
            boxed_content = boxed_match.group(1)
            # Remove commas from numbers
            boxed_content = boxed_content.replace(",", "")
            # Try scientific notation first
            sci_numbers = re.findall(r"-?\d+\.?\d*[eE][+-]?\d+", boxed_content)
            if sci_numbers:
                try:
                    return str(int(float(sci_numbers[0])))
                except (ValueError, OverflowError):
                    pass
            # Then regular numbers
            numbers = re.findall(r"-?\d+", boxed_content)
            if numbers:
                try:
                    return str(int(numbers[0]))
                except (ValueError, OverflowError):
                    pass

        # Try common answer patterns near the end
        end_section = response[-500:] if len(response) > 500 else response
        answer_patterns = [
            r"(?:the\s+)?answer\s+is\s+(-?[\d,]+\.?\d*[eE][+-]?\d+|-?[\d,]+)",
            r"answer:\s*(-?[\d,]+\.?\d*[eE][+-]?\d+|-?[\d,]+)",
            r"(?:total|result|left|remaining)(?:\s+is|\s+are|:)\s*(-?[\d,]+\.?\d*[eE][+-]?\d+|-?[\d,]+)",
        ]

        for pattern in answer_patterns:
            matches = list(re.finditer(pattern, end_section, re.IGNORECASE))
            if matches:
                try:
                    # Remove commas before converting
                    num_str = matches[-1].group(1).replace(",", "")
                    return str(int(float(num_str)))
                except (ValueError, OverflowError):
                    pass

        # Look for scientific notation anywhere in response
        sci_numbers = re.findall(r"-?\d+\.?\d*[eE][+-]?\d+", response)
        if sci_numbers:
            try:
                return str(int(float(sci_numbers[-1])))
            except (ValueError, OverflowError):
                pass

        # Fall back to finding all numbers (including comma-formatted) and taking the last one
        numbers = re.findall(r"-?[\d,]+", response)
        if numbers:
            try:
                # Remove commas and try the last number first
                return str(int(numbers[-1].replace(",", "")))
            except (ValueError, OverflowError):
                pass

    # === FLOAT EXTRACTION ===
    elif answer_type == "float":
        precision = int(problem.get("precision", 2))

        # First try to find boxed answer
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed_match:
            boxed_content = boxed_match.group(1)
            # Try scientific notation first
            sci_numbers = re.findall(r"-?\d+\.?\d*[eE][+-]?\d+", boxed_content)
            if sci_numbers:
                try:
                    return str(round(float(sci_numbers[0]), precision))
                except (ValueError, OverflowError):
                    pass
            # Then regular numbers
            numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
            if numbers:
                try:
                    return str(round(float(numbers[0]), precision))
                except (ValueError, OverflowError):
                    pass

        # Try common answer patterns near the end
        end_section = response[-500:] if len(response) > 500 else response
        answer_patterns = [
            r"(?:the\s+)?answer\s+is\s+(-?\d+\.?\d*[eE][+-]?\d+|-?\d+\.?\d*)",
            r"answer:\s*(-?\d+\.?\d*[eE][+-]?\d+|-?\d+\.?\d*)",
            r"d\s*=\s*(-?\d+\.?\d*[eE][+-]?\d+|-?\d+\.?\d*)",  # For physics problems
        ]

        for pattern in answer_patterns:
            matches = list(re.finditer(pattern, end_section, re.IGNORECASE))
            if matches:
                try:
                    return str(round(float(matches[-1].group(1)), precision))
                except (ValueError, OverflowError):
                    pass

        # Look for scientific notation anywhere in response
        sci_numbers = re.findall(r"-?\d+\.?\d*[eE][+-]?\d+", response)
        if sci_numbers:
            try:
                return str(round(float(sci_numbers[-1]), precision))
            except (ValueError, OverflowError):
                pass

        # Fall back to finding all numbers and taking the last one
        numbers = re.findall(r"-?\d+\.?\d*", response)
        if numbers:
            try:
                return str(round(float(numbers[-1]), precision))
            except (ValueError, OverflowError):
                pass

    return response


def evaluate_answer(prediction: Optional[str], ground_truth: str) -> bool:
    """
    Check if the prediction matches the ground truth.

    Handles exact string matching and numeric word representations (e.g., "one" == "1").

    Args:
        prediction: The normalized prediction from the model
        ground_truth: The ground truth answer from the dataset

    Returns:
        bool: True if prediction matches ground truth
    """
    if prediction is None:
        return False

    try:
        # First check exact match
        if str(prediction).strip() == str(ground_truth).strip():
            return True

        # Handle numeric word representations
        word_to_num = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
        }

        pred_normalized = str(prediction).strip().lower()
        gt_normalized = str(ground_truth).strip().lower()

        # Convert words to numbers
        if pred_normalized in word_to_num:
            pred_normalized = word_to_num[pred_normalized]
        if gt_normalized in word_to_num:
            gt_normalized = word_to_num[gt_normalized]

        return pred_normalized == gt_normalized
    except Exception:
        return False


def parse_args():
    """Parse command-line arguments for MathVista evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate vision-language models on MathVista benchmark (SMLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on testmini split with ground truth
  python -m smlx.evals.math_vista \\
      --model mlx-community/SmolVLM-256M-Instruct \\
      --split testmini

  # Quick test with 10 samples
  python -m smlx.evals.math_vista \\
      --model mlx-community/Moondream2-1.8B \\
      --split testmini \\
      --max-samples 10 \\
      --verbose

  # Full test split evaluation (no ground truth)
  python -m smlx.evals.math_vista \\
      --model mlx-community/SmolVLM-500M \\
      --split test \\
      --output-dir results/mathvista
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (e.g., mlx-community/SmolVLM-256M-Instruct)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional path for LoRA/adapter weights",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="AI4Math/MathVista",
        help="HuggingFace dataset name (default: AI4Math/MathVista)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="testmini",
        choices=["testmini", "test"],
        help="Dataset split: testmini (1k samples, has ground truth) or test (5k samples)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming dataset loading (for large datasets)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (useful for debugging)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mathvista",
        help="Directory to save CSV and JSON results",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic/greedy decoding)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    """Main evaluation loop for MathVista benchmark."""
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load model
    logging.info(f"Loading model from {args.model}")
    try:
        model, processor = load_model(
            args.model, adapter_path=args.adapter_path, trust_remote_code=True
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install evaluation dependencies:")
        print("  pip install 'smlx[evals]'")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load dataset
    logging.info(f"Loading dataset {args.dataset}, split {args.split}")
    try:
        from datasets import Dataset
    except ImportError:
        print("Error: datasets library not found.")
        print("Install with: pip install 'smlx[evals]'")
        return

    from smlx.evals.utils import resolve_eval_dataset

    dataset, _ = resolve_eval_dataset(
        "mathvista",
        args.dataset,
        args.split,
        prefer_local=(args.dataset == "AI4Math/MathVista"),
        streaming=args.streaming,
    )

    if args.max_samples and not args.streaming:
        # Type guard: when streaming=False and split is specified, we get a Dataset
        if isinstance(dataset, Dataset):
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tracking variables
    results = {}
    category_scores = {}
    correct = 0
    total = 0

    # Evaluate each sample
    sample_iterator = enumerate(dataset)
    if not args.streaming:
        sample_iterator = enumerate(tqdm(dataset, desc="Evaluating MathVista"))

    for idx, sample in sample_iterator:
        if args.streaming and args.max_samples and idx >= args.max_samples:
            break

        pid = sample.get("pid", f"sample_{idx}")

        try:
            # Load and process image
            image = None
            if "decoded_image" in sample and sample["decoded_image"]:
                if isinstance(sample["decoded_image"], str):
                    # Image path provided
                    image_path = sample["decoded_image"]
                    try:
                        image = Image.open(image_path).convert("RGB")
                    except Exception as e:
                        logging.warning(f"Cannot load image {image_path}: {e}, skipping {pid}")
                        continue
                else:
                    # Image already loaded (PIL Image)
                    image = sample["decoded_image"].convert("RGB")
            else:
                logging.warning(f"No image for sample {pid}, skipping")
                continue

            # Create prompt
            prompt = process_question(sample)

            # Generate response
            output = inference(
                model,
                processor,
                prompt,
                image=image,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=args.verbose,
            )

            response = output.strip()

            # Normalize answer
            prediction = normalize_answer(response, sample)

            # Evaluate (only for testmini split which has ground truth)
            ground_truth = sample.get("answer", "")
            is_correct = None
            if args.split == "testmini" and ground_truth:
                is_correct = evaluate_answer(prediction, ground_truth)
                if is_correct:
                    correct += 1

            total += 1

            # Store results
            results[pid] = {
                "pid": pid,
                "question": sample.get("question", ""),
                "query": sample.get("query", ""),
                "question_type": sample.get("question_type", ""),
                "answer_type": sample.get("answer_type", ""),
                "choices": sample.get("choices", []),
                "unit": sample.get("unit", ""),
                "precision": sample.get("precision", 0),
                "ground_truth": ground_truth,
                "response": response,
                "prediction": prediction,
                "correct": is_correct,
                "metadata": sample.get("metadata", {}),
            }

            # Track category-wise performance
            category = sample.get("metadata", {}).get("category", "unknown")
            if category not in category_scores:
                category_scores[category] = {"correct": 0, "total": 0}

            category_scores[category]["total"] += 1
            if is_correct:
                category_scores[category]["correct"] += 1

            if args.verbose:
                logging.info(f"\nSample {pid}:")
                logging.info(f"Question: {sample.get('question', '')}")
                logging.info(f"Response: {response}")
                logging.info(f"Prediction: {prediction}")
                logging.info(f"Ground Truth: {ground_truth}")
                logging.info(f"Correct: {is_correct}")

        except Exception as e:
            logging.error(f"Error processing sample {pid}: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            continue

    # Calculate accuracy if applicable
    if args.split == "testmini":
        accuracy = correct / total if total > 0 else 0.0
    else:
        accuracy = None
        correct = None

    # Save results to CSV
    model_name = args.model.split("/")[-1]
    results_file = output_dir / f"{model_name}_MathVista_{args.split}.csv"

    fieldnames = [
        "pid",
        "question",
        "query",
        "question_type",
        "answer_type",
        "choices",
        "unit",
        "precision",
        "ground_truth",
        "response",
        "prediction",
        "correct",
        "metadata",
    ]

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results.values():
            # Convert list and dict fields to strings for CSV
            row = result.copy()
            if isinstance(row.get("choices"), list):
                row["choices"] = "; ".join(str(c) for c in row["choices"])
            if isinstance(row.get("metadata"), dict):
                row["metadata"] = json.dumps(row["metadata"])
            writer.writerow(row)

    # Save summary to JSON
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "total_samples": total,
        "category_scores": category_scores,
    }

    if accuracy is not None:
        summary["correct"] = correct
        summary["accuracy"] = accuracy

    summary_file = output_dir / f"{model_name}_MathVista_{args.split}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\n{'='*80}")
    print("MathVista Evaluation Results (SMLX)")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Total Samples: {total}")
    if accuracy is not None:
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy*100:.2f}%")
    else:
        print("Accuracy not computed for this split (no ground truth labels)")

    if category_scores:
        print("\n" + "-" * 80)
        print("Category-wise Performance:")
        print("-" * 80)
        for category, scores in sorted(category_scores.items()):
            cat_total = scores["total"]
            cat_correct = scores["correct"]
            cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0.0
            print(f"  {category}: {cat_correct}/{cat_total} ({cat_accuracy*100:.2f}%)")

    print("=" * 80)
    print("\nResults saved to:")
    print(f"  CSV: {results_file}")
    print(f"  JSON: {summary_file}")


if __name__ == "__main__":
    main()
