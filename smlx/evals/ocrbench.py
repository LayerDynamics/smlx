"""
OCRBench evaluation benchmark for SMLX.

OCRBench is a comprehensive OCR (Optical Character Recognition) benchmark designed to
evaluate the OCR capabilities of Large Multimodal Models. It contains 1,000 question-answer
pairs from 29 different datasets covering 5 main OCR task components.

Dataset: echo840/OCRBench
Split: test (1,000 samples)
Tasks: Text Recognition, Scene Text VQA, Document VQA, KIE, Handwritten Math
Metrics: Overall accuracy, type-wise accuracy, dataset-wise breakdown

Example usage:
    python -m smlx.evals.ocrbench \\
        --model mlx-community/SmolVLM-256M-Instruct \\
        --max-samples 10
"""

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from smlx.evals.utils import inference, load_model


def process_question(sample: dict) -> str:
    """
    Format the question for OCRBench evaluation.

    For OCRBench, questions are used as-is without additional formatting.

    Args:
        sample: Dataset sample containing 'question' field

    Returns:
        str: The question text

    Example:
        >>> sample = {"question": "What is the text in this image?"}
        >>> process_question(sample)
        'What is the text in this image?'
    """
    return sample["question"]


def normalize_answer(response: str, problem: dict) -> Optional[str]:
    """
    Normalize the model's response to extract the answer.

    OCRBench uses minimal normalization - just strips whitespace.
    The evaluation uses substring matching against ground truth answers.

    Args:
        response: Raw model output text
        problem: Dataset sample (not used for OCRBench, kept for API consistency)

    Returns:
        Optional[str]: Stripped response or None if empty

    Example:
        >>> normalize_answer("  The answer is 42  ", {})
        'The answer is 42'
    """
    if not response:
        return None
    return response.strip()


def evaluate_answer(prediction: Optional[str], ground_truth: list) -> bool:
    """
    Check if any ground truth answer is contained in the prediction.

    OCRBench evaluation uses case-insensitive substring matching.
    A prediction is correct if ANY of the ground truth answers appears
    as a substring in the prediction.

    Args:
        prediction: The normalized prediction from the model
        ground_truth: List of acceptable ground truth answers

    Returns:
        bool: True if any ground truth is found in prediction

    Example:
        >>> evaluate_answer("The answer is 42", ["42", "forty-two"])
        True
        >>> evaluate_answer("The answer is 43", ["42", "forty-two"])
        False
    """
    if prediction is None:
        return False

    pred = prediction.strip().lower()
    return any(str(a).strip().lower() in pred for a in ground_truth)


def OCRBench_val(results_list: list, args, model_name: str, dataset: str = "OCRBench"):
    """
    Evaluate OCRBench results and generate reports.

    This function processes the collected results, computes accuracy metrics,
    and saves detailed CSV and JSON outputs with type-wise breakdowns.

    Args:
        results_list: List of result dictionaries from evaluation
        args: Parsed command-line arguments
        model_name: Name of the evaluated model
        dataset: Dataset name for output files (default: "OCRBench")

    Side Effects:
        - Writes CSV file with detailed predictions
        - Writes JSON file with summary statistics
        - Prints formatted results to console
    """
    correct = 0
    total = len(results_list)
    category_scores = {}

    for row in results_list:
        ground_truth = row["ground_truth"]

        # Handle both string and list ground truth formats
        if isinstance(ground_truth, str):
            # Split semicolon-separated answers
            ground_truth = [a.strip() for a in ground_truth.split(";")]

        prediction = row["prediction"]

        is_correct = evaluate_answer(prediction, ground_truth)
        row["correct"] = is_correct

        if is_correct:
            correct += 1

        # Track type-wise performance
        category = row["type"]
        if category not in category_scores:
            category_scores[category] = {"correct": 0, "total": 0}

        category_scores[category]["total"] += 1
        if is_correct:
            category_scores[category]["correct"] += 1

    accuracy = correct / total if total > 0 else 0

    # Save results to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{model_name}_{dataset}_{args.split}.csv"

    fieldnames = [
        "id",
        "question",
        "dataset",
        "type",
        "ground_truth",
        "response",
        "prediction",
        "correct",
    ]

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_list:
            out_row = row.copy()
            # Convert list ground truth to semicolon-separated string for CSV
            if isinstance(out_row["ground_truth"], list):
                out_row["ground_truth"] = "; ".join(map(str, out_row["ground_truth"]))
            writer.writerow(out_row)

    # Save summary to JSON
    summary = {
        "model": model_name,
        "dataset": args.dataset,
        "split": args.split,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "category_scores": category_scores,
    }

    summary_file = output_dir / f"{model_name}_{dataset}_{args.split}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\n{'='*80}")
    print(f"{dataset} Evaluation Results (SMLX)")
    print(f"{'='*80}")
    print(f"Model: {summary['model']}")
    print(f"Split: {args.split}")
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    if len(category_scores.items()) > 1:
        print("\n" + "-" * 80)
        print("Type-wise Performance:")
        print("-" * 80)
        for category, scores in sorted(category_scores.items()):
            cat_total = scores["total"]
            cat_correct = scores["correct"]
            cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0
            print(f"  {category}: {cat_correct}/{cat_total} ({cat_accuracy*100:.2f}%)")

    print("=" * 80)
    print("\nResults saved to:")
    print(f"  CSV: {results_file}")
    print(f"  JSON: {summary_file}")


def parse_args():
    """Parse command-line arguments for OCRBench evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate vision-language models on OCRBench benchmark (SMLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on full test split
  python -m smlx.evals.ocrbench \\
      --model mlx-community/SmolVLM-256M-Instruct

  # Quick test with 10 samples
  python -m smlx.evals.ocrbench \\
      --model mlx-community/Moondream2-1.8B \\
      --max-samples 10 \\
      --verbose

  # Evaluate pre-generated predictions
  python -m smlx.evals.ocrbench \\
      --predictions-file results/ocrbench/model_OCRBench_test.csv
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
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
        default="echo840/OCRBench",
        help="HuggingFace dataset name (default: echo840/OCRBench)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test"],
        help="Dataset split to evaluate on (default: test)",
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
        "--predictions-file",
        type=str,
        default=None,
        help="CSV file with pre-generated predictions to evaluate (skip inference)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ocrbench",
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
    """Main evaluation loop for OCRBench benchmark."""
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Handle evaluation-only mode (pre-generated predictions)
    if args.predictions_file:
        logging.info(f"Loading predictions from {args.predictions_file} for evaluation")
        try:
            with open(args.predictions_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                loaded_results = list(reader)

            model_name = Path(args.predictions_file).stem.split("_OCRBench")[0]
            dataset = (
                "OCRBench-v2" if "OCRBench-v2" in args.predictions_file else "OCRBench"
            )
            OCRBench_val(loaded_results, args, model_name, dataset)
            logging.info("Evaluation complete")
            return
        except Exception as e:
            print(f"Error loading predictions file: {e}")
            return

    # Validate model argument for inference mode
    if not args.model:
        print("Error: --model argument is required when not using --predictions-file")
        return

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
        from datasets import Dataset, IterableDataset, load_dataset
    except ImportError:
        print("Error: datasets library not found.")
        print("Install with: pip install 'smlx[evals]'")
        return

    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    # Handle sample limiting based on dataset type
    if args.max_samples:
        if args.streaming:
            # IterableDataset has take() method
            if isinstance(dataset, IterableDataset):
                dataset = dataset.take(args.max_samples)
        else:
            # Dataset has select() method
            if isinstance(dataset, Dataset):
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Initialize tracking variables
    results = {}

    # Evaluate each sample
    sample_iterator = enumerate(dataset)
    if not args.streaming:
        sample_iterator = enumerate(tqdm(dataset, desc="Evaluating OCRBench"))

    for idx, sample in sample_iterator:
        pid = sample.get("id", str(idx))

        try:
            # Load and process image
            image = None
            if "image" in sample and sample["image"]:
                image = sample["image"].convert("RGB")
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

            # Extract ground truth (handle both 'answers' and 'answer' fields)
            ground_truth = sample.get("answers", sample.get("answer", []))
            if not isinstance(ground_truth, list):
                ground_truth = [ground_truth]

            # Store results (evaluation happens in OCRBench_val)
            results[pid] = {
                "id": pid,
                "question": sample["question"],
                "dataset": sample.get("dataset", ""),
                "type": sample.get("type", sample.get("question_type", "")),
                "ground_truth": ground_truth,
                "response": response,
                "prediction": prediction,
                "correct": False,  # Will be filled by OCRBench_val
            }

            if args.verbose:
                logging.info(f"\nSample {pid}:")
                logging.info(f"Question: {sample['question']}")
                logging.info(f"Response: {response}")
                logging.info(f"Prediction: {prediction}")
                logging.info(f"Ground Truth: {ground_truth}")

        except Exception as e:
            logging.error(f"Error processing sample {pid}: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            continue

    # Evaluate results
    results_list = list(results.values())
    model_name = args.model.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]
    OCRBench_val(results_list, args, model_name, dataset_name)


if __name__ == "__main__":
    main()
