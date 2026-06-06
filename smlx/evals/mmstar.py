"""
MMStar evaluation benchmark for SMLX.

MMStar (Elite Vision-Indispensable Multi-Modal Benchmark) is a comprehensive benchmark
for evaluating vision-language models across 6 core capabilities and 18 subcategories.
It contains 1,500 carefully curated samples designed to minimize data leakage and
ensure true multimodal understanding.

Dataset: Lin-Chen/MMStar
Split: val (1,500 samples)
Categories: 6 core capabilities (coarse/fine-grained perception, reasoning, science, math)
Subcategories: 18 detailed evaluation axes
Metrics: Overall accuracy, category-wise and subcategory-wise breakdowns

Example usage:
    python -m smlx.evals.mmstar \\
        --model mlx-community/SmolVLM-256M-Instruct \\
        --max-samples 10
"""

import argparse
import csv
import logging
import random
import re
from copy import deepcopy
from json import dump
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

from smlx.evals.utils import inference, load_model


def extract_answer(predict: str, answer: str) -> bool:
    """
    Extract the answer from model predictions using priority-based pattern matching.

    This function uses a sophisticated two-tier pattern matching strategy:
    1. Concluding patterns (priority 2): Patterns that indicate a final answer
       (e.g., "the answer is A", "therefore A")
    2. General patterns (priority 1): Patterns that match answer appearances
       (e.g., "(A)", "option A")

    When multiple matches are found, the function prioritizes:
    - Higher priority patterns over lower priority
    - Later positions in text over earlier (final answer over intermediate reasoning)

    Args:
        predict: Model prediction text
        answer: Ground truth answer (A, B, C, D, or E)

    Returns:
        bool: True if extracted answer matches ground truth

    Example:
        >>> extract_answer("Let's think... Initially B seems right. However, the answer is A", "A")
        True
        >>> extract_answer("The options are (A) red, (B) blue. The answer is B", "B")
        True
    """
    text = predict.lower().replace("\n", " ").strip()
    answer_lower = answer.lower()

    # General patterns - lower priority (1)
    general_templates = [
        r"^{0}\b",  # Starts with answer letter
        r"^\({0}",  # Starts with (A)
        r"^option {0}\b",  # Starts with "option A"
        r"\b{0}\s*[:\.\)]",  # Letter followed by punctuation
        r"(?:^|\.|\s)\s*{0}\.",  # Letter with period
        r"\({0}\)",  # Parenthesized answer
        r"option\s+{0}\b",  # "option A"
        r"choice\s+{0}\b",  # "choice A"
    ]

    # Concluding patterns - higher priority (2)
    concluding_templates = [
        r"^the answer is {0}\b",
        r"answer:\s*{0}\b",
        r"answer\s+is\s+{0}\b",
        r"correct\s+(?:answer|option|choice)\s+is:?\s+{0}\b",
        r"the\s+answer\s+is\s+{0}\b",
        r"is\s+{0}\s*:",
        r"(?:therefore|thus|hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?{0}\b",
        r"(?:select|choose)\s+{0}\b",
        r"it\s+is\s+{0}\b",
        r"would\s+be\s+{0}\b",
        r"\*\*(?:revised\s+)?answer\*\*:\s*{0}\b",
        r"(?:correct\s+)?category\s+(?:for\s+this\s+image\s+)?is\s+\*\*{0}[:\s]",
    ]

    possible_answers = ["a", "b", "c", "d", "e"]
    matches = []

    # Find all matches with their positions and priorities
    for ans in possible_answers:
        for pri, template_list in [(2, concluding_templates), (1, general_templates)]:
            for template in template_list:
                pattern = template.format(ans)
                for match in re.finditer(pattern, text):
                    # Store (end_position, answer, priority)
                    matches.append((match.end(), ans, pri))

    if not matches:
        return False

    # Sort by priority (descending) then by position (descending)
    # This gives us the highest priority match that appears latest in the text
    matches.sort(key=lambda m: (-m[2], -m[0]))
    latest_ans = matches[0][1]

    return latest_ans == answer_lower


def MMStar_eval(data: list, args, model_name: str):
    """
    Evaluate MMStar results with category and subcategory scoring.

    This function processes results, computes accuracy at multiple levels
    (overall, category, subcategory), and generates detailed reports.

    Args:
        data: List of result dictionaries with predictions
        args: Parsed command-line arguments
        model_name: Name of the evaluated model

    Side Effects:
        - Writes CSV file with predictions and scores
        - Writes JSON file with detailed scores
        - Prints formatted results to console
    """
    # Initialize category hierarchy (6 categories, 18 subcategories)
    MMStar_score_l2 = {
        "coarse perception": {
            "image scene and topic": 0,
            "image style & quality": 0,
            "image emotion": 0,
        },
        "fine-grained perception": {
            "object counting": 0,
            "recognition": 0,
            "localization": 0,
        },
        "instance reasoning": {
            "single-instance reasoning": 0,
            "cross-instance attribute reasoning": 0,
            "cross-instance relation reasoning": 0,
        },
        "logical reasoning": {
            "code & sequence reasoning": 0,
            "diagram reasoning": 0,
            "common reasoning": 0,
        },
        "science & technology": {
            "biology & chemistry & physics": 0,
            "electronics & energy & mechanical eng.": 0,
            "geography & earth science & agriculture": 0,
        },
        "math": {
            "geometry": 0,
            "numeric commonsense and calculation": 0,
            "statistical reasoning": 0,
        },
    }

    # Deep copy for counting total samples per category
    MMStar_counter = deepcopy(MMStar_score_l2)

    # Evaluate each prediction
    for line in tqdm(data, desc="Evaluating MMStar"):
        predict = str(line["prediction"])
        answers = str(line["answer"])
        category = str(line["category"])
        l2_category = str(line["l2_category"])

        MMStar_counter[category][l2_category] += 1

        # Use priority-based extraction
        if extract_answer(predict, answers):
            MMStar_score_l2[category][l2_category] += 1
            line["score"] = 1
        else:
            line["score"] = 0

    # Calculate scores at all levels
    MMStar_score = {}
    MMStar_score["final score"] = 0
    total_correct = 0

    for k, v in MMStar_score_l2.items():
        cat_total = sum(MMStar_counter[k].values())
        cat_correct = 0

        # Calculate subcategory scores
        for l2_k, l2_v in v.items():
            count = MMStar_counter[k][l2_k]
            if count > 0:
                MMStar_score[f"{k}({l2_k})"] = float(l2_v) / float(count)
            else:
                MMStar_score[f"{k}({l2_k})"] = 0.0
            cat_correct += l2_v
            total_correct += l2_v

        # Calculate category score
        MMStar_score[k] = float(cat_correct) / cat_total if cat_total > 0 else 0.0
        MMStar_score["final score"] += cat_correct

    # Calculate overall accuracy
    if len(data) > 0:
        MMStar_score["final score"] = float(MMStar_score["final score"]) / float(len(data))

    # Save results to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{model_name}_MMStar_{args.split}_predictions.csv"

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    # Save summary to JSON
    summary_file = output_dir / f"{model_name}_MMStar_{args.split}_score.json"
    with open(summary_file, "w") as f:
        dump(MMStar_score, f, indent=2)

    # Print results
    print("\n" + "=" * 80)
    print("MMStar Evaluation Results (SMLX)")
    print("=" * 80)
    print(f"\nFinal Score: {total_correct}/{len(data)} = {MMStar_score['final score']*100:.2f}%\n")

    print("-" * 80)
    print("Category Scores:")
    print("-" * 80)
    category_order = [
        "coarse perception",
        "fine-grained perception",
        "instance reasoning",
        "logical reasoning",
        "science & technology",
        "math",
    ]

    for category in category_order:
        if category in MMStar_score:
            cat_total = sum(MMStar_counter[category].values())
            cat_correct = sum(MMStar_score_l2[category].values())
            print(
                f"{category:30s}: {cat_correct:4d}/{cat_total:4d} = {MMStar_score[category]*100:6.2f}%"
            )

    print("\n" + "-" * 80)
    print("Subcategory Scores:")
    print("-" * 80)

    for category in category_order:
        print(f"\n{category.upper()}:")
        for l2_cat, score in MMStar_score_l2[category].items():
            count = MMStar_counter[category][l2_cat]
            pct = (score / count * 100) if count > 0 else 0
            print(f"  {l2_cat:55s}: {score:4d}/{count:4d} = {pct:6.2f}%")

    print("\n" + "=" * 80)
    print("\nResults saved to:")
    print(f"  CSV: {results_file}")
    print(f"  JSON: {summary_file}")


def parse_args():
    """Parse command-line arguments for MMStar evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate vision-language models on MMStar benchmark (SMLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on full validation split
  python -m smlx.evals.mmstar \\
      --model mlx-community/SmolVLM-256M-Instruct

  # Quick test with 10 samples
  python -m smlx.evals.mmstar \\
      --model mlx-community/Moondream2-1.8B \\
      --max-samples 10 \\
      --verbose

  # Evaluate pre-generated predictions
  python -m smlx.evals.mmstar \\
      --prediction-file results/mmstar/model_MMStar_val_predictions.csv
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
        default="Lin-Chen/MMStar",
        help="HuggingFace dataset name (default: Lin-Chen/MMStar)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val"],
        help="Dataset split to evaluate on (default: val)",
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
        "--prediction-file",
        type=str,
        default=None,
        help="CSV file with pre-generated predictions to evaluate (skip inference)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mmstar",
        help="Directory to save CSV and JSON results",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Maximum number of tokens to generate per question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 for deterministic, 0.7 for diverse responses)",
    )
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs=2,
        default=None,
        help="Resize shape for images (height width)",
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
    """Main evaluation loop for MMStar benchmark."""
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Starting MMStar evaluation")

    # Handle evaluation-only mode (pre-generated predictions)
    if args.prediction_file:
        logging.info(f"Loading predictions from {args.prediction_file} for evaluation")
        try:
            results = []
            with open(args.prediction_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                results = list(reader)

            model_name = Path(args.prediction_file).stem.split("_MMStar")[0]
            MMStar_eval(results, args, model_name)
            logging.info("Evaluation complete")
            return
        except Exception as e:
            print(f"Error loading predictions file: {e}")
            return

    # Validate model argument for inference mode
    if not args.model:
        print("Error: --model argument is required when not using --prediction-file")
        return

    # Load dataset
    logging.info(f"Loading dataset {args.dataset}, split {args.split}")
    try:
        from datasets import Dataset, IterableDataset
    except ImportError:
        print("Error: datasets library not found.")
        print("Install with: pip install 'smlx[evals]'")
        return

    from smlx.evals.utils import resolve_eval_dataset

    dataset_loaded, _ = resolve_eval_dataset(
        "mmstar",
        args.dataset,
        args.split,
        prefer_local=(args.dataset == "Lin-Chen/MMStar"),
        streaming=args.streaming,
    )

    # Handle streaming vs non-streaming datasets
    if args.streaming:
        dataset: IterableDataset = dataset_loaded  # type: ignore
        if args.max_samples:
            dataset = dataset.take(args.max_samples)
    else:
        dataset: Dataset = dataset_loaded  # type: ignore
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

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

    # Initialize tracking variables
    results = []

    # Evaluate each sample
    sample_iterator = dataset
    if not args.streaming:
        sample_iterator = tqdm(dataset, desc="Evaluating MMStar")

    for example in sample_iterator:
        try:
            example_dict = cast(dict[str, Any], example)  # Type hint for dataset row
            question = example_dict["question"]
            image = example_dict["image"].convert("RGB")

            # Generate prediction
            prediction = inference(
                model,
                processor,
                question,
                image,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                resize_shape=tuple(args.resize_shape) if args.resize_shape else None,
                verbose=args.verbose,
            )

            results.append(
                {
                    "question": question,
                    "answer": example_dict["answer"],
                    "category": example_dict["category"],
                    "l2_category": example_dict["l2_category"],
                    "meta_info": str(example_dict.get("meta_info", "")),
                    "prediction": prediction,
                }
            )

            if args.verbose:
                logging.info(f"\nQuestion: {question}")
                logging.info(f"Answer: {example_dict['answer']}")
                logging.info(
                    f"Category: {example_dict['category']} / {example_dict['l2_category']}"
                )
                logging.info(f"Prediction: {prediction}")

        except Exception as e:
            logging.error(f"Error processing sample: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            continue

    if args.verbose:
        print("\nFirst 5 results:")
        for i, result in enumerate(results[:5]):
            print(
                f"{i+1}. Question: {result['question'][:50]}... | "
                f"Answer: {result['answer']} | "
                f"Prediction: {result['prediction'][:50]}..."
            )

    # Evaluate results
    model_name = args.model.split("/")[-1]
    MMStar_eval(results, args, model_name)

    logging.info("Evaluation complete")


if __name__ == "__main__":
    main()
