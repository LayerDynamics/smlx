"""
MMMU evaluation benchmark for SMLX.

MMMU (Massive Multi-discipline Multimodal Understanding) is a comprehensive benchmark
evaluating expert-level multimodal understanding across 30 subjects spanning 6 core disciplines
(Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, Tech & Engineering).

Dataset: MMMU/MMMU
Splits: dev (150), validation (900), test (10,500, no ground truth)
Subjects: 30 college-level domains with 183 subfields
Question Types: Multiple choice (A-F), open-ended (text/numeric)
Multi-image: Supports up to 7 images per question
Metrics: Overall accuracy, subject-wise breakdown

Example usage:
    # Evaluate all 30 subjects
    python -m smlx.evals.mmmu \\
        --model mlx-community/SmolVLM-256M-Instruct \\
        --split validation

    # Evaluate specific subject
    python -m smlx.evals.mmmu \\
        --model mlx-community/SmolVLM-256M-Instruct \\
        --subset Math \\
        --max-samples 10

    # List all available subjects
    python -m smlx.evals.mmmu --list-subjects
"""

import argparse
import csv
import logging
import random
import re
from json import dump
from pathlib import Path

from tqdm import tqdm

from smlx.evals.utils import inference, load_model

# All 30 MMMU subjects (confirmed from dataset)
MMMU_SUBJECTS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]

# MMMU Pro subjects (3 specialized evaluation sets)
MMMU_PRO_SUBJECTS = [
    "vision",
    "standard (10 options)",
    "standard (4 options)",
]


def normalize_number(s) -> float | str:
    """
    Normalize numeric strings for comparison.

    Removes commas and converts to float for numeric comparison.

    Args:
        s: String or number to normalize

    Returns:
        float or str: Normalized number as float, or original string if not numeric

    Example:
        >>> normalize_number("1,234.56")
        1234.56
        >>> normalize_number("text")
        'text'
    """
    try:
        return float(str(s).strip().replace(",", ""))
    except Exception:
        return str(s).strip()


def process_question(example: dict) -> str:
    """
    Process MMMU question to format it properly.

    MMMU questions may have:
    - Options field (string representation of list needing parsing)
    - <image n> tags that should be removed
    - Multiple images referenced in text

    Args:
        example: Dataset sample containing 'question' and optionally 'options'

    Returns:
        str: Formatted question with options appended as A., B., C., etc.

    Example:
        >>> example = {
        ...     "question": "What shape is shown in <image 1>?",
        ...     "options": "['circle', 'square', 'triangle']"
        ... }
        >>> process_question(example)
        'What shape is shown in?\\n\\nOptions:\\nA. circle\\nB. square\\nC. triangle'
    """
    question = example.get("question", "")

    # Parse options if they exist
    # Options come as string like "['option1', 'option2', 'option3']"
    options = example.get("options", None)
    if options:
        # Remove brackets, quotes, and split by comma
        options = re.sub(r'[\[\]"\']', "", options).split(", ")

    if options and isinstance(options, list) and len(options) > 0:
        question += "\n\nOptions:"
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D, ...
            question += f"\n{letter}. {option}"

    # Remove <image n> tags from the question
    # These are just placeholders indicating image presence
    question = re.sub(r"<image \d+>", "", question).strip()

    return question


def get_images(example: dict) -> list:
    """
    Extract images from MMMU example.

    MMMU can have multiple images per question (up to 7 images).
    Images are stored as 'image' or 'image_1' through 'image_7'.

    Args:
        example: Dataset sample potentially containing multiple image fields

    Returns:
        list: List of PIL Images in RGB format

    Example:
        >>> example = {"image_1": <PIL.Image>, "image_2": <PIL.Image>}
        >>> images = get_images(example)
        >>> len(images)
        2
    """
    images = []

    # Check for single 'image' field first
    if "image" in example and example["image"] is not None:
        try:
            img = example["image"].convert("RGB")
            images.append(img)
        except Exception as e:
            logging.warning(f"Could not process 'image' field: {e}")

    # Check for image_1 through image_7
    for i in range(1, 8):
        img_key = f"image_{i}"
        if img_key in example and example[img_key] is not None:
            try:
                img = example[img_key].convert("RGB")
                images.append(img)
            except Exception as e:
                logging.warning(f"Could not process {img_key}: {e}")
                continue

    return images


def MMMU_eval(data: list, args, model_name: str):
    """
    Evaluate MMMU results by subject.

    Handles both multiple choice (A-F, occasionally I) and open-ended questions.
    Uses prioritized regex patterns for multiple choice, substring matching for open-ended.

    Args:
        data: List of result dictionaries with predictions
        args: Parsed command-line arguments
        model_name: Name of the evaluated model

    Side Effects:
        - Writes CSV file with predictions and scores
        - Writes JSON file with detailed subject-wise scores
        - Prints formatted results to console
    """
    # Track by subject
    subject_scores = {}
    subject_counters = {}

    total_correct = 0
    total_questions = 0

    for line in data:
        predict = str(line["prediction"])
        answer = str(line["answer"])
        subject = str(line.get("subject", "Unknown"))

        # Initialize subject tracking if needed
        if subject not in subject_scores:
            subject_scores[subject] = 0
            subject_counters[subject] = 0

        # Count this question
        subject_counters[subject] += 1
        total_questions += 1

        # Normalize for comparison
        predict_lower = predict.lower().strip()
        answer_lower = answer.lower().strip()

        is_correct = False

        # Check if this is a multiple choice question (answer is A-F or I)
        if answer in ["A", "B", "C", "D", "E", "F", "I"]:
            # Multiple choice extraction with prioritized patterns
            patterns = [
                (r"option\s+([a-fi])\b", 10),  # High priority
                (r"answer\s+is:?\s+([a-fi])\b", 10),
                (r"choice\s+is:?\s+([a-fi])\b", 10),
                (r"correct\s+answer\s+is:?\s+([a-fi])\b", 10),
                (r"correct\s+option\s+is:?\s+\(?([a-fi])\)?", 10),
                (r"\(([a-fi])\)", 8),  # Medium priority
                (r"^([a-fi])[.:\)]\s", 8),
                (r"\b([a-fi])\b", 5),  # Low priority - isolated letters
            ]

            best_match = None
            best_priority = -1

            # Try each pattern, keeping the highest priority match
            for pattern, priority in patterns:
                matches = re.findall(pattern, predict_lower, re.IGNORECASE)
                if matches and priority > best_priority:
                    best_match = matches[0].lower()
                    best_priority = priority
                    # Stop early if we found a high-confidence pattern
                    if priority >= 10:
                        break

            # Check if match is correct
            if best_match and best_match == answer_lower:
                is_correct = True
            # Fallback: check first character
            elif not best_match and len(predict_lower) > 0 and predict_lower[0] in "abcdefi":
                if predict_lower[0] == answer_lower:
                    is_correct = True

        else:
            # Open-ended question - use multiple matching strategies
            # 1. Exact substring match (case-insensitive)
            if answer_lower in predict_lower:
                is_correct = True
            # 2. For numeric answers, try numeric comparison
            elif answer.replace(".", "").replace("-", "").replace(",", "").isdigit():
                numbers = re.findall(r"-?\d+\.?\d*", predict)
                answer_num = normalize_number(answer)
                for num_str in numbers:
                    try:
                        pred_num = normalize_number(num_str)
                        # Only compare if both are floats
                        if isinstance(pred_num, float) and isinstance(answer_num, float):
                            if abs(pred_num - answer_num) < 0.01:
                                is_correct = True
                                break
                    except Exception:
                        pass
            # 3. Word-level match for text answers
            else:
                answer_words = set(answer_lower.split())
                predict_words = set(predict_lower.split())
                if answer_words and answer_words.issubset(predict_words):
                    is_correct = True

        if is_correct:
            total_correct += 1
            subject_scores[subject] += 1
            line["score"] = 1
        else:
            line["score"] = 0

    # Calculate final scores
    results = {}
    results["overall_accuracy"] = (
        float(total_correct) / float(total_questions) if total_questions > 0 else 0.0
    )
    results["total_correct"] = total_correct
    results["total_questions"] = total_questions

    # Calculate subject scores
    for subject in sorted(subject_scores.keys()):
        if subject_counters[subject] > 0:
            results[f"subject_{subject}_accuracy"] = float(subject_scores[subject]) / float(
                subject_counters[subject]
            )
            results[f"subject_{subject}_correct"] = subject_scores[subject]
            results[f"subject_{subject}_total"] = subject_counters[subject]

    # Save results to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_name = args.subset if args.subset else "all"
    results_file = output_dir / f"{model_name}_MMMU_{subset_name}_{args.split}_predictions.csv"

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    # Save summary to JSON
    summary_file = output_dir / f"{model_name}_MMMU_{subset_name}_{args.split}_score.json"
    with open(summary_file, "w") as f:
        dump(results, f, indent=2)

    # Print results
    print("\n" + "=" * 80)
    print("MMMU Evaluation Results (SMLX)")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Total Questions: {total_questions}")
    print(f"Total Correct: {total_correct}")
    print(
        f"Overall Accuracy: {results['overall_accuracy']*100:.2f}% ({total_correct}/{total_questions})"
    )

    if len(subject_scores) > 1 or (len(subject_scores) == 1 and args.subset):
        print("\n" + "-" * 80)
        print("Subject Breakdown:")
        print("-" * 80)
        for subject in sorted(subject_scores.keys()):
            acc = results.get(f"subject_{subject}_accuracy", 0.0)
            correct = results.get(f"subject_{subject}_correct", 0)
            total = results.get(f"subject_{subject}_total", 0)
            print(f"  {subject:40s}: {acc*100:6.2f}% ({correct:3d}/{total:3d})")

    print("=" * 80)
    print("\nResults saved to:")
    print(f"  CSV: {results_file}")
    print(f"  JSON: {summary_file}")

    logging.info(f"MMMU evaluation complete, results saved to {summary_file}")


def list_subjects():
    """Print all available MMMU subjects."""
    print("\n" + "=" * 80)
    print("MMMU Available Subjects (30 total)")
    print("=" * 80)
    print("\nCore Disciplines:")
    print("-" * 80)

    disciplines = {
        "Art & Design": ["Art", "Art_Theory", "Design", "Music"],
        "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
        "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
        "Health & Medicine": [
            "Basic_Medical_Science",
            "Clinical_Medicine",
            "Diagnostics_and_Laboratory_Medicine",
            "Pharmacy",
            "Public_Health",
        ],
        "Humanities & Social Science": [
            "History",
            "Literature",
            "Psychology",
            "Sociology",
        ],
        "Tech & Engineering": [
            "Agriculture",
            "Architecture_and_Engineering",
            "Computer_Science",
            "Electronics",
            "Energy_and_Power",
            "Materials",
            "Mechanical_Engineering",
        ],
    }

    for discipline, subjects in disciplines.items():
        print(f"\n{discipline}:")
        for subject in subjects:
            print(f"  - {subject}")

    print("\n" + "=" * 80)
    print("MMMU Pro Subjects (3 specialized sets)")
    print("=" * 80)
    for i, subject in enumerate(MMMU_PRO_SUBJECTS, 1):
        print(f"{i}. {subject}")
    print("=" * 80 + "\n")


def parse_args():
    """Parse command-line arguments for MMMU evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate vision-language models on MMMU benchmark (SMLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available subjects
  python -m smlx.evals.mmmu --list-subjects

  # Evaluate specific subject
  python -m smlx.evals.mmmu \\
      --model mlx-community/SmolVLM-256M-Instruct \\
      --subset Math \\
      --max-samples 10

  # Evaluate all 30 subjects (validation split)
  python -m smlx.evals.mmmu \\
      --model mlx-community/SmolVLM-256M-Instruct \\
      --split validation

  # Evaluate pre-generated predictions
  python -m smlx.evals.mmmu \\
      --prediction-file results/mmmu/model_MMMU_Math_validation_predictions.csv
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
        default="MMMU/MMMU",
        help="HuggingFace dataset name (default: MMMU/MMMU)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["dev", "validation", "test"],
        help="Dataset split (default: validation). Note: test split has no ground truth",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Specific subject to evaluate (e.g., Math, Physics). See --list-subjects for all options",
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
        help="Maximum number of samples to evaluate per subject (useful for debugging)",
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="List all 30 available subjects and exit",
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
        default="results/mmmu",
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
        default=0.0,
        help="Sampling temperature (0.0 for deterministic/greedy decoding)",
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
    """Main evaluation loop for MMMU benchmark."""
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Handle --list-subjects flag
    if args.list_subjects:
        list_subjects()
        return

    # Determine which subjects to use
    if "pro" in args.dataset.lower():
        subjects = MMMU_PRO_SUBJECTS
    else:
        subjects = MMMU_SUBJECTS

    # Handle evaluation-only mode (pre-generated predictions)
    if args.prediction_file:
        logging.info("Loading predictions from %s for evaluation", args.prediction_file)
        try:
            results = []
            with open(args.prediction_file) as f:
                reader = csv.DictReader(f)
                results = list(reader)

            model_name = Path(args.prediction_file).stem.split("_MMMU_")[0]
            MMMU_eval(results, args, model_name)
            logging.info("Evaluation complete")
            return
        except Exception as e:
            print(f"Error loading predictions file: {e}")
            return

    # Validate model argument for inference mode
    if not args.model:
        print(
            "Error: --model argument is required when not using --prediction-file or --list-subjects"
        )
        return

    logging.info("Starting MMMU evaluation")

    # Validate subset if provided
    if args.subset and args.subset not in subjects:
        print(f"Error: Invalid subset '{args.subset}'")
        print(f"Valid subjects are: {', '.join(subjects)}")
        print("Run with --list-subjects to see all options")
        return

    # Load dataset
    logging.info(f"Loading dataset {args.dataset}, split {args.split}")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not found.")
        print("Install with: pip install 'smlx[evals]'")
        return

    # Load subject datasets
    datasets = {}

    # Prefer the bundled local copy when usable: no --subset override, default
    # repo, requested split present locally, non-streaming. The local MMMU is
    # the combined all-subjects dataset, so we regroup it per subject via the
    # id field ("<split>_<Subject>_<n>") to preserve the per-subject breakdown.
    if not args.subset and not args.streaming and args.dataset == "MMMU/MMMU":
        try:
            from smlx.data import local as _local

            if _local.is_available("mmmu") and args.split in _local.available_splits("mmmu"):
                combined = _local.load("mmmu", split=args.split)
                prefix = f"{args.split}_"
                buckets: dict[str, list[int]] = {}
                for i, ex_id in enumerate(combined["id"]):
                    core = ex_id[len(prefix) :] if ex_id.startswith(prefix) else ex_id
                    subject = core.rsplit("_", 1)[0] if "_" in core else core
                    buckets.setdefault(subject, []).append(i)
                for subject, idxs in buckets.items():
                    datasets[subject] = combined.select(idxs)
                rel = _local.local_path("mmmu").relative_to(_local.data_dir())
                logging.info(
                    f"Dataset source: local data/{rel} "
                    f"(split={args.split}, {len(datasets)} subjects)"
                )
        except Exception as e:
            logging.warning(f"Local MMMU load failed ({e}); using HuggingFace source")
            datasets = {}

    if not datasets and args.subset:
        logging.info(f"Using subset: {args.subset}")
        try:
            datasets[args.subset] = load_dataset(
                args.dataset,
                args.subset,
                split=args.split,
                streaming=args.streaming,
            )
        except Exception as e:
            print(f"Error loading dataset for {args.subset}: {e}")
            return
    elif not datasets:
        logging.info("Evaluating all 30 subjects")
        for subject in subjects:
            try:
                datasets[subject] = load_dataset(
                    args.dataset,
                    name=subject,
                    split=args.split,
                    streaming=args.streaming,
                )
            except Exception as e:
                logging.error(f"Error loading dataset for {subject}: {e}")
                continue

    if not datasets:
        print("Error: No datasets loaded successfully")
        return

    # Limit samples if specified
    if args.max_samples and not args.streaming:
        for subject in datasets:
            try:
                dataset_len = len(datasets[subject])
                datasets[subject] = datasets[subject].select(
                    range(min(args.max_samples, dataset_len))
                )
            except Exception:
                # Streaming datasets don't support select
                pass

    logging.info(f"Dataset subjects loaded: {len(datasets.keys())}")

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

    # Evaluate each subject
    for subject, dataset in tqdm(datasets.items(), desc="Processing subjects"):
        sample_iterator = dataset
        if not args.streaming:
            sample_iterator = tqdm(dataset, desc=f"Processing {subject}", leave=False)

        for idx, example in enumerate(sample_iterator):
            if args.streaming and args.max_samples and idx >= args.max_samples:
                break

            try:
                # Process question and get images
                question = process_question(example)
                images = get_images(example)

                if not images:
                    logging.warning(f"No images for sample {idx} in {subject}, skipping")
                    continue

                # Generate prediction
                prediction = inference(
                    model,
                    processor,
                    question,
                    images,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    resize_shape=tuple(args.resize_shape) if args.resize_shape else None,
                    verbose=args.verbose,
                )

                # Store result
                result = {
                    "id": example.get("id", idx),
                    "question": question,
                    "answer": example.get("answer", ""),
                    "subfield": example.get("subfield", "Unknown"),
                    "topic_difficulty": example.get("topic_difficulty", "Unknown"),
                    "question_type": example.get("question_type", "Unknown"),
                    "prediction": prediction,
                    "subject": example.get("subject", None) or subject,
                }
                results.append(result)

                if args.verbose:
                    logging.info(f"\n{subject} Sample {idx}:")
                    logging.info(f"Question: {question[:100]}...")
                    logging.info(f"Answer: {result['answer']}")
                    logging.info(f"Prediction: {prediction}")

            except Exception as e:
                logging.error(f"Error processing sample {idx} in {subject}: {e}")
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
    MMMU_eval(results, args, model_name)

    logging.info("Evaluation complete")


if __name__ == "__main__":
    main()
