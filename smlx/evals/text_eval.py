"""
Text Model Evaluation

Evaluate text-only models (e.g., SmolLM2) using standard NLP metrics:
- Perplexity (PPL) on WikiText-2, WikiText-103, Penn Treebank, etc.
- Token-level loss analysis
- Statistical measures (mean, std, confidence intervals)

Usage:
    # Evaluate perplexity on WikiText-2
    python -m smlx.evals.text_eval \\
        --model mlx-community/SmolLM2-135M-Instruct \\
        --dataset wikitext \\
        --split test

    # Evaluate on custom dataset with specific parameters
    python -m smlx.evals.text_eval \\
        --model mlx-community/SmolLM2-360M-Instruct \\
        --dataset HuggingFaceFW/fineweb-edu \\
        --split train \\
        --batch-size 16 \\
        --sequence-length 1024 \\
        --num-samples 500

Example (programmatic):
    from smlx.evals.text_eval import evaluate_perplexity
    from smlx.models.SmolLM2_135M import load

    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
    results = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset="wikitext",
        split="test"
    )
    print(f"Perplexity: {results['perplexity']:.2f}")
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm

from smlx.utils import (
    ensure_dir,
    format_duration,
    format_memory,
    get_results_dir,
    save_json,
)

# Import datasets at module level for easier mocking in tests
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# Common evaluation datasets
EVAL_DATASETS = {
    "wikitext": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "text_column": "text",
        "description": "WikiText-2 dataset (raw)",
    },
    "wikitext103": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "text_column": "text",
        "description": "WikiText-103 dataset (raw)",
    },
    "ptb": {
        "path": "ptb_text_only",
        "name": None,
        "text_column": "sentence",
        "description": "Penn Treebank dataset",
    },
    "openwebtext": {
        "path": "openwebtext",
        "name": None,
        "text_column": "text",
        "description": "OpenWebText corpus",
    },
}


def load_eval_dataset(
    dataset_name: str,
    tokenizer,
    split: str = "test",
    sequence_length: int = 512,
    num_samples: int = -1,
    seed: int = 42,
) -> mx.array:
    """
    Load and tokenize evaluation dataset.

    Args:
        dataset_name: Name of dataset (see EVAL_DATASETS) or HuggingFace path
        tokenizer: Tokenizer to use
        split: Dataset split (train/validation/test)
        sequence_length: Length of each sequence
        num_samples: Number of samples to use (-1 for all)
        seed: Random seed for sampling

    Returns:
        Tokenized data as mx.array of shape (num_samples, sequence_length)
    """
    if load_dataset is None:
        raise ImportError("datasets is required for evaluation. Install with: pip install datasets")

    # Set random seed
    np.random.seed(seed)

    # Get dataset configuration
    if dataset_name in EVAL_DATASETS:
        config = EVAL_DATASETS[dataset_name]
        dataset_path = config["path"]
        dataset_name_param = config["name"]
        text_column = config["text_column"]
        print(f"Loading dataset: {config['description']}")
    else:
        # Custom dataset path
        dataset_path = dataset_name
        dataset_name_param = None
        text_column = "text"
        print(f"Loading custom dataset: {dataset_name}")

    # Load dataset
    print(f"  Split: {split}")
    print(f"  Sequence length: {sequence_length}")

    if dataset_name_param:
        dataset = load_dataset(dataset_path, dataset_name_param, split=split)
    else:
        dataset = load_dataset(dataset_path, split=split)

    # Tokenize and concatenate all text
    try:
        num_examples = len(dataset)
        print(f"  Tokenizing {num_examples} examples...")
    except TypeError:
        # IterableDataset doesn't support len()
        print("  Tokenizing examples...")
        num_examples = None
    all_tokens = []

    for example in tqdm(dataset, desc="Tokenizing", disable=False, total=num_examples):
        text = example[text_column]
        if text and text.strip():  # Skip empty texts
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

    print(f"  Total tokens: {len(all_tokens):,}")

    # Split into sequences
    # Drop incomplete sequence at end
    num_sequences = len(all_tokens) // sequence_length
    all_tokens = all_tokens[: num_sequences * sequence_length]

    # Reshape into sequences
    data = mx.array(all_tokens).reshape(-1, sequence_length)

    print(f"  Total sequences: {len(data)}")

    # Sample if requested
    if num_samples > 0 and num_samples < len(data):
        # Random sampling
        indices = np.random.permutation(len(data))[:num_samples]
        indices.sort()  # Sort for cache efficiency
        data = data[indices]
        print(f"  Sampled {num_samples} sequences")

    return data


def evaluate_batch(
    model,
    batch: mx.array,
) -> mx.array:
    """
    Evaluate perplexity for a single batch.

    Args:
        model: Model to evaluate
        batch: Tokenized batch of shape (batch_size, sequence_length)

    Returns:
        Token-level losses of shape (batch_size * (sequence_length - 1),)
    """
    # Forward pass: get logits for all tokens except last
    logits = model(batch[:, :-1]).astype(mx.float32)

    # Calculate cross-entropy loss with next tokens
    losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")

    # Evaluate (MLX lazy evaluation)
    mx.eval(losses)

    # Return flattened losses
    return losses.flatten()


def evaluate_perplexity(
    model,
    tokenizer,
    dataset: str = "wikitext",
    split: str = "test",
    batch_size: int = 8,
    sequence_length: int = 512,
    num_samples: int = -1,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate model perplexity on a dataset.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: Dataset name or path
        split: Dataset split
        batch_size: Batch size for evaluation
        sequence_length: Sequence length
        num_samples: Number of samples (-1 for all)
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary with evaluation results:
            - perplexity: Perplexity value
            - std_error: Standard error of perplexity
            - mean_loss: Mean loss
            - tokens_evaluated: Number of tokens
            - samples_evaluated: Number of samples
            - eval_time: Evaluation time in seconds
            - tokens_per_second: Throughput
            - peak_memory_gb: Peak memory usage
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TEXT MODEL EVALUATION - PERPLEXITY")
        print("=" * 70)

    # Load data
    data = load_eval_dataset(
        dataset_name=dataset,
        tokenizer=tokenizer,
        split=split,
        sequence_length=sequence_length,
        num_samples=num_samples,
        seed=seed,
    )

    if verbose:
        print("\nEvaluating perplexity...")
        print(f"  Batch size: {batch_size}")
        print(f"  Number of batches: {(len(data) + batch_size - 1) // batch_size}")

    # Reset memory tracking
    mx.reset_peak_memory()

    # Evaluate
    start_time = time.time()
    all_losses = []

    num_batches = (len(data) + batch_size - 1) // batch_size

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        batch_losses = evaluate_batch(model, batch)
        all_losses.append(batch_losses)

        # Progress
        batch_num = i // batch_size + 1
        if verbose and (batch_num % max(1, num_batches // 20) == 0 or batch_num == num_batches):
            print(f"  Progress: {batch_num}/{num_batches} batches", end="\r")

    if verbose:
        print()  # New line after progress

    # Concatenate all losses
    all_losses = mx.concatenate(all_losses)

    # Calculate statistics
    mean_loss = all_losses.mean().item()
    perplexity = math.exp(mean_loss)

    # Standard error calculation
    std_dev = mx.sqrt(mx.var(all_losses, ddof=1)).item()
    num_tokens = all_losses.size
    standard_error = std_dev / math.sqrt(num_tokens)
    # Delta approximation for standard error of perplexity
    standard_error_ppl = perplexity * standard_error

    eval_time = time.time() - start_time
    tokens_evaluated = data.shape[0] * (data.shape[1] - 1)  # B * (L - 1)
    peak_memory = mx.get_peak_memory() / 1e9

    # Compile results
    results = {
        "perplexity": perplexity,
        "std_error": standard_error_ppl,
        "mean_loss": mean_loss,
        "std_dev": std_dev,
        "tokens_evaluated": int(tokens_evaluated),
        "samples_evaluated": len(data),
        "eval_time": eval_time,
        "tokens_per_second": tokens_evaluated / eval_time,
        "peak_memory_gb": peak_memory,
        "dataset": dataset,
        "split": split,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "seed": seed,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Perplexity:        {perplexity:.3f} ± {standard_error_ppl:.3f}")
        print(f"Mean loss:         {mean_loss:.4f}")
        print(f"Std deviation:     {std_dev:.4f}")
        print("\nPerformance:")
        print(f"  Evaluation time: {format_duration(eval_time)}")
        print(f"  Tokens/sec:      {results['tokens_per_second']:,.0f}")
        print(f"  Peak memory:     {format_memory(peak_memory * 1e9)}")
        print("\nDataset:")
        print(f"  Samples:         {len(data):,}")
        print(f"  Tokens:          {tokens_evaluated:,}")
        print("=" * 70)

    return results


def list_datasets():
    """Print available evaluation datasets and return list of dataset names."""
    print("\nAvailable Evaluation Datasets:")
    print("=" * 70)
    for name, config in EVAL_DATASETS.items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Path: {config['path']}")
        if config["name"]:
            print(f"  Config: {config['name']}")
    print("\nYou can also use any HuggingFace dataset path.")
    print("=" * 70)
    return list(EVAL_DATASETS.keys())


def main():
    """Main entry point for text evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate text-only models using perplexity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on WikiText-2
  python -m smlx.evals.text_eval --model mlx-community/SmolLM2-135M-Instruct

  # Evaluate on WikiText-103 with custom parameters
  python -m smlx.evals.text_eval \\
      --model mlx-community/SmolLM2-360M-Instruct \\
      --dataset wikitext103 \\
      --batch-size 16 \\
      --sequence-length 1024

  # Quick test with limited samples
  python -m smlx.evals.text_eval \\
      --model mlx-community/SmolLM2-135M-Instruct \\
      --num-samples 100 \\
      --verbose

  # List available datasets
  python -m smlx.evals.text_eval --list-datasets
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model path or HuggingFace model ID",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name (wikitext, wikitext103, ptb, openwebtext) or HuggingFace path (default: wikitext)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Sequence length (default: 512)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to use, -1 for all (default: -1)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path for results (JSON)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available evaluation datasets and exit",
    )

    args = parser.parse_args()

    # List datasets if requested
    if args.list_datasets:
        list_datasets()
        return

    # Validate required arguments
    if not args.model:
        parser.error("--model is required (unless using --list-datasets)")

    # Set random seeds
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # Load model - try SmolLM2_135M first, fall back to generic
    if args.verbose:
        print(f"Loading model: {args.model}")

    try:
        from smlx.models import mlx_backend

        bm = mlx_backend.load(args.model)
        model, tokenizer = bm.model, bm.processor
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        print("Ensure the model is a valid MLX language model.", file=sys.stderr)
        sys.exit(1)

    # Count parameters
    def count_params(tree):
        """Recursively count parameters in a nested dictionary structure."""
        count = 0
        if isinstance(tree, dict):
            for v in tree.values():
                count += count_params(v)
        elif hasattr(tree, "size"):
            # MLX array
            count += tree.size
        return count

    total_params = count_params(model.parameters())
    if args.verbose:
        print(f"Model parameters: {total_params / 1e6:.1f}M")

    # Evaluate perplexity
    results = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_samples=args.num_samples,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Add model info to results
    results["model"] = args.model
    results["model_parameters"] = total_params

    # Save results if output specified
    if args.output:
        ensure_dir(args.output.parent)
        save_json(results, args.output)
        print(f"\nResults saved to {args.output}")
    else:
        # Save to default location
        output_dir = get_results_dir()
        model_name = args.model.replace("/", "_")
        output_file = output_dir / f"text_eval_{model_name}_{args.dataset}.json"
        save_json(results, output_file)
        if args.verbose:
            print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
