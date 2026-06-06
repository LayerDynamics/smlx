"""
LM Evaluation Harness Integration for SMLX

Provides integration with EleutherAI's lm-evaluation-harness for comprehensive
language model evaluation across 200+ tasks.

Tasks include:
- MMLU (Massive Multitask Language Understanding)
- HellaSwag (commonsense reasoning)
- ARC (AI2 Reasoning Challenge)
- TruthfulQA (truthfulness)
- Winogrande (commonsense reasoning)
- GSM8K (math word problems)
- And many more...

Usage:
    # CLI
    python -m smlx.evals.lm_harness \
        --model mlx-community/SmolLM2-135M-Instruct \
        --tasks hellaswag winogrande arc_easy \
        --output results/lm_harness_results.json

    # Programmatic
    from smlx.evals.lm_harness import SMLXLM, run_evaluation

    results = run_evaluation(
        model_path="mlx-community/SmolLM2-135M-Instruct",
        tasks=["hellaswag", "winogrande"],
        num_fewshot=5,
        limit=100,
    )

Requirements:
    pip install lm-eval>=0.4.0

Reference:
    https://github.com/EleutherAI/lm-evaluation-harness
"""

import argparse
import collections
import json
import logging
from pathlib import Path
from typing import Any, Optional, cast

import lm_eval
import mlx.core as mx
import mlx.nn as nn
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models import huggingface
from tqdm import tqdm

# Import model functions at module level for testing/mocking
from smlx.models.SmolLM2_135M import load
from smlx.models.SmolLM2_135M.generate import generate

DEFAULT_MAX_TOKENS = 2048

# Popular evaluation tasks with metadata
POPULAR_TASKS = {
    "hellaswag": {
        "description": "Commonsense reasoning",
        "category": "Common Sense Reasoning",
        "num_fewshot": 10,
    },
    "winogrande": {
        "description": "Commonsense reasoning",
        "category": "Common Sense Reasoning",
        "num_fewshot": 5,
    },
    "arc_easy": {
        "description": "Science questions (easy)",
        "category": "Knowledge & QA",
        "num_fewshot": 25,
    },
    "arc_challenge": {
        "description": "Science questions (hard)",
        "category": "Knowledge & QA",
        "num_fewshot": 25,
    },
    "mmlu": {
        "description": "Massive multitask understanding",
        "category": "Knowledge & QA",
        "num_fewshot": 5,
    },
    "truthfulqa_mc": {
        "description": "Truthfulness",
        "category": "Truthfulness",
        "num_fewshot": 0,
    },
    "gsm8k": {
        "description": "Math word problems",
        "category": "Mathematical Reasoning",
        "num_fewshot": 5,
    },
    "piqa": {
        "description": "Physical commonsense",
        "category": "Common Sense Reasoning",
        "num_fewshot": 0,
    },
    "boolq": {
        "description": "Boolean questions",
        "category": "Reading Comprehension",
        "num_fewshot": 0,
    },
    "lambada_openai": {
        "description": "Reading comprehension",
        "category": "Reading Comprehension",
        "num_fewshot": 0,
    },
}


def _pad_inputs(inputs):
    """Pad inputs to the same length for batch processing."""
    lengths = [len(x) for x in inputs]
    maxlen = max(lengths)
    padded = []
    for x in inputs:
        padded.append(list(x) + [0] * (maxlen - len(x)))

    return mx.array(padded), mx.array(lengths)


def _rstrip_until(s: str, untils: list[str]) -> str:
    """Truncate string at first occurrence of any stop sequence."""
    min_idx = len(s)
    for until in untils:
        idx = s.find(until)
        if idx >= 0 and idx < min_idx:
            min_idx = idx
    return s[:min_idx]


@register_model("smlx")
class SMLXLM(LM):
    """
    SMLX Language Model wrapper for lm-evaluation-harness.

    Implements the required LM interface for evaluating SMLX models
    using the standardized lm-evaluation-harness framework.
    """

    # Inherit tokenizer naming from HuggingFace implementation
    tokenizer_name = huggingface.HFLM.tokenizer_name

    def __init__(
        self,
        model_path: str,
        max_tokens: Optional[int] = None,
        batch_size: int = 8,
        trust_remote_code: bool = False,
    ):
        """
        Initialize SMLX LM for evaluation.

        Args:
            model_path: Path to model or HuggingFace model ID
            max_tokens: Maximum context length (default: 2048)
            batch_size: Batch size for evaluation (default: 8)
            trust_remote_code: Trust remote code for tokenizer
        """
        super().__init__()

        self._model, self.tokenizer = load(model_path)
        self._max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self._batch_size = batch_size
        self._model_path = model_path

        logging.info(f"Loaded SMLX model: {model_path}")
        logging.info(f"Max tokens: {self._max_tokens}")
        logging.info(f"Batch size: {self._batch_size}")

    def tok_encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)

    def tok_decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(tokens)

    @property
    def eot_token_id(self) -> int:
        """End-of-text token ID."""
        return self.tokenizer.eos_token_id

    def _batch_score(self, tokens: mx.array, lengths: mx.array) -> list[float]:
        """Score a batch of token sequences for compatibility."""
        # This method is here for compatibility with some tests
        # Convert to list format and use _score_batch
        inputs: list[list[int]] = [
            cast(list[int], tokens[i, : lengths[i].item()].tolist()) for i in range(tokens.shape[0])
        ]
        scores, _, _ = self._score_batch(inputs)
        # Return list of total scores per sequence
        return [float(scores[i, : lengths[i].item() - 1].sum().item()) for i in range(len(inputs))]

    def _get_request_args(self, req):
        """Extract arguments from request (handle both tuple and object with .args)."""
        if isinstance(req, tuple):
            return req
        return req.args

    def _tokenize(self, texts: list[str]) -> list[tuple]:
        """Tokenize a list of texts."""
        return [tuple(self.tokenizer.encode(t)) for t in texts]

    def _score_batch(
        self,
        inputs: list[list[int]],
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Score a batch of token sequences.

        Returns:
            scores: Log probabilities for each token
            lengths: Actual length of each sequence
            is_greedy: Whether each token matches greedy sampling
        """
        inputs_padded, lengths = _pad_inputs(inputs)
        inputs_arr = inputs_padded[:, :-1]  # Remove last token for input
        targets_arr = inputs_padded[:, 1:]  # Remove first token for targets

        # Forward pass
        logits = self._model(inputs_arr).astype(mx.float32)
        log_probs = nn.log_softmax(logits, axis=-1)

        # Get log probability of actual targets
        target_logprobs = mx.take_along_axis(
            log_probs,
            targets_arr[..., mx.newaxis],
            axis=-1,
        )[..., 0]

        # Check if targets match greedy sampling
        is_greedy = targets_arr == mx.argmax(logits, axis=-1)

        # Mask padding tokens
        mask = mx.arange(target_logprobs.shape[-1]) < (lengths - 1)[:, mx.newaxis]
        is_greedy = mx.where(mask, is_greedy, False)

        mx.eval(target_logprobs, is_greedy)
        mx.clear_cache()

        return target_logprobs, lengths, is_greedy

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """
        Compute log-likelihood of generating continuations from contexts.

        This is the primary method used by most evaluation tasks.

        Args:
            requests: List of Instance objects with (context, continuation) pairs

        Returns:
            List of (log_prob, is_greedy) tuples
        """
        logging.info(f"Computing loglikelihood for {len(requests)} requests")

        # Group by context to reuse prompt processing
        context_groups = collections.defaultdict(list)
        for idx, req in enumerate(requests):
            context, continuation = self._get_request_args(req)
            context_groups[context].append((idx, continuation))

        results: list[tuple[float, bool]] = [(0.0, False)] * len(requests)

        for context, continuations in tqdm(
            context_groups.items(),
            desc="Processing contexts",
            disable=logging.getLogger().level > logging.INFO,
        ):
            # Tokenize context and full sequences
            context_tokens = self._tokenize([context])[0]
            full_sequences = self._tokenize([context + cont for _, cont in continuations])

            # Handle context length truncation
            max_seq_len = max(len(seq) for seq in full_sequences)
            if max_seq_len > self._max_tokens:
                # Truncate from the left
                truncation = max_seq_len - self._max_tokens
                context_tokens = context_tokens[truncation:]

                # If entire context was truncated, return -inf
                if len(context_tokens) == 0:
                    for idx, _ in continuations:
                        results[idx] = (-float("inf"), False)
                    continue

            # Score each full sequence
            for idx, continuation in continuations:
                full_seq = self._tokenize([context + continuation])[0]
                continuation_tokens = full_seq[len(context_tokens) :]

                if len(continuation_tokens) == 0:
                    results[idx] = (0.0, True)
                    continue

                # Score the continuation tokens
                scores, _, is_greedy_arr = self._score_batch([list(full_seq)])

                # Sum scores for continuation tokens only
                start_idx = len(context_tokens) - 1
                end_idx = len(full_seq) - 1
                cont_scores = scores[0, start_idx:end_idx]
                cont_greedy = is_greedy_arr[0, start_idx:end_idx]

                total_score = float(mx.sum(cont_scores).item())
                all_greedy = bool(mx.all(cont_greedy).item())

                results[idx] = (total_score, all_greedy)

        return results

    def loglikelihood_rolling(self, requests) -> list[float]:
        """
        Compute full log-likelihood for perplexity evaluation.

        Used for evaluating perplexity on full documents.

        Args:
            requests: List of Instance objects with (string,) tuples

        Returns:
            List of log probabilities
        """
        logging.info(f"Computing rolling loglikelihood for {len(requests)} sequences")

        inputs = self._tokenize([self._get_request_args(req)[0] for req in requests])
        all_scores: list[float] = []

        for i in tqdm(
            range(0, len(inputs), self._batch_size),
            desc="Processing batches",
            disable=logging.getLogger().level > logging.INFO,
        ):
            batch = inputs[i : i + self._batch_size]
            scores, lengths, _ = self._score_batch([list(seq) for seq in batch])

            # Mask and sum scores for each sequence
            mask = mx.arange(scores.shape[-1]) < (lengths - 1)[:, mx.newaxis]
            seq_scores = (mask * scores).sum(axis=-1)
            # Cast to list[float] for type checker
            scores_list: list[float] = cast(list[float], seq_scores.tolist())
            all_scores.extend(scores_list)

        return all_scores

    def generate_until(self, requests) -> list[str]:
        """
        Generate text until stopping criteria are met.

        Used for generation tasks like question answering.

        Args:
            requests: List of Instance objects with (context, generation_kwargs) pairs

        Returns:
            List of generated strings
        """
        logging.info(f"Generating continuations for {len(requests)} requests")

        contexts, options = zip(*[self._get_request_args(req) for req in requests])
        completions = []

        for context, opt in tqdm(
            zip(contexts, options),
            desc="Generating",
            disable=logging.getLogger().level > logging.INFO,
        ):
            # Get generation parameters
            max_gen_tokens = opt.get("max_gen_tokens", 256)
            until = opt.get("until", [])
            do_sample = opt.get("do_sample", False)
            temperature = opt.get("temperature", 0.0) if do_sample else 0.0

            # Generate
            try:
                output = generate(
                    model=self._model,
                    tokenizer=self.tokenizer,
                    prompt=context,
                    max_tokens=max_gen_tokens,
                    temperature=temperature,
                    verbose=False,
                )

                # Extract only the generated part (remove prompt)
                continuation = output[len(context) :]

                # Apply stopping criteria
                continuation = _rstrip_until(continuation, until)

                completions.append(continuation)

            except Exception as e:
                logging.warning(f"Generation failed: {e}")
                completions.append("")

        return completions


def list_tasks():
    """List available evaluation tasks."""
    print("\n" + "=" * 70)
    print("AVAILABLE LM EVALUATION HARNESS TASKS")
    print("=" * 70)
    print("\nPopular LM Evaluation Tasks:")

    # Group by category
    categories = {}
    for task_name, task_info in POPULAR_TASKS.items():
        category = task_info["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append((task_name, task_info["description"]))

    for category, tasks in sorted(categories.items()):
        print(f"\n{category}:")
        for task, description in tasks:
            print(f"  {task:<20} - {description}")

    print("\n" + "-" * 70)
    print("To see all tasks, run:")
    print("  lm_eval --tasks list")
    print("\nFor task details:")
    print("  lm_eval --tasks <task_name> --verbosity DEBUG")
    print("=" * 70)


def run_evaluation(
    model_path: str,
    tasks: list[str],
    output_path: Optional[Path] = None,
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    batch_size: int = 8,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run LM evaluation harness on SMLX model.

    Args:
        model_path: Path to model or HuggingFace ID
        tasks: List of task names to evaluate
        output_path: Path to save results JSON
        num_fewshot: Number of few-shot examples (default: task-specific)
        limit: Limit number of examples per task
        batch_size: Batch size for evaluation
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary of evaluation results

    Example:
        >>> results = run_evaluation(
        ...     model_path="mlx-community/SmolLM2-135M-Instruct",
        ...     tasks=["hellaswag", "winogrande"],
        ...     num_fewshot=5,
        ...     limit=100,
        ... )
        >>> print(f"HellaSwag: {results['results']['hellaswag']['acc']:.2%}")
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Set seed
    mx.random.seed(seed)

    # Create model wrapper
    lm = SMLXLM(
        model_path=model_path,
        batch_size=batch_size,
    )

    # Run evaluation
    # Note: Pylance may show false errors due to type stub issues
    # The parameters below are correct as verified by runtime signature inspection
    results = lm_eval.simple_evaluate(  # pyright: ignore[reportCallIssue, reportArgumentType]
        model=lm,  # pyright: ignore[reportCallIssue]
        tasks=tasks,  # pyright: ignore[reportCallIssue, reportArgumentType]
        num_fewshot=num_fewshot,  # pyright: ignore[reportCallIssue]
        limit=limit,
        batch_size=batch_size,  # pyright: ignore[reportCallIssue]
    )

    # Check if results are valid
    if results is None:
        raise RuntimeError("Evaluation failed: results is None")

    # Save results if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_data = results.get("results", {})
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)
        if verbose:
            print(f"\nResults saved to {output_path}")

    return results


def main():
    """CLI entry point for LM evaluation harness."""
    parser = argparse.ArgumentParser(
        description="Evaluate SMLX models using lm-evaluation-harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available tasks
  python -m smlx.evals.lm_harness --list-tasks

  # Run single task
  python -m smlx.evals.lm_harness \\
      --model mlx-community/SmolLM2-135M-Instruct \\
      --tasks hellaswag

  # Run multiple tasks with few-shot
  python -m smlx.evals.lm_harness \\
      --model mlx-community/SmolLM2-135M-Instruct \\
      --tasks hellaswag winogrande arc_easy \\
      --num-fewshot 5 \\
      --output results/lm_harness.json

  # Quick test with limited examples
  python -m smlx.evals.lm_harness \\
      --model mlx-community/SmolLM2-135M-Instruct \\
      --tasks hellaswag \\
      --limit 100

Popular tasks:
  hellaswag, winogrande, arc_easy, arc_challenge, mmlu,
  truthfulqa_mc, gsm8k, piqa, boolq, lambada_openai
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model path or HuggingFace ID",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Tasks to evaluate (e.g., hellaswag winogrande)",
    )

    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List popular evaluation tasks",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON)",
    )

    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (default: task-specific)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for testing)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Handle list tasks
    if args.list_tasks:
        list_tasks()
        return

    # Validate required arguments
    if not args.model or not args.tasks:
        parser.error("--model and --tasks are required (or use --list-tasks)")

    verbose = args.verbose and not args.quiet

    # Run evaluation
    results = run_evaluation(
        model_path=args.model,
        tasks=args.tasks,
        output_path=args.output,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=verbose,
    )

    # Print results
    if verbose and results:
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        results_data = results.get("results", {})
        for task_name, task_results in results_data.items():
            print(f"\n{task_name}:")
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, float):
                    print(f"  {metric_name}: {metric_value:.4f}")
                elif not metric_name.endswith("_stderr"):
                    print(f"  {metric_name}: {metric_value}")
        print("=" * 70)


if __name__ == "__main__":
    main()
