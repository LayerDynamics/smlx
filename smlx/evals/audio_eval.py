"""
Audio model evaluation metrics.

Provides Word Error Rate (WER) and Character Error Rate (CER) metrics
for evaluating automatic speech recognition (ASR) systems.

WER and CER are standard metrics that measure edit distance (insertions,
deletions, substitutions) between predicted and reference transcriptions.

Usage:
    # Basic usage
    from smlx.evals.audio_eval import compute_wer, compute_cer

    reference = "hello world"
    hypothesis = "hello word"

    wer = compute_wer(reference, hypothesis)
    cer = compute_cer(reference, hypothesis)

    print(f"WER: {wer:.2%}, CER: {cer:.2%}")

    # Batch evaluation
    from smlx.evals.audio_eval import AudioEvaluator

    evaluator = AudioEvaluator()
    results = evaluator.evaluate_batch(references, hypotheses)
    print(f"Average WER: {results['wer']:.2%}")
    print(f"Average CER: {results['cer']:.2%}")

Requirements:
    pip install jiwer  # For WER/CER computation
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    logging.warning(
        "jiwer not installed. Install with: pip install jiwer\n"
        "Audio evaluation metrics will not be available."
    )


@dataclass
class AudioEvalResult:
    """Result of audio evaluation.

    Attributes:
        wer: Word Error Rate (0.0 = perfect, 1.0 = completely wrong)
        cer: Character Error Rate (0.0 = perfect, 1.0 = completely wrong)
        insertions: Number of word insertions
        deletions: Number of word deletions
        substitutions: Number of word substitutions
        num_words: Total number of words in reference
        num_chars: Total number of characters in reference
    """

    wer: float
    cer: float
    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0
    num_words: int = 0
    num_chars: int = 0

    def __str__(self) -> str:
        return (
            f"WER: {self.wer:.2%} | CER: {self.cer:.2%}\n"
            f"Errors: {self.insertions} ins, {self.deletions} del, "
            f"{self.substitutions} sub\n"
            f"Words: {self.num_words} | Chars: {self.num_chars}"
        )


def compute_wer(
    reference: Union[str, list[str]],
    hypothesis: Union[str, list[str]],
    *,
    normalize: bool = True,
) -> float:
    """Compute Word Error Rate (WER).

    WER = (Insertions + Deletions + Substitutions) / Total Words

    Args:
        reference: Ground truth transcription(s)
        hypothesis: Predicted transcription(s)
        normalize: Whether to normalize text (lowercase, remove punctuation)

    Returns:
        Word Error Rate as a float between 0.0 and 1.0+
        (can exceed 1.0 if hypothesis is much longer than reference)

    Raises:
        ImportError: If jiwer is not installed

    Example:
        >>> compute_wer("hello world", "hello word")
        0.5  # 1 substitution out of 2 words
        >>> compute_wer("the cat sat", "cat sat")
        0.333  # 1 deletion out of 3 words
    """
    if not HAS_JIWER:
        raise ImportError(
            "jiwer is required for WER computation. Install with: pip install jiwer"
        )

    if normalize:
        transform = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ]
        )
        reference = transform(reference)
        hypothesis = transform(hypothesis)

    return jiwer.wer(reference, hypothesis)


def compute_cer(
    reference: Union[str, list[str]],
    hypothesis: Union[str, list[str]],
    *,
    normalize: bool = True,
) -> float:
    """Compute Character Error Rate (CER).

    CER = (Insertions + Deletions + Substitutions) / Total Characters

    Args:
        reference: Ground truth transcription(s)
        hypothesis: Predicted transcription(s)
        normalize: Whether to normalize text (lowercase, remove punctuation)

    Returns:
        Character Error Rate as a float between 0.0 and 1.0+

    Raises:
        ImportError: If jiwer is not installed

    Example:
        >>> compute_cer("hello", "helo")
        0.2  # 1 deletion out of 5 characters
        >>> compute_cer("cat", "cut")
        0.333  # 1 substitution out of 3 characters
    """
    if not HAS_JIWER:
        raise ImportError(
            "jiwer is required for CER computation. Install with: pip install jiwer"
        )

    if normalize:
        transform = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ]
        )
        reference = transform(reference)
        hypothesis = transform(hypothesis)

    return jiwer.cer(reference, hypothesis)


def compute_metrics(
    reference: Union[str, list[str]],
    hypothesis: Union[str, list[str]],
    *,
    normalize: bool = True,
) -> AudioEvalResult:
    """Compute detailed ASR metrics including WER, CER, and error breakdown.

    Args:
        reference: Ground truth transcription(s)
        hypothesis: Predicted transcription(s)
        normalize: Whether to normalize text (lowercase, remove punctuation)

    Returns:
        AudioEvalResult with WER, CER, and error counts

    Raises:
        ImportError: If jiwer is not installed

    Example:
        >>> result = compute_metrics("hello world", "hello word")
        >>> print(result)
        WER: 50.00% | CER: 9.09%
        Errors: 0 ins, 0 del, 1 sub
        Words: 2 | Chars: 11
    """
    if not HAS_JIWER:
        raise ImportError(
            "jiwer is required for metrics computation. Install with: pip install jiwer"
        )

    if normalize:
        transform = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ]
        )
        reference = transform(reference)
        hypothesis = transform(hypothesis)

    # Compute WER with detailed measures
    wer_output = jiwer.process_words(reference, hypothesis)
    wer = wer_output.wer

    # Compute CER
    cer = jiwer.cer(reference, hypothesis)

    return AudioEvalResult(
        wer=wer,
        cer=cer,
        insertions=wer_output.insertions,
        deletions=wer_output.deletions,
        substitutions=wer_output.substitutions,
        num_words=wer_output.hits + wer_output.deletions + wer_output.substitutions,
        num_chars=len(reference) if isinstance(reference, str) else sum(len(r) for r in reference),
    )


class AudioEvaluator:
    """Audio model evaluator for batch evaluation.

    Provides utilities for evaluating ASR models on datasets with
    reference transcriptions.

    Attributes:
        normalize: Whether to normalize text before comparison
        results: List of evaluation results for each sample
    """

    def __init__(self, normalize: bool = True):
        """Initialize evaluator.

        Args:
            normalize: Whether to normalize text (lowercase, remove punctuation)
        """
        if not HAS_JIWER:
            raise ImportError(
                "jiwer is required for AudioEvaluator. Install with: pip install jiwer"
            )

        self.normalize = normalize
        self.results: list[AudioEvalResult] = []

    def evaluate(
        self, reference: str, hypothesis: str
    ) -> AudioEvalResult:
        """Evaluate single prediction.

        Args:
            reference: Ground truth transcription
            hypothesis: Predicted transcription

        Returns:
            AudioEvalResult with metrics
        """
        result = compute_metrics(reference, hypothesis, normalize=self.normalize)
        self.results.append(result)
        return result

    def evaluate_batch(
        self,
        references: list[str],
        hypotheses: list[str],
        verbose: bool = False,
    ) -> dict[str, float]:
        """Evaluate batch of predictions.

        Args:
            references: List of ground truth transcriptions
            hypotheses: List of predicted transcriptions
            verbose: Whether to print progress

        Returns:
            Dictionary with average metrics

        Example:
            >>> evaluator = AudioEvaluator()
            >>> refs = ["hello world", "the cat sat"]
            >>> hyps = ["hello word", "cat sat"]
            >>> results = evaluator.evaluate_batch(refs, hyps)
            >>> print(f"Average WER: {results['wer']:.2%}")
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Number of references ({len(references)}) and hypotheses "
                f"({len(hypotheses)}) must match"
            )

        self.results = []
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            if verbose and i % 10 == 0:
                print(f"Evaluating {i}/{len(references)}")
            self.evaluate(ref, hyp)

        return self.summary()

    def summary(self) -> dict[str, float]:
        """Get summary statistics across all evaluations.

        Returns:
            Dictionary with average WER, CER, and total error counts

        Example:
            >>> evaluator = AudioEvaluator()
            >>> # ... evaluate samples ...
            >>> summary = evaluator.summary()
            >>> print(f"Average WER: {summary['wer']:.2%}")
        """
        if not self.results:
            return {
                "wer": 0.0,
                "cer": 0.0,
                "insertions": 0,
                "deletions": 0,
                "substitutions": 0,
                "num_samples": 0,
            }

        # Compute averages
        avg_wer = sum(r.wer for r in self.results) / len(self.results)
        avg_cer = sum(r.cer for r in self.results) / len(self.results)

        # Sum totals
        total_insertions = sum(r.insertions for r in self.results)
        total_deletions = sum(r.deletions for r in self.results)
        total_substitutions = sum(r.substitutions for r in self.results)
        total_words = sum(r.num_words for r in self.results)

        return {
            "wer": avg_wer,
            "cer": avg_cer,
            "insertions": total_insertions,
            "deletions": total_deletions,
            "substitutions": total_substitutions,
            "total_words": total_words,
            "num_samples": len(self.results),
        }

    def save_results(self, output_path: Union[str, Path]):
        """Save detailed results to JSON file.

        Args:
            output_path: Path to save results
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            "summary": self.summary(),
            "samples": [
                {
                    "wer": r.wer,
                    "cer": r.cer,
                    "insertions": r.insertions,
                    "deletions": r.deletions,
                    "substitutions": r.substitutions,
                    "num_words": r.num_words,
                    "num_chars": r.num_chars,
                }
                for r in self.results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {output_path}")

    def reset(self):
        """Reset evaluator state."""
        self.results = []


def evaluate_whisper(
    model,
    tokenizer,
    audio_files: list[Union[str, Path]],
    references: list[str],
    *,
    verbose: bool = True,
    **transcribe_kwargs,
) -> dict[str, float]:
    """Evaluate Whisper model on dataset.

    Convenience function for evaluating Whisper models.

    Args:
        model: Whisper model
        tokenizer: Whisper tokenizer
        audio_files: List of audio file paths
        references: List of reference transcriptions
        verbose: Whether to print progress
        **transcribe_kwargs: Additional arguments for transcribe()

    Returns:
        Dictionary with evaluation metrics

    Example:
        >>> from smlx.models.Whisper_tiny import load
        >>> from smlx.evals.audio_eval import evaluate_whisper
        >>>
        >>> model, tokenizer = load()
        >>> audio_files = ["audio1.wav", "audio2.wav"]
        >>> references = ["hello world", "the cat sat"]
        >>>
        >>> results = evaluate_whisper(
        ...     model, tokenizer, audio_files, references,
        ...     language="en", verbose=True
        ... )
        >>> print(f"WER: {results['wer']:.2%}")
    """
    import mlx_whisper

    del tokenizer  # mlx-whisper loads/caches by repo; no separate tokenizer object
    repo = getattr(model, "repo", model)

    if len(audio_files) != len(references):
        raise ValueError(
            f"Number of audio files ({len(audio_files)}) and references "
            f"({len(references)}) must match"
        )

    evaluator = AudioEvaluator()
    hypotheses = []

    for i, audio_file in enumerate(audio_files):
        if verbose:
            print(f"Transcribing {i + 1}/{len(audio_files)}: {audio_file}")

        result = mlx_whisper.transcribe(
            str(audio_file), path_or_hf_repo=repo, **transcribe_kwargs
        )
        hypotheses.append(result["text"])

    results = evaluator.evaluate_batch(references, hypotheses, verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"WER: {results['wer']:.2%}")
        print(f"CER: {results['cer']:.2%}")
        print(f"Errors: {results['insertions']} ins, {results['deletions']} del, "
              f"{results['substitutions']} sub")
        print(f"Total words: {results['total_words']}")
        print(f"Samples: {results['num_samples']}")
        print("=" * 70)

    return results
