#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
VLM Diagnostics Utilities.

Tools for debugging and validating vision-language model outputs.
"""

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def log_vision_features(
    vision_features: mx.array,
    label: str = "vision_features",
    expected_mean_range: tuple[float, float] = (-2.0, 2.0),
    expected_std_range: tuple[float, float] = (0.5, 3.0),
) -> dict[str, float]:
    """
    Log statistics about vision features for debugging.

    Args:
        vision_features: Vision feature tensor to analyze
        label: Label for logging messages
        expected_mean_range: (min, max) for expected mean value
        expected_std_range: (min, max) for expected std value

    Returns:
        Dictionary with statistics (mean, std, min, max)
    """
    # Compute statistics
    mean_val = float(mx.mean(vision_features))
    std_val = float(mx.std(vision_features))
    min_val = float(mx.min(vision_features))
    max_val = float(mx.max(vision_features))

    stats = {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
    }

    # Log shape and statistics
    logger.debug(f"{label} - shape: {vision_features.shape}")
    logger.debug(f"{label} - dtype: {vision_features.dtype}")
    logger.debug(f"{label} - mean: {mean_val:.4f}, std: {std_val:.4f}")
    logger.debug(f"{label} - range: [{min_val:.4f}, {max_val:.4f}]")

    # Validate against expected ranges
    if not (expected_mean_range[0] <= mean_val <= expected_mean_range[1]):
        logger.warning(
            f"{label} mean outside expected range {expected_mean_range}: {mean_val:.4f}"
        )

    if not (expected_std_range[0] <= std_val <= expected_std_range[1]):
        logger.warning(
            f"{label} std outside expected range {expected_std_range}: {std_val:.4f}"
        )

    return stats


def log_embedding_comparison(
    vision_embeds: mx.array,
    text_embeds: mx.array,
    label: str = "embeddings",
) -> dict[str, dict[str, float]]:
    """
    Compare vision and text embedding statistics.

    Args:
        vision_embeds: Vision embeddings
        text_embeds: Text embeddings
        label: Label for logging messages

    Returns:
        Dictionary with statistics for both embeddings
    """
    vision_stats = {
        "mean": float(mx.mean(vision_embeds)),
        "std": float(mx.std(vision_embeds)),
        "min": float(mx.min(vision_embeds)),
        "max": float(mx.max(vision_embeds)),
    }

    text_stats = {
        "mean": float(mx.mean(text_embeds)),
        "std": float(mx.std(text_embeds)),
        "min": float(mx.min(text_embeds)),
        "max": float(mx.max(text_embeds)),
    }

    logger.debug(f"{label} - Vision: shape={vision_embeds.shape}, dtype={vision_embeds.dtype}")
    logger.debug(f"{label} - Vision: mean={vision_stats['mean']:.4f}, std={vision_stats['std']:.4f}")
    logger.debug(f"{label} - Text: shape={text_embeds.shape}, dtype={text_embeds.dtype}")
    logger.debug(f"{label} - Text: mean={text_stats['mean']:.4f}, std={text_stats['std']:.4f}")

    # Check for scale mismatch
    scale_ratio = vision_stats["std"] / text_stats["std"] if text_stats["std"] > 0 else float("inf")
    if scale_ratio > 10.0 or scale_ratio < 0.1:
        logger.warning(
            f"{label} - Large scale mismatch: vision_std/text_std = {scale_ratio:.2f}"
        )

    return {"vision": vision_stats, "text": text_stats}


def log_logits_distribution(
    logits: mx.array,
    top_k: int = 10,
    label: str = "logits",
) -> dict[str, any]:
    """
    Log logits distribution for debugging generation issues.

    Args:
        logits: Logits tensor (typically last token logits)
        top_k: Number of top tokens to show
        label: Label for logging messages

    Returns:
        Dictionary with logits statistics and top tokens
    """
    # Compute statistics
    mean_val = float(mx.mean(logits))
    std_val = float(mx.std(logits))
    min_val = float(mx.min(logits))
    max_val = float(mx.max(logits))

    # Get top-k tokens
    # Flatten logits to 1D if needed
    logits_flat = logits.reshape(-1) if len(logits.shape) > 1 else logits
    probs = mx.softmax(logits_flat, axis=-1)

    top_k_indices = mx.argpartition(-logits_flat, kth=top_k)[:top_k]
    top_k_logits = logits_flat[top_k_indices]
    top_k_probs = probs[top_k_indices]

    logger.debug(f"{label} - shape: {logits.shape}")
    logger.debug(f"{label} - mean: {mean_val:.4f}, std: {std_val:.4f}")
    logger.debug(f"{label} - range: [{min_val:.4f}, {max_val:.4f}]")
    logger.debug(f"{label} - Top {top_k} token IDs: {top_k_indices.tolist()}")

    # Format probabilities properly - convert to list and format each value
    top_k_probs_list = [float(p) for p in top_k_probs.tolist()]
    top_k_probs_formatted = ', '.join(f'{p:.4f}' for p in top_k_probs_list)
    logger.debug(f"{label} - Top {top_k} probabilities: [{top_k_probs_formatted}]")

    # Check for degenerate distributions
    max_prob = float(mx.max(probs))
    if max_prob > 0.99:
        logger.warning(f"{label} - Highly peaked distribution (max_prob={max_prob:.4f})")

    entropy = -float(mx.sum(probs * mx.log(probs + 1e-10)))
    logger.debug(f"{label} - Entropy: {entropy:.4f}")

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "top_k_indices": top_k_indices.tolist(),
        "top_k_probs": top_k_probs.tolist(),
        "entropy": entropy,
    }


def log_attention_mask(
    mask: mx.array,
    label: str = "attention_mask",
) -> dict[str, any]:
    """
    Log attention mask statistics.

    Args:
        mask: Attention mask tensor
        label: Label for logging messages

    Returns:
        Dictionary with mask statistics
    """
    # Count unmasked positions
    if mask.dtype == mx.bool_:
        num_unmasked = int(mx.sum(mask))
        total_positions = mask.size
    else:
        # Assume -inf for masked, finite for unmasked
        num_unmasked = int(mx.sum(mx.isfinite(mask)))
        total_positions = mask.size

    pct_unmasked = 100.0 * num_unmasked / total_positions if total_positions > 0 else 0.0

    logger.debug(f"{label} - shape: {mask.shape}")
    logger.debug(f"{label} - dtype: {mask.dtype}")
    logger.debug(f"{label} - unmasked: {num_unmasked}/{total_positions} ({pct_unmasked:.1f}%)")

    return {
        "shape": mask.shape,
        "num_unmasked": num_unmasked,
        "total_positions": total_positions,
        "pct_unmasked": pct_unmasked,
    }


def compare_with_reference(
    smlx_output: str,
    reference_output: Optional[str] = None,
    label: str = "output",
) -> dict[str, any]:
    """
    Compare SMLX output with reference implementation output.

    Args:
        smlx_output: Output from SMLX model
        reference_output: Output from reference implementation (optional)
        label: Label for logging messages

    Returns:
        Dictionary with comparison metrics
    """
    smlx_len = len(smlx_output)
    smlx_unique_tokens = len(set(smlx_output.split()))
    smlx_total_tokens = len(smlx_output.split())

    logger.debug(f"{label} - SMLX output length: {smlx_len} chars")
    logger.debug(f"{label} - SMLX unique/total tokens: {smlx_unique_tokens}/{smlx_total_tokens}")

    result = {
        "smlx_length": smlx_len,
        "smlx_unique_tokens": smlx_unique_tokens,
        "smlx_total_tokens": smlx_total_tokens,
    }

    if reference_output is not None:
        ref_len = len(reference_output)
        ref_unique_tokens = len(set(reference_output.split()))
        ref_total_tokens = len(reference_output.split())

        logger.debug(f"{label} - Reference output length: {ref_len} chars")
        logger.debug(f"{label} - Reference unique/total tokens: {ref_unique_tokens}/{ref_total_tokens}")

        # Check for repetition issues
        smlx_repetition_ratio = smlx_unique_tokens / smlx_total_tokens if smlx_total_tokens > 0 else 1.0
        ref_repetition_ratio = ref_unique_tokens / ref_total_tokens if ref_total_tokens > 0 else 1.0

        if smlx_repetition_ratio < 0.5:
            logger.warning(f"{label} - SMLX output has high repetition (ratio={smlx_repetition_ratio:.2f})")

        result.update({
            "reference_length": ref_len,
            "reference_unique_tokens": ref_unique_tokens,
            "reference_total_tokens": ref_total_tokens,
            "smlx_repetition_ratio": smlx_repetition_ratio,
            "reference_repetition_ratio": ref_repetition_ratio,
        })

    return result


__all__ = [
    "log_vision_features",
    "log_embedding_comparison",
    "log_logits_distribution",
    "log_attention_mask",
    "compare_with_reference",
]
