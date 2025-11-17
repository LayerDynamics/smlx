"""
4-bit quantization utilities for SMLX.

Provides convenient 4-bit quantization wrappers optimized for Apple M4 chipsets.
4-bit quantization typically reduces model size by 8x with minimal accuracy loss.

This module offers simple interfaces for the most common quantization scenario
in SMLX - 4-bit uniform quantization of "smol" models.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import QuantizedEmbedding, QuantizedLinear


def quantize_4bit(
    model,
    group_size: int = 64,
    inplace: bool = True,
) -> Optional[object]:
    """
    Quantize a model to 4-bit precision.

    This is a convenience wrapper for 4-bit quantization, the most common
    quantization format for "smol" models on M4 chips. For better quality
    preservation on critical models, consider using GPTQ or AWQ instead.

    Args:
        model: MLX model to quantize
        group_size: Group size for quantization (default: 64, optimal for M4)
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_4bit

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        quantize_4bit(model)  # In-place 4-bit quantization
        # Model is now ~8x smaller with minimal quality loss
        ```

    Notes:
        - Uses symmetric per-group quantization
        - Quantizes nn.Linear and nn.Embedding layers
        - Leaves other layers (LayerNorm, etc.) in original precision
        - Group size 64 is optimal for M4 Metal GPU performance
        - Reduces model size by ~8x (from FP16)
    """
    nn.quantize(model, group_size=group_size, bits=4)
    if not inplace:
        return model
    return None


def quantize_weights_4bit(
    weight: mx.array,
    group_size: int = 64,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Quantize weight array to 4-bit format.

    Args:
        weight: Weight array to quantize (typically 2D for Linear layers)
        group_size: Group size for quantization (default: 64)

    Returns:
        Tuple of (quantized_weights, scales, biases)
        - quantized_weights: Packed uint32 array with 4-bit values
        - scales: Per-group scaling factors (float16)
        - biases: Per-group bias terms (float16)

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_weights_4bit

        weights = mx.random.normal((768, 768))  # Linear layer weights
        w_q, scales, biases = quantize_weights_4bit(weights)
        # w_q is 8x smaller (4 bits per weight, packed in uint32)
        ```

    Notes:
        - Output is packed: 8 weights per uint32 element
        - Actual storage is weight.size * 4 / 32 = weight.size / 8
        - Can be dequantized with mx.dequantize()
    """
    return mx.quantize(weight, group_size=group_size, bits=4)


def dequantize_weights_4bit(
    quantized_weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = 64,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """
    Dequantize 4-bit weights back to floating point.

    Args:
        quantized_weight: Packed uint32 array with 4-bit values
        scales: Per-group scaling factors
        biases: Per-group bias terms
        group_size: Group size used for quantization (default: 64)
        dtype: Output dtype (default: float16)

    Returns:
        Dequantized weight array in specified dtype

    Example:
        ```python
        from smlx.quant import quantize_weights_4bit, dequantize_weights_4bit

        # Quantize
        w_q, scales, biases = quantize_weights_4bit(weights)

        # Dequantize
        weights_restored = dequantize_weights_4bit(w_q, scales, biases)
        # weights_restored H weights (with quantization error)
        ```
    """
    return mx.dequantize(
        quantized_weight,
        scales,
        biases,
        group_size=group_size,
        bits=4,
        dtype=dtype,
    )


def estimate_4bit_size_reduction(model) -> dict:
    """
    Estimate size reduction from 4-bit quantization.

    Args:
        model: MLX model (unquantized or partially quantized)

    Returns:
        Dictionary with size estimates:
        - current_mb: Current model size in MB
        - quantized_mb: Size after full 4-bit quantization
        - reduction_ratio: Size reduction factor (e.g., 7.8x)
        - saved_mb: MB saved by quantization

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import estimate_4bit_size_reduction

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        stats = estimate_4bit_size_reduction(model)
        print(f"Will reduce from {stats['current_mb']:.1f} MB "
              f"to {stats['quantized_mb']:.1f} MB "
              f"({stats['reduction_ratio']:.1f}x smaller)")
        ```
    """
    # Count quantizable parameters (Linear and Embedding layers)
    quantizable_params = 0
    quantized_params = 0
    total_bytes = 0

    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            weight = module.weight
            param_count = weight.size

            if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
                # Already quantized
                quantized_params += param_count
                total_bytes += weight.nbytes
                # Add scales and biases
                if hasattr(module, "scales") and module.scales is not None:
                    total_bytes += module.scales.nbytes
                if hasattr(module, "biases") and module.biases is not None:
                    total_bytes += module.biases.nbytes
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                # Can be quantized
                quantizable_params += param_count
                total_bytes += weight.nbytes
            else:
                # Other layers (keep as-is)
                total_bytes += weight.nbytes

    # Calculate 4-bit quantized size
    # 4 bits per weight = 0.5 bytes
    # Plus scales and biases (float16 = 2 bytes each per group)
    group_size = 64
    num_groups = (quantizable_params + group_size - 1) // group_size
    quantized_weight_bytes = quantizable_params * 4 // 8  # 4 bits = 0.5 bytes
    scales_biases_bytes = num_groups * 2 * 2  # 2 arrays * 2 bytes (float16)

    # Already quantized parameters stay as-is
    # Estimate their size (assume 4-bit)
    already_quantized_bytes = quantized_params * 4 // 8
    already_quantized_groups = (quantized_params + group_size - 1) // group_size
    already_quantized_bytes += already_quantized_groups * 2 * 2

    # Non-quantizable parameters (everything else)
    non_quantizable_bytes = total_bytes - (
        quantizable_params * 2 + quantized_params * 0.5
    )  # Assume FP16

    quantized_total = (
        quantized_weight_bytes
        + scales_biases_bytes
        + already_quantized_bytes
        + non_quantizable_bytes
    )
    current_mb = total_bytes / (1024**2)
    quantized_mb = quantized_total / (1024**2)
    reduction_ratio = current_mb / quantized_mb if quantized_mb > 0 else 1.0

    return {
        "current_mb": current_mb,
        "quantized_mb": quantized_mb,
        "reduction_ratio": reduction_ratio,
        "saved_mb": current_mb - quantized_mb,
        "quantizable_params": quantizable_params,
        "already_quantized": quantized_params > 0,
    }


def is_4bit_quantized(module) -> bool:
    """
    Check if a module is quantized to 4-bit.

    Args:
        module: MLX module to check

    Returns:
        True if module is 4-bit quantized, False otherwise

    Example:
        ```python
        from smlx.quant import quantize_4bit, is_4bit_quantized

        linear = nn.Linear(768, 768)
        assert not is_4bit_quantized(linear)

        quantize_4bit(linear)
        assert is_4bit_quantized(linear)
        ```
    """
    if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
        return getattr(module, "bits", None) == 4
    return False


def get_quantization_info(module) -> Optional[dict]:
    """
    Get quantization information for a module.

    Args:
        module: MLX module to inspect

    Returns:
        Dictionary with quantization info if quantized, None otherwise:
        - bits: Bits per weight
        - group_size: Group size
        - is_4bit: Whether it's 4-bit quantized
        - weight_shape: Original weight shape
        - quantized_shape: Packed weight shape

    Example:
        ```python
        from smlx.quant import quantize_4bit, get_quantization_info

        linear = nn.Linear(768, 768)
        quantize_4bit(linear)
        info = get_quantization_info(linear)
        print(f"Quantized to {info['bits']} bits with "
              f"group size {info['group_size']}")
        ```
    """
    if not isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
        return None

    bits = getattr(module, "bits", None)
    group_size = getattr(module, "group_size", None)

    if bits is None or group_size is None:
        return None

    weight = module.weight
    # Calculate original shape
    # Packed weights: each uint32 holds 32/bits values
    elements_per_int = 32 // bits
    if isinstance(module, QuantizedLinear):
        # Linear: (out_features, in_features // elements_per_int)
        original_shape = (weight.shape[0], weight.shape[1] * elements_per_int)
    else:
        # Embedding: (num_embeddings, embedding_dim // elements_per_int)
        original_shape = (weight.shape[0], weight.shape[1] * elements_per_int)

    return {
        "bits": bits,
        "group_size": group_size,
        "is_4bit": bits == 4,
        "weight_shape": original_shape,
        "quantized_shape": weight.shape,
        "compression_ratio": (original_shape[0] * original_shape[1] * 16)
        / (weight.size * 32),
    }


__all__ = [
    "quantize_4bit",
    "quantize_weights_4bit",
    "dequantize_weights_4bit",
    "estimate_4bit_size_reduction",
    "is_4bit_quantized",
    "get_quantization_info",
]
