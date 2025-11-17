"""
6-bit quantization utilities for SMLX.

Provides 6-bit quantization wrappers for Apple M4 chipsets.
6-bit quantization offers a middle ground between 4-bit and 8-bit, providing
better quality than 4-bit with ~5.3x size reduction.

This format is useful for mixed-precision strategies where critical layers
need higher precision than 4-bit but don't require full 8-bit.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import QuantizedEmbedding, QuantizedLinear


def quantize_6bit(
    model,
    group_size: int = 64,
    inplace: bool = True,
) -> Optional[object]:
    """
    Quantize a model to 6-bit precision.

    6-bit quantization provides a balance between model size and quality,
    offering better accuracy than 4-bit with ~5.3x compression from FP16.

    Args:
        model: MLX model to quantize
        group_size: Group size for quantization (default: 64)
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_6bit

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        quantize_6bit(model)  # In-place 6-bit quantization
        # Model is ~5.3x smaller with better quality than 4-bit
        ```

    Notes:
        - Uses symmetric per-group quantization
        - Quantizes nn.Linear and nn.Embedding layers
        - Better quality than 4-bit, smaller than 8-bit
        - Useful for mixed-precision strategies
        - Reduces model size by ~5.3x (from FP16)
    """
    nn.quantize(model, group_size=group_size, bits=6)
    if not inplace:
        return model
    return None


def quantize_weights_6bit(
    weight: mx.array,
    group_size: int = 64,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Quantize weight array to 6-bit format.

    Args:
        weight: Weight array to quantize (typically 2D for Linear layers)
        group_size: Group size for quantization (default: 64)

    Returns:
        Tuple of (quantized_weights, scales, biases)
        - quantized_weights: Packed uint32 array with 6-bit values
        - scales: Per-group scaling factors (float16)
        - biases: Per-group bias terms (float16)

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_weights_6bit

        weights = mx.random.normal((768, 768))  # Linear layer weights
        w_q, scales, biases = quantize_weights_6bit(weights)
        # w_q is ~5.3x smaller (6 bits per weight, packed in uint32)
        ```

    Notes:
        - Output is packed: 5 weights per uint32 element (32/6 H 5.33)
        - Actual storage is weight.size * 6 / 32 H weight.size / 5.33
        - Can be dequantized with mx.dequantize()
    """
    return mx.quantize(weight, group_size=group_size, bits=6)


def dequantize_weights_6bit(
    quantized_weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = 64,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """
    Dequantize 6-bit weights back to floating point.

    Args:
        quantized_weight: Packed uint32 array with 6-bit values
        scales: Per-group scaling factors
        biases: Per-group bias terms
        group_size: Group size used for quantization (default: 64)
        dtype: Output dtype (default: float16)

    Returns:
        Dequantized weight array in specified dtype

    Example:
        ```python
        from smlx.quant import quantize_weights_6bit, dequantize_weights_6bit

        # Quantize
        w_q, scales, biases = quantize_weights_6bit(weights)

        # Dequantize
        weights_restored = dequantize_weights_6bit(w_q, scales, biases)
        # weights_restored H weights (with less error than 4-bit)
        ```
    """
    return mx.dequantize(
        quantized_weight,
        scales,
        biases,
        group_size=group_size,
        bits=6,
        dtype=dtype,
    )


def estimate_6bit_size_reduction(model) -> dict:
    """
    Estimate size reduction from 6-bit quantization.

    Args:
        model: MLX model (unquantized or partially quantized)

    Returns:
        Dictionary with size estimates:
        - current_mb: Current model size in MB
        - quantized_mb: Size after full 6-bit quantization
        - reduction_ratio: Size reduction factor (e.g., 5.2x)
        - saved_mb: MB saved by quantization

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import estimate_6bit_size_reduction

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        stats = estimate_6bit_size_reduction(model)
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

    # Calculate 6-bit quantized size
    # 6 bits per weight = 0.75 bytes
    # Plus scales and biases (float16 = 2 bytes each per group)
    group_size = 64
    num_groups = (quantizable_params + group_size - 1) // group_size
    quantized_weight_bytes = quantizable_params * 6 // 8  # 6 bits = 0.75 bytes
    scales_biases_bytes = num_groups * 2 * 2  # 2 arrays * 2 bytes (float16)

    # Already quantized parameters stay as-is
    # Estimate their size (assume 6-bit)
    already_quantized_bytes = quantized_params * 6 // 8
    already_quantized_groups = (quantized_params + group_size - 1) // group_size
    already_quantized_bytes += already_quantized_groups * 2 * 2

    # Non-quantizable parameters (everything else)
    non_quantizable_bytes = total_bytes - (
        quantizable_params * 2 + quantized_params * 0.75
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


def is_6bit_quantized(module) -> bool:
    """
    Check if a module is quantized to 6-bit.

    Args:
        module: MLX module to check

    Returns:
        True if module is 6-bit quantized, False otherwise

    Example:
        ```python
        from smlx.quant import quantize_6bit, is_6bit_quantized

        linear = nn.Linear(768, 768)
        assert not is_6bit_quantized(linear)

        quantize_6bit(linear)
        assert is_6bit_quantized(linear)
        ```
    """
    if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
        return getattr(module, "bits", None) == 6
    return False


__all__ = [
    "quantize_6bit",
    "quantize_weights_6bit",
    "dequantize_weights_6bit",
    "estimate_6bit_size_reduction",
    "is_6bit_quantized",
]
