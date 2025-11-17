"""
MXFP4 (Microscaling FP4) quantization for SMLX using MLX native support.

MXFP4 is part of the OCP Microscaling Formats (MX) specification v1.0. It uses:
- Element format: E2M1 (1 sign, 2 exponent, 1 mantissa = 4 bits per element)
- Scale format: E8M0 (8-bit exponent-only scale per block)
- Block size: 32 elements (fixed per OCP specification)

Key characteristics:
- Industry-standard format from OCP (AMD, Arm, Intel, Meta, Microsoft, NVIDIA, Qualcomm)
- Block-based shared exponent (microscaling)
- Hardware-accelerated via MLX Metal kernels
- 8-bit scale overhead vs 16-bit in regular FP4

This format provides better compression than regular FP4 due to smaller scale overhead,
while maintaining hardware acceleration and industry standard compliance.

Use cases:
- Maximum compression for edge deployment
- Industry-standard OCP compliance
- Hardware-accelerated inference on Apple Silicon
- Models compatible with 32-element block size

Example:
    ```python
    from smlx.models.SmolLM2_135M import load
    from smlx.quant import quantize_model_mxfp4, estimate_mxfp4_size

    model, _ = load("mlx-community/SmolLM2-135M-Instruct")

    # Estimate size reduction
    stats = estimate_mxfp4_size(model)
    print(f"MXFP4 reduces to {stats['mxfp4_mb']:.1f} MB")

    # Apply MXFP4 quantization
    quantize_model_mxfp4(model)
    ```

Notes:
    - Requires weight dimensions divisible by 32 (can auto-pad)
    - Uses MLX native implementation (optimized Metal kernels)
    - More efficient than regular FP4 (8-bit vs 16-bit scales)
    - Complies with OCP MX specification v1.0
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def validate_mxfp_shape(weight: mx.array, pad: bool = False) -> mx.array:
    """
    Validate and optionally pad weight array for MXFP quantization.

    MXFP requires the last dimension to be divisible by 32 (block size).

    Args:
        weight: Weight array to validate
        pad: If True, pad to nearest multiple of 32. If False, raise error.

    Returns:
        Original or padded weight array

    Raises:
        ValueError: If shape invalid and pad=False

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import validate_mxfp_shape

        # Valid shape
        w = mx.random.normal((768, 768))  # 768 % 32 == 0
        w_valid = validate_mxfp_shape(w)  # OK

        # Invalid shape with padding
        w = mx.random.normal((100, 100))  # 100 % 32 != 0
        w_padded = validate_mxfp_shape(w, pad=True)  # Pads to (100, 128)
        ```
    """
    last_dim = weight.shape[-1]

    if last_dim % 32 == 0:
        return weight

    if not pad:
        raise ValueError(
            f"MXFP requires last dimension divisible by 32. "
            f"Got shape {weight.shape} with last dim {last_dim}. "
            f"Set pad=True to auto-pad or reshape your weights."
        )

    # Pad to nearest multiple of 32
    padded_last_dim = ((last_dim + 31) // 32) * 32
    pad_size = padded_last_dim - last_dim

    # Create padding specification: [(0, 0), ..., (0, pad_size)]
    pad_spec = [(0, 0)] * (len(weight.shape) - 1) + [(0, pad_size)]

    return mx.pad(weight, pad_spec)


def quantize_to_mxfp4(
    weight: mx.array, validate: bool = True, pad_if_needed: bool = True
) -> tuple[mx.array, mx.array]:
    """
    Quantize weights to MXFP4 format using MLX native implementation.

    MXFP4 uses E2M1 element format (4 bits) with E8M0 shared scale (8 bits)
    per 32-element block.

    Args:
        weight: Weight array to quantize
        validate: Check shape compatibility (default: True)
        pad_if_needed: Auto-pad if shape not divisible by 32 (default: True)

    Returns:
        Tuple of (quantized_weights, scales)
        - quantized_weights: uint32 array (packed 4-bit values)
        - scales: uint8 array (E8M0 scales, one per 32 elements)

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_to_mxfp4, dequantize_from_mxfp4

        weights = mx.random.normal((768, 768))
        w_q, scales = quantize_to_mxfp4(weights)

        # w_q is uint32 (packed), scales is uint8
        print(f"Original: {weights.nbytes} bytes")
        print(f"Quantized: {w_q.nbytes + scales.nbytes} bytes")

        # Dequantize
        weights_restored = dequantize_from_mxfp4(w_q, scales)
        ```

    Notes:
        - Uses MLX native quantization with Metal GPU acceleration
        - Block size is fixed at 32 (OCP specification)
        - Returns 2-tuple (no bias, unlike INT quantization)
        - 8-bit scale overhead (vs 16-bit in regular FP4)
    """
    if validate:
        weight = validate_mxfp_shape(weight, pad=pad_if_needed)

    # MLX native MXFP4 quantization
    # Returns (quantized_weights: uint32, scales: uint8)
    return mx.quantize(weight, mode="mxfp4")


def dequantize_from_mxfp4(
    w_q: mx.array, scales: mx.array, dtype: mx.Dtype = mx.float16
) -> mx.array:
    """
    Dequantize MXFP4 weights back to floating point.

    Args:
        w_q: Quantized weights (uint32 packed format)
        scales: E8M0 scales (uint8, one per 32 elements)
        dtype: Target dtype for dequantized weights (default: float16)

    Returns:
        Dequantized weight array in specified dtype

    Example:
        ```python
        from smlx.quant import quantize_to_mxfp4, dequantize_from_mxfp4

        # Quantize
        w_q, scales = quantize_to_mxfp4(weights)

        # Dequantize to float16
        weights_fp16 = dequantize_from_mxfp4(w_q, scales, dtype=mx.float16)

        # Dequantize to bfloat16
        weights_bf16 = dequantize_from_mxfp4(w_q, scales, dtype=mx.bfloat16)
        ```
    """
    # MLX native MXFP4 dequantization
    result = mx.dequantize(w_q, scales, mode="mxfp4")

    # Convert to desired dtype
    if result.dtype != dtype:
        result = result.astype(dtype)

    return result


def quantize_model_mxfp4(model, inplace: bool = True) -> Optional[object]:
    """
    Quantize model to MXFP4 format using MLX native quantization.

    This uses MLX's nn.quantize() which replaces Linear layers with
    QuantizedLinear layers using MXFP4 format.

    Args:
        model: MLX model to quantize
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_model_mxfp4

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

        # Quantize model
        quantize_model_mxfp4(model)

        # Model now uses QuantizedLinear with MXFP4 weights
        # Inference uses optimized Metal kernels
        ```

    Notes:
        - Converts nn.Linear to nn.QuantizedLinear with MXFP4 mode
        - Uses hardware-accelerated quantized matrix operations
        - Block size fixed at 32 (OCP specification)
        - Weights must have last dimension divisible by 32
    """
    # MLX native model quantization
    # group_size=32 is default and required for MXFP4
    nn.quantize(model, group_size=32, mode="mxfp4")

    if not inplace:
        return model
    return None


def estimate_mxfp4_size(model) -> dict:
    """
    Estimate memory usage with MXFP4 quantization.

    Args:
        model: MLX model to analyze

    Returns:
        Dictionary with size estimates:
        - current_mb: Current model size in MB
        - mxfp4_mb: Size after MXFP4 quantization
        - reduction_ratio: Size reduction factor
        - saved_mb: MB saved
        - total_params: Total parameter count
        - quantizable_params: Parameters that can be quantized
        - scale_overhead_mb: Size of E8M0 scales

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import estimate_mxfp4_size

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        stats = estimate_mxfp4_size(model)

        print(f"Current: {stats['current_mb']:.1f} MB")
        print(f"MXFP4: {stats['mxfp4_mb']:.1f} MB")
        print(f"Reduction: {stats['reduction_ratio']:.2f}x")
        print(f"Scale overhead: {stats['scale_overhead_mb']:.2f} MB")
        ```
    """
    total_bytes = 0
    quantizable_params = 0
    total_params = 0

    for _, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            weight = module.weight
            total_bytes += weight.nbytes
            total_params += weight.size

            if isinstance(module, (nn.Linear, nn.Embedding)):
                quantizable_params += weight.size

        if hasattr(module, "bias") and module.bias is not None:
            bias = module.bias
            total_bytes += bias.nbytes
            total_params += bias.size

    # MXFP4 size calculation:
    # - Elements: 4 bits per weight (0.5 bytes)
    # - Scales: 8 bits (1 byte) per 32 elements

    element_bytes = quantizable_params // 2  # 4 bits = 0.5 bytes
    n_blocks = (quantizable_params + 31) // 32  # Round up to nearest block
    scale_bytes = n_blocks  # 1 byte (uint8) per block

    mxfp4_bytes = element_bytes + scale_bytes

    # Add non-quantizable parameters (assume they stay as-is)
    non_quantizable_bytes = total_bytes - (quantizable_params * 2)  # Assume FP16
    mxfp4_total = mxfp4_bytes + non_quantizable_bytes

    current_mb = total_bytes / (1024**2)
    mxfp4_mb = mxfp4_total / (1024**2)
    scale_overhead_mb = scale_bytes / (1024**2)
    reduction_ratio = current_mb / mxfp4_mb if mxfp4_mb > 0 else 1.0

    return {
        "current_mb": current_mb,
        "mxfp4_mb": mxfp4_mb,
        "reduction_ratio": reduction_ratio,
        "saved_mb": current_mb - mxfp4_mb,
        "total_params": total_params,
        "quantizable_params": quantizable_params,
        "scale_overhead_mb": scale_overhead_mb,
        "format": "MXFP4 (E2M1 + E8M0 scale, block size 32)",
        "scale_format": "E8M0 (8-bit exponent-only)",
        "block_size": 32,
    }


def compare_mxfp4_vs_fp4(weight: mx.array) -> dict:
    """
    Compare MXFP4 vs regular FP4 quantization quality.

    Args:
        weight: Weight array to compare

    Returns:
        Dictionary with comparison metrics:
        - mxfp4_error: Mean absolute error for MXFP4
        - fp4_error: Mean absolute error for regular FP4
        - mxfp4_max_error: Max error for MXFP4
        - fp4_max_error: Max error for regular FP4
        - mxfp4_better: True if MXFP4 has lower error
        - size_comparison: Size comparison details
        - recommendation: Which format is better

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import compare_mxfp4_vs_fp4

        weights = mx.random.normal((768, 768))
        comparison = compare_mxfp4_vs_fp4(weights)

        print(f"MXFP4 error: {comparison['mxfp4_error']:.4f}")
        print(f"FP4 error: {comparison['fp4_error']:.4f}")
        print(comparison['recommendation'])
        ```
    """
    from .fp4 import dequantize_from_fp4, quantize_to_fp4

    # Ensure weight is compatible with MXFP4 (pad if needed)
    weight_padded = validate_mxfp_shape(weight, pad=True)

    # MXFP4 quantization
    w_q_mxfp4, scales_mxfp4 = quantize_to_mxfp4(weight_padded, validate=False)
    restored_mxfp4 = dequantize_from_mxfp4(w_q_mxfp4, scales_mxfp4)

    # Trim padding if we added any
    if restored_mxfp4.shape != weight.shape:
        restored_mxfp4 = restored_mxfp4[: weight.shape[0], : weight.shape[1]]

    mxfp4_error = mx.mean(mx.abs(restored_mxfp4 - weight)).item()
    mxfp4_max_error = mx.max(mx.abs(restored_mxfp4 - weight)).item()

    # Regular FP4 quantization (group_size=64 to match typical usage)
    indices_fp4, scales_fp4 = quantize_to_fp4(weight, group_size=64)
    restored_fp4 = dequantize_from_fp4(indices_fp4, scales_fp4, group_size=64)
    fp4_error = mx.mean(mx.abs(restored_fp4 - weight)).item()
    fp4_max_error = mx.max(mx.abs(restored_fp4 - weight)).item()

    # Size comparison
    mxfp4_size = w_q_mxfp4.nbytes + scales_mxfp4.nbytes
    fp4_size = indices_fp4.nbytes + scales_fp4.nbytes

    # Determine recommendation
    if mxfp4_error < fp4_error * 0.9:  # MXFP4 significantly better
        recommendation = "MXFP4 (better quality + smaller size due to 8-bit scales)"
    elif fp4_error < mxfp4_error * 0.9:  # FP4 significantly better
        recommendation = "FP4 (better quality, flexible block size)"
    else:
        recommendation = "MXFP4 (similar quality, smaller size, hardware-accelerated)"

    return {
        "mxfp4_error": mxfp4_error,
        "fp4_error": fp4_error,
        "mxfp4_max_error": mxfp4_max_error,
        "fp4_max_error": fp4_max_error,
        "mxfp4_better": mxfp4_error < fp4_error,
        "improvement_ratio": fp4_error / mxfp4_error if mxfp4_error > 0 else float("inf"),
        "size_comparison": {
            "mxfp4_bytes": mxfp4_size,
            "fp4_bytes": fp4_size,
            "mxfp4_smaller": mxfp4_size < fp4_size,
            "size_ratio": fp4_size / mxfp4_size if mxfp4_size > 0 else 1.0,
        },
        "recommendation": recommendation,
    }


def compare_mxfp4_vs_int4(weight: mx.array) -> dict:
    """
    Compare MXFP4 vs INT4 quantization quality.

    Args:
        weight: Weight array to compare

    Returns:
        Dictionary with comparison metrics:
        - mxfp4_error: Mean absolute error for MXFP4
        - int4_error: Mean absolute error for INT4
        - mxfp4_max_error: Max error for MXFP4
        - int4_max_error: Max error for INT4
        - mxfp4_better: True if MXFP4 has lower error
        - recommendation: Which format is better for this weight

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import compare_mxfp4_vs_int4

        weights = mx.random.normal((768, 768))
        comparison = compare_mxfp4_vs_int4(weights)

        print(f"MXFP4 error: {comparison['mxfp4_error']:.4f}")
        print(f"INT4 error: {comparison['int4_error']:.4f}")
        print(comparison['recommendation'])
        ```
    """
    # Ensure weight is compatible with MXFP4
    weight_padded = validate_mxfp_shape(weight, pad=True)

    # MXFP4 quantization
    w_q_mxfp4, scales_mxfp4 = quantize_to_mxfp4(weight_padded, validate=False)
    restored_mxfp4 = dequantize_from_mxfp4(w_q_mxfp4, scales_mxfp4)

    # Trim padding if we added any
    if restored_mxfp4.shape != weight.shape:
        restored_mxfp4 = restored_mxfp4[: weight.shape[0], : weight.shape[1]]

    mxfp4_error = mx.mean(mx.abs(restored_mxfp4 - weight)).item()
    mxfp4_max_error = mx.max(mx.abs(restored_mxfp4 - weight)).item()

    # INT4 quantization (using MLX built-in)
    w_q_int4, scales_int4, biases_int4 = mx.quantize(weight, group_size=64, bits=4)
    restored_int4 = mx.dequantize(w_q_int4, scales_int4, biases_int4, group_size=64, bits=4)
    int4_error = mx.mean(mx.abs(restored_int4 - weight)).item()
    int4_max_error = mx.max(mx.abs(restored_int4 - weight)).item()

    # Determine recommendation
    if mxfp4_error < int4_error * 0.9:  # MXFP4 significantly better
        recommendation = "MXFP4 (better quality for wide dynamic range, hardware-accelerated)"
    elif int4_error < mxfp4_error * 0.9:  # INT4 significantly better
        recommendation = "INT4 (better quality for uniform distribution)"
    else:
        recommendation = "MXFP4 (similar quality, industry-standard OCP format)"

    return {
        "mxfp4_error": mxfp4_error,
        "int4_error": int4_error,
        "mxfp4_max_error": mxfp4_max_error,
        "int4_max_error": int4_max_error,
        "mxfp4_better": mxfp4_error < int4_error,
        "improvement_ratio": int4_error / mxfp4_error if mxfp4_error > 0 else float("inf"),
        "recommendation": recommendation,
    }


__all__ = [
    "validate_mxfp_shape",
    "quantize_to_mxfp4",
    "dequantize_from_mxfp4",
    "quantize_model_mxfp4",
    "estimate_mxfp4_size",
    "compare_mxfp4_vs_fp4",
    "compare_mxfp4_vs_int4",
]
