"""
MXFP8 (Microscaling FP8) quantization for SMLX using MLX native support.

MXFP8 is part of the OCP Microscaling Formats (MX) specification v1.0. It uses:
- Element format: E4M3 (1 sign, 4 exponent, 3 mantissa = 8 bits per element)
- Scale format: E8M0 (8-bit exponent-only scale per block)
- Block size: 32 elements (fixed per OCP specification)

Key characteristics:
- Industry-standard format from OCP (AMD, Arm, Intel, Meta, Microsoft, NVIDIA, Qualcomm)
- Block-based shared exponent (microscaling)
- Hardware-accelerated via MLX Metal kernels
- True 8-bit storage (vs simulated in regular FP8)
- 8-bit scale overhead vs 16-bit in regular FP8

This format provides 2x memory savings over FP16 with hardware acceleration,
while maintaining better precision than MXFP4.

Use cases:
- Balanced compression and quality
- Industry-standard OCP compliance
- Hardware-accelerated inference on Apple Silicon
- Models compatible with 32-element block size
- Replacement for simulated FP8 with true 8-bit storage

Example:
    ```python
    from smlx.models.SmolLM2_135M import load
    from smlx.quant import quantize_model_mxfp8, estimate_mxfp8_size

    model, _ = load("mlx-community/SmolLM2-135M-Instruct")

    # Estimate size reduction
    stats = estimate_mxfp8_size(model)
    print(f"MXFP8 reduces to {stats['mxfp8_mb']:.1f} MB")

    # Apply MXFP8 quantization
    quantize_model_mxfp8(model)
    ```

Notes:
    - Requires weight dimensions divisible by 32 (can auto-pad)
    - Uses MLX native implementation (optimized Metal kernels)
    - True 8-bit storage (not simulated like regular FP8)
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
            f"MXFP8 requires last dimension divisible by 32. "
            f"Got shape {weight.shape} with last dim {last_dim}. "
            f"Set pad=True to auto-pad or reshape your weights."
        )

    # Pad to nearest multiple of 32
    padded_last_dim = ((last_dim + 31) // 32) * 32
    pad_size = padded_last_dim - last_dim

    # Create padding specification: [(0, 0), ..., (0, pad_size)]
    pad_spec = [(0, 0)] * (len(weight.shape) - 1) + [(0, pad_size)]

    return mx.pad(weight, pad_spec)


def quantize_to_mxfp8(
    weight: mx.array, validate: bool = True, pad_if_needed: bool = True
) -> tuple[mx.array, mx.array]:
    """
    Quantize weights to MXFP8 format using MLX native implementation.

    MXFP8 uses E4M3 element format (8 bits) with E8M0 shared scale (8 bits)
    per 32-element block.

    Args:
        weight: Weight array to quantize
        validate: Check shape compatibility (default: True)
        pad_if_needed: Auto-pad if shape not divisible by 32 (default: True)

    Returns:
        Tuple of (quantized_weights, scales)
        - quantized_weights: uint8 array (E4M3 format)
        - scales: uint8 array (E8M0 scales, one per 32 elements)

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_to_mxfp8, dequantize_from_mxfp8

        weights = mx.random.normal((768, 768))
        w_q, scales = quantize_to_mxfp8(weights)

        # Both w_q and scales are uint8
        print(f"Original: {weights.nbytes} bytes")
        print(f"Quantized: {w_q.nbytes + scales.nbytes} bytes")

        # Dequantize
        weights_restored = dequantize_from_mxfp8(w_q, scales)
        ```

    Notes:
        - Uses MLX native quantization with Metal GPU acceleration
        - Block size is fixed at 32 (OCP specification)
        - Returns 2-tuple (no bias, unlike INT quantization)
        - True 8-bit storage (not simulated)
        - 8-bit scale overhead (vs 16-bit in regular FP8)
    """
    if validate:
        weight = validate_mxfp_shape(weight, pad=pad_if_needed)

    # MLX native MXFP8 quantization
    # Returns (quantized_weights: uint8, scales: uint8)
    return mx.quantize(weight, mode="mxfp8")


def dequantize_from_mxfp8(
    w_q: mx.array, scales: mx.array, dtype: mx.Dtype = mx.float16
) -> mx.array:
    """
    Dequantize MXFP8 weights back to floating point.

    Args:
        w_q: Quantized weights (uint8 E4M3 format)
        scales: E8M0 scales (uint8, one per 32 elements)
        dtype: Target dtype for dequantized weights (default: float16)

    Returns:
        Dequantized weight array in specified dtype

    Example:
        ```python
        from smlx.quant import quantize_to_mxfp8, dequantize_from_mxfp8

        # Quantize
        w_q, scales = quantize_to_mxfp8(weights)

        # Dequantize to float16
        weights_fp16 = dequantize_from_mxfp8(w_q, scales, dtype=mx.float16)

        # Dequantize to bfloat16
        weights_bf16 = dequantize_from_mxfp8(w_q, scales, dtype=mx.bfloat16)
        ```
    """
    # MLX native MXFP8 dequantization
    result = mx.dequantize(w_q, scales, mode="mxfp8")

    # Convert to desired dtype
    if result.dtype != dtype:
        result = result.astype(dtype)

    return result


def quantize_model_mxfp8(model, inplace: bool = True) -> Optional[object]:
    """
    Quantize model to MXFP8 format using MLX native quantization.

    This uses MLX's nn.quantize() which replaces Linear layers with
    QuantizedLinear layers using MXFP8 format.

    Args:
        model: MLX model to quantize
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_model_mxfp8

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

        # Quantize model
        quantize_model_mxfp8(model)

        # Model now uses QuantizedLinear with MXFP8 weights
        # Inference uses optimized Metal kernels
        ```

    Notes:
        - Converts nn.Linear to nn.QuantizedLinear with MXFP8 mode
        - Uses hardware-accelerated quantized matrix operations
        - Block size fixed at 32 (OCP specification)
        - Weights must have last dimension divisible by 32
        - True 8-bit storage (2x memory savings vs FP16)
    """
    # MLX native model quantization
    # group_size=32 is default and required for MXFP8
    # bits=8 is required for MXFP8 (8-bit element format)
    nn.quantize(model, group_size=32, bits=8, mode="mxfp8")

    if not inplace:
        return model
    return None


def estimate_mxfp8_size(model) -> dict:
    """
    Estimate memory usage with MXFP8 quantization.

    Args:
        model: MLX model to analyze

    Returns:
        Dictionary with size estimates:
        - current_mb: Current model size in MB
        - mxfp8_mb: Size after MXFP8 quantization
        - reduction_ratio: Size reduction factor
        - saved_mb: MB saved
        - total_params: Total parameter count
        - quantizable_params: Parameters that can be quantized
        - scale_overhead_mb: Size of E8M0 scales

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import estimate_mxfp8_size

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        stats = estimate_mxfp8_size(model)

        print(f"Current: {stats['current_mb']:.1f} MB")
        print(f"MXFP8: {stats['mxfp8_mb']:.1f} MB")
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

    # MXFP8 size calculation:
    # - Elements: 8 bits per weight (1 byte)
    # - Scales: 8 bits (1 byte) per 32 elements

    element_bytes = quantizable_params  # 8 bits = 1 byte
    n_blocks = (quantizable_params + 31) // 32  # Round up to nearest block
    scale_bytes = n_blocks  # 1 byte (uint8) per block

    mxfp8_bytes = element_bytes + scale_bytes

    # Add non-quantizable parameters (assume they stay as-is)
    non_quantizable_bytes = total_bytes - (quantizable_params * 2)  # Assume FP16
    mxfp8_total = mxfp8_bytes + non_quantizable_bytes

    current_mb = total_bytes / (1024**2)
    mxfp8_mb = mxfp8_total / (1024**2)
    scale_overhead_mb = scale_bytes / (1024**2)
    reduction_ratio = current_mb / mxfp8_mb if mxfp8_mb > 0 else 1.0

    return {
        "current_mb": current_mb,
        "mxfp8_mb": mxfp8_mb,
        "reduction_ratio": reduction_ratio,
        "saved_mb": current_mb - mxfp8_mb,
        "total_params": total_params,
        "quantizable_params": quantizable_params,
        "scale_overhead_mb": scale_overhead_mb,
        "format": "MXFP8 (E4M3 + E8M0 scale, block size 32)",
        "scale_format": "E8M0 (8-bit exponent-only)",
        "block_size": 32,
    }


def compare_mxfp8_vs_fp8(weight: mx.array) -> dict:
    """
    Compare MXFP8 vs regular FP8 quantization quality.

    Args:
        weight: Weight array to compare

    Returns:
        Dictionary with comparison metrics:
        - mxfp8_error: Mean absolute error for MXFP8
        - fp8_error: Mean absolute error for regular FP8
        - mxfp8_max_error: Max error for MXFP8
        - fp8_max_error: Max error for regular FP8
        - mxfp8_better: True if MXFP8 has lower error
        - size_comparison: Size comparison details
        - recommendation: Which format is better

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import compare_mxfp8_vs_fp8

        weights = mx.random.normal((768, 768))
        comparison = compare_mxfp8_vs_fp8(weights)

        print(f"MXFP8 error: {comparison['mxfp8_error']:.4f}")
        print(f"FP8 error: {comparison['fp8_error']:.4f}")
        print(comparison['recommendation'])
        ```
    """
    import warnings

    # Suppress deprecation warning for FP8 (we're comparing to show why MXFP8 is better)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="smlx.quant.fp8")
        from .fp8 import dequantize_from_fp8, quantize_to_fp8_e4m3

    # Ensure weight is compatible with MXFP8 (pad if needed)
    weight_padded = validate_mxfp_shape(weight, pad=True)

    # MXFP8 quantization
    w_q_mxfp8, scales_mxfp8 = quantize_to_mxfp8(weight_padded, validate=False)
    restored_mxfp8 = dequantize_from_mxfp8(w_q_mxfp8, scales_mxfp8)

    # Trim padding if we added any
    if restored_mxfp8.shape != weight.shape:
        restored_mxfp8 = restored_mxfp8[: weight.shape[0], : weight.shape[1]]

    mxfp8_error = mx.mean(mx.abs(restored_mxfp8 - weight)).item()
    mxfp8_max_error = mx.max(mx.abs(restored_mxfp8 - weight)).item()

    # Regular FP8 quantization (group_size=64 to match typical usage)
    # Suppress deprecation warning - we're comparing to demonstrate why MXFP8 is better
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        values_fp8, scales_fp8 = quantize_to_fp8_e4m3(weight, group_size=64)
        restored_fp8 = dequantize_from_fp8(values_fp8, scales_fp8, group_size=64)
    fp8_error = mx.mean(mx.abs(restored_fp8 - weight)).item()
    fp8_max_error = mx.max(mx.abs(restored_fp8 - weight)).item()

    # Size comparison
    mxfp8_size = w_q_mxfp8.nbytes + scales_mxfp8.nbytes
    # Regular FP8 is simulated as float16, so much larger
    fp8_size = values_fp8.nbytes + scales_fp8.nbytes

    # Determine recommendation
    if mxfp8_error < fp8_error * 0.9:  # MXFP8 significantly better
        recommendation = "MXFP8 (better quality + true 8-bit storage + hardware-accelerated)"
    elif fp8_error < mxfp8_error * 0.9:  # FP8 significantly better
        recommendation = "FP8 (better quality, flexible block size)"
    else:
        recommendation = "MXFP8 (similar quality, true 8-bit storage, hardware-accelerated)"

    return {
        "mxfp8_error": mxfp8_error,
        "fp8_error": fp8_error,
        "mxfp8_max_error": mxfp8_max_error,
        "fp8_max_error": fp8_max_error,
        "mxfp8_better": mxfp8_error < fp8_error,
        "improvement_ratio": fp8_error / mxfp8_error if mxfp8_error > 0 else float("inf"),
        "size_comparison": {
            "mxfp8_bytes": mxfp8_size,
            "fp8_bytes": fp8_size,
            "mxfp8_smaller": mxfp8_size < fp8_size,
            "size_ratio": fp8_size / mxfp8_size if mxfp8_size > 0 else 1.0,
        },
        "recommendation": recommendation,
    }


def compare_mxfp8_vs_int8(weight: mx.array) -> dict:
    """
    Compare MXFP8 vs INT8 quantization quality.

    Args:
        weight: Weight array to compare

    Returns:
        Dictionary with comparison metrics:
        - mxfp8_error: Mean absolute error for MXFP8
        - int8_error: Mean absolute error for INT8
        - mxfp8_max_error: Max error for MXFP8
        - int8_max_error: Max error for INT8
        - mxfp8_better: True if MXFP8 has lower error
        - recommendation: Which format is better for this weight

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import compare_mxfp8_vs_int8

        weights = mx.random.normal((768, 768))
        comparison = compare_mxfp8_vs_int8(weights)

        print(f"MXFP8 error: {comparison['mxfp8_error']:.4f}")
        print(f"INT8 error: {comparison['int8_error']:.4f}")
        print(comparison['recommendation'])
        ```
    """
    # Ensure weight is compatible with MXFP8
    weight_padded = validate_mxfp_shape(weight, pad=True)

    # MXFP8 quantization
    w_q_mxfp8, scales_mxfp8 = quantize_to_mxfp8(weight_padded, validate=False)
    restored_mxfp8 = dequantize_from_mxfp8(w_q_mxfp8, scales_mxfp8)

    # Trim padding if we added any
    if restored_mxfp8.shape != weight.shape:
        restored_mxfp8 = restored_mxfp8[: weight.shape[0], : weight.shape[1]]

    mxfp8_error = mx.mean(mx.abs(restored_mxfp8 - weight)).item()
    mxfp8_max_error = mx.max(mx.abs(restored_mxfp8 - weight)).item()

    # INT8 quantization (using MLX built-in)
    w_q_int8, scales_int8, biases_int8 = mx.quantize(weight, group_size=64, bits=8)
    restored_int8 = mx.dequantize(w_q_int8, scales_int8, biases_int8, group_size=64, bits=8)
    int8_error = mx.mean(mx.abs(restored_int8 - weight)).item()
    int8_max_error = mx.max(mx.abs(restored_int8 - weight)).item()

    # Determine recommendation
    if mxfp8_error < int8_error * 0.9:  # MXFP8 significantly better
        recommendation = "MXFP8 (better quality for wide dynamic range, hardware-accelerated)"
    elif int8_error < mxfp8_error * 0.9:  # INT8 significantly better
        recommendation = "INT8 (better quality for uniform distribution)"
    else:
        recommendation = "MXFP8 (similar quality, industry-standard OCP format)"

    return {
        "mxfp8_error": mxfp8_error,
        "int8_error": int8_error,
        "mxfp8_max_error": mxfp8_max_error,
        "int8_max_error": int8_max_error,
        "mxfp8_better": mxfp8_error < int8_error,
        "improvement_ratio": int8_error / mxfp8_error if mxfp8_error > 0 else float("inf"),
        "recommendation": recommendation,
    }


__all__ = [
    "validate_mxfp_shape",
    "quantize_to_mxfp8",
    "dequantize_from_mxfp8",
    "quantize_model_mxfp8",
    "estimate_mxfp8_size",
    "compare_mxfp8_vs_fp8",
    "compare_mxfp8_vs_int8",
]
