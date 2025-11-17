"""
FP8 (8-bit Floating Point) quantization for SMLX.

⚠️  DEPRECATION WARNING ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module provides SIMULATED FP8 quantization and is DEPRECATED.

**Key limitations:**
- ❌ Stores as float16 (16-bit) instead of true 8-bit → NO memory savings
- ❌ No hardware acceleration (doesn't use MLX Metal kernels)
- ❌ Simplified rounding instead of proper FP8 bit layout
- ❌ 2x memory overhead vs true 8-bit implementation

**Recommended alternative:** Use `smlx.quant.mxfp8` instead!

MXFP8 provides:
- ✅ True 8-bit storage (uint8) → Real memory savings
- ✅ Hardware-accelerated via Apple Metal GPU kernels
- ✅ Industry standard (OCP MX specification)
- ✅ Proper FP8 E4M3 format with E8M0 scales

Migration example:
    ```python
    # OLD (deprecated, simulated):
    from smlx.quant import quantize_to_fp8_e4m3
    values, scales = quantize_to_fp8_e4m3(weights, group_size=64)

    # NEW (recommended, true 8-bit):
    from smlx.quant import quantize_to_mxfp8
    values, scales = quantize_to_mxfp8(weights)  # group_size=32 (fixed by OCP)
    ```

See MIGRATION.md for full migration guide.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FP8 uses an 8-bit floating point representation providing better dynamic range
than INT8 quantization. Two standard formats exist:

1. E4M3 (4-bit exponent, 3-bit mantissa) - Better precision
2. E5M2 (5-bit exponent, 2-bit mantissa) - Better range

FP8 is particularly useful for:
- Training with mixed precision (FP8/BF16/FP32)
- Gradient quantization
- Activation quantization (better than INT8 for non-uniform distributions)
- Models requiring wide dynamic range

Note: This implementation is for research and educational purposes ONLY.
For production use, migrate to mxfp8.py which uses MLX's native FP8 support.
"""

import warnings

import mlx.core as mx
import mlx.nn as nn

# Deprecation warning message
_DEPRECATION_MSG = (
    "fp8.{func_name} is deprecated and uses simulated FP8 (stored as float16). "
    "Use smlx.quant.mxfp8.{alt_func} instead for true 8-bit storage with hardware acceleration. "
    "See MIGRATION.md for migration guide."
)


def quantize_to_fp8_e4m3(
    weight: mx.array,
    group_size: int = 64,
    max_value: float = 448.0,  # E4M3 max representable value
) -> tuple[mx.array, mx.array]:
    """
    Quantize weights to FP8 E4M3 format (4-bit exp, 3-bit mantissa).

    ⚠️  DEPRECATED: Use smlx.quant.mxfp8.quantize_to_mxfp8 for true 8-bit storage.

    E4M3 provides better precision but limited range compared to E5M2.
    Range: +/-448.0 with subnormal support down to +/-2^-9.

    Args:
        weight: Weight array to quantize
        group_size: Group size for per-group scaling (default: 64)
        max_value: Maximum representable value in E4M3 (default: 448.0)

    Returns:
        Tuple of (quantized_values, scales)
        - quantized_values: Approximated FP8 values as float16 (NOT true 8-bit!)
        - scales: Per-group scaling factors

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_to_fp8_e4m3

        weights = mx.random.normal((768, 768))
        fp8_values, scales = quantize_to_fp8_e4m3(weights)
        ```

    Notes:
        - ⚠️ Uses simulated FP8 (stored as float16 with reduced precision)
        - ⚠️ NO memory savings - this is for educational purposes only
        - Per-group scaling extends effective dynamic range
        - Clipping applied to values exceeding max_value
    """
    warnings.warn(
        _DEPRECATION_MSG.format(func_name="quantize_to_fp8_e4m3", alt_func="quantize_to_mxfp8"),
        DeprecationWarning,
        stacklevel=2,
    )
    original_shape = weight.shape
    weight_flat = weight.reshape(-1)
    n_elements = weight_flat.size

    # Pad to multiple of group_size
    n_groups = (n_elements + group_size - 1) // group_size
    padded_size = n_groups * group_size
    if padded_size > n_elements:
        weight_flat = mx.pad(weight_flat, [(0, padded_size - n_elements)])

    # Reshape into groups
    weight_grouped = weight_flat.reshape(n_groups, group_size)

    # Compute per-group scales
    scales = mx.max(mx.abs(weight_grouped), axis=1, keepdims=True)
    scales = mx.maximum(scales, 1e-10)  # Avoid division by zero

    # Normalize by scales
    weight_normalized = weight_grouped / scales

    # Clip to FP8 E4M3 range
    weight_clipped = mx.clip(weight_normalized, -max_value, max_value)

    # Simulate FP8 precision by rounding to fewer mantissa bits
    # E4M3 has 3 mantissa bits, so we quantize to 8 levels between powers of 2
    # This is a simplified simulation - real FP8 has more complex rounding
    scale_factor = 8.0  # 2^3 mantissa bits
    weight_quantized = mx.round(weight_clipped * scale_factor) / scale_factor

    # Flatten and truncate to original size
    quantized_flat = weight_quantized.reshape(-1)[:n_elements]
    quantized_result = quantized_flat.reshape(original_shape)

    # Remove keepdims from scales
    scales_flat = scales.squeeze(axis=1)

    return quantized_result.astype(mx.float16), scales_flat.astype(mx.float16)


def quantize_to_fp8_e5m2(
    weight: mx.array,
    group_size: int = 64,
    max_value: float = 57344.0,  # E5M2 max representable value
) -> tuple[mx.array, mx.array]:
    """
    Quantize weights to FP8 E5M2 format (5-bit exp, 2-bit mantissa).

    ⚠️  DEPRECATED: This is simulated FP8. Use smlx.quant.mxfp8 for true 8-bit storage.

    E5M2 provides better range but lower precision compared to E4M3.
    Range: ±57344.0 with subnormal support.

    Args:
        weight: Weight array to quantize
        group_size: Group size for per-group scaling (default: 64)
        max_value: Maximum representable value in E5M2 (default: 57344.0)

    Returns:
        Tuple of (quantized_values, scales)
        - quantized_values: Approximated FP8 values as float16 (NOT true 8-bit!)
        - scales: Per-group scaling factors

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_to_fp8_e5m2

        weights = mx.random.normal((768, 768))
        fp8_values, scales = quantize_to_fp8_e5m2(weights)
        ```

    Notes:
        - ⚠️ Uses simulated FP8 (stored as float16) - NO memory savings
        - Better for wide dynamic range (gradients, activations)
        - Less precision than E4M3
        - Commonly used for backward pass in training
    """
    warnings.warn(
        _DEPRECATION_MSG.format(func_name="quantize_to_fp8_e5m2", alt_func="quantize_to_mxfp8"),
        DeprecationWarning,
        stacklevel=2,
    )
    original_shape = weight.shape
    weight_flat = weight.reshape(-1)
    n_elements = weight_flat.size

    # Pad to multiple of group_size
    n_groups = (n_elements + group_size - 1) // group_size
    padded_size = n_groups * group_size
    if padded_size > n_elements:
        weight_flat = mx.pad(weight_flat, [(0, padded_size - n_elements)])

    # Reshape into groups
    weight_grouped = weight_flat.reshape(n_groups, group_size)

    # Compute per-group scales
    scales = mx.max(mx.abs(weight_grouped), axis=1, keepdims=True)
    scales = mx.maximum(scales, 1e-10)

    # Normalize by scales
    weight_normalized = weight_grouped / scales

    # Clip to FP8 E5M2 range
    weight_clipped = mx.clip(weight_normalized, -max_value, max_value)

    # Simulate FP8 precision with 2 mantissa bits (4 levels between powers of 2)
    scale_factor = 4.0  # 2^2 mantissa bits
    weight_quantized = mx.round(weight_clipped * scale_factor) / scale_factor

    # Flatten and truncate to original size
    quantized_flat = weight_quantized.reshape(-1)[:n_elements]
    quantized_result = quantized_flat.reshape(original_shape)

    # Remove keepdims from scales
    scales_flat = scales.squeeze(axis=1)

    return quantized_result.astype(mx.float16), scales_flat.astype(mx.float16)


def dequantize_from_fp8(
    quantized_values: mx.array, scales: mx.array, group_size: int = 64
) -> mx.array:
    """
    Dequantize FP8 weights back to full precision.

    ⚠️  DEPRECATED: Use smlx.quant.mxfp8.dequantize_from_mxfp8 instead.

    Args:
        quantized_values: Simulated FP8 values (float16)
        scales: Per-group scaling factors
        group_size: Group size used for quantization (default: 64)

    Returns:
        Dequantized weight array

    Example:
        ```python
        from smlx.quant import quantize_to_fp8_e4m3, dequantize_from_fp8

        # Quantize
        fp8_values, scales = quantize_to_fp8_e4m3(weights)

        # Dequantize
        weights_restored = dequantize_from_fp8(fp8_values, scales)
        ```
    """
    warnings.warn(
        _DEPRECATION_MSG.format(
            func_name="dequantize_from_fp8", alt_func="dequantize_from_mxfp8"
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    original_shape = quantized_values.shape
    values_flat = quantized_values.reshape(-1)
    n_elements = values_flat.size

    # Pad to multiple of group_size
    n_groups = (n_elements + group_size - 1) // group_size
    padded_size = n_groups * group_size
    if padded_size > n_elements:
        values_flat = mx.pad(values_flat, [(0, padded_size - n_elements)])

    # Reshape into groups
    values_grouped = values_flat.reshape(n_groups, group_size)

    # Apply scales (broadcast over group_size dimension)
    scales_expanded = scales[:, None]  # (n_groups, 1)
    dequantized = values_grouped * scales_expanded

    # Flatten and truncate to original size
    dequantized_flat = dequantized.reshape(-1)[:n_elements]
    dequantized_result = dequantized_flat.reshape(original_shape)

    return dequantized_result


def quantize_model_fp8(
    model, format: str = "e4m3", group_size: int = 64
) -> dict[str, tuple]:
    """
    Quantize model to FP8 format (experimental).

    ⚠️  DEPRECATED: Use smlx.quant.mxfp8.quantize_model_mxfp8 instead.

    Args:
        model: MLX model to quantize
        format: FP8 format ("e4m3" or "e5m2", default: "e4m3")
        group_size: Group size for quantization (default: 64)

    Returns:
        Dictionary mapping layer names to (quantized_values, scales) tuples

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_model_fp8

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        quantized_weights = quantize_model_fp8(model, format="e4m3")
        ```

    Notes:
        - ⚠️ Returns simulated FP8 (stored as float16) - NO true memory savings
        - E4M3: Better precision, use for forward pass
        - E5M2: Better range, use for backward pass (gradients)
        - This is experimental - use quantize_model_mxfp8 for production
    """
    warnings.warn(
        _DEPRECATION_MSG.format(
            func_name="quantize_model_fp8", alt_func="quantize_model_mxfp8"
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    if format not in ["e4m3", "e5m2"]:
        raise ValueError(f"Unsupported FP8 format: {format}. Use 'e4m3' or 'e5m2'.")

    quantize_fn = quantize_to_fp8_e4m3 if format == "e4m3" else quantize_to_fp8_e5m2
    quantized_weights = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)) and hasattr(module, "weight"):
            weight = module.weight
            values, scales = quantize_fn(weight, group_size=group_size)
            quantized_weights[name] = (values, scales)

    return quantized_weights


def estimate_fp8_size(model, group_size: int = 64) -> dict:
    """
    Estimate memory usage with FP8 quantization.

    ⚠️  DEPRECATED: Use smlx.quant.mxfp8.estimate_mxfp8_size instead.

    Args:
        model: MLX model
        group_size: Group size for quantization (default: 64)

    Returns:
        Dictionary with size estimates:
        - current_mb: Current model size in MB
        - fp8_mb: Size after FP8 quantization (simulated, NOT true 8-bit!)
        - reduction_ratio: Size reduction factor
        - saved_mb: MB saved

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import estimate_fp8_size

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        stats = estimate_fp8_size(model)
        print(f"FP8 reduces to {stats['fp8_mb']:.1f} MB")
        ```

    Notes:
        - ⚠️ Simulated FP8 stored as float16, so ~2x reduction from FP32 (NOT true 8-bit!)
        - For true 8-bit storage, use estimate_mxfp8_size instead
    """
    warnings.warn(
        _DEPRECATION_MSG.format(func_name="estimate_fp8_size", alt_func="estimate_mxfp8_size"),
        DeprecationWarning,
        stacklevel=2,
    )
    total_bytes = 0
    quantizable_params = 0

    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            weight = module.weight
            total_bytes += weight.nbytes

            if isinstance(module, (nn.Linear, nn.Embedding)):
                quantizable_params += weight.size

    # FP8 size (simulated as float16 + scales):
    # - Values: float16 (2 bytes per param)
    # - Scales: float16 per group (2 bytes)
    values_bytes = quantizable_params * 2  # Simulated FP8 as float16
    n_groups = (quantizable_params + group_size - 1) // group_size
    scales_bytes = n_groups * 2  # float16

    fp8_bytes = values_bytes + scales_bytes

    # Add non-quantizable parameters
    non_quantizable_bytes = total_bytes - (quantizable_params * 2)  # Assume FP16
    fp8_total = fp8_bytes + non_quantizable_bytes

    current_mb = total_bytes / (1024**2)
    fp8_mb = fp8_total / (1024**2)
    reduction_ratio = current_mb / fp8_mb if fp8_mb > 0 else 1.0

    return {
        "current_mb": current_mb,
        "fp8_mb": fp8_mb,
        "reduction_ratio": reduction_ratio,
        "saved_mb": current_mb - fp8_mb,
        "quantizable_params": quantizable_params,
        "note": "Simulated FP8 (stored as float16). Native FP8 would be 2x smaller.",
    }


def compare_fp8_formats(weight: mx.array, group_size: int = 64) -> dict:
    """
    Compare E4M3 vs E5M2 FP8 formats.

    Args:
        weight: Weight array to compare
        group_size: Group size for both formats

    Returns:
        Dictionary with comparison metrics:
        - e4m3_error: Mean absolute error for E4M3
        - e5m2_error: Mean absolute error for E5M2
        - e4m3_range: Effective range with E4M3
        - e5m2_range: Effective range with E5M2
        - recommendation: Which format to use

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import compare_fp8_formats

        weights = mx.random.normal((768, 768))
        comparison = compare_fp8_formats(weights)
        print(comparison['recommendation'])
        ```
    """
    # E4M3 quantization
    values_e4m3, scales_e4m3 = quantize_to_fp8_e4m3(weight, group_size=group_size)
    restored_e4m3 = dequantize_from_fp8(values_e4m3, scales_e4m3, group_size=group_size)
    e4m3_error = mx.mean(mx.abs(restored_e4m3 - weight)).item()

    # E5M2 quantization
    values_e5m2, scales_e5m2 = quantize_to_fp8_e5m2(weight, group_size=group_size)
    restored_e5m2 = dequantize_from_fp8(values_e5m2, scales_e5m2, group_size=group_size)
    e5m2_error = mx.mean(mx.abs(restored_e5m2 - weight)).item()

    # Calculate effective ranges
    weight_range = mx.max(mx.abs(weight)).item()
    e4m3_max = 448.0
    e5m2_max = 57344.0

    # Determine recommendation
    if e4m3_error < e5m2_error * 0.9 and weight_range < e4m3_max:
        recommendation = "E4M3 (better precision, sufficient range)"
    elif e5m2_error < e4m3_error * 0.9:
        recommendation = "E5M2 (better for this distribution)"
    elif weight_range > e4m3_max:
        recommendation = "E5M2 required (exceeds E4M3 range)"
    else:
        recommendation = "E4M3 (default choice for weights)"

    return {
        "e4m3_error": e4m3_error,
        "e5m2_error": e5m2_error,
        "e4m3_better": e4m3_error < e5m2_error,
        "weight_range": weight_range,
        "e4m3_max": e4m3_max,
        "e5m2_max": e5m2_max,
        "recommendation": recommendation,
        "use_cases": {
            "e4m3": "Forward pass, weight storage, high precision needed",
            "e5m2": "Backward pass, gradients, wide dynamic range",
        },
    }


def compare_fp8_vs_int8(weight: mx.array, group_size: int = 64) -> dict:
    """
    Compare FP8 vs INT8 quantization.

    Args:
        weight: Weight array to compare
        group_size: Group size for both methods

    Returns:
        Dictionary with comparison results

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import compare_fp8_vs_int8

        weights = mx.random.normal((768, 768))
        comparison = compare_fp8_vs_int8(weights)
        print(f"FP8 error: {comparison['fp8_error']:.4f}")
        print(f"INT8 error: {comparison['int8_error']:.4f}")
        ```
    """
    # FP8 E4M3 quantization
    values_fp8, scales_fp8 = quantize_to_fp8_e4m3(weight, group_size=group_size)
    restored_fp8 = dequantize_from_fp8(values_fp8, scales_fp8, group_size=group_size)
    fp8_error = mx.mean(mx.abs(restored_fp8 - weight)).item()

    # INT8 quantization (using MLX built-in)
    w_q_int8, scales_int8, biases_int8 = mx.quantize(weight, group_size=group_size, bits=8)
    restored_int8 = mx.dequantize(
        w_q_int8, scales_int8, biases_int8, group_size=group_size, bits=8
    )
    int8_error = mx.mean(mx.abs(restored_int8 - weight)).item()

    # Determine recommendation
    if fp8_error < int8_error * 0.9:
        recommendation = "FP8 (better quality for wide dynamic range)"
    elif int8_error < fp8_error * 0.9:
        recommendation = "INT8 (better quality, use with GPTQ/AWQ)"
    else:
        recommendation = "Similar quality - use INT8 for better hardware support"

    return {
        "fp8_error": fp8_error,
        "int8_error": int8_error,
        "fp8_better": fp8_error < int8_error,
        "recommendation": recommendation,
        "notes": {
            "fp8": "Better for gradients and activations with wide range",
            "int8": "Better hardware support, use with advanced quantization (AWQ/GPTQ)",
        },
    }


__all__ = [
    "quantize_to_fp8_e4m3",
    "quantize_to_fp8_e5m2",
    "dequantize_from_fp8",
    "quantize_model_fp8",
    "estimate_fp8_size",
    "compare_fp8_formats",
    "compare_fp8_vs_int8",
]
