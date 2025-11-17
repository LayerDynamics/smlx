"""
FP4 (4-bit Floating Point) quantization for SMLX.

This module provides comprehensive 4-bit quantization support with multiple formats:

**FP4 Formats:**

1. **E2M1** (Standard FP4: 1 sign, 2 exponent, 1 mantissa):
   - Range: ±0.5 to ±6.0 with exponential spacing
   - Best for: Wide dynamic range, non-uniform distributions
   - Implementation: Simulated (lookup table + scaling)
   - Group size: Flexible (default 64)

2. **MXFP4** (Microscaling FP4 - MLX Native):
   - Same E2M1 format, hardware accelerated
   - Group size: 32 (fixed requirement)
   - Best for: Production use with MLX acceleration
   - Performance: Direct computation on quantized weights

3. **NVFP4** (NVIDIA FP4 - MLX Native):
   - Same E2M1 format, NVIDIA GPU optimized
   - Group size: 16 (fixed requirement)
   - Best for: NVIDIA hardware acceleration
   - Performance: Optimized for NVIDIA GPUs

4. **NF4** (Normal Float 4 - QLoRA):
   - Information-theoretically optimal for N(0,1) distributions
   - Non-uniform quantization (higher precision near zero)
   - Best for: Neural network weights, QLoRA-style fine-tuning
   - Group size: Flexible (default 64)

**Quick Start:**

```python
import mlx.core as mx
from smlx.quant import quantize_fp4, dequantize_fp4

weight = mx.random.normal((768, 768))

# E2M1 (flexible group size, simulation)
q, s = quantize_fp4(weight, mode="e2m1", group_size=64)
restored = dequantize_fp4(q, s, mode="e2m1", group_size=64)

# MXFP4 (hardware accelerated, best performance)
q, s = quantize_fp4(weight, mode="mxfp4")
restored = dequantize_fp4(q, s, mode="mxfp4")

# NF4 (optimal for neural network weights)
q, s = quantize_fp4(weight, mode="nf4", group_size=64)
restored = dequantize_fp4(q, s, mode="nf4", group_size=64)
```

**When to Use Each Mode:**

- **e2m1**: Research, analysis, flexible group sizes, custom experiments
- **mxfp4**: Production inference (hardware accelerated, best speed)
- **nvfp4**: NVIDIA GPU inference (hardware optimized)
- **nf4**: QLoRA fine-tuning, normally distributed weights
- **INT4**: Best overall hardware support (see `smlx.quant.gptq`, `smlx.quant.awq`)

**Performance Notes:**

- MXFP4/NVFP4: Direct computation on quantized weights (fastest)
- E2M1: Requires dequantization before use (most flexible)
- NF4: Requires dequantization, optimal quality for normal distributions
- FP4 is ~78× faster to quantize than INT4 GPTQ
- FP4 better for long-tail distributions and large outliers
- INT4 better for broader hardware support
"""

from enum import Enum

import mlx.core as mx
import mlx.nn as nn


class FP4Mode(str, Enum):
    """FP4 quantization modes."""

    E2M1 = "e2m1"  # Standard FP4 simulation (flexible group size)
    MXFP4 = "mxfp4"  # MLX native, group_size=32 (hardware accelerated)
    NVFP4 = "nvfp4"  # MLX native, group_size=16 (NVIDIA optimized)
    NF4 = "nf4"  # QLoRA Normal Float 4 (information-theoretically optimal)


# FP4 E2M1 format lookup table (2-bit exponent, 1-bit mantissa)
# Format: sign(1) | exp(2) | mantissa(1)
# Values from -6.0 to +6.0 with exponential spacing
FP4_E2M1_VALUES = mx.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,  # Positive values
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,  # Negative values
    ]
)

# NF4 (Normal Float 4) lookup table from QLoRA
# Information-theoretically optimal quantization for N(0,1) distributions
# Non-uniform spacing with higher precision near zero
NF4_VALUES = mx.array(
    [
        -1.0000,
        -0.6962,
        -0.5251,
        -0.3949,
        -0.2844,
        -0.1848,
        -0.0911,
        0.0000,
        0.0796,
        0.1609,
        0.2461,
        0.3379,
        0.4407,
        0.5626,
        0.7230,
        1.0000,
    ]
)


def _quantize_to_e2m1_simulated(
    weight: mx.array, group_size: int = 64
) -> tuple[mx.array, mx.array]:
    """
    Quantize weights to FP4 E2M1 format (simulated via lookup table).

    This is an internal implementation that simulates E2M1 format.
    For public API, use quantize_fp4() with mode="e2m1".

    Args:
        weight: Weight array to quantize
        group_size: Group size for per-group scaling (default: 64)

    Returns:
        Tuple of (quantized_indices, scales)
        - quantized_indices: uint8 array with FP4 indices (0-15)
        - scales: Per-group scaling factors

    Notes:
        - Uses nearest neighbor quantization to FP4 E2M1 values
        - Per-group scaling improves accuracy
        - Stored as indices (uint8) + scales (float16)
    """

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

    # Compute per-group scales (max absolute value)
    scales = mx.max(mx.abs(weight_grouped), axis=1, keepdims=True)
    scales = mx.maximum(scales, 1e-8)  # Avoid division by zero

    # Normalize by scales
    weight_normalized = weight_grouped / scales

    # Quantize to nearest FP4 value
    # Expand dimensions for broadcasting: (n_groups, group_size, 1) vs (16,)
    weight_expanded = weight_normalized[:, :, None]  # (n_groups, group_size, 1)
    fp4_values_expanded = FP4_E2M1_VALUES[None, None, :]  # (1, 1, 16)

    # Find nearest FP4 value
    distances = mx.abs(weight_expanded - fp4_values_expanded)  # (n_groups, group_size, 16)
    indices = mx.argmin(distances, axis=2)  # (n_groups, group_size)

    # Flatten and truncate to original size
    indices_flat = indices.reshape(-1)[:n_elements]
    indices_result = indices_flat.reshape(original_shape)

    # Remove keepdims from scales
    scales_flat = scales.squeeze(axis=1)

    return indices_result.astype(mx.uint8), scales_flat.astype(mx.float16)


def _dequantize_from_e2m1_simulated(
    indices: mx.array, scales: mx.array, group_size: int = 64
) -> mx.array:
    """
    Dequantize FP4 E2M1 weights back to floating point (simulated).

    This is an internal implementation. For public API, use dequantize_fp4()
    with mode="e2m1".

    Args:
        indices: uint8 array with FP4 indices (0-15)
        scales: Per-group scaling factors
        group_size: Group size used for quantization (default: 64)

    Returns:
        Dequantized weight array in float16
    """

    original_shape = indices.shape
    indices_flat = indices.reshape(-1)
    n_elements = indices_flat.size

    # Pad to multiple of group_size
    n_groups = (n_elements + group_size - 1) // group_size
    padded_size = n_groups * group_size
    if padded_size > n_elements:
        indices_flat = mx.pad(indices_flat, [(0, padded_size - n_elements)])

    # Reshape into groups
    indices_grouped = indices_flat.reshape(n_groups, group_size)

    # Lookup FP4 values
    fp4_weights = FP4_E2M1_VALUES[indices_grouped]  # (n_groups, group_size)

    # Apply scales (broadcast over group_size dimension)
    scales_expanded = scales[:, None]  # (n_groups, 1)
    dequantized = fp4_weights * scales_expanded

    # Flatten and truncate to original size
    dequantized_flat = dequantized.reshape(-1)[:n_elements]
    dequantized_result = dequantized_flat.reshape(original_shape)

    return dequantized_result.astype(mx.float16)


# ============================================================================
# NF4 (Normal Float 4) - QLoRA Quantization
# ============================================================================


def _quantize_to_nf4(weight: mx.array, group_size: int = 64) -> tuple[mx.array, mx.array]:
    """
    Quantize weights to NF4 format (Normal Float 4 from QLoRA).

    NF4 is information-theoretically optimal for normally distributed weights.
    Uses non-uniform quantization with higher precision near zero.

    Args:
        weight: Weight array to quantize
        group_size: Group size for per-group scaling (default: 64)

    Returns:
        Tuple of (quantized_indices, scales)
        - quantized_indices: uint8 array with NF4 indices (0-15)
        - scales: Per-group max absolute value scales
    """
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

    # Compute per-group scales (max absolute value)
    scales = mx.max(mx.abs(weight_grouped), axis=1, keepdims=True)
    scales = mx.maximum(scales, 1e-8)  # Avoid division by zero

    # Normalize to [-1, 1] range
    weight_normalized = weight_grouped / scales

    # Quantize to nearest NF4 value
    weight_expanded = weight_normalized[:, :, None]  # (n_groups, group_size, 1)
    nf4_values_expanded = NF4_VALUES[None, None, :]  # (1, 1, 16)

    # Find nearest NF4 value
    distances = mx.abs(weight_expanded - nf4_values_expanded)
    indices = mx.argmin(distances, axis=2)

    # Flatten and truncate to original size
    indices_flat = indices.reshape(-1)[:n_elements]
    indices_result = indices_flat.reshape(original_shape)

    # Remove keepdims from scales
    scales_flat = scales.squeeze(axis=1)

    return indices_result.astype(mx.uint8), scales_flat.astype(mx.float16)


def _dequantize_from_nf4(indices: mx.array, scales: mx.array, group_size: int = 64) -> mx.array:
    """
    Dequantize NF4 weights back to floating point.

    Args:
        indices: uint8 array with NF4 indices (0-15)
        scales: Per-group scaling factors
        group_size: Group size used for quantization (default: 64)

    Returns:
        Dequantized weight array in float16
    """
    original_shape = indices.shape
    indices_flat = indices.reshape(-1)
    n_elements = indices_flat.size

    # Pad to multiple of group_size
    n_groups = (n_elements + group_size - 1) // group_size
    padded_size = n_groups * group_size
    if padded_size > n_elements:
        indices_flat = mx.pad(indices_flat, [(0, padded_size - n_elements)])

    # Reshape into groups
    indices_grouped = indices_flat.reshape(n_groups, group_size)

    # Lookup NF4 values
    nf4_weights = NF4_VALUES[indices_grouped]

    # Apply scales
    scales_expanded = scales[:, None]  # (n_groups, 1)
    dequantized = nf4_weights * scales_expanded

    # Flatten and truncate to original size
    dequantized_flat = dequantized.reshape(-1)[:n_elements]
    dequantized_result = dequantized_flat.reshape(original_shape)

    return dequantized_result.astype(mx.float16)


# ============================================================================
# MLX Native FP4 (MXFP4 and NVFP4)
# ============================================================================


def _quantize_to_mxfp4(weight: mx.array) -> tuple[mx.array, mx.array]:
    """
    Quantize weights using MLX native MXFP4 (Microscaling FP4).

    MXFP4 provides hardware acceleration with fixed group_size=32.

    Args:
        weight: Weight array to quantize

    Returns:
        Tuple of (quantized_weights, scales)
        - quantized_weights: MXFP4 quantized array
        - scales: Per-group scales (group_size=32)

    Notes:
        - Group size is fixed at 32 for MXFP4
        - Uses E2M1 format (same as simulated)
        - Hardware accelerated on supported devices
        - MLX quantize returns (quantized, scales) for mode="mxfp4" (no biases)
    """
    quantized, scales = mx.quantize(weight, mode="mxfp4")
    return quantized, scales


def _dequantize_from_mxfp4(
    quantized: mx.array, scales: mx.array, group_size: int = 32
) -> mx.array:
    """
    Dequantize MXFP4 weights back to floating point.

    Args:
        quantized: MXFP4 quantized array
        scales: Per-group scales
        group_size: Group size (must be 32 for MXFP4)

    Returns:
        Dequantized weight array
    """
    if group_size != 32:
        raise ValueError(f"MXFP4 requires group_size=32, got {group_size}")
    # MXFP4 mode requires mode parameter for dequantization
    return mx.dequantize(quantized, scales, group_size=32, bits=4, mode="mxfp4")


def _quantize_to_nvfp4(weight: mx.array) -> tuple[mx.array, mx.array]:
    """
    Quantize weights using MLX native NVFP4 (NVIDIA FP4).

    NVFP4 is optimized for NVIDIA hardware with fixed group_size=16.

    Args:
        weight: Weight array to quantize

    Returns:
        Tuple of (quantized_weights, scales)
        - quantized_weights: NVFP4 quantized array
        - scales: Per-group scales (group_size=16)

    Notes:
        - Group size is fixed at 16 for NVFP4
        - Uses E4M3 scale format (different from E2M1)
        - Optimized for NVIDIA GPUs
        - MLX quantize returns (quantized, scales) for mode="nvfp4" (no biases)
    """
    quantized, scales = mx.quantize(weight, mode="nvfp4")
    return quantized, scales


def _dequantize_from_nvfp4(
    quantized: mx.array, scales: mx.array, group_size: int = 16
) -> mx.array:
    """
    Dequantize NVFP4 weights back to floating point.

    Args:
        quantized: NVFP4 quantized array
        scales: Per-group scales
        group_size: Group size (must be 16 for NVFP4)

    Returns:
        Dequantized weight array
    """
    if group_size != 16:
        raise ValueError(f"NVFP4 requires group_size=16, got {group_size}")
    # NVFP4 mode requires mode parameter for dequantization
    return mx.dequantize(quantized, scales, group_size=16, bits=4, mode="nvfp4")


# ============================================================================
# Unified Public API
# ============================================================================


def quantize_fp4(
    weight: mx.array,
    mode: str = "e2m1",
    group_size: int = 64,
) -> tuple[mx.array, mx.array]:
    """
    Quantize weights to FP4 format with multiple mode support.

    Supports four FP4 quantization modes:
    - **e2m1**: Standard FP4 simulation (flexible group size)
    - **mxfp4**: MLX native MXFP4 (group_size=32, hardware accelerated)
    - **nvfp4**: MLX native NVFP4 (group_size=16, NVIDIA optimized)
    - **nf4**: QLoRA Normal Float 4 (optimal for N(0,1) distributions)

    Args:
        weight: Weight array to quantize
        mode: Quantization mode ("e2m1", "mxfp4", "nvfp4", or "nf4")
        group_size: Group size for per-group scaling (ignored for mxfp4/nvfp4)

    Returns:
        Tuple of (quantized_data, scales)
        - quantized_data: Quantized weights (format depends on mode)
        - scales: Per-group scaling factors

    Examples:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_fp4, dequantize_fp4

        weight = mx.random.normal((768, 768))

        # E2M1 simulation (flexible group size)
        q, s = quantize_fp4(weight, mode="e2m1", group_size=64)
        restored = dequantize_fp4(q, s, mode="e2m1", group_size=64)

        # MXFP4 (hardware accelerated, group_size=32)
        q, s = quantize_fp4(weight, mode="mxfp4")
        restored = dequantize_fp4(q, s, mode="mxfp4")

        # NF4 for QLoRA-style fine-tuning
        q, s = quantize_fp4(weight, mode="nf4", group_size=64)
        restored = dequantize_fp4(q, s, mode="nf4", group_size=64)
        ```

    Notes:
        - **e2m1**: Best for research, flexible group sizes, requires dequantization
        - **mxfp4**: Best for production, hardware accelerated, group_size fixed at 32
        - **nvfp4**: Best for NVIDIA GPUs, hardware accelerated, group_size fixed at 16
        - **nf4**: Best for QLoRA fine-tuning, optimal for normal distributions
    """
    mode_lower = mode.lower()

    if mode_lower == FP4Mode.E2M1:
        return _quantize_to_e2m1_simulated(weight, group_size)
    elif mode_lower == FP4Mode.MXFP4:
        if group_size != 32:
            print(f"Warning: MXFP4 requires group_size=32, ignoring provided value {group_size}")
        return _quantize_to_mxfp4(weight)
    elif mode_lower == FP4Mode.NVFP4:
        if group_size != 16:
            print(f"Warning: NVFP4 requires group_size=16, ignoring provided value {group_size}")
        return _quantize_to_nvfp4(weight)
    elif mode_lower == FP4Mode.NF4:
        return _quantize_to_nf4(weight, group_size)
    else:
        raise ValueError(
            f"Unsupported FP4 mode: {mode}. "
            f"Supported modes: {[m.value for m in FP4Mode]}"
        )


def dequantize_fp4(
    quantized: mx.array,
    scales: mx.array,
    mode: str = "e2m1",
    group_size: int | None = None,
) -> mx.array:
    """
    Dequantize FP4 weights back to floating point.

    Args:
        quantized: Quantized weight data
        scales: Per-group scaling factors
        mode: Quantization mode used ("e2m1", "mxfp4", "nvfp4", or "nf4")
        group_size: Group size used for quantization (default: auto-set based on mode)
                   - MXFP4: Must be 32 (auto-set if None)
                   - NVFP4: Must be 16 (auto-set if None)
                   - E2M1/NF4: Defaults to 64 if None

    Returns:
        Dequantized weight array in float16

    Examples:
        ```python
        from smlx.quant import quantize_fp4, dequantize_fp4

        # Quantize
        q, s = quantize_fp4(weight, mode="e2m1", group_size=64)

        # Dequantize (group_size auto-set based on mode)
        restored = dequantize_fp4(q, s, mode="e2m1")

        # Or explicitly specify
        restored = dequantize_fp4(q, s, mode="e2m1", group_size=64)
        ```

    See Also:
        quantize_fp4: Quantize weights to FP4
    """
    mode_lower = mode.lower()

    # Auto-set group_size based on mode if not provided
    if group_size is None:
        if mode_lower == FP4Mode.MXFP4:
            group_size = 32
        elif mode_lower == FP4Mode.NVFP4:
            group_size = 16
        else:
            group_size = 64  # Default for E2M1 and NF4

    if mode_lower == FP4Mode.E2M1:
        return _dequantize_from_e2m1_simulated(quantized, scales, group_size)
    elif mode_lower == FP4Mode.MXFP4:
        return _dequantize_from_mxfp4(quantized, scales, group_size=group_size)
    elif mode_lower == FP4Mode.NVFP4:
        return _dequantize_from_nvfp4(quantized, scales, group_size=group_size)
    elif mode_lower == FP4Mode.NF4:
        return _dequantize_from_nf4(quantized, scales, group_size)
    else:
        raise ValueError(
            f"Unsupported FP4 mode: {mode}. "
            f"Supported modes: {[m.value for m in FP4Mode]}"
        )


# Legacy aliases for backward compatibility
def quantize_to_fp4(
    weight: mx.array, group_size: int = 64, format: str = "e2m1"
) -> tuple[mx.array, mx.array]:
    """Legacy function. Use quantize_fp4() instead."""
    return quantize_fp4(weight, mode=format, group_size=group_size)


def dequantize_from_fp4(
    indices: mx.array, scales: mx.array, group_size: int = 64, format: str = "e2m1"
) -> mx.array:
    """Legacy function. Use dequantize_fp4() instead."""
    return dequantize_fp4(indices, scales, mode=format, group_size=group_size)


def quantize_model_fp4(
    model,
    mode: str = "e2m1",
    group_size: int = 64,
) -> dict[str, tuple]:
    """
    Quantize model to FP4 format (experimental).

    Note: This returns quantization artifacts but does not modify the model.
    For E2M1/NF4, returns (indices, scales). For MXFP4/NVFP4, returns (quantized, scales).

    Args:
        model: MLX model to quantize
        mode: FP4 quantization mode ("e2m1", "mxfp4", "nvfp4", or "nf4")
        group_size: Group size for quantization (default: 64, ignored for mxfp4/nvfp4)

    Returns:
        Dictionary mapping layer names to (quantized_data, scales) tuples

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_model_fp4, dequantize_fp4

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")

        # Quantize with E2M1
        quantized_weights = quantize_model_fp4(model, mode="e2m1", group_size=64)

        # Quantize with MXFP4 (hardware accelerated)
        quantized_weights = quantize_model_fp4(model, mode="mxfp4")

        # Dequantize a specific layer
        layer_name = "layers.0.self_attn.q_proj"
        q, s = quantized_weights[layer_name]
        restored = dequantize_fp4(q, s, mode="e2m1", group_size=64)
        ```

    Notes:
        - This is an experimental feature for research and analysis
        - Returns quantized weights for offline storage
        - Runtime dequantization required for inference
        - MXFP4/NVFP4 modes support hardware-accelerated inference
    """
    quantized_weights = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)) and hasattr(module, "weight"):
            weight = module.weight
            q_data, scales = quantize_fp4(weight, mode=mode, group_size=group_size)
            quantized_weights[name] = (q_data, scales)

    return quantized_weights


def estimate_fp4_size(model, group_size: int = 64) -> dict:
    """
    Estimate memory usage with FP4 quantization.

    Args:
        model: MLX model
        group_size: Group size for quantization (default: 64)

    Returns:
        Dictionary with size estimates:
        - current_mb: Current model size in MB
        - fp4_mb: Size after FP4 quantization
        - reduction_ratio: Size reduction factor
        - saved_mb: MB saved

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import estimate_fp4_size

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        stats = estimate_fp4_size(model)
        print(f"FP4 reduces to {stats['fp4_mb']:.1f} MB")
        ```
    """
    total_bytes = 0
    quantizable_params = 0

    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            weight = module.weight
            total_bytes += weight.nbytes

            if isinstance(module, (nn.Linear, nn.Embedding)):
                quantizable_params += weight.size

    # FP4 size calculation:
    # - Indices: 4 bits per weight (stored as uint8, so 1 byte per 2 weights)
    # - Scales: float16 per group (2 bytes)
    indices_bytes = quantizable_params // 2  # 4 bits = 0.5 bytes (stored as uint8)
    n_groups = (quantizable_params + group_size - 1) // group_size
    scales_bytes = n_groups * 2  # float16 = 2 bytes

    fp4_bytes = indices_bytes + scales_bytes

    # Add non-quantizable parameters (assume they stay as-is)
    non_quantizable_bytes = total_bytes - (quantizable_params * 2)  # Assume FP16
    fp4_total = fp4_bytes + non_quantizable_bytes

    current_mb = total_bytes / (1024**2)
    fp4_mb = fp4_total / (1024**2)
    reduction_ratio = current_mb / fp4_mb if fp4_mb > 0 else 1.0

    return {
        "current_mb": current_mb,
        "fp4_mb": fp4_mb,
        "reduction_ratio": reduction_ratio,
        "saved_mb": current_mb - fp4_mb,
        "quantizable_params": quantizable_params,
        "format": "E2M1 (2-bit exp, 1-bit mantissa)",
    }


def compare_fp4_vs_int4(weight: mx.array, group_size: int = 64) -> dict:
    """
    Compare FP4 vs INT4 quantization quality.

    Args:
        weight: Weight array to compare
        group_size: Group size for both methods

    Returns:
        Dictionary with comparison metrics:
        - fp4_error: Mean absolute error for FP4
        - int4_error: Mean absolute error for INT4
        - fp4_max_error: Max error for FP4
        - int4_max_error: Max error for INT4
        - recommendation: Which format is better for this weight

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import compare_fp4_vs_int4

        weights = mx.random.normal((768, 768))
        comparison = compare_fp4_vs_int4(weights)
        print(f"FP4 error: {comparison['fp4_error']:.4f}")
        print(f"INT4 error: {comparison['int4_error']:.4f}")
        ```
    """
    # FP4 quantization
    indices_fp4, scales_fp4 = quantize_to_fp4(weight, group_size=group_size)
    restored_fp4 = dequantize_from_fp4(indices_fp4, scales_fp4, group_size=group_size)
    fp4_error = mx.mean(mx.abs(restored_fp4 - weight)).item()
    fp4_max_error = mx.max(mx.abs(restored_fp4 - weight)).item()

    # INT4 quantization (using MLX built-in)
    w_q_int4, scales_int4, biases_int4 = mx.quantize(weight, group_size=group_size, bits=4)
    restored_int4 = mx.dequantize(
        w_q_int4, scales_int4, biases_int4, group_size=group_size, bits=4
    )
    int4_error = mx.mean(mx.abs(restored_int4 - weight)).item()
    int4_max_error = mx.max(mx.abs(restored_int4 - weight)).item()

    # Determine recommendation
    if fp4_error < int4_error * 0.9:  # FP4 significantly better
        recommendation = "FP4 (better quality for wide dynamic range)"
    elif int4_error < fp4_error * 0.9:  # INT4 significantly better
        recommendation = "INT4 (better quality for uniform distribution)"
    else:
        recommendation = "Similar quality - use INT4 for better hardware support"

    return {
        "fp4_error": fp4_error,
        "int4_error": int4_error,
        "fp4_max_error": fp4_max_error,
        "int4_max_error": int4_max_error,
        "fp4_better": fp4_error < int4_error,
        "improvement_ratio": int4_error / fp4_error if fp4_error > 0 else float("inf"),
        "recommendation": recommendation,
    }


__all__ = [
    # New unified API (recommended)
    "quantize_fp4",
    "dequantize_fp4",
    "FP4Mode",
    # Model-level functions
    "quantize_model_fp4",
    "estimate_fp4_size",
    "compare_fp4_vs_int4",
    # Lookup tables
    "FP4_E2M1_VALUES",
    "NF4_VALUES",
    # Legacy API (backward compatibility)
    "quantize_to_fp4",
    "dequantize_from_fp4",
]
