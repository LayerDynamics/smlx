"""
GGML Q8_0 quantization format for SMLX.

Q8_0 is a simple 8-bit GGML quantization format used in llama.cpp.
Format: 34 bytes per block (2-byte FP16 scale + 32 weights at 8 bits each)
Bias is computed as -128 * scale (not stored).

This format provides better quality than 4-bit formats while still offering ~2x compression.
Compatible with llama.cpp ecosystem.

Compression: ~2x from FP32, ~1x from FP16
Block size: 32 weights
Storage: 1.0625 bytes per weight (34 bytes / 32 weights)
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# GGML Q8_0 Constants
Q8_0_BLOCK_SIZE = 32  # Number of weights per block
Q8_0_BYTES_PER_BLOCK = 34  # 2 (scale) + 32 (weights)
Q8_0_SCALE_BYTES = 2  # FP16 scale
Q8_0_WEIGHT_BYTES = 32  # 32 weights at 8 bits each


def quantize_to_q8_0(
    weight: mx.array,
    block_size: int = Q8_0_BLOCK_SIZE
) -> tuple[mx.array, mx.array]:
    """
    Quantize weight array to GGML Q8_0 format.

    Q8_0 uses symmetric 8-bit quantization with implicit bias:
    - Each block has one FP16 scale
    - Bias = -128 * scale (not stored)
    - Weights quantized to 8-bit unsigned integers [0, 255]
    - Original value H scale * (quantized_weight - 128)

    Args:
        weight: Weight array to quantize (any shape)
        block_size: Block size for quantization (default: 32)

    Returns:
        Tuple of (quantized_weights, scales):
        - quantized_weights: uint8 array
        - scales: Per-block FP16 scaling factors

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_to_q8_0

        weights = mx.random.normal((768, 768))
        w_q, scales = quantize_to_q8_0(weights)
        # w_q is ~2x smaller with minimal quality loss
        ```
    """
    weight_flat = weight.flatten()

    # Pad to multiple of block_size
    remainder = weight_flat.size % block_size
    if remainder != 0:
        padding = block_size - remainder
        weight_flat = mx.concatenate([weight_flat, mx.zeros(padding, dtype=weight.dtype)])

    # Reshape into blocks
    num_blocks = weight_flat.size // block_size
    weight_blocks = weight_flat.reshape(num_blocks, block_size)

    # Compute per-block scale
    # For Q8_0: scale = max(abs(block)) / 128
    # This maps the range to [-128, 127] in quantized space
    max_vals = mx.max(mx.abs(weight_blocks), axis=1, keepdims=True)
    # Avoid division by zero
    scales = mx.maximum(max_vals / 128.0, 1e-10)

    # Quantize: map to [0, 255] range
    # quantized = round((weight / scale) + 128)
    # Bias of -128 * scale is implicit
    quantized = mx.round((weight_blocks / scales) + 128.0)
    quantized = mx.clip(quantized, 0, 255)

    # Convert to uint8
    quantized_uint8 = quantized.astype(mx.uint8)

    # Convert scales to float16
    scales_f16 = scales.squeeze().astype(mx.float16)

    return quantized_uint8, scales_f16


def dequantize_from_q8_0(
    quantized_weight: mx.array,
    scales: mx.array,
    original_shape: Optional[tuple[int, ...]] = None,
    block_size: int = Q8_0_BLOCK_SIZE
) -> mx.array:
    """
    Dequantize from GGML Q8_0 format back to float.

    Args:
        quantized_weight: uint8 array from quantize_to_q8_0
        scales: Per-block FP16 scales
        original_shape: Original weight shape to reshape to (optional)
        block_size: Block size used for quantization (default: 32)

    Returns:
        Dequantized weight array as float32

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_to_q8_0, dequantize_from_q8_0

        weights = mx.random.normal((768, 768))
        w_q, scales = quantize_to_q8_0(weights)
        w_dequant = dequantize_from_q8_0(w_q, scales, weights.shape)
        ```
    """
    # Convert to float
    quantized_float = quantized_weight.astype(mx.float32)

    # Expand scales for broadcasting
    scales_expanded = scales.reshape(-1, 1).astype(mx.float32)

    # Dequantize: weight = scale * (quantized - 128)
    # The implicit bias is -128 * scale
    dequantized = scales_expanded * (quantized_float - 128.0)

    # Flatten and reshape
    dequantized_flat = dequantized.flatten()

    if original_shape is not None:
        # Trim padding if necessary
        original_size = 1
        for dim in original_shape:
            original_size *= dim
        dequantized_flat = dequantized_flat[:original_size]
        dequantized = dequantized_flat.reshape(original_shape)
    else:
        dequantized = dequantized_flat

    return dequantized


def quantize_model_q8_0(
    model: nn.Module,
    block_size: int = Q8_0_BLOCK_SIZE,
    inplace: bool = True
) -> Optional[nn.Module]:
    """
    Quantize entire model to GGML Q8_0 format.

    Args:
        model: MLX model to quantize
        block_size: Block size for quantization (default: 32)
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_model_q8_0

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        quantize_model_q8_0(model)  # ~2x compression, high quality
        ```

    Notes:
        - Quantizes nn.Linear layers only (Embedding layers skipped)
        - Better quality than 4-bit formats
        - ~2x compression from FP16
        - Compatible with GGML/llama.cpp Q8_0 format
        - Good choice for quality-sensitive applications
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight"):
                original_weight = module.weight
                quantized, scales = quantize_to_q8_0(original_weight, block_size)

                # Store quantized weights and scales for reference
                module.weight_q8_0 = quantized
                module.scales_q8_0 = scales
                module.original_shape = original_weight.shape
                module.quantization_format = "q8_0"

                # Replace weight with dequantized version
                # Note: This keeps the model functional but doesn't save runtime memory
                module.weight = dequantize_from_q8_0(
                    quantized, scales, original_weight.shape
                )

    if not inplace:
        return model
    return None


def estimate_q8_0_size(model: nn.Module) -> dict:
    """
    Estimate model size after Q8_0 quantization.

    Args:
        model: MLX model to analyze

    Returns:
        Dictionary with size estimates:
        - original_mb: Original model size in MB
        - q8_0_mb: Estimated size after Q8_0 quantization
        - reduction_ratio: Compression ratio
        - bytes_per_weight: Average bytes per weight

    Example:
        ```python
        from smlx.quant import estimate_q8_0_size

        stats = estimate_q8_0_size(model)
        print(f"Q8_0: {stats['q8_0_mb']:.1f} MB ({stats['reduction_ratio']:.1f}x)")
        ```
    """
    total_params = 0
    quantizable_params = 0
    original_bytes = 0

    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            weight = module.weight
            param_count = weight.size
            total_params += param_count
            original_bytes += weight.nbytes

            if isinstance(module, (nn.Linear, nn.Embedding)):
                quantizable_params += param_count

    # Q8_0 storage: 1.0625 bytes per weight (34 bytes / 32 weights)
    # = 8 bits for weight + 0.5 bits for scale
    bytes_per_weight = Q8_0_BYTES_PER_BLOCK / Q8_0_BLOCK_SIZE
    q8_0_bytes = quantizable_params * bytes_per_weight
    q8_0_bytes += (total_params - quantizable_params) * 2  # Non-quantized params stay FP16

    return {
        "original_mb": original_bytes / (1024**2),
        "q8_0_mb": q8_0_bytes / (1024**2),
        "reduction_ratio": original_bytes / q8_0_bytes if q8_0_bytes > 0 else 1.0,
        "bytes_per_weight": bytes_per_weight,
        "quantizable_params": quantizable_params,
        "total_params": total_params,
    }


def compare_q8_0_vs_int8(weight: mx.array) -> dict:
    """
    Compare Q8_0 GGML format vs standard INT8 quantization.

    Args:
        weight: Weight array to compare

    Returns:
        Dictionary with comparison metrics:
        - q8_0_error: Mean absolute error for Q8_0
        - int8_error: Mean absolute error for INT8
        - q8_0_size_bytes: Q8_0 storage size
        - int8_size_bytes: INT8 storage size

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import compare_q8_0_vs_int8

        weights = mx.random.normal((1024, 1024))
        comparison = compare_q8_0_vs_int8(weights)
        print(f"Q8_0 error: {comparison['q8_0_error']:.6f}")
        ```
    """
    import importlib
    _bit8 = importlib.import_module("smlx.quant.8bit")
    quantize_weights_8bit = _bit8.quantize_weights_8bit
    dequantize_weights_8bit = _bit8.dequantize_weights_8bit

    # Quantize with both formats
    q8_0_quant, q8_0_scales = quantize_to_q8_0(weight)
    int8_quant, int8_scales, int8_biases = quantize_weights_8bit(weight)

    # Dequantize
    q8_0_dequant = dequantize_from_q8_0(q8_0_quant, q8_0_scales, weight.shape)
    int8_dequant = dequantize_weights_8bit(int8_quant, int8_scales, int8_biases)

    # Compute errors
    q8_0_error = float(mx.mean(mx.abs(weight - q8_0_dequant)))
    int8_error = float(mx.mean(mx.abs(weight - int8_dequant)))

    # Size comparison
    q8_0_size = q8_0_quant.nbytes + q8_0_scales.nbytes
    int8_size = int8_quant.nbytes + int8_scales.nbytes + int8_biases.nbytes

    return {
        "q8_0_error": q8_0_error,
        "int8_error": int8_error,
        "q8_0_size_bytes": q8_0_size,
        "int8_size_bytes": int8_size,
        "format": "Q8_0 uses implicit bias (-128*scale), INT8 uses explicit bias",
    }


__all__ = [
    "quantize_to_q8_0",
    "dequantize_from_q8_0",
    "quantize_model_q8_0",
    "estimate_q8_0_size",
    "compare_q8_0_vs_int8",
    "Q8_0_BLOCK_SIZE",
    "Q8_0_BYTES_PER_BLOCK",
]
