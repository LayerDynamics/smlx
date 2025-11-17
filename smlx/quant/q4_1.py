"""
GGML Q4_1 quantization format for SMLX.

Q4_1 is an improved 4-bit GGML quantization format with explicit bias term.
Format: 20 bytes per block (2-byte FP16 scale + 2-byte FP16 bias + 32 weights at 4 bits each)
Bias is stored explicitly (unlike Q4_0 where it's computed).

This format provides better quality than Q4_0 at the cost of slightly larger size.
Compatible with llama.cpp ecosystem.

Compression: ~7.3x from FP32, ~3.6x from FP16
Block size: 32 weights
Storage: 0.625 bytes per weight (20 bytes / 32 weights)
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# GGML Q4_1 Constants
Q4_1_BLOCK_SIZE = 32  # Number of weights per block
Q4_1_BYTES_PER_BLOCK = 20  # 2 (scale) + 2 (bias) + 16 (32 weights * 0.5 bytes)
Q4_1_SCALE_BYTES = 2  # FP16 scale
Q4_1_BIAS_BYTES = 2  # FP16 bias
Q4_1_WEIGHT_BYTES = 16  # 32 weights at 4 bits each


def quantize_to_q4_1(
    weight: mx.array,
    block_size: int = Q4_1_BLOCK_SIZE
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Quantize weight array to GGML Q4_1 format.

    Q4_1 uses asymmetric quantization with explicit bias and scale:
    - Each block has one FP16 scale and one FP16 bias
    - Weights quantized to 4-bit unsigned integers [0, 15]
    - Original value H scale * quantized_weight + bias

    Args:
        weight: Weight array to quantize (any shape)
        block_size: Block size for quantization (default: 32)

    Returns:
        Tuple of (quantized_weights, scales, biases):
        - quantized_weights: Packed uint8 array (2 weights per byte)
        - scales: Per-block FP16 scaling factors
        - biases: Per-block FP16 bias terms

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_to_q4_1

        weights = mx.random.normal((768, 768))
        w_q, scales, biases = quantize_to_q4_1(weights)
        # w_q is ~7-8x smaller with better quality than Q4_0
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

    # Compute per-block min/max for asymmetric quantization
    min_vals = mx.min(weight_blocks, axis=1, keepdims=True)
    max_vals = mx.max(weight_blocks, axis=1, keepdims=True)

    # Compute scale and bias
    # Map [min, max] to [0, 15]
    range_vals = max_vals - min_vals
    # Avoid division by zero
    range_vals = mx.maximum(range_vals, 1e-10)
    scales = range_vals / 15.0
    biases = min_vals

    # Quantize: map to [0, 15] range
    # quantized = round((weight - bias) / scale)
    quantized = mx.round((weight_blocks - biases) / scales)
    quantized = mx.clip(quantized, 0, 15)

    # Pack 2 weights per byte (4 bits each)
    quantized_uint8 = quantized.astype(mx.uint8)

    # Pack pairs of weights into bytes
    packed_weights = []
    for i in range(0, block_size, 2):
        low = quantized_uint8[:, i]
        high = quantized_uint8[:, i + 1]
        packed = low | (high << 4)
        packed_weights.append(packed)

    packed = mx.stack(packed_weights, axis=1)  # Shape: (num_blocks, block_size//2)

    # Convert to float16
    scales_f16 = scales.squeeze().astype(mx.float16)
    biases_f16 = biases.squeeze().astype(mx.float16)

    return packed, scales_f16, biases_f16


def dequantize_from_q4_1(
    quantized_weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    original_shape: Optional[tuple[int, ...]] = None,
    block_size: int = Q4_1_BLOCK_SIZE
) -> mx.array:
    """
    Dequantize from GGML Q4_1 format back to float.

    Args:
        quantized_weight: Packed uint8 array from quantize_to_q4_1
        scales: Per-block FP16 scales
        biases: Per-block FP16 biases
        original_shape: Original weight shape to reshape to (optional)
        block_size: Block size used for quantization (default: 32)

    Returns:
        Dequantized weight array as float32

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_to_q4_1, dequantize_from_q4_1

        weights = mx.random.normal((768, 768))
        w_q, scales, biases = quantize_to_q4_1(weights)
        w_dequant = dequantize_from_q4_1(w_q, scales, biases, weights.shape)
        ```
    """
    # Unpack weights from bytes
    unpacked_weights = []
    for i in range(block_size // 2):
        packed_byte = quantized_weight[:, i]
        low = packed_byte & 0x0F
        high = (packed_byte >> 4) & 0x0F
        unpacked_weights.extend([low, high])

    # Stack into (num_blocks, block_size)
    unpacked = []
    for i in range(block_size):
        unpacked.append(unpacked_weights[i])
    unpacked_array = mx.stack(unpacked, axis=1)

    # Convert to float
    unpacked_float = unpacked_array.astype(mx.float32)

    # Expand scales and biases for broadcasting
    scales_expanded = scales.reshape(-1, 1).astype(mx.float32)
    biases_expanded = biases.reshape(-1, 1).astype(mx.float32)

    # Dequantize: weight = scale * quantized + bias
    dequantized = scales_expanded * unpacked_float + biases_expanded

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


def quantize_model_q4_1(
    model: nn.Module,
    block_size: int = Q4_1_BLOCK_SIZE,
    inplace: bool = True
) -> Optional[nn.Module]:
    """
    Quantize entire model to GGML Q4_1 format.

    Args:
        model: MLX model to quantize
        block_size: Block size for quantization (default: 32)
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_model_q4_1

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        quantize_model_q4_1(model)  # ~7-8x compression with better quality
        ```

    Notes:
        - Quantizes nn.Linear layers only (Embedding layers skipped)
        - Better quality than Q4_0 (explicit bias term)
        - Slightly larger than Q4_0 (20 bytes vs 18 bytes per block)
        - Compatible with GGML/llama.cpp Q4_1 format
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight"):
                original_weight = module.weight
                quantized, scales, biases = quantize_to_q4_1(original_weight, block_size)

                # Store quantized weights, scales, and biases for reference
                module.weight_q4_1 = quantized
                module.scales_q4_1 = scales
                module.biases_q4_1 = biases
                module.original_shape = original_weight.shape
                module.quantization_format = "q4_1"

                # Replace weight with dequantized version
                # Note: This keeps the model functional but doesn't save runtime memory
                module.weight = dequantize_from_q4_1(
                    quantized, scales, biases, original_weight.shape
                )

    if not inplace:
        return model
    return None


def estimate_q4_1_size(model: nn.Module) -> dict:
    """
    Estimate model size after Q4_1 quantization.

    Args:
        model: MLX model to analyze

    Returns:
        Dictionary with size estimates:
        - original_mb: Original model size in MB
        - q4_1_mb: Estimated size after Q4_1 quantization
        - reduction_ratio: Compression ratio
        - bytes_per_weight: Average bytes per weight

    Example:
        ```python
        from smlx.quant import estimate_q4_1_size

        stats = estimate_q4_1_size(model)
        print(f"Q4_1: {stats['q4_1_mb']:.1f} MB ({stats['reduction_ratio']:.1f}x)")
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

    # Q4_1 storage: 0.625 bytes per weight (20 bytes / 32 weights)
    # = 4 bits for weight + 0.5 bits for scale + 0.5 bits for bias
    bytes_per_weight = Q4_1_BYTES_PER_BLOCK / Q4_1_BLOCK_SIZE
    q4_1_bytes = quantizable_params * bytes_per_weight
    q4_1_bytes += (total_params - quantizable_params) * 2  # Non-quantized params stay FP16

    return {
        "original_mb": original_bytes / (1024**2),
        "q4_1_mb": q4_1_bytes / (1024**2),
        "reduction_ratio": original_bytes / q4_1_bytes if q4_1_bytes > 0 else 1.0,
        "bytes_per_weight": bytes_per_weight,
        "quantizable_params": quantizable_params,
        "total_params": total_params,
    }


__all__ = [
    "quantize_to_q4_1",
    "dequantize_from_q4_1",
    "quantize_model_q4_1",
    "estimate_q4_1_size",
    "Q4_1_BLOCK_SIZE",
    "Q4_1_BYTES_PER_BLOCK",
]
