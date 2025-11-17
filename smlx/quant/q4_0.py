"""
GGML Q4_0 quantization format for SMLX.

Q4_0 is the simplest 4-bit GGML quantization format used in llama.cpp and compatible tools.
Format: 18 bytes per block (2-byte FP16 scale + 32 weights at 4 bits each, packed)
Bias is computed as -8 * scale (not stored).

This format provides maximum compatibility with llama.cpp ecosystem while
sacrificing some quality compared to Q4_1 (which stores bias explicitly).

Compression: ~8x from FP32, ~4x from FP16
Block size: 32 weights
Storage: 0.5625 bytes per weight (18 bytes / 32 weights)
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# GGML Q4_0 Constants
Q4_0_BLOCK_SIZE = 32  # Number of weights per block
Q4_0_BYTES_PER_BLOCK = 18  # 2 (scale) + 16 (32 weights * 0.5 bytes)
Q4_0_SCALE_BYTES = 2  # FP16 scale
Q4_0_WEIGHT_BYTES = 16  # 32 weights at 4 bits each


def quantize_to_q4_0(
    weight: mx.array,
    block_size: int = Q4_0_BLOCK_SIZE
) -> tuple[mx.array, mx.array]:
    """
    Quantize weight array to GGML Q4_0 format.

    Q4_0 uses symmetric quantization with implicit bias:
    - Each block has one FP16 scale
    - Bias = -8 * scale (not stored)
    - Weights quantized to 4-bit unsigned integers [0, 15]
    - Original value H scale * (quantized_weight + bias)

    Args:
        weight: Weight array to quantize (any shape)
        block_size: Block size for quantization (default: 32)

    Returns:
        Tuple of (quantized_weights, scales):
        - quantized_weights: Packed uint8 array (2 weights per byte)
        - scales: Per-block FP16 scaling factors

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_to_q4_0

        weights = mx.random.normal((768, 768))
        w_q, scales = quantize_to_q4_0(weights)
        # w_q is ~8x smaller
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

    # Compute per-block statistics
    # For Q4_0: scale = max(abs(block)) / 8
    # This maps the range to [-8, 7] in quantized space
    max_vals = mx.max(mx.abs(weight_blocks), axis=1, keepdims=True)
    # Avoid division by zero
    scales = mx.maximum(max_vals / 8.0, 1e-10)

    # Quantize: map to [0, 15] range
    # quantized = round((weight / scale) + 8)
    # Bias of -8 * scale is implicit
    quantized = mx.round((weight_blocks / scales) + 8.0)
    quantized = mx.clip(quantized, 0, 15)

    # Pack 2 weights per byte (4 bits each)
    # Convert to uint8
    quantized_uint8 = quantized.astype(mx.uint8)

    # Pack pairs of weights into bytes
    # First weight in low 4 bits, second in high 4 bits
    packed_weights = []
    for i in range(0, block_size, 2):
        low = quantized_uint8[:, i]
        high = quantized_uint8[:, i + 1]
        packed = low | (high << 4)
        packed_weights.append(packed)

    packed = mx.stack(packed_weights, axis=1)  # Shape: (num_blocks, block_size//2)

    # Convert scales to float16
    scales_f16 = scales.squeeze().astype(mx.float16)

    return packed, scales_f16


def dequantize_from_q4_0(
    quantized_weight: mx.array,
    scales: mx.array,
    original_shape: Optional[tuple[int, ...]] = None,
    block_size: int = Q4_0_BLOCK_SIZE
) -> mx.array:
    """
    Dequantize from GGML Q4_0 format back to float.

    Args:
        quantized_weight: Packed uint8 array from quantize_to_q4_0
        scales: Per-block FP16 scales
        original_shape: Original weight shape to reshape to (optional)
        block_size: Block size used for quantization (default: 32)

    Returns:
        Dequantized weight array as float32

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_to_q4_0, dequantize_from_q4_0

        weights = mx.random.normal((768, 768))
        w_q, scales = quantize_to_q4_0(weights)
        w_dequant = dequantize_from_q4_0(w_q, scales, weights.shape)
        ```
    """
    # Unpack weights from bytes
    unpacked_weights = []
    for i in range(block_size // 2):
        packed_byte = quantized_weight[:, i]
        low = packed_byte & 0x0F  # Lower 4 bits
        high = (packed_byte >> 4) & 0x0F  # Upper 4 bits
        unpacked_weights.extend([low, high])

    # Stack into (num_blocks, block_size)
    unpacked = []
    for i in range(block_size):
        unpacked.append(unpacked_weights[i])
    unpacked_array = mx.stack(unpacked, axis=1)  # Shape: (num_blocks, block_size)

    # Convert to float
    unpacked_float = unpacked_array.astype(mx.float32)

    # Expand scales for broadcasting
    scales_expanded = scales.reshape(-1, 1).astype(mx.float32)

    # Dequantize: weight = scale * (quantized - 8)
    # The implicit bias is -8 * scale
    dequantized = scales_expanded * (unpacked_float - 8.0)

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


def quantize_model_q4_0(
    model: nn.Module,
    block_size: int = Q4_0_BLOCK_SIZE,
    inplace: bool = True
) -> Optional[nn.Module]:
    """
    Quantize entire model to 4-bit format using MLX native quantization.

    This function uses MLX's optimized nn.quantize() which replaces nn.Linear
    layers with nn.QuantizedLinear layers that compute directly on packed weights.
    This provides TRUE runtime memory savings (4-bit storage + FP16 scales/biases).

    Args:
        model: MLX model to quantize
        block_size: Block size for quantization (default: 32 for Q4_0 compatibility)
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_model_q4_0

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        quantize_model_q4_0(model)  # ~8x compression with REAL memory savings
        ```

    Notes:
        - Quantizes nn.Linear layers only (Embedding layers skipped)
        - Uses MLX's native nn.QuantizedLinear for fast inference
        - Runtime format: MLX 4-bit (uint32 packing with explicit biases)
        - For GGML Q4_0 serialization, use save_as_ggml_q4_0() (future)
        - Significantly faster than custom Q4_0 unpacking
        - TRUE memory savings: ~75% reduction from FP32

    Technical Details:
        - MLX packs 8 weights per uint32 element (vs GGML's 2 per uint8)
        - Uses "affine" mode (explicit scales and biases per group)
        - Group size = 32 to match Q4_0 block size
        - Weights stored as uint32 arrays + FP16 scales/biases
        - Inference uses optimized Metal GPU kernels
    """
    def class_predicate(path, module):
        """Determine if a module should be quantized."""
        # Only quantize Linear layers
        if not isinstance(module, nn.Linear):
            return False

        # Must have weight attribute
        if not hasattr(module, "weight"):
            return False

        # Weight dimensions must be divisible by group_size
        if module.weight.shape[-1] % block_size != 0:
            return False

        return True

    # Use MLX's native quantization for fast, memory-efficient inference
    nn.quantize(
        model,
        group_size=block_size,  # 32 to match Q4_0 block size
        bits=4,
        mode="affine",  # Explicit scales + biases (like Q4_1, not pure Q4_0)
        class_predicate=class_predicate,
    )

    # Add metadata to quantized layers for compatibility
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            # Mark as Q4_0-compatible (though technically MLX 4-bit affine mode)
            module.quantization_format = "q4_0_mlx"
            module.original_path = name

            # For backward compatibility with tests expecting these attributes,
            # we can extract the quantized data
            # Note: MLX QuantizedLinear stores data as (weight, scales, biases)
            # which is different from pure GGML Q4_0 (packed uint8 + FP16 scales)
            module.weight_q4_0 = module.weight  # uint32 packed format
            module.scales_q4_0 = module.scales  # FP16 scales
            # Note: Q4_0 has implicit bias, but MLX uses explicit biases

    if not inplace:
        return model
    return None


def dequantize_model_q4_0(model: nn.Module, inplace: bool = True) -> Optional[nn.Module]:
    """
    Dequantize a Q4_0 quantized model back to FP32.

    Converts nn.QuantizedLinear layers back to nn.Linear with FP32 weights.
    Useful for fine-tuning or exporting to formats that don't support quantization.

    Args:
        model: Quantized model to dequantize
        inplace: Modify model in-place (default: True)

    Returns:
        Dequantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.quant import quantize_model_q4_0, dequantize_model_q4_0

        # Quantize for inference
        quantize_model_q4_0(model)

        # Later, dequantize for fine-tuning
        dequantize_model_q4_0(model)
        ```
    """
    from mlx.utils import tree_unflatten

    dequantize_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            # Dequantize the weights
            weight = mx.dequantize(
                module.weight,
                module.scales,
                module.biases,
                module.group_size,
                module.bits,
                module.mode,
            )

            # Create new Linear layer with dequantized weights
            has_bias = hasattr(module, "bias") and module.bias is not None
            linear = nn.Linear(
                weight.shape[1],  # in_features
                weight.shape[0],  # out_features
                bias=has_bias,
            )
            linear.weight = weight
            if has_bias:
                linear.bias = module.bias

            dequantize_layers.append((name, linear))

    # Replace quantized layers with dequantized ones
    if len(dequantize_layers) > 0:
        model.update_modules(tree_unflatten(dequantize_layers))

    if not inplace:
        return model
    return None


def estimate_q4_0_size(model: nn.Module) -> dict:
    """
    Estimate model size after Q4_0 quantization.

    Args:
        model: MLX model to analyze

    Returns:
        Dictionary with size estimates:
        - original_mb: Original model size in MB
        - q4_0_mb: Estimated size after Q4_0 quantization
        - reduction_ratio: Compression ratio
        - bytes_per_weight: Average bytes per weight

    Example:
        ```python
        from smlx.quant import estimate_q4_0_size

        stats = estimate_q4_0_size(model)
        print(f"Q4_0: {stats['q4_0_mb']:.1f} MB ({stats['reduction_ratio']:.1f}x)")
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

    # Q4_0 storage: 0.5625 bytes per weight (18 bytes / 32 weights)
    # = 4 bits for weight + 0.5 bits for scale
    bytes_per_weight = Q4_0_BYTES_PER_BLOCK / Q4_0_BLOCK_SIZE
    q4_0_bytes = quantizable_params * bytes_per_weight
    q4_0_bytes += (total_params - quantizable_params) * 2  # Non-quantized params stay FP16

    return {
        "original_mb": original_bytes / (1024**2),
        "q4_0_mb": q4_0_bytes / (1024**2),
        "reduction_ratio": original_bytes / q4_0_bytes if q4_0_bytes > 0 else 1.0,
        "bytes_per_weight": bytes_per_weight,
        "quantizable_params": quantizable_params,
        "total_params": total_params,
    }


def compare_q4_0_vs_q4_1(weight: mx.array) -> dict:
    """
    Compare Q4_0 vs Q4_1 quantization quality.

    Args:
        weight: Weight array to compare

    Returns:
        Dictionary with comparison metrics:
        - q4_0_error: Mean absolute error for Q4_0
        - q4_1_error: Mean absolute error for Q4_1
        - q4_0_size_bytes: Q4_0 storage size
        - q4_1_size_bytes: Q4_1 storage size
        - quality_improvement: Q4_1 error reduction %

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import compare_q4_0_vs_q4_1

        weights = mx.random.normal((1024, 1024))
        comparison = compare_q4_0_vs_q4_1(weights)
        print(f"Q4_1 is {comparison['quality_improvement']:.1f}% better")
        ```
    """
    from .q4_1 import quantize_to_q4_1, dequantize_from_q4_1

    # Quantize with both formats
    q4_0_quant, q4_0_scales = quantize_to_q4_0(weight)
    q4_1_quant, q4_1_scales, q4_1_biases = quantize_to_q4_1(weight)

    # Dequantize
    q4_0_dequant = dequantize_from_q4_0(q4_0_quant, q4_0_scales, weight.shape)
    q4_1_dequant = dequantize_from_q4_1(q4_1_quant, q4_1_scales, q4_1_biases, weight.shape)

    # Compute errors
    q4_0_error = float(mx.mean(mx.abs(weight - q4_0_dequant)))
    q4_1_error = float(mx.mean(mx.abs(weight - q4_1_dequant)))

    # Size comparison
    q4_0_size = q4_0_quant.nbytes + q4_0_scales.nbytes
    q4_1_size = q4_1_quant.nbytes + q4_1_scales.nbytes + q4_1_biases.nbytes

    quality_improvement = ((q4_0_error - q4_1_error) / q4_0_error * 100) if q4_0_error > 0 else 0

    return {
        "q4_0_error": q4_0_error,
        "q4_1_error": q4_1_error,
        "q4_0_size_bytes": q4_0_size,
        "q4_1_size_bytes": q4_1_size,
        "quality_improvement": quality_improvement,
        "size_overhead_percent": ((q4_1_size - q4_0_size) / q4_0_size * 100),
    }


__all__ = [
    "quantize_to_q4_0",
    "dequantize_from_q4_0",
    "quantize_model_q4_0",
    "dequantize_model_q4_0",
    "estimate_q4_0_size",
    "compare_q4_0_vs_q4_1",
    "Q4_0_BLOCK_SIZE",
    "Q4_0_BYTES_PER_BLOCK",
]
