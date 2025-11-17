"""
GGML Q6_K quantization format for SMLX.

Q6_K is a 6-bit GGML quantization format with hierarchical quantization.
Used in Q4_K_M strategy for important layers (attention.wv, feed_forward.w2).

**Q6_K Format Structure (GGML-compatible):**
- Super-blocks: 256 weights (16 sub-blocks of 16 weights each)
- Each super-block has:
  - 1 × FP16 d_scale (super-block scale)
  - 16 × int8 scales (sub-block scales, quantized to 8-bit signed)
  - 128 bytes ql (lower 4 bits of 256 weights)
  - 64 bytes qh (upper 2 bits of 256 weights, 4 per byte)
- Total: 210 bytes per 256 weights = 6.5625 bits/weight

**Dequantization:**
- Reconstruct 6-bit weight from ql and qh
- Scale = d_scale * (scales[i] / 256.0)
- Weight = scale * (weight_6bit - 32)

This implementation is MLX-native and dequantizes to FP16 for compatibility.

Compression: 6.5625 bits/weight
Block size: 256 weights (super-block)
Quality: Better than Q4_K, used for important layers in Q4_K_M
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# GGML Q6_K Constants (matching GGML specification)
Q6_K_BLOCK_SIZE = 256  # Super-block size (QK_K in GGML)
Q6_K_NUM_SUBBLOCKS = 16  # Number of sub-blocks per super-block
Q6_K_SUBBLOCK_SIZE = 16  # Weights per sub-block
Q6_K_BYTES_PER_BLOCK = 210  # Total bytes per super-block (128+64+16+2)


def pack_weights_6bit(weights: mx.array) -> tuple[mx.array, mx.array]:
    """
    Pack 6-bit weights into ql (lower 4 bits, packed) and qh (upper 2 bits, packed) format.

    Args:
        weights: Array of values in range [0, 63] (6-bit)

    Returns:
        Tuple of (ql, qh):
        - ql: Lower 4 bits, packed 2 per byte, shape (..., N//2) uint8
        - qh: Upper 2 bits, packed 4 per byte, shape (..., N//4) uint8
    """
    weights = mx.clip(weights, 0, 63).astype(mx.uint8)

    # Extract lower 4 bits and upper 2 bits
    ql_unpacked = weights & 0x0F  # Lower 4 bits
    qh_unpacked = (weights >> 4) & 0x03  # Upper 2 bits

    *batch_dims, n_weights = weights.shape
    if n_weights % 4 != 0:
        raise ValueError(f"Number of weights must be divisible by 4 for Q6_K, got {n_weights}")

    # Pack ql: 2 weights per byte (4 bits each)
    # ql[0] = w0[3:0] | w1[3:0]<<4
    ql_unpacked_reshaped = ql_unpacked.reshape(-1, n_weights // 2, 2)
    ql = ql_unpacked_reshaped[..., 0] | (ql_unpacked_reshaped[..., 1] << 4)

    # Pack qh: 4 weights per byte (2 bits each)
    # qh[0] = w0[1:0] | w1[1:0]<<2 | w2[1:0]<<4 | w3[1:0]<<6
    qh_unpacked_reshaped = qh_unpacked.reshape(-1, n_weights // 4, 4)
    qh = (
        qh_unpacked_reshaped[..., 0]
        | (qh_unpacked_reshaped[..., 1] << 2)
        | (qh_unpacked_reshaped[..., 2] << 4)
        | (qh_unpacked_reshaped[..., 3] << 6)
    )

    if batch_dims:
        ql = ql.reshape(*batch_dims, n_weights // 2)
        qh = qh.reshape(*batch_dims, n_weights // 4)
    else:
        ql = ql.squeeze(0)
        qh = qh.squeeze(0)

    return ql, qh


def unpack_weights_6bit(ql: mx.array, qh: mx.array, n_weights: int) -> mx.array:
    """
    Unpack 6-bit weights from ql (lower 4 bits, packed) and qh (upper 2 bits, packed) format.

    Args:
        ql: Lower 4 bits packed, shape (..., N//2) uint8
        qh: Upper 2 bits packed, shape (..., N//4) uint8
        n_weights: Number of weights to extract

    Returns:
        Unpacked 6-bit weights, shape (..., N) uint8
    """
    *batch_dims, n_ql = ql.shape
    ql_reshaped = ql.reshape(-1, n_ql, 1)

    # Unpack ql: extract 2 × 4-bit values from each byte
    ql0 = ql_reshaped & 0x0F
    ql1 = (ql_reshaped >> 4) & 0x0F
    ql_unpacked = mx.concatenate([ql0, ql1], axis=-1).reshape(-1, n_weights)

    # Unpack qh: extract 4 × 2-bit values from each byte
    qh_reshaped = qh.reshape(-1, n_weights // 4, 1)

    qh0 = qh_reshaped & 0x03
    qh1 = (qh_reshaped >> 2) & 0x03
    qh2 = (qh_reshaped >> 4) & 0x03
    qh3 = (qh_reshaped >> 6) & 0x03

    qh_unpacked = mx.concatenate([qh0, qh1, qh2, qh3], axis=-1).reshape(-1, n_weights)

    # Combine: weight = ql | (qh << 4)
    weights = ql_unpacked | (qh_unpacked << 4)

    # Reshape back to original batch shape
    if batch_dims:
        weights = weights.reshape(*batch_dims, n_weights)
    else:
        weights = weights.squeeze(0)

    return weights


def quantize_to_q6_k(
    weight: mx.array, block_size: int = Q6_K_BLOCK_SIZE
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Quantize weight array to Q6_K format (MLX-compatible variant).

    Q6_K uses hierarchical quantization:
    1. Super-blocks of 256 weights divided into 16 sub-blocks of 16 weights
    2. Per super-block: FP16 d_scale for dequantizing scales
    3. Per sub-block: int8 scale (quantized using d_scale)
    4. Per weight: 6-bit quantized value [0, 63]

    Args:
        weight: Weight array to quantize (any shape)
        block_size: Super-block size (default: 256, must match Q6_K_BLOCK_SIZE)

    Returns:
        Tuple of (ql, qh, scales, d_scales):
        - ql: Lower 4 bits of weights, shape (num_blocks, 128) uint8
        - qh: Upper 2 bits of weights, shape (num_blocks, 64) uint8
        - scales: Quantized int8 scales, shape (num_blocks, 16) int8
        - d_scales: Per-block FP16 scale for scales, shape (num_blocks,)

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant.q6_k import quantize_to_q6_k

        weights = mx.random.normal((768, 768))
        ql, qh, scales, d_sc = quantize_to_q6_k(weights)
        # GGML-compatible Q6_K format, 6.5625 bits/weight
        ```

    Notes:
        - GGML-compatible format (210 bytes per 256 weights)
        - Better quality than Q4_K at ~45% more storage
        - Used for important layers in Q4_K_M strategy
    """
    if block_size != Q6_K_BLOCK_SIZE:
        raise ValueError(f"Q6_K requires block_size={Q6_K_BLOCK_SIZE}, got {block_size}")

    weight_flat = weight.flatten()

    # Pad to multiple of block_size
    remainder = weight_flat.size % block_size
    if remainder != 0:
        padding = block_size - remainder
        weight_flat = mx.concatenate([weight_flat, mx.zeros(padding, dtype=weight.dtype)])

    # Reshape into super-blocks
    num_superblocks = weight_flat.size // block_size
    weight_superblocks = weight_flat.reshape(num_superblocks, block_size)

    # Further divide into sub-blocks
    weight_subblocks = weight_superblocks.reshape(
        num_superblocks, Q6_K_NUM_SUBBLOCKS, Q6_K_SUBBLOCK_SIZE
    )

    # Lists to collect per-block data
    ql_list = []
    qh_list = []
    scales_list = []
    d_scales_list = []

    for block_idx in range(num_superblocks):
        subblocks = weight_subblocks[block_idx]  # (16, 16)

        # Compute sub-block scales
        sb_scales = []
        sb_quantized_weights = []

        for sb_idx in range(Q6_K_NUM_SUBBLOCKS):
            subblock = subblocks[sb_idx]  # (16,)

            # Compute sub-block absolute max for symmetric quantization
            max_abs = float(mx.max(mx.abs(subblock)))
            scale = max(max_abs / 32.0, 1e-10)  # 6-bit signed range [-32, 31]

            sb_scales.append(scale)

            # Quantize weights to 6-bit signed [-32, 31], then shift to [0, 63]
            weights_normalized = subblock / scale
            weights_q = mx.round(mx.clip(weights_normalized, -32, 31)).astype(mx.int8)
            weights_q_unsigned = (weights_q + 32).astype(mx.uint8)  # Shift to [0, 63]
            sb_quantized_weights.append(weights_q_unsigned)

        # Convert scales to array
        sb_scales = mx.array(sb_scales, dtype=mx.float32)

        # Compute super-block d_scale for quantizing scales to int8
        max_scale = float(mx.max(mx.abs(sb_scales)))
        d_scale = max(max_scale / 127.0, 1e-10)  # int8 range [-128, 127]

        # Quantize sub-block scales to int8
        scales_q = mx.round(mx.clip(sb_scales / d_scale, -128, 127)).astype(mx.int8)

        # Concatenate and pack weights
        weights_concat = mx.concatenate(sb_quantized_weights)  # (256,)
        ql, qh = pack_weights_6bit(weights_concat)  # ql: (128,), qh: (64,)

        # Store
        ql_list.append(ql)
        qh_list.append(qh)
        scales_list.append(scales_q)
        d_scales_list.append(d_scale)

    # Stack all blocks
    ql = mx.stack(ql_list, axis=0)  # (num_blocks, 128) - packed lower 4 bits
    qh = mx.stack(qh_list, axis=0)  # (num_blocks, 64) - packed upper 2 bits
    scales = mx.stack(scales_list, axis=0)  # (num_blocks, 16)
    d_scales = mx.array(d_scales_list, dtype=mx.float16)  # (num_blocks,)

    return ql, qh, scales, d_scales


def dequantize_from_q6_k(
    ql: mx.array,
    qh: mx.array,
    scales: mx.array,
    d_scales: mx.array,
    original_shape: Optional[tuple[int, ...]] = None,
) -> mx.array:
    """
    Dequantize from Q6_K format back to float (MLX-compatible variant).

    Args:
        ql: Lower 4 bits of weights (packed), shape (num_blocks, 128)
        qh: Upper 2 bits of weights (packed), shape (num_blocks, 64)
        scales: Quantized int8 scales, shape (num_blocks, 16)
        d_scales: Per-block FP16 scales, shape (num_blocks,)
        original_shape: Original weight shape to reshape to (optional)

    Returns:
        Dequantized weight array as float32

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant.q6_k import quantize_to_q6_k, dequantize_from_q6_k

        weights = mx.random.normal((768, 768))
        ql, qh, scales, d_sc = quantize_to_q6_k(weights)
        w_dequant = dequantize_from_q6_k(ql, qh, scales, d_sc, weights.shape)
        # w_dequant ≈ weights (with quantization error)
        ```
    """
    num_blocks = ql.shape[0]

    # Lists to collect dequantized blocks
    dequantized_blocks = []

    for block_idx in range(num_blocks):
        d_scale = float(d_scales[block_idx])
        scales_q = scales[block_idx]  # (16,)
        ql_block = ql[block_idx]  # (256,)
        qh_block = qh[block_idx]  # (64,)

        # Unpack weights
        weights_q = unpack_weights_6bit(ql_block, qh_block, Q6_K_BLOCK_SIZE)  # (256,)

        # Dequantize scales
        scales_float = scales_q.astype(mx.float32) * d_scale  # (16,)

        # Reshape to sub-blocks
        weights_q_sb = weights_q.reshape(Q6_K_NUM_SUBBLOCKS, Q6_K_SUBBLOCK_SIZE)  # (16, 16)

        # Dequantize each sub-block
        sb_dequantized = []
        for sb_idx in range(Q6_K_NUM_SUBBLOCKS):
            scale = scales_float[sb_idx]
            weights_sb = weights_q_sb[sb_idx].astype(mx.float32)  # (16,)

            # Q6_K formula: weight = scale * (weight_6bit - 32)
            dequant_sb = scale * (weights_sb - 32.0)
            sb_dequantized.append(dequant_sb)

        # Concatenate sub-blocks
        block_dequantized = mx.concatenate(sb_dequantized)  # (256,)
        dequantized_blocks.append(block_dequantized)

    # Concatenate all blocks
    dequantized_flat = mx.concatenate(dequantized_blocks)

    if original_shape is not None:
        # Trim padding
        original_size = 1
        for dim in original_shape:
            original_size *= dim
        dequantized_flat = dequantized_flat[:original_size]
        dequantized = dequantized_flat.reshape(original_shape)
    else:
        dequantized = dequantized_flat

    return dequantized


def quantize_model_q6_k(
    model: nn.Module, block_size: int = Q6_K_BLOCK_SIZE, inplace: bool = True
) -> Optional[nn.Module]:
    """
    Quantize entire model to GGML Q6_K format.

    This applies Q6_K quantization to all Linear layers in the model.
    Weights are dequantized to FP16/FP32 after quantization for MLX compatibility.

    Args:
        model: MLX model to quantize
        block_size: Super-block size (default: 256, must match Q6_K_BLOCK_SIZE)
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant.q6_k import quantize_model_q6_k

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        quantize_model_q6_k(model)  # Apply Q6_K quantization
        ```

    Notes:
        - GGML-compatible Q6_K format (6.5625 bits/weight)
        - Better quality than Q4_K at ~45% more storage
        - Quantizes nn.Linear layers only (Embedding layers skipped)
        - Dequantizes to FP16/FP32 for MLX compatibility (no runtime memory savings)
        - For Q4_K_M strategy (selective Q6_K), use quantize_model_q4_k_m()
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight"):
                original_weight = module.weight
                ql, qh, scales, d_scales = quantize_to_q6_k(original_weight, block_size)

                # Store quantized components for reference
                module.weight_q6_k_ql = ql
                module.weight_q6_k_qh = qh
                module.scales_q6_k = scales
                module.d_scales_q6_k = d_scales
                module.original_shape = original_weight.shape
                module.quantization_format = "q6_k"

                # Replace weight with dequantized version
                module.weight = dequantize_from_q6_k(ql, qh, scales, d_scales, original_weight.shape)

    if not inplace:
        return model
    return None


def estimate_q6_k_size(model: nn.Module) -> dict:
    """
    Estimate model size after Q6_K quantization.

    Args:
        model: MLX model to analyze

    Returns:
        Dictionary with size estimates:
        - original_mb: Original model size in MB
        - q6_k_mb: Estimated size after Q6_K quantization
        - reduction_ratio: Compression ratio
        - avg_bits_per_weight: Average bits per weight (6.5625 for Q6_K)

    Example:
        ```python
        from smlx.quant.q6_k import estimate_q6_k_size

        stats = estimate_q6_k_size(model)
        print(f"Q6_K: {stats['q6_k_mb']:.1f} MB ({stats['reduction_ratio']:.1f}x)")
        ```

    Notes:
        - Q6_K format: 210 bytes per 256 weights = 6.5625 bits/weight
        - Breakdown per 256-weight block:
          - 128 bytes: lower 4 bits of 256 weights (ql)
          - 64 bytes: upper 2 bits of 256 weights (qh, 4 per byte)
          - 16 bytes: 16 int8 scales
          - 2 bytes: FP16 d_scale
          - Total: 210 bytes / 256 weights = 0.8203125 bytes/weight = 6.5625 bits/weight
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

    # Q6_K: 210 bytes per 256 weights = 6.5625 bits per weight
    avg_bits_per_weight = 6.5625
    bytes_per_weight = Q6_K_BYTES_PER_BLOCK / Q6_K_BLOCK_SIZE  # 210 / 256
    q6_k_bytes = quantizable_params * bytes_per_weight
    q6_k_bytes += (total_params - quantizable_params) * 2  # Non-quantized stay FP16

    return {
        "original_mb": original_bytes / (1024**2),
        "q6_k_mb": q6_k_bytes / (1024**2),
        "reduction_ratio": original_bytes / q6_k_bytes if q6_k_bytes > 0 else 1.0,
        "avg_bits_per_weight": avg_bits_per_weight,
        "bytes_per_weight": bytes_per_weight,
        "quantizable_params": quantizable_params,
        "total_params": total_params,
    }


__all__ = [
    "quantize_to_q6_k",
    "dequantize_from_q6_k",
    "quantize_model_q6_k",
    "estimate_q6_k_size",
    "pack_weights_6bit",
    "unpack_weights_6bit",
    "Q6_K_BLOCK_SIZE",
    "Q6_K_NUM_SUBBLOCKS",
    "Q6_K_SUBBLOCK_SIZE",
    "Q6_K_BYTES_PER_BLOCK",
]
