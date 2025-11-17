"""
GGML Q4_K (and Q4_K_M) quantization format for SMLX.

Q4_K is an advanced 4-bit GGML quantization format with hierarchical K-quantization.
Q4_K_M is a model-level strategy that mixes Q4_K and Q6_K for optimal quality.

**Q4_K Format Structure (GGML-compatible):**
- Super-blocks: 256 weights (8 sub-blocks of 32 weights each)
- Each super-block has:
  - 1 × FP16 d_scale (super-block scale for dequantizing scales)
  - 1 × FP16 d_min (super-block min offset)
  - 1 × FP16 d_min_scale (super-block scale for dequantizing mins)
  - 12 bytes packed scales/mins (8 × 6-bit scales + 8 × 6-bit mins)
  - 128 bytes packed weights (256 × 4-bit weights)
- Total: 146 bytes per 256 weights = 4.5625 bits/weight

**Q4_K_M Strategy (Model-level):**
- Applies Q6_K to half of `attention.wv` and `feed_forward.w2` tensors
- Applies Q4_K to all other tensors
- Provides better quality at ~4.8 bits/weight average

This implementation is MLX-native and dequantizes to FP16 for compatibility.

Compression: 4.5625 bits/weight (Q4_K), ~4.8 bits/weight (Q4_K_M mixed)
Block size: 256 weights (super-block)
Quality: Superior to Q4_0/Q4_1, comparable to Q5_0
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .utils import (
    pack_scales_mins_6bit,
    pack_weights_4bit,
    unpack_scales_mins_6bit,
    unpack_weights_4bit,
)


# GGML Q4_K Constants (matching GGML specification)
Q4_K_BLOCK_SIZE = 256  # Super-block size (QK_K in GGML)
Q4_K_NUM_SUBBLOCKS = 8  # Number of sub-blocks per super-block
Q4_K_SUBBLOCK_SIZE = 32  # Weights per sub-block
Q4_K_SCALE_BITS = 6  # Bits for sub-block scales and mins
Q4_K_BYTES_PER_BLOCK = 146  # Total bytes per super-block (2+2+2+12+128)


def quantize_to_q4_k(
    weight: mx.array, block_size: int = Q4_K_BLOCK_SIZE
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """
    Quantize weight array to Q4_K format (MLX-compatible variant).

    Q4_K uses hierarchical two-tier quantization:
    1. Super-blocks of 256 weights divided into 8 sub-blocks of 32 weights
    2. Per super-block: FP16 d_scale, d_min (offset), d_min_scale for quantizing scales/mins
    3. Per sub-block: 6-bit scale and 6-bit min (quantized using d_scale/d_min_scale)
    4. Per weight: 4-bit quantized value

    Dequantization formula: weight = (d_scale * scale_6bit/63) * q_4bit + (d_min + d_min_scale * min_6bit/63)
    where scale_6bit and min_6bit are 6-bit quantized values, q_4bit is 4-bit weight.

    Args:
        weight: Weight array to quantize (any shape)
        block_size: Super-block size (default: 256, must match Q4_K_BLOCK_SIZE)

    Returns:
        Tuple of (packed_weights, d_scales, d_mins, d_min_scales, packed_scales_mins):
        - packed_weights: Packed 4-bit weights, shape (num_blocks, 128) uint8
        - d_scales: Per-block FP16 scale for dequantizing scales, shape (num_blocks,)
        - d_mins: Per-block FP16 min offset, shape (num_blocks,)
        - d_min_scales: Per-block FP16 scale for dequantizing mins, shape (num_blocks,)
        - packed_scales_mins: Packed 6-bit scales/mins, shape (num_blocks, 12) uint8

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant.q4_k_m import quantize_to_q4_k

        weights = mx.random.normal((768, 768))
        w_q, d_sc, d_mn, d_mn_sc, sm_packed = quantize_to_q4_k(weights)
        # GGML-compatible Q4_K format, 4.5 bits/weight
        ```

    Notes:
        - GGML-compatible format (144 bytes per 256 weights)
        - Better quality than Q4_0/Q4_1 at similar compression
        - All weights use 4-bit (not mixed precision - that's Q4_K_M strategy)
        - Uses proper two-tier quantization with packed storage
    """
    if block_size != Q4_K_BLOCK_SIZE:
        raise ValueError(f"Q4_K requires block_size={Q4_K_BLOCK_SIZE}, got {block_size}")

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
        num_superblocks, Q4_K_NUM_SUBBLOCKS, Q4_K_SUBBLOCK_SIZE
    )

    # Lists to collect per-block data
    d_scales_list = []
    d_mins_list = []
    d_min_scales_list = []
    packed_scales_mins_list = []
    packed_weights_list = []

    for block_idx in range(num_superblocks):
        subblocks = weight_subblocks[block_idx]  # (8, 32)

        # Compute sub-block scales and mins
        sb_scales = []
        sb_mins = []
        sb_quantized_weights = []

        for sb_idx in range(Q4_K_NUM_SUBBLOCKS):
            subblock = subblocks[sb_idx]  # (32,)

            # Compute sub-block min and scale
            sb_min = float(mx.min(subblock))
            sb_max = float(mx.max(subblock))
            sb_scale = max((sb_max - sb_min) / 15.0, 1e-10)  # 4-bit range [0, 15]

            sb_scales.append(sb_scale)
            sb_mins.append(sb_min)

            # Quantize weights to 4-bit
            weights_normalized = (subblock - sb_min) / sb_scale
            weights_q = mx.round(mx.clip(weights_normalized, 0, 15)).astype(mx.uint8)
            sb_quantized_weights.append(weights_q)

        # Convert to arrays
        sb_scales = mx.array(sb_scales, dtype=mx.float32)
        sb_mins = mx.array(sb_mins, dtype=mx.float32)

        # Compute super-block d_scale and d_min for quantizing scales/mins
        max_scale = float(mx.max(sb_scales))
        # For mins, we need to find the range (can be negative)
        min_min = float(mx.min(sb_mins))
        max_min = float(mx.max(sb_mins))
        min_range = max(max_min - min_min, 1e-10)

        d_scale = max(max_scale / 63.0, 1e-10)  # Scale for 6-bit scales [0, 63]
        d_min = min_min  # Store the base offset
        d_min_scale = min_range / 63.0  # Scale for quantizing mins to [0, 63]

        # Quantize sub-block scales to 6-bit [0, 63]
        scales_q = mx.round(mx.clip(sb_scales / d_scale, 0, 63)).astype(mx.uint8)
        # Quantize mins: map [min_min, max_min] to [0, 63]
        mins_q = mx.round(mx.clip((sb_mins - d_min) / max(d_min_scale, 1e-10), 0, 63)).astype(
            mx.uint8
        )

        # Pack scales and mins into 12 bytes
        packed_sm = pack_scales_mins_6bit(scales_q, mins_q)  # (12,)

        # Stack and pack weights into 128 bytes (256 weights × 4 bits / 8 bits/byte)
        weights_concat = mx.concatenate(sb_quantized_weights)  # (256,)
        packed_w = pack_weights_4bit(weights_concat)  # (128,)

        # Store
        d_scales_list.append(d_scale)
        d_mins_list.append(d_min)
        d_min_scales_list.append(d_min_scale)
        packed_scales_mins_list.append(packed_sm)
        packed_weights_list.append(packed_w)

    # Stack all blocks
    d_scales = mx.array(d_scales_list, dtype=mx.float16)
    d_mins = mx.array(d_mins_list, dtype=mx.float16)
    d_min_scales = mx.array(d_min_scales_list, dtype=mx.float16)
    packed_scales_mins = mx.stack(packed_scales_mins_list, axis=0)  # (num_blocks, 12)
    packed_weights = mx.stack(packed_weights_list, axis=0)  # (num_blocks, 128)

    return packed_weights, d_scales, d_mins, d_min_scales, packed_scales_mins


# Alias for backward compatibility
quantize_to_q4_k_m = quantize_to_q4_k


def dequantize_from_q4_k(
    packed_weights: mx.array,
    d_scales: mx.array,
    d_mins: mx.array,
    d_min_scales: mx.array,
    packed_scales_mins: mx.array,
    original_shape: Optional[tuple[int, ...]] = None,
) -> mx.array:
    """
    Dequantize from Q4_K format back to float (MLX-compatible variant).

    Uses proper two-tier dequantization:
    1. Unpack 6-bit scales and mins from packed_scales_mins
    2. Dequantize scales: scale_float = d_scale * scale_6bit (d_scale already includes /63.0)
    3. Dequantize mins: min_float = d_min + d_min_scale * min_6bit (d_min_scale already includes /63.0)
    4. Unpack 4-bit weights from packed_weights
    5. Dequantize weights: weight = scale_float * weight_4bit + min_float

    Args:
        packed_weights: Packed 4-bit weights, shape (num_blocks, 128)
        d_scales: Per-block FP16 scales for scales, shape (num_blocks,)
        d_mins: Per-block FP16 min offsets, shape (num_blocks,)
        d_min_scales: Per-block FP16 scales for mins, shape (num_blocks,)
        packed_scales_mins: Packed 6-bit scales/mins, shape (num_blocks, 12)
        original_shape: Original weight shape to reshape to (optional)

    Returns:
        Dequantized weight array as float32

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant.q4_k_m import quantize_to_q4_k, dequantize_from_q4_k

        weights = mx.random.normal((768, 768))
        w_q, d_sc, d_mn, d_mn_sc, sm_packed = quantize_to_q4_k(weights)
        w_dequant = dequantize_from_q4_k(w_q, d_sc, d_mn, d_mn_sc, sm_packed, weights.shape)
        # w_dequant ≈ weights (with quantization error)
        ```

    Notes:
        - Implements proper two-tier dequantization formula
        - Properly unpacks bit-packed scales, mins, and weights
        - Returns FP32 for numerical stability
    """
    num_blocks = packed_weights.shape[0]

    # Lists to collect dequantized blocks
    dequantized_blocks = []

    for block_idx in range(num_blocks):
        # Get block data
        d_scale = float(d_scales[block_idx])
        d_min = float(d_mins[block_idx])
        d_min_scale = float(d_min_scales[block_idx])
        packed_sm = packed_scales_mins[block_idx]  # (12,)
        packed_w = packed_weights[block_idx]  # (128,)

        # Unpack scales and mins (8 each, 6-bit)
        scales_q, mins_q = unpack_scales_mins_6bit(packed_sm)  # Both (8,)

        # Dequantize scales and mins
        # Note: d_scale already includes /63.0, so just multiply
        scales_float = scales_q.astype(mx.float32) * d_scale
        # Note: d_min_scale already includes /63.0, so just multiply
        mins_float = d_min + mins_q.astype(mx.float32) * d_min_scale

        # Unpack weights (256 weights, 4-bit)
        weights_q = unpack_weights_4bit(packed_w, Q4_K_BLOCK_SIZE)  # (256,)

        # Reshape to sub-blocks for proper scale/min application
        weights_q_sb = weights_q.reshape(Q4_K_NUM_SUBBLOCKS, Q4_K_SUBBLOCK_SIZE)  # (8, 32)

        # Dequantize each sub-block
        sb_dequantized = []
        for sb_idx in range(Q4_K_NUM_SUBBLOCKS):
            scale = scales_float[sb_idx]
            min_val = mins_float[sb_idx]
            weights_sb = weights_q_sb[sb_idx].astype(mx.float32)  # (32,)

            # Q4_K formula: weight = scale * q + min
            dequant_sb = scale * weights_sb + min_val
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


# Alias for backward compatibility
dequantize_from_q4_k_m = dequantize_from_q4_k


def quantize_model_q4_k(
    model: nn.Module, block_size: int = Q4_K_BLOCK_SIZE, inplace: bool = True
) -> Optional[nn.Module]:
    """
    Quantize entire model to GGML Q4_K format.

    This applies Q4_K quantization to all Linear layers in the model.
    Weights are dequantized to FP16/FP32 after quantization for MLX compatibility.

    Args:
        model: MLX model to quantize
        block_size: Super-block size (default: 256, must match Q4_K_BLOCK_SIZE)
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant.q4_k_m import quantize_model_q4_k

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        quantize_model_q4_k(model)  # Apply Q4_K quantization
        ```

    Notes:
        - GGML-compatible Q4_K format (4.5 bits/weight)
        - Superior quality to Q4_0/Q4_1 at similar compression
        - Quantizes nn.Linear layers only (Embedding layers skipped)
        - Dequantizes to FP16/FP32 for MLX compatibility (no runtime memory savings)
        - For model-level Q4_K_M strategy (mixed Q4_K/Q6_K), use quantize_model_q4_k_m()
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight"):
                original_weight = module.weight
                packed_w, d_scales, d_mins, d_min_scales, packed_sm = quantize_to_q4_k(
                    original_weight, block_size
                )

                # Store quantized components for reference
                module.weight_q4_k_packed = packed_w
                module.d_scales_q4_k = d_scales
                module.d_mins_q4_k = d_mins
                module.d_min_scales_q4_k = d_min_scales
                module.packed_scales_mins_q4_k = packed_sm
                module.original_shape = original_weight.shape
                module.quantization_format = "q4_k"

                # Replace weight with dequantized version
                # Note: This keeps the model functional but doesn't save runtime memory
                # MLX-native approach: dequantize to FP16/FP32 for inference
                module.weight = dequantize_from_q4_k(
                    packed_w, d_scales, d_mins, d_min_scales, packed_sm, original_weight.shape
                )

    if not inplace:
        return model
    return None


def quantize_model_q4_k_m(
    model: nn.Module,
    block_size: int = 64,
    inplace: bool = True,
    use_mlx_native: bool = True,
) -> Optional[nn.Module]:
    """
    Quantize model using Q4_K_M-style mixed precision.

    **Q4_K_M** is a model-level quantization strategy (not a format) that uses:
    - Q6_K (6-bit) for half of attention.wv and feed_forward.w2 tensors
    - Q4_K (4-bit) for all other tensors
    - Average: ~4.8 bits/weight with better quality than pure Q4_K

    **Two modes:**

    1. **MLX Native (default, RECOMMENDED)**: Uses MLX's QuantizedLinear with
       intelligent 4-bit/6-bit layer selection. TRUE runtime memory savings.

    2. **GGML Compatible**: Quantizes to Q4_K format for GGUF export. Dequantizes
       for compatibility. NO runtime memory savings.

    Args:
        model: MLX model to quantize
        block_size: Group size for MLX quantization (default: 64)
        inplace: Modify model in-place (default: True)
        use_mlx_native: Use MLX native quantization (default: True, recommended)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_model_q4_k_m

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

        # Recommended: MLX native with Q4_K_M-style mixed precision
        quantize_model_q4_k_m(model)  # Fast, memory-efficient

        # GGML mode (for export only, not runtime)
        quantize_model_q4_k_m(model, use_mlx_native=False)
        ```

    Notes:
        - **MLX mode**: Uses QuantizedLinear (fast GPU kernels, true memory savings)
        - **GGML mode**: For GGUF file export only (no runtime benefits)
        - For more control over mixed precision, use quantize_model_mixed()
        - Quantizes nn.Linear layers only (Embedding layers skipped)

    See Also:
        - quantize_model_mixed(): More flexible mixed-precision options
        - quantize_model_q4_k(): Pure Q4_K (no mixed precision)
        - quantize_to_q4_k(): Low-level Q4_K quantization function
    """
    if use_mlx_native:
        # Use MLX's optimized quantization with Q4_K_M-style mixed precision
        from .mlx_mixed import quantize_model_mixed

        quantize_model_mixed(
            model, style="q4_k_m", low_bits=4, high_bits=6, group_size=block_size, inplace=True
        )

    else:
        # GGML-compatible mode: quantize to Q4_K/Q6_K format but dequantize for compatibility
        import warnings

        warnings.warn(
            "use_mlx_native=False mode is for GGUF export only. "
            "It does NOT provide runtime memory savings. "
            "Use use_mlx_native=True (default) for actual quantization benefits.",
            UserWarning,
            stacklevel=2,
        )

        # Apply true Q4_K_M strategy: Q6_K for important layers, Q4_K for others
        quantize_model_q4_k_m_ggml(model, inplace=True)

    if not inplace:
        return model
    return None


def quantize_model_q4_k_m_ggml(model: nn.Module, inplace: bool = True) -> Optional[nn.Module]:
    """
    Apply true GGML Q4_K_M mixed precision quantization.

    Q4_K_M strategy (from llama.cpp):
    - Uses Q6_K (6-bit) for half of attention.v_proj and feed_forward.down_proj tensors
    - Uses Q4_K (4-bit) for all other tensors
    - Average: ~4.8 bits/weight with better quality than pure Q4_K

    This is the GGML-compatible implementation that uses actual Q4_K and Q6_K formats.
    Weights are dequantized for MLX compatibility (no runtime memory savings).

    Args:
        model: MLX model to quantize
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant.q4_k_m import quantize_model_q4_k_m_ggml

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        quantize_model_q4_k_m_ggml(model)  # True Q4_K_M with Q6_K for important layers
        ```

    Notes:
        - For GGUF export compatibility
        - No runtime memory savings (dequantizes for MLX)
        - For runtime efficiency, use quantize_model_q4_k_m() with use_mlx_native=True
    """
    from .q6_k import dequantize_from_q6_k, quantize_to_q6_k

    # Track quantization statistics
    q4_layers = []
    q6_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight"):
            original_weight = module.weight

            # Determine if this is an "important" layer for Q6_K
            use_q6k = False

            # Q4_K_M uses Q6_K for:
            # 1. Half of v_proj layers (attention values - quality critical)
            # 2. Half of down_proj layers (MLP output - quality critical)
            # The "half" means layers with even indices
            if any(pattern in name for pattern in ["v_proj", "v_a_proj", "v_b_proj"]):
                # Extract layer index
                layer_idx = 0
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part in ["layers", "blocks", "h"] and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            continue

                # Use Q6_K for half of v_proj layers (even indices)
                if layer_idx % 2 == 0:
                    use_q6k = True

            if "down_proj" in name:
                # Extract layer index
                layer_idx = 0
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part in ["layers", "blocks", "h"] and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            continue

                # Use Q6_K for half of down_proj layers (even indices)
                if layer_idx % 2 == 0:
                    use_q6k = True

            # Apply appropriate quantization
            if use_q6k:
                # Quantize to Q6_K format
                ql, qh, scales, d_scales = quantize_to_q6_k(original_weight)

                # Store quantized components
                module.weight_q6_k_ql = ql
                module.weight_q6_k_qh = qh
                module.weight_q6_k_scales = scales
                module.weight_q6_k_d_scales = d_scales
                module.original_shape = original_weight.shape
                module.quantization_format = "q6_k"

                # Dequantize for MLX compatibility
                module.weight = dequantize_from_q6_k(ql, qh, scales, d_scales, original_weight.shape)

                q6_layers.append(name)

            else:
                # Quantize to Q4_K format
                packed_w, d_scales, d_mins, d_min_scales, packed_sm = quantize_to_q4_k(
                    original_weight, block_size=Q4_K_BLOCK_SIZE
                )

                # Store quantized components
                module.weight_q4_k_packed = packed_w
                module.d_scales_q4_k = d_scales
                module.d_mins_q4_k = d_mins
                module.d_min_scales_q4_k = d_min_scales
                module.packed_scales_mins_q4_k = packed_sm
                module.original_shape = original_weight.shape
                module.quantization_format = "q4_k"

                # Dequantize for MLX compatibility
                module.weight = dequantize_from_q4_k(
                    packed_w, d_scales, d_mins, d_min_scales, packed_sm, original_weight.shape
                )

                q4_layers.append(name)

    # Print quantization summary
    print(f"Q4_K_M GGML Quantization Applied:")
    print(f"  Q6_K layers (6-bit): {len(q6_layers)}")
    print(f"  Q4_K layers (4-bit): {len(q4_layers)}")
    print(f"  Total quantized: {len(q6_layers) + len(q4_layers)}")

    if not inplace:
        return model
    return None


def estimate_q4_k_size(model: nn.Module) -> dict:
    """
    Estimate model size after Q4_K quantization.

    Args:
        model: MLX model to analyze

    Returns:
        Dictionary with size estimates:
        - original_mb: Original model size in MB
        - q4_k_mb: Estimated size after Q4_K quantization
        - reduction_ratio: Compression ratio
        - avg_bits_per_weight: Average bits per weight (4.5 for Q4_K)

    Example:
        ```python
        from smlx.quant.q4_k_m import estimate_q4_k_size

        stats = estimate_q4_k_size(model)
        print(f"Q4_K: {stats['q4_k_mb']:.1f} MB ({stats['reduction_ratio']:.1f}x)")
        ```

    Notes:
        - Q4_K format: 146 bytes per 256 weights = 4.5625 bits/weight
        - Breakdown per 256-weight block:
          - 128 bytes: 256 × 4-bit weights (packed)
          - 12 bytes: 8 × 6-bit scales + 8 × 6-bit mins (packed)
          - 2 bytes: FP16 d_scale (scale for dequantizing scales)
          - 2 bytes: FP16 d_min (min offset)
          - 2 bytes: FP16 d_min_scale (scale for dequantizing mins)
          - Total: 146 bytes / 256 weights = 0.5703125 bytes/weight = 4.5625 bits/weight
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

    # Q4_K: 146 bytes per 256 weights = 4.5625 bits per weight
    # Calculation:
    # - 256 weights × 4 bits = 1024 bits = 128 bytes (packed)
    # - 8 scales × 6 bits + 8 mins × 6 bits = 96 bits = 12 bytes (packed)
    # - 1 d_scale (FP16) = 16 bits = 2 bytes
    # - 1 d_min (FP16) = 16 bits = 2 bytes
    # - 1 d_min_scale (FP16) = 16 bits = 2 bytes
    # Total: 128 + 12 + 2 + 2 + 2 = 146 bytes per 256 weights
    # Per weight: 146 / 256 = 0.5703125 bytes = 4.5625 bits
    avg_bits_per_weight = 4.5625
    bytes_per_weight = Q4_K_BYTES_PER_BLOCK / Q4_K_BLOCK_SIZE  # 146 / 256 = 0.5703125
    q4_k_bytes = quantizable_params * bytes_per_weight
    q4_k_bytes += (total_params - quantizable_params) * 2  # Non-quantized stay FP16

    return {
        "original_mb": original_bytes / (1024**2),
        "q4_k_mb": q4_k_bytes / (1024**2),
        "reduction_ratio": original_bytes / q4_k_bytes if q4_k_bytes > 0 else 1.0,
        "avg_bits_per_weight": avg_bits_per_weight,
        "bytes_per_weight": bytes_per_weight,
        "quantizable_params": quantizable_params,
        "total_params": total_params,
    }


# Alias for backward compatibility
estimate_q4_k_m_size = estimate_q4_k_size


__all__ = [
    # Primary Q4_K functions (GGML-compatible)
    "quantize_to_q4_k",
    "dequantize_from_q4_k",
    "quantize_model_q4_k",
    "estimate_q4_k_size",
    # Q4_K_M mixed precision (model-level strategy)
    "quantize_model_q4_k_m",
    "quantize_model_q4_k_m_ggml",
    # Backward compatibility aliases
    "quantize_to_q4_k_m",
    "dequantize_from_q4_k_m",
    "estimate_q4_k_m_size",
    # Constants
    "Q4_K_BLOCK_SIZE",
    "Q4_K_NUM_SUBBLOCKS",
    "Q4_K_SUBBLOCK_SIZE",
    "Q4_K_SCALE_BITS",
    "Q4_K_BYTES_PER_BLOCK",
]
