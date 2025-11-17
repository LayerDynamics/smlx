"""
BFloat16 (Brain Float16) utilities for SMLX.

BFloat16 is a 16-bit floating point format that trades mantissa precision
for extended exponent range compared to FP16. It's particularly useful for
training and inference as it maintains FP32's dynamic range.

Key characteristics:
- 1 sign bit, 8 exponent bits, 7 mantissa bits
- Same exponent range as FP32 (vs 5 exponent bits in FP16)
- Lower precision than FP16 but more stable for training
- Native support on M-series chips

This is NOT traditional quantization (which uses integer math), but rather
a lower-precision floating point format that reduces memory by 2x from FP32
while maintaining numerical stability.
"""

from typing import Optional

import mlx.core as mx


def convert_to_bfloat16(model, inplace: bool = True) -> Optional[object]:
    """
    Convert model weights to BFloat16 precision.

    BFloat16 reduces memory by 2x compared to FP32 while maintaining
    better numerical stability than FP16 for many operations.

    Args:
        model: MLX model to convert
        inplace: Modify model in-place (default: True)

    Returns:
        Converted model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import convert_to_bfloat16

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        convert_to_bfloat16(model)  # Convert to BF16
        # Model uses 2x less memory than FP32, more stable than FP16
        ```

    Notes:
        - BFloat16 maintains FP32's dynamic range
        - Better for training than FP16 (less overflow/underflow)
        - Native M-series hardware support
        - Reduces memory by 2x from FP32 (same as FP16)
        - Not as aggressive as 4-bit/8-bit integer quantization
    """
    # Convert all parameters to bfloat16
    for _, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            module.weight = module.weight.astype(mx.bfloat16)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias = module.bias.astype(mx.bfloat16)

    if not inplace:
        return model
    return None


def weights_to_bfloat16(weight: mx.array) -> mx.array:
    """
    Convert weight array to BFloat16.

    Args:
        weight: Weight array to convert

    Returns:
        Weight array in BFloat16 format

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import weights_to_bfloat16

        weights = mx.random.normal((768, 768), dtype=mx.float32)
        weights_bf16 = weights_to_bfloat16(weights)
        # weights_bf16 uses half the memory of weights
        ```
    """
    return weight.astype(mx.bfloat16)


def weights_from_bfloat16(weight: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
    """
    Convert BFloat16 weights to another dtype.

    Args:
        weight: BFloat16 weight array
        dtype: Target dtype (default: float32)

    Returns:
        Weight array in target dtype

    Example:
        ```python
        from smlx.quant import weights_to_bfloat16, weights_from_bfloat16

        # Convert to BF16 and back
        weights_bf16 = weights_to_bfloat16(weights)
        weights_restored = weights_from_bfloat16(weights_bf16)
        # weights_restored H weights (with BF16 precision loss)
        ```
    """
    return weight.astype(dtype)


def estimate_bfloat16_size(model) -> dict:
    """
    Estimate memory usage with BFloat16.

    Args:
        model: MLX model

    Returns:
        Dictionary with size estimates:
        - current_mb: Current model size in MB
        - bfloat16_mb: Size after BFloat16 conversion
        - reduction_ratio: Size reduction factor
        - saved_mb: MB saved
        - current_dtype: Current predominant dtype

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import estimate_bfloat16_size

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        stats = estimate_bfloat16_size(model)
        print(f"BF16 will reduce from {stats['current_mb']:.1f} MB "
              f"to {stats['bfloat16_mb']:.1f} MB")
        ```
    """
    total_bytes = 0
    bf16_bytes = 0
    dtypes = {}

    for _, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            weight = module.weight
            total_bytes += weight.nbytes

            # Track dtypes
            dtype_str = str(weight.dtype)
            dtypes[dtype_str] = dtypes.get(dtype_str, 0) + 1

            # BFloat16 always uses 2 bytes per element
            bf16_bytes += weight.size * 2

        if hasattr(module, "bias") and module.bias is not None:
            bias = module.bias
            total_bytes += bias.nbytes
            bf16_bytes += bias.size * 2

    current_mb = total_bytes / (1024**2)
    bf16_mb = bf16_bytes / (1024**2)
    reduction_ratio = current_mb / bf16_mb if bf16_mb > 0 else 1.0

    # Determine predominant dtype
    if dtypes:
        current_dtype = max(dtypes.items(), key=lambda x: x[1])[0]
    else:
        current_dtype = "unknown"

    return {
        "current_mb": current_mb,
        "bfloat16_mb": bf16_mb,
        "reduction_ratio": reduction_ratio,
        "saved_mb": current_mb - bf16_mb,
        "current_dtype": current_dtype,
        "dtype_distribution": dtypes,
    }


def is_bfloat16(module) -> bool:
    """
    Check if a module's weights are in BFloat16.

    Args:
        module: MLX module to check

    Returns:
        True if weights are BFloat16, False otherwise

    Example:
        ```python
        from smlx.quant import convert_to_bfloat16, is_bfloat16

        linear = nn.Linear(768, 768)
        assert not is_bfloat16(linear)

        convert_to_bfloat16(linear)
        assert is_bfloat16(linear)
        ```
    """
    if hasattr(module, "weight") and module.weight is not None:
        return module.weight.dtype == mx.bfloat16
    return False


def mixed_precision_bf16_fp32(
    model,
    bf16_layers: Optional[list[str]] = None,
    fp32_layers: Optional[list[str]] = None,
) -> None:
    """
    Apply mixed precision: BFloat16 for most layers, FP32 for sensitive layers.

    This is useful for maintaining numerical stability in critical layers
    (like final output projections) while using BF16 for most of the model.

    Args:
        model: MLX model to convert
        bf16_layers: List of layer name patterns to convert to BF16 (default: all)
        fp32_layers: List of layer name patterns to keep in FP32 (overrides bf16_layers)

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import mixed_precision_bf16_fp32

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")

        # Keep embeddings and final layer in FP32, rest in BF16
        mixed_precision_bf16_fp32(
            model,
            fp32_layers=["embed_tokens", "lm_head"]
        )
        ```

    Notes:
        - If both lists are None, converts everything to BF16
        - fp32_layers takes precedence over bf16_layers
        - Layer matching is done via substring matching
    """
    fp32_patterns = fp32_layers or []
    bf16_patterns = bf16_layers or [""]  # Empty string matches all

    for name, module in model.named_modules():
        # Check if this layer should stay in FP32
        keep_fp32 = any(pattern in name for pattern in fp32_patterns)
        convert_bf16 = any(pattern in name for pattern in bf16_patterns)

        if keep_fp32:
            # Ensure FP32
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.dtype != mx.float32:
                    module.weight = module.weight.astype(mx.float32)
            if hasattr(module, "bias") and module.bias is not None:
                if module.bias.dtype != mx.float32:
                    module.bias = module.bias.astype(mx.float32)
        elif convert_bf16:
            # Convert to BF16
            if hasattr(module, "weight") and module.weight is not None:
                module.weight = module.weight.astype(mx.bfloat16)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = module.bias.astype(mx.bfloat16)


def compare_dtypes(model) -> dict:
    """
    Compare different dtype options for the model.

    Args:
        model: MLX model to analyze

    Returns:
        Dictionary comparing dtype options:
        - fp32_mb: Size in FP32
        - fp16_mb: Size in FP16
        - bfloat16_mb: Size in BFloat16
        - current_mb: Current size
        - recommendations: Usage recommendations

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import compare_dtypes

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        comparison = compare_dtypes(model)
        print(comparison['recommendations'])
        ```
    """
    total_params = 0
    current_bytes = 0

    for _, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            weight = module.weight
            total_params += weight.size
            current_bytes += weight.nbytes
        if hasattr(module, "bias") and module.bias is not None:
            bias = module.bias
            total_params += bias.size
            current_bytes += bias.nbytes

    # Calculate sizes for different dtypes
    fp32_mb = (total_params * 4) / (1024**2)  # 4 bytes per param
    fp16_mb = (total_params * 2) / (1024**2)  # 2 bytes per param
    bf16_mb = (total_params * 2) / (1024**2)  # 2 bytes per param
    current_mb = current_bytes / (1024**2)

    recommendations = {
        "fp32": "Use for training sensitive models or when maximum precision is required",
        "fp16": "Use for inference when range is limited, slightly higher precision than BF16",
        "bfloat16": "Recommended for training and inference - good balance of range and size",
        "4bit/8bit": "Use integer quantization (4-bit/8-bit) for even more compression",
    }

    return {
        "fp32_mb": fp32_mb,
        "fp16_mb": fp16_mb,
        "bfloat16_mb": bf16_mb,
        "current_mb": current_mb,
        "total_params": total_params,
        "recommendations": recommendations,
        "tradeoffs": {
            "precision": "FP32 > FP16 > BFloat16 (mantissa)",
            "range": "FP32 = BFloat16 > FP16 (exponent)",
            "training_stability": "FP32 > BFloat16 > FP16",
            "memory": "FP32 (4 bytes) > FP16/BF16 (2 bytes) > 8-bit (1 byte) > 4-bit (0.5 bytes)",
        },
    }


__all__ = [
    "convert_to_bfloat16",
    "weights_to_bfloat16",
    "weights_from_bfloat16",
    "estimate_bfloat16_size",
    "is_bfloat16",
    "mixed_precision_bf16_fp32",
    "compare_dtypes",
]
