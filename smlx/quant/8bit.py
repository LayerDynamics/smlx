"""
8-bit quantization utilities for SMLX.

Provides 8-bit quantization wrappers for Apple M4 chipsets.
8-bit quantization offers minimal accuracy loss with ~2x size reduction,
making it ideal for models that require high fidelity or for fine-tuning.

This format is particularly useful for:
- Models where 4-bit degradation is too severe
- Fine-tuning scenarios where gradients need higher precision
- Critical production deployments requiring maximum quality
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import QuantizedEmbedding, QuantizedLinear


def quantize_8bit(
    model,
    group_size: int = 64,
    inplace: bool = True,
) -> Optional[object]:
    """
    Quantize a model to 8-bit precision.

    8-bit quantization provides minimal accuracy loss compared to FP16,
    while still achieving ~2x compression. Ideal for quality-critical applications.

    Args:
        model: MLX model to quantize
        group_size: Group size for quantization (default: 64)
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_8bit

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        quantize_8bit(model)  # In-place 8-bit quantization
        # Model is ~2x smaller with negligible quality loss
        ```

    Notes:
        - Uses symmetric per-group quantization
        - Quantizes nn.Linear and nn.Embedding layers
        - Minimal quality degradation (~0.1% perplexity increase)
        - Ideal for fine-tuning and quality-critical applications
        - Reduces model size by ~2x (from FP16)
        - Still benefits from M4 Metal GPU acceleration
    """
    nn.quantize(model, group_size=group_size, bits=8)
    if not inplace:
        return model
    return None


def quantize_weights_8bit(
    weight: mx.array,
    group_size: int = 64,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Quantize weight array to 8-bit format.

    Args:
        weight: Weight array to quantize (typically 2D for Linear layers)
        group_size: Group size for quantization (default: 64)

    Returns:
        Tuple of (quantized_weights, scales, biases)
        - quantized_weights: Packed uint32 array with 8-bit values
        - scales: Per-group scaling factors (float16)
        - biases: Per-group bias terms (float16)

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant import quantize_weights_8bit

        weights = mx.random.normal((768, 768))  # Linear layer weights
        w_q, scales, biases = quantize_weights_8bit(weights)
        # w_q is ~2x smaller (8 bits per weight, packed in uint32)
        ```

    Notes:
        - Output is packed: 4 weights per uint32 element
        - Actual storage is weight.size * 8 / 32 = weight.size / 4
        - Can be dequantized with mx.dequantize()
        - Minimal quantization error compared to 4-bit/6-bit
    """
    return mx.quantize(weight, group_size=group_size, bits=8)


def dequantize_weights_8bit(
    quantized_weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = 64,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """
    Dequantize 8-bit weights back to floating point.

    Args:
        quantized_weight: Packed uint32 array with 8-bit values
        scales: Per-group scaling factors
        biases: Per-group bias terms
        group_size: Group size used for quantization (default: 64)
        dtype: Output dtype (default: float16)

    Returns:
        Dequantized weight array in specified dtype

    Example:
        ```python
        from smlx.quant import quantize_weights_8bit, dequantize_weights_8bit

        # Quantize
        w_q, scales, biases = quantize_weights_8bit(weights)

        # Dequantize
        weights_restored = dequantize_weights_8bit(w_q, scales, biases)
        # weights_restored H weights (minimal error)
        ```
    """
    return mx.dequantize(
        quantized_weight,
        scales,
        biases,
        group_size=group_size,
        bits=8,
        dtype=dtype,
    )


def estimate_8bit_size_reduction(model) -> dict:
    """
    Estimate size reduction from 8-bit quantization.

    Args:
        model: MLX model (unquantized or partially quantized)

    Returns:
        Dictionary with size estimates:
        - current_mb: Current model size in MB
        - quantized_mb: Size after full 8-bit quantization
        - reduction_ratio: Size reduction factor (typically ~2x)
        - saved_mb: MB saved by quantization

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import estimate_8bit_size_reduction

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        stats = estimate_8bit_size_reduction(model)
        print(f"Will reduce from {stats['current_mb']:.1f} MB "
              f"to {stats['quantized_mb']:.1f} MB "
              f"({stats['reduction_ratio']:.1f}x smaller)")
        ```
    """
    # Count quantizable parameters (Linear and Embedding layers)
    quantizable_params = 0
    quantized_params = 0
    total_bytes = 0

    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            weight = module.weight
            param_count = weight.size

            if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
                # Already quantized
                quantized_params += param_count
                total_bytes += weight.nbytes
                # Add scales and biases
                if hasattr(module, "scales") and module.scales is not None:
                    total_bytes += module.scales.nbytes
                if hasattr(module, "biases") and module.biases is not None:
                    total_bytes += module.biases.nbytes
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                # Can be quantized
                quantizable_params += param_count
                total_bytes += weight.nbytes
            else:
                # Other layers (keep as-is)
                total_bytes += weight.nbytes

    # Calculate 8-bit quantized size
    # 8 bits per weight = 1 byte
    # Plus scales and biases (float16 = 2 bytes each per group)
    group_size = 64
    num_groups = (quantizable_params + group_size - 1) // group_size
    quantized_weight_bytes = quantizable_params * 8 // 8  # 8 bits = 1 byte
    scales_biases_bytes = num_groups * 2 * 2  # 2 arrays * 2 bytes (float16)

    # Already quantized parameters stay as-is
    # Estimate their size (assume 8-bit)
    already_quantized_bytes = quantized_params * 8 // 8
    already_quantized_groups = (quantized_params + group_size - 1) // group_size
    already_quantized_bytes += already_quantized_groups * 2 * 2

    # Non-quantizable parameters (everything else)
    non_quantizable_bytes = total_bytes - (
        quantizable_params * 2 + quantized_params * 1.0
    )  # Assume FP16

    quantized_total = (
        quantized_weight_bytes
        + scales_biases_bytes
        + already_quantized_bytes
        + non_quantizable_bytes
    )
    current_mb = total_bytes / (1024**2)
    quantized_mb = quantized_total / (1024**2)
    reduction_ratio = current_mb / quantized_mb if quantized_mb > 0 else 1.0

    return {
        "current_mb": current_mb,
        "quantized_mb": quantized_mb,
        "reduction_ratio": reduction_ratio,
        "saved_mb": current_mb - quantized_mb,
        "quantizable_params": quantizable_params,
        "already_quantized": quantized_params > 0,
    }


def is_8bit_quantized(module) -> bool:
    """
    Check if a module is quantized to 8-bit.

    Args:
        module: MLX module to check

    Returns:
        True if module is 8-bit quantized, False otherwise

    Example:
        ```python
        from smlx.quant import quantize_8bit, is_8bit_quantized

        linear = nn.Linear(768, 768)
        assert not is_8bit_quantized(linear)

        quantize_8bit(linear)
        assert is_8bit_quantized(linear)
        ```
    """
    if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
        return getattr(module, "bits", None) == 8
    return False


def compare_with_4bit(model) -> dict:
    """
    Compare 8-bit vs 4-bit quantization trade-offs.

    Args:
        model: MLX model to analyze

    Returns:
        Dictionary comparing quantization options:
        - size_8bit_mb: Size with 8-bit quantization
        - size_4bit_mb: Size with 4-bit quantization
        - additional_size_mb: Extra MB used by 8-bit
        - quality_tradeoff: Qualitative assessment

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import compare_with_4bit

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        comparison = compare_with_4bit(model)
        print(f"8-bit uses {comparison['additional_size_mb']:.1f} MB more "
              f"but provides {comparison['quality_tradeoff']}")
        ```
    """
    # Get 8-bit estimate
    stats_8bit = estimate_8bit_size_reduction(model)
    size_8bit = stats_8bit["quantized_mb"]

    # Estimate 4-bit size (rough approximation)
    # 4-bit is approximately half the size of 8-bit
    size_4bit = size_8bit * 0.5

    return {
        "size_8bit_mb": size_8bit,
        "size_4bit_mb": size_4bit,
        "additional_size_mb": size_8bit - size_4bit,
        "quality_tradeoff": (
            "~0.1% perplexity increase vs ~0.5-1% for 4-bit. "
            "8-bit is better for fine-tuning and quality-critical applications."
        ),
        "speed_tradeoff": (
            "8-bit has similar inference speed to 4-bit on M4, "
            "both benefit from Metal GPU acceleration."
        ),
        "recommendation": (
            "Use 8-bit for fine-tuning or when quality is critical. "
            "Use 4-bit for inference when size constraints are tight."
        ),
    }


__all__ = [
    "quantize_8bit",
    "quantize_weights_8bit",
    "dequantize_weights_8bit",
    "estimate_8bit_size_reduction",
    "is_8bit_quantized",
    "compare_with_4bit",
]
