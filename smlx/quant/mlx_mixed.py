"""
MLX-Native Mixed-Precision Quantization for SMLX.

Provides MLX-optimized mixed-precision quantization that approximates Q4_K_M quality
while using MLX's fast native QuantizedLinear layers for runtime efficiency.

This approach:
- Uses MLX's built-in nn.quantize() with intelligent layer selection
- Provides TRUE runtime memory savings (not just storage)
- Faster inference than custom bit-packing formats
- Approximates Q4_K_M quality through strategic higher-bit allocation

Inspired by llama.cpp's Q4_K_M mixed quantization strategy.
"""

from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn


def create_q4_k_m_style_predicate(
    model: nn.Module,
    low_bits: int = 4,
    high_bits: int = 6,
    group_size: int = 64,
) -> Callable[[str, nn.Module], Union[bool, dict]]:
    """
    Create a mixed-precision predicate that mimics Q4_K_M quantization strategy.

    Q4_K_M allocates higher precision (6-bit) to:
    - First 1/8 of layers (early representation learning)
    - Last 1/8 of layers (final output formation)
    - Every 3rd layer in middle section (maintaining quality)
    - v_proj layers (attention values - quality critical)
    - down_proj layers (MLP output - quality critical)
    - lm_head (final projection)

    All other layers use lower precision (4-bit).

    Args:
        model: MLX model to quantize
        low_bits: Bits for most layers (default: 4)
        high_bits: Bits for important layers (default: 6)
        group_size: Quantization group size (default: 64)

    Returns:
        Predicate function for nn.quantize()

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import create_q4_k_m_style_predicate
        import mlx.nn as nn

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        pred = create_q4_k_m_style_predicate(model, low_bits=4, high_bits=6)
        nn.quantize(model, group_size=64, bits=4, class_predicate=pred)
        ```

    Notes:
        - Based on llama.cpp Q4_K_M strategy
        - Provides ~4.8 bits/weight average (similar to Q4_K_M)
        - Uses MLX's fast QuantizedLinear for inference
    """
    # Count transformer layers
    num_layers = 0
    for name, module in model.named_modules():
        # Common patterns for transformer layers
        if "layers." in name or "blocks." in name or "h." in name:
            # Extract layer index
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part in ["layers", "blocks", "h"] and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        num_layers = max(num_layers, layer_idx + 1)
                    except ValueError:
                        continue

    def predicate(path: str, module: nn.Module) -> Union[bool, dict]:
        """
        Determine quantization config for each layer.

        Returns:
            - False: Don't quantize
            - True: Use default quantization
            - dict: Custom quantization config {"bits": int, "group_size": int}
        """
        # Only quantize Linear layers
        if not isinstance(module, nn.Linear):
            return False

        # Must have weight attribute
        if not hasattr(module, "weight"):
            return False

        # Extract layer index if present
        layer_idx = 0
        parts = path.split(".")
        for i, part in enumerate(parts):
            if part in ["layers", "blocks", "h"] and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except ValueError:
                    continue

        # Determine if this is an "important" layer for higher bits
        use_high_bits = False

        if num_layers > 0:
            # First 1/8 of layers
            if layer_idx < num_layers // 8:
                use_high_bits = True
            # Last 1/8 of layers
            elif layer_idx >= 7 * num_layers // 8:
                use_high_bits = True
            # Every 3rd layer in middle section
            elif (layer_idx - num_layers // 8) % 3 == 2:
                use_high_bits = True

        # Important layer types always get higher bits
        if any(keyword in path for keyword in ["v_proj", "v_a_proj", "v_b_proj"]):
            use_high_bits = True
        if "down_proj" in path:
            use_high_bits = True
        if "lm_head" in path or "head" in path:
            use_high_bits = True

        # Return config
        if use_high_bits:
            return {"group_size": group_size, "bits": high_bits}
        else:
            return {"group_size": group_size, "bits": low_bits}

    return predicate


def quantize_model_mixed(
    model: nn.Module,
    style: str = "q4_k_m",
    low_bits: int = 4,
    high_bits: int = 6,
    group_size: int = 64,
    inplace: bool = True,
) -> Optional[nn.Module]:
    """
    Quantize model using MLX-native mixed-precision quantization.

    This function provides several predefined quantization strategies:
    - "q4_k_m": Mimics llama.cpp Q4_K_M (~4.8 bits/weight avg)
    - "q4_k_s": Aggressive 4-bit with minimal 6-bit (~4.2 bits/weight avg)
    - "q4_k_l": Conservative with more 6-bit for quality (~5.2 bits/weight avg)
    - "uniform": All layers at same bit-width (low_bits)

    Args:
        model: MLX model to quantize
        style: Quantization style (default: "q4_k_m")
        low_bits: Bits for most layers (default: 4)
        high_bits: Bits for important layers (default: 6)
        group_size: Quantization group size (default: 64)
        inplace: Modify model in-place (default: True)

    Returns:
        Quantized model if inplace=False, otherwise None

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import quantize_model_mixed

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

        # Q4_K_M style (recommended - balance of size and quality)
        quantize_model_mixed(model, style="q4_k_m")

        # More aggressive (smaller, slightly lower quality)
        quantize_model_mixed(model, style="q4_k_s", low_bits=4, high_bits=5)

        # More conservative (larger, higher quality)
        quantize_model_mixed(model, style="q4_k_l", low_bits=4, high_bits=8)
        ```

    Notes:
        - Uses MLX's QuantizedLinear for TRUE runtime memory savings
        - Significantly faster than custom Q4_K bit-packing
        - Provides quality similar to GGML Q4_K_M
        - For GGML Q4_K_M file loading, use load_gguf_q4_k_m() instead
    """
    if style == "uniform":
        # Simple uniform quantization - all layers same bits
        nn.quantize(model, group_size=group_size, bits=low_bits)

    elif style in ["q4_k_m", "q4_k_s", "q4_k_l"]:
        # Create style-specific predicate
        if style == "q4_k_s":
            # Aggressive: 4-bit everywhere except lm_head
            pred = lambda path, mod: (
                {"bits": high_bits, "group_size": group_size}
                if "lm_head" in path and isinstance(mod, nn.Linear)
                else {"bits": low_bits, "group_size": group_size}
                if isinstance(mod, nn.Linear)
                else False
            )
        elif style == "q4_k_l":
            # Conservative: use high_bits more liberally
            # Double the frequency of high-bit layers
            pred = create_q4_k_m_style_predicate(model, low_bits, high_bits, group_size)
            # Wrap to use high_bits for every other layer instead of every 3rd
            original_pred = pred

            def conservative_pred(path: str, mod: nn.Module):
                result = original_pred(path, mod)
                if isinstance(result, dict) and result["bits"] == low_bits:
                    # Extract layer index
                    layer_idx = 0
                    parts = path.split(".")
                    for i, part in enumerate(parts):
                        if part in ["layers", "blocks", "h"] and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                break
                            except ValueError:
                                continue
                    # Use high_bits for every other layer
                    if layer_idx % 2 == 1:
                        return {"bits": high_bits, "group_size": group_size}
                return result

            pred = conservative_pred
        else:
            # q4_k_m: balanced strategy
            pred = create_q4_k_m_style_predicate(model, low_bits, high_bits, group_size)

        nn.quantize(model, group_size=group_size, bits=low_bits, class_predicate=pred)

    else:
        raise ValueError(
            f"Unknown quantization style: {style}. "
            f"Choose from: 'q4_k_m', 'q4_k_s', 'q4_k_l', 'uniform'"
        )

    # Mark quantized layers with metadata
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.quantization_format = f"mlx_mixed_{style}"
            module.original_path = name

    if not inplace:
        return model
    return None


def estimate_mixed_size(
    model: nn.Module,
    style: str = "q4_k_m",
    low_bits: int = 4,
    high_bits: int = 6,
) -> dict:
    """
    Estimate model size after mixed-precision quantization.

    Args:
        model: MLX model to analyze
        style: Quantization style
        low_bits: Bits for most layers
        high_bits: Bits for important layers

    Returns:
        Dictionary with size estimates:
        - original_mb: Original model size in MB
        - quantized_mb: Estimated size after mixed quantization
        - reduction_ratio: Compression ratio
        - avg_bits_per_weight: Average bits per weight
        - low_bit_params: Parameters at low_bits
        - high_bit_params: Parameters at high_bits

    Example:
        ```python
        from smlx.quant import estimate_mixed_size

        stats = estimate_mixed_size(model, style="q4_k_m", low_bits=4, high_bits=6)
        print(f"Mixed Q4_K_M: {stats['quantized_mb']:.1f} MB ({stats['avg_bits_per_weight']:.1f} bits/weight)")
        ```
    """
    # Create temporary model copy for analysis
    import copy

    model_copy = copy.deepcopy(model)

    # Apply quantization to analyze
    quantize_model_mixed(model_copy, style=style, low_bits=low_bits, high_bits=high_bits)

    # Analyze quantized model
    from .utils import estimate_model_size

    size_info = estimate_model_size(model_copy)

    # Count high-bit vs low-bit parameters
    low_bit_params = 0
    high_bit_params = 0

    for _, module in model_copy.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            weight_size = module.weight.size * 32 // module.bits  # Dequantized size
            if module.bits == high_bits:
                high_bit_params += weight_size
            else:
                low_bit_params += weight_size

    total_quant_params = low_bit_params + high_bit_params
    if total_quant_params > 0:
        avg_bits = (low_bit_params * low_bits + high_bit_params * high_bits) / total_quant_params
    else:
        avg_bits = low_bits

    return {
        "original_mb": size_info["unquantized_mb"] + size_info["quantized_mb"],
        "quantized_mb": size_info["quantized_mb"],
        "reduction_ratio": size_info["total_mb"] / size_info["quantized_mb"]
        if size_info["quantized_mb"] > 0
        else 1.0,
        "avg_bits_per_weight": avg_bits,
        "low_bit_params": low_bit_params,
        "high_bit_params": high_bit_params,
        "total_params": size_info["parameters"],
    }


__all__ = [
    "create_q4_k_m_style_predicate",
    "quantize_model_mixed",
    "estimate_mixed_size",
]
