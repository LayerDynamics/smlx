"""
3-6 Bit Mixed-Precision Quantization Strategy for SMLX.

A specific mixed-precision strategy that uses:
- 3-bit quantization for less sensitive layers (maximum compression)
- 6-bit quantization for sensitive layers (better quality than 4-bit)
- Targets ~4.5 bits per weight on average

This strategy is designed for aggressive compression while maintaining
acceptable quality for "smol" models (<1B parameters) on Apple M4 chipsets.

Use cases:
- Maximum compression for edge deployment
- Models where 4-bit is borderline acceptable
- Experimentation with ultra-low-bit quantization

Example:
    ```python
    from smlx.models.SmolLM2_135M import load
    from smlx.quant.mixed_3_6 import quantize_3_6_mixed

    model, _ = load("mlx-community/SmolLM2-135M-Instruct")

    # Apply 3-6 bit mixed quantization
    config = quantize_3_6_mixed(model, strategy="balanced")
    # Average BPW: ~4.5 bits
    ```

Notes:
    - 3-bit quantization is very aggressive and may cause quality degradation
    - Use AWQ or GPTQ for better quality at same BPW
    - Recommended for SmolLM2-135M and similar tiny models
    - Test thoroughly for your specific use case
"""

from typing import Literal

import mlx.nn as nn

from .mixed_bit import (
    MixedBitStrategy,
    analyze_quantization_distribution,
    apply_mixed_bit_quantization,
)


def quantize_3_6_mixed(
    model: nn.Module,
    strategy: Literal["aggressive", "balanced", "conservative"] = "balanced",
    verbose: bool = True,
) -> dict[str, tuple[int, int]]:
    """
    Apply 3-6 bit mixed-precision quantization.

    Args:
        model: MLX model to quantize
        strategy: Quantization strategy:
            - "aggressive": Most layers 3-bit, avg ~3.5 BPW
            - "balanced": Mix of 3/4/6-bit, avg ~4.5 BPW
            - "conservative": Most layers 4/6-bit, avg ~5.0 BPW
        verbose: Print quantization decisions (default: True)

    Returns:
        Dictionary mapping layer names to (bits, group_size) tuples

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant.mixed_3_6 import quantize_3_6_mixed

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")

        # Aggressive: smallest size
        config = quantize_3_6_mixed(model, strategy="aggressive")

        # Balanced: good size/quality tradeoff
        config = quantize_3_6_mixed(model, strategy="balanced")

        # Conservative: better quality
        config = quantize_3_6_mixed(model, strategy="conservative")
        ```

    Strategy Details:
        Aggressive (avg ~3.5 BPW):
        - Embeddings/Output: 6-bit
        - Attention: 4-bit
        - FFN/MLP: 3-bit

        Balanced (avg ~4.5 BPW):
        - Embeddings/Output: 6-bit
        - Attention: 6-bit
        - FFN/MLP: 4-bit
        - Norms/Bias: 6-bit

        Conservative (avg ~5.0 BPW):
        - Embeddings/Output: 6-bit
        - Attention: 6-bit
        - FFN/MLP: 4-bit
        - All others: 6-bit
    """
    if strategy not in ["aggressive", "balanced", "conservative"]:
        raise ValueError(
            f"Invalid strategy: {strategy}. "
            "Choose from 'aggressive', 'balanced', or 'conservative'."
        )

    mixed_strategy = MixedBitStrategy()

    if strategy == "aggressive":
        # Aggressive: maximize compression (~3.5 BPW)
        # Highest priority: Embeddings and output (6-bit)
        mixed_strategy.add_rule(pattern="embed_tokens", bits=6, priority=10)
        mixed_strategy.add_rule(pattern="wte", bits=6, priority=10)
        mixed_strategy.add_rule(pattern="lm_head", bits=6, priority=10)
        mixed_strategy.add_rule(pattern="embed", bits=6, priority=10)

        # Medium priority: Attention (4-bit)
        mixed_strategy.add_rule(pattern="self_attn", bits=4, priority=7)
        mixed_strategy.add_rule(pattern="attn", bits=4, priority=7)

        # Low priority: FFN/MLP (3-bit - most aggressive)
        mixed_strategy.add_rule(pattern="mlp", bits=3, priority=5)
        mixed_strategy.add_rule(pattern="ffn", bits=3, priority=5)
        mixed_strategy.add_rule(pattern="fc", bits=3, priority=5)

        # Default: 3-bit (aggressive)
        mixed_strategy.set_default(bits=3)

    elif strategy == "balanced":
        # Balanced: good size/quality tradeoff (~4.5 BPW)
        # Highest priority: Embeddings and output (6-bit)
        mixed_strategy.add_rule(pattern="embed_tokens", bits=6, priority=10)
        mixed_strategy.add_rule(pattern="wte", bits=6, priority=10)
        mixed_strategy.add_rule(pattern="lm_head", bits=6, priority=10)
        mixed_strategy.add_rule(pattern="embed", bits=6, priority=10)

        # High priority: Layer norms and important projections (6-bit)
        mixed_strategy.add_rule(pattern="norm", bits=6, priority=9)
        mixed_strategy.add_rule(pattern="ln", bits=6, priority=9)

        # Medium priority: Attention (6-bit for quality)
        mixed_strategy.add_rule(pattern="self_attn", bits=6, priority=7)
        mixed_strategy.add_rule(pattern="attn", bits=6, priority=7)

        # Low priority: FFN/MLP (4-bit)
        mixed_strategy.add_rule(pattern="mlp", bits=4, priority=5)
        mixed_strategy.add_rule(pattern="ffn", bits=4, priority=5)
        mixed_strategy.add_rule(pattern="fc", bits=4, priority=5)

        # Default: 4-bit
        mixed_strategy.set_default(bits=4)

    else:  # conservative
        # Conservative: better quality (~5.0 BPW)
        # Highest priority: Embeddings and output (6-bit)
        mixed_strategy.add_rule(pattern="embed_tokens", bits=6, priority=10)
        mixed_strategy.add_rule(pattern="wte", bits=6, priority=10)
        mixed_strategy.add_rule(pattern="lm_head", bits=6, priority=10)
        mixed_strategy.add_rule(pattern="embed", bits=6, priority=10)

        # High priority: Layer norms (6-bit)
        mixed_strategy.add_rule(pattern="norm", bits=6, priority=9)
        mixed_strategy.add_rule(pattern="ln", bits=6, priority=9)

        # Medium priority: Attention (6-bit)
        mixed_strategy.add_rule(pattern="self_attn", bits=6, priority=7)
        mixed_strategy.add_rule(pattern="attn", bits=6, priority=7)

        # Medium-low priority: FFN/MLP (4-bit, but could be 6-bit)
        mixed_strategy.add_rule(pattern="mlp.gate", bits=6, priority=6)
        mixed_strategy.add_rule(pattern="mlp.up", bits=6, priority=6)
        mixed_strategy.add_rule(pattern="mlp.down", bits=4, priority=5)
        mixed_strategy.add_rule(pattern="mlp", bits=4, priority=4)

        # Default: 6-bit (conservative)
        mixed_strategy.set_default(bits=6)

    # Apply strategy
    config = apply_mixed_bit_quantization(model, mixed_strategy, verbose=verbose)

    # Print summary
    if verbose:
        stats = analyze_quantization_distribution(config, model)
        print(f"\nMixed 3-6 Bit Quantization Summary ({strategy} strategy):")
        print(f"  Average BPW: {stats['avg_bpw']:.2f}")
        print(f"  Memory reduction: {stats['memory_reduction_vs_fp16']:.2f}x vs FP16")
        print("  Bit distribution:")
        for bits in sorted(stats['bit_distribution'].keys()):
            pct = stats['bit_distribution'][bits]
            print(f"    {bits}-bit: {pct:.1f}% of parameters")

    return config


def create_custom_3_6_strategy(
    embed_bits: int = 6,
    attn_bits: int = 6,
    mlp_bits: int = 4,
    norm_bits: int = 6,
    default_bits: int = 4,
) -> MixedBitStrategy:
    """
    Create a custom 3-6 bit strategy with explicit bit assignments.

    Args:
        embed_bits: Bits for embeddings and output layers (default: 6)
        attn_bits: Bits for attention layers (default: 6)
        mlp_bits: Bits for FFN/MLP layers (default: 4)
        norm_bits: Bits for normalization layers (default: 6)
        default_bits: Default bits for unmatched layers (default: 4)

    Returns:
        MixedBitStrategy configured with specified bits

    Example:
        ```python
        from smlx.quant.mixed_3_6 import create_custom_3_6_strategy

        # Custom strategy: 6/4/3 bit allocation
        strategy = create_custom_3_6_strategy(
            embed_bits=6,
            attn_bits=4,
            mlp_bits=3,
            norm_bits=6,
            default_bits=4
        )

        # Apply to model
        from smlx.quant.mixed_bit import apply_mixed_bit_quantization
        config = apply_mixed_bit_quantization(model, strategy)
        ```
    """
    strategy = MixedBitStrategy()

    # Embeddings and output
    strategy.add_rule(pattern="embed_tokens", bits=embed_bits, priority=10)
    strategy.add_rule(pattern="wte", bits=embed_bits, priority=10)
    strategy.add_rule(pattern="lm_head", bits=embed_bits, priority=10)
    strategy.add_rule(pattern="embed", bits=embed_bits, priority=10)

    # Norms
    strategy.add_rule(pattern="norm", bits=norm_bits, priority=9)
    strategy.add_rule(pattern="ln", bits=norm_bits, priority=9)

    # Attention
    strategy.add_rule(pattern="self_attn", bits=attn_bits, priority=7)
    strategy.add_rule(pattern="attn", bits=attn_bits, priority=7)

    # FFN/MLP
    strategy.add_rule(pattern="mlp", bits=mlp_bits, priority=5)
    strategy.add_rule(pattern="ffn", bits=mlp_bits, priority=5)
    strategy.add_rule(pattern="fc", bits=mlp_bits, priority=5)

    # Default
    strategy.set_default(bits=default_bits)

    return strategy


def get_recommended_strategy(model_size_params: int) -> str:
    """
    Get recommended 3-6 bit strategy based on model size.

    Args:
        model_size_params: Number of model parameters

    Returns:
        Recommended strategy: "aggressive", "balanced", or "conservative"

    Example:
        ```python
        from smlx.quant.mixed_3_6 import get_recommended_strategy

        # For 135M parameter model
        strategy = get_recommended_strategy(135_000_000)
        # Returns: "aggressive"

        # For 360M parameter model
        strategy = get_recommended_strategy(360_000_000)
        # Returns: "balanced"
        ```
    """
    if model_size_params < 200_000_000:  # <200M params
        # Tiny models can handle aggressive quantization
        return "aggressive"
    elif model_size_params < 500_000_000:  # 200-500M params
        # Small models work well with balanced
        return "balanced"
    else:  # >500M params
        # Larger models need conservative approach
        return "conservative"


__all__ = [
    "quantize_3_6_mixed",
    "create_custom_3_6_strategy",
    "get_recommended_strategy",
]
