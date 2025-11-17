"""
General Mixed-Bit Quantization Framework for SMLX.

Provides flexible mixed-precision quantization with custom bit allocation strategies.
Unlike dynamic_quant.py which uses sensitivity analysis, this module provides
explicit control over layer-wise bit assignments and allocation strategies.

Use cases:
- Custom bit allocation based on model architecture
- Experimentation with different mixed-precision configurations
- Fine-grained control over quantization strategy
- Model-specific optimizations

Example:
    ```python
    from smlx.quant import MixedBitStrategy, apply_mixed_bit_quantization

    # Define custom strategy: 8-bit for embeddings, 4-bit for FFN, 6-bit for attention
    strategy = MixedBitStrategy()
    strategy.add_rule(pattern="embed", bits=8, priority=10)
    strategy.add_rule(pattern="mlp", bits=4, priority=5)
    strategy.add_rule(pattern="attn", bits=6, priority=7)
    strategy.set_default(bits=4)

    # Apply to model
    quantized_model = apply_mixed_bit_quantization(model, strategy)
    ```
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.nn as nn


@dataclass
class QuantizationRule:
    """
    Rule for layer quantization.

    Attributes:
        pattern: Substring to match in layer name (e.g., "mlp", "attn", "embed")
        bits: Number of bits for quantization
        group_size: Group size for quantization
        priority: Rule priority (higher = applied first)
        condition: Optional function(name, module) -> bool for advanced matching
    """

    pattern: str
    bits: int
    group_size: int = 64
    priority: int = 0
    condition: Optional[Callable] = None

    def matches(self, name: str, module: nn.Module) -> bool:
        """Check if rule matches the given layer."""
        # Check pattern match
        if self.pattern not in name:
            return False

        # Check custom condition if provided
        if self.condition is not None:
            return self.condition(name, module)

        return True


@dataclass
class MixedBitStrategy:
    """
    Mixed-bit quantization strategy with prioritized rules.

    Example:
        ```python
        strategy = MixedBitStrategy()
        strategy.add_rule(pattern="embed_tokens", bits=8, priority=10)
        strategy.add_rule(pattern="lm_head", bits=8, priority=10)
        strategy.add_rule(pattern="mlp.gate_proj", bits=4, priority=5)
        strategy.add_rule(pattern="mlp.down_proj", bits=4, priority=5)
        strategy.add_rule(pattern="self_attn", bits=6, priority=7)
        strategy.set_default(bits=4)
        ```
    """

    rules: list[QuantizationRule] = field(default_factory=list)
    default_bits: int = 4
    default_group_size: int = 64

    def add_rule(
        self,
        pattern: str,
        bits: int,
        group_size: int = 64,
        priority: int = 0,
        condition: Optional[Callable] = None,
    ) -> None:
        """Add a quantization rule."""
        rule = QuantizationRule(
            pattern=pattern,
            bits=bits,
            group_size=group_size,
            priority=priority,
            condition=condition,
        )
        self.rules.append(rule)
        # Keep rules sorted by priority (descending)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def set_default(self, bits: int, group_size: int = 64) -> None:
        """Set default quantization for layers not matching any rule."""
        self.default_bits = bits
        self.default_group_size = group_size

    def get_quantization_config(self, name: str, module: nn.Module) -> tuple[int, int]:
        """
        Get quantization config (bits, group_size) for a layer.

        Args:
            name: Layer name
            module: Layer module

        Returns:
            Tuple of (bits, group_size)
        """
        # Try rules in priority order
        for rule in self.rules:
            if rule.matches(name, module):
                return rule.bits, rule.group_size

        # Fall back to default
        return self.default_bits, self.default_group_size


def apply_mixed_bit_quantization(
    model: nn.Module,
    strategy: MixedBitStrategy,
    verbose: bool = True,
) -> dict[str, tuple[int, int]]:
    """
    Apply mixed-bit quantization to model according to strategy.

    Args:
        model: MLX model to quantize
        strategy: Mixed-bit quantization strategy
        verbose: Print quantization decisions (default: True)

    Returns:
        Dictionary mapping layer names to (bits, group_size) tuples

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import MixedBitStrategy, apply_mixed_bit_quantization

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")

        strategy = MixedBitStrategy()
        strategy.add_rule(pattern="embed", bits=8, priority=10)
        strategy.add_rule(pattern="mlp", bits=4, priority=5)
        strategy.set_default(bits=6)

        config = apply_mixed_bit_quantization(model, strategy)
        ```
    """
    quantization_config = {}

    # Build quantization configuration for class_predicate
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)) and hasattr(module, "weight"):
            bits, group_size = strategy.get_quantization_config(name, module)
            quantization_config[name] = (bits, group_size)

            if verbose and bits > 0:
                print(f"{name}: {bits}-bit (group_size={group_size})")

    # Define class_predicate for nn.quantize()
    # This function is called for each layer and returns quantization params
    def class_predicate(path: str, module: nn.Module):
        """Return quantization config for each layer based on strategy."""
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False

        if path in quantization_config:
            bits, group_size = quantization_config[path]

            # bits=0 means skip quantization
            if bits == 0:
                return False

            # Return quantization parameters as dict
            return {"group_size": group_size, "bits": bits}

        return False

    # Apply mixed-bit quantization using MLX's built-in nn.quantize
    # This properly replaces Linear/Embedding layers with QuantizedLinear/QuantizedEmbedding
    nn.quantize(model, class_predicate=class_predicate)

    return quantization_config


def compute_average_bpw(quantization_config: dict[str, tuple[int, int]], model) -> float:
    """
    Compute average bits-per-weight for a quantization configuration.

    Args:
        quantization_config: Dict mapping layer names to (bits, group_size)
        model: MLX model

    Returns:
        Average bits per weight

    Example:
        ```python
        config = apply_mixed_bit_quantization(model, strategy, verbose=False)
        avg_bpw = compute_average_bpw(config, model)
        print(f"Average BPW: {avg_bpw:.2f}")
        ```
    """
    total_params = 0
    total_bits = 0

    for name, module in model.named_modules():
        if name in quantization_config and hasattr(module, "weight"):
            bits, _ = quantization_config[name]
            num_params = module.weight.size
            total_params += num_params
            total_bits += num_params * bits

    return total_bits / total_params if total_params > 0 else 0.0


def create_balanced_strategy(
    target_bpw: float, low_bits: int = 4, high_bits: int = 8
) -> MixedBitStrategy:
    """
    Create a balanced mixed-bit strategy targeting specific average BPW.

    Uses common architectural patterns to distribute bits.

    Args:
        target_bpw: Target average bits per weight (e.g., 4.5)
        low_bits: Minimum bits to use (default: 4)
        high_bits: Maximum bits to use (default: 8)

    Returns:
        MixedBitStrategy configured for target BPW

    Example:
        ```python
        # Target 4.5 bits per weight on average
        strategy = create_balanced_strategy(target_bpw=4.5, low_bits=4, high_bits=6)
        ```

    Notes:
        - Embeddings and output layers use high_bits (most sensitive)
        - FFN/MLP layers use low_bits (least sensitive)
        - Attention layers use mid_bits
        - Adjust based on target_bpw
    """
    strategy = MixedBitStrategy()

    # Calculate mid bits based on target
    if target_bpw >= (low_bits + high_bits) / 2:
        mid_bits = high_bits - 1
    else:
        mid_bits = low_bits + 1

    # High priority: embeddings and output (use high_bits)
    strategy.add_rule(pattern="embed_tokens", bits=high_bits, priority=10)
    strategy.add_rule(pattern="wte", bits=high_bits, priority=10)  # GPT-style
    strategy.add_rule(pattern="lm_head", bits=high_bits, priority=10)

    # Medium priority: attention layers (use mid_bits)
    strategy.add_rule(pattern="self_attn", bits=mid_bits, priority=7)
    strategy.add_rule(pattern="attn", bits=mid_bits, priority=7)

    # Low priority: FFN/MLP (use low_bits)
    strategy.add_rule(pattern="mlp", bits=low_bits, priority=5)
    strategy.add_rule(pattern="ffn", bits=low_bits, priority=5)

    # Default: use mid_bits
    strategy.set_default(bits=mid_bits)

    return strategy


def create_layerwise_strategy(
    num_layers: int, bits_schedule: Callable[[int], int]
) -> MixedBitStrategy:
    """
    Create a layer-wise mixed-bit strategy with custom schedule.

    Args:
        num_layers: Total number of transformer layers
        bits_schedule: Function mapping layer_idx -> bits

    Returns:
        MixedBitStrategy with layer-wise bit allocation

    Example:
        ```python
        # Higher bits for early and late layers
        def schedule(layer_idx):
            if layer_idx < 4 or layer_idx >= 20:
                return 8  # Early/late layers: 8-bit
            else:
                return 4  # Middle layers: 4-bit

        strategy = create_layerwise_strategy(num_layers=24, bits_schedule=schedule)
        ```
    """
    strategy = MixedBitStrategy()

    for layer_idx in range(num_layers):
        bits = bits_schedule(layer_idx)
        # Match layer by index (works for most transformer architectures)
        pattern = f"layers.{layer_idx}."
        strategy.add_rule(pattern=pattern, bits=bits, priority=layer_idx)

    strategy.set_default(bits=bits_schedule(num_layers // 2))  # Use mid-layer bits

    return strategy


def analyze_quantization_distribution(
    quantization_config: dict[str, tuple[int, int]], model
) -> dict:
    """
    Analyze the distribution of quantization bits across the model.

    Args:
        quantization_config: Dict mapping layer names to (bits, group_size)
        model: MLX model

    Returns:
        Dictionary with distribution statistics

    Example:
        ```python
        config = apply_mixed_bit_quantization(model, strategy, verbose=False)
        stats = analyze_quantization_distribution(config, model)
        print(f"Average BPW: {stats['avg_bpw']:.2f}")
        print(f"Distribution: {stats['bit_distribution']}")
        ```
    """
    bit_counts = {}  # bits -> num_params
    total_params = 0

    for name, module in model.named_modules():
        if name in quantization_config and hasattr(module, "weight"):
            bits, _ = quantization_config[name]
            num_params = module.weight.size
            bit_counts[bits] = bit_counts.get(bits, 0) + num_params
            total_params += num_params

    # Calculate distribution percentages
    bit_distribution = {
        bits: (count / total_params * 100) if total_params > 0 else 0
        for bits, count in bit_counts.items()
    }

    # Calculate average BPW
    avg_bpw = compute_average_bpw(quantization_config, model)

    # Calculate weighted memory reduction (vs FP16)
    memory_reduction = 16 / avg_bpw if avg_bpw > 0 else 1.0

    return {
        "avg_bpw": avg_bpw,
        "bit_distribution": bit_distribution,
        "bit_counts": bit_counts,
        "total_params": total_params,
        "memory_reduction_vs_fp16": memory_reduction,
        "layers_quantized": len(quantization_config),
    }


__all__ = [
    "QuantizationRule",
    "MixedBitStrategy",
    "apply_mixed_bit_quantization",
    "compute_average_bpw",
    "create_balanced_strategy",
    "create_layerwise_strategy",
    "analyze_quantization_distribution",
]
