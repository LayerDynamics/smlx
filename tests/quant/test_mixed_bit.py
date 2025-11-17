"""
Tests for mixed-bit quantization strategies.

This module tests:
- MixedBitStrategy configuration
- QuantizationRule pattern matching
- apply_mixed_bit_quantization
- create_balanced_strategy, create_layerwise_strategy
- analyze_quantization_distribution
- quantize_3_6_mixed (specific 3-6 bit strategy)
- create_custom_3_6_strategy
- get_recommended_strategy
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant import (
    MixedBitStrategy,
    QuantizationRule,
    analyze_quantization_distribution,
    apply_mixed_bit_quantization,
    compute_average_bpw,
    create_balanced_strategy,
    create_custom_3_6_strategy,
    create_layerwise_strategy,
    get_recommended_strategy,
    quantize_3_6_mixed,
)


class TransformerBlock(nn.Module):
    """Simplified transformer block for testing."""

    def __init__(self, dim=256):
        super().__init__()
        self.self_attn_q = nn.Linear(dim, dim)
        self.self_attn_k = nn.Linear(dim, dim)
        self.self_attn_v = nn.Linear(dim, dim)
        self.self_attn_out = nn.Linear(dim, dim)
        self.mlp_gate = nn.Linear(dim, dim * 4)
        self.mlp_up = nn.Linear(dim, dim * 4)
        self.mlp_down = nn.Linear(dim * 4, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def __call__(self, x):
        return x


class TransformerModel(nn.Module):
    """Simple transformer model for testing."""

    def __init__(self, vocab_size=1000, dim=256, n_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = [TransformerBlock(dim) for _ in range(n_layers)]
        self.lm_head = nn.Linear(dim, vocab_size)

    def __call__(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


@pytest.mark.unit
def test_quantization_rule_creation():
    """Test QuantizationRule creation."""
    rule = QuantizationRule(
        pattern="attn",
        bits=6,
        group_size=64,
        priority=10
    )

    assert rule.pattern == "attn"
    assert rule.bits == 6
    assert rule.group_size == 64
    assert rule.priority == 10


@pytest.mark.unit
def test_mixed_bit_strategy_creation():
    """Test MixedBitStrategy creation."""
    strategy = MixedBitStrategy()

    # Should have empty rules initially
    assert len(strategy.rules) == 0

    # Should have default config
    assert strategy.default_bits == 4
    assert strategy.default_group_size == 64


@pytest.mark.unit
def test_mixed_bit_strategy_add_rule():
    """Test adding rules to strategy."""
    strategy = MixedBitStrategy()

    strategy.add_rule(pattern="embed", bits=6, priority=10)
    strategy.add_rule(pattern="attn", bits=6, priority=8)
    strategy.add_rule(pattern="mlp", bits=4, priority=5)

    assert len(strategy.rules) == 3


@pytest.mark.unit
def test_mixed_bit_strategy_priority():
    """Test rule priority ordering."""
    strategy = MixedBitStrategy()

    # Add rules in random order
    strategy.add_rule(pattern="mlp", bits=4, priority=5)
    strategy.add_rule(pattern="embed", bits=6, priority=10)
    strategy.add_rule(pattern="attn", bits=6, priority=8)

    # Rules should be sorted by priority (highest first)
    assert strategy.rules[0].priority == 10
    assert strategy.rules[1].priority == 8
    assert strategy.rules[2].priority == 5


@pytest.mark.unit
def test_mixed_bit_strategy_get_config():
    """Test getting quantization config for a module."""
    strategy = MixedBitStrategy()
    strategy.add_rule(pattern="attn", bits=6, priority=10)
    strategy.add_rule(pattern="mlp", bits=4, priority=5)
    strategy.set_default(bits=8)

    # Test pattern matching
    bits, group_size = strategy.get_quantization_config("self_attn_q", None)
    assert bits == 6  # Matches "attn" pattern

    bits, group_size = strategy.get_quantization_config("mlp_down", None)
    assert bits == 4  # Matches "mlp" pattern

    bits, group_size = strategy.get_quantization_config("some_other_layer", None)
    assert bits == 8  # Uses default


@pytest.mark.unit
def test_apply_mixed_bit_quantization():
    """Test applying mixed-bit quantization to a model."""
    model = TransformerModel()
    mx.eval(model.parameters())

    strategy = MixedBitStrategy()
    strategy.add_rule(pattern="embed", bits=6, priority=10)
    strategy.add_rule(pattern="attn", bits=6, priority=8)
    strategy.add_rule(pattern="mlp", bits=4, priority=5)
    strategy.set_default(bits=4)

    config = apply_mixed_bit_quantization(model, strategy, verbose=False)

    # Should have configs for all layers
    assert len(config) > 0

    # Check that different layers got different bit widths
    bits_used = set(bits for bits, _ in config.values())
    assert len(bits_used) > 1  # Should have mixed precision


@pytest.mark.unit
def test_compute_average_bpw():
    """Test computing average bits per weight."""
    config = {
        "layer1": (4, 64),  # 4-bit
        "layer2": (6, 64),  # 6-bit
        "layer3": (8, 64),  # 8-bit
    }

    model = nn.Module()
    model.layer1 = nn.Linear(100, 100)  # 10,000 params
    model.layer2 = nn.Linear(100, 100)  # 10,000 params
    model.layer3 = nn.Linear(100, 100)  # 10,000 params
    mx.eval(model.parameters())

    avg_bpw = compute_average_bpw(config, model)

    # Should be average of 4, 6, 8 = 6.0
    assert 5.9 < avg_bpw < 6.1


@pytest.mark.unit
def test_analyze_quantization_distribution():
    """Test analyzing quantization distribution."""
    config = {
        "layer1": (4, 64),
        "layer2": (4, 64),
        "layer3": (6, 64),
        "layer4": (8, 64),
    }

    model = nn.Module()
    model.layer1 = nn.Linear(100, 100)
    model.layer2 = nn.Linear(100, 100)
    model.layer3 = nn.Linear(100, 100)
    model.layer4 = nn.Linear(100, 100)
    mx.eval(model.parameters())

    stats = analyze_quantization_distribution(config, model)

    assert "avg_bpw" in stats
    assert "bit_distribution" in stats
    assert "memory_reduction_vs_fp16" in stats

    # Check bit distribution
    assert 4 in stats["bit_distribution"]
    assert 6 in stats["bit_distribution"]
    assert 8 in stats["bit_distribution"]

    # 4-bit should be 50% (2 out of 4 layers)
    assert abs(stats["bit_distribution"][4] - 50.0) < 1.0


@pytest.mark.unit
def test_create_balanced_strategy():
    """Test creating a balanced mixed-bit strategy."""
    strategy = create_balanced_strategy(target_bpw=5.5, low_bits=4, high_bits=8)

    # Should have rules for common layer types
    assert len(strategy.rules) > 0

    # Should prioritize embeddings and attention
    high_priority_patterns = [rule.pattern for rule in strategy.rules if rule.priority >= 8]
    assert any("embed" in p for p in high_priority_patterns)


@pytest.mark.unit
def test_create_layerwise_strategy():
    """Test creating a layerwise strategy."""
    model = TransformerModel(n_layers=4)
    mx.eval(model.parameters())

    # Define a bits schedule: first/last layers use 8-bit, middle use 4-bit
    def bits_schedule(layer_idx: int) -> int:
        if layer_idx == 0 or layer_idx == 3:  # First and last layer
            return 8
        else:
            return 4

    strategy = create_layerwise_strategy(
        num_layers=4,
        bits_schedule=bits_schedule
    )

    # Should have rules
    assert len(strategy.rules) > 0


@pytest.mark.unit
def test_quantize_3_6_mixed_aggressive():
    """Test 3-6 bit mixed quantization with aggressive profile."""
    model = TransformerModel()
    mx.eval(model.parameters())

    config = quantize_3_6_mixed(model, strategy="aggressive", verbose=False)

    # Should return configuration
    assert len(config) > 0

    # Check that we have mixed precision
    bits_used = set(bits for bits, _ in config.values())
    assert len(bits_used) > 1


@pytest.mark.unit
def test_quantize_3_6_mixed_balanced():
    """Test 3-6 bit mixed quantization with balanced profile."""
    model = TransformerModel()
    mx.eval(model.parameters())

    config = quantize_3_6_mixed(model, strategy="balanced", verbose=False)

    assert len(config) > 0

    # Balanced should use higher bits on average than aggressive
    bits_used = [bits for bits, _ in config.values()]
    avg_bits = sum(bits_used) / len(bits_used)
    assert avg_bits > 3.5  # Should be around 4.5 for balanced


@pytest.mark.unit
def test_quantize_3_6_mixed_conservative():
    """Test 3-6 bit mixed quantization with conservative profile."""
    model = TransformerModel()
    mx.eval(model.parameters())

    config = quantize_3_6_mixed(model, strategy="conservative", verbose=False)

    assert len(config) > 0

    # Conservative should use highest bits on average
    bits_used = [bits for bits, _ in config.values()]
    avg_bits = sum(bits_used) / len(bits_used)
    assert avg_bits > 4.5  # Should be around 5.0 for conservative


@pytest.mark.unit
def test_create_custom_3_6_strategy():
    """Test creating custom 3-6 bit strategy."""
    strategy = create_custom_3_6_strategy(
        embed_bits=6,
        attn_bits=6,
        mlp_bits=4,
        norm_bits=6,
        default_bits=4
    )

    # Should have rules
    assert len(strategy.rules) > 0

    # Test pattern matching
    bits, _ = strategy.get_quantization_config("embed_tokens", None)
    assert bits == 6

    bits, _ = strategy.get_quantization_config("self_attn_q", None)
    assert bits == 6

    bits, _ = strategy.get_quantization_config("mlp_down", None)
    assert bits == 4


@pytest.mark.unit
def test_get_recommended_strategy():
    """Test getting recommended strategy based on model size."""
    # Tiny model (<200M)
    strategy = get_recommended_strategy(100_000_000)
    assert strategy == "aggressive"

    # Small model (200-500M)
    strategy = get_recommended_strategy(300_000_000)
    assert strategy == "balanced"

    # Medium model (500M-1B)
    strategy = get_recommended_strategy(700_000_000)
    assert strategy == "conservative"


@pytest.mark.unit
def test_mixed_bit_with_conditional_rule():
    """Test mixed-bit strategy with conditional rules."""
    # Condition function receives (name, module) arguments
    def is_large_layer(name, module):
        if hasattr(module, "weight"):
            return module.weight.size > 50000
        return False

    strategy = MixedBitStrategy()
    strategy.add_rule(
        pattern="mlp",
        bits=6,
        priority=10,
        condition=is_large_layer
    )
    strategy.add_rule(
        pattern="mlp",
        bits=4,
        priority=5
    )

    # Large MLP layer should get 6 bits
    large_mlp = nn.Linear(256, 1024)  # 262,144 params > 50,000
    mx.eval(large_mlp.parameters())
    bits, _ = strategy.get_quantization_config("mlp_gate", large_mlp)
    assert bits == 6

    # Small MLP layer should get 4 bits
    small_mlp = nn.Linear(32, 32)  # 1,024 params < 50,000
    mx.eval(small_mlp.parameters())
    bits, _ = strategy.get_quantization_config("mlp_down", small_mlp)
    assert bits == 4


@pytest.mark.unit
def test_mixed_bit_pattern_specificity():
    """Test that more specific patterns take precedence."""
    strategy = MixedBitStrategy()
    strategy.add_rule(pattern="self_attn_q", bits=8, priority=10)
    strategy.add_rule(pattern="attn", bits=6, priority=8)
    strategy.add_rule(pattern="self", bits=4, priority=6)

    # Most specific pattern should win
    bits, _ = strategy.get_quantization_config("self_attn_q", None)
    assert bits == 8  # Exact match has highest priority


@pytest.mark.unit
def test_mixed_bit_empty_strategy():
    """Test mixed-bit with only default settings."""
    strategy = MixedBitStrategy()
    strategy.set_default(bits=6, group_size=32)

    model = TransformerModel()
    mx.eval(model.parameters())

    config = apply_mixed_bit_quantization(model, strategy, verbose=False)

    # All layers should get default settings
    for bits, group_size in config.values():
        assert bits == 6
        assert group_size == 32


@pytest.mark.unit
def test_quantize_3_6_invalid_strategy():
    """Test that invalid strategy raises error."""
    model = TransformerModel()
    mx.eval(model.parameters())

    with pytest.raises(ValueError, match="Invalid strategy"):
        quantize_3_6_mixed(model, strategy="invalid_strategy")


@pytest.mark.unit
def test_mixed_bit_comprehensive_analysis():
    """Test comprehensive quantization analysis."""
    model = TransformerModel(n_layers=3)
    mx.eval(model.parameters())

    # Apply balanced strategy
    config = quantize_3_6_mixed(model, strategy="balanced", verbose=False)
    stats = analyze_quantization_distribution(config, model)

    # Should have complete statistics
    assert "avg_bpw" in stats
    assert "bit_distribution" in stats
    assert "memory_reduction_vs_fp16" in stats
    assert "total_params" in stats

    # Memory reduction should be significant
    assert stats["memory_reduction_vs_fp16"] > 2.0  # At least 2x compression


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
