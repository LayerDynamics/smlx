"""
Tests for BFloat16 quantization utilities.

This module tests:
- convert_to_bfloat16 (model-level conversion)
- weights_to_bfloat16 (array conversion to BF16)
- weights_from_bfloat16 (array conversion from BF16)
- estimate_bfloat16_size (size estimation)
- is_bfloat16 (dtype checking)
- mixed_precision_bf16_fp32 (mixed precision conversion)
- compare_dtypes (dtype comparison)
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant import (
    compare_dtypes,
    convert_to_bfloat16,
    estimate_bfloat16_size,
    is_bfloat16,
    mixed_precision_bf16_fp32,
    weights_from_bfloat16,
    weights_to_bfloat16,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)

    def __call__(self, x):
        x = self.fc1(x)
        return self.fc2(x)


class TransformerModel(nn.Module):
    """Transformer-like model for testing mixed precision."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, 256)
        self.self_attn_q = nn.Linear(256, 256)
        self.self_attn_k = nn.Linear(256, 256)
        self.self_attn_v = nn.Linear(256, 256)
        self.mlp_gate = nn.Linear(256, 1024)
        self.mlp_down = nn.Linear(1024, 256)
        self.lm_head = nn.Linear(256, 1000)
        self.norm = nn.LayerNorm(256)

    def __call__(self, x):
        return x


@pytest.mark.unit
def test_weights_to_bfloat16():
    """Test converting weight array to BFloat16."""
    weight = mx.random.normal((64, 64), dtype=mx.float32)

    weight_bf16 = weights_to_bfloat16(weight)

    # Check dtype conversion
    assert weight_bf16.dtype == mx.bfloat16
    assert weight_bf16.shape == weight.shape

    # Check that values are preserved (with BF16 precision)
    max_diff = mx.max(mx.abs(weight.astype(mx.bfloat16) - weight_bf16))
    assert float(max_diff) < 1e-6


@pytest.mark.unit
def test_weights_from_bfloat16():
    """Test converting BFloat16 weights to another dtype."""
    weight = mx.random.normal((64, 64), dtype=mx.float32)
    weight_bf16 = weights_to_bfloat16(weight)

    # Convert back to FP32
    weight_restored = weights_from_bfloat16(weight_bf16, dtype=mx.float32)

    assert weight_restored.dtype == mx.float32
    assert weight_restored.shape == weight.shape

    # Should match the BF16 precision (not original FP32)
    expected = weight_bf16.astype(mx.float32)
    max_diff = mx.max(mx.abs(weight_restored - expected))
    assert float(max_diff) < 1e-6


@pytest.mark.unit
def test_weights_from_bfloat16_to_fp16():
    """Test converting BFloat16 to FP16."""
    weight_bf16 = mx.random.normal((32, 32), dtype=mx.bfloat16)

    weight_fp16 = weights_from_bfloat16(weight_bf16, dtype=mx.float16)

    assert weight_fp16.dtype == mx.float16
    assert weight_fp16.shape == weight_bf16.shape


@pytest.mark.unit
def test_convert_to_bfloat16_inplace():
    """Test in-place BFloat16 conversion."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Check original dtype (should be float32 by default)
    assert model.fc1.weight.dtype == mx.float32

    # Convert in-place
    result = convert_to_bfloat16(model, inplace=True)

    # Should return None for in-place
    assert result is None

    # Check conversion
    assert model.fc1.weight.dtype == mx.bfloat16
    assert model.fc2.weight.dtype == mx.bfloat16


@pytest.mark.unit
def test_convert_to_bfloat16_not_inplace():
    """Test non-inplace BFloat16 conversion."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Original dtype
    assert model.fc1.weight.dtype == mx.float32

    # Convert not in-place (returns model)
    result = convert_to_bfloat16(model, inplace=False)

    # Should return the model
    assert result is not None

    # Model should still be converted
    assert model.fc1.weight.dtype == mx.bfloat16
    assert model.fc2.weight.dtype == mx.bfloat16


@pytest.mark.unit
def test_convert_to_bfloat16_with_bias():
    """Test BFloat16 conversion with bias."""
    # Create linear layer with bias
    linear = nn.Linear(128, 64, bias=True)
    mx.eval(linear.parameters())

    assert linear.weight.dtype == mx.float32
    assert linear.bias.dtype == mx.float32

    convert_to_bfloat16(linear, inplace=True)

    # Both weight and bias should be converted
    assert linear.weight.dtype == mx.bfloat16
    assert linear.bias.dtype == mx.bfloat16


@pytest.mark.unit
def test_is_bfloat16_true():
    """Test is_bfloat16 detection (positive case)."""
    linear = nn.Linear(64, 32)
    mx.eval(linear.parameters())

    # Initially not BF16
    assert not is_bfloat16(linear)

    # Convert to BF16
    convert_to_bfloat16(linear, inplace=True)

    # Now should detect BF16
    assert is_bfloat16(linear)


@pytest.mark.unit
def test_is_bfloat16_false():
    """Test is_bfloat16 detection (negative case)."""
    linear = nn.Linear(64, 32)
    mx.eval(linear.parameters())

    # FP32 should not be detected as BF16
    assert not is_bfloat16(linear)


@pytest.mark.unit
def test_is_bfloat16_no_weight():
    """Test is_bfloat16 with module without weight."""
    module = nn.Module()

    # Module without weight should return False
    assert not is_bfloat16(module)


@pytest.mark.unit
def test_estimate_bfloat16_size():
    """Test BFloat16 size estimation."""
    model = SimpleModel()
    mx.eval(model.parameters())

    stats = estimate_bfloat16_size(model)

    # Check all expected keys
    assert "current_mb" in stats
    assert "bfloat16_mb" in stats
    assert "reduction_ratio" in stats
    assert "saved_mb" in stats
    assert "current_dtype" in stats
    assert "dtype_distribution" in stats

    # Check values make sense
    assert stats["current_mb"] > 0
    assert stats["bfloat16_mb"] > 0
    assert stats["saved_mb"] >= 0

    # If model is FP32, BF16 should be ~2x smaller
    if "float32" in stats["current_dtype"]:
        assert 1.9 < stats["reduction_ratio"] < 2.1


@pytest.mark.unit
def test_estimate_bfloat16_size_fp16_model():
    """Test size estimation for FP16 model (should be similar size)."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Convert to FP16
    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            module.weight = module.weight.astype(mx.float16)

    stats = estimate_bfloat16_size(model)

    # FP16 to BF16 should be ~1x (same size)
    assert 0.95 < stats["reduction_ratio"] < 1.05
    assert "float16" in stats["current_dtype"]


@pytest.mark.unit
def test_mixed_precision_bf16_fp32_default():
    """Test mixed precision with default settings (all to BF16)."""
    model = TransformerModel()
    mx.eval(model.parameters())

    # Convert all to BF16 (default behavior)
    mixed_precision_bf16_fp32(model)

    # All layers should be BF16
    assert model.embed_tokens.weight.dtype == mx.bfloat16
    assert model.self_attn_q.weight.dtype == mx.bfloat16
    assert model.mlp_gate.weight.dtype == mx.bfloat16
    assert model.lm_head.weight.dtype == mx.bfloat16


@pytest.mark.unit
def test_mixed_precision_bf16_fp32_selective():
    """Test mixed precision with selective FP32 layers."""
    model = TransformerModel()
    mx.eval(model.parameters())

    # Keep embeddings and lm_head in FP32, rest in BF16
    mixed_precision_bf16_fp32(model, fp32_layers=["embed_tokens", "lm_head"])

    # Check FP32 layers
    assert model.embed_tokens.weight.dtype == mx.float32
    assert model.lm_head.weight.dtype == mx.float32

    # Check BF16 layers
    assert model.self_attn_q.weight.dtype == mx.bfloat16
    assert model.mlp_gate.weight.dtype == mx.bfloat16


@pytest.mark.unit
def test_mixed_precision_bf16_fp32_patterns():
    """Test mixed precision with pattern matching."""
    model = TransformerModel()
    mx.eval(model.parameters())

    # Keep all "attn" layers in FP32
    mixed_precision_bf16_fp32(model, fp32_layers=["attn"])

    # Attention layers should be FP32
    assert model.self_attn_q.weight.dtype == mx.float32
    assert model.self_attn_k.weight.dtype == mx.float32
    assert model.self_attn_v.weight.dtype == mx.float32

    # Other layers should be BF16
    assert model.mlp_gate.weight.dtype == mx.bfloat16
    assert model.mlp_down.weight.dtype == mx.bfloat16


@pytest.mark.unit
def test_mixed_precision_bf16_fp32_priority():
    """Test that fp32_layers takes precedence over bf16_layers."""
    model = TransformerModel()
    mx.eval(model.parameters())

    # Try to convert "mlp" to BF16, but keep "mlp_gate" in FP32
    mixed_precision_bf16_fp32(
        model, bf16_layers=["mlp"], fp32_layers=["mlp_gate"]  # Should match all mlp layers
    )  # Should override

    # mlp_gate should stay FP32 (fp32_layers takes precedence)
    assert model.mlp_gate.weight.dtype == mx.float32

    # mlp_down should be BF16
    assert model.mlp_down.weight.dtype == mx.bfloat16


@pytest.mark.unit
def test_compare_dtypes():
    """Test dtype comparison functionality."""
    model = SimpleModel()
    mx.eval(model.parameters())

    comparison = compare_dtypes(model)

    # Check all expected keys
    assert "fp32_mb" in comparison
    assert "fp16_mb" in comparison
    assert "bfloat16_mb" in comparison
    assert "current_mb" in comparison
    assert "total_params" in comparison
    assert "recommendations" in comparison
    assert "tradeoffs" in comparison

    # Check values make sense
    assert comparison["fp32_mb"] > 0
    assert comparison["fp16_mb"] > 0
    assert comparison["bfloat16_mb"] > 0

    # FP32 should be 2x larger than FP16/BF16
    assert comparison["fp32_mb"] > comparison["fp16_mb"] * 1.9
    assert comparison["fp32_mb"] > comparison["bfloat16_mb"] * 1.9

    # FP16 and BF16 should be same size (both 2 bytes)
    assert abs(comparison["fp16_mb"] - comparison["bfloat16_mb"]) < 0.01

    # Check recommendations exist
    assert "fp32" in comparison["recommendations"]
    assert "bfloat16" in comparison["recommendations"]


@pytest.mark.unit
def test_compare_dtypes_recommendations():
    """Test that dtype recommendations are sensible."""
    model = SimpleModel()
    mx.eval(model.parameters())

    comparison = compare_dtypes(model)

    # Check that recommendations mention key concepts
    assert "training" in comparison["recommendations"]["bfloat16"].lower()
    assert "precision" in comparison["recommendations"]["fp32"].lower()


@pytest.mark.unit
def test_compare_dtypes_tradeoffs():
    """Test that tradeoffs are documented."""
    model = SimpleModel()
    mx.eval(model.parameters())

    comparison = compare_dtypes(model)

    # Check tradeoffs exist
    assert "precision" in comparison["tradeoffs"]
    assert "range" in comparison["tradeoffs"]
    assert "memory" in comparison["tradeoffs"]
    assert "training_stability" in comparison["tradeoffs"]

    # Verify precision ordering
    precision_order = comparison["tradeoffs"]["precision"]
    assert "FP32" in precision_order
    assert "BFloat16" in precision_order


@pytest.mark.unit
def test_bfloat16_roundtrip():
    """Test full conversion roundtrip."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Save original weights
    orig_fc1 = mx.array(model.fc1.weight)

    # Convert to BF16
    convert_to_bfloat16(model, inplace=True)
    assert is_bfloat16(model.fc1)

    # Convert back to FP32
    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            module.weight = weights_from_bfloat16(module.weight, dtype=mx.float32)

    # Should match BF16 precision (not original FP32 precision)
    expected = orig_fc1.astype(mx.bfloat16).astype(mx.float32)
    max_diff = mx.max(mx.abs(model.fc1.weight - expected))
    assert float(max_diff) < 1e-6


@pytest.mark.unit
def test_bfloat16_preserves_shape():
    """Test that BF16 conversion preserves shapes."""
    weights = mx.random.normal((123, 456), dtype=mx.float32)

    weights_bf16 = weights_to_bfloat16(weights)

    assert weights_bf16.shape == weights.shape
    assert weights_bf16.shape == (123, 456)


@pytest.mark.unit
def test_bfloat16_handles_zero():
    """Test BF16 conversion of zero values."""
    weights = mx.zeros((32, 32), dtype=mx.float32)

    weights_bf16 = weights_to_bfloat16(weights)
    weights_restored = weights_from_bfloat16(weights_bf16)

    # Zeros should remain zeros
    assert float(mx.max(mx.abs(weights_restored))) < 1e-6


@pytest.mark.unit
def test_bfloat16_handles_large_values():
    """Test BF16 conversion with large values (benefits from extended range)."""
    # BFloat16 has same exponent range as FP32
    weights = mx.array([[1e10, -1e10], [1e-10, -1e-10]], dtype=mx.float32)

    weights_bf16 = weights_to_bfloat16(weights)
    weights_restored = weights_from_bfloat16(weights_bf16)

    # Should handle large values better than FP16
    assert mx.all(mx.isfinite(weights_restored))


@pytest.mark.unit
def test_estimate_size_with_embedding():
    """Test size estimation with embedding layers."""
    model = TransformerModel()
    mx.eval(model.parameters())

    stats = estimate_bfloat16_size(model)

    # Should handle embeddings correctly
    assert stats["current_mb"] > 0
    assert stats["bfloat16_mb"] > 0


@pytest.mark.unit
def test_mixed_precision_empty_patterns():
    """Test mixed precision with empty pattern lists."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Empty fp32_layers should convert everything to BF16
    mixed_precision_bf16_fp32(model, fp32_layers=[])

    assert model.fc1.weight.dtype == mx.bfloat16
    assert model.fc2.weight.dtype == mx.bfloat16


@pytest.mark.unit
def test_convert_already_bfloat16():
    """Test converting a model that's already BFloat16."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Convert to BF16
    convert_to_bfloat16(model, inplace=True)
    assert is_bfloat16(model.fc1)

    # Save weights
    orig_weights = mx.array(model.fc1.weight)

    # Convert again (should be idempotent)
    convert_to_bfloat16(model, inplace=True)
    assert is_bfloat16(model.fc1)

    # Weights should be unchanged
    max_diff = mx.max(mx.abs(model.fc1.weight - orig_weights))
    assert float(max_diff) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
