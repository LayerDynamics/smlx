"""
Tests for automatic quantization strategy selection.

This module tests:
- analyze_model (model analysis with sensitivity)
- select_strategy (strategy selection logic)
- autoquant (automatic quantization)
- recommend_strategy (get all recommendations)
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant import (
    analyze_model,
    autoquant,
    recommend_strategy,
    select_strategy,
)


class TinyModel(nn.Module):
    """Tiny model <200M params for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.embed = nn.Embedding(100, 64)

    def __call__(self, x):
        return self.fc2(self.fc1(x))


class SmallModel(nn.Module):
    """Small model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.embed = nn.Embedding(1000, 256)

    def __call__(self, x):
        return self.fc2(self.fc1(x))


class TransformerModel(nn.Module):
    """Transformer model for testing."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, 256)
        self.self_attn_q = nn.Linear(256, 256)
        self.self_attn_k = nn.Linear(256, 256)
        self.self_attn_v = nn.Linear(256, 256)
        self.mlp_gate = nn.Linear(256, 1024)
        self.mlp_down = nn.Linear(1024, 256)

    def __call__(self, x):
        return x


@pytest.mark.unit
def test_analyze_model_basic():
    """Test basic model analysis."""
    model = SmallModel()
    mx.eval(model.parameters())

    info = analyze_model(model, calibration_data=None, use_sensitivity=False)

    # Check all expected keys
    assert "total_params" in info
    assert "quantizable_params" in info
    assert "model_size_mb" in info
    assert "architecture_type" in info
    assert "has_embeddings" in info
    assert "has_attention" in info
    assert "layer_count" in info
    assert "quantizable_ratio" in info

    # Check values make sense
    assert info["total_params"] > 0
    assert info["quantizable_params"] > 0
    assert info["model_size_mb"] > 0
    assert 0 <= info["quantizable_ratio"] <= 1.0


@pytest.mark.unit
def test_analyze_model_architecture_detection():
    """Test architecture type detection."""
    # Test embedding model
    embed_model = nn.Embedding(1000, 128)
    info = analyze_model(embed_model)
    assert info["architecture_type"] == "embedding"
    assert info["has_embeddings"] is True

    # Test transformer model
    transformer = TransformerModel()
    mx.eval(transformer.parameters())
    info = analyze_model(transformer)
    assert info["has_attention"] is True
    assert info["has_embeddings"] is True


@pytest.mark.unit
def test_analyze_model_sensitivity():
    """Test model analysis with sensitivity estimation."""
    model = SmallModel()
    mx.eval(model.parameters())

    # Create fake calibration data
    calibration_data = [mx.random.normal((10, 256)) for _ in range(5)]

    info = analyze_model(model, calibration_data=calibration_data, use_sensitivity=True)

    # Should have sensitivity scores
    assert "sensitivities" in info
    assert "avg_sensitivity" in info

    # avg_sensitivity should be a float
    assert isinstance(info["avg_sensitivity"], float)


@pytest.mark.unit
def test_select_strategy_aggressive():
    """Test strategy selection with aggressive profile."""
    model = SmallModel()
    mx.eval(model.parameters())

    info = analyze_model(model)
    strategy = select_strategy(info, profile="aggressive")

    assert "method" in strategy
    assert "reason" in strategy
    assert "expected_size_mb" in strategy
    assert "expected_quality" in strategy

    # Aggressive should target high compression
    assert strategy["expected_size_mb"] < info["model_size_mb"]


@pytest.mark.unit
def test_select_strategy_balanced():
    """Test strategy selection with balanced profile."""
    model = SmallModel()
    mx.eval(model.parameters())

    info = analyze_model(model)
    strategy = select_strategy(info, profile="balanced")

    assert "method" in strategy
    assert strategy["expected_quality"] in ["high", "very_high", "medium"]


@pytest.mark.unit
def test_select_strategy_conservative():
    """Test strategy selection with conservative profile."""
    model = SmallModel()
    mx.eval(model.parameters())

    info = analyze_model(model)
    strategy = select_strategy(info, profile="conservative")

    assert "method" in strategy
    # Conservative should prefer quality
    assert strategy["expected_quality"] in ["high", "very_high", "maximum"]


@pytest.mark.unit
def test_select_strategy_with_calibration():
    """Test strategy selection with calibration data available."""
    model = SmallModel()
    mx.eval(model.parameters())

    info = analyze_model(model)
    info["avg_sensitivity"] = 0.6  # High sensitivity

    strategy = select_strategy(info, profile="aggressive", calibration_available=True)

    # High sensitivity + calibration should pick an advanced, quality-preserving
    # method rather than a naive uniform one. "Advanced" includes calibration
    # methods (GPTQ/AWQ/DWQ), sensitivity-aware dynamic per-layer quantization,
    # and hardware-accelerated MXFP formats (preferred on M-series with OCP
    # microscaling) — not just GPTQ/AWQ.
    advanced_methods = {
        "gptq",
        "awq",
        "dwq",
        "dynamic",
        "quantize_mxfp4",
        "quantize_mxfp8",
        "mixed_2_4",
        "mixed_3_6",
        "mixed_4_6",
    }
    assert strategy["method"] in advanced_methods


@pytest.mark.unit
def test_select_strategy_memory_target():
    """Test strategy selection with memory target."""
    model = SmallModel()
    mx.eval(model.parameters())

    info = analyze_model(model)
    current_size = info["model_size_mb"]

    # Target 50% reduction
    target_mb = current_size * 0.5

    strategy = select_strategy(info, target_memory_mb=target_mb)

    # Should select method that achieves target
    assert strategy["expected_size_mb"] <= target_mb * 1.1  # Allow 10% margin


@pytest.mark.unit
def test_autoquant_basic():
    """Test basic autoquant functionality."""
    model = SmallModel()
    mx.eval(model.parameters())

    result = autoquant(model, profile="balanced", apply=False, verbose=False)

    # Check result structure
    assert "strategy" in result
    assert "model_info" in result
    assert "reason" in result
    assert "applied" in result
    assert "before_size_mb" in result

    assert result["applied"] is False  # We didn't apply


@pytest.mark.unit
def test_autoquant_all_profiles():
    """Test autoquant with all profiles."""
    for profile in ["aggressive", "balanced", "conservative"]:
        model = SmallModel()
        mx.eval(model.parameters())

        result = autoquant(model, profile=profile, apply=False, verbose=False)

        assert "strategy" in result
        assert result["strategy"]["method"] is not None


@pytest.mark.unit
def test_autoquant_with_target_memory():
    """Test autoquant with memory target."""
    model = SmallModel()
    mx.eval(model.parameters())

    # Get current size
    info = analyze_model(model)
    current_size = info["model_size_mb"]

    # Target very aggressive compression
    result = autoquant(model, target_memory_mb=current_size * 0.2, apply=False, verbose=False)

    # Should recommend aggressive method
    assert "strategy" in result
    # Expected size should be close to target
    expected_size = result["strategy"]["expected_size_mb"]
    assert expected_size < current_size * 0.3


@pytest.mark.unit
def test_autoquant_tiny_model():
    """Test autoquant behavior with tiny models."""
    model = TinyModel()
    mx.eval(model.parameters())

    result = autoquant(model, profile="aggressive", apply=False, verbose=False)

    # For tiny models, might recommend mixed precision
    assert "strategy" in result
    method = result["strategy"]["method"]
    assert method in ["mixed_3_6", "quantize_4bit", "quantize_6bit"]


@pytest.mark.unit
def test_recommend_strategy_basic():
    """Test strategy recommendations."""
    model = SmallModel()
    mx.eval(model.parameters())

    recommendations = recommend_strategy(model, verbose=False)

    # Should have recommendations for all profiles
    assert "aggressive" in recommendations
    assert "balanced" in recommendations
    assert "conservative" in recommendations
    assert "model_info" in recommendations

    # Each recommendation should have required fields
    for profile in ["aggressive", "balanced", "conservative"]:
        rec = recommendations[profile]
        assert "method" in rec
        assert "expected_size_mb" in rec
        assert "expected_quality" in rec


@pytest.mark.unit
def test_recommend_strategy_ordering():
    """Test that recommendations are ordered by compression."""
    model = SmallModel()
    mx.eval(model.parameters())

    recommendations = recommend_strategy(model, verbose=False)

    aggressive_size = recommendations["aggressive"]["expected_size_mb"]
    balanced_size = recommendations["balanced"]["expected_size_mb"]
    conservative_size = recommendations["conservative"]["expected_size_mb"]

    # Aggressive should be smallest
    assert aggressive_size <= balanced_size
    assert balanced_size <= conservative_size


@pytest.mark.unit
def test_autoquant_with_calibration():
    """Test autoquant with calibration data."""
    model = SmallModel()
    mx.eval(model.parameters())

    # Create fake calibration data
    calibration_data = [mx.random.normal((10, 256)) for _ in range(5)]

    result = autoquant(
        model, calibration_data=calibration_data, profile="balanced", apply=False, verbose=False
    )

    # Should have sensitivity info
    assert "model_info" in result
    assert "avg_sensitivity" in result["model_info"]


@pytest.mark.unit
def test_autoquant_transformer_model():
    """Test autoquant with transformer architecture."""
    model = TransformerModel()
    mx.eval(model.parameters())

    result = autoquant(model, profile="balanced", apply=False, verbose=False)

    # Should detect transformer architecture
    assert result["model_info"]["architecture_type"] == "transformer"

    # Might recommend mixed precision for transformers
    method = result["strategy"]["method"]
    assert method in ["mixed_3_6", "quantize_4bit", "quantize_6bit", "quantize_8bit"]


@pytest.mark.unit
def test_autoquant_sensitivity_aware():
    """Test that autoquant uses sensitivity for decisions."""
    model = SmallModel()
    mx.eval(model.parameters())

    # Manually set high sensitivity
    info = analyze_model(model)
    info["avg_sensitivity"] = 0.8  # Very high sensitivity
    info["sensitivities"] = {"fc1": 0.8, "fc2": 0.8}

    strategy = select_strategy(info, profile="balanced", calibration_available=True)

    # Should recommend an advanced, quality-preserving method for high
    # sensitivity — calibration methods, sensitivity-aware dynamic quantization,
    # or hardware-accelerated MXFP formats (not a naive uniform method).
    advanced_methods = {
        "gptq",
        "awq",
        "dwq",
        "dynamic",
        "quantize_mxfp4",
        "quantize_mxfp8",
        "mixed_2_4",
        "mixed_3_6",
        "mixed_4_6",
    }
    assert strategy["method"] in advanced_methods
    # The decision must be justified by sensitivity (or by naming the method).
    reason = strategy["reason"].lower()
    assert "sensitivity" in reason or strategy["method"] in reason


@pytest.mark.unit
def test_autoquant_fine_tuning_scenario():
    """Test autoquant for fine-tuning scenario (minimal compression)."""
    model = SmallModel()
    mx.eval(model.parameters())

    info = analyze_model(model)
    current_size = info["model_size_mb"]

    # Target 95% of current size (minimal compression)
    result = autoquant(model, target_memory_mb=current_size * 0.95, apply=False, verbose=False)

    # Should recommend BFloat16 for minimal compression
    assert result["strategy"]["method"] == "convert_to_bfloat16"


@pytest.mark.unit
def test_autoquant_error_handling():
    """Test autoquant error handling."""
    model = SmallModel()
    mx.eval(model.parameters())

    # Apply with invalid method should handle gracefully
    result = autoquant(model, profile="balanced", apply=False, verbose=False)

    # Should not crash
    assert "strategy" in result
    assert "error" not in result  # No errors when not applying


@pytest.mark.unit
def test_select_strategy_size_categories():
    """Test strategy selection across model size categories."""
    # Tiny model (<200M)
    tiny_info = {
        "total_params": 100_000_000,
        "model_size_mb": 200.0,
        "architecture_type": "transformer",
    }
    strategy = select_strategy(tiny_info, profile="aggressive")
    assert "method" in strategy

    # Small model (200-500M)
    small_info = {
        "total_params": 300_000_000,
        "model_size_mb": 600.0,
        "architecture_type": "transformer",
    }
    strategy = select_strategy(small_info, profile="balanced")
    assert "method" in strategy

    # Medium model (500M-1B)
    medium_info = {
        "total_params": 700_000_000,
        "model_size_mb": 1400.0,
        "architecture_type": "transformer",
    }
    strategy = select_strategy(medium_info, profile="conservative", calibration_available=True)
    # Medium models with calibration should use advanced methods
    assert strategy["method"] in ["gptq", "quantize_8bit"]


@pytest.mark.unit
def test_autoquant_reason_explanations():
    """Test that autoquant provides clear reasoning."""
    model = SmallModel()
    mx.eval(model.parameters())

    result = autoquant(model, profile="balanced", apply=False, verbose=False)

    # Should have explanation
    assert "reason" in result
    assert len(result["reason"]) > 0
    assert isinstance(result["reason"], str)

    # Reason should mention key factors
    reason_lower = result["reason"].lower()
    # Should mention either compression, quantization, or quality
    assert any(word in reason_lower for word in ["compression", "quantization", "quality", "bit"])


@pytest.mark.unit
def test_autoquant_expected_vs_actual():
    """Test autoquant expected size estimation."""
    model = SmallModel()
    mx.eval(model.parameters())

    result = autoquant(model, profile="balanced", apply=False, verbose=False)

    # Expected size should be reasonable
    expected = result["strategy"]["expected_size_mb"]
    current = result["before_size_mb"]

    # Expected should be smaller
    assert expected < current

    # But not impossibly small
    assert expected > current * 0.1  # At least 10% of original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
