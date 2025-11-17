"""
Tests for floating-point quantization formats (FP4 and FP8).

This module tests:
- FP4 E2M1 quantization and dequantization
- FP8 E4M3 quantization and dequantization (DEPRECATED - simulated only)
- FP8 E5M2 quantization and dequantization (DEPRECATED - simulated only)
- Model-level FP4/FP8 quantization
- Format comparisons
- Deprecation warnings for FP8 (use MXFP8 instead)
"""

import warnings

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant import (
    FP4_E2M1_VALUES,
    compare_fp4_vs_int4,
    compare_fp8_formats,
    compare_fp8_vs_int8,
    dequantize_from_fp4,
    dequantize_from_fp8,
    estimate_fp4_size,
    estimate_fp8_size,
    quantize_model_fp4,
    quantize_model_fp8,
    quantize_to_fp4,
    quantize_to_fp8_e4m3,
    quantize_to_fp8_e5m2,
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


@pytest.mark.unit
def test_fp4_e2m1_values():
    """Test FP4 E2M1 lookup table."""
    assert FP4_E2M1_VALUES.shape == (16,)
    # Check that values are sorted (positive then negative)
    positive = FP4_E2M1_VALUES[:8]
    negative = FP4_E2M1_VALUES[8:]
    assert all(positive >= 0)
    assert all(negative <= 0)


@pytest.mark.unit
def test_quantize_to_fp4():
    """Test FP4 weight quantization."""
    weight = mx.random.normal((64, 64))

    quantized, scales = quantize_to_fp4(weight, group_size=16)

    # Check shapes
    assert quantized.shape == weight.shape
    # FP4 has 16 values, stored as indices
    # Scales are flattened across all groups (1D array)
    total_elements = weight.size
    expected_groups = (total_elements + 15) // 16
    assert scales.shape == (expected_groups,)


@pytest.mark.unit
def test_fp4_quantize_dequantize():
    """Test FP4 quantization round-trip."""
    weight = mx.random.normal((32, 32)) * 2.0  # Scale to FP4 range

    quantized, scales = quantize_to_fp4(weight)
    restored = dequantize_from_fp4(quantized, scales)

    # Check shape preserved
    assert restored.shape == weight.shape

    # FP4 is lossy but should be somewhat close
    # With only 16 values, expect significant error
    error = mx.abs(weight - restored)
    mean_error = mx.mean(error)
    assert float(mean_error) < 1.0  # Loose bound for FP4


@pytest.mark.unit
def test_quantize_model_fp4():
    """Test model-level FP4 quantization."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Get original weights
    orig_fc1 = mx.array(model.fc1.weight)

    # Quantize model (returns dict of quantized weights, doesn't modify model)
    quantized_weights = quantize_model_fp4(model)

    # Should have entries for linear layers
    assert len(quantized_weights) > 0

    # Check that quantized data exists for fc1
    fc1_key = next((k for k in quantized_weights.keys() if 'fc1' in k), None)
    assert fc1_key is not None

    indices, scales = quantized_weights[fc1_key]
    assert indices.shape == orig_fc1.shape
    assert scales.ndim == 1  # Flattened scales


@pytest.mark.unit
def test_estimate_fp4_size():
    """Test FP4 size estimation."""
    model = SimpleModel()
    mx.eval(model.parameters())

    stats = estimate_fp4_size(model)

    assert "current_mb" in stats
    assert "fp4_mb" in stats
    assert "reduction_ratio" in stats

    # FP4 should be smaller than FP16
    assert stats["fp4_mb"] < stats["current_mb"]
    assert stats["reduction_ratio"] > 1.0


@pytest.mark.unit
def test_compare_fp4_vs_int4():
    """Test FP4 vs INT4 comparison."""
    weight = mx.random.normal((64, 64))

    comparison = compare_fp4_vs_int4(weight)

    assert "fp4_error" in comparison
    assert "int4_error" in comparison
    assert "recommendation" in comparison

    # Both errors should be reasonable
    assert comparison["fp4_error"] < 1.0
    assert comparison["int4_error"] < 1.0


@pytest.mark.unit
def test_quantize_to_fp8_e4m3():
    """Test FP8 E4M3 weight quantization."""
    weight = mx.random.normal((64, 64)) * 10.0

    # Suppress deprecation warning for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        quantized, scales = quantize_to_fp8_e4m3(weight, group_size=32)

    # Check shapes
    assert quantized.shape == weight.shape
    # Scales are flattened across all groups (1D array)
    total_elements = weight.size
    expected_groups = (total_elements + 31) // 32
    assert scales.shape == (expected_groups,)


@pytest.mark.unit
def test_quantize_to_fp8_e5m2():
    """Test FP8 E5M2 weight quantization."""
    weight = mx.random.normal((64, 64)) * 100.0  # Larger range for E5M2

    # Suppress deprecation warning for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        quantized, scales = quantize_to_fp8_e5m2(weight, group_size=32)

    # Check shapes
    assert quantized.shape == weight.shape
    # Scales are flattened across all groups (1D array)
    total_elements = weight.size
    expected_groups = (total_elements + 31) // 32
    assert scales.shape == (expected_groups,)


@pytest.mark.unit
def test_fp8_e4m3_quantize_dequantize():
    """Test FP8 E4M3 quantization round-trip."""
    weight = mx.random.normal((32, 32)) * 10.0

    # Suppress deprecation warnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        quantized, scales = quantize_to_fp8_e4m3(weight)
        restored = dequantize_from_fp8(quantized, scales)

    # Check shape preserved
    assert restored.shape == weight.shape

    # FP8 simulation with limited mantissa bits is fairly lossy
    # E4M3 has only 3 mantissa bits (8 levels between powers of 2)
    rel_error = mx.mean(mx.abs(weight - restored) / (mx.abs(weight) + 1e-6))
    assert float(rel_error) < 0.35  # Within 35% relative error (due to simulation)


@pytest.mark.unit
def test_fp8_e5m2_quantize_dequantize():
    """Test FP8 E5M2 quantization round-trip."""
    weight = mx.random.normal((32, 32)) * 100.0

    # Suppress deprecation warnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        quantized, scales = quantize_to_fp8_e5m2(weight)
        restored = dequantize_from_fp8(quantized, scales)

    # Check shape preserved
    assert restored.shape == weight.shape

    # E5M2 has wider range but less precision (only 2 mantissa bits)
    # Simulation is fairly lossy
    rel_error = mx.mean(mx.abs(weight - restored) / (mx.abs(weight) + 1e-6))
    assert float(rel_error) < 0.5  # Within 50% relative error (due to only 2 mantissa bits)


@pytest.mark.unit
def test_quantize_model_fp8():
    """Test model-level FP8 quantization."""
    model = SimpleModel()
    mx.eval(model.parameters())

    orig_fc1 = mx.array(model.fc1.weight)

    # Suppress deprecation warning for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # Quantize model with E4M3 (returns dict, doesn't modify model)
        quantized_weights = quantize_model_fp8(model, format="e4m3")

    # Should have entries for linear layers
    assert len(quantized_weights) > 0

    # Check that quantized data exists for fc1
    fc1_key = next((k for k in quantized_weights.keys() if 'fc1' in k), None)
    assert fc1_key is not None

    values, scales = quantized_weights[fc1_key]
    assert values.shape == orig_fc1.shape
    assert scales.ndim == 1  # Flattened scales


@pytest.mark.unit
def test_estimate_fp8_size():
    """Test FP8 size estimation."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Suppress deprecation warning for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        stats = estimate_fp8_size(model)

    assert "current_mb" in stats
    assert "fp8_mb" in stats
    assert "reduction_ratio" in stats

    # FP8 is simulated as float16, so reduction ratio is ~1.0
    # (Native FP8 hardware would provide true 8-bit storage and ~2x compression)
    assert 0.95 < stats["reduction_ratio"] < 1.05  # Minimal reduction in simulation


@pytest.mark.unit
def test_compare_fp8_formats():
    """Test FP8 E4M3 vs E5M2 comparison."""
    weight = mx.random.normal((64, 64)) * 50.0

    # Suppress deprecation warnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        comparison = compare_fp8_formats(weight)

    assert "e4m3_error" in comparison
    assert "e5m2_error" in comparison
    assert "recommendation" in comparison

    # Errors depend on weight distribution and simulation precision
    # E4M3 has better precision, E5M2 has wider range but coarser quantization
    assert comparison["e4m3_error"] < 10.0
    assert comparison["e5m2_error"] < 15.0


@pytest.mark.unit
def test_compare_fp8_vs_int8():
    """Test FP8 vs INT8 comparison."""
    weight = mx.random.normal((64, 64)) * 10.0

    # Suppress deprecation warnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        comparison = compare_fp8_vs_int8(weight)

    assert "fp8_error" in comparison
    assert "int8_error" in comparison
    assert "recommendation" in comparison

    # Both errors should be reasonable
    assert comparison["fp8_error"] < 1.0
    assert comparison["int8_error"] < 1.0


@pytest.mark.unit
def test_fp4_extreme_values():
    """Test FP4 with extreme values."""
    # Test with values outside FP4 range
    weight = mx.array([[10.0, -10.0], [100.0, -100.0]])

    quantized, scales = quantize_to_fp4(weight)
    restored = dequantize_from_fp4(quantized, scales)

    # Should not crash with extreme values
    assert restored.shape == weight.shape

    # With per-group scaling, extreme values are preserved within their group
    # The scale factor normalizes the group, so actual values are maintained
    # Check that restoration is reasonable (values exist and are finite)
    assert mx.all(mx.isfinite(restored))

    # Check that quantization preserves relative magnitudes
    # (larger values should still be larger after quantization)
    assert float(mx.abs(restored[1, 0])) > float(mx.abs(restored[0, 0]))


@pytest.mark.unit
def test_fp8_extreme_values():
    """Test FP8 with extreme values."""
    # Test E4M3 with values outside range
    weight = mx.array([[1000.0, -1000.0], [10000.0, -10000.0]])

    # Suppress deprecation warnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        quantized, scales = quantize_to_fp8_e4m3(weight)
        restored = dequantize_from_fp8(quantized, scales)

    # Should clip but not crash
    assert restored.shape == weight.shape


@pytest.mark.unit
def test_fp4_zero_handling():
    """Test FP4 quantization of zero values."""
    weight = mx.zeros((16, 16))

    quantized, scales = quantize_to_fp4(weight)
    restored = dequantize_from_fp4(quantized, scales)

    # Zeros should remain zeros (or very close)
    assert float(mx.max(mx.abs(restored))) < 0.1


@pytest.mark.unit
def test_fp8_zero_handling():
    """Test FP8 quantization of zero values."""
    weight = mx.zeros((16, 16))

    # Suppress deprecation warnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        quantized, scales = quantize_to_fp8_e4m3(weight)
        restored = dequantize_from_fp8(quantized, scales)

    # Zeros should remain zeros (or very close)
    assert float(mx.max(mx.abs(restored))) < 0.01


@pytest.mark.unit
def test_fp4_different_group_sizes():
    """Test FP4 with different group sizes."""
    weight = mx.random.normal((64, 64))

    for group_size in [8, 16, 32, 64]:
        quantized, scales = quantize_to_fp4(weight, group_size=group_size)
        restored = dequantize_from_fp4(quantized, scales, group_size=group_size)

        assert restored.shape == weight.shape


@pytest.mark.unit
def test_fp8_different_group_sizes():
    """Test FP8 with different group sizes."""
    weight = mx.random.normal((64, 64))

    for group_size in [16, 32, 64]:
        # Suppress deprecation warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            quantized, scales = quantize_to_fp8_e4m3(weight, group_size=group_size)
            restored = dequantize_from_fp8(quantized, scales, group_size=group_size)

        assert restored.shape == weight.shape


@pytest.mark.unit
def test_fp8_deprecation_warnings():
    """Test that FP8 functions emit deprecation warnings."""
    weight = mx.random.normal((32, 32))

    # Test quantize_to_fp8_e4m3
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        quantize_to_fp8_e4m3(weight)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "mxfp8" in str(w[0].message).lower()
        assert "deprecated" in str(w[0].message).lower()

    # Test quantize_to_fp8_e5m2
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        quantize_to_fp8_e5m2(weight)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "mxfp8" in str(w[0].message).lower()

    # Test dequantize_from_fp8
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Need to quantize first (which will also emit warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values, scales = quantize_to_fp8_e4m3(weight)

        # Now test dequantize
        warnings.simplefilter("always")
        dequantize_from_fp8(values, scales)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "mxfp8" in str(w[0].message).lower()

    # Test quantize_model_fp8
    model = SimpleModel()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        quantize_model_fp8(model)

        # Model quantization may emit multiple warnings (one per layer)
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "mxfp8" in str(w[0].message).lower()

    # Test estimate_fp8_size
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        estimate_fp8_size(model)

        # Model size estimation may emit multiple warnings
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "mxfp8" in str(w[0].message).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
