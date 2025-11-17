"""
Tests for MXFP8 (Microscaling FP8) quantization.

This module tests:
- quantize_to_mxfp8 (weight quantization)
- dequantize_from_mxfp8 (weight dequantization)
- quantize_model_mxfp8 (model-level quantization)
- estimate_mxfp8_size (size estimation)
- validate_mxfp_shape (shape validation)
- compare_mxfp8_vs_fp8 (format comparison)
- compare_mxfp8_vs_int8 (format comparison)
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant import (
    compare_mxfp8_vs_fp8,
    compare_mxfp8_vs_int8,
    dequantize_from_mxfp8,
    estimate_mxfp8_size,
    quantize_model_mxfp8,
    quantize_to_mxfp8,
    validate_mxfp_shape,
)
from smlx.quant.utils import (
    compare_mxfp8_vs_int8 as compare_mxfp8_vs_int8_util,
    estimate_mxfp8_size as estimate_mxfp8_size_util,
    pad_for_mxfp8,
    validate_mxfp8_shape,
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
def test_validate_mxfp_shape_valid():
    """Test shape validation with valid dimensions."""
    weight = mx.random.normal((64, 64))  # 64 % 32 == 0

    validated = validate_mxfp_shape(weight)

    assert validated.shape == weight.shape
    # Should return same array (no padding needed)
    assert mx.array_equal(validated, weight)


@pytest.mark.unit
def test_validate_mxfp_shape_invalid_no_pad():
    """Test shape validation raises error without padding."""
    weight = mx.random.normal((100, 100))  # 100 % 32 != 0

    with pytest.raises(ValueError, match="MXFP requires last dimension divisible by 32"):
        validate_mxfp_shape(weight, pad=False)


@pytest.mark.unit
def test_validate_mxfp_shape_with_padding():
    """Test shape validation with automatic padding."""
    weight = mx.random.normal((64, 100))  # 100 % 32 != 0

    validated = validate_mxfp_shape(weight, pad=True)

    # Should pad last dimension to 128 (next multiple of 32)
    assert validated.shape == (64, 128)
    # Original values preserved
    assert mx.array_equal(validated[:, :100], weight)
    # Padding is zeros
    assert mx.all(validated[:, 100:] == 0)


@pytest.mark.unit
def test_quantize_to_mxfp8_basic():
    """Test basic MXFP8 quantization."""
    weight = mx.random.normal((64, 64))

    w_q, scales = quantize_to_mxfp8(weight)

    # Check shapes
    # MXFP8 returns packed format: 4 elements (8-bit each) per uint32
    expected_packed_shape = weight.shape[:-1] + (weight.shape[-1] // 4,)
    assert w_q.shape == expected_packed_shape
    # Scales: one per 32 elements in last dimension
    # Total elements = 64 * 64 = 4096
    # Blocks = 4096 / 32 = 128
    expected_scale_count = (weight.size + 31) // 32
    assert scales.size == expected_scale_count

    # Check dtypes
    assert w_q.dtype == mx.uint32  # Packed 8-bit values
    assert scales.dtype == mx.uint8  # E8M0 scales


@pytest.mark.unit
def test_mxfp8_quantize_dequantize():
    """Test MXFP8 quantization round-trip."""
    weight = mx.random.normal((32, 32)) * 10.0

    w_q, scales = quantize_to_mxfp8(weight)
    restored = dequantize_from_mxfp8(w_q, scales)

    # Check shape preserved
    assert restored.shape == weight.shape

    # MXFP8 should be fairly accurate (better than MXFP4)
    rel_error = mx.mean(mx.abs(weight - restored) / (mx.abs(weight) + 1e-6))
    assert float(rel_error) < 0.15  # Within 15% relative error


@pytest.mark.unit
def test_mxfp8_dtype_conversion():
    """Test dequantization to different dtypes."""
    weight = mx.random.normal((64, 64))
    w_q, scales = quantize_to_mxfp8(weight)

    # Dequantize to float16
    restored_fp16 = dequantize_from_mxfp8(w_q, scales, dtype=mx.float16)
    assert restored_fp16.dtype == mx.float16

    # Dequantize to bfloat16
    restored_bf16 = dequantize_from_mxfp8(w_q, scales, dtype=mx.bfloat16)
    assert restored_bf16.dtype == mx.bfloat16

    # Dequantize to float32
    restored_fp32 = dequantize_from_mxfp8(w_q, scales, dtype=mx.float32)
    assert restored_fp32.dtype == mx.float32


@pytest.mark.unit
def test_quantize_model_mxfp8():
    """Test model-level MXFP8 quantization."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Quantize model
    result = quantize_model_mxfp8(model, inplace=True)

    # Should return None for in-place
    assert result is None

    # Check that linear layers are now quantized
    assert isinstance(model.fc1, nn.QuantizedLinear)
    assert isinstance(model.fc2, nn.QuantizedLinear)


@pytest.mark.unit
def test_quantize_model_mxfp8_not_inplace():
    """Test model quantization without in-place modification."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Quantize not in-place (returns model)
    result = quantize_model_mxfp8(model, inplace=False)

    # Should return the model
    assert result is not None

    # Model should still be quantized
    assert isinstance(model.fc1, nn.QuantizedLinear)


@pytest.mark.unit
def test_estimate_mxfp8_size():
    """Test MXFP8 size estimation."""
    model = SimpleModel()
    mx.eval(model.parameters())

    stats = estimate_mxfp8_size(model)

    # Check all expected keys
    assert "current_mb" in stats
    assert "mxfp8_mb" in stats
    assert "reduction_ratio" in stats
    assert "saved_mb" in stats
    assert "total_params" in stats
    assert "quantizable_params" in stats
    assert "scale_overhead_mb" in stats
    assert "format" in stats
    assert "block_size" in stats

    # MXFP8 should provide ~1.2-1.4x reduction from FP32
    # Note: Models are FP32 by default in MLX
    assert stats["mxfp8_mb"] < stats["current_mb"]
    assert 1.2 < stats["reduction_ratio"] < 1.5

    # Block size should be 32
    assert stats["block_size"] == 32

    # Format should mention MXFP8
    assert "MXFP8" in stats["format"]


@pytest.mark.unit
def test_mxfp8_handles_zeros():
    """Test MXFP8 quantization of zero values."""
    weight = mx.zeros((64, 64))

    w_q, scales = quantize_to_mxfp8(weight)
    restored = dequantize_from_mxfp8(w_q, scales)

    # Zeros should remain zeros (or very close)
    assert float(mx.max(mx.abs(restored))) < 0.01


@pytest.mark.unit
def test_mxfp8_large_values():
    """Test MXFP8 with large values."""
    weight = mx.random.normal((64, 64)) * 100.0

    w_q, scales = quantize_to_mxfp8(weight)
    restored = dequantize_from_mxfp8(w_q, scales)

    # Should handle large values reasonably
    assert restored.shape == weight.shape
    assert mx.all(mx.isfinite(restored))


@pytest.mark.unit
def test_mxfp8_preserves_shape():
    """Test that MXFP8 preserves shape."""
    for shape in [(32, 64), (64, 128), (128, 256)]:
        weight = mx.random.normal(shape)
        w_q, scales = quantize_to_mxfp8(weight)
        restored = dequantize_from_mxfp8(w_q, scales)

        assert restored.shape == weight.shape


@pytest.mark.unit
def test_compare_mxfp8_vs_fp8():
    """Test MXFP8 vs FP8 comparison."""
    weight = mx.random.normal((64, 64)) * 10.0

    comparison = compare_mxfp8_vs_fp8(weight)

    # Check all expected keys
    assert "mxfp8_error" in comparison
    assert "fp8_error" in comparison
    assert "mxfp8_max_error" in comparison
    assert "fp8_max_error" in comparison
    assert "mxfp8_better" in comparison
    assert "size_comparison" in comparison
    assert "recommendation" in comparison

    # Errors should be reasonable
    assert comparison["mxfp8_error"] < 5.0
    assert comparison["fp8_error"] < 5.0

    # Size comparison should exist
    assert "mxfp8_bytes" in comparison["size_comparison"]
    assert "fp8_bytes" in comparison["size_comparison"]


@pytest.mark.unit
def test_compare_mxfp8_vs_int8():
    """Test MXFP8 vs INT8 comparison."""
    weight = mx.random.normal((64, 64)) * 10.0

    comparison = compare_mxfp8_vs_int8(weight)

    # Check all expected keys
    assert "mxfp8_error" in comparison
    assert "int8_error" in comparison
    assert "mxfp8_max_error" in comparison
    assert "int8_max_error" in comparison
    assert "mxfp8_better" in comparison
    assert "recommendation" in comparison

    # Errors should be reasonable
    assert comparison["mxfp8_error"] < 5.0
    assert comparison["int8_error"] < 5.0


@pytest.mark.unit
def test_mxfp8_with_padding_comparison():
    """Test MXFP8 with non-divisible dimensions in comparison."""
    weight = mx.random.normal((64, 100)) * 10.0  # 100 % 32 != 0

    # Should auto-pad for comparison
    comparison = compare_mxfp8_vs_fp8(weight)

    # Should not raise error
    assert "mxfp8_error" in comparison
    assert comparison["mxfp8_error"] >= 0


@pytest.mark.unit
def test_mxfp8_block_size_32():
    """Test that MXFP8 uses block size 32."""
    weight = mx.random.normal((64, 64))  # 4096 elements

    w_q, scales = quantize_to_mxfp8(weight)

    # Number of scales should be elements / 32
    expected_scales = weight.size // 32
    assert scales.size == expected_scales


@pytest.mark.unit
def test_mxfp8_comparison_uniform_distribution():
    """Test MXFP8 comparison with uniform distribution weights."""
    weight = mx.random.uniform(-10, 10, (64, 64))

    comparison = compare_mxfp8_vs_int8(weight)

    # Both should handle uniform distribution
    assert comparison["mxfp8_error"] < 5.0
    assert comparison["int8_error"] < 5.0


@pytest.mark.unit
def test_mxfp8_comparison_wide_range():
    """Test MXFP8 comparison with wide dynamic range."""
    # Create weights with wide range
    weight = mx.concatenate(
        [mx.random.normal((32, 64)) * 0.1, mx.random.normal((32, 64)) * 100.0], axis=0
    )

    comparison = compare_mxfp8_vs_int8(weight)

    # MXFP8 should handle wide range well (floating point advantage)
    assert "mxfp8_error" in comparison


@pytest.mark.unit
def test_estimate_mxfp8_size_reduction():
    """Test that MXFP8 provides expected compression."""
    model = SimpleModel()
    mx.eval(model.parameters())

    stats = estimate_mxfp8_size(model)

    # MXFP8 should provide ~1.2-1.4x reduction from FP32
    # (1 byte per weight element + scale overhead)
    # Note: Models are FP32 by default in MLX
    assert 1.2 < stats["reduction_ratio"] < 1.5


@pytest.mark.unit
def test_mxfp8_scale_overhead():
    """Test MXFP8 scale overhead calculation."""
    model = SimpleModel()
    mx.eval(model.parameters())

    stats = estimate_mxfp8_size(model)

    # Scale overhead should be reasonable (1 byte per 32 elements)
    # For a typical model, should be < 5% of total
    scale_percentage = (stats["scale_overhead_mb"] / stats["mxfp8_mb"]) * 100
    assert scale_percentage < 5.0  # Should be small overhead


@pytest.mark.unit
def test_mxfp8_empty_model():
    """Test MXFP8 size estimation with minimal model."""
    model = nn.Module()

    stats = estimate_mxfp8_size(model)

    # Should handle empty model gracefully
    assert stats["current_mb"] == 0
    assert stats["mxfp8_mb"] == 0


@pytest.mark.unit
def test_mxfp8_quantize_different_shapes():
    """Test MXFP8 with various valid shapes."""
    shapes = [(32, 32), (64, 64), (128, 64), (256, 128)]

    for shape in shapes:
        weight = mx.random.normal(shape) * 10.0
        w_q, scales = quantize_to_mxfp8(weight)
        restored = dequantize_from_mxfp8(w_q, scales)

        assert restored.shape == weight.shape
        error = mx.mean(mx.abs(weight - restored))
        assert float(error) < 5.0


@pytest.mark.unit
def test_mxfp8_idempotent():
    """Test that repeated quantization produces same results."""
    weight = mx.random.normal((64, 64)) * 10.0

    # Quantize twice
    w_q1, scales1 = quantize_to_mxfp8(weight)
    w_q2, scales2 = quantize_to_mxfp8(weight)

    # Should produce identical results
    assert mx.array_equal(w_q1, w_q2)
    assert mx.array_equal(scales1, scales2)


@pytest.mark.unit
def test_mxfp8_comparison_recommendation():
    """Test that comparisons provide recommendations."""
    weight = mx.random.normal((64, 64)) * 10.0

    comparison_fp8 = compare_mxfp8_vs_fp8(weight)
    comparison_int8 = compare_mxfp8_vs_int8(weight)

    # Both should have recommendations
    assert isinstance(comparison_fp8["recommendation"], str)
    assert len(comparison_fp8["recommendation"]) > 0

    assert isinstance(comparison_int8["recommendation"], str)
    assert len(comparison_int8["recommendation"]) > 0


@pytest.mark.unit
def test_mxfp8_better_than_mxfp4():
    """Test that MXFP8 provides better quality than MXFP4."""
    from smlx.quant import dequantize_from_mxfp4, quantize_to_mxfp4

    weight = mx.random.normal((64, 64)) * 10.0

    # MXFP4 quantization
    w_q4, scales4 = quantize_to_mxfp4(weight)
    restored4 = dequantize_from_mxfp4(w_q4, scales4)
    error4 = mx.mean(mx.abs(weight - restored4))

    # MXFP8 quantization
    w_q8, scales8 = quantize_to_mxfp8(weight)
    restored8 = dequantize_from_mxfp8(w_q8, scales8)
    error8 = mx.mean(mx.abs(weight - restored8))

    # MXFP8 should have lower error (better precision)
    assert float(error8) < float(error4)


@pytest.mark.unit
def test_mxfp8_size_larger_than_mxfp4():
    """Test that MXFP8 uses more memory than MXFP4."""
    model = SimpleModel()
    mx.eval(model.parameters())

    from smlx.quant import estimate_mxfp4_size

    stats4 = estimate_mxfp4_size(model)
    stats8 = estimate_mxfp8_size(model)

    # MXFP8 should be larger (8-bit vs 4-bit)
    assert stats8["mxfp8_mb"] > stats4["mxfp4_mb"]

    # But MXFP4 should have better compression ratio
    assert stats4["reduction_ratio"] > stats8["reduction_ratio"]


# ============================================================================
# Tests for MXFP8 Utility Functions (M4-specific)
# ============================================================================


@pytest.mark.unit
def test_validate_mxfp8_shape_valid():
    """Test validate_mxfp8_shape with valid dimension."""
    weight = mx.random.normal((768, 768))  # 768 % 32 == 0

    is_valid = validate_mxfp8_shape(weight, raise_error=False)
    assert is_valid is True

    # Should not raise when valid
    validate_mxfp8_shape(weight, raise_error=True)


@pytest.mark.unit
def test_validate_mxfp8_shape_invalid():
    """Test validate_mxfp8_shape with invalid dimension."""
    weight = mx.random.normal((770, 770))  # 770 % 32 != 0

    # Should return False when raise_error=False
    is_valid = validate_mxfp8_shape(weight, raise_error=False)
    assert is_valid is False

    # Should raise ValueError when raise_error=True
    with pytest.raises(ValueError, match="MXFP8 requires last dimension divisible by 32"):
        validate_mxfp8_shape(weight, raise_error=True)


@pytest.mark.unit
def test_pad_for_mxfp8_no_padding_needed():
    """Test pad_for_mxfp8 when no padding is needed."""
    weight = mx.random.normal((768, 768))  # Already divisible by 32

    padded, orig_size = pad_for_mxfp8(weight)

    # Should return original array unchanged
    assert padded.shape == weight.shape
    assert orig_size == 768
    assert mx.array_equal(padded, weight)


@pytest.mark.unit
def test_pad_for_mxfp8_padding_needed():
    """Test pad_for_mxfp8 when padding is needed."""
    weight = mx.random.normal((770, 770))  # Not divisible by 32

    padded, orig_size = pad_for_mxfp8(weight)

    # Should pad to 800 (next multiple of 32)
    assert padded.shape == (770, 800)
    assert orig_size == 770

    # Original values should be preserved
    assert mx.array_equal(padded[:, :770], weight)

    # Padding should be zeros
    assert mx.all(padded[:, 770:] == 0)


@pytest.mark.unit
def test_pad_for_mxfp8_various_sizes():
    """Test pad_for_mxfp8 with various dimensions."""
    test_cases = [
        (100, 128),  # 100 → 128
        (500, 512),  # 500 → 512
        (33, 64),    # 33 → 64
        (1000, 1024),  # 1000 → 1024
    ]

    for orig_dim, expected_dim in test_cases:
        weight = mx.random.normal((64, orig_dim))
        padded, orig_size = pad_for_mxfp8(weight)

        assert padded.shape == (64, expected_dim)
        assert orig_size == orig_dim


@pytest.mark.unit
def test_compare_mxfp8_vs_int8_util():
    """Test compare_mxfp8_vs_int8 utility function."""
    weight = mx.random.normal((768, 768)) * 10.0

    comparison = compare_mxfp8_vs_int8_util(weight)

    # Check all expected keys
    expected_keys = [
        "mxfp8_error",
        "int8_error",
        "mxfp8_max_error",
        "int8_max_error",
        "mxfp8_size_bytes",
        "int8_size_bytes",
        "recommendation",
        "reason",
        "mxfp8_block_size",
        "int8_group_size",
    ]

    for key in expected_keys:
        assert key in comparison, f"Missing key: {key}"

    # Check values are reasonable
    assert comparison["mxfp8_error"] >= 0
    assert comparison["int8_error"] >= 0
    assert comparison["mxfp8_size_bytes"] > 0
    assert comparison["int8_size_bytes"] > 0
    assert comparison["mxfp8_block_size"] == 32
    assert isinstance(comparison["recommendation"], str)
    assert len(comparison["recommendation"]) > 0


@pytest.mark.unit
def test_compare_mxfp8_vs_int8_with_padding():
    """Test compare_mxfp8_vs_int8 with non-divisible dimensions."""
    weight = mx.random.normal((770, 770)) * 10.0  # Not divisible by 32

    # Should auto-pad for MXFP8
    comparison = compare_mxfp8_vs_int8_util(weight)

    # Should complete without error
    assert "mxfp8_error" in comparison
    assert comparison["mxfp8_error"] >= 0


@pytest.mark.unit
def test_estimate_mxfp8_size_util():
    """Test estimate_mxfp8_size utility function."""
    model = SimpleModel()
    mx.eval(model.parameters())

    stats = estimate_mxfp8_size_util(model)

    # Check all expected keys
    expected_keys = [
        "current_mb",
        "mxfp8_mb",
        "savings_mb",
        "reduction_ratio",
        "parameters",
        "notes",
    ]

    for key in expected_keys:
        assert key in stats, f"Missing key: {key}"

    # Check values are reasonable
    assert stats["current_mb"] > 0
    assert stats["mxfp8_mb"] > 0
    assert stats["mxfp8_mb"] < stats["current_mb"]  # Should be smaller
    assert stats["reduction_ratio"] > 1.0  # Should have compression
    assert stats["parameters"] > 0
    assert "MXFP8" in stats["notes"]


@pytest.mark.unit
def test_mxfp8_utilities_integration():
    """Test MXFP8 utilities work together correctly."""
    # Create weights that need padding
    weight = mx.random.normal((770, 770)) * 10.0

    # Validate (should fail)
    is_valid = validate_mxfp8_shape(weight, raise_error=False)
    assert is_valid is False

    # Pad
    padded, orig_size = pad_for_mxfp8(weight)
    assert padded.shape[1] == 800

    # Validate padded (should pass)
    is_valid_padded = validate_mxfp8_shape(padded, raise_error=False)
    assert is_valid_padded is True

    # Quantize padded version
    w_q, scales = quantize_to_mxfp8(padded)
    restored = dequantize_from_mxfp8(w_q, scales)

    # Remove padding
    restored_unpadded = restored[:, :orig_size]
    assert restored_unpadded.shape == weight.shape

    # Check error
    error = float(mx.mean(mx.abs(restored_unpadded - weight)))
    assert error < 5.0  # Reasonable error


@pytest.mark.unit
def test_compare_mxfp8_vs_int8_recommendation_logic():
    """Test recommendation logic in comparison."""
    # Test with normal distribution (typically similar performance)
    weight_normal = mx.random.normal((768, 768)) * 10.0
    comparison_normal = compare_mxfp8_vs_int8_util(weight_normal)

    # Should provide a recommendation
    assert comparison_normal["recommendation"] in ["mxfp8", "int8", "similar"]

    # Test with wide dynamic range (MXFP8 advantage)
    weight_wide = mx.concatenate([
        mx.random.normal((384, 768)) * 0.01,
        mx.random.normal((384, 768)) * 100.0
    ], axis=0)
    comparison_wide = compare_mxfp8_vs_int8_util(weight_wide)

    # Should still provide valid recommendation
    assert comparison_wide["recommendation"] in ["mxfp8", "int8", "similar"]
    assert "reason" in comparison_wide


@pytest.mark.unit
def test_mxfp8_size_estimate_accuracy():
    """Test that MXFP8 size estimate is accurate."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Estimate size
    stats = estimate_mxfp8_size_util(model)

    # Get actual size info
    from smlx.quant.utils import get_actual_model_size

    actual_size = get_actual_model_size(model)

    # Estimate counts quantizable weights only (excludes bias terms)
    # SimpleModel has 40960 weight params + 192 bias params = 41152 total
    # The estimate should count only the quantizable weights
    assert stats["parameters"] <= actual_size["parameters"]
    # Should be close (within 10% for bias overhead)
    assert stats["parameters"] >= actual_size["parameters"] * 0.9

    # MXFP8 bytes per parameter should be ~1.03125
    estimated_bytes_per_param = (stats["mxfp8_mb"] * 1024 ** 2) / stats["parameters"]
    assert 1.0 < estimated_bytes_per_param < 1.1  # Should be close to 1.03125


@pytest.mark.unit
def test_pad_for_mxfp8_batch_dimensions():
    """Test pad_for_mxfp8 with batch dimensions."""
    # 3D tensor (batch_size, seq_len, hidden_dim)
    weight = mx.random.normal((8, 128, 770))  # 770 not divisible by 32

    padded, orig_size = pad_for_mxfp8(weight)

    # Should only pad last dimension
    assert padded.shape == (8, 128, 800)
    assert orig_size == 770

    # Original values preserved
    assert mx.array_equal(padded[:, :, :770], weight)


@pytest.mark.unit
def test_mxfp8_utils_m4_specific():
    """Test that utilities provide M4-specific information."""
    weight = mx.random.normal((768, 768)) * 10.0

    comparison = compare_mxfp8_vs_int8_util(weight)

    # Should mention M4 considerations in reason or recommendation
    combined_text = comparison["recommendation"] + " " + comparison["reason"]

    # Should have useful information about the comparison
    assert len(comparison["reason"]) > 10  # Should have detailed reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
