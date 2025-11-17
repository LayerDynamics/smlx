"""
Tests for MXFP4 (Microscaling FP4) quantization.

This module tests:
- quantize_to_mxfp4 (weight quantization)
- dequantize_from_mxfp4 (weight dequantization)
- quantize_model_mxfp4 (model-level quantization)
- estimate_mxfp4_size (size estimation)
- validate_mxfp_shape (shape validation)
- compare_mxfp4_vs_fp4 (format comparison)
- compare_mxfp4_vs_int4 (format comparison)
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant import (
    compare_mxfp4_vs_fp4,
    compare_mxfp4_vs_int4,
    dequantize_from_mxfp4,
    estimate_mxfp4_size,
    quantize_model_mxfp4,
    quantize_to_mxfp4,
    validate_mxfp_shape,
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
def test_quantize_to_mxfp4_basic():
    """Test basic MXFP4 quantization."""
    weight = mx.random.normal((64, 64))

    w_q, scales = quantize_to_mxfp4(weight)

    # Check shapes
    # MXFP4 returns packed format: 8 elements (4-bit each) per uint32
    expected_packed_shape = weight.shape[:-1] + (weight.shape[-1] // 8,)
    assert w_q.shape == expected_packed_shape
    # Scales: one per 32 elements in last dimension
    # Total elements = 64 * 64 = 4096
    # Blocks = 4096 / 32 = 128
    expected_scale_count = (weight.size + 31) // 32
    assert scales.size == expected_scale_count

    # Check dtypes
    assert w_q.dtype == mx.uint32  # Packed 4-bit values
    assert scales.dtype == mx.uint8  # E8M0 scales


@pytest.mark.unit
def test_mxfp4_quantize_dequantize():
    """Test MXFP4 quantization round-trip."""
    weight = mx.random.normal((32, 32)) * 2.0

    w_q, scales = quantize_to_mxfp4(weight)
    restored = dequantize_from_mxfp4(w_q, scales)

    # Check shape preserved
    assert restored.shape == weight.shape

    # MXFP4 is lossy but should be reasonably close
    error = mx.mean(mx.abs(weight - restored))
    assert float(error) < 0.5  # Reasonable bound for MXFP4


@pytest.mark.unit
def test_mxfp4_dtype_conversion():
    """Test dequantization to different dtypes."""
    weight = mx.random.normal((64, 64))
    w_q, scales = quantize_to_mxfp4(weight)

    # Dequantize to float16
    restored_fp16 = dequantize_from_mxfp4(w_q, scales, dtype=mx.float16)
    assert restored_fp16.dtype == mx.float16

    # Dequantize to bfloat16
    restored_bf16 = dequantize_from_mxfp4(w_q, scales, dtype=mx.bfloat16)
    assert restored_bf16.dtype == mx.bfloat16

    # Dequantize to float32
    restored_fp32 = dequantize_from_mxfp4(w_q, scales, dtype=mx.float32)
    assert restored_fp32.dtype == mx.float32


@pytest.mark.unit
def test_quantize_model_mxfp4():
    """Test model-level MXFP4 quantization."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Quantize model
    result = quantize_model_mxfp4(model, inplace=True)

    # Should return None for in-place
    assert result is None

    # Check that linear layers are now quantized
    assert isinstance(model.fc1, nn.QuantizedLinear)
    assert isinstance(model.fc2, nn.QuantizedLinear)


@pytest.mark.unit
def test_quantize_model_mxfp4_not_inplace():
    """Test model quantization without in-place modification."""
    model = SimpleModel()
    mx.eval(model.parameters())

    # Quantize not in-place (returns model)
    result = quantize_model_mxfp4(model, inplace=False)

    # Should return the model
    assert result is not None

    # Model should still be quantized
    assert isinstance(model.fc1, nn.QuantizedLinear)


@pytest.mark.unit
def test_estimate_mxfp4_size():
    """Test MXFP4 size estimation."""
    model = SimpleModel()
    mx.eval(model.parameters())

    stats = estimate_mxfp4_size(model)

    # Check all expected keys
    assert "current_mb" in stats
    assert "mxfp4_mb" in stats
    assert "reduction_ratio" in stats
    assert "saved_mb" in stats
    assert "total_params" in stats
    assert "quantizable_params" in stats
    assert "scale_overhead_mb" in stats
    assert "format" in stats
    assert "block_size" in stats

    # MXFP4 should be significantly smaller
    assert stats["mxfp4_mb"] < stats["current_mb"]
    assert stats["reduction_ratio"] > 1.0

    # Block size should be 32
    assert stats["block_size"] == 32

    # Format should mention MXFP4
    assert "MXFP4" in stats["format"]


@pytest.mark.unit
def test_mxfp4_handles_zeros():
    """Test MXFP4 quantization of zero values."""
    weight = mx.zeros((64, 64))

    w_q, scales = quantize_to_mxfp4(weight)
    restored = dequantize_from_mxfp4(w_q, scales)

    # Zeros should remain zeros (or very close)
    assert float(mx.max(mx.abs(restored))) < 0.1


@pytest.mark.unit
def test_mxfp4_large_values():
    """Test MXFP4 with large values."""
    weight = mx.random.normal((64, 64)) * 100.0

    w_q, scales = quantize_to_mxfp4(weight)
    restored = dequantize_from_mxfp4(w_q, scales)

    # Should handle large values reasonably
    assert restored.shape == weight.shape
    assert mx.all(mx.isfinite(restored))


@pytest.mark.unit
def test_mxfp4_preserves_shape():
    """Test that MXFP4 preserves shape."""
    for shape in [(32, 64), (64, 128), (128, 256)]:
        weight = mx.random.normal(shape)
        w_q, scales = quantize_to_mxfp4(weight)
        restored = dequantize_from_mxfp4(w_q, scales)

        assert restored.shape == weight.shape


@pytest.mark.unit
def test_compare_mxfp4_vs_fp4():
    """Test MXFP4 vs FP4 comparison."""
    weight = mx.random.normal((64, 64))

    comparison = compare_mxfp4_vs_fp4(weight)

    # Check all expected keys
    assert "mxfp4_error" in comparison
    assert "fp4_error" in comparison
    assert "mxfp4_max_error" in comparison
    assert "fp4_max_error" in comparison
    assert "mxfp4_better" in comparison
    assert "size_comparison" in comparison
    assert "recommendation" in comparison

    # Errors should be reasonable
    assert comparison["mxfp4_error"] < 1.0
    assert comparison["fp4_error"] < 1.0

    # Size comparison should exist
    assert "mxfp4_bytes" in comparison["size_comparison"]
    assert "fp4_bytes" in comparison["size_comparison"]


@pytest.mark.unit
def test_compare_mxfp4_vs_int4():
    """Test MXFP4 vs INT4 comparison."""
    weight = mx.random.normal((64, 64))

    comparison = compare_mxfp4_vs_int4(weight)

    # Check all expected keys
    assert "mxfp4_error" in comparison
    assert "int4_error" in comparison
    assert "mxfp4_max_error" in comparison
    assert "int4_max_error" in comparison
    assert "mxfp4_better" in comparison
    assert "recommendation" in comparison

    # Errors should be reasonable
    assert comparison["mxfp4_error"] < 1.0
    assert comparison["int4_error"] < 1.0


@pytest.mark.unit
def test_mxfp4_with_padding_comparison():
    """Test MXFP4 with non-divisible dimensions in comparison."""
    weight = mx.random.normal((64, 100))  # 100 % 32 != 0

    # Should auto-pad for comparison
    comparison = compare_mxfp4_vs_fp4(weight)

    # Should not raise error
    assert "mxfp4_error" in comparison
    assert comparison["mxfp4_error"] >= 0


@pytest.mark.unit
def test_mxfp4_block_size_32():
    """Test that MXFP4 uses block size 32."""
    weight = mx.random.normal((64, 64))  # 4096 elements

    w_q, scales = quantize_to_mxfp4(weight)

    # Number of scales should be elements / 32
    expected_scales = weight.size // 32
    assert scales.size == expected_scales


@pytest.mark.unit
def test_mxfp4_comparison_uniform_distribution():
    """Test MXFP4 comparison with uniform distribution weights."""
    weight = mx.random.uniform(-5, 5, (64, 64))

    comparison = compare_mxfp4_vs_int4(weight)

    # Both should handle uniform distribution
    assert comparison["mxfp4_error"] < 1.0
    assert comparison["int4_error"] < 1.0


@pytest.mark.unit
def test_mxfp4_comparison_wide_range():
    """Test MXFP4 comparison with wide dynamic range."""
    # Create weights with wide range
    weight = mx.concatenate(
        [mx.random.normal((32, 64)) * 0.1, mx.random.normal((32, 64)) * 10.0], axis=0
    )

    comparison = compare_mxfp4_vs_int4(weight)

    # MXFP4 should handle wide range well (floating point advantage)
    assert "mxfp4_error" in comparison


@pytest.mark.unit
def test_estimate_mxfp4_size_reduction():
    """Test that MXFP4 provides expected compression."""
    model = SimpleModel()
    mx.eval(model.parameters())

    stats = estimate_mxfp4_size(model)

    # MXFP4 should provide ~1.5-2x reduction from FP32
    # (0.5 bytes per weight element + scale overhead)
    # Note: Models are FP32 by default in MLX
    assert 1.4 < stats["reduction_ratio"] < 2.0


@pytest.mark.unit
def test_mxfp4_scale_overhead():
    """Test MXFP4 scale overhead calculation."""
    model = SimpleModel()
    mx.eval(model.parameters())

    stats = estimate_mxfp4_size(model)

    # Scale overhead should be reasonable (1 byte per 32 elements)
    # For a typical model, should be < 5% of total
    scale_percentage = (stats["scale_overhead_mb"] / stats["mxfp4_mb"]) * 100
    assert scale_percentage < 10.0  # Should be small overhead


@pytest.mark.unit
def test_mxfp4_empty_model():
    """Test MXFP4 size estimation with minimal model."""
    model = nn.Module()

    stats = estimate_mxfp4_size(model)

    # Should handle empty model gracefully
    assert stats["current_mb"] == 0
    assert stats["mxfp4_mb"] == 0


@pytest.mark.unit
def test_mxfp4_quantize_different_shapes():
    """Test MXFP4 with various valid shapes."""
    shapes = [(32, 32), (64, 64), (128, 64), (256, 128)]

    for shape in shapes:
        weight = mx.random.normal(shape)
        w_q, scales = quantize_to_mxfp4(weight)
        restored = dequantize_from_mxfp4(w_q, scales)

        assert restored.shape == weight.shape
        error = mx.mean(mx.abs(weight - restored))
        assert float(error) < 1.0


@pytest.mark.unit
def test_mxfp4_idempotent():
    """Test that repeated quantization produces same results."""
    weight = mx.random.normal((64, 64))

    # Quantize twice
    w_q1, scales1 = quantize_to_mxfp4(weight)
    w_q2, scales2 = quantize_to_mxfp4(weight)

    # Should produce identical results
    assert mx.array_equal(w_q1, w_q2)
    assert mx.array_equal(scales1, scales2)


@pytest.mark.unit
def test_mxfp4_comparison_recommendation():
    """Test that comparisons provide recommendations."""
    weight = mx.random.normal((64, 64))

    comparison_fp4 = compare_mxfp4_vs_fp4(weight)
    comparison_int4 = compare_mxfp4_vs_int4(weight)

    # Both should have recommendations
    assert isinstance(comparison_fp4["recommendation"], str)
    assert len(comparison_fp4["recommendation"]) > 0

    assert isinstance(comparison_int4["recommendation"], str)
    assert len(comparison_int4["recommendation"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
