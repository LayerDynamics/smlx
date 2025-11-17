"""
Tests for bit-width specific quantization wrappers.

Tests cover:
- 4-bit quantization (4bit.py)
- 6-bit quantization (6bit.py)
- 8-bit quantization (8bit.py)
- BFloat16 conversion (bf16.py)
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

# Import using importlib since module names start with numbers
import importlib

bit4_module = importlib.import_module("smlx.quant.4bit")
bit6_module = importlib.import_module("smlx.quant.6bit")
bit8_module = importlib.import_module("smlx.quant.8bit")
bf16_module = importlib.import_module("smlx.quant.bf16")

# Import from 4bit module
quantize_4bit = bit4_module.quantize_4bit
quantize_weights_4bit = bit4_module.quantize_weights_4bit
dequantize_weights_4bit = bit4_module.dequantize_weights_4bit
is_4bit_quantized = bit4_module.is_4bit_quantized
estimate_4bit_size_reduction = bit4_module.estimate_4bit_size_reduction
get_quantization_info = bit4_module.get_quantization_info

# Import from 6bit module
quantize_6bit = bit6_module.quantize_6bit
quantize_weights_6bit = bit6_module.quantize_weights_6bit
dequantize_weights_6bit = bit6_module.dequantize_weights_6bit
is_6bit_quantized = bit6_module.is_6bit_quantized
estimate_6bit_size_reduction = bit6_module.estimate_6bit_size_reduction

# Import from 8bit module
quantize_8bit = bit8_module.quantize_8bit
quantize_weights_8bit = bit8_module.quantize_weights_8bit
dequantize_weights_8bit = bit8_module.dequantize_weights_8bit
is_8bit_quantized = bit8_module.is_8bit_quantized
estimate_8bit_size_reduction = bit8_module.estimate_8bit_size_reduction
compare_with_4bit = bit8_module.compare_with_4bit

# Import from bf16 module
convert_to_bfloat16 = bf16_module.convert_to_bfloat16
weights_to_bfloat16 = bf16_module.weights_to_bfloat16
weights_from_bfloat16 = bf16_module.weights_from_bfloat16
is_bfloat16 = bf16_module.is_bfloat16
estimate_bfloat16_size = bf16_module.estimate_bfloat16_size
mixed_precision_bf16_fp32 = bf16_module.mixed_precision_bf16_fp32
compare_dtypes = bf16_module.compare_dtypes


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.embed = nn.Embedding(1000, input_dim)

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        return x


@pytest.mark.unit
class Test4BitQuantization:
    """Tests for 4-bit quantization."""

    def test_quantize_4bit_inplace(self):
        """Test in-place 4-bit quantization."""
        model = SimpleModel()
        result = quantize_4bit(model, inplace=True)

        # Should return None for inplace
        assert result is None

        # Check that linear layers are quantized
        assert is_4bit_quantized(model.fc1)
        assert is_4bit_quantized(model.fc2)

    def test_quantize_4bit_not_inplace(self):
        """Test non-inplace 4-bit quantization."""
        model = SimpleModel()
        result = quantize_4bit(model, inplace=False)

        # Should return model
        assert result is not None
        assert is_4bit_quantized(result.fc1)

    def test_quantize_weights_4bit(self):
        """Test 4-bit weight quantization."""
        weight = mx.random.normal((768, 768))
        w_q, scales, biases = quantize_weights_4bit(weight)

        # Check shapes
        assert w_q.ndim == 2
        assert scales.ndim == 2
        assert biases.ndim == 2

        # Packed weights should be smaller
        assert w_q.size < weight.size

    def test_dequantize_weights_4bit(self):
        """Test 4-bit weight dequantization."""
        weight = mx.random.normal((768, 768))
        w_q, scales, biases = quantize_weights_4bit(weight)

        # Dequantize
        weight_restored = dequantize_weights_4bit(w_q, scales, biases)

        # Check shape and dtype
        assert weight_restored.shape == weight.shape
        assert weight_restored.dtype == weight.dtype

        # Should be close but not exact
        assert not mx.allclose(weight_restored, weight, atol=1e-6)
        max_diff = mx.max(mx.abs(weight_restored - weight))
        assert max_diff < 1.0

    def test_is_4bit_quantized(self):
        """Test 4-bit quantization detection."""
        model = SimpleModel()
        assert not is_4bit_quantized(model.fc1)

        quantize_4bit(model, inplace=True)
        assert is_4bit_quantized(model.fc1)

    def test_get_quantization_info(self):
        """Test getting quantization information."""
        model = SimpleModel()
        quantize_4bit(model, inplace=True)

        info = get_quantization_info(model.fc1)
        assert info is not None
        assert info["bits"] == 4
        assert info["group_size"] == 64
        assert info["is_4bit"] is True
        assert "weight_shape" in info
        assert "compression_ratio" in info

    def test_estimate_4bit_size_reduction(self):
        """Test size reduction estimation."""
        model = SimpleModel()
        stats = estimate_4bit_size_reduction(model)

        assert "current_mb" in stats
        assert "quantized_mb" in stats
        assert "reduction_ratio" in stats
        assert "saved_mb" in stats

        # Should show significant reduction (4-bit gives ~8x compression theoretically)
        # But in practice with scales and biases, expect ~5-6x
        assert stats["reduction_ratio"] > 1.5

    def test_different_group_sizes(self):
        """Test 4-bit quantization with different group sizes."""
        weight = mx.random.normal((512, 512))

        for group_size in [32, 64, 128]:
            w_q, scales, biases = quantize_weights_4bit(weight, group_size=group_size)
            weight_restored = dequantize_weights_4bit(
                w_q, scales, biases, group_size=group_size
            )
            assert weight_restored.shape == weight.shape


@pytest.mark.unit
class Test6BitQuantization:
    """Tests for 6-bit quantization."""

    def test_quantize_6bit_inplace(self):
        """Test in-place 6-bit quantization."""
        model = SimpleModel()
        result = quantize_6bit(model, inplace=True)

        assert result is None
        assert is_6bit_quantized(model.fc1)
        assert is_6bit_quantized(model.fc2)

    def test_quantize_weights_6bit(self):
        """Test 6-bit weight quantization."""
        weight = mx.random.normal((512, 512))
        w_q, scales, biases = quantize_weights_6bit(weight)

        # Check that output exists
        assert w_q is not None
        assert scales is not None
        assert biases is not None

    def test_dequantize_weights_6bit(self):
        """Test 6-bit weight dequantization."""
        weight = mx.random.normal((512, 512))
        w_q, scales, biases = quantize_weights_6bit(weight)
        weight_restored = dequantize_weights_6bit(w_q, scales, biases)

        assert weight_restored.shape == weight.shape
        assert weight_restored.dtype == weight.dtype

    def test_is_6bit_quantized(self):
        """Test 6-bit quantization detection."""
        model = SimpleModel()
        assert not is_6bit_quantized(model.fc1)

        quantize_6bit(model, inplace=True)
        assert is_6bit_quantized(model.fc1)

    def test_estimate_6bit_size_reduction(self):
        """Test 6-bit size reduction estimation."""
        model = SimpleModel()
        stats = estimate_6bit_size_reduction(model)

        assert "current_mb" in stats
        assert "quantized_mb" in stats
        assert "reduction_ratio" in stats

        # 6-bit should give ~1.3-2x reduction in practice with overhead
        assert 1.2 < stats["reduction_ratio"] < 3.0

    def test_6bit_better_quality_than_4bit(self):
        """Test that 6-bit has less error than 4-bit."""
        weight = mx.random.normal((512, 512))

        # 4-bit quantization
        w_q_4, scales_4, biases_4 = quantize_weights_4bit(weight)
        restored_4 = dequantize_weights_4bit(w_q_4, scales_4, biases_4)
        error_4 = mx.mean(mx.abs(restored_4 - weight))

        # 6-bit quantization
        w_q_6, scales_6, biases_6 = quantize_weights_6bit(weight)
        restored_6 = dequantize_weights_6bit(w_q_6, scales_6, biases_6)
        error_6 = mx.mean(mx.abs(restored_6 - weight))

        # 6-bit should have less error
        assert error_6 < error_4


@pytest.mark.unit
class Test8BitQuantization:
    """Tests for 8-bit quantization."""

    def test_quantize_8bit_inplace(self):
        """Test in-place 8-bit quantization."""
        model = SimpleModel()
        result = quantize_8bit(model, inplace=True)

        assert result is None
        assert is_8bit_quantized(model.fc1)
        assert is_8bit_quantized(model.fc2)

    def test_quantize_weights_8bit(self):
        """Test 8-bit weight quantization."""
        weight = mx.random.normal((256, 256))
        w_q, scales, biases = quantize_weights_8bit(weight)

        assert w_q is not None
        assert scales is not None
        assert biases is not None

    def test_dequantize_weights_8bit(self):
        """Test 8-bit weight dequantization."""
        weight = mx.random.normal((256, 256))
        w_q, scales, biases = quantize_weights_8bit(weight)
        weight_restored = dequantize_weights_8bit(w_q, scales, biases)

        assert weight_restored.shape == weight.shape
        assert weight_restored.dtype == weight.dtype

    def test_is_8bit_quantized(self):
        """Test 8-bit quantization detection."""
        model = SimpleModel()
        assert not is_8bit_quantized(model.fc1)

        quantize_8bit(model, inplace=True)
        assert is_8bit_quantized(model.fc1)

    def test_estimate_8bit_size_reduction(self):
        """Test 8-bit size reduction estimation."""
        model = SimpleModel()
        stats = estimate_8bit_size_reduction(model)

        assert "current_mb" in stats
        assert "quantized_mb" in stats
        assert "reduction_ratio" in stats

        # 8-bit should give ~2x reduction (in practice ~1.3-1.8x with overhead)
        assert 1.2 < stats["reduction_ratio"] < 2.5

    def test_8bit_minimal_error(self):
        """Test that 8-bit has minimal quantization error."""
        weight = mx.random.normal((512, 512))

        w_q, scales, biases = quantize_weights_8bit(weight)
        restored = dequantize_weights_8bit(w_q, scales, biases)

        # Error should be very small
        error = mx.mean(mx.abs(restored - weight))
        assert error < 0.01

    def test_compare_with_4bit(self):
        """Test comparison function between 8-bit and 4-bit."""
        model = SimpleModel()
        comparison = compare_with_4bit(model)

        assert "size_8bit_mb" in comparison
        assert "size_4bit_mb" in comparison
        assert "additional_size_mb" in comparison
        assert "quality_tradeoff" in comparison

        # 8-bit should be larger than 4-bit
        assert comparison["size_8bit_mb"] > comparison["size_4bit_mb"]

    def test_quality_progression(self):
        """Test that quality improves with more bits."""
        weight = mx.random.normal((512, 512))

        # Test 4-bit, 6-bit, 8-bit
        w_q_4, scales_4, biases_4 = quantize_weights_4bit(weight)
        restored_4 = dequantize_weights_4bit(w_q_4, scales_4, biases_4)
        error_4 = mx.mean(mx.abs(restored_4 - weight))

        w_q_6, scales_6, biases_6 = quantize_weights_6bit(weight)
        restored_6 = dequantize_weights_6bit(w_q_6, scales_6, biases_6)
        error_6 = mx.mean(mx.abs(restored_6 - weight))

        w_q_8, scales_8, biases_8 = quantize_weights_8bit(weight)
        restored_8 = dequantize_weights_8bit(w_q_8, scales_8, biases_8)
        error_8 = mx.mean(mx.abs(restored_8 - weight))

        # More bits = less error
        assert error_8 < error_6 < error_4


@pytest.mark.unit
class TestBFloat16:
    """Tests for BFloat16 conversion."""

    def test_convert_to_bfloat16_inplace(self):
        """Test in-place BFloat16 conversion."""
        model = SimpleModel()
        result = convert_to_bfloat16(model, inplace=True)

        assert result is None
        assert is_bfloat16(model.fc1)
        assert is_bfloat16(model.fc2)

    def test_weights_to_bfloat16(self):
        """Test weight array conversion to BFloat16."""
        weight = mx.random.normal((512, 512), dtype=mx.float32)
        weight_bf16 = weights_to_bfloat16(weight)

        assert weight_bf16.dtype == mx.bfloat16
        assert weight_bf16.shape == weight.shape

    def test_weights_from_bfloat16(self):
        """Test weight conversion from BFloat16."""
        weight = mx.random.normal((512, 512), dtype=mx.float32)
        weight_bf16 = weights_to_bfloat16(weight)
        weight_restored = weights_from_bfloat16(weight_bf16, dtype=mx.float32)

        assert weight_restored.dtype == mx.float32
        assert weight_restored.shape == weight.shape

    def test_is_bfloat16(self):
        """Test BFloat16 detection."""
        linear = nn.Linear(512, 256)
        assert not is_bfloat16(linear)

        convert_to_bfloat16(linear)
        assert is_bfloat16(linear)

    def test_estimate_bfloat16_size(self):
        """Test BFloat16 size estimation."""
        model = SimpleModel()
        stats = estimate_bfloat16_size(model)

        assert "current_mb" in stats
        assert "bfloat16_mb" in stats
        assert "reduction_ratio" in stats
        assert "current_dtype" in stats

    def test_mixed_precision_bf16_fp32(self):
        """Test mixed precision with BF16 and FP32."""
        model = SimpleModel()

        # Convert most to BF16, keep fc1 in FP32
        mixed_precision_bf16_fp32(model, fp32_layers=["fc1"])

        # fc1 should be FP32
        assert model.fc1.weight.dtype == mx.float32
        # fc2 should be BF16
        assert model.fc2.weight.dtype == mx.bfloat16

    def test_compare_dtypes(self):
        """Test dtype comparison function."""
        model = SimpleModel()
        comparison = compare_dtypes(model)

        assert "fp32_mb" in comparison
        assert "fp16_mb" in comparison
        assert "bfloat16_mb" in comparison
        assert "recommendations" in comparison
        assert "tradeoffs" in comparison

        # FP32 should be largest
        assert comparison["fp32_mb"] > comparison["fp16_mb"]
        assert comparison["fp32_mb"] > comparison["bfloat16_mb"]

    def test_bfloat16_precision_loss(self):
        """Test that BFloat16 introduces precision loss."""
        weight = mx.random.normal((256, 256), dtype=mx.float32)
        weight_bf16 = weights_to_bfloat16(weight)
        weight_restored = weights_from_bfloat16(weight_bf16)

        # Should not be exactly equal
        assert not mx.allclose(weight, weight_restored, atol=1e-6)

        # But should be reasonably close
        error = mx.mean(mx.abs(weight - weight_restored))
        assert error < 0.01


@pytest.mark.unit
class TestCrossFormat:
    """Tests comparing different quantization formats."""

    def test_size_comparison(self):
        """Test that size reductions follow expected order."""
        model = SimpleModel()

        stats_4bit = estimate_4bit_size_reduction(model)
        stats_6bit = estimate_6bit_size_reduction(model)
        stats_8bit = estimate_8bit_size_reduction(model)

        # 4-bit should be smallest
        assert stats_4bit["quantized_mb"] < stats_6bit["quantized_mb"]
        assert stats_6bit["quantized_mb"] < stats_8bit["quantized_mb"]

        # Reduction ratios should decrease with more bits
        assert stats_4bit["reduction_ratio"] > stats_6bit["reduction_ratio"]
        assert stats_6bit["reduction_ratio"] > stats_8bit["reduction_ratio"]

    def test_error_comparison(self):
        """Test that quantization error decreases with more bits."""
        weight = mx.random.normal((512, 512))

        # Quantize with different bit widths
        w_q_4, scales_4, biases_4 = quantize_weights_4bit(weight)
        restored_4 = dequantize_weights_4bit(w_q_4, scales_4, biases_4)

        w_q_6, scales_6, biases_6 = quantize_weights_6bit(weight)
        restored_6 = dequantize_weights_6bit(w_q_6, scales_6, biases_6)

        w_q_8, scales_8, biases_8 = quantize_weights_8bit(weight)
        restored_8 = dequantize_weights_8bit(w_q_8, scales_8, biases_8)

        # Calculate mean absolute errors
        error_4 = mx.mean(mx.abs(restored_4 - weight)).item()
        error_6 = mx.mean(mx.abs(restored_6 - weight)).item()
        error_8 = mx.mean(mx.abs(restored_8 - weight)).item()

        # Errors should decrease with more bits
        assert error_8 < error_6 < error_4

    def test_model_forward_pass_quantized(self):
        """Test that quantized models can still perform forward pass."""
        model = SimpleModel()
        x = mx.random.normal((4, 768))

        # Get baseline output
        output_original = model(x)

        # Test with different quantizations
        for quantize_fn in [quantize_4bit, quantize_6bit, quantize_8bit]:
            model_copy = SimpleModel()
            quantize_fn(model_copy)
            output_quantized = model_copy(x)

            # Shape should be preserved
            assert output_quantized.shape == output_original.shape

    def test_bfloat16_vs_quantization(self):
        """Test BFloat16 vs integer quantization."""
        model = SimpleModel()

        # BFloat16 conversion
        stats_bf16 = estimate_bfloat16_size(model)

        # 8-bit quantization
        stats_8bit = estimate_8bit_size_reduction(model)

        # Both should give ~1.3-2x reduction with overhead
        # Exact ratio depends on implementation details
        assert stats_bf16["reduction_ratio"] > 1.0
        assert stats_8bit["reduction_ratio"] > 1.0
        # Ratios should be in same ballpark (within 2x of each other)
        assert 0.5 < (stats_8bit["reduction_ratio"] / stats_bf16["reduction_ratio"]) < 2.0


@pytest.mark.gpu
class TestQuantizationGPU:
    """GPU-specific tests for quantization."""

    def test_quantized_inference_on_gpu(self):
        """Test that quantized models work on GPU."""
        model = SimpleModel()
        quantize_4bit(model)

        x = mx.random.normal((8, 768))
        output = model(x)

        assert output is not None
        assert output.shape == (8, 256)

    def test_bfloat16_inference_on_gpu(self):
        """Test that BFloat16 models work on GPU."""
        model = SimpleModel()
        convert_to_bfloat16(model)

        x = mx.random.normal((8, 768))
        output = model(x)

        assert output is not None
        assert output.shape == (8, 256)
