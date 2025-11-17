"""
Tests for quantization utilities.

Tests cover:
- Calibration data loading
- Model size estimation
- Quantize-dequantize operations
- M4 compatibility checks
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant.utils import (
    check_m4_compatibility,
    estimate_model_size,
    quantize_dequantize,
)


@pytest.mark.unit
class TestQuantizeDequantize:
    """Tests for quantize_dequantize utility."""

    def test_basic_quantization(self):
        """Test basic quantize-dequantize operation."""
        weight = mx.random.normal((768, 768))

        # Quantize and dequantize
        result = quantize_dequantize(weight, bits=4, group_size=64)

        # Shape should be preserved
        assert result.shape == weight.shape
        # Dtype should be preserved
        assert result.dtype == weight.dtype

    def test_quantization_introduces_error(self):
        """Test that quantization introduces some error."""
        weight = mx.random.normal((768, 768))

        # Quantize and dequantize
        result = quantize_dequantize(weight, bits=4, group_size=64)

        # Should not be exactly equal (quantization error)
        assert not mx.allclose(result, weight, atol=1e-6)

        # But should be reasonably close
        max_diff = mx.max(mx.abs(result - weight))
        assert max_diff < 1.0  # Reasonable error bound

    def test_different_bit_widths(self):
        """Test quantization with different bit widths."""
        weight = mx.random.normal((512, 512))

        # Lower bits = more error
        result_2bit = quantize_dequantize(weight, bits=2, group_size=64)
        result_4bit = quantize_dequantize(weight, bits=4, group_size=64)
        result_8bit = quantize_dequantize(weight, bits=8, group_size=64)

        # Calculate errors
        error_2bit = mx.mean(mx.abs(result_2bit - weight))
        error_4bit = mx.mean(mx.abs(result_4bit - weight))
        error_8bit = mx.mean(mx.abs(result_8bit - weight))

        # More bits = less error
        assert error_2bit > error_4bit
        assert error_4bit > error_8bit

    def test_different_group_sizes(self):
        """Test quantization with different group sizes."""
        weight = mx.random.normal((512, 512))

        # Smaller groups = better precision (more scales/biases)
        result_32 = quantize_dequantize(weight, bits=4, group_size=32)
        result_64 = quantize_dequantize(weight, bits=4, group_size=64)
        result_128 = quantize_dequantize(weight, bits=4, group_size=128)

        # All should have same shape
        assert result_32.shape == weight.shape
        assert result_64.shape == weight.shape
        assert result_128.shape == weight.shape

    def test_preserves_dtype(self):
        """Test that dtype is preserved through quantization."""
        for dtype in [mx.float32, mx.float16]:
            weight = mx.random.normal((256, 256)).astype(dtype)
            result = quantize_dequantize(weight, bits=4, group_size=64)
            assert result.dtype == dtype


@pytest.mark.unit
@pytest.mark.gpu
class TestEstimateModelSize:
    """Tests for estimate_model_size utility."""

    def test_unquantized_model(self):
        """Test size estimation for unquantized model."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(768, 768)
                self.linear2 = nn.Linear(768, 512)

            def __call__(self, x):
                return self.linear2(self.linear1(x))

        model = SimpleModel()
        size_info = estimate_model_size(model, dtype=mx.float16)

        # Should have size estimation
        assert "total_mb" in size_info
        assert "parameters" in size_info
        assert "quantized_mb" in size_info
        assert "unquantized_mb" in size_info

        # All parameters should be unquantized
        assert size_info["quantized_mb"] == 0.0
        assert size_info["unquantized_mb"] > 0.0

        # Total should match unquantized
        assert size_info["total_mb"] == size_info["unquantized_mb"]

        # Parameter count should be correct
        # linear1: 768 * 768 = 589,824
        # linear2: 768 * 512 = 393,216
        # Total: 983,040
        expected_params = 768 * 768 + 768 * 512
        assert size_info["parameters"] == expected_params

    def test_quantized_model(self):
        """Test size estimation for quantized model."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(768, 768)
                self.linear2 = nn.Linear(768, 512)

            def __call__(self, x):
                return self.linear2(self.linear1(x))

        model = SimpleModel()

        # Quantize the model
        model.linear1 = nn.QuantizedLinear.from_linear(  # type: ignore[assignment]
            model.linear1, group_size=64, bits=4
        )
        model.linear2 = nn.QuantizedLinear.from_linear(  # type: ignore[assignment]
            model.linear2, group_size=64, bits=4
        )

        size_info = estimate_model_size(model, dtype=mx.float16)

        # Should have both quantized and no unquantized
        assert size_info["quantized_mb"] > 0.0
        assert size_info["unquantized_mb"] == 0.0

        # Total should match quantized
        assert size_info["total_mb"] == size_info["quantized_mb"]

        # Quantized size should be much smaller than unquantized
        # For 4-bit: approximately 4x smaller (plus overhead for scales/biases)
        expected_params = 768 * 768 + 768 * 512
        assert size_info["parameters"] == expected_params

    def test_mixed_quantization(self):
        """Test size estimation for partially quantized model."""

        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(768, 768)
                self.linear2 = nn.Linear(768, 512)

            def __call__(self, x):
                return self.linear2(self.linear1(x))

        model = MixedModel()

        # Quantize only one layer
        model.linear1 = nn.QuantizedLinear.from_linear(  # type: ignore[assignment]
            model.linear1, group_size=64, bits=4
        )

        size_info = estimate_model_size(model, dtype=mx.float16)

        # Should have both quantized and unquantized
        assert size_info["quantized_mb"] > 0.0
        assert size_info["unquantized_mb"] > 0.0

        # Total should be sum of both
        assert (
            abs(
                size_info["total_mb"]
                - (size_info["quantized_mb"] + size_info["unquantized_mb"])
            )
            < 0.01
        )

    def test_with_embeddings(self):
        """Test size estimation with embedding layers."""

        class ModelWithEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(50000, 768)
                self.linear = nn.Linear(768, 768)

            def __call__(self, x):
                return self.linear(self.embedding(x))

        model = ModelWithEmbedding()
        size_info = estimate_model_size(model, dtype=mx.float16)

        # Should include embedding parameters
        # embedding: 50000 * 768 = 38,400,000
        # linear: 768 * 768 = 589,824
        expected_params = 50000 * 768 + 768 * 768
        assert size_info["parameters"] == expected_params


@pytest.mark.unit
@pytest.mark.gpu
class TestCheckM4Compatibility:
    """Tests for M4 compatibility checking."""

    def test_small_model_compatible(self):
        """Test that small models are compatible with M4."""

        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(256, 256)
                self.linear2 = nn.Linear(256, 128)

            def __call__(self, x):
                return self.linear2(self.linear1(x))

        model = SmallModel()
        compat = check_m4_compatibility(model)

        # Should have all required keys
        assert "is_smol" in compat
        assert "estimated_size_mb" in compat
        assert "total_parameters" in compat
        assert "recommended_quantization" in compat
        assert "fits_in_memory" in compat
        assert "available_memory_mb" in compat

        # Small model should be smol and fit in memory
        assert compat["is_smol"] is True
        assert compat["fits_in_memory"] is True

        # Should recommend no quantization or 8-bit
        assert "No quantization" in compat["recommended_quantization"] or "8-bit" in compat["recommended_quantization"]

    def test_medium_model_recommendations(self):
        """Test recommendations for medium-sized models."""

        class MediumModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Simulate ~5B parameter model with smaller layers for testing
                self.linear1 = nn.Linear(2048, 2048)
                self.linear2 = nn.Linear(2048, 2048)
                self.linear3 = nn.Linear(2048, 2048)

            def __call__(self, x):
                return self.linear3(self.linear2(self.linear1(x)))

        model = MediumModel()
        compat = check_m4_compatibility(model)

        # Should still be smol (<10B)
        assert compat["is_smol"] is True

        # Should recommend some quantization
        assert "4-bit" in compat["recommended_quantization"] or "quantization" in compat["recommended_quantization"].lower()

    def test_parameter_count_threshold(self):
        """Test that parameter count threshold is correctly applied."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = TinyModel()
        compat = check_m4_compatibility(model)

        # Should be well under 10B parameters
        assert compat["total_parameters"] < 10e9
        assert compat["is_smol"] is True

    def test_memory_availability(self):
        """Test that available memory is reported correctly."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = TinyModel()
        compat = check_m4_compatibility(model)

        # Should report 26GB available (36GB - 10GB for system)
        assert compat["available_memory_mb"] == 26 * 1024

    def test_quantized_model_size(self):
        """Test M4 compatibility check on quantized models."""

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(1024, 1024)
                self.linear2 = nn.Linear(1024, 1024)

            def __call__(self, x):
                return self.linear2(self.linear1(x))

        model = TestModel()

        # Check unquantized
        compat_unquant = check_m4_compatibility(model)
        size_unquant = compat_unquant["estimated_size_mb"]

        # Quantize
        model.linear1 = nn.QuantizedLinear.from_linear(  # type: ignore[assignment]
            model.linear1, group_size=64, bits=4
        )
        model.linear2 = nn.QuantizedLinear.from_linear(  # type: ignore[assignment]
            model.linear2, group_size=64, bits=4
        )

        # Check quantized
        compat_quant = check_m4_compatibility(model)
        size_quant = compat_quant["estimated_size_mb"]

        # Quantized should be smaller
        assert size_quant < size_unquant


@pytest.mark.slow
@pytest.mark.integration
class TestLoadCalibrationData:
    """Tests for calibration data loading.

    Note: These tests are slow as they download data and use real tokenizers.
    """

    def test_calibration_data_loading_mock(self):
        """Test calibration data loading with mock tokenizer."""

        class MockTokenizer:
            """Mock tokenizer for testing."""

            def encode(self, text, return_tensors=None):
                # Return mock token IDs (just use character codes)
                tokens = [ord(c) % 256 for c in text[:1000]]
                return mx.array(tokens)

        from smlx.quant.utils import load_calibration_data

        tokenizer = MockTokenizer()

        # Should work with mock tokenizer
        data = load_calibration_data(tokenizer, num_samples=4, sequence_length=32)

        # Should return correct shape
        assert data.shape == (4, 32)
        assert isinstance(data, mx.array)


@pytest.mark.unit
class TestUtilsEdgeCases:
    """Tests for edge cases in utility functions."""

    def test_quantize_dequantize_small_tensor(self):
        """Test quantization on small tensors."""
        # Very small tensor
        weight = mx.random.normal((32, 32))
        result = quantize_dequantize(weight, bits=4, group_size=32)

        assert result.shape == weight.shape

    def test_quantize_dequantize_large_group_size(self):
        """Test quantization with group size larger than tensor."""
        weight = mx.random.normal((64, 64))

        # Group size larger than dimensions should raise ValueError in MLX
        with pytest.raises(ValueError, match="divisible by the quantization group size"):
            quantize_dequantize(weight, bits=4, group_size=128)

    def test_estimate_size_empty_model(self):
        """Test size estimation on minimal model."""

        class EmptyModel(nn.Module):
            def __call__(self, x):
                return x

        model = EmptyModel()
        size_info = estimate_model_size(model)

        # Should return zero sizes
        assert size_info["total_mb"] == 0.0
        assert size_info["parameters"] == 0

    def test_check_m4_minimal_model(self):
        """Test M4 compatibility check on minimal model."""

        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

            def __call__(self, x):
                return self.linear(x)

        model = MinimalModel()
        compat = check_m4_compatibility(model)

        # Should succeed with minimal model
        assert compat["is_smol"] is True
        assert compat["fits_in_memory"] is True
        assert compat["total_parameters"] == 8 * 8
