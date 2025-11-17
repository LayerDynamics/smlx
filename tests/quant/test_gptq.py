"""
Tests for GPTQ (GPT Quantization) implementation.

Tests cover:
- Catcher class for Hessian computation
- Weight quantization and packing
- Full GPTQ quantization pipeline
- Selective quantization (only Linear layers)
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant.gptq import Catcher, _quantize_weights, gptq_quantize


@pytest.mark.unit
@pytest.mark.gpu
class TestCatcher:
    """Tests for the Catcher module that computes Hessians."""

    def test_catcher_wraps_module(self):
        """Test that Catcher properly wraps a module."""
        linear = nn.Linear(64, 64)
        catcher = Catcher(linear)

        assert catcher.module is linear
        assert isinstance(catcher.H, mx.array)

    def test_catcher_accumulates_hessian(self):
        """Test that Catcher accumulates Hessian matrix."""
        linear = nn.Linear(64, 64)
        catcher = Catcher(linear)

        # Initial Hessian should be zero
        assert catcher.H.item() == 0.0

        # Forward pass should accumulate H = X^T @ X
        x = mx.random.normal((4, 64))
        output = catcher(x)

        # H should now be non-zero
        assert catcher.H.shape == (64, 64)
        assert not mx.allclose(catcher.H, mx.zeros_like(catcher.H))

        # Output should match wrapped module
        expected = linear(x)
        assert mx.allclose(output, expected, atol=1e-5)

    def test_catcher_accumulates_across_batches(self):
        """Test that Hessian accumulates across multiple forward passes."""
        linear = nn.Linear(32, 32)
        catcher = Catcher(linear)

        # First batch
        x1 = mx.random.normal((2, 32))
        catcher(x1)
        H1 = mx.array(catcher.H)

        # Second batch
        x2 = mx.random.normal((2, 32))
        catcher(x2)
        H2 = catcher.H

        # H2 should be greater than H1 (accumulated)
        assert mx.sum(mx.abs(H2)) > mx.sum(mx.abs(H1))


@pytest.mark.unit
@pytest.mark.gpu
class TestQuantizeWeights:
    """Tests for weight quantization and packing."""

    def test_quantize_4bit(self):
        """Test 4-bit weight quantization."""
        # Create test weights and scales/biases
        w = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        scales = mx.array([[0.5]])
        biases = mx.array([[0.0]])

        # Quantize to 4-bit (8 values per uint32)
        packed = _quantize_weights(w, bits=4, scales=scales, biases=biases)

        # Should pack 8 values into 1 uint32
        assert packed.shape == (1, 1)
        assert packed.dtype == mx.uint32

    def test_quantize_2bit(self):
        """Test 2-bit weight quantization."""
        w = mx.array([[0.0, 1.0, 2.0, 3.0] * 4])  # 16 values
        scales = mx.array([[1.0]])
        biases = mx.array([[0.0]])

        # Quantize to 2-bit (16 values per uint32)
        packed = _quantize_weights(w, bits=2, scales=scales, biases=biases)

        # Should pack 16 values into 1 uint32
        assert packed.shape == (1, 1)
        assert packed.dtype == mx.uint32

    def test_quantize_8bit(self):
        """Test 8-bit weight quantization."""
        w = mx.array([[1.0, 2.0, 3.0, 4.0]])
        scales = mx.array([[0.01]])
        biases = mx.array([[0.0]])

        # Quantize to 8-bit (4 values per uint32)
        packed = _quantize_weights(w, bits=8, scales=scales, biases=biases)

        # Should pack 4 values into 1 uint32
        assert packed.shape == (1, 1)
        assert packed.dtype == mx.uint32


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
class TestGPTQQuantize:
    """Integration tests for full GPTQ quantization."""

    def create_test_model(self):
        """Create a small test model with Linear layers."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 64)
                self.linear2 = nn.Linear(64, 64)
                self.embedding = nn.Embedding(100, 64)

            def __call__(self, x):
                # x is token indices
                embedded = self.embedding(x)
                h = self.linear1(embedded)
                h = nn.relu(h)
                h = self.linear2(h)
                return h

        return SimpleModel()

    def test_gptq_quantize_basic(self):
        """Test basic GPTQ quantization."""
        model = self.create_test_model()

        # Create small calibration data (token indices)
        calibration_data = mx.random.randint(0, 100, (16, 8))

        # Quantize with GPTQ
        quantized_model = gptq_quantize(
            model,
            calibration_data,
            bits=4,
            group_size=32,  # Minimum supported group size
            batch_size=4
        )

        # Model should still be callable
        test_input = mx.random.randint(0, 100, (2, 8))
        output = quantized_model(test_input)
        assert output.shape == (2, 8, 64)

    def test_gptq_quantizes_linear_layers(self):
        """Test that GPTQ quantizes Linear layers."""
        model = self.create_test_model()
        calibration_data = mx.random.randint(0, 100, (8, 8))

        # Before quantization - layers should be regular Linear
        assert isinstance(model.linear1, nn.Linear)
        assert isinstance(model.linear2, nn.Linear)

        # Quantize
        quantized_model = gptq_quantize(
            model,
            calibration_data,
            bits=4,
            group_size=32,
            batch_size=4
        )

        # After quantization - Linear layers should be QuantizedLinear
        assert isinstance(quantized_model.linear1, nn.QuantizedLinear)
        assert isinstance(quantized_model.linear2, nn.QuantizedLinear)

        # Check quantization parameters
        assert quantized_model.linear1.bits == 4
        assert quantized_model.linear1.group_size == 32

    def test_gptq_leaves_non_linear_unquantized(self):
        """Test that non-Linear layers remain unquantized."""
        model = self.create_test_model()
        calibration_data = mx.random.randint(0, 100, (8, 8))

        # Quantize with GPTQ (only Linear layers)
        quantized_model = gptq_quantize(
            model,
            calibration_data,
            bits=4,
            group_size=32,
            batch_size=4
        )

        # Linear layers should be quantized
        assert isinstance(quantized_model.linear1, nn.QuantizedLinear)
        assert isinstance(quantized_model.linear2, nn.QuantizedLinear)

        # Embedding should remain unquantized (GPTQ only handles Linear layers)
        assert isinstance(quantized_model.embedding, nn.Embedding)
        assert not isinstance(quantized_model.embedding, nn.QuantizedEmbedding)

    def test_gptq_output_close_to_original(self):
        """Test that GPTQ quantized model produces similar outputs."""
        model = self.create_test_model()
        calibration_data = mx.random.randint(0, 100, (16, 8))

        # Get output from original model
        test_input = mx.random.randint(0, 100, (4, 8))
        original_output = model(test_input)

        # Quantize
        quantized_model = gptq_quantize(
            model,
            calibration_data,
            bits=4,
            group_size=32,
            batch_size=4
        )

        # Get output from quantized model
        quantized_output = quantized_model(test_input)

        # Outputs should be reasonably close (allowing for quantization error)
        # Using loose tolerance since 4-bit quantization loses precision
        assert quantized_output.shape == original_output.shape
        # Check that outputs are in similar range
        orig_mean = mx.mean(mx.abs(original_output))
        quant_mean = mx.mean(mx.abs(quantized_output))
        assert mx.abs(orig_mean - quant_mean) / orig_mean < 0.5  # Within 50%

    def test_gptq_with_different_bit_widths(self):
        """Test GPTQ with different bit widths."""
        calibration_data = mx.random.randint(0, 100, (8, 8))

        for bits in [2, 4, 8]:
            # Create fresh model for each bit width
            model = self.create_test_model()

            quantized_model = gptq_quantize(
                model,
                calibration_data,
                bits=bits,
                group_size=32,
                batch_size=4
            )

            # Check quantization applied
            assert isinstance(quantized_model.linear1, nn.QuantizedLinear)
            assert quantized_model.linear1.bits == bits

            # Model should still work
            test_input = mx.random.randint(0, 100, (2, 8))
            output = quantized_model(test_input)
            assert output.shape == (2, 8, 64)


@pytest.mark.unit
@pytest.mark.gpu
class TestGPTQM4Optimization:
    """Tests for M4-specific optimizations."""

    def test_default_parameters_optimized_for_m4(self):
        """Test that default parameters are optimized for M4."""
        # Create model that's large enough for default group_size=64
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)

            def __call__(self, x):
                return self.linear(x)

        model = TinyModel()
        # GPTQ expects the model to handle calibration_data, which in this case is float input
        calibration_data = mx.random.normal((8, 128))

        # Use default parameters (should be M4-optimized: 4-bit, group_size=64)
        quantized_model = gptq_quantize(
            model,
            calibration_data,
            # bits=4 (default)
            # group_size=64 (default)
            batch_size=4
        )

        # Should use 4-bit quantization with group_size=64 by default
        assert isinstance(quantized_model.linear, nn.QuantizedLinear)
        assert quantized_model.linear.bits == 4
        assert quantized_model.linear.group_size == 64
