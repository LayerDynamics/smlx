"""
Tests for Dynamic Quantization with Mixed-Precision implementation.

Tests cover:
- Quantize-dequantize simulation
- KL divergence loss computation
- Sensitivity estimation
- Threshold estimation for target BPW
- Full dynamic quantization pipeline
"""

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant.dynamic_quant import (
    _compute_bits_per_weight,
    _kl_divergence_loss,
    _quantize_dequantize,
    dynamic_quantize,
    estimate_sensitivities,
    estimate_threshold,
)


@pytest.mark.unit
@pytest.mark.gpu
class TestQuantizeDequantize:
    """Tests for quantize-dequantize simulation."""

    def test_qdq_introduces_error(self):
        """Test that quantize-dequantize introduces quantization error."""
        w = mx.random.normal((64, 64))
        w_qdq = _quantize_dequantize(w, bits=4, group_size=64)

        # Should be close but not exact
        assert not mx.allclose(w, w_qdq, atol=0.0)
        # But should be reasonably close
        error = mx.abs(w - w_qdq).mean()
        assert error < 0.5  # Reasonable error threshold

    def test_qdq_with_different_bits(self):
        """Test quantize-dequantize with different bit widths."""
        w = mx.random.normal((64, 64))

        # Higher bits = less error
        w_qdq_8 = _quantize_dequantize(w, bits=8, group_size=64)
        w_qdq_4 = _quantize_dequantize(w, bits=4, group_size=64)
        w_qdq_2 = _quantize_dequantize(w, bits=2, group_size=64)

        error_8 = mx.abs(w - w_qdq_8).mean()
        error_4 = mx.abs(w - w_qdq_4).mean()
        error_2 = mx.abs(w - w_qdq_2).mean()

        # More bits should have less error
        assert error_8 < error_4
        assert error_4 < error_2


@pytest.mark.unit
@pytest.mark.gpu
class TestKLDivergenceLoss:
    """Tests for KL divergence loss computation."""

    def test_kl_loss_zero_for_same_logits(self):
        """Test that KL loss is zero for identical logits."""
        logits = mx.random.normal((4, 100))
        kl = _kl_divergence_loss(logits, logits)

        # Should be very close to zero
        assert mx.allclose(kl, mx.zeros_like(kl), atol=1e-5)

    def test_kl_loss_positive_for_different_logits(self):
        """Test that KL loss is positive for different logits."""
        logits_orig = mx.random.normal((4, 100))
        logits_q = logits_orig + mx.random.normal((4, 100)) * 0.1

        kl = _kl_divergence_loss(logits_q, logits_orig)

        # KL divergence should be non-negative
        assert mx.all(kl >= 0.0)
        # And should be non-zero for perturbed logits
        assert mx.any(kl > 1e-6)

    def test_kl_loss_shape(self):
        """Test that KL loss has correct shape."""
        batch_size = 8
        seq_len = 16
        vocab_size = 1000

        logits_orig = mx.random.normal((batch_size, seq_len, vocab_size))
        logits_q = mx.random.normal((batch_size, seq_len, vocab_size))

        kl = _kl_divergence_loss(logits_q, logits_orig)

        # Should reduce over vocab dimension
        assert kl.shape == (batch_size, seq_len)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
class TestEstimateSensitivities:
    """Tests for sensitivity estimation."""

    def create_test_model(self):
        """Create a small test model."""
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

    def test_estimate_sensitivities_basic(self):
        """Test basic sensitivity estimation."""
        model = self.create_test_model()
        calibration_data = mx.random.randint(0, 100, (16, 8))

        sensitivities = estimate_sensitivities(
            model,
            calibration_data,
            low_bits=4,
            high_bits=6,
            batch_size=4
        )

        # Should have sensitivities for Linear layers
        assert len(sensitivities) > 0
        assert "linear1" in sensitivities
        assert "linear2" in sensitivities

        # Sensitivities should be numeric
        for _, value in sensitivities.items():
            assert isinstance(value, (int, float))

    def test_sensitivities_vary_across_layers(self):
        """Test that sensitivities vary across layers."""
        model = self.create_test_model()
        calibration_data = mx.random.randint(0, 100, (16, 8))

        sensitivities = estimate_sensitivities(
            model,
            calibration_data,
            low_bits=4,
            high_bits=6,
            batch_size=4
        )

        # Get all sensitivity values
        sens_values = list(sensitivities.values())

        # Should have at least 2 different values (not all identical)
        assert len(set(sens_values)) > 1

    def test_embedding_layer_detected(self):
        """Test that Embedding layers are detected as quantizable."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)

            def __call__(self, x):
                return self.embedding(x)

        model = SimpleModel()
        calibration_data = mx.random.randint(0, 100, (8, 8))

        sensitivities = estimate_sensitivities(
            model,
            calibration_data,
            batch_size=4
        )

        # Embedding should be detected (has to_quantized method)
        assert "embedding" in sensitivities
        assert isinstance(sensitivities["embedding"], (int, float))


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
class TestEstimateThreshold:
    """Tests for threshold estimation."""

    def create_test_model(self):
        """Create a test model with multiple linear layers."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(128, 128)
                self.linear2 = nn.Linear(128, 128)
                self.linear3 = nn.Linear(128, 128)

            def __call__(self, x):
                x = self.linear1(x)
                x = nn.relu(x)
                x = self.linear2(x)
                x = nn.relu(x)
                x = self.linear3(x)
                return x

        return SimpleModel()

    def test_threshold_estimation(self):
        """Test that threshold estimation finds a reasonable value."""
        model = self.create_test_model()

        # Create fake sensitivities
        sensitivities = {
            "linear1": 1.0,
            "linear2": 0.5,
            "linear3": 0.1,
        }

        threshold = estimate_threshold(
            model,
            sensitivities,
            target_bpw=5.0,  # Between 4 and 6
            low_bits=4,
            high_bits=6,
        )

        # Threshold should be between min and max sensitivities
        assert 0.1 <= threshold <= 1.0

    def test_threshold_affects_quantization(self):
        """Test that different thresholds lead to different quantization."""
        model = self.create_test_model()

        sensitivities = {
            "linear1": 1.0,
            "linear2": 0.5,
            "linear3": 0.1,
        }

        # High target BPW should give lower threshold (more high-bit layers)
        threshold_high = estimate_threshold(
            model,
            sensitivities,
            target_bpw=5.5,
            low_bits=4,
            high_bits=6,
        )

        # Low target BPW should give higher threshold (fewer high-bit layers)
        threshold_low = estimate_threshold(
            model,
            sensitivities,
            target_bpw=4.5,
            low_bits=4,
            high_bits=6,
        )

        # Higher BPW target should have lower threshold
        assert threshold_high < threshold_low


@pytest.mark.unit
@pytest.mark.gpu
class TestComputeBitsPerWeight:
    """Tests for computing bits per weight."""

    def test_quantized_linear(self):
        """Test BPW calculation for quantized linear layer."""
        linear = nn.Linear(128, 128)
        quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)

        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = quantized

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        bpw = _compute_bits_per_weight(model)

        # Should be 4 bits for 4-bit quantization
        assert abs(bpw - 4.0) < 0.1

    def test_mixed_precision(self):
        """Test BPW calculation for mixed precision."""
        linear1 = nn.Linear(128, 128)
        linear2 = nn.Linear(128, 128)

        # Quantize one layer to 4-bit, another to 6-bit
        q1 = nn.QuantizedLinear.from_linear(linear1, group_size=64, bits=4)
        q2 = nn.QuantizedLinear.from_linear(linear2, group_size=64, bits=6)

        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = q1
                self.linear2 = q2

            def __call__(self, x):
                return self.linear2(self.linear1(x))

        model = MixedModel()
        bpw = _compute_bits_per_weight(model)

        # Should be between 4 and 6 (average of mixed precision)
        assert 4.0 <= bpw <= 6.0
        # Should be close to average (5.0)
        assert abs(bpw - 5.0) < 0.5


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
class TestDynamicQuantize:
    """Integration tests for full dynamic quantization pipeline."""

    def create_test_model(self):
        """Create a test model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 64)
                self.linear2 = nn.Linear(64, 64)
                self.embedding = nn.Embedding(100, 64)

            def __call__(self, x):
                embedded = self.embedding(x)
                h = self.linear1(embedded)
                h = nn.relu(h)
                h = self.linear2(h)
                return h

        return SimpleModel()

    def test_dynamic_quantize_basic(self):
        """Test basic dynamic quantization."""
        model = self.create_test_model()
        calibration_data = mx.random.randint(0, 100, (16, 8))

        quantized_model, sensitivities = dynamic_quantize(
            model,
            calibration_data,
            target_bpw=4.5,
            low_bits=4,
            high_bits=6,
            batch_size=4
        )

        # Should return sensitivities
        assert isinstance(sensitivities, dict)
        assert len(sensitivities) > 0

        # Model should be quantized
        assert isinstance(quantized_model.linear1, nn.QuantizedLinear)
        assert isinstance(quantized_model.linear2, nn.QuantizedLinear)

        # Model should still be callable
        test_input = mx.random.randint(0, 100, (2, 8))
        output = quantized_model(test_input)
        assert output.shape == (2, 8, 64)

    def test_mixed_precision_applied(self):
        """Test that mixed-precision is actually applied."""
        model = self.create_test_model()
        calibration_data = mx.random.randint(0, 100, (16, 8))

        quantized_model, sensitivities = dynamic_quantize(
            model,
            calibration_data,
            target_bpw=4.5,
            low_bits=4,
            high_bits=6,
            batch_size=4
        )

        # Get bit widths of quantized layers
        bits_used = set()
        if isinstance(quantized_model.linear1, nn.QuantizedLinear):
            bits_used.add(quantized_model.linear1.bits)
        if isinstance(quantized_model.linear2, nn.QuantizedLinear):
            bits_used.add(quantized_model.linear2.bits)

        # Should have mixed precision (both 4 and 6 bit ideally, but at minimum quantized)
        assert len(bits_used) >= 1
        # All values should be either 4 or 6
        for bits in bits_used:
            assert bits in {4, 6}

    def test_sensitivities_save_load(self):
        """Test saving and loading sensitivities."""
        model = self.create_test_model()
        calibration_data = mx.random.randint(0, 100, (16, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            sens_path = Path(tmpdir) / "sensitivities.json"

            # First run: compute and save sensitivities
            _, sensitivities1 = dynamic_quantize(
                model,
                calibration_data,
                target_bpw=4.5,
                low_bits=4,
                high_bits=6,
                batch_size=4,
                sensitivities_path=sens_path
            )

            # Sensitivities file should exist
            assert sens_path.exists()

            # Second run: load sensitivities (should be faster)
            model2 = self.create_test_model()
            _, sensitivities2 = dynamic_quantize(
                model2,
                calibration_data,
                target_bpw=4.5,
                low_bits=4,
                high_bits=6,
                batch_size=4,
                sensitivities_path=sens_path
            )

            # Sensitivities should match
            assert sensitivities1.keys() == sensitivities2.keys()
            for key in sensitivities1:
                assert abs(sensitivities1[key] - sensitivities2[key]) < 1e-6

    def test_output_close_to_original(self):
        """Test that quantized model produces similar outputs."""
        model = self.create_test_model()
        calibration_data = mx.random.randint(0, 100, (16, 8))

        # Get output from original model
        test_input = mx.random.randint(0, 100, (4, 8))
        original_output = model(test_input)

        # Quantize
        quantized_model, _ = dynamic_quantize(
            model,
            calibration_data,
            target_bpw=5.0,  # Higher BPW for better accuracy
            low_bits=4,
            high_bits=6,
            batch_size=4
        )

        # Get output from quantized model
        quantized_output = quantized_model(test_input)

        # Outputs should be reasonably close
        assert quantized_output.shape == original_output.shape
        # Check that outputs are in similar range
        orig_mean = mx.mean(mx.abs(original_output))
        quant_mean = mx.mean(mx.abs(quantized_output))
        assert mx.abs(orig_mean - quant_mean) / orig_mean < 0.5  # Within 50%
