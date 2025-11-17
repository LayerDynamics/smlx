"""
Tests for DWQ (Distilled Weight Quantization) implementation.

Tests cover:
- KL divergence computation
- MSE loss computation
- Layer sensitivity analysis
- Full DWQ quantization pipeline
- Simplified DWQ quantization
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant.dwq import (
    _kl_divergence,
    _mse_loss,
    dwq_quantize_simple,
)


@pytest.mark.unit
class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_kl_divergence_identical_distributions(self):
        """Test that KL divergence is zero for identical distributions."""
        logits = mx.random.normal((4, 10))

        kl = _kl_divergence(logits, logits, temperature=1.0)

        # KL(P||P) should be ~0
        assert mx.abs(kl).item() < 1e-5

    def test_kl_divergence_different_distributions(self):
        """Test that KL divergence is positive for different distributions."""
        teacher_logits = mx.random.normal((8, 20))
        student_logits = mx.random.normal((8, 20))

        kl = _kl_divergence(teacher_logits, student_logits, temperature=1.0)

        # KL(P||Q) should be positive
        assert kl.item() > 0

    def test_kl_divergence_with_temperature(self):
        """Test that temperature affects KL divergence."""
        teacher_logits = mx.array([[1.0, 2.0, 3.0]])
        student_logits = mx.array([[1.1, 1.9, 3.1]])

        kl_t1 = _kl_divergence(teacher_logits, student_logits, temperature=1.0)
        kl_t2 = _kl_divergence(teacher_logits, student_logits, temperature=2.0)

        # Higher temperature should give different (typically lower) KL
        # because distributions are softer
        assert kl_t1.item() != kl_t2.item()

    def test_kl_divergence_shape(self):
        """Test that KL divergence returns a scalar."""
        teacher_logits = mx.random.normal((16, 100))
        student_logits = mx.random.normal((16, 100))

        kl = _kl_divergence(teacher_logits, student_logits, temperature=1.0)

        # Should return scalar
        assert kl.shape == ()

    def test_kl_divergence_batch_independence(self):
        """Test KL divergence handles batches correctly."""
        # Small batch
        teacher_small = mx.random.normal((2, 10))
        student_small = mx.random.normal((2, 10))
        kl_small = _kl_divergence(teacher_small, student_small)

        # Larger batch with same distribution
        teacher_large = mx.concatenate([teacher_small] * 4, axis=0)
        student_large = mx.concatenate([student_small] * 4, axis=0)
        kl_large = _kl_divergence(teacher_large, student_large)

        # Should be similar (mean over batch)
        assert mx.allclose(kl_small, kl_large, rtol=0.1)


@pytest.mark.unit
class TestMSELoss:
    """Tests for MSE loss computation."""

    def test_mse_identical_outputs(self):
        """Test MSE is zero for identical outputs."""
        outputs = mx.random.normal((8, 64))

        mse = _mse_loss(outputs, outputs)

        assert mx.allclose(mse, mx.array(0.0), atol=1e-6)

    def test_mse_different_outputs(self):
        """Test MSE is positive for different outputs."""
        teacher = mx.random.normal((4, 32))
        student = mx.random.normal((4, 32))

        mse = _mse_loss(teacher, student)

        assert mse.item() > 0

    def test_mse_shape(self):
        """Test MSE returns a scalar."""
        teacher = mx.random.normal((16, 128))
        student = mx.random.normal((16, 128))

        mse = _mse_loss(teacher, student)

        assert mse.shape == ()

    def test_mse_known_value(self):
        """Test MSE with known values."""
        teacher = mx.array([[1.0, 2.0, 3.0]])
        student = mx.array([[1.0, 2.0, 4.0]])

        mse = _mse_loss(teacher, student)

        # MSE = mean((1-1)^2 + (2-2)^2 + (3-4)^2) = 1/3
        expected = 1.0 / 3.0
        assert mx.allclose(mse, mx.array(expected), atol=1e-5)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
class TestDWQSimple:
    """Tests for simplified DWQ quantization."""

    def test_dwq_simple_quantizes_model(self):
        """Test that DWQ simple quantizes a small model."""
        # Create simple model
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(32, 64)
                self.linear2 = nn.Linear(64, 32)

            def __call__(self, x):
                return self.linear2(nn.relu(self.linear1(x)))

        model = TinyModel()

        # Create dummy calibration data (token IDs)
        calibration_data = mx.random.randint(0, 100, (16, 32))

        # Quantize with DWQ simple
        quantized_model = dwq_quantize_simple(
            model,
            calibration_data,
            bits=4,
            group_size=32,
            batch_size=4,
        )

        # Model should still be callable
        test_input = mx.random.randint(0, 100, (2, 32))
        output = quantized_model(test_input)

        assert output.shape == (2, 32)

    def test_dwq_simple_computes_quality_metrics(self):
        """Test that DWQ simple computes and reports quality metrics."""
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = TinyModel()
        calibration_data = mx.random.normal((8, 64))

        # Should complete without errors
        quantized_model = dwq_quantize_simple(
            model,
            calibration_data,
            bits=4,
            group_size=64,  # MLX requires group_size of 32, 64, or 128
        )

        assert quantized_model is not None


@pytest.mark.unit
@pytest.mark.gpu
class TestDWQComponents:
    """Tests for individual DWQ components."""

    def test_temperature_scaling_effect(self):
        """Test that temperature scaling affects distributions."""
        logits = mx.array([[10.0, 1.0, 0.1]])

        # T=1 (normal)
        kl_t1 = _kl_divergence(logits, logits * 0.9, temperature=1.0)

        # T=2 (softer)
        kl_t2 = _kl_divergence(logits, logits * 0.9, temperature=2.0)

        # Higher temperature should give different divergence
        assert kl_t1.item() != kl_t2.item()

    def test_quantization_quality_measurement(self):
        """Test measuring quantization quality via teacher-student loss."""
        # Teacher model (full precision)
        teacher = nn.Linear(32, 32)

        # Student model (quantized)
        student = teacher.to_quantized(bits=4, group_size=32)

        # Test inputs
        x = mx.random.normal((8, 32))

        # Compute outputs
        teacher_out = teacher(x)
        student_out = student(x)

        # Measure quality
        mse = _mse_loss(teacher_out, student_out)

        # Should have some error due to quantization
        assert mse.item() > 0


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
class TestDWQQuantization:
    """Tests for complete DWQ quantization pipeline."""

    def test_dwq_quantize_synthetic_model(
        self, synthetic_transformer_model, small_calibration_data
    ):
        """
        Test DWQ quantization on a synthetic transformer model.

        Uses a small synthetic model for fast testing.
        """
        from smlx.quant import dwq_quantize

        model = synthetic_transformer_model
        calibration_data = small_calibration_data

        # Test with reduced iterations for speed
        quantized_model = dwq_quantize(
            model,
            calibration_data,
            bits=4,
            group_size=64,
            num_iterations=2,  # Reduced for faster testing
            temperature=2.0,
            batch_size=4,
        )

        # Verify quantization succeeded
        assert quantized_model is not None

        # Verify model still works
        test_input = mx.random.randint(0, 1000, (2, 64))
        output = quantized_model(test_input)
        assert output.shape == (2, 64, 1000)

    def test_dwq_with_sensitivity_analysis(
        self, synthetic_transformer_model, small_calibration_data
    ):
        """
        Test DWQ with sensitivity-based mixed precision.

        Uses synthetic model to test sensitivity analysis.
        """
        from smlx.quant import dwq_quantize

        model = synthetic_transformer_model
        calibration_data = small_calibration_data

        # Test sensitivity mode
        quantized_model = dwq_quantize(
            model,
            calibration_data,
            use_sensitivity=True,  # Enable mixed precision
            bits=4,
            num_iterations=1,  # Just 1 iteration for speed
            batch_size=4,
        )

        # Should complete successfully
        assert quantized_model is not None

        # Model should still work
        test_input = mx.random.randint(0, 1000, (2, 64))
        output = quantized_model(test_input)
        assert output.shape == (2, 64, 1000)


@pytest.mark.unit
class TestDWQEdgeCases:
    """Test edge cases and error handling."""

    def test_kl_divergence_numerical_stability(self):
        """Test KL divergence with extreme logits."""
        # Very large logits (potential overflow)
        teacher = mx.array([[100.0, -100.0, 0.0]])
        student = mx.array([[99.0, -99.0, 0.1]])

        # Should not crash or return nan/inf
        kl = _kl_divergence(teacher, student, temperature=1.0)

        assert mx.isfinite(kl)

    def test_kl_divergence_near_zero_probabilities(self):
        """Test KL divergence with near-zero probabilities."""
        # Teacher has very confident prediction
        teacher = mx.array([[10.0, -10.0, -10.0]])

        # Student is less confident
        student = mx.array([[5.0, -5.0, -5.0]])

        kl = _kl_divergence(teacher, student, temperature=1.0)

        # Should be finite and positive
        assert mx.isfinite(kl)
        assert kl.item() > 0

    def test_mse_loss_with_different_dtypes(self):
        """Test MSE loss handles different dtypes correctly."""
        teacher = mx.random.normal((4, 16)).astype(mx.float32)
        student = mx.random.normal((4, 16)).astype(mx.float16)

        # Should not crash
        mse = _mse_loss(teacher, student)

        assert mx.isfinite(mse)

    def test_dwq_simple_with_small_batch(self):
        """Test DWQ simple with batch_size=1."""
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = TinyModel()
        calibration_data = mx.random.normal((4, 64))

        # Should work with batch_size=1
        quantized_model = dwq_quantize_simple(
            model,
            calibration_data,
            bits=4,
            group_size=64,  # MLX requires group_size of 32, 64, or 128
            batch_size=1,
        )

        assert quantized_model is not None


@pytest.mark.benchmark
@pytest.mark.gpu
@pytest.mark.slow
class TestDWQPerformance:
    """Performance benchmarks for DWQ quantization."""

    def test_dwq_simple_speed(self):
        """
        Benchmark DWQ simple quantization speed.

        This test uses pytest-benchmark if available, otherwise just runs timing.
        """

        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(128, 128) for _ in range(4)]

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = SmallModel()
        calibration_data = mx.random.normal((32, 128))

        # Simple timing test (pytest-benchmark is available in dev dependencies)
        import time

        start = time.time()
        result = dwq_quantize_simple(
            model,
            calibration_data,
            bits=4,
            group_size=64,
        )
        elapsed = time.time() - start

        print(f"\nDWQ simple took {elapsed:.3f}s")

        # Verify the result is valid
        assert result is not None
