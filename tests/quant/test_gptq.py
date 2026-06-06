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

import smlx.quant.gptq as gptq_module
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
            batch_size=4,
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
            model, calibration_data, bits=4, group_size=32, batch_size=4
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
            model, calibration_data, bits=4, group_size=32, batch_size=4
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
            model, calibration_data, bits=4, group_size=32, batch_size=4
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
                model, calibration_data, bits=bits, group_size=32, batch_size=4
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
            batch_size=4,
        )

        # Should use 4-bit quantization with group_size=64 by default
        assert isinstance(quantized_model.linear, nn.QuantizedLinear)
        assert quantized_model.linear.bits == 4
        assert quantized_model.linear.group_size == 64


@pytest.mark.unit
@pytest.mark.gpu
class TestGPTQDeadFeatures:
    """Regression tests for the dead-feature / ill-conditioned-Hessian guard.

    A singular Hessian (dead input channels — common for attention o_proj, whose
    inputs can be all-zero on some dimensions) used to make the Cholesky inverse
    return NaN, corrupting the quantized weights so the model emitted only EOS.
    """

    def _quantize_with_dead_inputs(self, dead_cols):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 64, bias=False)

            def __call__(self, x):
                return self.linear(x)

        model = TinyModel()
        mx.eval(model.parameters())

        # Calibration data with some input channels forced to zero -> those
        # columns of H = X^T X are zero -> singular Hessian.
        calibration_data = mx.random.normal((16, 128))
        if dead_cols:
            mask = mx.ones((128,))
            for c in dead_cols:
                mask[c] = 0.0
            calibration_data = calibration_data * mask

        quantized = gptq_quantize(model, calibration_data, bits=4, group_size=64, batch_size=4)
        ql = quantized.linear
        return mx.dequantize(ql.weight, ql.scales, ql.biases, ql.group_size, ql.bits)

    def test_dead_input_channels_do_not_produce_nan(self):
        """Dead/zero input channels must not yield NaN/inf quantized weights."""
        weights = self._quantize_with_dead_inputs(dead_cols=[0, 5, 17, 63, 100])
        assert bool(mx.all(mx.isfinite(weights))), "GPTQ produced non-finite weights"

    def test_all_zero_calibration_is_stable(self):
        """A fully degenerate (all-zero) Hessian still yields finite weights."""
        weights = self._quantize_with_dead_inputs(dead_cols=list(range(128)))
        assert bool(mx.all(mx.isfinite(weights)))

    def _quantize_with_dead_value(self, dead_cols, dead_value):
        """Deterministically seeded so two calls differ ONLY in `dead_value`:
        identical model init and identical base calibration noise. This isolates
        the dead-feature handling — any output difference is attributable to the
        masked channels, not to a different random model."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 64, bias=False)

            def __call__(self, x):
                return self.linear(x)

        mx.random.seed(20240607)
        model = TinyModel()
        mx.eval(model.parameters())

        calibration_data = mx.random.normal((16, 128))
        mask = mx.ones((128,))
        for c in dead_cols:
            mask[c] = dead_value
        calibration_data = calibration_data * mask

        quantized = gptq_quantize(model, calibration_data, bits=4, group_size=64, batch_size=4)
        ql = quantized.linear
        return mx.dequantize(ql.weight, ql.scales, ql.biases, ql.group_size, ql.bits)

    def test_dead_feature_mask_catches_tiny_but_nonzero(self):
        """The guard must flag effectively-dead channels, not just exact zeros.

        An exact ``diag == 0`` test misses a channel whose squared-activation
        sum is tiny-but-nonzero (e.g. 1e-30) — it still makes H singular yet
        slips past equality. The magnitude-relative cutoff flags it; strong
        (live) channels are never flagged.
        """
        from smlx.quant.gptq import _dead_feature_mask

        # idx 0,1,4 live; idx 2,5 tiny-but-nonzero; idx 3 exact zero.
        diag = mx.array([10.0, 8.0, 1e-30, 0.0, 12.0, 1e-12])
        dead = _dead_feature_mask(diag)

        assert bool(dead[2]), "tiny-but-nonzero (1e-30) channel not flagged dead"
        assert bool(dead[3]), "exact-zero channel not flagged dead"
        assert bool(dead[5]), "tiny-but-nonzero (1e-12) channel not flagged dead"
        assert (
            not bool(dead[0]) and not bool(dead[1]) and not bool(dead[4])
        ), "live channel misflagged as dead"

    def test_dead_feature_mask_all_dead(self):
        """When every channel is dead (max == 0) all are flagged, matching the
        all-zero degenerate path."""
        from smlx.quant.gptq import _dead_feature_mask

        assert bool(mx.all(_dead_feature_mask(mx.zeros((8,)))))

    def test_tiny_dead_channels_match_exact_zero(self):
        """Tiny-but-nonzero dead inputs quantize identically to exact zeros.

        Safety lock for the magnitude-relative guard: genuinely-dead channels
        (whether their masked activation is exactly 0 or a tiny 1e-20 residue)
        must produce the same end-to-end quantized weights. This holds even
        before the guard change — the downstream damping already neutralizes a
        tiny residue — so this test does not by itself prove the guard; it
        proves the guard is behavior-preserving for the dead case and never
        misflags it. The white-box behavior of the guard is asserted directly
        in test_dead_feature_mask_catches_tiny_but_nonzero.
        """
        cols = [0, 5, 17, 63, 100]
        w_zero = self._quantize_with_dead_value(dead_cols=cols, dead_value=0.0)
        w_tiny = self._quantize_with_dead_value(dead_cols=cols, dead_value=1e-20)
        assert bool(mx.all(mx.isfinite(w_tiny)))
        assert bool(mx.allclose(w_zero, w_tiny, atol=1e-6)), "tiny-dead diverged from exact-zero"


@pytest.mark.unit
@pytest.mark.gpu
class TestGPTQEvalStrideInvariance:
    """Regression tests for the per-column -> strided lazy-graph eval cadence.

    `gptq_module._GPTQ_EVAL_COLUMN_STRIDE` controls how many columns of the inner
    error-compensation loop are deferred between `mx.eval` calls. The loop is a
    strict read-after-write chain on W, so the cadence is purely a host-sync /
    peak-memory trade-off and MUST NOT change the quantized weights. This test
    locks that invariant: any future change to the stride (or to the loop's eval
    placement) that perturbs the output bit-for-bit will fail here.
    """

    def _make_model(self):
        # Non-square, bias-free Linears with several groups per layer so the
        # strided eval actually spans multiple group boundaries.
        # in_features of every Linear must be a multiple of group_size (64), and
        # >1 group so the strided eval spans group boundaries: 256 -> 4 groups,
        # 128 -> 2 groups.
        class StridedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(256, 128, bias=False)
                self.linear2 = nn.Linear(128, 64, bias=False)

            def __call__(self, x):
                return self.linear2(nn.relu(self.linear1(x)))

        return StridedModel()

    def _quantize_dequantized(self, stride, calibration_data):
        """Quantize a deterministically-seeded model at the given eval stride and
        return the dequantized weights of each Linear layer."""
        original_stride = gptq_module._GPTQ_EVAL_COLUMN_STRIDE
        gptq_module._GPTQ_EVAL_COLUMN_STRIDE = stride
        try:
            # Same seed -> identical Linear init across every call.
            mx.random.seed(1234)
            model = self._make_model()
            mx.eval(model.parameters())

            quantized = gptq_quantize(model, calibration_data, bits=4, group_size=64, batch_size=4)
            result = {}
            for name in ("linear1", "linear2"):
                ql = getattr(quantized, name)
                result[name] = mx.dequantize(
                    ql.weight, ql.scales, ql.biases, ql.group_size, ql.bits
                )
                mx.eval(result[name])
            return result
        finally:
            gptq_module._GPTQ_EVAL_COLUMN_STRIDE = original_stride

    def test_output_is_bit_identical_across_strides(self):
        """Quantized weights must be byte-for-byte identical for any eval stride."""
        # Fixed calibration data shared across every stride (not mutated by GPTQ).
        mx.random.seed(99)
        calibration_data = mx.random.normal((16, 256))
        mx.eval(calibration_data)

        reference = self._quantize_dequantized(1, calibration_data)  # per-column
        for stride in (2, 3, 8):
            candidate = self._quantize_dequantized(stride, calibration_data)
            for name, ref in reference.items():
                assert mx.array_equal(ref, candidate[name]), (
                    f"GPTQ output diverged for {name} at eval stride {stride}; "
                    "the eval cadence must not affect quantization results."
                )

    def test_default_stride_is_the_measured_knee(self):
        """The shipped stride is the verified zero-memory-regression value (2)."""
        assert gptq_module._GPTQ_EVAL_COLUMN_STRIDE == 2
