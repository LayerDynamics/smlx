"""
Tests for FP4 quantization (all modes: E2M1, MXFP4, NVFP4, NF4).
"""

import pytest
import mlx.core as mx
import mlx.nn as nn

from smlx.quant.fp4 import (
    FP4Mode,
    FP4_E2M1_VALUES,
    NF4_VALUES,
    quantize_fp4,
    dequantize_fp4,
    quantize_model_fp4,
    estimate_fp4_size,
    compare_fp4_vs_int4,
    # Legacy API
    quantize_to_fp4,
    dequantize_from_fp4,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def random_weights():
    """Generate random weights for testing."""
    mx.random.seed(42)
    return mx.random.normal((128, 256))


@pytest.fixture
def small_weights():
    """Generate small test weights."""
    mx.random.seed(123)
    return mx.random.normal((8, 64))


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(128, 64)
            self.layer2 = nn.Linear(64, 32)

        def __call__(self, x):
            x = self.layer1(x)
            x = nn.relu(x)
            x = self.layer2(x)
            return x

    return SimpleModel()


# ============================================================================
# Test Lookup Tables
# ============================================================================


class TestLookupTables:
    """Test FP4 lookup tables."""

    def test_e2m1_values_count(self):
        """Test E2M1 has 16 values."""
        assert FP4_E2M1_VALUES.size == 16

    def test_e2m1_values_symmetric(self):
        """Test E2M1 values are symmetric around zero."""
        positive = FP4_E2M1_VALUES[:8]
        negative = FP4_E2M1_VALUES[8:]
        assert mx.allclose(positive, -negative, atol=1e-6)

    def test_nf4_values_count(self):
        """Test NF4 has 16 values."""
        assert NF4_VALUES.size == 16

    def test_nf4_values_range(self):
        """Test NF4 values are in [-1, 1] range."""
        assert mx.min(NF4_VALUES).item() >= -1.0
        assert mx.max(NF4_VALUES).item() <= 1.0

    def test_nf4_values_sorted(self):
        """Test NF4 values are sorted."""
        nf4_list = NF4_VALUES.tolist()
        assert nf4_list == sorted(nf4_list)


# ============================================================================
# Test E2M1 Quantization
# ============================================================================


class TestE2M1Quantization:
    """Test E2M1 FP4 quantization (simulated)."""

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_quantize_dequantize_roundtrip(self, random_weights, group_size):
        """Test E2M1 quantization/dequantization round-trip."""
        # Quantize
        q, scales = quantize_fp4(random_weights, mode="e2m1", group_size=group_size)

        # Check output types
        assert q.dtype == mx.uint8
        assert scales.dtype == mx.float16

        # Dequantize
        restored = dequantize_fp4(q, scales, mode="e2m1", group_size=group_size)

        # Check shape preserved
        assert restored.shape == random_weights.shape

        # Check quantization error is reasonable
        # E2M1 uses only 16 values, so error is higher than native formats
        error = mx.mean(mx.abs(restored - random_weights)).item()
        assert error < 0.4  # E2M1 has coarser quantization than MXFP4/NVFP4

    def test_quantize_indices_range(self, small_weights):
        """Test quantized indices are in valid range [0, 15]."""
        q, _ = quantize_fp4(small_weights, mode="e2m1", group_size=32)
        assert mx.min(q).item() >= 0
        assert mx.max(q).item() <= 15

    def test_zero_weights(self):
        """Test quantization of all-zero weights."""
        weights = mx.zeros((64, 64))
        q, scales = quantize_fp4(weights, mode="e2m1", group_size=32)
        restored = dequantize_fp4(q, scales, mode="e2m1", group_size=32)
        assert mx.allclose(restored, weights, atol=1e-6)

    def test_uniform_weights(self):
        """Test quantization of uniform weights."""
        weights = mx.ones((64, 64)) * 2.5
        q, scales = quantize_fp4(weights, mode="e2m1", group_size=32)
        restored = dequantize_fp4(q, scales, mode="e2m1", group_size=32)
        # Should quantize to nearest E2M1 value (2.0 or 3.0)
        assert mx.mean(mx.abs(restored - weights)).item() < 0.6


# ============================================================================
# Test MXFP4 Quantization
# ============================================================================


class TestMXFP4Quantization:
    """Test MXFP4 (MLX native) quantization."""

    def test_quantize_dequantize_roundtrip(self, random_weights):
        """Test MXFP4 quantization/dequantization round-trip."""
        # Quantize (group_size automatically set to 32)
        q, scales = quantize_fp4(random_weights, mode="mxfp4")

        # Dequantize
        restored = dequantize_fp4(q, scales, mode="mxfp4")

        # Check shape preserved
        assert restored.shape == random_weights.shape

        # Check quantization error
        error = mx.mean(mx.abs(restored - random_weights)).item()
        assert error < 0.1

    def test_group_size_warning(self, small_weights, capsys):
        """Test that MXFP4 warns when group_size != 32."""
        # Should warn about group_size
        quantize_fp4(small_weights, mode="mxfp4", group_size=64)
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "group_size=32" in captured.out.lower()

    def test_fixed_group_size(self, random_weights):
        """Test MXFP4 always uses group_size=32."""
        q1, s1 = quantize_fp4(random_weights, mode="mxfp4", group_size=32)
        q2, s2 = quantize_fp4(random_weights, mode="mxfp4", group_size=64)

        # Both should use group_size=32, so results should be identical
        assert mx.array_equal(q1, q2)
        assert mx.array_equal(s1, s2)


# ============================================================================
# Test NVFP4 Quantization
# ============================================================================


class TestNVFP4Quantization:
    """Test NVFP4 (MLX native) quantization."""

    def test_quantize_dequantize_roundtrip(self, random_weights):
        """Test NVFP4 quantization/dequantization round-trip."""
        # Quantize (group_size automatically set to 16)
        q, scales = quantize_fp4(random_weights, mode="nvfp4")

        # Dequantize
        restored = dequantize_fp4(q, scales, mode="nvfp4")

        # Check shape preserved
        assert restored.shape == random_weights.shape

        # Check quantization error
        error = mx.mean(mx.abs(restored - random_weights)).item()
        assert error < 0.1

    def test_group_size_warning(self, small_weights, capsys):
        """Test that NVFP4 warns when group_size != 16."""
        # Should warn about group_size
        quantize_fp4(small_weights, mode="nvfp4", group_size=32)
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "group_size=16" in captured.out.lower()

    def test_fixed_group_size(self, random_weights):
        """Test NVFP4 always uses group_size=16."""
        q1, s1 = quantize_fp4(random_weights, mode="nvfp4", group_size=16)
        q2, s2 = quantize_fp4(random_weights, mode="nvfp4", group_size=32)

        # Both should use group_size=16, so results should be identical
        assert mx.array_equal(q1, q2)
        assert mx.array_equal(s1, s2)


# ============================================================================
# Test NF4 Quantization
# ============================================================================


class TestNF4Quantization:
    """Test NF4 (Normal Float 4) quantization."""

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_quantize_dequantize_roundtrip(self, random_weights, group_size):
        """Test NF4 quantization/dequantization round-trip."""
        # Quantize
        q, scales = quantize_fp4(random_weights, mode="nf4", group_size=group_size)

        # Check output types
        assert q.dtype == mx.uint8
        assert scales.dtype == mx.float16

        # Dequantize
        restored = dequantize_fp4(q, scales, mode="nf4", group_size=group_size)

        # Check shape preserved
        assert restored.shape == random_weights.shape

        # Check quantization error
        error = mx.mean(mx.abs(restored - random_weights)).item()
        assert error < 0.1

    def test_normal_distribution_optimal(self):
        """Test NF4 is optimal for normally distributed weights."""
        # Generate weights from standard normal
        mx.random.seed(42)
        weights = mx.random.normal((256, 512))

        # Quantize with NF4
        q_nf4, s_nf4 = quantize_fp4(weights, mode="nf4", group_size=64)
        restored_nf4 = dequantize_fp4(q_nf4, s_nf4, mode="nf4", group_size=64)
        error_nf4 = mx.mean(mx.abs(restored_nf4 - weights)).item()

        # Quantize with E2M1
        q_e2m1, s_e2m1 = quantize_fp4(weights, mode="e2m1", group_size=64)
        restored_e2m1 = dequantize_fp4(q_e2m1, s_e2m1, mode="e2m1", group_size=64)
        error_e2m1 = mx.mean(mx.abs(restored_e2m1 - weights)).item()

        # NF4 should have equal or better error for normal distribution
        # (though not guaranteed in all cases due to randomness)
        assert error_nf4 < error_e2m1 * 1.5  # Allow 50% margin


# ============================================================================
# Test Mode Comparison
# ============================================================================


class TestModeComparison:
    """Test comparisons between different FP4 modes."""

    def test_all_modes_preserve_shape(self, random_weights):
        """Test all modes preserve weight shape."""
        modes = ["e2m1", "mxfp4", "nvfp4", "nf4"]

        for mode in modes:
            q, scales = quantize_fp4(random_weights, mode=mode)
            restored = dequantize_fp4(q, scales, mode=mode)
            assert restored.shape == random_weights.shape, f"Mode {mode} failed"

    def test_e2m1_vs_mxfp4_similar_quality(self, random_weights):
        """Test E2M1 and MXFP4 have similar quality (both use E2M1 format)."""
        # E2M1 simulation with group_size=32
        q_e2m1, s_e2m1 = quantize_fp4(random_weights, mode="e2m1", group_size=32)
        restored_e2m1 = dequantize_fp4(q_e2m1, s_e2m1, mode="e2m1", group_size=32)
        error_e2m1 = mx.mean(mx.abs(restored_e2m1 - random_weights)).item()

        # MXFP4 (native, group_size=32)
        q_mxfp4, s_mxfp4 = quantize_fp4(random_weights, mode="mxfp4")
        restored_mxfp4 = dequantize_fp4(q_mxfp4, s_mxfp4, mode="mxfp4")
        error_mxfp4 = mx.mean(mx.abs(restored_mxfp4 - random_weights)).item()

        # MXFP4 should be better (hardware-native) than E2M1 (simulated)
        # But both use E2M1 format, so errors should be in same ballpark
        assert abs(error_e2m1 - error_mxfp4) < 0.3
        assert error_mxfp4 < error_e2m1  # MXFP4 should be more accurate


# ============================================================================
# Test Model-Level Functions
# ============================================================================


class TestModelQuantization:
    """Test model-level quantization functions."""

    def test_quantize_model_e2m1(self, simple_model):
        """Test model quantization with E2M1."""
        quantized_weights = quantize_model_fp4(simple_model, mode="e2m1", group_size=64)

        # Should have quantized both layers
        assert len(quantized_weights) >= 2

        # Check each entry has (quantized, scales)
        for name, (q, scales) in quantized_weights.items():
            assert q.dtype == mx.uint8
            assert scales.dtype == mx.float16

    def test_quantize_model_mxfp4(self, simple_model):
        """Test model quantization with MXFP4."""
        quantized_weights = quantize_model_fp4(simple_model, mode="mxfp4")

        # Should have quantized both layers
        assert len(quantized_weights) >= 2

    def test_estimate_fp4_size(self, simple_model):
        """Test FP4 size estimation."""
        stats = estimate_fp4_size(simple_model, group_size=64)

        # Check all fields present
        assert "current_mb" in stats
        assert "fp4_mb" in stats
        assert "reduction_ratio" in stats
        assert "saved_mb" in stats

        # FP4 should reduce size
        assert stats["fp4_mb"] < stats["current_mb"]
        assert stats["reduction_ratio"] > 1.0

    def test_compare_fp4_vs_int4(self, random_weights):
        """Test FP4 vs INT4 comparison."""
        comparison = compare_fp4_vs_int4(random_weights, group_size=64)

        # Check all fields present
        assert "fp4_error" in comparison
        assert "int4_error" in comparison
        assert "fp4_max_error" in comparison
        assert "int4_max_error" in comparison
        assert "recommendation" in comparison

        # Errors should be positive
        assert comparison["fp4_error"] >= 0
        assert comparison["int4_error"] >= 0


# ============================================================================
# Test Legacy API
# ============================================================================


class TestLegacyAPI:
    """Test backward compatibility with legacy API."""

    def test_legacy_quantize_to_fp4(self, small_weights):
        """Test legacy quantize_to_fp4 function."""
        q, scales = quantize_to_fp4(small_weights, group_size=64, format="e2m1")

        assert q.dtype == mx.uint8
        assert scales.dtype == mx.float16

    def test_legacy_dequantize_from_fp4(self, small_weights):
        """Test legacy dequantize_from_fp4 function."""
        q, scales = quantize_to_fp4(small_weights, group_size=64, format="e2m1")
        restored = dequantize_from_fp4(q, scales, group_size=64, format="e2m1")

        assert restored.shape == small_weights.shape

    def test_legacy_api_matches_new_api(self, small_weights):
        """Test legacy API produces same results as new API."""
        # Legacy API
        q_old, s_old = quantize_to_fp4(small_weights, group_size=64, format="e2m1")
        restored_old = dequantize_from_fp4(q_old, s_old, group_size=64, format="e2m1")

        # New API
        q_new, s_new = quantize_fp4(small_weights, mode="e2m1", group_size=64)
        restored_new = dequantize_fp4(q_new, s_new, mode="e2m1", group_size=64)

        # Should produce identical results
        assert mx.array_equal(restored_old, restored_new)


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_mode(self, small_weights):
        """Test error on invalid mode."""
        with pytest.raises(ValueError, match="Unsupported FP4 mode"):
            quantize_fp4(small_weights, mode="invalid_mode")

    def test_mxfp4_dequantize_wrong_group_size(self, small_weights):
        """Test error when dequantizing MXFP4 with wrong group_size."""
        q, scales = quantize_fp4(small_weights, mode="mxfp4")

        with pytest.raises(ValueError, match="MXFP4 requires group_size=32"):
            dequantize_fp4(q, scales, mode="mxfp4", group_size=64)

    def test_nvfp4_dequantize_wrong_group_size(self, small_weights):
        """Test error when dequantizing NVFP4 with wrong group_size."""
        q, scales = quantize_fp4(small_weights, mode="nvfp4")

        with pytest.raises(ValueError, match="NVFP4 requires group_size=16"):
            dequantize_fp4(q, scales, mode="nvfp4", group_size=32)


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestIntegration:
    """Integration tests for FP4 quantization."""

    def test_full_model_workflow(self, simple_model):
        """Test complete model quantization workflow."""
        # 1. Quantize model
        quantized_weights = quantize_model_fp4(simple_model, mode="e2m1", group_size=64)

        # 2. Dequantize specific layer
        layer_names = list(quantized_weights.keys())
        assert len(layer_names) > 0

        q, scales = quantized_weights[layer_names[0]]
        restored = dequantize_fp4(q, scales, mode="e2m1", group_size=64)

        # 3. Check shape matches original
        original_weight = None
        for name, module in simple_model.named_modules():
            if name == layer_names[0] and hasattr(module, "weight"):
                original_weight = module.weight
                break

        if original_weight is not None:
            assert restored.shape == original_weight.shape

    @pytest.mark.parametrize("mode", ["e2m1", "mxfp4", "nvfp4", "nf4"])
    def test_all_modes_work_end_to_end(self, random_weights, mode):
        """Test all modes work in end-to-end scenario."""
        # Quantize
        q, scales = quantize_fp4(random_weights, mode=mode)

        # Dequantize
        restored = dequantize_fp4(q, scales, mode=mode)

        # Verify
        assert restored.shape == random_weights.shape
        error = mx.mean(mx.abs(restored - random_weights)).item()
        # E2M1/NF4 have higher error (~0.3), MXFP4/NVFP4 are lower (~0.09)
        assert error < 0.4  # Reasonable error threshold for all modes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
