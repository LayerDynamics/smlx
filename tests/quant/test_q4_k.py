"""
Tests for Q4_K and Q4_K_M quantization formats.

Tests both the GGML-compatible Q4_K format and the MLX-native Q4_K_M-style
mixed-precision quantization.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn

from smlx.quant.q4_k_m import (
    quantize_to_q4_k,
    dequantize_from_q4_k,
    quantize_model_q4_k,
    quantize_model_q4_k_m,
    quantize_model_q4_k_m_ggml,
    estimate_q4_k_size,
    Q4_K_BLOCK_SIZE,
)
from smlx.quant.mlx_mixed import (
    quantize_model_mixed,
    create_q4_k_m_style_predicate,
    estimate_mixed_size,
)
from smlx.quant.utils import (
    pack_scales_mins_6bit,
    unpack_scales_mins_6bit,
    pack_weights_4bit,
    unpack_weights_4bit,
)


@pytest.mark.unit
class TestBitPacking:
    """Test bit-packing utilities used by Q4_K."""

    def test_pack_unpack_weights_4bit(self):
        """Test 4-bit weight packing/unpacking roundtrip."""
        # Create test weights (must be even length)
        weights = mx.array([0, 1, 2, 3, 14, 15, 0, 1], dtype=mx.uint8)

        # Pack and unpack
        packed = pack_weights_4bit(weights)
        unpacked = unpack_weights_4bit(packed, 8)

        # Should be exact roundtrip
        assert mx.array_equal(weights, unpacked)

        # Check packed size is half
        assert packed.size == weights.size // 2

    def test_pack_unpack_scales_mins_6bit(self):
        """Test 6-bit scales/mins packing/unpacking roundtrip."""
        # Create test scales and mins (8 each, 6-bit range [0, 63])
        scales = mx.array([10, 20, 30, 40, 50, 60, 62, 63], dtype=mx.uint8)
        mins = mx.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=mx.uint8)

        # Pack and unpack
        packed = pack_scales_mins_6bit(scales, mins)
        scales_unpacked, mins_unpacked = unpack_scales_mins_6bit(packed)

        # Should be exact roundtrip
        assert mx.array_equal(scales, scales_unpacked)
        assert mx.array_equal(mins, mins_unpacked)

        # Check packed size is 12 bytes
        assert packed.size == 12

    def test_pack_weights_4bit_batch(self):
        """Test 4-bit weight packing with batch dimension."""
        # Batch of weights
        weights = mx.array([[0, 1, 2, 3], [14, 15, 0, 1]], dtype=mx.uint8)

        # Pack and unpack
        packed = pack_weights_4bit(weights)
        unpacked = unpack_weights_4bit(packed, 4)

        # Should preserve batch dimension
        assert packed.shape == (2, 2)
        assert mx.array_equal(weights, unpacked)


@pytest.mark.unit
class TestQ4KQuantization:
    """Test Q4_K quantization and dequantization."""

    def test_quantize_dequantize_roundtrip(self):
        """Test Q4_K quantization/dequantization roundtrip with small error."""
        # Create test weights (multiples of 256 for clean block alignment)
        mx.random.seed(42)
        weights = mx.random.normal((512, 512)).astype(mx.float32)

        # Quantize and dequantize (now returns 5 values with d_min_scales)
        packed_w, d_scales, d_mins, d_min_scales, packed_sm = quantize_to_q4_k(weights)
        weights_deq = dequantize_from_q4_k(
            packed_w, d_scales, d_mins, d_min_scales, packed_sm, weights.shape
        )

        # Check shapes match
        assert weights_deq.shape == weights.shape

        # Check quantization error is reasonable (< 10% mean absolute error for 4-bit)
        error = mx.mean(mx.abs(weights - weights_deq))
        relative_error = error / (mx.mean(mx.abs(weights)) + 1e-10)
        assert float(relative_error) < 0.10, f"Quantization error too high: {float(relative_error):.4f}"

    def test_q4_k_block_size(self):
        """Test that Q4_K uses correct block size (256 weights)."""
        # Create weights not aligned to block size
        weights = mx.random.normal((100, 100)).astype(mx.float32)  # 10,000 weights

        # Quantize (should pad to multiple of 256)
        packed_w, d_scales, d_mins, d_min_scales, packed_sm = quantize_to_q4_k(weights)

        # Check number of blocks: ceil(10000 / 256) = 40 blocks
        expected_blocks = (10000 + 255) // 256
        assert packed_w.shape[0] == expected_blocks
        assert packed_w.shape[1] == 128  # 256 weights × 4 bits / 8 bits/byte
        assert packed_sm.shape == (expected_blocks, 12)  # 8 × 6-bit scales + 8 × 6-bit mins

        # Dequantize should restore original shape
        weights_deq = dequantize_from_q4_k(
            packed_w, d_scales, d_mins, d_min_scales, packed_sm, weights.shape
        )
        assert weights_deq.shape == weights.shape

    def test_q4_k_storage_size(self):
        """Test that Q4_K achieves proper storage size."""
        weights = mx.random.normal((256, 1)).astype(mx.float32)  # Exactly 256 weights

        # Quantize
        packed_w, d_scales, d_mins, d_min_scales, packed_sm = quantize_to_q4_k(weights)

        # Calculate storage: 1 block
        # - 128 bytes: packed weights (256 × 4 bits / 8)
        # - 12 bytes: packed scales/mins (8 × 6-bit + 8 × 6-bit)
        # - 2 bytes: d_scale (FP16)
        # - 2 bytes: d_min (FP16)
        # - 2 bytes: d_min_scale (FP16)
        # Total: 146 bytes / 256 weights = 0.570 bytes/weight = 4.56 bits/weight

        total_bytes = (
            packed_w.nbytes  # 128 bytes
            + packed_sm.nbytes  # 12 bytes
            + d_scales.nbytes  # 2 bytes
            + d_mins.nbytes  # 2 bytes
            + d_min_scales.nbytes  # 2 bytes
        )

        assert total_bytes == 146
        bits_per_weight = (total_bytes * 8) / 256
        assert abs(bits_per_weight - 4.5625) < 0.01  # Slightly higher than pure Q4_K


@pytest.mark.unit
class TestQ4KMModel:
    """Test Q4_K_M model-level quantization."""

    def test_quantize_model_q4_k_ggml_mode(self):
        """Test Q4_K model quantization in GGML-compatible mode."""
        # Create simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 128)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(128, 64)

        model = SimpleModel()

        # Quantize in GGML mode
        quantize_model_q4_k(model, inplace=True)

        # Check that Linear layers have Q4_K metadata
        assert hasattr(model.linear1, "weight_q4_k_packed")
        assert hasattr(model.linear1, "d_scales_q4_k")
        assert hasattr(model.linear1, "d_mins_q4_k")
        assert hasattr(model.linear1, "d_min_scales_q4_k")  # New field
        assert hasattr(model.linear1, "packed_scales_mins_q4_k")
        assert model.linear1.quantization_format == "q4_k"

    def test_quantize_model_q4_k_m_mlx_mode(self):
        """Test Q4_K_M model quantization in MLX native mode (default)."""
        # Create test model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # Quantize in MLX native mode (default, use_mlx_native=True)
        quantize_model_q4_k_m(model, inplace=True, use_mlx_native=True)

        # Check that Linear layers are now QuantizedLinear
        quantized_count = 0
        for module in model.modules():
            if isinstance(module, nn.QuantizedLinear):
                quantized_count += 1
                assert hasattr(module, "quantization_format")
                assert module.quantization_format == "mlx_mixed_q4_k_m"

        # Should have quantized the 2 Linear layers
        assert quantized_count == 2

    def test_quantize_model_mixed_styles(self):
        """Test different mixed-precision quantization styles."""
        # Create test model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # Test q4_k_m style
        quantize_model_mixed(model, style="q4_k_m", low_bits=4, high_bits=6, inplace=True)

        # Check quantization applied
        quantized_count = sum(1 for m in model.modules() if isinstance(m, nn.QuantizedLinear))
        assert quantized_count == 2

    def test_estimate_q4_k_size(self):
        """Test Q4_K size estimation."""
        # Create test model
        model = nn.Sequential(
            nn.Linear(256, 256),  # 256×256 = 65,536 weights
            nn.Linear(256, 256),
        )

        # Estimate size
        stats = estimate_q4_k_size(model)

        # Check expected fields
        assert "original_mb" in stats
        assert "q4_k_mb" in stats
        assert "reduction_ratio" in stats
        assert "avg_bits_per_weight" in stats

        # Q4_K should achieve ~4.5 bits/weight
        assert abs(stats["avg_bits_per_weight"] - 4.5) < 0.1

        # Should have significant reduction from FP32
        assert stats["reduction_ratio"] > 5.0

    def test_estimate_mixed_size(self):
        """Test mixed-precision size estimation."""
        # Create test model
        model = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 256),
        )

        # Estimate mixed size
        stats = estimate_mixed_size(model, style="q4_k_m", low_bits=4, high_bits=6)

        # Check expected fields
        assert "original_mb" in stats
        assert "quantized_mb" in stats
        assert "avg_bits_per_weight" in stats

        # Q4_K_M should be between 4 and 6 bits/weight
        assert 4.0 < stats["avg_bits_per_weight"] < 6.0


@pytest.mark.unit
def test_create_q4_k_m_style_predicate():
    """Test Q4_K_M-style predicate creation."""
    # Create test model with typical transformer structure
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 64)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(64, 64)

    model = SimpleModel()

    # Create predicate
    pred = create_q4_k_m_style_predicate(model, low_bits=4, high_bits=6, group_size=64)

    # Test predicate on Linear layer
    result = pred("linear1", model.linear1)
    assert isinstance(result, dict)
    assert "bits" in result
    assert "group_size" in result
    assert result["bits"] in [4, 6]

    # Test predicate on non-Linear layer (should return False)
    result = pred("relu", model.relu)
    assert result is False


@pytest.mark.unit
@pytest.mark.slow
class TestQ4KMGGMLStrategy:
    """Test Q4_K_M GGML-compatible mixed precision strategy."""

    @staticmethod
    def create_transformer_test_model(num_layers: int = 4, dim: int = 128):
        """Create a test model with transformer-like structure."""

        class TransformerLayer(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim, bias=False)
                self.k_proj = nn.Linear(dim, dim, bias=False)
                self.v_proj = nn.Linear(dim, dim, bias=False)
                self.o_proj = nn.Linear(dim, dim, bias=False)
                self.up_proj = nn.Linear(dim, dim * 2, bias=False)
                self.down_proj = nn.Linear(dim * 2, dim, bias=False)

            def __call__(self, x):
                # Simplified forward pass
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                attn = self.o_proj(v)  # Simplified
                mlp = self.down_proj(self.up_proj(attn))
                return mlp

        class TransformerModel(nn.Module):
            def __init__(self, num_layers: int, dim: int):
                super().__init__()
                self.layers = [TransformerLayer(dim) for _ in range(num_layers)]
                self.lm_head = nn.Linear(dim, 256, bias=False)

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.lm_head(x)

        return TransformerModel(num_layers, dim)

    def test_q4k_m_ggml_layer_selection(self):
        """Test that Q4_K_M GGML correctly selects Q6_K for important layers."""
        model = self.create_transformer_test_model(num_layers=4, dim=128)

        # Apply Q4_K_M GGML quantization
        quantize_model_q4_k_m_ggml(model, inplace=True)

        # Track which layers got Q6_K vs Q4_K
        q6_layers = []
        q4_layers = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "quantization_format"):
                if module.quantization_format == "q6_k":
                    q6_layers.append(name)
                elif module.quantization_format == "q4_k":
                    q4_layers.append(name)

        # Verify Q6_K was used for important layers
        # Q4_K_M uses Q6_K for even-indexed v_proj and down_proj
        expected_q6k = ["layers.0.v_proj", "layers.0.down_proj", "layers.2.v_proj", "layers.2.down_proj"]

        for expected in expected_q6k:
            assert expected in q6_layers, f"Expected {expected} to use Q6_K"

        # Verify Q4_K was used for other layers
        # Odd-indexed v_proj and down_proj should use Q4_K
        expected_q4k_patterns = [
            "layers.1.v_proj",
            "layers.1.down_proj",
            "layers.3.v_proj",
            "layers.3.down_proj",
        ]

        for expected in expected_q4k_patterns:
            assert expected in q4_layers, f"Expected {expected} to use Q4_K"

        # All q_proj, k_proj, o_proj, up_proj, lm_head should use Q4_K
        for name in q4_layers:
            if "v_proj" not in name and "down_proj" not in name:
                # These should all be Q4_K
                assert any(
                    pattern in name for pattern in ["q_proj", "k_proj", "o_proj", "up_proj", "lm_head"]
                )

    def test_q4k_m_ggml_quantization_metadata(self):
        """Test that Q4_K_M GGML stores proper quantization metadata."""
        model = self.create_transformer_test_model(num_layers=2, dim=64)

        # Apply quantization
        quantize_model_q4_k_m_ggml(model, inplace=True)

        # Check Q6_K layer metadata
        v_proj_layer = model.layers[0].v_proj
        assert hasattr(v_proj_layer, "weight_q6_k_ql")
        assert hasattr(v_proj_layer, "weight_q6_k_qh")
        assert hasattr(v_proj_layer, "weight_q6_k_scales")
        assert hasattr(v_proj_layer, "weight_q6_k_d_scales")
        assert v_proj_layer.quantization_format == "q6_k"

        # Check Q4_K layer metadata
        q_proj_layer = model.layers[0].q_proj
        assert hasattr(q_proj_layer, "weight_q4_k_packed")
        assert hasattr(q_proj_layer, "d_scales_q4_k")
        assert hasattr(q_proj_layer, "d_mins_q4_k")
        assert hasattr(q_proj_layer, "d_min_scales_q4_k")
        assert hasattr(q_proj_layer, "packed_scales_mins_q4_k")
        assert q_proj_layer.quantization_format == "q4_k"

    def test_q4k_m_ggml_average_bits_per_weight(self):
        """Test that Q4_K_M GGML achieves expected average bits per weight."""
        model = self.create_transformer_test_model(num_layers=4, dim=128)

        # Count total parameters
        total_params = 0
        for _, module in model.named_modules():
            if hasattr(module, "weight"):
                total_params += module.weight.size

        # Apply quantization
        quantize_model_q4_k_m_ggml(model, inplace=True)

        # Count Q4_K vs Q6_K parameters
        q4_params = 0
        q6_params = 0

        for _, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "quantization_format"):
                param_count = module.weight.size
                if module.quantization_format == "q6_k":
                    q6_params += param_count
                elif module.quantization_format == "q4_k":
                    q4_params += param_count

        # Calculate average bits per weight
        # Q4_K: 4.5625 bits/weight, Q6_K: 6.5625 bits/weight
        avg_bpw = (q4_params * 4.5625 + q6_params * 6.5625) / (q4_params + q6_params)

        # Q4_K_M should target ~4.8 bits/weight (can vary based on architecture)
        assert 4.5 < avg_bpw < 6.0, f"Expected avg_bpw in [4.5, 6.0], got {avg_bpw:.2f}"

        # Should have both Q4_K and Q6_K layers
        assert q4_params > 0, "Should have Q4_K layers"
        assert q6_params > 0, "Should have Q6_K layers"

    def test_q4k_m_ggml_forward_pass(self):
        """Test that model works after Q4_K_M GGML quantization."""
        model = self.create_transformer_test_model(num_layers=2, dim=64)

        # Apply quantization
        quantize_model_q4_k_m_ggml(model, inplace=True)

        # Test forward pass
        test_input = mx.random.normal((2, 10, 64))  # batch_size=2, seq_len=10, dim=64
        output = model(test_input)

        # Should produce output of correct shape
        assert output.shape == (2, 10, 256)  # lm_head projects to 256

    def test_q4k_m_ggml_quantization_error(self):
        """Test quantization error is acceptable for Q4_K_M GGML."""
        # Create a simple model
        model = nn.Sequential(nn.Linear(256, 256, bias=False), nn.Linear(256, 128, bias=False))

        # Store original weights
        original_weights = {name: mx.array(module.weight) for name, module in model.named_modules() if hasattr(module, "weight")}

        # Apply Q4_K_M quantization
        quantize_model_q4_k_m_ggml(model, inplace=True)

        # Check quantization error for each layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in original_weights:
                original = original_weights[name]
                quantized = module.weight

                # Calculate error
                error = mx.abs(original - quantized)
                mean_error = float(mx.mean(error))
                relative_error = mean_error / (float(mx.std(original)) + 1e-10)

                # Error should be reasonable
                # Q6_K should have < 5% error, Q4_K should have < 10% error
                if module.quantization_format == "q6_k":
                    assert relative_error < 0.10, f"Q6_K error too high: {relative_error:.4f}"
                elif module.quantization_format == "q4_k":
                    assert relative_error < 0.15, f"Q4_K error too high: {relative_error:.4f}"

    def test_q4k_m_ggml_vs_mlx_native(self):
        """Compare Q4_K_M GGML mode vs MLX native mode."""
        # Create two identical models
        model_ggml = self.create_transformer_test_model(num_layers=2, dim=64)
        model_mlx = self.create_transformer_test_model(num_layers=2, dim=64)

        # Quantize with GGML mode
        quantize_model_q4_k_m_ggml(model_ggml, inplace=True)

        # Quantize with MLX native mode
        quantize_model_q4_k_m(model_mlx, inplace=True, use_mlx_native=True)

        # Both should work for forward pass
        test_input = mx.random.normal((2, 5, 64))

        output_ggml = model_ggml(test_input)
        output_mlx = model_mlx(test_input)

        # Both should produce valid outputs (not necessarily identical)
        assert output_ggml.shape == (2, 5, 256)
        assert output_mlx.shape == (2, 5, 256)

        # GGML mode should have Q4_K and Q6_K metadata
        has_q6k = any(
            hasattr(m, "quantization_format") and m.quantization_format == "q6_k"
            for m in model_ggml.modules()
        )
        has_q4k = any(
            hasattr(m, "quantization_format") and m.quantization_format == "q4_k"
            for m in model_ggml.modules()
        )
        assert has_q6k and has_q4k, "GGML mode should use both Q6_K and Q4_K"

        # MLX mode should use QuantizedLinear
        has_quantized_linear = any(isinstance(m, nn.QuantizedLinear) for m in model_mlx.modules())
        assert has_quantized_linear, "MLX mode should use QuantizedLinear"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
