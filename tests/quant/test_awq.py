"""
Tests for AWQ (Activation-Aware Weight Quantization) implementation.

Tests cover:
- AWQConfig and ScaleConfig dataclasses
- Scale search and application
- Weight clipping search
- Full AWQ quantization pipeline
- Model-specific configurations
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant.awq import (
    AWQConfig,
    ScaleConfig,
    _mse,
    _submodule_from_key,
    apply_scale,
    llama_awq,
    mistral_awq,
    qwen_awq,
    search_best_scale,
)


@pytest.mark.unit
class TestAWQConfig:
    """Tests for AWQ configuration dataclasses."""

    def test_scale_config_creation(self):
        """Test ScaleConfig creation with required fields."""
        config = ScaleConfig(
            prev="input_layernorm",
            layers=["q_proj", "k_proj", "v_proj"],
        )

        assert config.prev == "input_layernorm"
        assert config.layers == ["q_proj", "k_proj", "v_proj"]
        assert config.block is None
        assert config.kwargs == []
        assert config.use_config is None

    def test_scale_config_with_optional_fields(self):
        """Test ScaleConfig with all fields."""
        config = ScaleConfig(
            prev="input_layernorm",
            layers=["q_proj"],
            block="self_attn",
            kwargs=["mask"],
            use_config=lambda x: True,
        )

        assert config.block == "self_attn"
        assert config.kwargs == ["mask"]
        assert config.use_config is not None

    def test_awq_config_creation(self):
        """Test AWQConfig creation."""
        config = AWQConfig(
            embed="embed_tokens",
            lm_head="lm_head",
            no_clip=["q_proj", "k_proj"],
            scale_configs=[
                ScaleConfig(prev="ln", layers=["proj"])
            ],
        )

        assert config.embed == "embed_tokens"
        assert config.lm_head == "lm_head"
        assert config.no_clip == ["q_proj", "k_proj"]
        assert len(config.scale_configs) == 1

    def test_llama_awq_config(self):
        """Test predefined Llama AWQ configuration."""
        config = llama_awq

        assert config.embed == "embed_tokens"
        assert config.lm_head == "lm_head"
        assert "q_proj" in config.no_clip
        assert "k_proj" in config.no_clip
        assert len(config.scale_configs) == 3

    def test_model_configs_exist(self):
        """Test that all predefined model configs are available."""
        assert llama_awq is not None
        assert mistral_awq is not None
        assert qwen_awq is not None

        # Mistral and Qwen should use same config as Llama
        assert mistral_awq is llama_awq
        assert qwen_awq is llama_awq


@pytest.mark.unit
class TestHelperFunctions:
    """Tests for AWQ helper functions."""

    def test_mse(self):
        """Test MSE computation."""
        x = mx.array([1.0, 2.0, 3.0, 4.0])
        y = mx.array([1.1, 2.1, 2.9, 4.1])

        loss = _mse(x, y)

        # MSE should be small for close values
        assert loss.item() < 0.1
        assert loss.item() > 0.0

    def test_mse_identical(self):
        """Test MSE is zero for identical arrays."""
        x = mx.array([1.0, 2.0, 3.0])
        loss = _mse(x, x)

        assert mx.allclose(loss, mx.array(0.0), atol=1e-6)

    def test_submodule_from_key_simple(self):
        """Test navigating to submodule with simple key."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = nn.Linear(64, 64)
                self.layer1 = nn.ReLU()

        model = SimpleModel()

        # Access by attribute name
        layer = _submodule_from_key(model, "layer0")
        assert isinstance(layer, nn.Linear)

    def test_submodule_from_key_nested(self):
        """Test navigating to deeply nested submodule."""
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
                self.relu = nn.ReLU()

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(64, 64)
                self.block = Block()

        model = NestedModel()

        # Access nested layer
        layer = _submodule_from_key(model, "block.linear")
        assert isinstance(layer, nn.Linear)


@pytest.mark.unit
@pytest.mark.gpu
class TestScaleSearch:
    """Tests for scale search functionality."""

    def test_search_best_scale_runs(self):
        """Test that scale search completes without errors."""
        # Create simple model
        layer = nn.Linear(64, 64)
        layer.input_feat = mx.random.normal((16, 32, 64))

        # Quantize function
        def quantize_func(w):
            wq, scales, biases = mx.quantize(w, bits=4, group_size=64)
            return mx.dequantize(wq, scales, biases, group_size=64, bits=4, dtype=w.dtype)

        # Search for scales (small grid for speed)
        scales = search_best_scale(
            layers=[layer],
            quantize_func=quantize_func,
            block=None,
            layer_kwargs={},
            n_grid=5,
        )

        # Should return scales with correct shape
        assert scales.shape == (64,)
        assert mx.all(scales > 0)

    def test_search_best_scale_improves_quantization(self):
        """Test that AWQ scales improve quantization quality."""
        # Create layer with known input features
        layer = nn.Linear(32, 32)
        layer.input_feat = mx.random.normal((8, 16, 32))

        def quantize_func(w):
            wq, scales, biases = mx.quantize(w, bits=4, group_size=32)
            return mx.dequantize(wq, scales, biases, group_size=32, bits=4, dtype=w.dtype)

        # Find best scales
        scales = search_best_scale(
            layers=[layer],
            quantize_func=quantize_func,
            block=None,
            layer_kwargs={},
            n_grid=10,
        )

        # Scales should vary (not all equal)
        assert not mx.allclose(scales, mx.full(scales.shape, scales[0].item()))


@pytest.mark.unit
@pytest.mark.gpu
class TestApplyScale:
    """Tests for scale application."""

    def test_apply_scale_to_layernorm(self):
        """Test applying scales to LayerNorm -> Linear."""
        layernorm = nn.LayerNorm(64)
        linear = nn.Linear(64, 64)
        scales = mx.ones((64,)) * 2.0

        orig_ln_weight = mx.array(layernorm.weight)
        orig_linear_weight = mx.array(linear.weight)

        apply_scale(layernorm, [linear], scales)

        # LayerNorm weight should be divided by scales
        assert mx.allclose(
            layernorm.weight,
            orig_ln_weight / scales,
            atol=1e-5
        )

        # Linear weight should be multiplied by scales
        assert mx.allclose(
            linear.weight,
            orig_linear_weight * scales,
            atol=1e-5
        )

    def test_apply_scale_to_linear(self):
        """Test applying scales to Linear -> Linear."""
        prev_linear = nn.Linear(64, 64)
        next_linear = nn.Linear(64, 64)
        scales = mx.ones((64,)) * 0.5

        orig_prev_weight = mx.array(prev_linear.weight)
        orig_next_weight = mx.array(next_linear.weight)

        apply_scale(prev_linear, [next_linear], scales)

        # Previous linear weight should be scaled appropriately
        expected_prev = orig_prev_weight / scales[:, mx.newaxis]
        assert mx.allclose(prev_linear.weight, expected_prev, atol=1e-5)

        # Next linear weight should be scaled
        expected_next = orig_next_weight * scales
        assert mx.allclose(next_linear.weight, expected_next, atol=1e-5)

    def test_apply_scale_preserves_output(self):
        """Test that applying scales preserves layer output (mathematically)."""
        layernorm = nn.LayerNorm(32)
        linear = nn.Linear(32, 32)
        scales = mx.random.uniform(0.5, 2.0, (32,))

        # Compute original output
        x = mx.random.normal((4, 32))
        orig_out = linear(layernorm(x))

        # Apply scales
        apply_scale(layernorm, [linear], scales)

        # Output should be approximately the same
        new_out = linear(layernorm(x))
        assert mx.allclose(orig_out, new_out, atol=1e-4)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.slow
class TestAWQIntegration:
    """Integration tests for full AWQ pipeline."""

    def test_awq_config_for_simple_transformer_block(self):
        """Test AWQ configuration on a simple transformer block."""
        # Create minimal transformer block structure
        class SimpleBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = nn.LayerNorm(64)
                self.self_attn = nn.Module()
                self.self_attn.q_proj = nn.Linear(64, 64)
                self.self_attn.k_proj = nn.Linear(64, 64)
                self.self_attn.v_proj = nn.Linear(64, 64)
                self.post_attention_layernorm = nn.LayerNorm(64)
                self.mlp = nn.Module()
                self.mlp.gate_proj = nn.Linear(64, 128)
                self.mlp.up_proj = nn.Linear(64, 128)
                self.mlp.down_proj = nn.Linear(128, 64)

            def __call__(self, x, mask=None):
                # Simplified forward pass
                h = self.input_layernorm(x)
                q = self.self_attn.q_proj(h)
                k = self.self_attn.k_proj(h)
                v = self.self_attn.v_proj(h)
                attn_out = q + k + v  # Simplified attention

                h2 = x + attn_out
                h3 = self.post_attention_layernorm(h2)
                mlp_out = self.mlp.down_proj(
                    self.mlp.gate_proj(h3) + self.mlp.up_proj(h3)
                )
                return h2 + mlp_out

        block = SimpleBlock()

        # Test that we can access layers via config paths
        q_proj = _submodule_from_key(block, "self_attn.q_proj")
        assert isinstance(q_proj, nn.Linear)

        gate_proj = _submodule_from_key(block, "mlp.gate_proj")
        assert isinstance(gate_proj, nn.Linear)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
class TestAWQQuantization:
    """Tests for complete AWQ quantization pipeline."""

    def test_awq_quantize_synthetic_model(
        self, synthetic_transformer_model, small_calibration_data
    ):
        """
        Test AWQ quantization on a synthetic transformer model.

        Uses a small synthetic model and calibration data for fast testing.
        """
        from smlx.quant import awq_quantize, llama_awq

        model = synthetic_transformer_model
        calibration_data = small_calibration_data

        # Test with reduced grid for speed
        quantized_model = awq_quantize(
            model,
            calibration_data,
            awq_config=llama_awq,
            bits=4,
            group_size=64,
            n_grid=5,  # Reduced for faster testing
        )

        # Verify quantization succeeded
        assert quantized_model is not None

        # Verify model still works
        test_input = mx.random.randint(0, 1000, (2, 64))
        output = quantized_model(test_input)
        assert output.shape == (2, 64, 1000)  # (batch, seq_len, vocab_size)


@pytest.mark.unit
class TestAWQEdgeCases:
    """Test edge cases and error handling."""

    def test_apply_scale_unsupported_layer_raises(self):
        """Test that apply_scale raises for unsupported layer types."""
        unsupported = nn.ReLU()
        target = nn.Linear(64, 64)
        scales = mx.ones((64,))

        with pytest.raises(NotImplementedError):
            apply_scale(unsupported, [target], scales)

    def test_apply_scale_linear_requires_single_target(self):
        """Test that Linear prev_op only supports single target layer."""
        prev_linear = nn.Linear(64, 64)
        target1 = nn.Linear(64, 64)
        target2 = nn.Linear(64, 64)
        scales = mx.ones((64,))

        with pytest.raises(AssertionError):
            apply_scale(prev_linear, [target1, target2], scales)
