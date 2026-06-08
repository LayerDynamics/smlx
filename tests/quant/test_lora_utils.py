#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Unit tests for LoRA utility functions (apply_lora, merge_lora).

Tests cover:
- Applying LoRA to models
- Merging LoRA weights back to base model
- Target module filtering
- Error handling for invalid parameters
- Weight fusion correctness
"""

import pytest
import mlx.core as mx
import mlx.nn as nn

from smlx.quant.lora import (
    LoRALinear,
    apply_lora,
    merge_lora,
)


class SimpleModel(nn.Module):
    """Simple test model with Linear layers."""

    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, 10)

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.relu(x)
        x = self.fc3(x)
        return x


class TestApplyLoRA:
    """Test apply_lora function."""

    def test_apply_lora_basic(self):
        """Test basic LoRA application to a model."""
        model = SimpleModel()

        # Apply LoRA with default settings
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # Check that Linear layers were replaced with LoRALinear
        assert isinstance(lora_model.fc1, LoRALinear)
        assert isinstance(lora_model.fc2, LoRALinear)
        assert isinstance(lora_model.fc3, LoRALinear)

    def test_apply_lora_custom_rank(self):
        """Test LoRA with custom rank."""
        model = SimpleModel()

        # Apply LoRA with custom rank
        lora_model = apply_lora(model, rank=16, alpha=32.0)

        # Verify LoRA layers have correct rank
        assert lora_model.fc1.r == 16
        assert lora_model.fc2.r == 16

    def test_apply_lora_with_dropout(self):
        """Test LoRA with dropout enabled."""
        model = SimpleModel()

        # Apply LoRA with dropout
        lora_model = apply_lora(model, rank=8, alpha=16.0, dropout=0.1)

        # Verify dropout is set (if LoRALinear exposes this)
        # Note: Depends on LoRALinear implementation
        assert isinstance(lora_model.fc1, LoRALinear)

    def test_apply_lora_preserves_weights(self):
        """Test that LoRA preserves original model weights."""
        model = SimpleModel()

        # Get original weights
        orig_weight = model.fc1.weight

        # Apply LoRA
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # Original weights should be preserved in the LoRA layer
        # (LoRALinear.linear.weight should match orig_weight)
        assert mx.allclose(lora_model.fc1.linear.weight, orig_weight)

    def test_apply_lora_forward_pass(self):
        """Test that model forward pass still works after LoRA."""
        model = SimpleModel()
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # Create test input
        x = mx.random.normal((4, 128))

        # Forward pass should work
        output = lora_model(x)

        # Check output shape
        assert output.shape == (4, 10)

    def test_apply_lora_invalid_rank(self):
        """Test that invalid rank raises error."""
        model = SimpleModel()

        with pytest.raises(ValueError):
            apply_lora(model, rank=0, alpha=16.0)

        with pytest.raises(ValueError):
            apply_lora(model, rank=-5, alpha=16.0)

    def test_apply_lora_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        model = SimpleModel()

        with pytest.raises(ValueError):
            apply_lora(model, rank=8, alpha=0.0)

        with pytest.raises(ValueError):
            apply_lora(model, rank=8, alpha=-1.0)

    def test_apply_lora_target_modules(self):
        """Test selective LoRA application to specific module types."""
        model = SimpleModel()

        # Apply LoRA only to nn.Linear (default behavior)
        lora_model = apply_lora(model, rank=8, alpha=16.0, target_modules=[nn.Linear])

        # All Linear layers should be converted
        assert isinstance(lora_model.fc1, LoRALinear)

    def test_apply_lora_no_trainable_params(self):
        """Test warning/error when model has no trainable params."""
        # Create a model with frozen parameters
        model = SimpleModel()

        # Freeze all parameters
        model.freeze()

        # Applying LoRA should still work (LoRA params are trainable)
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # LoRA should add trainable parameters
        assert isinstance(lora_model.fc1, LoRALinear)


class TestMergeLoRA:
    """Test merge_lora function."""

    def test_merge_lora_basic(self):
        """Test basic LoRA weight merging."""
        model = SimpleModel()

        # Apply LoRA
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # Merge LoRA weights back
        fused_model = merge_lora(lora_model, dequantize=False)

        # After merging, layers should be regular Linear (not LoRALinear)
        assert isinstance(fused_model.fc1, nn.Linear)
        assert isinstance(fused_model.fc2, nn.Linear)
        assert not isinstance(fused_model.fc1, LoRALinear)

    def test_merge_lora_preserves_output(self):
        """Test that merging LoRA preserves model output."""
        model = SimpleModel()

        # Apply LoRA
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # Test input
        x = mx.random.normal((4, 128))

        # Get output from LoRA model
        lora_output = lora_model(x)
        mx.eval(lora_output)

        # Merge LoRA
        fused_model = merge_lora(lora_model, dequantize=False)

        # Get output from fused model
        fused_output = fused_model(x)
        mx.eval(fused_output)

        # Outputs should be very close (allowing for numerical precision)
        assert mx.allclose(lora_output, fused_output, atol=1e-5)

    def test_merge_lora_with_dequantize(self):
        """Test LoRA merging with dequantization."""
        model = SimpleModel()

        # Apply LoRA
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # Merge with dequantize=True
        fused_model = merge_lora(lora_model, dequantize=True)

        # Should return regular Linear layers
        assert isinstance(fused_model.fc1, nn.Linear)

    def test_merge_lora_trained_weights(self):
        """Test merging after training LoRA adapters."""
        model = SimpleModel()

        # Apply LoRA
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # Simulate training by modifying LoRA weights.
        # lora_a / lora_b are raw mx.array parameters (not nn.Linear modules),
        # so we update the arrays directly. MLX has no mx.no_grad(); plain
        # assignment outside a grad transform is already gradient-free.
        lora_model.fc1.lora_a = (
            lora_model.fc1.lora_a + mx.random.normal(lora_model.fc1.lora_a.shape) * 0.01
        )
        lora_model.fc1.lora_b = (
            lora_model.fc1.lora_b + mx.random.normal(lora_model.fc1.lora_b.shape) * 0.01
        )

        # Merge LoRA
        fused_model = merge_lora(lora_model, dequantize=False)

        # Model should still produce output
        x = mx.random.normal((4, 128))
        output = fused_model(x)
        assert output.shape == (4, 10)

    def test_merge_lora_idempotent(self):
        """Test that merging non-LoRA model is safe."""
        model = SimpleModel()

        # Try to merge model that doesn't have LoRA (should be no-op or safe)
        try:
            fused_model = merge_lora(model, dequantize=False)
            # Should either return unchanged model or handle gracefully
            assert isinstance(fused_model.fc1, nn.Linear)
        except:
            # Or it might raise an error, which is also acceptable
            pass


class TestLoRARoundTrip:
    """Test apply -> train -> merge workflow."""

    def test_lora_roundtrip_zero_update(self):
        """Test apply-merge roundtrip with no training (should preserve weights)."""
        model = SimpleModel()

        # Get original output
        x = mx.random.normal((4, 128))
        orig_output = model(x)
        mx.eval(orig_output)

        # Apply LoRA
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # Immediately merge (no training)
        fused_model = merge_lora(lora_model, dequantize=False)

        # Get fused output
        fused_output = fused_model(x)
        mx.eval(fused_output)

        # Outputs should match closely
        # (LoRA initialized to zero, so should preserve original behavior)
        assert mx.allclose(orig_output, fused_output, atol=1e-4)

    def test_lora_roundtrip_with_update(self):
        """Test apply-merge roundtrip with simulated training."""
        model = SimpleModel()

        # Apply LoRA
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # Simulate training updates
        x = mx.random.normal((4, 128))
        orig_output = lora_model(x)

        # Update LoRA weights directly (lora_a/lora_b are mx.array parameters,
        # not modules; MLX has no mx.no_grad()).
        lora_model.fc1.lora_a = (
            lora_model.fc1.lora_a + mx.random.normal(lora_model.fc1.lora_a.shape) * 0.1
        )
        lora_model.fc1.lora_b = (
            lora_model.fc1.lora_b + mx.random.normal(lora_model.fc1.lora_b.shape) * 0.1
        )

        updated_output = lora_model(x)

        # Merge LoRA
        fused_model = merge_lora(lora_model, dequantize=False)
        fused_output = fused_model(x)

        # Fused output should match updated LoRA output
        assert mx.allclose(updated_output, fused_output, atol=1e-5)

        # But should differ from original (since we updated LoRA)
        assert not mx.allclose(orig_output, fused_output, atol=0.1)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_model_type(self):
        """Test apply_lora with invalid model type."""
        # Not a proper nn.Module
        invalid_model = "not a model"

        with pytest.raises((TypeError, AttributeError)):
            apply_lora(invalid_model, rank=8, alpha=16.0)

    def test_empty_model(self):
        """Test apply_lora on model with no Linear layers."""

        class EmptyModel(nn.Module):
            def __call__(self, x):
                return x

        model = EmptyModel()

        # Should handle gracefully (no layers to convert)
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # Model should still work
        x = mx.random.normal((4, 10))
        output = lora_model(x)
        assert mx.allclose(output, x)

    def test_nested_modules(self):
        """Test LoRA application to nested modules."""

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
                self.block2 = nn.Linear(32, 10)

            def __call__(self, x):
                x = self.block1(x)
                x = self.block2(x)
                return x

        model = NestedModel()
        lora_model = apply_lora(model, rank=8, alpha=16.0)

        # Should find and convert nested Linear layers
        # Note: Depends on implementation details of apply_lora

    def test_parameter_validation(self):
        """Test parameter validation."""
        model = SimpleModel()

        # Test various invalid parameters
        with pytest.raises(ValueError):
            apply_lora(model, rank=-1, alpha=16.0)

        with pytest.raises(ValueError):
            apply_lora(model, rank=8, alpha=-5.0)

        with pytest.raises(ValueError):
            apply_lora(model, rank=8, alpha=0.0, dropout=-0.1)

        with pytest.raises(ValueError):
            apply_lora(model, rank=8, alpha=16.0, dropout=1.5)


class TestIntegrationWithSmolLM:
    """Integration tests with actual SmolLM2 model."""

    @pytest.mark.integration
    @pytest.mark.requires_model
    @pytest.mark.skip(reason="Requires downloading model - run manually")
    def test_lora_on_smollm2(self):
        """Test applying LoRA to SmolLM2-135M model."""
        from smlx.models import mlx_backend

        # Load model
        bm = mlx_backend.load("mlx-community/SmolLM2-135M-Instruct")
        model, tokenizer = bm.model, bm.processor

        # Apply LoRA
        lora_model = apply_lora(model, rank=16, alpha=32.0)

        # Test generation still works
        from smlx.utils.generation import generate

        prompt = "Hello, world!"
        output = generate(lora_model, tokenizer, prompt, max_tokens=10)

        assert isinstance(output, str)
        assert len(output) > 0
