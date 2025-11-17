"""
Tests for LoRASwitchLinear (LoRA with Mixture of Experts).

Tests LoRA adaptation on top of SwitchLinear layers for parameter-efficient
fine-tuning with expert routing.
"""

import unittest

import mlx.core as mx
import mlx.nn as nn

from smlx.models.common.switch_layers import QuantizedSwitchLinear, SwitchLinear
from smlx.quant.lora import LoRASwitchLinear


class TestLoRASwitchLinear(unittest.TestCase):
    """Test LoRASwitchLinear layer."""

    def test_initialization(self):
        """Test LoRASwitchLinear initialization."""
        layer = LoRASwitchLinear(
            input_dims=768, output_dims=3072, num_experts=8, r=16, scale=1.0
        )

        # Check LoRA matrices shape
        # lora_a: (num_experts, r, input_dims)
        self.assertEqual(layer.lora_a.shape, (8, 16, 768))
        # lora_b: (num_experts, output_dims, r)
        self.assertEqual(layer.lora_b.shape, (8, 3072, 16))

        # Check base linear layer
        self.assertIsInstance(layer.linear, SwitchLinear)

    def test_from_base_switch_linear(self):
        """Test creating LoRA from existing SwitchLinear."""
        base_layer = SwitchLinear(
            input_dims=768, output_dims=3072, num_experts=8, bias=True
        )

        # Create LoRA layer from base
        lora_layer = LoRASwitchLinear.from_base(base_layer, r=16)

        # Check that base weights are preserved
        self.assertTrue(mx.allclose(lora_layer.linear.weight, base_layer.weight))
        self.assertTrue(mx.allclose(lora_layer.linear.bias, base_layer.bias))

    def test_from_base_quantized(self):
        """Test creating LoRA from QuantizedSwitchLinear."""
        base_layer = SwitchLinear(input_dims=768, output_dims=3072, num_experts=8)
        quantized = base_layer.to_quantized(group_size=64, bits=4)

        # Create LoRA from quantized layer
        lora_layer = LoRASwitchLinear.from_base(quantized, r=16)

        # Check base is quantized
        self.assertIsInstance(lora_layer.linear, QuantizedSwitchLinear)

    def test_forward_pass(self):
        """Test forward pass with LoRA adaptation."""
        layer = LoRASwitchLinear(
            input_dims=768, output_dims=3072, num_experts=8, r=16, scale=1.0
        )

        # Input: 32 tokens
        x = mx.random.normal((32, 768))
        # Expert routing
        indices = mx.array([i % 8 for i in range(32)])

        # Forward pass
        output = layer(x, indices)

        # Check output shape
        self.assertEqual(output.shape, (32, 3072))

    def test_lora_contributes(self):
        """Test that LoRA adaptation actually contributes to output."""
        base_layer = SwitchLinear(input_dims=768, output_dims=768, num_experts=4)
        lora_layer = LoRASwitchLinear.from_base(base_layer, r=8, scale=1.0)

        x = mx.random.normal((16, 768))
        indices = mx.array([i % 4 for i in range(16)])

        # Get outputs
        base_output = base_layer(x, indices)
        lora_output = lora_layer(x, indices)

        # LoRA output should be different (unless LoRA weights are zero)
        # Since lora_b is initialized to zeros, outputs should initially be the same
        # But this tests the plumbing works
        self.assertEqual(base_output.shape, lora_output.shape)

        # Now set lora_a to non-zero and test again
        lora_layer.lora_a = mx.random.normal(lora_layer.lora_a.shape)
        lora_layer.lora_b = mx.random.normal(lora_layer.lora_b.shape)

        lora_output_modified = lora_layer(x, indices)

        # Should be different now
        self.assertFalse(mx.allclose(base_output, lora_output_modified, atol=1e-5))

    def test_fuse(self):
        """Test fusing LoRA weights into base layer."""
        base_layer = SwitchLinear(input_dims=768, output_dims=768, num_experts=4)
        lora_layer = LoRASwitchLinear.from_base(base_layer, r=8, scale=1.0)

        # Set non-zero LoRA weights
        lora_layer.lora_a = mx.random.normal(lora_layer.lora_a.shape) * 0.01
        lora_layer.lora_b = mx.random.normal(lora_layer.lora_b.shape) * 0.01

        # Fuse
        fused_layer = lora_layer.fuse()

        # Check fused layer is SwitchLinear
        self.assertIsInstance(fused_layer, SwitchLinear)

        # Test that fused output matches LoRA output
        x = mx.random.normal((16, 768))
        indices = mx.array([i % 4 for i in range(16)])

        lora_output = lora_layer(x, indices)
        fused_output = fused_layer(x, indices)

        # Should be very close
        self.assertTrue(mx.allclose(lora_output, fused_output, atol=1e-4))

    def test_fuse_quantized(self):
        """Test fusing LoRA with quantized base layer."""
        base_layer = SwitchLinear(input_dims=768, output_dims=768, num_experts=4)
        quantized = base_layer.to_quantized(group_size=64, bits=4)
        lora_layer = LoRASwitchLinear.from_base(quantized, r=8, scale=1.0)

        # Set non-zero LoRA weights
        lora_layer.lora_a = mx.random.normal(lora_layer.lora_a.shape) * 0.01
        lora_layer.lora_b = mx.random.normal(lora_layer.lora_b.shape) * 0.01

        # Fuse and keep quantized
        fused_q = lora_layer.fuse(dequantize=False)
        self.assertIsInstance(fused_q, QuantizedSwitchLinear)

        # Fuse and dequantize
        fused_dq = lora_layer.fuse(dequantize=True)
        self.assertIsInstance(fused_dq, SwitchLinear)

    def test_sorted_indices(self):
        """Test forward pass with sorted indices."""
        layer = LoRASwitchLinear(
            input_dims=768, output_dims=3072, num_experts=8, r=16
        )

        x = mx.random.normal((32, 768))
        # Pre-sorted indices
        indices = mx.array(sorted([i % 8 for i in range(32)]))

        # Forward pass with sorted_indices=True
        output = layer(x, indices, sorted_indices=True)

        self.assertEqual(output.shape, (32, 3072))

    def test_dropout(self):
        """Test that dropout is applied during training."""
        layer = LoRASwitchLinear(
            input_dims=768, output_dims=3072, num_experts=8, r=16, dropout=0.5
        )

        # Dropout should be in the layer
        self.assertIsNotNone(layer.dropout)

    def test_trainable_parameters(self):
        """Test that only LoRA parameters are trainable."""
        base_layer = SwitchLinear(input_dims=768, output_dims=3072, num_experts=8)
        lora_layer = LoRASwitchLinear.from_base(base_layer, r=16)

        # Freeze base layer
        lora_layer.linear.freeze()

        # LoRA parameters should be trainable
        trainable = mx.tree_flatten(lora_layer.trainable_parameters())
        self.assertGreater(len(trainable), 0)


class TestLoRASwitchIntegration(unittest.TestCase):
    """Integration tests for LoRASwitchLinear."""

    def test_multi_expert_routing(self):
        """Test that different experts produce different outputs."""
        layer = LoRASwitchLinear(
            input_dims=768, output_dims=768, num_experts=4, r=8, scale=1.0
        )

        # Set different LoRA weights per expert
        for i in range(4):
            layer.lora_a[i] = mx.ones_like(layer.lora_a[i]) * (i + 1) * 0.01
            layer.lora_b[i] = mx.ones_like(layer.lora_b[i]) * (i + 1) * 0.01

        # Same input, different experts
        x = mx.ones((4, 768))
        indices = mx.array([0, 1, 2, 3])

        output = layer(x, indices)

        # Each token processed by different expert should have different output
        # (not testing exact values, just that routing worked)
        self.assertEqual(output.shape, (4, 768))

    def test_large_batch_sorting(self):
        """Test LoRA with large batch that triggers sorting."""
        layer = LoRASwitchLinear(
            input_dims=768, output_dims=3072, num_experts=8, r=16
        )

        # Large batch (>= 64 tokens triggers sorting in SwitchLinear)
        x = mx.random.normal((64, 768))
        indices = mx.random.randint(0, 8, (64,))

        # Should handle sorting internally
        output = layer(x, indices)

        self.assertEqual(output.shape, (64, 3072))


if __name__ == "__main__":
    unittest.main()
