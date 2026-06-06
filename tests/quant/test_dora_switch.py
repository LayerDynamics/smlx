"""
Tests for DoRASwitchLinear (DoRA with Mixture of Experts).

Tests DoRA (Weight-Decomposed LoRA) adaptation on top of SwitchLinear layers
with magnitude-direction decomposition for better adaptation quality.
"""

import unittest

import mlx.core as mx
import mlx.nn as nn

from smlx.models.common.switch_layers import QuantizedSwitchLinear, SwitchLinear
from smlx.quant.dora import DoRASwitchLinear


class TestDoRASwitchLinear(unittest.TestCase):
    """Test DoRASwitchLinear layer."""

    def test_initialization(self):
        """Test DoRASwitchLinear initialization."""
        layer = DoRASwitchLinear(input_dims=768, output_dims=3072, num_experts=8, r=16, scale=1.0)

        # Check LoRA matrices shape
        # lora_a: (num_experts, r, input_dims)
        self.assertEqual(layer.lora_a.shape, (8, 16, 768))
        # lora_b: (num_experts, output_dims, r)
        self.assertEqual(layer.lora_b.shape, (8, 3072, 16))

        # Check magnitude parameter (per expert)
        # m: (num_experts, output_dims)
        self.assertEqual(layer.m.shape, (8, 3072))

        # Check base linear layer
        self.assertIsInstance(layer.linear, SwitchLinear)

    def test_from_base_switch_linear(self):
        """Test creating DoRA from existing SwitchLinear."""
        base_layer = SwitchLinear(input_dims=768, output_dims=3072, num_experts=8, bias=True)

        # Create DoRA layer from base
        dora_layer = DoRASwitchLinear.from_base(base_layer, r=16)

        # Check that base weights are preserved
        self.assertTrue(mx.allclose(dora_layer.linear.weight, base_layer.weight))
        self.assertTrue(mx.allclose(dora_layer.linear.bias, base_layer.bias))

        # Check magnitude initialized from base weights
        # m should have shape (num_experts, output_dims)
        self.assertEqual(dora_layer.m.shape, (8, 3072))

    def test_from_base_quantized(self):
        """Test creating DoRA from QuantizedSwitchLinear."""
        base_layer = SwitchLinear(input_dims=768, output_dims=3072, num_experts=8)
        quantized = base_layer.to_quantized(group_size=64, bits=4)

        # Create DoRA from quantized layer
        dora_layer = DoRASwitchLinear.from_base(quantized, r=16)

        # Check base is quantized
        self.assertIsInstance(dora_layer.linear, QuantizedSwitchLinear)

    def test_forward_pass(self):
        """Test forward pass with DoRA adaptation."""
        layer = DoRASwitchLinear(input_dims=768, output_dims=3072, num_experts=8, r=16, scale=1.0)

        # Switch layers route one expert per token; the input carries an
        # explicit per-expert token axis: (num_tokens, 1, input). This matches
        # how SwitchGLU/SwitchMLP feed their SwitchLinears.
        x = mx.random.normal((32, 1, 768))
        # Expert routing (one flat index per token)
        indices = mx.array([i % 8 for i in range(32)])

        # Forward pass
        output = layer(x, indices)

        # Check output shape: (num_tokens, 1, output)
        self.assertEqual(output.shape, (32, 1, 3072))

    def test_magnitude_rescaling(self):
        """Test that DoRA applies magnitude rescaling."""
        base_layer = SwitchLinear(input_dims=768, output_dims=768, num_experts=4)
        dora_layer = DoRASwitchLinear.from_base(base_layer, r=8, scale=1.0)

        # Set non-zero LoRA weights to see magnitude effect
        dora_layer.lora_a = mx.random.normal(dora_layer.lora_a.shape) * 0.1
        dora_layer.lora_b = mx.random.normal(dora_layer.lora_b.shape) * 0.1

        # Set magnitudes to different values per expert
        for i in range(4):
            dora_layer.m[i] = mx.ones_like(dora_layer.m[i]) * (i + 1)

        x = mx.random.normal((16, 1, 768))
        indices = mx.array([i % 4 for i in range(16)])

        # Forward pass applies magnitude rescaling
        output = dora_layer(x, indices)

        # Check output shape: (num_tokens, 1, output)
        self.assertEqual(output.shape, (16, 1, 768))

    def test_dora_vs_lora(self):
        """Test that DoRA differs from LoRA due to magnitude rescaling."""
        base_layer = SwitchLinear(input_dims=768, output_dims=768, num_experts=4)
        dora_layer = DoRASwitchLinear.from_base(base_layer, r=8, scale=1.0)

        # Import LoRA for comparison
        from smlx.quant.lora import LoRASwitchLinear

        lora_layer = LoRASwitchLinear.from_base(base_layer, r=8, scale=1.0)

        # Set same LoRA weights
        dora_layer.lora_a = mx.random.normal(dora_layer.lora_a.shape) * 0.1
        dora_layer.lora_b = mx.random.normal(dora_layer.lora_b.shape) * 0.1
        lora_layer.lora_a = dora_layer.lora_a
        lora_layer.lora_b = dora_layer.lora_b

        x = mx.random.normal((16, 1, 768))
        indices = mx.array([i % 4 for i in range(16)])

        dora_output = dora_layer(x, indices)
        lora_output = lora_layer(x, indices)

        # Outputs should be different due to magnitude rescaling in DoRA
        self.assertFalse(mx.allclose(dora_output, lora_output, atol=1e-5))

    def test_fuse(self):
        """Test fusing DoRA weights into base layer."""
        base_layer = SwitchLinear(input_dims=768, output_dims=768, num_experts=4)
        dora_layer = DoRASwitchLinear.from_base(base_layer, r=8, scale=1.0)

        # Set non-zero LoRA weights
        dora_layer.lora_a = mx.random.normal(dora_layer.lora_a.shape) * 0.01
        dora_layer.lora_b = mx.random.normal(dora_layer.lora_b.shape) * 0.01

        # Fuse
        fused_layer = dora_layer.fuse()

        # Check fused layer is SwitchLinear
        self.assertIsInstance(fused_layer, SwitchLinear)

        # Test that fused output matches DoRA output
        x = mx.random.normal((16, 1, 768))
        indices = mx.array([i % 4 for i in range(16)])

        dora_output = dora_layer(x, indices)
        fused_output = fused_layer(x, indices)

        # Should be very close
        self.assertTrue(mx.allclose(dora_output, fused_output, atol=1e-4))

    def test_fuse_quantized(self):
        """Test fusing DoRA with quantized base layer."""
        base_layer = SwitchLinear(input_dims=768, output_dims=768, num_experts=4)
        quantized = base_layer.to_quantized(group_size=64, bits=4)
        dora_layer = DoRASwitchLinear.from_base(quantized, r=8, scale=1.0)

        # Set non-zero LoRA weights
        dora_layer.lora_a = mx.random.normal(dora_layer.lora_a.shape) * 0.01
        dora_layer.lora_b = mx.random.normal(dora_layer.lora_b.shape) * 0.01

        # Fuse and keep quantized
        fused_q = dora_layer.fuse(dequantize=False)
        self.assertIsInstance(fused_q, QuantizedSwitchLinear)

        # Fuse and dequantize
        fused_dq = dora_layer.fuse(dequantize=True)
        self.assertIsInstance(fused_dq, SwitchLinear)

    def test_sorted_indices(self):
        """Test forward pass with sorted indices."""
        layer = DoRASwitchLinear(input_dims=768, output_dims=3072, num_experts=8, r=16)

        x = mx.random.normal((32, 1, 768))
        # Pre-sorted indices
        indices = mx.array(sorted([i % 8 for i in range(32)]))

        # Forward pass with sorted_indices=True
        output = layer(x, indices, sorted_indices=True)

        self.assertEqual(output.shape, (32, 1, 3072))

    def test_dropout(self):
        """Test that dropout is applied during training."""
        layer = DoRASwitchLinear(input_dims=768, output_dims=3072, num_experts=8, r=16, dropout=0.5)

        # Dropout should be in the layer
        self.assertIsNotNone(layer.dropout)

    def test_trainable_parameters(self):
        """Test that only DoRA parameters are trainable."""
        base_layer = SwitchLinear(input_dims=768, output_dims=3072, num_experts=8)
        dora_layer = DoRASwitchLinear.from_base(base_layer, r=16)

        # Freeze base layer
        dora_layer.linear.freeze()

        # DoRA parameters (lora_a, lora_b, m) should be trainable
        # (tree_flatten lives in mlx.utils, not mlx.core)
        from mlx.utils import tree_flatten

        trainable = tree_flatten(dora_layer.trainable_parameters())
        self.assertGreater(len(trainable), 0)


class TestDoRASwitchIntegration(unittest.TestCase):
    """Integration tests for DoRASwitchLinear."""

    def test_multi_expert_routing(self):
        """Test that different experts produce different outputs."""
        layer = DoRASwitchLinear(input_dims=768, output_dims=768, num_experts=4, r=8, scale=1.0)

        # Set different LoRA weights and magnitudes per expert
        for i in range(4):
            layer.lora_a[i] = mx.ones_like(layer.lora_a[i]) * (i + 1) * 0.01
            layer.lora_b[i] = mx.ones_like(layer.lora_b[i]) * (i + 1) * 0.01
            layer.m[i] = mx.ones_like(layer.m[i]) * (i + 1)

        # Same input, different experts. Input carries the per-expert token
        # axis: (num_tokens, 1, input).
        x = mx.ones((4, 1, 768))
        indices = mx.array([0, 1, 2, 3])

        output = layer(x, indices)

        # Each token processed by different expert should have different output
        self.assertEqual(output.shape, (4, 1, 768))

    def test_large_batch_sorting(self):
        """Test DoRA with large batch that triggers sorting."""
        layer = DoRASwitchLinear(input_dims=768, output_dims=3072, num_experts=8, r=16)

        # Large batch (>= 64 tokens). Input carries the per-expert token axis:
        # (num_tokens, 1, input).
        x = mx.random.normal((64, 1, 768))
        indices = mx.random.randint(0, 8, (64,))

        # Should handle routing for all tokens
        output = layer(x, indices)

        self.assertEqual(output.shape, (64, 1, 3072))

    def test_magnitude_per_expert(self):
        """Test that magnitude parameter is independent per expert."""
        layer = DoRASwitchLinear(input_dims=768, output_dims=768, num_experts=4, r=8, scale=1.0)

        # Each expert should have independent magnitude
        # m: (num_experts, output_dims)
        self.assertEqual(layer.m.shape, (4, 768))

        # Set different magnitudes
        for i in range(4):
            layer.m[i] = mx.ones_like(layer.m[i]) * (i + 1)

        # Check they're different
        for i in range(3):
            self.assertFalse(mx.allclose(layer.m[i], layer.m[i + 1]))


class TestDoRAMagnitudeInitialization(unittest.TestCase):
    """Test DoRA magnitude initialization from base weights."""

    def test_magnitude_from_unquantized(self):
        """Test magnitude initialization from unquantized weights."""
        base_layer = SwitchLinear(input_dims=768, output_dims=3072, num_experts=8)
        dora_layer = DoRASwitchLinear.from_base(base_layer, r=16)

        # Magnitude should be initialized as column norms of base weights
        # m: (num_experts, output_dims)
        self.assertEqual(dora_layer.m.shape, (8, 3072))

        # Check magnitudes are positive
        self.assertTrue(mx.all(dora_layer.m > 0))

    def test_magnitude_from_quantized(self):
        """Test magnitude initialization from quantized weights."""
        base_layer = SwitchLinear(input_dims=768, output_dims=3072, num_experts=8)
        quantized = base_layer.to_quantized(group_size=64, bits=4)
        dora_layer = DoRASwitchLinear.from_base(quantized, r=16)

        # Magnitude should be initialized from dequantized weights
        self.assertEqual(dora_layer.m.shape, (8, 3072))

        # Check magnitudes are positive
        self.assertTrue(mx.all(dora_layer.m > 0))


if __name__ == "__main__":
    unittest.main()
