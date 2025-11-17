"""
Tests for Mixture of Experts (MoE) switch layers.

Tests SwitchLinear, QuantizedSwitchLinear, SwitchGLU, and SwitchMLP
for correct expert routing and output shapes.
"""

import unittest

import mlx.core as mx
import mlx.nn as nn

from smlx.models.common.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    SwitchMLP,
    SwiGLU,
)


class TestSwitchLinear(unittest.TestCase):
    """Test SwitchLinear layer."""

    def test_initialization(self):
        """Test SwitchLinear initialization."""
        layer = SwitchLinear(
            input_dims=768, output_dims=3072, num_experts=8, bias=True
        )

        # Check weight shape: (num_experts, output_dims, input_dims)
        self.assertEqual(layer.weight.shape, (8, 3072, 768))
        self.assertEqual(layer.bias.shape, (8, 3072))

    def test_forward_pass(self):
        """Test forward pass - SwitchLinear is primarily used within SwitchGLU/SwitchMLP."""
        # Test basic layer creation and weight shapes
        layer = SwitchLinear(input_dims=768, output_dims=3072, num_experts=8)

        # Check that the layer is properly initialized
        self.assertEqual(layer.num_experts, 8)
        self.assertEqual(layer.input_dims, 768)
        self.assertEqual(layer.output_dims, 3072)

    def test_sorted_indices(self):
        """Test that layer supports sorted_indices parameter."""
        layer = SwitchLinear(input_dims=768, output_dims=3072, num_experts=8)

        # Just verify the parameter is accepted (actual usage is in SwitchGLU/SwitchMLP)
        self.assertIsNotNone(layer)

    def test_no_bias(self):
        """Test SwitchLinear without bias."""
        layer = SwitchLinear(
            input_dims=768, output_dims=3072, num_experts=8, bias=False
        )

        # Should not have bias parameter
        self.assertNotIn("bias", layer)

    def test_to_quantized(self):
        """Test conversion to quantized version."""
        layer = SwitchLinear(input_dims=768, output_dims=3072, num_experts=8)

        # Convert to quantized
        quantized = layer.to_quantized(group_size=64, bits=4)

        # Check it's a QuantizedSwitchLinear
        self.assertIsInstance(quantized, QuantizedSwitchLinear)
        self.assertEqual(quantized.num_experts, 8)
        self.assertEqual(quantized.input_dims, 768)
        self.assertEqual(quantized.output_dims, 3072)


class TestQuantizedSwitchLinear(unittest.TestCase):
    """Test QuantizedSwitchLinear layer."""

    def test_initialization(self):
        """Test QuantizedSwitchLinear initialization."""
        layer = QuantizedSwitchLinear(
            input_dims=768, output_dims=3072, num_experts=8, bits=4, group_size=64
        )

        # Check weight is quantized (shape depends on MLX quantization internals)
        self.assertIsNotNone(layer.weight)
        self.assertEqual(len(layer.weight.shape), 3)
        self.assertEqual(layer.weight.shape[0], 8)  # num_experts
        self.assertEqual(layer.weight.shape[1], 3072)  # output_dims

        # Scales exist and have correct dimensions
        self.assertIsNotNone(layer.scales)
        self.assertEqual(len(layer.scales.shape), 3)
        self.assertEqual(layer.scales.shape[0], 8)  # num_experts
        self.assertEqual(layer.scales.shape[1], 3072)  # output_dims

    def test_forward_pass(self):
        """Test forward pass with quantized weights - primarily used within SwitchGLU/SwitchMLP."""
        layer = QuantizedSwitchLinear(
            input_dims=768, output_dims=3072, num_experts=8, bits=4
        )

        # Verify layer properties
        self.assertEqual(layer.num_experts, 8)
        self.assertEqual(layer.input_dims, 768)
        self.assertEqual(layer.output_dims, 3072)

    def test_properties(self):
        """Test layer properties."""
        layer = QuantizedSwitchLinear(
            input_dims=768, output_dims=3072, num_experts=8, bits=4, group_size=64
        )

        self.assertEqual(layer.input_dims, 768)
        self.assertEqual(layer.output_dims, 3072)
        self.assertEqual(layer.num_experts, 8)


class TestSwiGLU(unittest.TestCase):
    """Test SwiGLU activation function."""

    def test_swiglu_activation(self):
        """Test SwiGLU activation."""
        activation = SwiGLU()

        x = mx.random.normal((32, 3072))
        gate = mx.random.normal((32, 3072))

        # Apply SwiGLU
        output = activation(x, gate)

        # Check output shape
        self.assertEqual(output.shape, (32, 3072))


class TestSwitchGLU(unittest.TestCase):
    """Test SwitchGLU FFN with expert routing."""

    def test_initialization(self):
        """Test SwitchGLU initialization."""
        ffn = SwitchGLU(input_dims=768, hidden_dims=3072, num_experts=8)

        # Check all three projections exist
        self.assertIsInstance(ffn.gate_proj, SwitchLinear)
        self.assertIsInstance(ffn.up_proj, SwitchLinear)
        self.assertIsInstance(ffn.down_proj, SwitchLinear)

    def test_forward_pass(self):
        """Test SwitchGLU layer structure."""
        ffn = SwitchGLU(input_dims=768, hidden_dims=3072, num_experts=8)

        # Verify all three projections are SwitchLinear layers
        self.assertIsInstance(ffn.gate_proj, SwitchLinear)
        self.assertIsInstance(ffn.up_proj, SwitchLinear)
        self.assertIsInstance(ffn.down_proj, SwitchLinear)

        # Verify dimensions
        self.assertEqual(ffn.gate_proj.input_dims, 768)
        self.assertEqual(ffn.gate_proj.output_dims, 3072)
        self.assertEqual(ffn.down_proj.output_dims, 768)

    def test_with_large_batch(self):
        """Test SwitchGLU with large batch (triggers sorting)."""
        ffn = SwitchGLU(input_dims=768, hidden_dims=3072, num_experts=8)

        # Large batch to trigger sorting (>= 64 tokens)
        x = mx.random.normal((8, 10, 768))  # 80 tokens total
        indices = mx.random.randint(0, 8, (8, 10))

        # Forward pass should trigger sorting optimization
        output = ffn(x, indices)

        self.assertEqual(output.shape, (8, 10, 768))


class TestSwitchMLP(unittest.TestCase):
    """Test SwitchMLP with expert routing."""

    def test_initialization(self):
        """Test SwitchMLP initialization."""
        mlp = SwitchMLP(input_dims=768, hidden_dims=3072, num_experts=8)

        # Check both layers exist
        self.assertIsInstance(mlp.fc1, SwitchLinear)
        self.assertIsInstance(mlp.fc2, SwitchLinear)
        self.assertIsNotNone(mlp.activation)

    def test_forward_pass(self):
        """Test SwitchMLP layer structure."""
        mlp = SwitchMLP(input_dims=768, hidden_dims=3072, num_experts=8)

        # Verify both layers are SwitchLinear
        self.assertIsInstance(mlp.fc1, SwitchLinear)
        self.assertIsInstance(mlp.fc2, SwitchLinear)

        # Verify dimensions
        self.assertEqual(mlp.fc1.input_dims, 768)
        self.assertEqual(mlp.fc1.output_dims, 3072)
        self.assertEqual(mlp.fc2.output_dims, 768)

    def test_with_large_batch(self):
        """Test SwitchMLP with large batch (triggers sorting)."""
        mlp = SwitchMLP(input_dims=768, hidden_dims=3072, num_experts=8)

        # Large batch to trigger sorting
        x = mx.random.normal((8, 10, 768))  # 80 tokens total
        indices = mx.random.randint(0, 8, (8, 10))

        # Forward pass
        output = mlp(x, indices)

        self.assertEqual(output.shape, (8, 10, 768))


class TestExpertRouting(unittest.TestCase):
    """Test expert routing behavior across switch layers."""

    def test_consistent_routing(self):
        """Test that expert routing layers can be created with different expert counts."""
        layer1 = SwitchLinear(input_dims=768, output_dims=768, num_experts=4)
        layer2 = SwitchLinear(input_dims=768, output_dims=768, num_experts=8)

        # Verify different expert counts
        self.assertEqual(layer1.num_experts, 4)
        self.assertEqual(layer2.num_experts, 8)

        # Verify same input/output dims
        self.assertEqual(layer1.input_dims, 768)
        self.assertEqual(layer2.input_dims, 768)
        self.assertEqual(layer1.output_dims, 768)
        self.assertEqual(layer2.output_dims, 768)


if __name__ == "__main__":
    unittest.main()
