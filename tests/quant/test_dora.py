"""
Tests for DoRA (Weight-Decomposed Low-Rank Adaptation) implementation.

Tests cover:
- DoRALinear layer creation and forward pass
- DoRAEmbedding layer creation and forward pass
- Conversion from base layers
- QDoRA with quantized base layers
- Weight fusion with magnitude rescaling
- Magnitude-direction decomposition
- Trainable parameter verification
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant import DoRAEmbedding, DoRALinear


@pytest.mark.unit
@pytest.mark.gpu
class TestDoRALinear:
    """Tests for DoRALinear layer."""

    def test_creation_from_scratch(self):
        """Test creating DoRALinear layer from scratch."""
        layer = DoRALinear(
            input_dims=768,
            output_dims=768,
            r=8,
            dropout=0.1,
            scale=20.0,
        )

        assert layer.lora_a.shape == (768, 8)
        assert layer.lora_b.shape == (8, 768)
        assert layer.scale == 20.0
        assert isinstance(layer.linear, nn.Linear)
        assert isinstance(layer.dropout, nn.Dropout)
        # DoRA-specific: magnitude vector
        assert hasattr(layer, "m")
        assert layer.m.shape == (768,)

    def test_from_base_linear(self):
        """Test creating DoRALinear from existing nn.Linear."""
        base_linear = nn.Linear(768, 768)
        dora_layer = DoRALinear.from_base(base_linear, r=8)

        assert dora_layer.lora_a.shape == (768, 8)
        assert dora_layer.lora_b.shape == (8, 768)
        assert dora_layer.linear is base_linear
        # DoRA-specific: magnitude vector initialized from base weights
        assert hasattr(dora_layer, "m")
        assert dora_layer.m.shape == (768,)

    def test_from_base_quantized(self):
        """Test creating DoRALinear from quantized layer (QDoRA)."""
        # Create and quantize a base layer
        base_linear = nn.Linear(768, 768)
        quantized = nn.QuantizedLinear.from_linear(
            base_linear,
            group_size=64,
            bits=4,
        )

        # Create DoRA from quantized layer
        dora_layer = DoRALinear.from_base(quantized, r=8)

        assert dora_layer.lora_a.shape == (768, 8)
        assert dora_layer.lora_b.shape == (8, 768)
        assert isinstance(dora_layer.linear, nn.QuantizedLinear)
        # Magnitude vector should still be computed
        assert hasattr(dora_layer, "m")
        assert dora_layer.m.shape == (768,)

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        layer = DoRALinear(input_dims=768, output_dims=512, r=8)
        x = mx.random.normal((4, 768))

        output = layer(x)

        assert output.shape == (4, 512)

    def test_forward_pass_dtype_preservation(self):
        """Test that forward pass preserves input dtype when base layer matches."""
        layer = DoRALinear(input_dims=768, output_dims=768, r=8)

        # Test with float32
        x_f32 = mx.random.normal((2, 768)).astype(mx.float32)
        out_f32 = layer(x_f32)
        assert out_f32.dtype == mx.float32

    def test_magnitude_direction_decomposition(self):
        """Test that DoRA applies magnitude-direction decomposition."""
        base_linear = nn.Linear(768, 768)
        dora_layer = DoRALinear.from_base(base_linear, r=8, dropout=0.0)

        # Store initial magnitude
        initial_m = mx.array(dora_layer.m)

        x = mx.random.normal((2, 768))

        # Set lora_b to non-zero for meaningful adaptation
        dora_layer.lora_b = mx.random.normal(dora_layer.lora_b.shape) * 0.01

        # DoRA output should differ from base due to magnitude rescaling
        base_output = base_linear(x)
        dora_output = dora_layer(x)

        # Outputs should differ (DoRA applies magnitude rescaling)
        assert not mx.allclose(base_output, dora_output, atol=1e-5)

        # Magnitude vector should be unchanged (it's a learned parameter but not updated here)
        assert mx.allclose(dora_layer.m, initial_m)

    def test_fuse_weights(self):
        """Test fusing DoRA weights into base layer with magnitude rescaling."""
        base_linear = nn.Linear(768, 768)
        dora_layer = DoRALinear.from_base(base_linear, r=8, dropout=0.0, scale=20.0)

        # Set lora_b to non-zero for meaningful fusion
        dora_layer.lora_b = mx.random.normal(dora_layer.lora_b.shape) * 0.01

        # Fuse weights
        fused_layer = dora_layer.fuse(dequantize=True)

        assert isinstance(fused_layer, nn.Linear)
        assert fused_layer.weight.shape == base_linear.weight.shape

        # Test that fused layer produces same output as DoRA layer
        x = mx.random.normal((2, 768))
        dora_output = dora_layer(x)
        fused_output = fused_layer(x)

        # Should be very close (within numerical precision)
        assert mx.allclose(dora_output, fused_output, atol=1e-4)

    def test_fuse_quantized_layer(self):
        """Test fusing DoRA with quantized base layer."""
        base_linear = nn.Linear(768, 768)
        quantized = nn.QuantizedLinear.from_linear(base_linear, group_size=64, bits=4)
        dora_layer = DoRALinear.from_base(quantized, r=8, dropout=0.0)

        # Set lora_b to non-zero
        dora_layer.lora_b = mx.random.normal(dora_layer.lora_b.shape) * 0.01

        # Fuse without dequantizing (should return QuantizedLinear)
        fused_quantized = dora_layer.fuse(dequantize=False)
        assert isinstance(fused_quantized, nn.QuantizedLinear)

        # Fuse with dequantizing (should return regular Linear)
        fused_dequantized = dora_layer.fuse(dequantize=True)
        assert isinstance(fused_dequantized, nn.Linear)

    def test_trainable_parameters(self):
        """Test that DoRA parameters exist as separate trainable components."""
        base_linear = nn.Linear(768, 768)
        dora_layer = DoRALinear.from_base(base_linear, r=8)

        # Freeze base layer (simulate training setup)
        dora_layer.linear.freeze()

        # Verify DoRA parameters exist with correct shapes
        assert hasattr(dora_layer, "lora_a")
        assert hasattr(dora_layer, "lora_b")
        assert hasattr(dora_layer, "m")  # DoRA-specific magnitude
        assert dora_layer.lora_a.shape == (768, 8)
        assert dora_layer.lora_b.shape == (8, 768)
        assert dora_layer.m.shape == (768,)

        # Verify base layer is accessible
        assert hasattr(dora_layer, "linear")
        assert dora_layer.linear is base_linear

        # DoRA has slightly more parameters than LoRA due to magnitude vector
        dora_params = dora_layer.lora_a.size + dora_layer.lora_b.size + dora_layer.m.size
        base_params = base_linear.weight.size
        assert dora_params < base_params * 0.1  # DoRA is still <10% of base size

    def test_different_ranks(self):
        """Test DoRA with different rank values."""
        for r in [4, 8, 16, 32]:
            layer = DoRALinear(input_dims=768, output_dims=768, r=r)
            assert layer.lora_a.shape == (768, r)
            assert layer.lora_b.shape == (r, 768)
            assert layer.m.shape == (768,)

            # Test forward pass works
            x = mx.random.normal((2, 768))
            output = layer(x)
            assert output.shape == (2, 768)

    def test_with_bias(self):
        """Test DoRALinear with bias enabled."""
        layer = DoRALinear(input_dims=768, output_dims=768, r=8, bias=True)
        assert "bias" in layer.linear

        x = mx.random.normal((2, 768))
        output = layer(x)
        assert output.shape == (2, 768)

    def test_dequantized_weight(self):
        """Test that _dequantized_weight method works correctly."""
        # Test with regular linear
        base_linear = nn.Linear(768, 768)
        dora_layer = DoRALinear.from_base(base_linear, r=8)

        weight = dora_layer._dequantized_weight()
        assert weight.shape == (768, 768)
        assert mx.allclose(weight, base_linear.weight)

        # Test with quantized linear
        quantized = nn.QuantizedLinear.from_linear(base_linear, group_size=64, bits=4)
        dora_quant = DoRALinear.from_base(quantized, r=8)

        dequant_weight = dora_quant._dequantized_weight()
        assert dequant_weight.shape == (768, 768)


@pytest.mark.unit
@pytest.mark.gpu
class TestDoRAEmbedding:
    """Tests for DoRAEmbedding layer."""

    def test_creation_from_scratch(self):
        """Test creating DoRAEmbedding from scratch."""
        layer = DoRAEmbedding(
            num_embeddings=50000,
            dims=768,
            r=8,
            dropout=0.1,
            scale=20.0,
        )

        assert layer.lora_a.shape == (50000, 8)
        assert layer.lora_b.shape == (8, 768)
        assert layer.scale == 20.0
        assert isinstance(layer.embedding, nn.Embedding)
        # DoRA-specific: magnitude vector
        assert hasattr(layer, "m")
        assert layer.m.shape == (50000,)

    def test_from_base_embedding(self):
        """Test creating DoRAEmbedding from existing nn.Embedding."""
        base_embedding = nn.Embedding(50000, 768)
        dora_emb = DoRAEmbedding.from_base(base_embedding, r=8)

        assert dora_emb.lora_a.shape == (50000, 8)
        assert dora_emb.lora_b.shape == (8, 768)
        assert dora_emb.embedding is base_embedding
        # DoRA-specific: magnitude vector
        assert hasattr(dora_emb, "m")
        assert dora_emb.m.shape == (50000,)

    def test_quantized_embedding_support(self):
        """Test DoRA with quantized embeddings (QDoRA)."""
        base_embedding = nn.Embedding(1000, 768)
        quantized_emb = nn.QuantizedEmbedding.from_embedding(
            base_embedding, group_size=64, bits=4
        )

        # Should successfully create DoRA from quantized embedding
        dora_emb = DoRAEmbedding.from_base(quantized_emb, r=8)

        # Verify dimensions are adjusted for quantization
        assert dora_emb.lora_a.shape == (1000, 8)
        assert dora_emb.lora_b.shape == (8, 768)
        assert isinstance(dora_emb.embedding, nn.QuantizedEmbedding)

        # Test forward pass works
        x = mx.array([1, 2, 3, 4, 5])
        output = dora_emb(x)
        assert output.shape == (5, 768)

        # Test fusion with re-quantization
        fused_q = dora_emb.fuse(dequantize=False)
        assert isinstance(fused_q, nn.QuantizedEmbedding)

        # Test fusion with dequantization
        fused_dq = dora_emb.fuse(dequantize=True)
        assert isinstance(fused_dq, nn.Embedding)
        assert not isinstance(fused_dq, nn.QuantizedEmbedding)

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        layer = DoRAEmbedding(num_embeddings=50000, dims=768, r=8)

        # Single sequence
        x = mx.array([1, 2, 3, 4, 5])
        output = layer(x)
        assert output.shape == (5, 768)

        # Batch of sequences
        x_batch = mx.array([[1, 2, 3], [4, 5, 6]])
        output_batch = layer(x_batch)
        assert output_batch.shape == (2, 3, 768)

    def test_forward_pass_dtype_preservation(self):
        """Test that forward pass preserves embedding dtype."""
        layer = DoRAEmbedding(num_embeddings=1000, dims=768, r=8)
        x = mx.array([1, 2, 3, 4])

        output = layer(x)

        # Should match base embedding dtype
        assert output.dtype == layer.embedding.weight.dtype

    def test_as_linear(self):
        """Test using embedding as linear layer (output projection)."""
        layer = DoRAEmbedding(num_embeddings=50000, dims=768, r=8)

        # Input: (batch, dims) -> Output: (batch, vocab_size)
        x = mx.random.normal((2, 768))
        logits = layer.as_linear(x)

        assert logits.shape == (2, 50000)

    def test_fuse_weights(self):
        """Test fusing DoRA weights into base embedding with magnitude rescaling."""
        base_embedding = nn.Embedding(1000, 768)
        dora_emb = DoRAEmbedding.from_base(base_embedding, r=8, dropout=0.0, scale=20.0)

        # Set lora_b to non-zero
        dora_emb.lora_b = mx.random.normal(dora_emb.lora_b.shape) * 0.01

        # Fuse weights
        fused_embedding = dora_emb.fuse()

        assert isinstance(fused_embedding, nn.Embedding)
        assert fused_embedding.weight.shape == base_embedding.weight.shape

        # Test that fused embedding produces same output as DoRA embedding
        x = mx.array([1, 2, 3, 4, 5])
        dora_output = dora_emb(x)
        fused_output = fused_embedding(x)

        # Should be very close
        assert mx.allclose(dora_output, fused_output, atol=1e-4)

    def test_magnitude_direction_decomposition(self):
        """Test that DoRA applies magnitude-direction decomposition to embeddings."""
        base_embedding = nn.Embedding(1000, 768)
        dora_emb = DoRAEmbedding.from_base(base_embedding, r=8, dropout=0.0)

        # Store initial magnitude
        initial_m = mx.array(dora_emb.m)

        x = mx.array([1, 2, 3])

        # Set lora_b non-zero
        dora_emb.lora_b = mx.random.normal(dora_emb.lora_b.shape) * 0.01

        # DoRA output should differ from base due to magnitude rescaling
        base_output = base_embedding(x)
        dora_output = dora_emb(x)

        # Should differ from base
        assert not mx.allclose(base_output, dora_output, atol=1e-5)

        # Magnitude vector unchanged (learned parameter but not updated here)
        assert mx.allclose(dora_emb.m, initial_m)

    def test_different_ranks(self):
        """Test DoRAEmbedding with different rank values."""
        for r in [4, 8, 16, 32]:
            layer = DoRAEmbedding(num_embeddings=1000, dims=768, r=r)
            assert layer.lora_a.shape == (1000, r)
            assert layer.lora_b.shape == (r, 768)
            assert layer.m.shape == (1000,)

            # Test forward pass works
            x = mx.array([1, 2, 3, 4])
            output = layer(x)
            assert output.shape == (4, 768)


@pytest.mark.unit
@pytest.mark.gpu
class TestDoRAIntegration:
    """Integration tests for DoRA functionality."""

    def test_dora_reduces_trainable_params(self):
        """Test that DoRA dramatically reduces trainable parameters (similar to LoRA)."""
        input_dims = 768
        output_dims = 768
        r = 8

        # Regular linear layer parameters
        regular_params = input_dims * output_dims  # 589,824

        # DoRA parameters (LoRA params + magnitude vector)
        dora_params = input_dims * r + r * output_dims + output_dims  # 12,288 + 768 = 13,056

        # DoRA should be much smaller (~2% in this case)
        reduction_factor = dora_params / regular_params
        assert reduction_factor < 0.05  # Less than 5% of original

        # Verify actual implementation
        layer = DoRALinear(input_dims, output_dims, r=r)
        actual_dora_params = layer.lora_a.size + layer.lora_b.size + layer.m.size
        assert actual_dora_params == dora_params

    def test_dora_linear_and_embedding_compatibility(self):
        """Test that DoRALinear and DoRAEmbedding work together."""
        vocab_size = 1000
        dims = 768
        r = 8

        # Create embedding and linear layer
        dora_emb = DoRAEmbedding(vocab_size, dims, r=r)
        dora_linear = DoRALinear(dims, dims, r=r)

        # Forward pass through both
        tokens = mx.array([1, 2, 3, 4])
        embedded = dora_emb(tokens)  # (4, 768)
        output = dora_linear(embedded)  # (4, 768)

        assert output.shape == (4, dims)

        # Use embedding as output projection
        logits = dora_emb.as_linear(output)  # (4, vocab_size)
        assert logits.shape == (4, vocab_size)

    def test_scale_parameter_effect(self):
        """Test that scale parameter affects adaptation magnitude."""
        base_linear = nn.Linear(768, 768)

        # Create two DoRA layers with different scales
        dora_small = DoRALinear.from_base(base_linear, r=8, scale=1.0, dropout=0.0)
        dora_large = DoRALinear.from_base(base_linear, r=8, scale=20.0, dropout=0.0)

        # Set same non-zero lora_b
        lora_b_value = mx.random.normal((8, 768)) * 0.01
        dora_small.lora_b = lora_b_value
        dora_large.lora_b = lora_b_value

        # Same lora_a
        lora_a_value = mx.random.normal((768, 8))
        dora_small.lora_a = lora_a_value
        dora_large.lora_a = lora_a_value

        # Forward pass
        x = mx.random.normal((2, 768))
        base_output = base_linear(x)
        small_output = dora_small(x)
        large_output = dora_large(x)

        # Large scale should have bigger difference from base
        small_diff = mx.abs(small_output - base_output).mean()
        large_diff = mx.abs(large_output - base_output).mean()

        # DoRA's magnitude rescaling normalizes the effect, so expect modest difference
        assert large_diff > small_diff  # scale is 20x larger, but magnitude rescaling moderates this

    def test_magnitude_learning_independence(self):
        """Test that magnitude vector is learned independently from direction."""
        base_linear = nn.Linear(768, 768)
        dora_layer = DoRALinear.from_base(base_linear, r=8, dropout=0.0)

        # Get initial magnitude
        initial_m = mx.array(dora_layer.m)

        # Modify lora_b (changes direction)
        dora_layer.lora_b = mx.random.normal(dora_layer.lora_b.shape) * 0.01

        # Magnitude should be unchanged (it's independent)
        assert mx.allclose(dora_layer.m, initial_m)

        # But we can modify magnitude independently
        dora_layer.m = dora_layer.m * 2.0
        assert not mx.allclose(dora_layer.m, initial_m)
        assert mx.allclose(dora_layer.m, initial_m * 2.0)

    def test_dora_vs_lora_output_difference(self):
        """Test that DoRA produces different outputs than LoRA due to magnitude rescaling."""
        from smlx.quant import LoRALinear

        base_linear = nn.Linear(768, 768)

        # Create LoRA and DoRA with same parameters
        lora_layer = LoRALinear.from_base(base_linear, r=8, dropout=0.0, scale=20.0)
        dora_layer = DoRALinear.from_base(base_linear, r=8, dropout=0.0, scale=20.0)

        # Set same lora_a and lora_b
        lora_a = mx.random.normal((768, 8))
        lora_b = mx.random.normal((8, 768)) * 0.01

        lora_layer.lora_a = lora_a
        lora_layer.lora_b = lora_b
        dora_layer.lora_a = lora_a
        dora_layer.lora_b = lora_b

        # Forward pass
        x = mx.random.normal((2, 768))
        lora_output = lora_layer(x)
        dora_output = dora_layer(x)

        # Outputs should differ due to DoRA's magnitude rescaling
        assert not mx.allclose(lora_output, dora_output, atol=1e-5)

        # But should be in similar range
        lora_mean = mx.mean(mx.abs(lora_output))
        dora_mean = mx.mean(mx.abs(dora_output))
        # Within reasonable range (magnitude rescaling affects this)
        assert mx.abs(lora_mean - dora_mean) / lora_mean < 2.0
