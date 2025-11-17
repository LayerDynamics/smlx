"""
Tests for LoRA (Low-Rank Adaptation) implementation.

Tests cover:
- LoRALinear layer creation and forward pass
- LoRAEmbedding layer creation and forward pass
- Conversion from base layers
- QLoRA with quantized base layers
- Weight fusion
- Trainable parameter verification
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.quant import LoRAEmbedding, LoRALinear


@pytest.mark.unit
@pytest.mark.gpu
class TestLoRALinear:
    """Tests for LoRALinear layer."""

    def test_creation_from_scratch(self):
        """Test creating LoRALinear layer from scratch."""
        layer = LoRALinear(
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

    def test_from_base_linear(self):
        """Test creating LoRALinear from existing nn.Linear."""
        base_linear = nn.Linear(768, 768)
        lora_layer = LoRALinear.from_base(base_linear, r=8)

        assert lora_layer.lora_a.shape == (768, 8)
        assert lora_layer.lora_b.shape == (8, 768)
        assert lora_layer.linear is base_linear

    def test_from_base_quantized(self):
        """Test creating LoRALinear from quantized layer (QLoRA)."""
        # Create and quantize a base layer
        base_linear = nn.Linear(768, 768)
        quantized = nn.QuantizedLinear.from_linear(
            base_linear,
            group_size=64,
            bits=4,
        )

        # Create LoRA from quantized layer
        lora_layer = LoRALinear.from_base(quantized, r=8)

        assert lora_layer.lora_a.shape == (768, 8)
        assert lora_layer.lora_b.shape == (8, 768)
        assert isinstance(lora_layer.linear, nn.QuantizedLinear)

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        layer = LoRALinear(input_dims=768, output_dims=512, r=8)
        x = mx.random.normal((4, 768))

        output = layer(x)

        assert output.shape == (4, 512)

    def test_forward_pass_dtype_preservation(self):
        """Test that forward pass preserves input dtype when base layer matches."""
        layer = LoRALinear(input_dims=768, output_dims=768, r=8)

        # Test with float32
        x_f32 = mx.random.normal((2, 768)).astype(mx.float32)
        out_f32 = layer(x_f32)
        assert out_f32.dtype == mx.float32

        # Note: float16 input will be upcast to float32 if base layer is float32
        # This is expected behavior when base layer dtype differs from input dtype

    def test_forward_with_lora_adaptation(self):
        """Test that LoRA adaptation affects output."""
        base_linear = nn.Linear(768, 768)
        lora_layer = LoRALinear.from_base(base_linear, r=8, dropout=0.0)

        x = mx.random.normal((2, 768))

        # Base output
        base_output = base_linear(x)

        # LoRA output (should differ since lora_a is initialized randomly)
        # Note: lora_b is initialized to zeros, so initially no adaptation
        # But lora_a is random, so if we set lora_b non-zero, outputs differ
        lora_layer.lora_b = mx.random.normal(lora_layer.lora_b.shape) * 0.01
        lora_output_with_b = lora_layer(x)

        # With non-zero lora_b, outputs should differ
        assert not mx.allclose(base_output, lora_output_with_b, atol=1e-5)

    def test_fuse_weights(self):
        """Test fusing LoRA weights into base layer."""
        base_linear = nn.Linear(768, 768)
        lora_layer = LoRALinear.from_base(base_linear, r=8, dropout=0.0, scale=20.0)

        # Set lora_b to non-zero for meaningful fusion
        lora_layer.lora_b = mx.random.normal(lora_layer.lora_b.shape) * 0.01

        # Fuse weights
        fused_layer = lora_layer.fuse(dequantize=True)

        assert isinstance(fused_layer, nn.Linear)
        assert fused_layer.weight.shape == base_linear.weight.shape

        # Test that fused layer produces same output as LoRA layer
        x = mx.random.normal((2, 768))
        lora_output = lora_layer(x)
        fused_output = fused_layer(x)

        # Should be very close (within numerical precision)
        assert mx.allclose(lora_output, fused_output, atol=1e-4)

    def test_fuse_quantized_layer(self):
        """Test fusing LoRA with quantized base layer."""
        base_linear = nn.Linear(768, 768)
        quantized = nn.QuantizedLinear.from_linear(base_linear, group_size=64, bits=4)
        lora_layer = LoRALinear.from_base(quantized, r=8, dropout=0.0)

        # Set lora_b to non-zero
        lora_layer.lora_b = mx.random.normal(lora_layer.lora_b.shape) * 0.01

        # Fuse without dequantizing (should return QuantizedLinear)
        fused_quantized = lora_layer.fuse(dequantize=False)
        assert isinstance(fused_quantized, nn.QuantizedLinear)

        # Fuse with dequantizing (should return regular Linear)
        fused_dequantized = lora_layer.fuse(dequantize=True)
        assert isinstance(fused_dequantized, nn.Linear)

    def test_trainable_parameters(self):
        """Test that LoRA parameters exist as separate trainable components."""
        base_linear = nn.Linear(768, 768)
        lora_layer = LoRALinear.from_base(base_linear, r=8)

        # Freeze base layer (simulate training setup)
        lora_layer.linear.freeze()

        # Verify LoRA parameters exist with correct shapes
        assert hasattr(lora_layer, "lora_a")
        assert hasattr(lora_layer, "lora_b")
        assert lora_layer.lora_a.shape == (768, 8)
        assert lora_layer.lora_b.shape == (8, 768)

        # Verify base layer is accessible
        assert hasattr(lora_layer, "linear")
        assert lora_layer.linear is base_linear

        # The key insight: LoRA has far fewer parameters than the base layer
        lora_params = lora_layer.lora_a.size + lora_layer.lora_b.size
        base_params = base_linear.weight.size
        assert lora_params < base_params * 0.1  # LoRA is <10% of base size

    def test_different_ranks(self):
        """Test LoRA with different rank values."""
        for r in [4, 8, 16, 32]:
            layer = LoRALinear(input_dims=768, output_dims=768, r=r)
            assert layer.lora_a.shape == (768, r)
            assert layer.lora_b.shape == (r, 768)

            # Test forward pass works
            x = mx.random.normal((2, 768))
            output = layer(x)
            assert output.shape == (2, 768)

    def test_with_bias(self):
        """Test LoRALinear with bias enabled."""
        layer = LoRALinear(input_dims=768, output_dims=768, r=8, bias=True)
        assert "bias" in layer.linear

        x = mx.random.normal((2, 768))
        output = layer(x)
        assert output.shape == (2, 768)


@pytest.mark.unit
@pytest.mark.gpu
class TestLoRAEmbedding:
    """Tests for LoRAEmbedding layer."""

    def test_creation_from_scratch(self):
        """Test creating LoRAEmbedding from scratch."""
        layer = LoRAEmbedding(
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

    def test_from_base_embedding(self):
        """Test creating LoRAEmbedding from existing nn.Embedding."""
        base_embedding = nn.Embedding(50000, 768)
        lora_emb = LoRAEmbedding.from_base(base_embedding, r=8)

        assert lora_emb.lora_a.shape == (50000, 8)
        assert lora_emb.lora_b.shape == (8, 768)
        assert lora_emb.embedding is base_embedding

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        layer = LoRAEmbedding(num_embeddings=50000, dims=768, r=8)

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
        layer = LoRAEmbedding(num_embeddings=1000, dims=768, r=8)
        x = mx.array([1, 2, 3, 4])

        output = layer(x)

        # Should match base embedding dtype
        assert output.dtype == layer.embedding.weight.dtype

    def test_as_linear(self):
        """Test using embedding as linear layer (output projection)."""
        layer = LoRAEmbedding(num_embeddings=50000, dims=768, r=8)

        # Input: (batch, dims) -> Output: (batch, vocab_size)
        x = mx.random.normal((2, 768))
        logits = layer.as_linear(x)

        assert logits.shape == (2, 50000)

    def test_fuse_weights(self):
        """Test fusing LoRA weights into base embedding."""
        base_embedding = nn.Embedding(1000, 768)
        lora_emb = LoRAEmbedding.from_base(base_embedding, r=8, dropout=0.0, scale=20.0)

        # Set lora_b to non-zero
        lora_emb.lora_b = mx.random.normal(lora_emb.lora_b.shape) * 0.01

        # Fuse weights
        fused_embedding = lora_emb.fuse()

        assert isinstance(fused_embedding, nn.Embedding)
        assert fused_embedding.weight.shape == base_embedding.weight.shape

        # Test that fused embedding produces same output as LoRA embedding
        x = mx.array([1, 2, 3, 4, 5])
        lora_output = lora_emb(x)
        fused_output = fused_embedding(x)

        # Should be very close
        assert mx.allclose(lora_output, fused_output, atol=1e-4)

    def test_with_lora_adaptation(self):
        """Test that LoRA adaptation affects embeddings."""
        base_embedding = nn.Embedding(1000, 768)
        lora_emb = LoRAEmbedding.from_base(base_embedding, r=8, dropout=0.0)

        x = mx.array([1, 2, 3])

        # Base output
        base_output = base_embedding(x)

        # LoRA output with zero lora_b (should be same as base)
        lora_output_zero = lora_emb(x)
        # Should be very close since lora_b starts at zero
        assert mx.allclose(base_output, lora_output_zero, atol=1e-5)

        # Set lora_b non-zero
        lora_emb.lora_b = mx.random.normal(lora_emb.lora_b.shape) * 0.01
        lora_output_nonzero = lora_emb(x)

        # Should differ from base
        assert not mx.allclose(base_output, lora_output_nonzero, atol=1e-5)

    def test_different_ranks(self):
        """Test LoRAEmbedding with different rank values."""
        for r in [4, 8, 16, 32]:
            layer = LoRAEmbedding(num_embeddings=1000, dims=768, r=r)
            assert layer.lora_a.shape == (1000, r)
            assert layer.lora_b.shape == (r, 768)

            # Test forward pass works
            x = mx.array([1, 2, 3, 4])
            output = layer(x)
            assert output.shape == (4, 768)


@pytest.mark.unit
@pytest.mark.gpu
class TestLoRAIntegration:
    """Integration tests for LoRA functionality."""

    def test_lora_reduces_trainable_params(self):
        """Test that LoRA dramatically reduces trainable parameters."""
        input_dims = 768
        output_dims = 768
        r = 8

        # Regular linear layer parameters
        regular_params = input_dims * output_dims  # 589,824

        # LoRA parameters
        lora_params = input_dims * r + r * output_dims  # 6,144 + 6,144 = 12,288

        # LoRA should be much smaller (~2% in this case)
        reduction_factor = lora_params / regular_params
        assert reduction_factor < 0.05  # Less than 5% of original

        # Verify actual implementation
        layer = LoRALinear(input_dims, output_dims, r=r)
        actual_lora_params = layer.lora_a.size + layer.lora_b.size
        assert actual_lora_params == lora_params

    def test_lora_linear_and_embedding_compatibility(self):
        """Test that LoRALinear and LoRAEmbedding work together."""
        vocab_size = 1000
        dims = 768
        r = 8

        # Create embedding and linear layer
        lora_emb = LoRAEmbedding(vocab_size, dims, r=r)
        lora_linear = LoRALinear(dims, dims, r=r)

        # Forward pass through both
        tokens = mx.array([1, 2, 3, 4])
        embedded = lora_emb(tokens)  # (4, 768)
        output = lora_linear(embedded)  # (4, 768)

        assert output.shape == (4, dims)

        # Use embedding as output projection
        logits = lora_emb.as_linear(output)  # (4, vocab_size)
        assert logits.shape == (4, vocab_size)

    def test_scale_parameter_effect(self):
        """Test that scale parameter affects adaptation magnitude."""
        base_linear = nn.Linear(768, 768)

        # Create two LoRA layers with different scales
        lora_small = LoRALinear.from_base(base_linear, r=8, scale=1.0, dropout=0.0)
        lora_large = LoRALinear.from_base(base_linear, r=8, scale=20.0, dropout=0.0)

        # Set same non-zero lora_b
        lora_b_value = mx.random.normal((8, 768)) * 0.01
        lora_small.lora_b = lora_b_value
        lora_large.lora_b = lora_b_value

        # Same lora_a
        lora_a_value = mx.random.normal((768, 8))
        lora_small.lora_a = lora_a_value
        lora_large.lora_a = lora_a_value

        # Forward pass
        x = mx.random.normal((2, 768))
        base_output = base_linear(x)
        small_output = lora_small(x)
        large_output = lora_large(x)

        # Large scale should have bigger difference from base
        small_diff = mx.abs(small_output - base_output).mean()
        large_diff = mx.abs(large_output - base_output).mean()

        assert large_diff > small_diff * 10  # scale is 20x larger
