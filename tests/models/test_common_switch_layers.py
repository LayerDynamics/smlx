"""
Tests for switch layers (Mixture of Experts).
"""

import mlx.core as mx
import pytest

from smlx.models.common.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    SwitchMLP,
)


@pytest.mark.unit
def test_switch_linear_properties():
    """Test SwitchLinear properties."""
    layer = SwitchLinear(256, 512, num_experts=8)

    assert layer.input_dims == 256
    assert layer.output_dims == 512
    assert layer.num_experts == 8


@pytest.mark.unit
def test_switch_linear_no_bias():
    """Test SwitchLinear without bias."""
    layer = SwitchLinear(256, 512, num_experts=8, bias=False)

    assert "bias" not in layer


@pytest.mark.unit
def test_switch_linear_to_quantized():
    """Test converting SwitchLinear to quantized version."""
    layer = SwitchLinear(256, 512, num_experts=8)

    quantized = layer.to_quantized(group_size=64, bits=4)

    assert isinstance(quantized, QuantizedSwitchLinear)
    assert quantized.input_dims == 256
    assert quantized.output_dims == 512
    assert quantized.num_experts == 8
    assert quantized.group_size == 64
    assert quantized.bits == 4


@pytest.mark.unit
def test_quantized_switch_linear_properties():
    """Test QuantizedSwitchLinear properties."""
    layer = QuantizedSwitchLinear(256, 512, num_experts=8, bits=4, group_size=64)

    assert layer.input_dims == 256
    assert layer.output_dims == 512
    assert layer.num_experts == 8
    assert layer.bits == 4
    assert layer.group_size == 64


@pytest.mark.unit
def test_switch_glu():
    """Test SwitchGLU layer."""
    input_dims = 256
    hidden_dims = 1024
    num_experts = 8
    batch_size = 4
    seq_len = 32  # Total tokens = 128 >= 64, triggers sorting path

    layer = SwitchGLU(
        input_dims=input_dims,
        hidden_dims=hidden_dims,
        num_experts=num_experts,
    )

    # Create input and expert indices
    x = mx.random.normal((batch_size, seq_len, input_dims))
    indices = mx.random.randint(0, num_experts, (batch_size, seq_len))

    output = layer(x, indices)

    assert output.shape == (batch_size, seq_len, input_dims)


@pytest.mark.unit
def test_switch_mlp():
    """Test SwitchMLP layer."""
    input_dims = 256
    hidden_dims = 1024
    num_experts = 8
    batch_size = 4
    seq_len = 32  # Total tokens = 128 >= 64, triggers sorting path

    layer = SwitchMLP(
        input_dims=input_dims,
        hidden_dims=hidden_dims,
        num_experts=num_experts,
    )

    # Create input and expert indices
    x = mx.random.normal((batch_size, seq_len, input_dims))
    indices = mx.random.randint(0, num_experts, (batch_size, seq_len))

    output = layer(x, indices)

    assert output.shape == (batch_size, seq_len, input_dims)




@pytest.mark.unit
def test_switch_glu_large_batch():
    """Test SwitchGLU with large batch (triggers sorting)."""
    input_dims = 256
    hidden_dims = 1024
    num_experts = 8
    batch_size = 4
    seq_len = 32  # Total tokens = 128 > 64, should trigger sorting

    layer = SwitchGLU(
        input_dims=input_dims,
        hidden_dims=hidden_dims,
        num_experts=num_experts,
    )

    x = mx.random.normal((batch_size, seq_len, input_dims))
    indices = mx.random.randint(0, num_experts, (batch_size, seq_len))

    output = layer(x, indices)

    assert output.shape == (batch_size, seq_len, input_dims)


@pytest.mark.unit
def test_switch_mlp_large_batch():
    """Test SwitchMLP with large batch (triggers sorting)."""
    input_dims = 256
    hidden_dims = 1024
    num_experts = 8
    batch_size = 4
    seq_len = 32  # Total tokens = 128 > 64, should trigger sorting

    layer = SwitchMLP(
        input_dims=input_dims,
        hidden_dims=hidden_dims,
        num_experts=num_experts,
    )

    x = mx.random.normal((batch_size, seq_len, input_dims))
    indices = mx.random.randint(0, num_experts, (batch_size, seq_len))

    output = layer(x, indices)

    assert output.shape == (batch_size, seq_len, input_dims)


@pytest.mark.unit
def test_switch_layers_different_expert_routing():
    """Test that different routing gives different results."""
    input_dims = 256
    output_dims = 512
    num_experts = 4

    layer = SwitchLinear(input_dims, output_dims, num_experts)

    x = mx.random.normal((8, input_dims))

    # Route to expert 0
    indices_0 = mx.zeros((8,), dtype=mx.int32)
    output_0 = layer(x, indices_0)

    # Route to expert 1
    indices_1 = mx.ones((8,), dtype=mx.int32)
    output_1 = layer(x, indices_1)

    # Results should be different (different experts)
    assert not mx.allclose(output_0, output_1)




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
