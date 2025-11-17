"""
Tests for common MLP layers.
"""

import mlx.core as mx
import pytest

from smlx.models.common.mlp import (
    ExpertMLP,
    GeGLU,
    ParallelMLP,
    ReluSquared,
    StandardMLP,
    SwiGLU,
    create_mlp,
)


@pytest.mark.unit
def test_swiglu():
    """Test SwiGLU MLP layer."""
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    intermediate_size = 1024

    mlp = SwiGLU(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = mlp(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.unit
def test_swiglu_with_bias():
    """Test SwiGLU with bias."""
    mlp = SwiGLU(
        hidden_size=256,
        intermediate_size=1024,
        bias=True,
    )

    x = mx.random.normal((2, 16, 256))
    output = mlp(x)

    assert output.shape == (2, 16, 256)


@pytest.mark.unit
def test_geglu():
    """Test GeGLU MLP layer."""
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    intermediate_size = 1024

    mlp = GeGLU(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = mlp(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.unit
def test_geglu_approximate():
    """Test GeGLU with approximate GELU."""
    mlp = GeGLU(
        hidden_size=256,
        intermediate_size=1024,
        approximate="tanh",
    )

    x = mx.random.normal((2, 16, 256))
    output = mlp(x)

    assert output.shape == (2, 16, 256)


@pytest.mark.unit
def test_standard_mlp():
    """Test StandardMLP layer."""
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    intermediate_size = 1024

    mlp = StandardMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="gelu",
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = mlp(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.unit
def test_standard_mlp_activations():
    """Test StandardMLP with different activations."""
    activations = ["relu", "gelu", "silu", "tanh"]

    for activation in activations:
        mlp = StandardMLP(
            hidden_size=256,
            intermediate_size=1024,
            activation=activation,
        )

        x = mx.random.normal((2, 16, 256))
        output = mlp(x)

        assert output.shape == (2, 16, 256)


@pytest.mark.unit
def test_standard_mlp_invalid_activation():
    """Test StandardMLP with invalid activation."""
    with pytest.raises(ValueError):
        StandardMLP(
            hidden_size=256,
            intermediate_size=1024,
            activation="invalid",
        )


@pytest.mark.unit
def test_relu_squared():
    """Test ReluSquared MLP layer."""
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    intermediate_size = 1024

    mlp = ReluSquared(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = mlp(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.unit
def test_expert_mlp():
    """Test ExpertMLP layer."""
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    intermediate_size = 1024

    mlp = ExpertMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="swiglu",
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = mlp(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.unit
def test_expert_mlp_different_activations():
    """Test ExpertMLP with different activations."""
    activations = ["swiglu", "geglu", "relu", "gelu"]

    for activation in activations:
        mlp = ExpertMLP(
            hidden_size=256,
            intermediate_size=1024,
            activation=activation,
        )

        x = mx.random.normal((2, 16, 256))
        output = mlp(x)

        assert output.shape == (2, 16, 256)


@pytest.mark.unit
def test_expert_mlp_invalid_activation():
    """Test ExpertMLP with invalid activation."""
    with pytest.raises(ValueError):
        ExpertMLP(
            hidden_size=256,
            intermediate_size=1024,
            activation="invalid",
        )


@pytest.mark.unit
def test_parallel_mlp_sum():
    """Test ParallelMLP with sum combination."""
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    intermediate_size = 1024

    mlp = ParallelMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_parallel=2,
        combine="sum",
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = mlp(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.unit
def test_parallel_mlp_concat():
    """Test ParallelMLP with concatenation."""
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    intermediate_size = 1024

    mlp = ParallelMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_parallel=2,
        combine="concat",
    )

    x = mx.random.normal((batch_size, seq_len, hidden_size))
    output = mlp(x)

    assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.unit
def test_parallel_mlp_invalid_combine():
    """Test ParallelMLP with invalid combine method."""
    mlp = ParallelMLP(
        hidden_size=256,
        intermediate_size=1024,
        num_parallel=2,
        combine="invalid",
    )

    x = mx.random.normal((2, 16, 256))

    with pytest.raises(ValueError):
        mlp(x)


@pytest.mark.unit
def test_create_mlp_factory():
    """Test MLP factory function."""
    mlp_types = ["swiglu", "geglu", "standard", "relu_squared", "expert"]

    for mlp_type in mlp_types:
        mlp = create_mlp(
            hidden_size=256,
            intermediate_size=1024,
            mlp_type=mlp_type,
        )

        x = mx.random.normal((2, 16, 256))
        output = mlp(x)

        assert output.shape == (2, 16, 256)


@pytest.mark.unit
def test_create_mlp_invalid_type():
    """Test factory with invalid MLP type."""
    with pytest.raises(ValueError):
        create_mlp(
            hidden_size=256,
            intermediate_size=1024,
            mlp_type="invalid",
        )


@pytest.mark.unit
def test_mlp_shapes_preservation():
    """Test that all MLP types preserve input shape."""
    batch_size = 4
    seq_len = 32
    hidden_size = 512
    intermediate_size = 2048

    mlp_instances = [
        SwiGLU(hidden_size, intermediate_size),
        GeGLU(hidden_size, intermediate_size),
        StandardMLP(hidden_size, intermediate_size),
        ReluSquared(hidden_size, intermediate_size),
        ExpertMLP(hidden_size, intermediate_size),
    ]

    x = mx.random.normal((batch_size, seq_len, hidden_size))

    for mlp in mlp_instances:
        output = mlp(x)
        assert output.shape == x.shape, f"Failed for {mlp.__class__.__name__}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
