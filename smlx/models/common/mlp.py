"""
Common MLP (Multi-Layer Perceptron) and FFN (Feed-Forward Network) layers.

This module provides reusable MLP implementations that can be shared
across different model architectures.

Implementations:
- SwiGLU: Gated linear unit with SiLU activation (used in LLaMA, SmolLM, etc.)
- GeGLU: Gated linear unit with GELU activation
- StandardMLP: Traditional MLP with configurable activation
- ExpertMLP: MLP for Mixture of Experts
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SwiGLU(nn.Module):
    """
    SwiGLU MLP layer.

    Implements the gated linear unit pattern with SiLU (Swish) activation:
        output = down_proj(silu(gate_proj(x)) * up_proj(x))

    This is used in LLaMA, SmolLM, and many modern transformer models.

    References:
    - "GLU Variants Improve Transformer"
    - "Swish: a Self-Gated Activation Function"

    Args:
        hidden_size: Dimension of input/output
        intermediate_size: Dimension of the intermediate layer
        bias: Whether to use bias in linear projections
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply SwiGLU transformation.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class GeGLU(nn.Module):
    """
    GeGLU MLP layer.

    Implements the gated linear unit pattern with GELU activation:
        output = down_proj(gelu(gate_proj(x)) * up_proj(x))

    This is an alternative to SwiGLU, using GELU instead of SiLU.

    References:
    - "GLU Variants Improve Transformer"

    Args:
        hidden_size: Dimension of input/output
        intermediate_size: Dimension of the intermediate layer
        bias: Whether to use bias in linear projections
        approximate: Whether to use approximation for GELU
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        approximate: str = "none",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.approximate = approximate

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply GeGLU transformation.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        if self.approximate == "tanh":
            gate = nn.gelu_approx(self.gate_proj(x))
        elif self.approximate == "fast":
            gate = nn.gelu_fast_approx(self.gate_proj(x))
        else:
            gate = nn.gelu(self.gate_proj(x))

        return self.down_proj(gate * self.up_proj(x))


class StandardMLP(nn.Module):
    """
    Traditional MLP with configurable activation function.

    Implements a standard two-layer MLP:
        output = down_proj(activation(up_proj(x)))

    Args:
        hidden_size: Dimension of input/output
        intermediate_size: Dimension of the intermediate layer
        activation: Activation function ("relu", "gelu", "silu", "tanh")
        bias: Whether to use bias in linear projections
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_name = activation

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        # Select activation function
        if activation == "relu":
            self.activation = nn.relu
        elif activation == "gelu":
            self.activation = nn.gelu
        elif activation == "silu":
            self.activation = nn.silu
        elif activation == "tanh":
            self.activation = mx.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = dropout

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply MLP transformation.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        h = self.activation(self.up_proj(x))

        # Apply dropout if specified (though MLX doesn't have built-in dropout yet)
        # This is a placeholder for when MLX adds dropout support
        if self.dropout > 0.0:
            # TODO: Implement dropout when MLX supports it
            pass

        return self.down_proj(h)


class ReluSquared(nn.Module):
    """
    ReLU-Squared activation MLP.

    Uses ReLU^2 activation function: relu(x)^2
    This has been shown to work well in some transformer variants.

    Args:
        hidden_size: Dimension of input/output
        intermediate_size: Dimension of the intermediate layer
        bias: Whether to use bias in linear projections
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply ReLU� MLP transformation.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        h = nn.relu(self.up_proj(x))
        h = h * h  # Square the activation
        return self.down_proj(h)


class ExpertMLP(nn.Module):
    """
    Expert MLP for Mixture of Experts (MoE).

    This is a standard MLP that can be used as an expert in MoE layers.
    It's essentially identical to other MLPs but is designed to be used
    in an ensemble with a routing mechanism.

    Args:
        hidden_size: Dimension of input/output
        intermediate_size: Dimension of the intermediate layer
        activation: Type of activation/gating ("swiglu", "geglu", "relu", "gelu")
        bias: Whether to use bias in linear projections
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "swiglu",
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = activation

        if activation == "swiglu":
            self.mlp = SwiGLU(hidden_size, intermediate_size, bias=bias)
        elif activation == "geglu":
            self.mlp = GeGLU(hidden_size, intermediate_size, bias=bias)
        elif activation in ["relu", "gelu", "silu"]:
            self.mlp = StandardMLP(hidden_size, intermediate_size, activation=activation, bias=bias)
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply expert MLP transformation.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        return self.mlp(x)


class ParallelMLP(nn.Module):
    """
    Parallel MLP configuration.

    Some architectures use parallel MLP configurations where multiple
    MLPs process the input simultaneously and their outputs are combined.

    Args:
        hidden_size: Dimension of input/output
        intermediate_size: Dimension of each intermediate layer
        num_parallel: Number of parallel MLPs
        activation: Activation function
        bias: Whether to use bias
        combine: How to combine outputs ("sum", "concat")
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_parallel: int = 2,
        activation: str = "swiglu",
        bias: bool = False,
        combine: str = "sum",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_parallel = num_parallel
        self.combine = combine

        # Create parallel experts
        self.experts = [
            ExpertMLP(hidden_size, intermediate_size, activation=activation, bias=bias)
            for _ in range(num_parallel)
        ]

        # If concatenating, we need a projection back to hidden_size
        if combine == "concat":
            self.output_proj = nn.Linear(hidden_size * num_parallel, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply parallel MLP transformation.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Process through all parallel MLPs
        outputs = [expert(x) for expert in self.experts]

        # Combine outputs
        if self.combine == "sum":
            return mx.sum(mx.stack(outputs), axis=0)
        elif self.combine == "concat":
            concatenated = mx.concatenate(outputs, axis=-1)
            return self.output_proj(concatenated)
        else:
            raise ValueError(f"Unsupported combine method: {self.combine}")


def create_mlp(
    hidden_size: int,
    intermediate_size: int,
    mlp_type: str = "swiglu",
    bias: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create MLP layers.

    Args:
        hidden_size: Dimension of input/output
        intermediate_size: Dimension of intermediate layer
        mlp_type: Type of MLP ("swiglu", "geglu", "standard", "relu_squared", "expert")
        bias: Whether to use bias
        **kwargs: Additional arguments passed to the MLP constructor

    Returns:
        MLP module instance
    """
    if mlp_type == "swiglu":
        return SwiGLU(hidden_size, intermediate_size, bias=bias)
    elif mlp_type == "geglu":
        return GeGLU(hidden_size, intermediate_size, bias=bias, **kwargs)
    elif mlp_type == "standard":
        return StandardMLP(hidden_size, intermediate_size, bias=bias, **kwargs)
    elif mlp_type == "relu_squared":
        return ReluSquared(hidden_size, intermediate_size, bias=bias)
    elif mlp_type == "expert":
        return ExpertMLP(hidden_size, intermediate_size, bias=bias, **kwargs)
    elif mlp_type == "parallel":
        return ParallelMLP(hidden_size, intermediate_size, bias=bias, **kwargs)
    else:
        raise ValueError(f"Unsupported MLP type: {mlp_type}")
