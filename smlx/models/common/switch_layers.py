"""
Switch Layers for Mixture of Experts (MoE) models in SMLX.

Switch layers enable efficient Mixture of Experts architectures by allowing
different tokens to be routed to different expert networks. This module provides
both regular and quantized switch linear layers optimized for Apple M4 chipsets.

Key features:
- SwitchLinear: MoE linear layer with expert routing
- QuantizedSwitchLinear: Quantized version for memory efficiency
- SwitchGLU: Gated Linear Unit with expert routing
- SwitchMLP: Standard MLP with expert routing

Optimized for "smol" models (<1B parameters) on Apple Silicon.

Example:
    ```python
    import mlx.core as mx
    from smlx.models.switch_layers import SwitchLinear, QuantizedSwitchLinear

    # Create a switch layer with 8 experts
    switch_layer = SwitchLinear(
        input_dims=768,
        output_dims=768,
        num_experts=8
    )

    # Route tokens to experts
    x = mx.random.normal((32, 768))  # 32 tokens
    indices = mx.array([0, 1, 2, 0, 1, 2, ...])  # Expert assignments
    output = switch_layer(x, indices)

    # Quantize for efficiency
    quantized = switch_layer.to_quantized(group_size=64, bits=4)
    ```

Reference:
    Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
    https://arxiv.org/abs/2101.03961
"""

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn


def _gather_sort(x, indices):
    """
    Sort tokens by expert indices for efficient grouped processing.

    Args:
        x: Input tensor
        indices: Expert indices for each token

    Returns:
        Tuple of (sorted_x, sorted_indices, inverse_order)
    """
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    """
    Unsort tokens back to original order after expert processing.

    Args:
        x: Sorted tensor
        inv_order: Inverse sorting order
        shape: Optional shape to unflatten to

    Returns:
        Unsorted tensor in original order
    """
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    """
    Quantized switch linear layer for memory-efficient MoE models.

    Uses MLX's gather_qmm operation for efficient quantized matrix multiplication
    with expert routing.

    Args:
        input_dims: Input feature dimensions
        output_dims: Output feature dimensions
        num_experts: Number of expert networks
        bias: Whether to include bias terms (default: True)
        group_size: Group size for quantization (default: 64, optimal for M4)
        bits: Bits per weight (default: 4)
        mode: Quantization mode ("affine" or "symmetric", default: "affine")

    Example:
        ```python
        import mlx.core as mx
        from smlx.models.switch_layers import QuantizedSwitchLinear

        # Create quantized switch layer (8 experts, 4-bit)
        layer = QuantizedSwitchLinear(768, 768, num_experts=8, bits=4)

        # Route tokens to experts
        x = mx.random.normal((32, 768))
        indices = mx.array([0, 1, 2, 0, 1, 2, 0, 1] * 4)  # Expert routing
        output = layer(x, indices)
        ```
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        # Initialize weights and quantize
        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters (for base models)
        self.freeze()

    @property
    def input_dims(self):
        """Input feature dimensions."""
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        """Output feature dimensions."""
        return self.weight.shape[1]

    @property
    def num_experts(self):
        """Number of expert networks."""
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        """
        Forward pass with expert routing.

        Args:
            x: Input tensor of shape (..., input_dims)
            indices: Expert indices for each token (shape: ...)
            sorted_indices: Whether indices are already sorted (default: False)

        Returns:
            Output tensor of shape (..., output_dims)
        """
        # Quantized gather matrix multiplication
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    """
    Switch linear layer for Mixture of Experts models.

    Routes different tokens to different expert linear layers based on indices.
    Uses MLX's gather_mm operation for efficient expert-specific computation.

    Args:
        input_dims: Input feature dimensions
        output_dims: Output feature dimensions
        num_experts: Number of expert networks
        bias: Whether to include bias terms (default: True)

    Example:
        ```python
        import mlx.core as mx
        from smlx.models.switch_layers import SwitchLinear

        # Create switch layer with 8 experts
        layer = SwitchLinear(768, 3072, num_experts=8)

        # Process tokens with different experts
        x = mx.random.normal((32, 768))  # 32 tokens
        indices = mx.array([0, 1, 2, 3, 4, 5, 6, 7] * 4)  # Assign to experts
        output = layer(x, indices)  # Shape: (32, 3072)

        # Quantize for deployment
        quantized = layer.to_quantized(group_size=64, bits=4)
        ```

    Notes:
        - Each expert has its own weight matrix
        - Weights shape: (num_experts, output_dims, input_dims)
        - Supports conversion to quantized version via to_quantized()
    """

    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()

        # Initialize expert weights
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        """Input feature dimensions."""
        return self.weight.shape[2]

    @property
    def output_dims(self):
        """Output feature dimensions."""
        return self.weight.shape[1]

    @property
    def num_experts(self):
        """Number of expert networks."""
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        """
        Forward pass with expert routing.

        Args:
            x: Input tensor of shape (..., input_dims)
            indices: Expert indices for each token (shape: ...)
            sorted_indices: Whether indices are already sorted (default: False)
                           Sorting can improve performance for large batches

        Returns:
            Output tensor of shape (..., output_dims)
        """
        # Gather matrix multiplication: route each token to its expert
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        """
        Convert to quantized switch linear layer.

        Args:
            group_size: Group size for quantization (default: 64)
            bits: Bits per weight (default: 4)
            mode: Quantization mode (default: "affine")

        Returns:
            QuantizedSwitchLinear layer with same weights

        Example:
            ```python
            # Create and quantize
            layer = SwitchLinear(768, 768, num_experts=8)
            quantized = layer.to_quantized(group_size=64, bits=4)

            # ~8x memory reduction
            ```
        """
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


@partial(mx.compile, shapeless=True)
def swiglu(x, gate):
    """SwiGLU activation: SiLU(gate) * x"""
    return nn.silu(gate) * x


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.

    SwiGLU(x, gate) = SiLU(gate) * x
    where SiLU(x) = x * sigmoid(x)

    Commonly used in modern transformer FFNs.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        """
        Apply SwiGLU activation.

        Args:
            x: Value tensor
            gate: Gate tensor

        Returns:
            SwiGLU(x, gate)
        """
        return swiglu(x, gate)


class SwitchGLU(nn.Module):
    """
    Gated Linear Unit with Mixture of Experts.

    Implements GLU-style FFN with expert routing:
        output = down_proj(activation(up_proj(x), gate_proj(x)))

    Each projection (gate, up, down) is a SwitchLinear with its own experts.

    Args:
        input_dims: Input feature dimensions
        hidden_dims: Hidden layer dimensions (typically 4x input_dims)
        num_experts: Number of expert networks
        activation: Activation function (default: SwiGLU)
        bias: Whether to include bias (default: False)

    Example:
        ```python
        import mlx.core as mx
        from smlx.models.switch_layers import SwitchGLU

        # Create SwitchGLU FFN with 8 experts
        ffn = SwitchGLU(
            input_dims=768,
            hidden_dims=3072,  # 4x expansion
            num_experts=8
        )

        # Process tokens with expert routing
        x = mx.random.normal((32, 768))
        indices = mx.array([0, 1, 2, 0, 1, 2, 0, 1] * 4)
        output = ffn(x, indices)  # Shape: (32, 768)
        ```
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        """
        Forward pass through gated FFN with expert routing.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dims)
            indices: Expert routing indices of shape (batch, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, input_dims)
        """
        x = mx.expand_dims(x, (-2, -3))

        # Sort tokens by expert for efficient processing (if many tokens)
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        # Apply gated activation: activation(up(x), gate(x))
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        # Restore original token order
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    """
    Standard MLP with Mixture of Experts routing.

    Implements standard 2-layer MLP with expert routing:
        output = fc2(activation(fc1(x)))

    Args:
        input_dims: Input feature dimensions
        hidden_dims: Hidden layer dimensions
        num_experts: Number of expert networks
        activation: Activation function (default: GELU)
        bias: Whether to include bias (default: False)

    Example:
        ```python
        import mlx.core as mx
        from smlx.models.switch_layers import SwitchMLP

        # Create SwitchMLP with 8 experts
        mlp = SwitchMLP(
            input_dims=768,
            hidden_dims=3072,
            num_experts=8
        )

        # Process tokens
        x = mx.random.normal((32, 768))
        indices = mx.array([i % 8 for i in range(32)])  # Round-robin routing
        output = mlp(x, indices)  # Shape: (32, 768)
        ```
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        """
        Forward pass through MLP with expert routing.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dims)
            indices: Expert routing indices of shape (batch, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, input_dims)
        """
        x = mx.expand_dims(x, (-2, -3))

        # Sort tokens by expert for efficient processing (if many tokens)
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        # Standard MLP: fc2(activation(fc1(x)))
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        # Restore original token order
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


__all__ = [
    "SwitchLinear",
    "QuantizedSwitchLinear",
    "SwitchGLU",
    "SwitchMLP",
    "SwiGLU",
]
