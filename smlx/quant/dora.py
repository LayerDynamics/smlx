"""
DoRA (Weight-Decomposed Low-Rank Adaptation) for fine-tuning.

DoRA extends LoRA by decomposing weights into magnitude and direction components,
applying LoRA to the directional component while learning the magnitude separately.
This provides better adaptation while maintaining efficiency.

For weight W, DoRA decomposes: W = m * (w / ||w||)
where m is learned magnitude and w is the directional component.

Reference:
    DoRA: Weight-Decomposed Low-Rank Adaptation
    https://arxiv.org/abs/2402.09353

Optimized for "smol" models (<10B parameters) on Apple M4 chipsets.
"""

import math
from typing import Union

import mlx.core as mx
import mlx.nn as nn


class DoRALinear(nn.Module):
    """
    DoRA-enhanced linear layer.

    Similar to LoRA but with magnitude-direction decomposition for better
    preservation of model behavior during adaptation.

    Args:
        input_dims: Input feature dimensions
        output_dims: Output feature dimensions
        r: Rank of the low-rank decomposition (default: 8)
        dropout: Dropout probability for regularization (default: 0.0)
        scale: Scaling factor for DoRA updates (default: 20.0)
        bias: Whether to include bias in the base layer (default: False)

    Note:
        DoRA typically provides better adaptation quality than LoRA but
        requires additional norm computations during forward pass.

    Example:
        ```python
        # Create from existing layer
        linear = nn.Linear(768, 768)
        dora_layer = DoRALinear.from_base(linear, r=8)

        # Use with quantized layer
        q_linear = nn.QuantizedLinear(768, 768, bits=4, group_size=64)
        dora_layer = DoRALinear.from_base(q_linear, r=8)
        ```
    """

    @staticmethod
    def from_base(
        linear: Union[nn.Linear, nn.QuantizedLinear],
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        """
        Create DoRA layer from existing Linear or QuantizedLinear layer.

        Args:
            linear: Base linear layer to wrap
            r: DoRA rank
            dropout: Dropout probability
            scale: DoRA scaling factor

        Returns:
            DoRALinear layer wrapping the base layer
        """
        output_dims, input_dims = linear.weight.shape

        # Adjust for quantized layers
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits

        dora_lin = DoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        dora_lin.set_linear(linear)
        return dora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()

        # Frozen base layer
        self.set_linear(nn.Linear(input_dims, output_dims, bias=bias))
        self.dropout = nn.Dropout(p=dropout)

        # DoRA scaling factor
        self.scale = scale

        # Low-rank adaptation matrices (same as LoRA)
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

        # Magnitude vector (learned, initialized from base weights)
        # Will be set in set_linear()

    def set_linear(self, linear):
        """
        Set the base linear layer and compute magnitude vector.

        Args:
            linear: Base linear layer
        """
        self.linear = linear
        # Compute magnitude: m = ||W||_2 for each output dimension
        weight = self._dequantized_weight().astype(mx.float32)
        self.m = mx.linalg.norm(weight, axis=1)

    def _dequantized_weight(self):
        """
        Return the dequantized weight of the linear layer.

        Returns:
            Dequantized weight array
        """
        weight = self.linear.weight

        if self._is_quantized():
            weight = mx.dequantize(
                weight,
                self.linear.scales,
                self.linear.biases,
                self.linear.group_size,
                self.linear.bits,
                dtype=self.linear.scales.dtype,
            )

        return weight

    def _is_quantized(self):
        """Check if the base layer is quantized."""
        return isinstance(self.linear, nn.QuantizedLinear)

    def __call__(self, x):
        """
        Forward pass with DoRA magnitude-direction decomposition.

        Args:
            x: Input tensor of shape (..., input_dims)

        Returns:
            Output tensor of shape (..., output_dims)
        """
        # Get dequantized weight for norm computation
        w = self._dequantized_weight()

        # Regular LoRA computation (without bias for now)
        dtype = x.dtype
        y = x @ w.T
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        out = y + (self.scale * z).astype(dtype)

        # DoRA-specific: magnitude-direction decomposition
        # Compute adapted weight: W + scale * �W
        adapted = w + (self.scale * self.lora_b.T) @ self.lora_a.T

        # Compute norm of adapted weight (direction)
        denom = mx.stop_gradient(mx.linalg.norm(adapted, axis=1))

        # Rescale by learned magnitude
        # out = (m / ||W + �W||) * out
        out = (self.m / denom).astype(dtype) * out

        # Add bias if present
        if "bias" in self.linear:
            out = out + self.linear.bias

        return out

    def fuse(self, dequantize: bool = False):
        """
        Merge DoRA weights into base layer.

        Creates a new layer with DoRA adaptation fused into the base weights,
        including the magnitude rescaling.

        Args:
            dequantize: If True, return dequantized layer

        Returns:
            Fused linear layer
        """
        linear = self.linear
        bias = "bias" in linear
        weight = self._dequantized_weight()

        # Use the same dtype as the base weight
        dtype = weight.dtype

        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Compute LoRA delta
        lora_b = (self.scale * self.lora_b.T).astype(dtype)
        lora_a = self.lora_a.T.astype(dtype)

        # Apply LoRA and magnitude rescaling
        weight = weight + lora_b @ lora_a
        norm_scale = self.m / mx.linalg.norm(weight, axis=1)
        fused_linear.weight = norm_scale[:, None] * weight

        if bias:
            fused_linear.bias = linear.bias

        # Re-quantize if needed
        if self._is_quantized() and not dequantize:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
            )

        return fused_linear


class DoRAEmbedding(nn.Module):
    """
    DoRA-enhanced embedding layer.

    Applies magnitude-direction decomposition to embedding weights.

    Args:
        num_embeddings: Size of the vocabulary
        dims: Embedding dimension
        r: DoRA rank (default: 8)
        dropout: Dropout probability (default: 0.0)
        scale: DoRA scaling factor (default: 20.0)

    Note:
        Currently does not support quantized embeddings.

    Example:
        ```python
        # Create from existing embedding
        embedding = nn.Embedding(50000, 768)
        dora_emb = DoRAEmbedding.from_base(embedding, r=8)
        ```
    """

    @staticmethod
    def from_base(
        embedding: Union[nn.Embedding, nn.QuantizedEmbedding],
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        """
        Create DoRA embedding from existing Embedding layer.

        Args:
            embedding: Base embedding layer
            r: DoRA rank
            dropout: Dropout probability
            scale: DoRA scaling factor

        Returns:
            DoRAEmbedding layer wrapping the base embedding
        """
        num_embeddings, dims = embedding.weight.shape

        # Adjust for quantized embeddings
        if isinstance(embedding, nn.QuantizedEmbedding):
            dims = dims * 32 // embedding.bits

        dora_embedding = DoRAEmbedding(
            num_embeddings=num_embeddings,
            dims=dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        dora_embedding.set_embedding(embedding)
        return dora_embedding

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        super().__init__()

        # Frozen base embedding
        self.set_embedding(nn.Embedding(num_embeddings, dims))
        self.dropout = nn.Dropout(p=dropout)

        # DoRA scaling factor
        self.scale = scale

        # Low-rank adaptation matrices
        init_scale = 1 / math.sqrt(num_embeddings)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(num_embeddings, r),
        )
        self.lora_b = mx.zeros(shape=(r, dims))

    def set_embedding(self, embedding: nn.Embedding):
        """
        Set the base embedding layer and compute magnitude vector.

        Args:
            embedding: Base embedding layer
        """
        self.embedding = embedding
        # Compute magnitude for each embedding vector (dequantize if needed)
        weight = self._dequantized_weight().astype(mx.float32)
        self.m = mx.linalg.norm(weight, axis=1)

    def _dequantized_weight(self):
        """
        Return the dequantized weight of the embedding layer.

        Returns:
            Dequantized weight array
        """
        weight = self.embedding.weight

        if self._is_quantized():
            weight = mx.dequantize(
                weight,
                self.embedding.scales,
                self.embedding.biases,
                self.embedding.group_size,
                self.embedding.bits,
                dtype=self.embedding.scales.dtype,
            )

        return weight

    def _is_quantized(self):
        """Check if the base embedding is quantized."""
        return isinstance(self.embedding, nn.QuantizedEmbedding)

    def __call__(self, x):
        """
        Forward pass with DoRA magnitude-direction decomposition.

        Args:
            x: Token indices of shape (...)

        Returns:
            Embedded vectors of shape (..., dims)
        """
        embedding = self.embedding

        # Base embedding
        y = embedding(x)

        # LoRA adaptation
        z = self.scale * self.lora_a[x] @ self.lora_b
        dtype = y.dtype
        out = y + self.dropout(z).astype(dtype)

        # DoRA: compute norm of adapted embeddings
        adapted = y + z
        denom = mx.stop_gradient(mx.linalg.norm(adapted, axis=-1))

        # Rescale by learned magnitude
        out = (self.m[x] / denom)[..., None] * out

        return out

    def as_linear(self, x):
        """
        Use embedding as linear layer (for output projection).

        Args:
            x: Input tensor of shape (..., dims)

        Returns:
            Output logits of shape (..., num_embeddings)
        """
        # Base transformation
        y = self.embedding.as_linear(x)

        # LoRA transformation
        z = (self.dropout(x) @ self.lora_b.T) @ self.lora_a.T
        dtype = x.dtype
        out = y + (self.scale * z).astype(dtype)

        # DoRA: compute norm of adapted weights
        adapted = self.embedding.weight + (self.scale * self.lora_a) @ self.lora_b
        denom = mx.stop_gradient(mx.linalg.norm(adapted, axis=1))

        # Rescale by learned magnitude
        out = (self.m / denom) * out

        return out

    def fuse(self, dequantize: bool = False):
        """
        Merge DoRA weights into base embedding.

        Args:
            dequantize: If True, return dequantized embedding

        Returns:
            Fused embedding layer
        """
        embedding = self.embedding
        weight = self._dequantized_weight()
        is_quantized = self._is_quantized()

        # Use the same dtype as the base weight
        dtype = weight.dtype

        num_embeddings, dims = weight.shape
        fused_embedding = nn.Embedding(num_embeddings, dims)

        # Compute LoRA delta
        lora_a = (self.scale * self.lora_a).astype(dtype)
        lora_b = self.lora_b.astype(dtype)

        # Apply LoRA and magnitude rescaling
        fused_weight = weight + lora_a @ lora_b
        norm_scale = self.m / mx.linalg.norm(fused_weight, axis=1)
        fused_embedding.weight = norm_scale[:, None] * fused_weight

        # Re-quantize if needed
        if is_quantized and not dequantize:
            fused_embedding = nn.QuantizedEmbedding.from_embedding(
                fused_embedding,
                embedding.group_size,
                embedding.bits,
            )

        return fused_embedding


class DoRASwitchLinear(nn.Module):
    """
    DoRA-enhanced switch linear layer for Mixture of Experts models.

    Applies magnitude-direction decomposition to SwitchLinear layers, combining
    DoRA's benefits with efficient MoE architectures.

    Args:
        input_dims: Input feature dimensions
        output_dims: Output feature dimensions
        num_experts: Number of expert networks
        r: DoRA rank (default: 8)
        dropout: Dropout probability (default: 0.0)
        scale: DoRA scaling factor (default: 20.0)
        bias: Whether base layer has bias (default: False)

    Example:
        ```python
        from smlx.models.switch_layers import SwitchLinear, QuantizedSwitchLinear
        from smlx.quant import DoRASwitchLinear

        # Create from existing switch layer
        base_layer = SwitchLinear(768, 768, num_experts=8)
        dora_layer = DoRASwitchLinear.from_base(base_layer, r=8)

        # Use with quantized switch layer (QDoRA for MoE)
        quantized = QuantizedSwitchLinear(768, 768, num_experts=8, bits=4)
        qdora_layer = DoRASwitchLinear.from_base(quantized, r=8)
        ```

    Notes:
        - DoRA weights have expert dimension: (num_experts, ...)
        - Each expert gets its own magnitude-direction decomposition
        - Supports both SwitchLinear and QuantizedSwitchLinear
    """

    @staticmethod
    def from_base(
        linear,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        """
        Create DoRA switch layer from existing switch linear layer.

        Args:
            linear: Base switch linear layer (SwitchLinear or QuantizedSwitchLinear)
            r: DoRA rank
            dropout: Dropout probability
            scale: DoRA scaling factor

        Returns:
            DoRASwitchLinear layer wrapping the base layer
        """
        dora_lin = DoRASwitchLinear(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            num_experts=linear.num_experts,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        dora_lin.set_linear(linear)
        return dora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()

        # Import here to avoid circular dependency
        from ..models.common.switch_layers import SwitchLinear

        # Frozen base layer
        self.set_linear(SwitchLinear(input_dims, output_dims, num_experts, bias=bias))
        self.dropout = nn.Dropout(p=dropout)

        # DoRA rank and scaling factor
        self.r = r
        self.scale = scale

        # Low-rank adaptation matrices (per expert)
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(num_experts, r, input_dims),
        )
        self.lora_b = mx.zeros(shape=(num_experts, output_dims, r))
        self.num_experts = num_experts

        # Magnitude vector will be set in set_linear()

    def set_linear(self, linear):
        """
        Set the base linear layer and compute magnitude vectors.

        Args:
            linear: Base switch linear layer
        """
        self.linear = linear
        # Compute magnitude: m = ||W||_2 for each output dimension, per expert
        weight = self._dequantized_weight().astype(mx.float32)
        # Shape: (num_experts, output_dims)
        self.m = mx.linalg.norm(weight, axis=2)

    def _dequantized_weight(self):
        """
        Return the dequantized weight of the switch linear layer.

        Returns:
            Dequantized weight array
        """
        # Import here to avoid circular dependency
        from ..models.common.switch_layers import QuantizedSwitchLinear

        weight = self.linear.weight

        if isinstance(self.linear, QuantizedSwitchLinear):
            weight = mx.dequantize(
                weight,
                self.linear.scales,
                self.linear.biases,
                self.linear.group_size,
                self.linear.bits,
                dtype=self.linear.scales.dtype,
            )

        return weight

    def __call__(self, x, indices, sorted_indices=False):
        """
        Forward pass with DoRA magnitude-direction decomposition and expert routing.

        Args:
            x: Input tensor of shape (..., input_dims)
            indices: Expert routing indices
            sorted_indices: Whether indices are sorted (default: False)

        Returns:
            Output tensor of shape (..., output_dims)
        """
        # Get dequantized weight for norm computation
        w = self._dequantized_weight()

        # Regular LoRA computation (without bias for now)
        dtype = x.dtype
        y = self.linear(x, indices, sorted_indices=sorted_indices)

        # Low-rank adaptation
        z = mx.gather_mm(
            self.dropout(x),
            self.lora_a.swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        z = mx.gather_mm(
            z,
            self.lora_b.swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )

        out = y + (self.scale * z).astype(dtype)

        # DoRA-specific: magnitude-direction decomposition
        # Compute adapted weight per expert: W[i] + scale * B[i] @ A[i]
        # Shape: (num_experts, output_dims, input_dims)
        lora_b_scaled = (self.scale * self.lora_b).astype(mx.float32)
        lora_a_reshaped = self.lora_a.reshape(self.num_experts, -1, self.lora_a.shape[-1]).astype(
            mx.float32
        )
        adapted = w.astype(mx.float32) + lora_b_scaled @ lora_a_reshaped

        # Compute norm of adapted weight (direction)
        # Shape: (num_experts, output_dims)
        denom = mx.stop_gradient(mx.linalg.norm(adapted, axis=2))

        # Rescale by learned magnitude: (m / ||W + ΔW||) per expert
        # Gather the correct expert's magnitude scaling
        # Shape for indices: (...,) -> index into (num_experts, output_dims)
        m_scaled = (self.m / denom)[indices]  # Shape: (..., output_dims)

        # `out` carries the per-expert token axis (M=1): shape (..., 1, output).
        # Insert the matching singleton axis into the magnitude so it rescales
        # each token's output in place. Without this, (..., output) broadcasts
        # against (..., 1, output) and explodes to (..., tokens, output).
        m_scaled = mx.expand_dims(m_scaled, -2)  # Shape: (..., 1, output_dims)

        # Apply magnitude rescaling
        out = m_scaled.astype(dtype) * out

        return out

    def fuse(self, dequantize: bool = False):
        """
        Merge DoRA weights into base switch layer.

        Creates a new switch layer with DoRA adaptation fused into base weights,
        including the magnitude rescaling.

        Args:
            dequantize: If True, return dequantized layer

        Returns:
            Fused switch linear layer

        Example:
            ```python
            # Train with DoRA
            dora_layer = DoRASwitchLinear.from_base(base_layer, r=8)
            # ... training ...

            # Fuse for deployment
            fused_layer = dora_layer.fuse(dequantize=False)
            ```
        """
        # Import here to avoid circular dependency
        from ..models.common.switch_layers import QuantizedSwitchLinear, SwitchLinear

        linear = self.linear
        bias = "bias" in linear
        weight = self._dequantized_weight()
        is_quantized = isinstance(linear, QuantizedSwitchLinear)

        # Use the same dtype as the base weight
        dtype = weight.dtype

        num_experts, output_dims, input_dims = weight.shape
        fused_linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        # Compute LoRA delta
        lora_b = (self.scale * self.lora_b).astype(dtype)
        lora_a = self.lora_a.reshape(num_experts, -1, input_dims).astype(dtype)

        # Apply LoRA and magnitude rescaling
        weight = weight + lora_b @ lora_a
        norm_scale = self.m / mx.linalg.norm(weight, axis=2)
        # Broadcast magnitude scale to all input dims
        fused_linear.weight = norm_scale[:, :, None] * weight

        if bias:
            fused_linear.bias = linear.bias

        # Re-quantize if needed
        if is_quantized and not dequantize:
            fused_linear = fused_linear.to_quantized(linear.group_size, linear.bits)

        return fused_linear


__all__ = [
    "DoRALinear",
    "DoRAEmbedding",
    "DoRASwitchLinear",
]
