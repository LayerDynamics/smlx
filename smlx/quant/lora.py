"""
LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

LoRA adds trainable low-rank decomposition matrices to frozen pre-trained weights,
enabling efficient fine-tuning with minimal trainable parameters (~0.1-1% of original).

For a weight matrix W  R^(d_out � d_in), LoRA adds:
    �W = scale * B^T @ A^T
where A  R^(d_in � r), B  R^(r � d_out), and r << min(d_in, d_out)

Optimized for "smol" models (<10B parameters) on Apple M4 chipsets.

Reference:
    LoRA: Low-Rank Adaptation of Large Language Models
    https://arxiv.org/abs/2106.09685
"""

import math
from typing import Union

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA-enhanced linear layer.

    Wraps an existing `nn.Linear` or `nn.QuantizedLinear` layer and adds
    trainable low-rank adaptation matrices. The base layer weights remain frozen.

    Args:
        input_dims: Input feature dimensions
        output_dims: Output feature dimensions
        r: Rank of the low-rank decomposition (default: 8)
            Lower rank = fewer parameters but less expressive
            Typical range: 4-64
        dropout: Dropout probability for regularization (default: 0.0)
        scale: Scaling factor for LoRA updates (default: 20.0)
            Controls the magnitude of adaptation
        bias: Whether to include bias in the base layer (default: False)

    Example:
        ```python
        # Create from scratch
        lora_layer = LoRALinear(768, 768, r=8)

        # Convert existing layer
        linear = nn.Linear(768, 768)
        lora_layer = LoRALinear.from_base(linear, r=8)

        # Use with quantized layer (QLoRA)
        q_linear = nn.QuantizedLinear(768, 768, bits=4, group_size=64)
        lora_layer = LoRALinear.from_base(q_linear, r=8)
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
        Create LoRA layer from existing Linear or QuantizedLinear layer.

        Args:
            linear: Base linear layer to wrap
            r: LoRA rank
            dropout: Dropout probability
            scale: LoRA scaling factor

        Returns:
            LoRALinear layer wrapping the base layer
        """
        output_dims, input_dims = linear.weight.shape

        # Adjust for quantized layers
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = input_dims * 32 // linear.bits

        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

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
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        # LoRA scaling factor
        self.scale = scale

        # Low-rank adaptation matrices
        # A: down-projection, initialized with uniform distribution
        # B: up-projection, initialized to zeros (no adaptation initially)
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        """
        Forward pass: base output + scaled low-rank adaptation.

        Args:
            x: Input tensor of shape (..., input_dims)

        Returns:
            Output tensor of shape (..., output_dims)
        """
        # Base transformation (frozen weights)
        y = self.linear(x)

        # Low-rank adaptation: (x @ A) @ B
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b

        # Combine: y + scale * z
        dtype = x.dtype
        return y + (self.scale * z).astype(dtype)

    def fuse(self, dequantize: bool = False):
        """
        Merge LoRA weights into base layer.

        Creates a new layer with LoRA adaptation fused into the base weights.
        Useful for deployment to avoid the overhead of separate LoRA computation.

        Args:
            dequantize: If True, return dequantized layer even if base is quantized
                       If False, re-quantize after fusion (default)

        Returns:
            Fused linear layer (nn.Linear or nn.QuantizedLinear)

        Example:
            ```python
            # Train with LoRA
            lora_layer = LoRALinear.from_base(linear, r=8)
            # ... training ...

            # Fuse for deployment
            fused_layer = lora_layer.fuse(dequantize=False)
            ```
        """
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)

        # Use the same dtype as the base weight
        dtype = weight.dtype

        # Dequantize if needed
        if is_quantized:
            dtype = linear.scales.dtype
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
                dtype=dtype,
            )

        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Compute LoRA delta: scale * B^T @ A^T
        delta = ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(dtype)

        # Fuse: W_new = W_base + �W
        fused_linear.weight = weight + delta

        if bias:
            fused_linear.bias = linear.bias

        # Re-quantize if needed
        if is_quantized and not dequantize:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
            )

        return fused_linear


class LoRAEmbedding(nn.Module):
    """
    LoRA-enhanced embedding layer.

    Adds trainable low-rank adaptation to frozen embedding weights.
    Useful for adapting vocabulary embeddings to new domains.

    Args:
        num_embeddings: Size of the vocabulary
        dims: Embedding dimension
        r: LoRA rank (default: 8)
        dropout: Dropout probability (default: 0.0)
        scale: LoRA scaling factor (default: 20.0)

    Example:
        ```python
        # Create from existing embedding
        embedding = nn.Embedding(50000, 768)
        lora_emb = LoRAEmbedding.from_base(embedding, r=8)

        # Use in model
        tokens = mx.array([1, 2, 3, 4])
        embedded = lora_emb(tokens)  # Shape: (4, 768)
        ```
    """

    @staticmethod
    def from_base(
        embedding: nn.Embedding,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        """
        Create LoRA embedding from existing Embedding layer.

        Args:
            embedding: Base embedding layer
            r: LoRA rank
            dropout: Dropout probability
            scale: LoRA scaling factor

        Returns:
            LoRAEmbedding layer wrapping the base embedding
        """
        num_embeddings, dims = embedding.weight.shape

        # Adjust for quantized embeddings
        if isinstance(embedding, nn.QuantizedEmbedding):
            dims = dims * 32 // embedding.bits

        lora_embedding = LoRAEmbedding(
            num_embeddings=num_embeddings,
            dims=dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_embedding.embedding = embedding
        return lora_embedding

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
        self.embedding = nn.Embedding(num_embeddings, dims)
        self.dropout = nn.Dropout(p=dropout)

        # LoRA scaling factor
        self.scale = scale

        # Low-rank adaptation matrices
        init_scale = 1 / math.sqrt(num_embeddings)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(num_embeddings, r),
        )
        self.lora_b = mx.zeros(shape=(r, dims))

    def __call__(self, x):
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Token indices of shape (...)

        Returns:
            Embedded vectors of shape (..., dims)
        """
        # Base embedding (frozen)
        y = self.embedding(x)

        # LoRA adaptation: lookup A, multiply by B
        z = self.dropout(self.lora_a[x] @ self.lora_b)

        # Combine
        dtype = y.dtype
        out = y + (self.scale * z).astype(dtype)
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
        return y + (self.scale * z).astype(dtype)

    def fuse(self, dequantize: bool = False):
        """
        Merge LoRA weights into base embedding.

        Args:
            dequantize: If True, return dequantized embedding

        Returns:
            Fused embedding layer
        """
        embedding = self.embedding
        weight = embedding.weight
        is_quantized = isinstance(embedding, nn.QuantizedEmbedding)

        # Use the same dtype as the base weight
        dtype = weight.dtype

        # Dequantize if needed
        if is_quantized:
            dtype = embedding.scales.dtype
            weight = mx.dequantize(
                weight,
                embedding.scales,
                embedding.biases,
                embedding.group_size,
                embedding.bits,
                dtype=dtype,
            )

        num_embeddings, dims = weight.shape
        fused_embedding = nn.Embedding(num_embeddings, dims)

        # Fuse: scale * A @ B
        lora_a = (self.scale * self.lora_a).astype(dtype)
        lora_b = self.lora_b.astype(dtype)
        fused_embedding.weight = weight + lora_a @ lora_b

        # Re-quantize if needed
        if is_quantized and not dequantize:
            fused_embedding = nn.QuantizedEmbedding.from_embedding(
                fused_embedding,
                embedding.group_size,
                embedding.bits,
            )

        return fused_embedding


class LoRASwitchLinear(nn.Module):
    """
    LoRA-enhanced switch linear layer for Mixture of Experts models.

    Applies low-rank adaptation to SwitchLinear layers, enabling efficient
    fine-tuning of MoE models with minimal trainable parameters.

    Args:
        input_dims: Input feature dimensions
        output_dims: Output feature dimensions
        num_experts: Number of expert networks
        r: LoRA rank (default: 8)
        dropout: Dropout probability (default: 0.0)
        scale: LoRA scaling factor (default: 20.0)
        bias: Whether base layer has bias (default: False)

    Example:
        ```python
        from smlx.models.switch_layers import SwitchLinear, QuantizedSwitchLinear
        from smlx.quant import LoRASwitchLinear

        # Create from existing switch layer
        base_layer = SwitchLinear(768, 768, num_experts=8)
        lora_layer = LoRASwitchLinear.from_base(base_layer, r=8)

        # Use with quantized switch layer (QLoRA for MoE)
        quantized = QuantizedSwitchLinear(768, 768, num_experts=8, bits=4)
        qlora_layer = LoRASwitchLinear.from_base(quantized, r=8)
        ```

    Notes:
        - LoRA weights have expert dimension: (num_experts, ...)
        - Each expert gets its own low-rank adaptation
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
        Create LoRA switch layer from existing switch linear layer.

        Args:
            linear: Base switch linear layer (SwitchLinear or QuantizedSwitchLinear)
            r: LoRA rank
            dropout: Dropout probability
            scale: LoRA scaling factor

        Returns:
            LoRASwitchLinear layer wrapping the base layer
        """
        lora_lin = LoRASwitchLinear(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            num_experts=linear.num_experts,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

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
        self.linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        # LoRA scaling factor
        self.scale = scale

        # Low-rank adaptation matrices (per expert)
        # Shape: (num_experts, r, input_dims) and (num_experts, output_dims, r)
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(num_experts, r, input_dims),
        )
        self.lora_b = mx.zeros(shape=(num_experts, output_dims, r))
        self.num_experts = num_experts

    def __call__(self, x, indices, sorted_indices=False):
        """
        Forward pass with LoRA adaptation and expert routing.

        Args:
            x: Input tensor of shape (..., input_dims)
            indices: Expert routing indices
            sorted_indices: Whether indices are sorted (default: False)

        Returns:
            Output tensor of shape (..., output_dims)
        """
        # Base transformation (frozen weights)
        y = self.linear(x, indices, sorted_indices=sorted_indices)

        # LoRA adaptation: use gather_mm for expert-specific low-rank updates
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

        # Combine: y + scale * z
        dtype = x.dtype
        return y + (self.scale * z).astype(dtype)

    def fuse(self, dequantize: bool = False):
        """
        Merge LoRA weights into base switch layer.

        Creates a new switch layer with LoRA adaptation fused into base weights.

        Args:
            dequantize: If True, return dequantized layer even if base is quantized

        Returns:
            Fused switch linear layer

        Example:
            ```python
            # Train with LoRA
            lora_layer = LoRASwitchLinear.from_base(base_layer, r=8)
            # ... training ...

            # Fuse for deployment
            fused_layer = lora_layer.fuse(dequantize=False)
            ```
        """
        # Import here to avoid circular dependency
        from ..models.common.switch_layers import QuantizedSwitchLinear, SwitchLinear

        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, QuantizedSwitchLinear)

        # Use the same dtype as the base weight
        dtype = weight.dtype

        # Dequantize if needed
        if is_quantized:
            dtype = mx.float16
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )

        num_experts, output_dims, input_dims = weight.shape
        fused_linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        # Compute LoRA delta: scale * B @ A
        # Reshape lora_a for batch matmul
        lora_b = (self.scale * self.lora_b).astype(dtype)
        lora_a = self.lora_a.reshape(num_experts, -1, input_dims).astype(dtype)

        # Fuse: W_new = W_base + scale * B @ A
        fused_linear.weight = weight + lora_b @ lora_a

        if bias:
            fused_linear.bias = linear.bias

        # Re-quantize if needed
        if is_quantized and not dequantize:
            fused_linear = fused_linear.to_quantized(linear.group_size, linear.bits)

        return fused_linear


__all__ = [
    "LoRALinear",
    "LoRAEmbedding",
    "LoRASwitchLinear",
]
