# Copyright � 2025 SMLX Project

"""
Dynamic cache size limits based on available memory.

Provides utilities for computing safe KV cache sizes based on model parameters
and available device memory, with support for GQA (Grouped Query Attention).
"""

from __future__ import annotations

from smlx.utils.memory import get_active_memory_gb, get_device_info


class CacheLimitManager:
    """
    Manages cache size limits based on available memory.

    Dynamically computes safe maximum KV cache sizes by accounting for:
    - Model weights and activations
    - Target memory usage
    - Device memory limits
    - Number of layers and attention heads
    - Grouped Query Attention (GQA) if applicable

    Attributes:
        model_size_gb: Model size in GB
        target_memory_gb: Target total memory usage in GB
        device_info: Device information from MLX

    Example:
        >>> manager = CacheLimitManager(
        ...     model_size_gb=0.5,
        ...     target_memory_gb=32.0
        ... )
        >>>
        >>> max_tokens = manager.compute_max_kv_size(
        ...     num_layers=24,
        ...     head_dim=64,
        ...     num_heads=12
        ... )
        >>> print(f"Max tokens: {max_tokens}")
    """

    def __init__(
        self,
        model_size_gb: float,
        target_memory_gb: float = 32.0,
        activation_overhead: float = 0.1,
    ):
        """
        Initialize cache limit manager.

        Args:
            model_size_gb: Model size in GB
            target_memory_gb: Target total memory usage in GB (default: 32.0)
            activation_overhead: Fraction of model size for activations (default: 0.1)
        """
        self.model_size_gb = model_size_gb
        self.target_memory_gb = target_memory_gb
        self.activation_overhead = activation_overhead
        self.device_info = get_device_info()

    def compute_max_kv_size(
        self,
        num_layers: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dtype_bytes: int = 2,  # fp16 by default
        safety_margin: float = 0.9,
    ) -> int:
        """
        Compute maximum safe KV cache size in tokens.

        Accounts for GQA (Grouped Query Attention) if num_kv_heads is provided.

        Args:
            num_layers: Number of transformer layers
            head_dim: Dimension of each attention head
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (None for MHA, < num_heads for GQA)
            dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)
            safety_margin: Safety factor to avoid OOM (default: 0.9)

        Returns:
            Maximum number of tokens that can fit in cache

        Example:
            >>> # Multi-Head Attention (MHA)
            >>> max_tokens = manager.compute_max_kv_size(
            ...     num_layers=24,
            ...     head_dim=64,
            ...     num_heads=12
            ... )
            >>>
            >>> # Grouped Query Attention (GQA)
            >>> max_tokens = manager.compute_max_kv_size(
            ...     num_layers=24,
            ...     head_dim=64,
            ...     num_heads=12,
            ...     num_kv_heads=4  # 4 KV heads, 12 query heads
            ... )
        """
        # Use num_heads for both Q and KV if num_kv_heads not specified (MHA)
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # Get device max memory
        device_max_gb = self.device_info.get("max_recommended_working_set_size_gb", 36.0)

        # Use the smaller of target and device max
        effective_max_gb = min(self.target_memory_gb, device_max_gb)

        # Account for current active memory
        current_active_gb = get_active_memory_gb()

        # Calculate available memory for KV cache
        # Reserve: model weights + activation overhead + current active memory
        reserved_gb = self.model_size_gb * (1.0 + self.activation_overhead) + current_active_gb

        # Apply safety margin
        available_gb = (effective_max_gb - reserved_gb) * safety_margin

        # Ensure we have positive available memory
        if available_gb <= 0:
            return 256  # Minimum safe cache size

        # Calculate bytes per token for KV cache
        # K and V: num_layers * num_kv_heads * head_dim * 2 (K and V) * dtype_bytes
        bytes_per_token = num_layers * num_kv_heads * head_dim * 2 * dtype_bytes

        # Convert to max tokens
        max_tokens = int((available_gb * 1e9) / bytes_per_token)

        # Ensure minimum of 256 tokens
        return max(max_tokens, 256)

    def should_use_rotating_cache(
        self,
        requested_tokens: int,
        num_layers: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
    ) -> bool:
        """
        Determine if rotating cache is needed for requested token count.

        Args:
            requested_tokens: Desired number of tokens to cache
            num_layers: Number of transformer layers
            head_dim: Dimension of each attention head
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (optional, for GQA)

        Returns:
            True if rotating cache should be used, False otherwise

        Example:
            >>> should_rotate = manager.should_use_rotating_cache(
            ...     requested_tokens=4096,
            ...     num_layers=24,
            ...     head_dim=64,
            ...     num_heads=12
            ... )
            >>> if should_rotate:
            ...     print("Use RotatingKVCache")
        """
        max_safe = self.compute_max_kv_size(
            num_layers=num_layers,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        return requested_tokens > max_safe

    def should_use_quantized_cache(
        self,
        requested_tokens: int,
        num_layers: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        quantization_bits: int = 4,
    ) -> bool:
        """
        Determine if quantized cache is needed for requested token count.

        Args:
            requested_tokens: Desired number of tokens to cache
            num_layers: Number of transformer layers
            head_dim: Dimension of each attention head
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (optional, for GQA)
            quantization_bits: Target quantization bits (4 or 8)

        Returns:
            True if quantized cache should be used, False otherwise

        Example:
            >>> should_quantize = manager.should_use_quantized_cache(
            ...     requested_tokens=8192,
            ...     num_layers=24,
            ...     head_dim=64,
            ...     num_heads=12,
            ...     quantization_bits=4
            ... )
        """
        # Quantization reduces memory by compression_ratio
        compression_ratio = 16 / quantization_bits  # e.g., 4x for 4-bit, 2x for 8-bit

        # Compute max with fp16
        max_fp16 = self.compute_max_kv_size(
            num_layers=num_layers,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dtype_bytes=2,  # fp16
        )

        # Check if quantization would help
        max_quantized = max_fp16 * compression_ratio

        # Use quantized if requested exceeds fp16 capacity but fits with quantization
        return requested_tokens > max_fp16 and requested_tokens <= max_quantized

    def recommend_cache_type(
        self,
        requested_tokens: int,
        num_layers: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
    ) -> dict:
        """
        Recommend cache configuration based on requested tokens and available memory.

        Args:
            requested_tokens: Desired number of tokens to cache
            num_layers: Number of transformer layers
            head_dim: Dimension of each attention head
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (optional, for GQA)

        Returns:
            Dictionary with recommendations:
            - cache_type: "standard", "rotating", or "quantized"
            - max_kv_size: Recommended maximum cache size
            - keep: Recommended number of tokens to keep (for rotating cache)
            - bits: Recommended quantization bits (for quantized cache)
            - reason: Explanation of recommendation

        Example:
            >>> recommendation = manager.recommend_cache_type(
            ...     requested_tokens=4096,
            ...     num_layers=24,
            ...     head_dim=64,
            ...     num_heads=12
            ... )
            >>> print(recommendation)
            {'cache_type': 'rotating', 'max_kv_size': 2048, ...}
        """
        max_safe = self.compute_max_kv_size(
            num_layers=num_layers,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        if requested_tokens <= max_safe:
            # Can use standard cache
            return {
                "cache_type": "standard",
                "max_kv_size": None,
                "keep": 0,
                "bits": None,
                "reason": f"Requested {requested_tokens} tokens fits in memory "
                f"(max: {max_safe} tokens)",
            }

        # Check if quantization would help
        # But limit quantization to reasonable sizes (max 64K tokens)
        # For very large requests, use rotating cache instead
        max_quantized = max_safe * 4  # 4x compression for 4-bit
        should_quantize_4bit = self.should_use_quantized_cache(
            requested_tokens, num_layers, head_dim, num_heads, num_kv_heads, quantization_bits=4
        )

        # Use quantized cache only if reasonable (< 64K tokens)
        if should_quantize_4bit and requested_tokens <= 65536:
            # Use 4-bit quantized cache
            return {
                "cache_type": "quantized",
                "max_kv_size": min(requested_tokens, max_quantized),
                "keep": 256,  # Keep first 256 tokens
                "bits": 4,
                "reason": f"Requested {requested_tokens} tokens exceeds fp16 capacity "
                f"({max_safe}), using 4-bit quantization",
            }

        # Use rotating cache as fallback
        return {
            "cache_type": "rotating",
            "max_kv_size": max_safe,
            "keep": min(256, max_safe // 4),  # Keep 25% or 256 tokens, whichever is smaller
            "bits": None,
            "reason": f"Requested {requested_tokens} tokens exceeds capacity "
            f"({max_safe}), using rotating cache",
        }

    def get_memory_estimate(
        self,
        num_tokens: int,
        num_layers: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dtype_bytes: int = 2,
    ) -> dict:
        """
        Estimate memory usage for a given cache configuration.

        Args:
            num_tokens: Number of tokens in cache
            num_layers: Number of transformer layers
            head_dim: Dimension of each attention head
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (optional, for GQA)
            dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)

        Returns:
            Dictionary with memory estimates:
            - total_bytes: Total memory in bytes
            - total_gb: Total memory in GB
            - per_token_bytes: Bytes per token
            - gqa_savings: Memory savings from GQA (if applicable)

        Example:
            >>> estimate = manager.get_memory_estimate(
            ...     num_tokens=2048,
            ...     num_layers=24,
            ...     head_dim=64,
            ...     num_heads=12,
            ...     num_kv_heads=4
            ... )
            >>> print(f"Cache size: {estimate['total_gb']:.2f} GB")
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # Bytes per token
        bytes_per_token = num_layers * num_kv_heads * head_dim * 2 * dtype_bytes

        # Total bytes
        total_bytes = bytes_per_token * num_tokens
        total_gb = total_bytes / 1e9

        # Calculate GQA savings if applicable
        gqa_savings_gb = 0.0
        if num_kv_heads < num_heads:
            bytes_per_token_mha = num_layers * num_heads * head_dim * 2 * dtype_bytes
            total_bytes_mha = bytes_per_token_mha * num_tokens
            gqa_savings_gb = (total_bytes_mha - total_bytes) / 1e9

        return {
            "total_bytes": total_bytes,
            "total_gb": total_gb,
            "per_token_bytes": bytes_per_token,
            "gqa_savings_gb": gqa_savings_gb,
            "num_tokens": num_tokens,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
        }


__all__ = ["CacheLimitManager"]
