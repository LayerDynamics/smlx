#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model Memory Profiles for SMLX.

Pre-defined safe memory parameters for each supported model to prevent
Out of Memory (OOM) errors. Profiles include recommended max_tokens,
KV cache sizes, and batch sizes based on model architecture and size.

Features:
- Per-model safe parameter presets
- Automatic parameter selection based on available memory
- Conservative defaults to prevent crashes
- Support for different memory tiers (low, medium, high)

Example:
    >>> from smlx.config.model_profiles import get_model_profile
    >>>
    >>> # Get recommended params for model
    >>> profile = get_model_profile("SmolLM2-135M")
    >>> print(f"Max tokens: {profile['max_tokens']}")
    >>> print(f"KV cache size: {profile['max_kv_size']}")
    >>>
    >>> # Auto-select based on available memory
    >>> from smlx.config.model_profiles import auto_select_params
    >>> params = auto_select_params("SmolVLM-256M", available_memory_gb=20.0)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from smlx.utils.memory import get_device_info, get_active_memory_gb


logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    """
    Memory availability tiers for parameter selection.

    LOW: < 16GB available (conservative settings)
    MEDIUM: 16-32GB available (balanced settings)
    HIGH: > 32GB available (aggressive settings)
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ModelProfile:
    """
    Memory profile for a specific model.

    Attributes:
        model_name: Model identifier
        num_parameters: Approximate parameter count
        base_memory_gb: Base memory usage (weights only)
        max_tokens: Recommended max tokens per generation
        max_kv_size: Recommended KV cache size
        batch_size: Recommended batch size
        use_rotating_cache: Whether to use rotating cache
        supports_quantization: Whether model supports quantization
        quantized_memory_gb: Memory usage when quantized (4-bit)
        description: Human-readable description
    """

    model_name: str
    num_parameters: int
    base_memory_gb: float
    max_tokens: int
    max_kv_size: int
    batch_size: int = 1
    use_rotating_cache: bool = True
    supports_quantization: bool = True
    quantized_memory_gb: Optional[float] = None
    description: str = ""


# Model profiles database
MODEL_PROFILES: Dict[str, Dict[MemoryTier, ModelProfile]] = {
    # SmolLM2-135M: Smallest language model
    "SmolLM2-135M": {
        MemoryTier.LOW: ModelProfile(
            model_name="SmolLM2-135M",
            num_parameters=135_000_000,
            base_memory_gb=0.27,
            max_tokens=512,
            max_kv_size=1024,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=0.14,
            description="Conservative: 512 tokens, rotating cache",
        ),
        MemoryTier.MEDIUM: ModelProfile(
            model_name="SmolLM2-135M",
            num_parameters=135_000_000,
            base_memory_gb=0.27,
            max_tokens=1000,
            max_kv_size=2048,
            batch_size=2,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=0.14,
            description="Balanced: 1000 tokens, 2048 cache",
        ),
        MemoryTier.HIGH: ModelProfile(
            model_name="SmolLM2-135M",
            num_parameters=135_000_000,
            base_memory_gb=0.27,
            max_tokens=2048,
            max_kv_size=4096,
            batch_size=4,
            use_rotating_cache=False,
            supports_quantization=True,
            quantized_memory_gb=0.14,
            description="Aggressive: 2048 tokens, full cache",
        ),
    },
    # SmolLM2-360M: Medium language model
    "SmolLM2-360M": {
        MemoryTier.LOW: ModelProfile(
            model_name="SmolLM2-360M",
            num_parameters=360_000_000,
            base_memory_gb=0.72,
            max_tokens=256,
            max_kv_size=512,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=0.36,
            description="Conservative: 256 tokens, small cache",
        ),
        MemoryTier.MEDIUM: ModelProfile(
            model_name="SmolLM2-360M",
            num_parameters=360_000_000,
            base_memory_gb=0.72,
            max_tokens=800,
            max_kv_size=1536,
            batch_size=2,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=0.36,
            description="Balanced: 800 tokens, medium cache",
        ),
        MemoryTier.HIGH: ModelProfile(
            model_name="SmolLM2-360M",
            num_parameters=360_000_000,
            base_memory_gb=0.72,
            max_tokens=1536,
            max_kv_size=3072,
            batch_size=4,
            use_rotating_cache=False,
            supports_quantization=True,
            quantized_memory_gb=0.36,
            description="Aggressive: 1536 tokens, large cache",
        ),
    },
    # SmolVLM-256M: Small vision-language model
    "SmolVLM-256M": {
        MemoryTier.LOW: ModelProfile(
            model_name="SmolVLM-256M",
            num_parameters=256_000_000,
            base_memory_gb=1.2,  # Includes vision encoder
            max_tokens=128,
            max_kv_size=512,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=0.6,
            description="Conservative: 128 tokens for VLM",
        ),
        MemoryTier.MEDIUM: ModelProfile(
            model_name="SmolVLM-256M",
            num_parameters=256_000_000,
            base_memory_gb=1.2,
            max_tokens=300,
            max_kv_size=1024,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=0.6,
            description="Balanced: 300 tokens, rotating cache",
        ),
        MemoryTier.HIGH: ModelProfile(
            model_name="SmolVLM-256M",
            num_parameters=256_000_000,
            base_memory_gb=1.2,
            max_tokens=500,
            max_kv_size=2048,
            batch_size=2,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=0.6,
            description="Aggressive: 500 tokens, standard cache",
        ),
    },
    # SmolVLM-500M-Instruct: Larger vision-language model
    "SmolVLM-500M": {
        MemoryTier.LOW: ModelProfile(
            model_name="SmolVLM-500M",
            num_parameters=500_000_000,
            base_memory_gb=2.0,
            max_tokens=100,
            max_kv_size=512,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=1.0,
            description="Conservative: 100 tokens for larger VLM",
        ),
        MemoryTier.MEDIUM: ModelProfile(
            model_name="SmolVLM-500M",
            num_parameters=500_000_000,
            base_memory_gb=2.0,
            max_tokens=250,
            max_kv_size=1024,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=1.0,
            description="Balanced: 250 tokens, rotating cache",
        ),
        MemoryTier.HIGH: ModelProfile(
            model_name="SmolVLM-500M",
            num_parameters=500_000_000,
            base_memory_gb=2.0,
            max_tokens=400,
            max_kv_size=1536,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=1.0,
            description="Aggressive: 400 tokens, medium cache",
        ),
    },
    # nanoVLM: Minimal 222M VLM
    "nanoVLM": {
        MemoryTier.LOW: ModelProfile(
            model_name="nanoVLM",
            num_parameters=222_000_000,
            base_memory_gb=1.0,
            max_tokens=128,
            max_kv_size=512,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=0.5,
            description="Conservative: minimal VLM settings",
        ),
        MemoryTier.MEDIUM: ModelProfile(
            model_name="nanoVLM",
            num_parameters=222_000_000,
            base_memory_gb=1.0,
            max_tokens=256,
            max_kv_size=1024,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=0.5,
            description="Balanced: 256 tokens",
        ),
        MemoryTier.HIGH: ModelProfile(
            model_name="nanoVLM",
            num_parameters=222_000_000,
            base_memory_gb=1.0,
            max_tokens=400,
            max_kv_size=2048,
            batch_size=2,
            use_rotating_cache=False,
            supports_quantization=True,
            quantized_memory_gb=0.5,
            description="Aggressive: 400 tokens, full cache",
        ),
    },
    # Moondream2: ~500M VLM with region detection
    "Moondream2": {
        MemoryTier.LOW: ModelProfile(
            model_name="Moondream2",
            num_parameters=500_000_000,
            base_memory_gb=2.2,
            max_tokens=128,
            max_kv_size=512,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=1.1,
            description="Conservative: region detection VLM",
        ),
        MemoryTier.MEDIUM: ModelProfile(
            model_name="Moondream2",
            num_parameters=500_000_000,
            base_memory_gb=2.2,
            max_tokens=256,
            max_kv_size=1024,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=1.1,
            description="Balanced: 256 tokens with tiling",
        ),
        MemoryTier.HIGH: ModelProfile(
            model_name="Moondream2",
            num_parameters=500_000_000,
            base_memory_gb=2.2,
            max_tokens=384,
            max_kv_size=1536,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=1.1,
            description="Aggressive: 384 tokens, medium cache",
        ),
    },
    # TinyLLaVA: Compact LLaVA variant
    "TinyLLaVA": {
        MemoryTier.LOW: ModelProfile(
            model_name="TinyLLaVA",
            num_parameters=300_000_000,
            base_memory_gb=1.5,
            max_tokens=150,
            max_kv_size=512,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=0.75,
            description="Conservative: compact LLaVA",
        ),
        MemoryTier.MEDIUM: ModelProfile(
            model_name="TinyLLaVA",
            num_parameters=300_000_000,
            base_memory_gb=1.5,
            max_tokens=300,
            max_kv_size=1024,
            batch_size=1,
            use_rotating_cache=True,
            supports_quantization=True,
            quantized_memory_gb=0.75,
            description="Balanced: 300 tokens",
        ),
        MemoryTier.HIGH: ModelProfile(
            model_name="TinyLLaVA",
            num_parameters=300_000_000,
            base_memory_gb=1.5,
            max_tokens=450,
            max_kv_size=2048,
            batch_size=2,
            use_rotating_cache=False,
            supports_quantization=True,
            quantized_memory_gb=0.75,
            description="Aggressive: 450 tokens, full cache",
        ),
    },
    # Whisper-tiny: Audio transcription
    "Whisper-tiny": {
        MemoryTier.LOW: ModelProfile(
            model_name="Whisper-tiny",
            num_parameters=39_000_000,
            base_memory_gb=0.15,
            max_tokens=448,  # Whisper uses fixed chunks
            max_kv_size=1500,  # Max audio length
            batch_size=1,
            use_rotating_cache=False,
            supports_quantization=True,
            quantized_memory_gb=0.08,
            description="Conservative: single audio chunk",
        ),
        MemoryTier.MEDIUM: ModelProfile(
            model_name="Whisper-tiny",
            num_parameters=39_000_000,
            base_memory_gb=0.15,
            max_tokens=448,
            max_kv_size=3000,
            batch_size=2,
            use_rotating_cache=False,
            supports_quantization=True,
            quantized_memory_gb=0.08,
            description="Balanced: batch processing",
        ),
        MemoryTier.HIGH: ModelProfile(
            model_name="Whisper-tiny",
            num_parameters=39_000_000,
            base_memory_gb=0.15,
            max_tokens=448,
            max_kv_size=6000,
            batch_size=4,
            use_rotating_cache=False,
            supports_quantization=True,
            quantized_memory_gb=0.08,
            description="Aggressive: larger batches",
        ),
    },
}


def determine_memory_tier(available_memory_gb: Optional[float] = None) -> MemoryTier:
    """
    Determine appropriate memory tier based on available memory.

    Args:
        available_memory_gb: Available memory in GB (auto-detect if None)

    Returns:
        Memory tier classification

    Example:
        >>> tier = determine_memory_tier(20.0)
        >>> print(tier)  # MemoryTier.MEDIUM
    """
    if available_memory_gb is None:
        # Auto-detect
        device_info = get_device_info()
        max_memory_gb = device_info['max_recommended_working_set_size_gb']
        active_memory_gb = get_active_memory_gb()
        available_memory_gb = max_memory_gb - active_memory_gb

    if available_memory_gb < 16:
        return MemoryTier.LOW
    elif available_memory_gb < 32:
        return MemoryTier.MEDIUM
    else:
        return MemoryTier.HIGH


def get_model_profile(
    model_name: str,
    memory_tier: Optional[MemoryTier] = None,
    available_memory_gb: Optional[float] = None,
) -> ModelProfile:
    """
    Get memory profile for a specific model.

    Args:
        model_name: Model identifier
        memory_tier: Memory tier (auto-detect if None)
        available_memory_gb: Available memory in GB (for auto-tier detection)

    Returns:
        ModelProfile for the model and tier

    Raises:
        ValueError: If model not found

    Example:
        >>> profile = get_model_profile("SmolLM2-135M")
        >>> print(f"Max tokens: {profile.max_tokens}")
        >>>
        >>> # Specific tier
        >>> profile = get_model_profile("SmolVLM-256M", MemoryTier.LOW)
    """
    # Normalize model name
    model_key = model_name
    if model_key not in MODEL_PROFILES:
        # Try fuzzy matching
        for key in MODEL_PROFILES.keys():
            if key.lower() in model_name.lower() or model_name.lower() in key.lower():
                model_key = key
                break
        else:
            raise ValueError(
                f"Model profile not found for '{model_name}'. "
                f"Available: {list(MODEL_PROFILES.keys())}"
            )

    # Determine tier if not specified
    if memory_tier is None:
        memory_tier = determine_memory_tier(available_memory_gb)

    profile = MODEL_PROFILES[model_key][memory_tier]
    logger.info(
        f"Selected profile for {profile.model_name} ({memory_tier.value}): "
        f"{profile.description}"
    )
    return profile


def auto_select_params(
    model_name: str, available_memory_gb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Automatically select safe parameters for a model based on available memory.

    Args:
        model_name: Model identifier
        available_memory_gb: Available memory in GB (auto-detect if None)

    Returns:
        Dictionary of recommended parameters

    Example:
        >>> params = auto_select_params("SmolLM2-135M", available_memory_gb=20.0)
        >>> print(params)
        {'max_tokens': 1000, 'max_kv_size': 2048, 'batch_size': 2, ...}
    """
    profile = get_model_profile(model_name, available_memory_gb=available_memory_gb)

    return {
        'max_tokens': profile.max_tokens,
        'max_kv_size': profile.max_kv_size,
        'batch_size': profile.batch_size,
        'use_rotating_cache': profile.use_rotating_cache,
    }


def list_models() -> list[str]:
    """
    List all models with defined profiles.

    Returns:
        List of model names

    Example:
        >>> models = list_models()
        >>> print(models)
    """
    return list(MODEL_PROFILES.keys())


__all__ = [
    'MemoryTier',
    'ModelProfile',
    'MODEL_PROFILES',
    'determine_memory_tier',
    'get_model_profile',
    'auto_select_params',
    'list_models',
]
