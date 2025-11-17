# Copyright © 2025 SMLX Project

"""
SmolLM2-360M-Instruct: Lightweight language model (360M parameters).

This module provides a complete implementation of SmolLM2-360M-Instruct,
which uses the SmolLM3 architecture with NoPE (No Positional Encoding).

Quick Start:
    >>> from smlx.models.SmolLM2_360M import load, generate
    >>>
    >>> # Load model
    >>> model, tokenizer = load("mlx-community/SmolLM2-360M-Instruct")
    >>>
    >>> # Generate text
    >>> prompt = "Write a Python function to"
    >>> response = generate(
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     prompt=prompt,
    ...     max_tokens=100,
    ...     temperature=0.7
    ... )
    >>> print(response)

Architecture:
    - 360M parameters
    - 32 layers (960 hidden dim)
    - Grouped Query Attention (15 heads, 5 KV heads)
    - SmolLM3 with NoPE (selective RoPE disabling)
    - Context length: 8192 tokens
    - Vocabulary: 49,152 tokens

Available Functions:
    - load(): Load model and tokenizer from HuggingFace
    - generate(): Generate text from prompt
    - chat(): Interactive chat interface
    - stream_generate(): Stream generated tokens

Available Classes:
    - Model: Main model class
    - ModelArgs: Configuration dataclass
"""

from .cache import KVCache, RotatingKVCache, make_cache
from .config import (
    DEFAULT_CONFIG,
    get_default_config,
    load_config,
    print_config,
    validate_config,
)
from .generate import (
    GenerationConfig,
    chat,
    complete,
    generate,
    generate_step,
    sample,
    stream_generate,
)
from .loader import load, load_model_from_path, save_model
from .model import (
    MLP,
    Attention,
    LlamaModel,
    Model,
    ModelArgs,
    NoPE,
    TransformerBlock,
)

__all__ = [
    # Core model classes
    "Model",
    "ModelArgs",
    "Attention",
    "MLP",
    "TransformerBlock",
    "LlamaModel",
    "NoPE",
    # Configuration
    "DEFAULT_CONFIG",
    "load_config",
    "get_default_config",
    "validate_config",
    "print_config",
    # Cache
    "KVCache",
    "RotatingKVCache",
    "make_cache",
    # Loading
    "load",
    "load_model_from_path",
    "save_model",
    # Generation
    "generate",
    "stream_generate",
    "chat",
    "complete",
    "generate_step",
    "sample",
    "GenerationConfig",
]

# Model metadata
__version__ = "0.1.0"
__model_name__ = "SmolLM2-360M-Instruct"
__model_size__ = "360M"
__architecture__ = "SmolLM3 (Llama with NoPE)"
__context_length__ = 8192
__vocab_size__ = 49152
