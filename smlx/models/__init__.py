"""
SMLX Models

Collection of "smol" models (< 1B parameters) optimized for Apple Silicon.

This module provides:
- Fully implemented models (SmolLM2-135M, Whisper-tiny)
- Model registry for universal loading
- Automatic model type detection
- Model Execution Framework (Router, Runner, Manager)

Example:
    >>> from smlx.models import load_model
    >>>
    >>> # Load any model with auto-detection
    >>> model, tokenizer = load_model("mlx-community/SmolLM2-135M-Instruct")
    >>>
    >>> # Load with quantization
    >>> model, tokenizer = load_model(
    ...     "mlx-community/SmolLM2-135M-Instruct",
    ...     quantization="4bit"
    ... )
    >>>
    >>> # Use Model Execution Framework
    >>> from smlx.models import ModelRunner, InferenceConfig, get_router
    >>>
    >>> # Simple inference
    >>> runner = ModelRunner()
    >>> config = InferenceConfig(max_tokens=100, temperature=0.7)
    >>> result = runner.run(
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     model_type="smollm2-135m",
    ...     prompt="Hello world"
    ... )
"""

# Import registry functions for easy access
# Unified upstream backend (the recommended API): runs models through the correct
# mlx-lm / mlx-vlm implementations and applies SMLX's quantization on top.
#   from smlx.models import load, generate
#   m = load("smolvlm-256m", quantize="4bit"); generate(m, prompt, image=path)
from . import mlx_backend

# Import MoE/Switch layers for easy access
from .common.switch_layers import (
    QuantizedSwitchLinear,
    SwiGLU,
    SwitchGLU,
    SwitchLinear,
    SwitchMLP,
)
from .mlx_backend import ZOO, Backend, BackendModel
from .mlx_backend import generate as generate
from .mlx_backend import load as load
from .registry import (
    MODEL_REGISTRY,
    get_model_info,
    get_model_loader,
    infer_model_type,
    is_model_implemented,
    list_available_models,
    list_models_by_category,
    load_model,
)
from .smlx_manager import (
    CacheConfig,
    ModelCache,
    ModelLifecycleManager,
    ModelStats,
    ModelTelemetry,
    TelemetryConfig,
    get_manager,
)

# Import Model Execution Framework
from .smlx_router import ModelCapabilities, ModelRouter, get_router
from .smlx_runner import (
    InferenceConfig,
    ModelRunner,
    preprocess_audio_input,
    preprocess_chat_input,
    preprocess_image_input,
    preprocess_text_input,
)

__all__ = [
    # Unified upstream backend (recommended)
    "load",
    "generate",
    "mlx_backend",
    "BackendModel",
    "Backend",
    "ZOO",
    # Universal model loading (legacy custom implementations)
    "load_model",
    "infer_model_type",
    # Registry queries
    "list_available_models",
    "list_models_by_category",
    "is_model_implemented",
    "get_model_info",
    "get_model_loader",
    # Constants
    "MODEL_REGISTRY",
    # Model Execution Framework - Router
    "ModelRouter",
    "ModelCapabilities",
    "get_router",
    # Model Execution Framework - Runner
    "ModelRunner",
    "InferenceConfig",
    "preprocess_text_input",
    "preprocess_chat_input",
    "preprocess_image_input",
    "preprocess_audio_input",
    # Model Execution Framework - Manager
    "ModelLifecycleManager",
    "get_manager",
    "CacheConfig",
    "TelemetryConfig",
    "ModelCache",
    "ModelTelemetry",
    "ModelStats",
    # MoE/Switch layers
    "SwitchLinear",
    "QuantizedSwitchLinear",
    "SwitchGLU",
    "SwitchMLP",
    "SwiGLU",
]
