"""
SMLX Models

Small models for Apple Silicon, run through maintained upstream MLX
implementations (mlx-lm / mlx-vlm / mlx-whisper / mlx-embeddings / mlx-audio) plus
SMLX's curation, quantization, unified API, and the fail-closed correctness gate.

Example:
    >>> from smlx.models import load, generate
    >>> m = load("smolvlm-256m")
    >>> generate(m, "What is in this image?", image="photo.jpg")
    >>>
    >>> # Run any curated model through one entrypoint
    >>> from smlx.models import runner
    >>> runner.produce("smollm2-135m", text="What is MLX?")
"""

from . import mlx_backend, runner
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
    infer_model_type,
    is_model_implemented,
    list_available_models,
    list_models_by_category,
    load_model,
)

__all__ = [
    # Unified upstream backend (recommended)
    "load",
    "generate",
    "mlx_backend",
    "BackendModel",
    "Backend",
    "ZOO",
    # Unified runner (smlx run)
    "runner",
    # Legacy universal loader (delegates to the backend)
    "load_model",
    "infer_model_type",
    "list_available_models",
    "list_models_by_category",
    "is_model_implemented",
    "get_model_info",
    "MODEL_REGISTRY",
    # MoE/Switch layers
    "SwitchLinear",
    "QuantizedSwitchLinear",
    "SwitchGLU",
    "SwitchMLP",
    "SwiGLU",
]
