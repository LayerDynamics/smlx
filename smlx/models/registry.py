"""
Model Registry for Universal Model Loading

Provides a centralized registry for all supported models in SMLX,
enabling automatic model discovery and loading by model ID or path.

Example:
    >>> from smlx.models.registry import load_model
    >>>
    >>> # Load by HuggingFace ID
    >>> model, tokenizer = load_model("mlx-community/SmolLM2-135M-Instruct")
    >>>
    >>> # Load with quantization
    >>> model, tokenizer = load_model(
    ...     "mlx-community/SmolLM2-135M-Instruct",
    ...     quantization="4bit"
    ... )
"""

from typing import Any, Dict, Optional


def _build_registry() -> Dict[str, str]:
    """Map each runnable alias to its modality, sourced from the runner."""
    from . import runner

    return {e.key: e.modality for e in runner.list_entries()}


# alias -> modality (language|vlm|asr|tts|ocr|embeddings|vad|audio_cls|cad)
MODEL_REGISTRY: Dict[str, str] = _build_registry()


def infer_model_type(model_path_or_id: str) -> Optional[str]:
    """
    Infer model type from model path or HuggingFace ID.

    Args:
        model_path_or_id: Model path or HuggingFace ID

    Returns:
        Model type key from MODEL_REGISTRY or None if unknown

    Example:
        >>> infer_model_type("mlx-community/SmolLM2-135M-Instruct")
        'smollm2-135m'
        >>> infer_model_type("openai/whisper-tiny")
        'whisper-tiny'
    """
    # Normalize the identifier
    identifier = model_path_or_id.lower()

    # Check for direct matches
    for key in MODEL_REGISTRY.keys():
        if key in identifier:
            return key

    # Check for common patterns
    patterns = {
        "smollm2-135m": ["smollm2-135m", "smollm-135m"],
        "smollm2-360m": ["smollm2-360m", "smollm-360m"],
        "whisper-tiny": ["whisper-tiny", "whisper_tiny"],
        "smolvlm-256m": ["smolvlm-256m", "smolvlm_256m"],
        "smolvlm-500m": ["smolvlm-500m", "smolvlm_500m"],
    }

    for model_type, pattern_list in patterns.items():
        if any(p in identifier for p in pattern_list):
            return model_type

    return None


def load_model(
    model_path_or_id: str,
    model_type: Optional[str] = None,
    quantization: Optional[str] = None,
    quantization_config: Optional[Dict[str, Any]] = None,
    detect_prequantized: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Any:
    """
    Universal model loader - loads any supported model by ID or path.

    This function provides a unified interface for loading all SMLX models,
    with automatic model type detection, pre-quantized model detection, and
    optional quantization.

    Args:
        model_path_or_id: HuggingFace model ID or local path
        model_type: Explicit model type (auto-detected if None)
        quantization: Quantization method to apply:
            - None: No quantization (default, or use pre-quantized if detected)
            - "4bit": 4-bit quantization
            - "8bit": 8-bit quantization
            - "gptq": GPTQ 4-bit
            - "awq": AWQ 4-bit
        quantization_config: Optional dict with quantization parameters:
            - group_size: Quantization group size
            - bits: Number of bits (overrides quantization method)
            - mode: Quantization mode ("affine")
        detect_prequantized: If True, detect and use pre-quantized weights (default: True)
        verbose: Print quantization info (default: False)
        **kwargs: Additional arguments passed to model's load() function

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If model type cannot be determined
        ImportError: If model is not implemented

    Examples:
        >>> # Basic loading (auto-detects pre-quantized models)
        >>> model, tokenizer = load_model("mlx-community/SmolLM2-135M-Instruct")
        >>>
        >>> # With quantization
        >>> model, tokenizer = load_model(
        ...     "mlx-community/SmolLM2-135M-Instruct",
        ...     quantization="4bit"
        ... )
        >>>
        >>> # Custom quantization config
        >>> model, tokenizer = load_model(
        ...     "mlx-community/SmolLM2-135M-Instruct",
        ...     quantization="4bit",
        ...     quantization_config={"group_size": 128, "bits": 4}
        ... )
        >>>
        >>> # Explicit model type
        >>> model, tokenizer = load_model(
        ...     "local/path/to/model",
        ...     model_type="smollm2-135m"
        ... )
    """
    del model_type, quantization_config, detect_prequantized, verbose  # legacy, unused
    from smlx.models import mlx_backend

    lazy = bool(kwargs.pop("lazy", False))
    return mlx_backend.load(model_path_or_id, quantize=quantization, lazy=lazy)


def list_available_models() -> Dict[str, str]:
    """
    List all available models in the registry.

    Returns:
        Dictionary mapping model types to module paths

    Example:
        >>> from smlx.models.registry import list_available_models
        >>> models = list_available_models()
        >>> for model_type, module_path in models.items():
        ...     print(f"{model_type}: {module_path}")
    """
    return MODEL_REGISTRY.copy()


def is_model_implemented(model_type: str) -> bool:
    """
    Check if a model type is implemented and loadable.

    Args:
        model_type: Model type key from MODEL_REGISTRY

    Returns:
        True if model can be loaded, False otherwise

    Example:
        >>> is_model_implemented("smollm2-135m")
        True
        >>> is_model_implemented("unknown-model")
        False
    """
    return model_type in MODEL_REGISTRY


def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get information about a model type.

    Args:
        model_type: Alias key from MODEL_REGISTRY

    Returns:
        Dictionary with model_type, modality, and is_implemented.

    Example:
        >>> get_model_info("smollm2-135m")["modality"]
        'language'
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    return {
        "model_type": model_type,
        "modality": MODEL_REGISTRY[model_type],
        "is_implemented": True,
    }


def list_models_by_category() -> Dict[str, list]:
    """
    List models grouped by modality.

    Example:
        >>> models = list_models_by_category()
        >>> models["language"]
        ['smollm2-135m', 'smollm2-360m']
    """
    categories: Dict[str, list] = {}
    for model_type, modality in MODEL_REGISTRY.items():
        categories.setdefault(modality, []).append(model_type)
    return categories
