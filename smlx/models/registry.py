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

import importlib
from typing import Any, Dict, Optional, Tuple

import mlx.nn as nn

# Model registry mapping model patterns to loader modules
MODEL_REGISTRY = {
    # Language Models
    "smollm2-135m": "smlx.models.SmolLM2_135M",
    "smollm2-360m": "smlx.models.SmolLM2_360M",
    "smollm2": "smlx.models.SmolLM2_135M",  # Default to 135M
    "chatterbox": "smlx.models.Chatterbox",
    # Vision-Language Models
    "smolvlm-256m": "smlx.models.SmolVLM_256M",
    "smolvlm-500m": "smlx.models.SmolVLM_500M_Instruct",
    "smolvlm": "smlx.models.SmolVLM_256M",  # Default to 256M
    "moondream2": "smlx.models.Moondream2",
    "tinyllava": "smlx.models.TinyLLaVA",
    "nanovlm": "smlx.models.nanoVLM",
    # Audio Models
    "whisper-tiny": "smlx.models.Whisper_tiny",
    "whisper": "smlx.models.Whisper_tiny",  # Default to tiny
    "orpheus-150m": "smlx.models.Orpheus_150M",
    "orpheus": "smlx.models.Orpheus_150M",
    "yamnet": "smlx.models.YAMNet",
    "silero-vad": "smlx.models.SileroVAD",
    # Document/OCR Models
    "trocr-small": "smlx.models.TrOCR_small",
    "trocr": "smlx.models.TrOCR_small",
    "donut-base": "smlx.models.Donut_base",
    "donut": "smlx.models.Donut_base",
    # Embedding Models
    "minilm": "smlx.models.MiniLM",
    "all-minilm-l6-v2": "smlx.models.all_MiniLM_L6_v2",
}


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


def get_model_loader(model_type: str):
    """
    Get the loader module for a given model type.

    Args:
        model_type: Model type key from MODEL_REGISTRY

    Returns:
        Module with load() function

    Raises:
        ImportError: If model module cannot be loaded
        ValueError: If model type is unknown
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}\nSupported types: {', '.join(MODEL_REGISTRY.keys())}"
        )

    module_path = MODEL_REGISTRY[model_type]

    try:
        module = importlib.import_module(module_path)
        if not hasattr(module, "load"):
            raise ImportError(f"Module {module_path} does not have a load() function")
        return module
    except ImportError as e:
        raise ImportError(
            f"Failed to import model module {module_path}: {e}\n"
            f"The model may not be implemented yet."
        )


def load_model(
    model_path_or_id: str,
    model_type: Optional[str] = None,
    quantization: Optional[str] = None,
    quantization_config: Optional[Dict[str, Any]] = None,
    detect_prequantized: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Tuple[nn.Module, Any]:
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
    # Infer model type if not provided
    if model_type is None:
        model_type = infer_model_type(model_path_or_id)
        if model_type is None:
            raise ValueError(
                f"Could not infer model type from: {model_path_or_id}\n"
                f"Please specify model_type explicitly.\n"
                f"Supported types: {', '.join(MODEL_REGISTRY.keys())}"
            )

    # Get the loader module
    loader_module = get_model_loader(model_type)

    # Check for pre-quantized weights before loading
    prequant_info = None
    if detect_prequantized:
        from smlx.utils.loading import detect_quantization, load_weights, resolve_model_path

        try:
            # Resolve model path and load weights to check for quantization
            model_path = resolve_model_path(model_path_or_id)
            weights = load_weights(model_path, lazy=True)
            prequant_info = detect_quantization(weights)

            if prequant_info and verbose:
                print("✓ Detected pre-quantized model:")
                print(f"  - Quantized layers: {prequant_info['num_quantized_layers']}")
                print(f"  - Estimated bits: {prequant_info['estimated_bits']}")

        except Exception:
            # If detection fails, proceed without it
            pass

    # Load the model
    model, tokenizer = loader_module.load(model_path_or_id, **kwargs)

    # Handle quantization based on detection results
    if prequant_info:
        # Model is pre-quantized
        if quantization is not None:
            # User requested explicit quantization on pre-quantized model
            if verbose:
                print(
                    f"Warning: Applying {quantization} to already quantized model. "
                    f"This may reduce quality."
                )
            from smlx.utils.quantization import apply_quantization

            quant_kwargs = quantization_config or {}
            model = apply_quantization(model, method=quantization, verbose=verbose, **quant_kwargs)
        elif verbose:
            print("✓ Using pre-quantized weights")

    elif quantization is not None:
        # Model is not pre-quantized, apply requested quantization
        from smlx.utils.quantization import apply_quantization

        quant_kwargs = quantization_config or {}
        model = apply_quantization(model, method=quantization, verbose=verbose, **quant_kwargs)

    return model, tokenizer


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
    if model_type not in MODEL_REGISTRY:
        return False

    try:
        get_model_loader(model_type)
        return True
    except ImportError:
        return False


def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get information about a model type.

    Args:
        model_type: Model type key from MODEL_REGISTRY

    Returns:
        Dictionary with model information:
            - model_type: The model type key
            - module_path: Python module path
            - is_implemented: Whether model is implemented
            - category: Model category (language, vision, audio, etc.)

    Example:
        >>> info = get_model_info("smollm2-135m")
        >>> print(info["category"])
        'language'
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    # Determine category based on module path
    module_path = MODEL_REGISTRY[model_type]

    if "SmolLM" in module_path or "Chatterbox" in module_path:
        category = "language"
    elif "VLM" in module_path or "LLaVA" in module_path or "Moondream" in module_path:
        category = "vision-language"
    elif (
        "Whisper" in module_path
        or "Orpheus" in module_path
        or "YAMNet" in module_path
        or "VAD" in module_path
    ):
        category = "audio"
    elif "TrOCR" in module_path or "Donut" in module_path:
        category = "document"
    elif "MiniLM" in module_path:
        category = "embedding"
    else:
        category = "unknown"

    return {
        "model_type": model_type,
        "module_path": module_path,
        "is_implemented": is_model_implemented(model_type),
        "category": category,
    }


def list_models_by_category() -> Dict[str, list]:
    """
    List models grouped by category.

    Returns:
        Dictionary mapping categories to lists of model types

    Example:
        >>> models = list_models_by_category()
        >>> print("Language models:", models["language"])
        >>> print("Audio models:", models["audio"])
    """
    categories = {}

    for model_type in MODEL_REGISTRY.keys():
        info = get_model_info(model_type)
        category = info["category"]

        if category not in categories:
            categories[category] = []
        categories[category].append(model_type)

    return categories
