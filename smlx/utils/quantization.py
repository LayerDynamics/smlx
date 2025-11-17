"""
Quantization Utilities for MLX Models

Provides utilities to apply different quantization methods to loaded MLX models.
Supports:
- FP16 (no quantization)
- 4-bit and 8-bit quantization
- GPTQ, AWQ, DWQ methods (with calibration)

Usage:
    from smlx.utils.quantization import apply_quantization
    from smlx.models.SmolLM2_135M import load

    # Load model
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Apply 4-bit quantization
    model = apply_quantization(model, method="4bit")

    # Apply 8-bit quantization
    model = apply_quantization(model, method="8bit")

Reference:
    - MLX quantization: https://ml-explore.github.io/mlx/build/html/python/nn.html#mlx.nn.quantize
    - mlx-lm implementation: resources/mlx-lm/mlx_lm/utils.py
"""

from typing import Any, Callable, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def has_quantizable_layers(model: nn.Module) -> bool:
    """
    Check if model has layers that support quantization.

    Args:
        model: Model to check

    Returns:
        True if model has layers with to_quantized() method
    """
    for name, module in model.named_modules():
        if hasattr(module, "to_quantized"):
            return True
    return False


def count_quantizable_layers(model: nn.Module) -> int:
    """
    Count number of quantizable layers in model.

    Args:
        model: Model to check

    Returns:
        Number of layers with to_quantized() method
    """
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "to_quantized"):
            count += 1
    return count


def get_quantization_config(method: str) -> Dict[str, Any]:
    """
    Get quantization configuration for a given method.

    Args:
        method: Quantization method name

    Returns:
        Dictionary with quantization parameters

    Raises:
        ValueError: If method is unknown
    """
    configs = {
        "fp16": None,  # No quantization
        "4bit": {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
        },
        "8bit": {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
        },
        # Future: GPTQ, AWQ, DWQ with calibration
        "gptq": {
            "bits": 4,
            "group_size": 128,
            "mode": "affine",
            "requires_calibration": True,
        },
        "awq": {
            "bits": 4,
            "group_size": 128,
            "mode": "affine",
            "requires_calibration": True,
        },
        "dwq": {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
            "requires_calibration": False,
        },
    }

    if method not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(
            f"Unknown quantization method: {method}. "
            f"Available methods: {available}"
        )

    return configs[method]


def create_class_predicate(
    weights: Optional[Dict[str, mx.array]] = None,
) -> Callable[[str, nn.Module], bool]:
    """
    Create a class predicate function for nn.quantize().

    The predicate determines which layers should be quantized. By default,
    quantizes all layers that have a to_quantized() method. If weights are
    provided, only quantizes layers that have corresponding scale weights
    (indicating they were pre-quantized).

    Args:
        weights: Optional model weights dict to check for pre-quantized layers

    Returns:
        Predicate function (path, module) -> bool
    """
    def predicate(path: str, module: nn.Module) -> bool:
        # Check if module supports quantization
        if not hasattr(module, "to_quantized"):
            return False

        # If weights provided, check if layer has quantization scales
        if weights is not None:
            return f"{path}.scales" in weights

        # Otherwise, quantize all quantizable layers
        return True

    return predicate


def apply_quantization(
    model: nn.Module,
    method: str = "fp16",
    group_size: Optional[int] = None,
    bits: Optional[int] = None,
    mode: Optional[str] = None,
    weights: Optional[Dict[str, mx.array]] = None,
    class_predicate: Optional[Callable[[str, nn.Module], bool]] = None,
    calibration_data: Optional[mx.array] = None,
    tokenizer: Optional[Any] = None,
    awq_config: Optional[Any] = None,
    verbose: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Apply quantization to a loaded model.

    This function applies quantization to an already-loaded MLX model using
    the MLX nn.quantize() API or advanced methods like GPTQ/AWQ. It modifies
    the model in-place by replacing quantizable layers (typically nn.Linear)
    with their quantized equivalents (QuantizedLinear).

    Args:
        model: The loaded model to quantize
        method: Quantization method name:
            - "fp16": No quantization (returns model unchanged)
            - "4bit": 4-bit quantization
            - "8bit": 8-bit quantization
            - "gptq": GPTQ 4-bit (requires calibration_data)
            - "awq": AWQ 4-bit (requires calibration_data and awq_config)
            - "dwq": Dynamic Weight Quantization 8-bit
        group_size: Quantization group size (overrides method default)
        bits: Number of bits (overrides method default)
        mode: Quantization mode (overrides method default), typically "affine"
        weights: Optional model weights dict (for detecting pre-quantized layers)
        class_predicate: Optional custom predicate to select which layers to quantize
        calibration_data: Calibration data for GPTQ/AWQ (required for those methods)
        tokenizer: Tokenizer for loading calibration data (alternative to calibration_data)
        awq_config: AWQ configuration for model-specific scaling (required for AWQ)
        verbose: Print quantization progress
        **kwargs: Additional arguments passed to GPTQ/AWQ functions

    Returns:
        The quantized model (modified in-place, returned for convenience)

    Raises:
        ValueError: If method is unknown or missing required parameters
        RuntimeError: If model has no quantizable layers

    Example:
        >>> from smlx.models.SmolLM2_135M import load
        >>> from smlx.utils.quantization import apply_quantization
        >>>
        >>> # Load model
        >>> model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        >>>
        >>> # Apply 4-bit quantization
        >>> model = apply_quantization(model, method="4bit", verbose=True)
        >>> # Quantizing 52 layers with 4-bit precision...
        >>>
        >>> # Apply GPTQ quantization
        >>> from smlx.quant.utils import load_calibration_data
        >>> calibration_data = load_calibration_data(tokenizer)
        >>> model = apply_quantization(
        ...     model,
        ...     method="gptq",
        ...     calibration_data=calibration_data,
        ...     verbose=True
        ... )
        >>>
        >>> # Apply AWQ quantization
        >>> from smlx.quant.awq import llama_awq
        >>> model = apply_quantization(
        ...     model,
        ...     method="awq",
        ...     calibration_data=calibration_data,
        ...     awq_config=llama_awq,
        ...     verbose=True
        ... )

    Notes:
        - Quantization is applied IN-PLACE. The original model is modified.
        - fp16 method returns the model unchanged (no quantization).
        - The model must have layers with to_quantized() method (e.g., nn.Linear).
        - After quantization, the model size and memory usage are reduced.
        - Generation speed may improve on Apple Silicon with quantized models.
        - GPTQ and AWQ require calibration data for optimal accuracy.

    Reference:
        Based on mlx-lm implementation:
        https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
    """
    # Get configuration for method
    config = get_quantization_config(method)

    # FP16 means no quantization
    if config is None:
        if verbose:
            print("FP16: No quantization applied")
        return model

    # Override config with explicit parameters
    if group_size is not None:
        config["group_size"] = group_size
    if bits is not None:
        config["bits"] = bits
    if mode is not None:
        config["mode"] = mode

    # Handle methods that require calibration
    if config.get("requires_calibration", False):
        # Load calibration data if tokenizer provided
        if calibration_data is None and tokenizer is not None:
            from smlx.quant.utils import load_calibration_data

            if verbose:
                print("Loading calibration data...")
            calibration_data = load_calibration_data(
                tokenizer,
                num_samples=kwargs.get("num_samples", 128),
                sequence_length=kwargs.get("sequence_length", 512),
                verbose=verbose,
            )

        # Check if calibration data is provided
        if calibration_data is None:
            raise ValueError(
                f"Method '{method}' requires calibration_data parameter. "
                f"Either provide calibration_data directly or pass tokenizer to load it automatically."
            )

        # Apply GPTQ quantization
        if method == "gptq":
            from smlx.quant.gptq import gptq_quantize

            return gptq_quantize(
                model,
                calibration_data,
                bits=config["bits"],
                group_size=config["group_size"],
                batch_size=kwargs.get("batch_size", 8),
            )

        # Apply AWQ quantization
        elif method == "awq":
            from smlx.quant.awq import awq_quantize, llama_awq

            # Use provided config or default to llama_awq
            if awq_config is None:
                if verbose:
                    print("No AWQ config provided, using default llama_awq configuration")
                awq_config = llama_awq

            return awq_quantize(
                model,
                calibration_data,
                awq_config=awq_config,
                bits=config["bits"],
                group_size=config["group_size"],
                embed_bits=kwargs.get("embed_bits", 4),
                embed_group_size=kwargs.get("embed_group_size", 32),
                n_grid=kwargs.get("n_grid", 20),
            )

    # Check if model has quantizable layers (for basic quantization)
    if not has_quantizable_layers(model):
        raise RuntimeError(
            "Model has no quantizable layers. "
            "Ensure model uses nn.Linear or other layers with to_quantized() method."
        )

    if verbose:
        num_layers = count_quantizable_layers(model)
        print(
            f"Quantizing {num_layers} layers with {config['bits']}-bit precision "
            f"(group_size={config['group_size']}, mode={config['mode']})..."
        )

    # Create class predicate if not provided
    if class_predicate is None:
        class_predicate = create_class_predicate(weights)

    # Apply quantization using MLX nn.quantize()
    # This walks the model tree and calls to_quantized() on eligible layers
    nn.quantize(
        model,
        group_size=config["group_size"],
        bits=config["bits"],
        mode=config.get("mode", "affine"),
        class_predicate=class_predicate,
    )

    if verbose:
        print(f"Quantization complete!")

    return model


def estimate_quantized_size(
    model: nn.Module,
    method: str = "fp16",
) -> float:
    """
    Estimate size of model after quantization.

    Args:
        model: Model to estimate
        method: Quantization method

    Returns:
        Estimated size in GB
    """
    # Count parameters (handle nested dict structure)
    def count_params(params):
        count = 0
        for v in params.values():
            if isinstance(v, dict):
                count += count_params(v)
            else:
                count += v.size
        return count

    try:
        total_params = count_params(model.parameters())
    except Exception:
        # Fallback: estimate from model type
        total_params = 135_000_000  # Assume SmolLM2-135M

    # Bytes per parameter based on quantization
    bytes_per_param = {
        "fp16": 2,      # 16-bit = 2 bytes
        "8bit": 1,      # 8-bit = 1 byte
        "4bit": 0.5,    # 4-bit = 0.5 bytes
        "gptq": 0.5,    # GPTQ typically 4-bit
        "awq": 0.5,     # AWQ typically 4-bit
        "dwq": 1,       # DWQ typically 8-bit
    }

    bytes_per = bytes_per_param.get(method, 2)
    size_bytes = total_params * bytes_per
    size_gb = size_bytes / (1024**3)

    return size_gb


def get_quantization_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about model quantization status.

    Args:
        model: Model to inspect

    Returns:
        Dictionary with quantization information:
            - is_quantized: Whether model has quantized layers
            - num_quantized: Number of quantized layers
            - num_quantizable: Number of layers that could be quantized
            - quantized_layers: List of quantized layer paths
    """
    quantized_layers = []
    quantizable_layers = []

    for path, module in model.named_modules():
        # Check if layer is already quantized
        if hasattr(module, "scales"):  # QuantizedLinear has scales
            quantized_layers.append(path)
        # Check if layer can be quantized
        elif hasattr(module, "to_quantized"):
            quantizable_layers.append(path)

    return {
        "is_quantized": len(quantized_layers) > 0,
        "num_quantized": len(quantized_layers),
        "num_quantizable": len(quantizable_layers),
        "quantized_layers": quantized_layers,
        "quantizable_layers": quantizable_layers,
    }
