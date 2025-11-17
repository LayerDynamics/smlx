"""
Quantization and parameter-efficient fine-tuning for SMLX.

This module provides quantization techniques and low-rank adaptation methods
optimized for "smol" models (<10B parameters) on Apple M4 chipsets.

Supported Methods:
    - LoRA: Low-Rank Adaptation for parameter-efficient fine-tuning
    - DoRA: Weight-Decomposed Low-Rank Adaptation
    - GPTQ: Post-training quantization with Hessian-based error compensation
    - Dynamic: Mixed-precision quantization based on layer sensitivity
    - AWQ: Activation-aware weight quantization (MLSys 2024 Best Paper)
    - DWQ: Distilled weight quantization with knowledge distillation

Bit-Width Specific:
    - 4bit: 4-bit integer quantization wrappers
    - 6bit: 6-bit integer quantization wrappers
    - 8bit: 8-bit integer quantization wrappers
    - BFloat16: Brain Float 16 conversion utilities

Floating Point Formats:
    - FP4: 4-bit floating point (E2M1) - simulated
    - FP8: 8-bit floating point (E4M3, E5M2) - ⚠️ DEPRECATED (simulated, use MXFP8)
    - MXFP4: Microscaling FP4 (E2M1 + E8M0 scale, OCP standard) - ✅ True 4-bit
    - MXFP8: Microscaling FP8 (E4M3 + E8M0 scale, OCP standard) - ✅ True 8-bit

Mixed-Precision:
    - mixed_bit: General mixed-bit framework with custom strategies
    - mixed_3_6: Specific 3-6 bit mixed-precision strategy

Example:
    ```python
    import mlx.nn as nn
    from smlx.quant import LoRALinear, load_calibration_data, quantize_4bit

    # Create LoRA layer from existing layer
    linear = nn.Linear(768, 768)
    lora_layer = LoRALinear.from_base(linear, r=8)

    # Train with LoRA
    # ... training loop ...

    # Fuse LoRA weights for deployment
    fused_layer = lora_layer.fuse()

    # Or quantize a model to 4-bit
    from smlx.models.SmolLM2_135M import load
    model, _ = load("mlx-community/SmolLM2-135M-Instruct")
    quantize_4bit(model)  # In-place quantization
    ```
"""

# LoRA (Low-Rank Adaptation)
# AWQ (Activation-Aware Weight Quantization)
# Bit-width specific quantization (use importlib for numeric module names)
import importlib

from .awq import (
    AWQConfig,
    ScaleConfig,
    awq_quantize,
    llama_awq,
    mistral_awq,
    qwen_awq,
)

# DoRA (Weight-Decomposed Low-Rank Adaptation)
from .dora import DoRAEmbedding, DoRALinear, DoRASwitchLinear

# DWQ (Distilled Weight Quantization)
from .dwq import dwq_quantize, dwq_quantize_simple

# Dynamic Quantization (Mixed-Precision)
from .dynamic_quant import dynamic_quantize, estimate_sensitivities, estimate_threshold

# GPTQ (GPT Quantization)
from .gptq import Catcher, gptq_quantize
from .lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

# Utilities
from .utils import (
    check_m4_compatibility,
    estimate_model_size,
    get_actual_model_size,
    load_calibration_data,
    measure_memory_savings,
    quantize_dequantize,
)

_bit4 = importlib.import_module("smlx.quant.4bit")
_bit6 = importlib.import_module("smlx.quant.6bit")
_bit8 = importlib.import_module("smlx.quant.8bit")

# 4-bit wrappers
quantize_4bit = _bit4.quantize_4bit
quantize_weights_4bit = _bit4.quantize_weights_4bit
dequantize_weights_4bit = _bit4.dequantize_weights_4bit
estimate_4bit_size_reduction = _bit4.estimate_4bit_size_reduction
is_4bit_quantized = _bit4.is_4bit_quantized
get_quantization_info = _bit4.get_quantization_info

# 6-bit wrappers
quantize_6bit = _bit6.quantize_6bit
quantize_weights_6bit = _bit6.quantize_weights_6bit
dequantize_weights_6bit = _bit6.dequantize_weights_6bit
estimate_6bit_size_reduction = _bit6.estimate_6bit_size_reduction
is_6bit_quantized = _bit6.is_6bit_quantized

# 8-bit wrappers
quantize_8bit = _bit8.quantize_8bit
quantize_weights_8bit = _bit8.quantize_weights_8bit
dequantize_weights_8bit = _bit8.dequantize_weights_8bit
estimate_8bit_size_reduction = _bit8.estimate_8bit_size_reduction
is_8bit_quantized = _bit8.is_8bit_quantized
compare_with_4bit = _bit8.compare_with_4bit

# Automatic quantization
from .autoquant import (
    analyze_model,
    autoquant,
    detect_hardware_capabilities,
    recommend_strategy,
    select_strategy,
)

# BFloat16
from .bf16 import (
    compare_dtypes,
    convert_to_bfloat16,
    estimate_bfloat16_size,
    is_bfloat16,
    mixed_precision_bf16_fp32,
    weights_from_bfloat16,
    weights_to_bfloat16,
)

# Floating point quantization
from .fp4 import (
    FP4_E2M1_VALUES,
    NF4_VALUES,
    FP4Mode,
    # New unified API
    quantize_fp4,
    dequantize_fp4,
    # Legacy API
    quantize_to_fp4,
    dequantize_from_fp4,
    # Model-level
    quantize_model_fp4,
    estimate_fp4_size,
    compare_fp4_vs_int4,
)

# FP8 - DEPRECATED (simulated only, use MXFP8 instead)
# These imports are kept for backward compatibility but emit deprecation warnings
from .fp8 import (
    compare_fp8_formats,
    compare_fp8_vs_int8,
    dequantize_from_fp8,
    estimate_fp8_size,
    quantize_model_fp8,
    quantize_to_fp8_e4m3,
    quantize_to_fp8_e5m2,
)

# Mixed-precision
from .mixed_3_6 import (
    create_custom_3_6_strategy,
    get_recommended_strategy,
    quantize_3_6_mixed,
)
from .mixed_bit import (
    MixedBitStrategy,
    QuantizationRule,
    analyze_quantization_distribution,
    apply_mixed_bit_quantization,
    compute_average_bpw,
    create_balanced_strategy,
    create_layerwise_strategy,
)

# MLX-native mixed-precision (Q4_K_M-style)
from .mlx_mixed import (
    create_q4_k_m_style_predicate,
    estimate_mixed_size,
    quantize_model_mixed,
)

# Microscaling floating point (MXFP - OCP standard)
from .mxfp4 import (
    compare_mxfp4_vs_fp4,
    compare_mxfp4_vs_int4,
    dequantize_from_mxfp4,
    estimate_mxfp4_size,
    quantize_model_mxfp4,
    quantize_to_mxfp4,
    validate_mxfp_shape,
)
from .mxfp8 import (
    compare_mxfp8_vs_fp8,
    compare_mxfp8_vs_int8,
    dequantize_from_mxfp8,
    estimate_mxfp8_size,
    quantize_model_mxfp8,
    quantize_to_mxfp8,
)

# GGML formats (llama.cpp compatible)
from .q4_0 import (
    Q4_0_BLOCK_SIZE,
    Q4_0_BYTES_PER_BLOCK,
    compare_q4_0_vs_q4_1,
    dequantize_from_q4_0,
    dequantize_model_q4_0,
    estimate_q4_0_size,
    quantize_model_q4_0,
    quantize_to_q4_0,
)
from .q4_1 import (
    Q4_1_BLOCK_SIZE,
    Q4_1_BYTES_PER_BLOCK,
    dequantize_from_q4_1,
    estimate_q4_1_size,
    quantize_model_q4_1,
    quantize_to_q4_1,
)
from .q4_k_m import (
    Q4_K_BLOCK_SIZE,
    Q4_K_NUM_SUBBLOCKS,
    Q4_K_SUBBLOCK_SIZE,
    dequantize_from_q4_k,
    estimate_q4_k_size,
    quantize_model_q4_k,
    quantize_model_q4_k_m,
    quantize_to_q4_k,
)
from .q8_0 import (
    Q8_0_BLOCK_SIZE,
    Q8_0_BYTES_PER_BLOCK,
    compare_q8_0_vs_int8,
    dequantize_from_q8_0,
    estimate_q8_0_size,
    quantize_model_q8_0,
    quantize_to_q8_0,
)


def quantize_model(model, bits: int = 4, group_size: int = 64):
    """
    Simple helper to quantize a model using MLX's built-in quantization.

    This is a convenience wrapper for demos and quick quantization. For production
    use with better quality preservation, use gptq_quantize or awq_quantize instead.

    Args:
        model: MLX model to quantize
        bits: Bits per weight (4 or 8, default: 4)
        group_size: Group size for quantization (default: 64)

    Returns:
        Quantized model

    Example:
        >>> from smlx.models.SmolLM2_135M import load
        >>> from smlx.quant import quantize_model
        >>> model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        >>> quantized_model = quantize_model(model, bits=4, group_size=64)
    """
    import mlx.nn as nn

    # Use MLX's built-in nn.quantize which converts Linear/Embedding layers
    nn.quantize(model, group_size=group_size, bits=bits)
    return model


__all__ = [
    # LoRA
    "LoRALinear",
    "LoRAEmbedding",
    "LoRASwitchLinear",
    # DoRA
    "DoRALinear",
    "DoRAEmbedding",
    "DoRASwitchLinear",
    # GPTQ
    "gptq_quantize",
    "Catcher",
    # Dynamic Quantization
    "dynamic_quantize",
    "estimate_sensitivities",
    "estimate_threshold",
    # AWQ
    "awq_quantize",
    "AWQConfig",
    "ScaleConfig",
    "llama_awq",
    "mistral_awq",
    "qwen_awq",
    # DWQ
    "dwq_quantize",
    "dwq_quantize_simple",
    # Utilities
    "load_calibration_data",
    "estimate_model_size",
    "get_actual_model_size",
    "measure_memory_savings",
    "quantize_dequantize",
    "check_m4_compatibility",
    "quantize_model",
    # 4-bit
    "quantize_4bit",
    "quantize_weights_4bit",
    "dequantize_weights_4bit",
    "estimate_4bit_size_reduction",
    "is_4bit_quantized",
    "get_quantization_info",
    # 6-bit
    "quantize_6bit",
    "quantize_weights_6bit",
    "dequantize_weights_6bit",
    "estimate_6bit_size_reduction",
    "is_6bit_quantized",
    # 8-bit
    "quantize_8bit",
    "quantize_weights_8bit",
    "dequantize_weights_8bit",
    "estimate_8bit_size_reduction",
    "is_8bit_quantized",
    "compare_with_4bit",
    # BFloat16
    "convert_to_bfloat16",
    "weights_to_bfloat16",
    "weights_from_bfloat16",
    "is_bfloat16",
    "estimate_bfloat16_size",
    "mixed_precision_bf16_fp32",
    "compare_dtypes",
    # FP4 (new unified API)
    "quantize_fp4",
    "dequantize_fp4",
    "FP4Mode",
    # FP4 (legacy API)
    "quantize_to_fp4",
    "dequantize_from_fp4",
    # FP4 (model-level and utils)
    "quantize_model_fp4",
    "estimate_fp4_size",
    "compare_fp4_vs_int4",
    "FP4_E2M1_VALUES",
    "NF4_VALUES",
    # FP8 (DEPRECATED - use MXFP8 instead)
    "quantize_to_fp8_e4m3",
    "quantize_to_fp8_e5m2",
    "dequantize_from_fp8",
    "quantize_model_fp8",
    "estimate_fp8_size",
    "compare_fp8_formats",
    "compare_fp8_vs_int8",
    # MXFP4
    "quantize_to_mxfp4",
    "dequantize_from_mxfp4",
    "quantize_model_mxfp4",
    "estimate_mxfp4_size",
    "compare_mxfp4_vs_fp4",
    "compare_mxfp4_vs_int4",
    "validate_mxfp_shape",
    # MXFP8
    "quantize_to_mxfp8",
    "dequantize_from_mxfp8",
    "quantize_model_mxfp8",
    "estimate_mxfp8_size",
    "compare_mxfp8_vs_fp8",
    "compare_mxfp8_vs_int8",
    # GGML Q4_0
    "quantize_to_q4_0",
    "dequantize_from_q4_0",
    "quantize_model_q4_0",
    "dequantize_model_q4_0",
    "estimate_q4_0_size",
    "compare_q4_0_vs_q4_1",
    "Q4_0_BLOCK_SIZE",
    "Q4_0_BYTES_PER_BLOCK",
    # GGML Q4_1
    "quantize_to_q4_1",
    "dequantize_from_q4_1",
    "quantize_model_q4_1",
    "estimate_q4_1_size",
    "Q4_1_BLOCK_SIZE",
    "Q4_1_BYTES_PER_BLOCK",
    # GGML Q4_K and Q4_K_M
    "quantize_to_q4_k",
    "dequantize_from_q4_k",
    "quantize_model_q4_k",
    "quantize_model_q4_k_m",
    "estimate_q4_k_size",
    "Q4_K_BLOCK_SIZE",
    "Q4_K_NUM_SUBBLOCKS",
    "Q4_K_SUBBLOCK_SIZE",
    # GGML Q8_0
    "quantize_to_q8_0",
    "dequantize_from_q8_0",
    "quantize_model_q8_0",
    "estimate_q8_0_size",
    "compare_q8_0_vs_int8",
    "Q8_0_BLOCK_SIZE",
    "Q8_0_BYTES_PER_BLOCK",
    # Mixed-bit
    "MixedBitStrategy",
    "QuantizationRule",
    "apply_mixed_bit_quantization",
    "compute_average_bpw",
    "create_balanced_strategy",
    "create_layerwise_strategy",
    "analyze_quantization_distribution",
    # Mixed 3-6
    "quantize_3_6_mixed",
    "create_custom_3_6_strategy",
    "get_recommended_strategy",
    # Automatic quantization
    "autoquant",
    "analyze_model",
    "select_strategy",
    "recommend_strategy",
]

__version__ = "0.1.0"
