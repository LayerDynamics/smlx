"""
Automatic quantization strategy selection for SMLX.

This module automatically analyzes models and selects optimal quantization strategies
based on model characteristics, memory constraints, and quality requirements.

The autoquant system intelligently chooses between:
- Uniform bit-width quantization (4/6/8-bit)
- Advanced methods (GPTQ, AWQ, DWQ, Dynamic)
- Mixed-precision strategies (mixed_3_6, mixed_bit, layerwise)
- Floating point formats (FP4/FP8, MXFP4/MXFP8, BFloat16)
- Hardware-optimized formats (OCP Microscaling on M4+)

Example:
    ```python
    from smlx.models.SmolLM2_135M import load
    from smlx.quant import autoquant

    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Automatic quantization with default settings
    result = autoquant(model)
    print(f"Selected strategy: {result['strategy']}")
    print(f"Reason: {result['reason']}")

    # Specify quality/size tradeoff
    result = autoquant(model, profile="aggressive")  # Maximize compression
    result = autoquant(model, profile="balanced")    # Balance quality/size
    result = autoquant(model, profile="conservative")  # Maximize quality

    # For training/fine-tuning
    result = autoquant(model, use_case="training")  # Optimizes for gradient precision

    # With memory constraints
    result = autoquant(model, target_memory_mb=100)
    ```
"""

import platform
import re
import subprocess
from typing import Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from .dynamic_quant import estimate_sensitivities


def detect_hardware_capabilities() -> dict:
    """
    Detect hardware capabilities for optimal quantization format selection.

    Returns:
        Dictionary with hardware capabilities:
        - ocp_microscaling: Whether OCP Microscaling formats (MXFP4/8) are supported
        - mlx_version: MLX version string
        - chip: Chip identifier (e.g., "M4", "M3", "M2")
        - supports_metal: Whether Metal GPU is available
    """
    capabilities = {
        "ocp_microscaling": False,
        "mlx_version": mx.__version__ if hasattr(mx, "__version__") else "unknown",
        "chip": "unknown",
        "supports_metal": True,  # MLX always uses Metal on Apple Silicon
    }

    # Detect Apple Silicon chip generation
    try:
        # On macOS, we can detect the chip from platform info
        machine = platform.machine()
        if machine == "arm64":
            # Use system_profiler to get exact chip model (most reliable)
            try:
                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                output = result.stdout

                # Parse chip name: "Chip: Apple M4 Max"
                chip_match = re.search(r"Chip:\s+Apple (M\d+)(\s+\w+)?", output)
                if chip_match:
                    generation = chip_match.group(1)  # "M1", "M2", "M3", "M4"
                    variant = chip_match.group(2)  # " Pro", " Max", " Ultra" or None

                    capabilities["chip"] = f"Apple {generation}{variant or ''}".strip()
                    capabilities["chip_generation"] = generation

                    # OCP Microscaling (MXFP) available on M3+
                    # M3 has preliminary support, M4 has full hardware acceleration
                    if generation in ("M3", "M4"):
                        capabilities["ocp_microscaling"] = True
                        capabilities["supports_mxfp4"] = True
                        capabilities["supports_mxfp8"] = True
                    elif generation in ("M1", "M2"):
                        # M1/M2 can use MXFP via software emulation
                        capabilities["ocp_microscaling"] = False
                        capabilities["supports_mxfp4"] = False
                        capabilities["supports_mxfp8"] = False

                    # Neural Engine on all M-series
                    capabilities["neural_engine"] = True

                # Parse memory: "Memory: 36 GB"
                memory_match = re.search(r"Memory:\s+(\d+)\s+GB", output)
                if memory_match:
                    capabilities["memory_gb"] = int(memory_match.group(1))

            except Exception:
                # Fallback: try sysctl
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    brand = result.stdout.strip()

                    # "Apple M4 Max"
                    chip_match = re.search(r"Apple (M\d+)", brand)
                    if chip_match:
                        generation = chip_match.group(1)
                        capabilities["chip"] = brand
                        capabilities["chip_generation"] = generation

                        if generation in ("M3", "M4"):
                            capabilities["ocp_microscaling"] = True
                            capabilities["supports_mxfp4"] = True
                            capabilities["supports_mxfp8"] = True
                except Exception:
                    # Last resort: generic Apple Silicon
                    capabilities["chip"] = "Apple Silicon (generic)"
                    capabilities["chip_generation"] = "unknown"
    except Exception:
        pass

    # Probe whether the installed MLX build can actually perform MXFP
    # microscaling quantization via the real mode="mxfp4"/"mxfp8" path. Plain
    # affine INT quantization (mx.quantize(..., bits=4)) always succeeds and
    # therefore says nothing about MXFP support, so it must NOT be used here.
    #
    # This probe determines *library* (software) support. Hardware acceleration
    # is the chip-based `ocp_microscaling` flag set during chip detection above
    # and is intentionally left untouched here.
    def _mxfp_mode_supported(mode: str) -> bool:
        try:
            probe = mx.random.normal((32, 32))
            result = mx.quantize(probe, mode=mode)
            mx.eval(result[0])
            return True
        except Exception:
            return False

    mlx_supports_mxfp4 = _mxfp_mode_supported("mxfp4")
    mlx_supports_mxfp8 = _mxfp_mode_supported("mxfp8")
    capabilities["mlx_supports_mxfp4"] = mlx_supports_mxfp4
    capabilities["mlx_supports_mxfp8"] = mlx_supports_mxfp8

    # MXFP is usable iff the MLX library supports the mode. On M1/M2 this runs as
    # software emulation (no hardware acceleration); on M3/M4 it is hardware
    # accelerated and `ocp_microscaling` remains True from chip detection.
    capabilities["supports_mxfp4"] = bool(mlx_supports_mxfp4)
    capabilities["supports_mxfp8"] = bool(mlx_supports_mxfp8)
    if not (mlx_supports_mxfp4 or mlx_supports_mxfp8):
        # The library cannot do MXFP at all -> no microscaling on any chip.
        capabilities["ocp_microscaling"] = False

    return capabilities


def analyze_model(
    model: nn.Module,
    calibration_data: Optional[list] = None,
    use_sensitivity: bool = True,
    use_case: Optional[Literal["inference", "training", "export"]] = None,
) -> dict:
    """
    Analyze model characteristics for quantization strategy selection.

    Args:
        model: MLX model to analyze
        calibration_data: Optional calibration data for sensitivity analysis
        use_sensitivity: Whether to compute layer sensitivities (requires calibration_data)
        use_case: Intended use case (inference, training, export) - affects format selection

    Returns:
        Dictionary with model analysis:
        - total_params: Total parameter count
        - quantizable_params: Parameters that can be quantized
        - model_size_mb: Current model size in MB
        - architecture_type: Detected architecture (transformer, cnn, hybrid)
        - has_embeddings: Whether model has embedding layers
        - has_attention: Whether model has attention layers
        - layer_count: Number of transformer/conv layers
        - sensitivities: Per-layer sensitivity scores (if calibration_data provided)
        - avg_sensitivity: Average sensitivity score
        - hardware: Hardware capabilities (OCP Microscaling, etc.)
        - use_case: Detected or specified use case
    """
    total_params = 0
    quantizable_params = 0
    total_bytes = 0
    has_embeddings = False
    has_attention = False
    layer_count = 0

    # Analyze modules
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            weight = module.weight
            param_count = weight.size
            total_params += param_count
            total_bytes += weight.nbytes

            # Count quantizable parameters (Linear and Embedding layers)
            if isinstance(module, (nn.Linear, nn.Embedding)):
                quantizable_params += param_count

            # Detect architecture patterns
            if isinstance(module, nn.Embedding):
                has_embeddings = True
            if "attn" in name.lower() or "attention" in name.lower():
                has_attention = True
            if "layer" in name.lower() or "block" in name.lower():
                if "." not in name[name.index("layer"):]:  # Count unique layers
                    layer_count += 1

    # Determine architecture type
    if has_attention:
        architecture_type = "transformer"
    elif has_embeddings and not has_attention:
        architecture_type = "embedding"
    else:
        architecture_type = "cnn_or_mlp"

    # Detect hardware capabilities
    hardware = detect_hardware_capabilities()

    # Auto-detect use case if not specified
    if use_case is None:
        # Default to inference for quantization
        use_case = "inference"

    result = {
        "total_params": total_params,
        "quantizable_params": quantizable_params,
        "model_size_mb": total_bytes / (1024**2),
        "architecture_type": architecture_type,
        "has_embeddings": has_embeddings,
        "has_attention": has_attention,
        "layer_count": layer_count,
        "quantizable_ratio": quantizable_params / total_params if total_params > 0 else 0,
        "hardware": hardware,
        "use_case": use_case,
    }

    # Compute sensitivity scores if calibration data available
    if use_sensitivity and calibration_data is not None:
        try:
            sensitivities = estimate_sensitivities(model, calibration_data)
            result["sensitivities"] = sensitivities
            # Compute average sensitivity using mx operations
            if sensitivities:
                sensitivity_values = mx.array(list(sensitivities.values()))
                result["avg_sensitivity"] = float(mx.mean(sensitivity_values))
            else:
                result["avg_sensitivity"] = 0.0
        except Exception:
            # If sensitivity estimation fails, just skip it
            result["sensitivities"] = {}
            result["avg_sensitivity"] = 0.0
    else:
        result["sensitivities"] = {}
        result["avg_sensitivity"] = 0.0

    return result


def select_strategy(
    model_info: dict,
    profile: Literal["aggressive", "balanced", "conservative"] = "balanced",
    target_memory_mb: Optional[float] = None,
    calibration_available: bool = False,
    use_case: Optional[Literal["inference", "training", "export"]] = None,
) -> dict:
    """
    Select optimal quantization strategy based on model analysis.

    Args:
        model_info: Model analysis from analyze_model()
        profile: Quality/size tradeoff profile
        target_memory_mb: Target memory budget (optional)
        calibration_available: Whether calibration data is available for GPTQ/AWQ
        use_case: Intended use case (overrides model_info use_case if provided)

    Returns:
        Dictionary with strategy selection:
        - method: Quantization method to use
        - bits: Bit width (for uniform quantization)
        - group_size: Group size
        - reason: Human-readable explanation
        - expected_size_mb: Expected model size after quantization
        - expected_quality: Expected quality level (high/medium/low)
    """
    total_params = model_info["total_params"]
    current_size_mb = model_info["model_size_mb"]
    architecture = model_info["architecture_type"]
    avg_sensitivity = model_info.get("avg_sensitivity", 0.0)
    has_sensitivity_data = bool(model_info.get("sensitivities"))
    hardware = model_info.get("hardware", {})
    use_case = use_case or model_info.get("use_case", "inference")

    # Hardware capabilities
    supports_ocp_microscaling = hardware.get("ocp_microscaling", False)

    # Calculate target compression ratio
    if target_memory_mb is not None:
        target_compression = current_size_mb / target_memory_mb
    else:
        # Default compression targets based on profile
        if profile == "aggressive":
            target_compression = 8.0  # 4-bit or mixed 3-6
        elif profile == "balanced":
            target_compression = 5.0  # 6-bit or mixed 4-6
        else:  # conservative
            target_compression = 2.0  # 8-bit

    # Model size categories
    is_tiny = total_params < 200_000_000  # <200M
    is_small = 200_000_000 <= total_params < 500_000_000  # 200-500M
    is_medium = 500_000_000 <= total_params < 1_000_000_000  # 500M-1B

    # Use sensitivity to decide between simple vs advanced quantization
    # High sensitivity (>0.5) means model is sensitive to quantization errors
    # and would benefit from GPTQ/AWQ or MXFP formats
    use_advanced_quant = calibration_available and (
        avg_sensitivity > 0.5 or has_sensitivity_data
    )

    # Strategy selection logic
    strategy = {}

    # ============================================================================
    # TRAINING/FINE-TUNING USE CASE
    # ============================================================================
    # For training, prefer floating point formats (FP8/MXFP8) for better gradients
    if use_case == "training":
        if supports_ocp_microscaling and profile != "aggressive":
            # MXFP8 is ideal for training - hardware accelerated, good precision
            strategy = {
                "method": "quantize_mxfp8",
                "reason": (
                    "Training use case detected with OCP Microscaling support. "
                    "MXFP8 provides hardware-accelerated 8-bit floating point "
                    "with excellent gradient precision for fine-tuning."
                ),
                "expected_size_mb": current_size_mb / 2.0,
                "expected_quality": "very_high",
            }
        else:
            # Fall back to BFloat16 for training without MXFP support
            strategy = {
                "method": "convert_to_bfloat16",
                "reason": (
                    "Training use case detected. BFloat16 provides better "
                    "gradient precision and training stability with minimal memory overhead."
                ),
                "expected_size_mb": current_size_mb,
                "expected_quality": "maximum",
            }
        return strategy

    # ============================================================================
    # EXPORT USE CASE (for cross-platform compatibility)
    # ============================================================================
    # For export, might want standard formats
    # (Future: could add GGML format selection here when implemented)

    # ============================================================================
    # INFERENCE USE CASE - Main quantization logic
    # ============================================================================

    # For very aggressive compression (>7x)
    if target_compression > 7.0:
        # Prefer MXFP4 if hardware supports it and model is sensitive
        if supports_ocp_microscaling and (avg_sensitivity > 0.5 or use_advanced_quant):
            strategy = {
                "method": "quantize_mxfp4",
                "reason": (
                    "Aggressive compression with OCP Microscaling support. "
                    "MXFP4 provides hardware-accelerated 4-bit floating point "
                    "with better quality than INT4 (~8x compression)."
                    + (f" Model sensitivity ({avg_sensitivity:.2f}) justifies FP over INT."
                       if avg_sensitivity > 0.5 else "")
                ),
                "expected_size_mb": current_size_mb / 7.8,
                "expected_quality": "high",
            }
        elif is_tiny and architecture == "transformer":
            # Tiny models can handle aggressive mixed quantization
            strategy = {
                "method": "mixed_3_6",
                "profile": "aggressive",
                "group_size": 64,
                "reason": (
                    f"Tiny model ({total_params/1e6:.0f}M params) with aggressive "
                    "compression target. Mixed 3-6 bit (avg ~3.5 BPW) provides "
                    "maximum compression while maintaining acceptable quality."
                ),
                "expected_size_mb": current_size_mb / 7.5,
                "expected_quality": "medium",
            }
        elif use_advanced_quant and calibration_available:
            # Use DWQ or GPTQ with 4-bit for best quality at high compression
            # DWQ is newer and often better than GPTQ for dynamic workloads
            if avg_sensitivity > 0.6:
                # Very sensitive model - use DWQ
                reason = (
                    f"Very high sensitivity ({avg_sensitivity:.2f}) with calibration data. "
                    "DWQ (Dynamic Weight Quantization) provides adaptive 4-bit "
                    "quantization for quality preservation at ~8x compression."
                )
                strategy = {
                    "method": "dwq",
                    "bits": 4,
                    "group_size": 64,
                    "reason": reason,
                    "expected_size_mb": current_size_mb / 7.8,
                    "expected_quality": "high",
                }
            else:
                # Moderately sensitive - use GPTQ
                reason = (
                    "High compression target with calibration data available. "
                    "GPTQ 4-bit provides Hessian-based quantization at ~8x compression."
                )
                if avg_sensitivity > 0.5:
                    reason += f" Model sensitivity ({avg_sensitivity:.2f}) suggests advanced quantization."
                strategy = {
                    "method": "gptq",
                    "bits": 4,
                    "group_size": 64,
                    "reason": reason,
                    "expected_size_mb": current_size_mb / 7.8,
                    "expected_quality": "high",
                }
        else:
            # Fall back to simple 4-bit
            strategy = {
                "method": "quantize_4bit",
                "bits": 4,
                "group_size": 64,
                "reason": (
                    "High compression target without calibration data. "
                    "Simple 4-bit integer quantization provides ~8x compression."
                ),
                "expected_size_mb": current_size_mb / 7.8,
                "expected_quality": "medium",
            }

    # For moderate compression (4-7x)
    elif 4.0 < target_compression <= 7.0:
        # Consider dynamic/sensitivity-based mixed precision if calibration available
        if calibration_available and has_sensitivity_data and avg_sensitivity > 0.4:
            # Use dynamic quantization for optimal per-layer bit allocation
            target_bpw = 32.0 / target_compression  # Convert compression to bits-per-weight
            strategy = {
                "method": "dynamic",
                "target_bpw": target_bpw,
                "reason": (
                    f"Calibration data with sensitivity analysis available. "
                    f"Dynamic quantization allocates bits per-layer based on "
                    f"sensitivity (target: {target_bpw:.1f} BPW, avg sensitivity: {avg_sensitivity:.2f})."
                ),
                "expected_size_mb": current_size_mb / target_compression,
                "expected_quality": "very_high",
            }
        elif architecture == "transformer" and (is_small or is_medium):
            # Use mixed 4-6 for better quality
            strategy = {
                "method": "mixed_3_6",
                "profile": "balanced",
                "group_size": 64,
                "reason": (
                    f"Model size ({total_params/1e6:.0f}M params) benefits from "
                    "mixed-precision. 4-6 bit strategy (avg ~4.5 BPW) balances "
                    "quality and compression."
                ),
                "expected_size_mb": current_size_mb / 5.3,
                "expected_quality": "high",
            }
        elif use_advanced_quant:
            # AWQ 4-bit for activation-aware quantization
            reason = (
                "Calibration data available. AWQ 4-bit provides activation-aware "
                "quantization for better quality at moderate compression."
            )
            if avg_sensitivity > 0.5:
                reason += f" Sensitivity analysis ({avg_sensitivity:.2f}) recommends AWQ."

            strategy = {
                "method": "awq",
                "bits": 4,
                "group_size": 64,
                "reason": reason,
                "expected_size_mb": current_size_mb / 7.8,
                "expected_quality": "high",
            }
        else:
            # 6-bit uniform quantization (or FP4 if available and sensitive)
            if supports_ocp_microscaling and avg_sensitivity > 0.5:
                # Use MXFP4 for better quality
                strategy = {
                    "method": "quantize_mxfp4",
                    "reason": (
                        "Moderate compression with OCP Microscaling support. "
                        f"Model sensitivity ({avg_sensitivity:.2f}) suggests MXFP4 "
                        "for better quality than INT6 at similar compression."
                    ),
                    "expected_size_mb": current_size_mb / 7.8,
                    "expected_quality": "very_high",
                }
            else:
                # Standard 6-bit integer
                strategy = {
                    "method": "quantize_6bit",
                    "bits": 6,
                    "group_size": 64,
                    "reason": (
                        "Moderate compression target. 6-bit integer quantization provides "
                        "~5.3x compression with better quality than 4-bit."
                    ),
                    "expected_size_mb": current_size_mb / 5.3,
                    "expected_quality": "high",
                }

    # For conservative compression (2-4x)
    elif 2.0 <= target_compression <= 4.0:
        # Consider MXFP8 for high quality with hardware acceleration
        if supports_ocp_microscaling:
            strategy = {
                "method": "quantize_mxfp8",
                "reason": (
                    "Conservative compression with OCP Microscaling support. "
                    "MXFP8 provides hardware-accelerated 8-bit floating point "
                    "with excellent quality (~2x compression)."
                ),
                "expected_size_mb": current_size_mb / 2.0,
                "expected_quality": "very_high",
            }
        elif is_medium and use_advanced_quant:
            # Use AWQ or GPTQ with 6-bit for larger models
            reason = (
                f"Large model ({total_params/1e6:.0f}M params) with conservative "
                "compression. GPTQ 6-bit provides excellent quality preservation."
            )
            if avg_sensitivity > 0.5:
                reason += f" High sensitivity ({avg_sensitivity:.2f}) justifies GPTQ."

            strategy = {
                "method": "gptq",
                "bits": 6,
                "group_size": 64,
                "reason": reason,
                "expected_size_mb": current_size_mb / 5.3,
                "expected_quality": "very_high",
            }
        else:
            # 8-bit quantization for maximum quality
            strategy = {
                "method": "quantize_8bit",
                "bits": 8,
                "group_size": 64,
                "reason": (
                    "Conservative compression target. 8-bit integer quantization provides "
                    "~2x compression with minimal quality loss."
                ),
                "expected_size_mb": current_size_mb / 2.0,
                "expected_quality": "very_high",
            }

    # For minimal compression (<2x) or fine-tuning scenarios
    else:
        strategy = {
            "method": "convert_to_bfloat16",
            "reason": (
                "Minimal compression or fine-tuning scenario. BFloat16 provides "
                "better training stability with minimal memory reduction."
            ),
            "expected_size_mb": current_size_mb,  # No compression, just format change
            "expected_quality": "maximum",
        }

    return strategy


def autoquant(
    model: nn.Module,
    profile: Literal["aggressive", "balanced", "conservative"] = "balanced",
    target_memory_mb: Optional[float] = None,
    calibration_data: Optional[list] = None,
    use_case: Optional[Literal["inference", "training", "export"]] = None,
    apply: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Automatically select and apply optimal quantization strategy.

    This is the main entry point for automatic quantization. It analyzes the model,
    selects the best quantization strategy, and optionally applies it.

    Args:
        model: MLX model to quantize
        profile: Quality/size tradeoff profile:
            - "aggressive": Maximum compression (avg ~8x, lower quality)
            - "balanced": Balanced compression (avg ~5x, good quality)
            - "conservative": Minimal compression (avg ~2x, high quality)
        target_memory_mb: Target memory budget in MB (optional, overrides profile)
        calibration_data: Calibration data for GPTQ/AWQ/DWQ/Dynamic (optional)
        use_case: Intended use case:
            - "inference": Optimize for inference (default)
            - "training": Optimize for training/fine-tuning (prefers FP formats)
            - "export": Optimize for export/compatibility
        apply: Whether to apply quantization (default: True)
        verbose: Print analysis and recommendations (default: True)

    Returns:
        Dictionary with quantization results:
        - strategy: Selected quantization method
        - model_info: Model analysis results
        - reason: Explanation for strategy selection
        - applied: Whether quantization was applied
        - before_size_mb: Model size before quantization
        - after_size_mb: Model size after quantization (if applied)

    Example:
        ```python
        from smlx.models.SmolLM2_135M import load
        from smlx.quant import autoquant

        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

        # Automatic quantization
        result = autoquant(model, profile="balanced")
        print(f"Reduced size from {result['before_size_mb']:.1f} MB "
              f"to {result['after_size_mb']:.1f} MB")

        # With memory target
        result = autoquant(model, target_memory_mb=50)

        # For training/fine-tuning
        result = autoquant(model, use_case="training")

        # Just analyze without applying
        result = autoquant(model, apply=False)
        print(f"Recommended: {result['strategy']}")
        ```
    """
    # Step 1: Analyze model
    if verbose:
        print("[*] Analyzing model...")

    model_info = analyze_model(model, calibration_data=calibration_data, use_case=use_case)

    if verbose:
        print(f"  Total parameters: {model_info['total_params']/1e6:.1f}M")
        print(f"  Quantizable parameters: {model_info['quantizable_params']/1e6:.1f}M "
              f"({model_info['quantizable_ratio']*100:.1f}%)")
        print(f"  Current size: {model_info['model_size_mb']:.1f} MB")
        print(f"  Architecture: {model_info['architecture_type']}")
        if model_info.get('sensitivities'):
            print(f"  Average sensitivity: {model_info['avg_sensitivity']:.3f}")

    # Step 2: Select strategy
    if verbose:
        print(f"\n[*] Selecting quantization strategy (profile: {profile})...")
        if use_case:
            print(f"  Use case: {use_case}")
        if model_info.get("hardware", {}).get("ocp_microscaling"):
            print("  Hardware: OCP Microscaling supported")

    calibration_available = calibration_data is not None
    strategy = select_strategy(
        model_info,
        profile=profile,
        target_memory_mb=target_memory_mb,
        calibration_available=calibration_available,
        use_case=use_case
    )

    if verbose:
        print(f"  Selected method: {strategy['method']}")
        if 'bits' in strategy:
            print(f"  Bit width: {strategy['bits']}-bit")
        if 'group_size' in strategy:
            print(f"  Group size: {strategy['group_size']}")
        print(f"  Expected size: {strategy['expected_size_mb']:.1f} MB "
              f"({model_info['model_size_mb']/strategy['expected_size_mb']:.1f}x reduction)")
        print(f"  Expected quality: {strategy['expected_quality']}")
        print(f"\n  Reason: {strategy['reason']}")

    # Step 3: Apply quantization
    result = {
        "strategy": strategy,
        "model_info": model_info,
        "reason": strategy["reason"],
        "applied": False,
        "before_size_mb": model_info["model_size_mb"],
    }

    if apply:
        if verbose:
            print(f"\n[*] Applying {strategy['method']}...")

        try:
            # Import and apply the selected method
            method = strategy["method"]

            # ========================================================================
            # Integer Quantization (4/6/8-bit)
            # ========================================================================
            if method == "quantize_4bit":
                from . import quantize_4bit
                quantize_4bit(model, group_size=strategy.get("group_size", 64))

            elif method == "quantize_6bit":
                from . import quantize_6bit
                quantize_6bit(model, group_size=strategy.get("group_size", 64))

            elif method == "quantize_8bit":
                from . import quantize_8bit
                quantize_8bit(model, group_size=strategy.get("group_size", 64))

            # ========================================================================
            # Floating Point Quantization (FP4/FP8, MXFP4/MXFP8)
            # ========================================================================
            elif method == "quantize_fp4":
                from .fp4 import quantize_model_fp4
                quantize_model_fp4(model)

            elif method == "quantize_fp8":
                # DEPRECATED: FP8 is simulated (stored as float16). Use MXFP8 instead.
                import warnings
                warnings.warn(
                    "quantize_fp8 is deprecated (simulated only). "
                    "Automatically using quantize_mxfp8 for true 8-bit storage. "
                    "Update your code to use 'quantize_mxfp8' method instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                from .mxfp8 import quantize_model_mxfp8
                quantize_model_mxfp8(model)

            elif method == "quantize_mxfp4":
                from .mxfp4 import quantize_model_mxfp4
                quantize_model_mxfp4(model)

            elif method == "quantize_mxfp8":
                from .mxfp8 import quantize_model_mxfp8
                quantize_model_mxfp8(model)

            elif method == "convert_to_bfloat16":
                from .bf16 import convert_to_bfloat16
                convert_to_bfloat16(model)

            # ========================================================================
            # Mixed-Precision Quantization
            # ========================================================================
            elif method == "mixed_3_6":
                from .mixed_3_6 import quantize_3_6_mixed
                quantize_3_6_mixed(
                    model,
                    strategy=strategy.get("profile", "balanced"),
                    verbose=False
                )

            elif method == "mixed_bit_custom":
                from .mixed_bit import apply_mixed_bit_quantization, create_balanced_strategy
                target_bpw = strategy.get("target_bpw", 4.5)
                strat = create_balanced_strategy(target_bpw=target_bpw)
                apply_mixed_bit_quantization(model, strat)

            # ========================================================================
            # Advanced Quantization Methods (GPTQ, AWQ, DWQ)
            # ========================================================================
            elif method == "gptq":
                from .gptq import gptq_quantize
                if calibration_data is None:
                    if verbose:
                        print("  Warning: GPTQ requires calibration data. "
                              "Falling back to simple quantization.")
                    from . import quantize_4bit
                    quantize_4bit(model, group_size=strategy.get("group_size", 64))
                else:
                    gptq_quantize(
                        model,
                        calibration_data,
                        bits=strategy.get("bits", 4),
                        group_size=strategy.get("group_size", 64)
                    )

            elif method == "awq":
                from .awq import awq_quantize
                if calibration_data is None:
                    if verbose:
                        print("  Warning: AWQ requires calibration data. "
                              "Falling back to simple quantization.")
                    from . import quantize_4bit
                    quantize_4bit(model, group_size=strategy.get("group_size", 64))
                else:
                    awq_quantize(
                        model,
                        calibration_data,
                        bits=strategy.get("bits", 4),
                        group_size=strategy.get("group_size", 64)
                    )

            elif method == "dwq":
                from .dwq import dwq_quantize
                if calibration_data is None:
                    if verbose:
                        print("  Warning: DWQ requires calibration data. "
                              "Falling back to simple quantization.")
                    from . import quantize_4bit
                    quantize_4bit(model, group_size=strategy.get("group_size", 64))
                else:
                    dwq_quantize(
                        model,
                        calibration_data,
                        bits=strategy.get("bits", 4),
                        group_size=strategy.get("group_size", 64)
                    )

            # ========================================================================
            # Dynamic/Sensitivity-Based Quantization
            # ========================================================================
            elif method == "dynamic":
                from .dynamic_quant import dynamic_quantize
                if calibration_data is None:
                    if verbose:
                        print("  Warning: Dynamic quantization requires calibration data. "
                              "Falling back to mixed-precision.")
                    from .mixed_3_6 import quantize_3_6_mixed
                    quantize_3_6_mixed(model, strategy="balanced", verbose=False)
                else:
                    target_bpw = strategy.get("target_bpw", 4.5)
                    dynamic_quantize(
                        model,
                        calibration_data,
                        target_bpw=target_bpw
                    )

            else:
                raise ValueError(f"Unknown quantization method: {method}")

            # Measure final size
            final_size = 0
            for _, module in model.named_modules():
                if hasattr(module, "weight"):
                    final_size += module.weight.nbytes
                    if hasattr(module, "scales") and module.scales is not None:
                        final_size += module.scales.nbytes
                    if hasattr(module, "biases") and module.biases is not None:
                        final_size += module.biases.nbytes

            result["after_size_mb"] = final_size / (1024**2)
            result["applied"] = True
            result["actual_reduction"] = model_info["model_size_mb"] / result["after_size_mb"]

            if verbose:
                print(f"\n[✓] Quantization complete!")
                print(f"  Final size: {result['after_size_mb']:.1f} MB")
                print(f"  Actual reduction: {result['actual_reduction']:.1f}x")

        except Exception as e:
            if verbose:
                print(f"\n[ERROR] Error applying quantization: {e}")
            result["error"] = str(e)

    return result


def recommend_strategy(
    model: nn.Module,
    verbose: bool = True,
) -> dict:
    """
    Analyze model and recommend quantization strategies without applying.

    Provides detailed recommendations for different quality/size tradeoffs.

    Args:
        model: MLX model to analyze
        verbose: Print recommendations (default: True)

    Returns:
        Dictionary with recommendations for each profile:
        - aggressive: Maximum compression strategy
        - balanced: Balanced quality/size strategy
        - conservative: Maximum quality strategy
        - model_info: Model analysis

    Example:
        ```python
        from smlx.quant import recommend_strategy

        recommendations = recommend_strategy(model)
        print("Aggressive:", recommendations['aggressive']['method'])
        print("Balanced:", recommendations['balanced']['method'])
        print("Conservative:", recommendations['conservative']['method'])
        ```
    """
    model_info = analyze_model(model)

    recommendations = {
        "model_info": model_info,
        "aggressive": select_strategy(model_info, profile="aggressive"),
        "balanced": select_strategy(model_info, profile="balanced"),
        "conservative": select_strategy(model_info, profile="conservative"),
    }

    if verbose:
        print("[*] Quantization Strategy Recommendations")
        print("=" * 60)
        print(f"\nModel: {model_info['total_params']/1e6:.1f}M parameters, "
              f"{model_info['model_size_mb']:.1f} MB")
        print(f"Architecture: {model_info['architecture_type']}")

        for profile in ["aggressive", "balanced", "conservative"]:
            strat = recommendations[profile]
            print(f"\n{profile.upper()}:")
            print(f"  Method: {strat['method']}")
            if 'bits' in strat:
                print(f"  Bits: {strat.get('bits')}")
            print(f"  Size: {strat['expected_size_mb']:.1f} MB "
                  f"({model_info['model_size_mb']/strat['expected_size_mb']:.1f}x)")
            print(f"  Quality: {strat['expected_quality']}")
            print(f"  Reason: {strat['reason']}")

    return recommendations


__all__ = [
    "autoquant",
    "analyze_model",
    "select_strategy",
    "recommend_strategy",
    "detect_hardware_capabilities",
]
