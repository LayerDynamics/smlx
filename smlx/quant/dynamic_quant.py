"""
Dynamic Quantization with Mixed-Precision for SMLX.

Dynamic quantization uses sensitivity analysis to determine which layers are most
important for model accuracy. Sensitive layers receive higher bit widths while less
sensitive layers can use aggressive quantization, achieving better compression with
minimal accuracy loss.

Algorithm:
1. Quantize model to low bits and compute KL divergence with original model
2. Accumulate gradients of KL loss to estimate layer sensitivities
3. Use binary search to find sensitivity threshold for target bits-per-weight
4. Apply mixed-precision: high bits for sensitive layers, low bits for others

Optimized for "smol" models (<10B parameters) on Apple M4 chipsets.

Reference:
    Mixed-Precision Quantization
    https://arxiv.org/abs/2106.08295

Example:
    ```python
    import mlx.core as mx
    from smlx.quant import dynamic_quantize, load_calibration_data

    # Load model and calibration data
    model = load_your_model()
    calibration_data = load_calibration_data(tokenizer, num_samples=128)

    # Quantize with dynamic mixed-precision (target 4.5 bits per weight)
    quantized_model, sensitivities = dynamic_quantize(
        model,
        calibration_data,
        target_bpw=4.5,
        low_bits=4,
        high_bits=6,
        group_size=64
    )
    ```
"""

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map


def _quantize_dequantize(w: mx.array, bits: int, group_size: int) -> mx.array:
    """
    Quantize and dequantize weight to simulate quantization error.

    Args:
        w: Weight array
        bits: Bits per weight
        group_size: Group size for quantization

    Returns:
        Dequantized weight (with quantization error)
    """
    w_q, scales, biases = mx.quantize(w, bits=bits, group_size=group_size)
    return mx.dequantize(
        w_q, scales=scales, biases=biases, bits=bits, group_size=group_size, dtype=w.dtype
    )


def _kl_divergence_loss(logits_q: mx.array, logits_orig: mx.array) -> mx.array:
    """
    Compute KL divergence loss between quantized and original model outputs.

    Args:
        logits_q: Logits from quantized model
        logits_orig: Logits from original model

    Returns:
        KL divergence loss
    """
    # Convert to float32 for numerical stability
    logits_q = logits_q.astype(mx.float32)
    logits_orig = logits_orig.astype(mx.float32)

    # Compute log probabilities
    log_probs_q = logits_q - mx.logsumexp(logits_q, axis=-1, keepdims=True)
    log_probs_orig = logits_orig - mx.logsumexp(logits_orig, axis=-1, keepdims=True)

    # KL(orig || q) = sum(p_orig * (log p_orig - log p_q))
    probs_orig = mx.exp(log_probs_orig)
    kl = probs_orig * (log_probs_orig - log_probs_q)
    return kl.sum(axis=-1)


def estimate_sensitivities(
    model: nn.Module,
    calibration_data: mx.array,
    low_bits: int = 4,
    low_group_size: int = 64,
    high_bits: int = 6,
    high_group_size: int = 64,
    batch_size: int = 4,
) -> dict[str, float]:
    """
    Estimate layer sensitivities to quantization.

    Sensitivity is computed as the gradient alignment of KL divergence loss
    with respect to the difference between low-bit and high-bit quantization.
    Higher sensitivity indicates layers that benefit more from higher bit widths.

    Args:
        model: MLX model to analyze
        calibration_data: Calibration data (token indices or inputs)
        low_bits: Low bit width for aggressive quantization (default: 4)
        low_group_size: Group size for low bits (default: 64)
        high_bits: High bit width for sensitive layers (default: 6)
        high_group_size: Group size for high bits (default: 64)
        batch_size: Batch size for sensitivity estimation (default: 4)

    Returns:
        Dictionary mapping layer paths to sensitivity scores

    Note:
        - Higher sensitivity = layer benefits more from high bits
        - Only layers with `to_quantized()` method are analyzed
        - Sensitivities can be saved to JSON for reuse
    """
    import copy

    # Find quantizable layers
    all_layers = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    quantizable_layers = [(path, mod) for path, mod in all_layers if hasattr(mod, "to_quantized")]

    if len(quantizable_layers) == 0:
        print("Warning: No quantizable layers found in model")
        return {}

    layers = dict(quantizable_layers)

    # Create quantized model copy with low-bit weights
    q_model = copy.deepcopy(model)

    # Modify the quantizable layers in the copied model
    q_all_layers = tree_flatten(q_model.leaf_modules(), is_leaf=nn.Module.is_module)
    for _, module in q_all_layers:
        # Type guard: ensure module is nn.Module
        if not isinstance(module, nn.Module):
            continue
        if hasattr(module, "to_quantized") and hasattr(module, "weight"):
            weight = getattr(module, "weight", None)
            if isinstance(weight, mx.array):
                module.weight = _quantize_dequantize(weight, low_bits, low_group_size)

    # Freeze all parameters except the quantizable weights we want to track
    q_model.freeze()
    for _, module in tree_flatten(q_model.leaf_modules(), is_leaf=nn.Module.is_module):
        # Type guard: ensure module is nn.Module
        if not isinstance(module, nn.Module):
            continue
        if hasattr(module, "to_quantized"):
            module.unfreeze(keys=["weight"])

    # Define loss function
    def loss_fn(batch, targets):
        return _kl_divergence_loss(q_model(batch), targets).mean()

    # Accumulate gradients across calibration data
    print(f"Estimating sensitivities for {len(layers)} layers...")
    grad_accum = tree_map(
        lambda x: mx.zeros(x.shape, dtype=mx.float32),
        q_model.trainable_parameters(),
    )

    for start_idx in range(0, len(calibration_data), batch_size):
        batch = calibration_data[start_idx : start_idx + batch_size]

        # Get targets from original model
        targets = model(batch)
        mx.eval(targets)

        # Compute gradients
        _, grads = nn.value_and_grad(q_model, loss_fn)(batch, targets)
        grad_accum = tree_map(lambda x, y: x + y, grad_accum, grads)
        del grads
        mx.eval(grad_accum)

    # Compute sensitivity for each layer
    def compute_sensitivity(gradient, low_q_weight, original_weight):
        n_batches = (len(calibration_data) + batch_size - 1) // batch_size
        gradient = gradient / n_batches

        # Simulate high-bit quantization
        high_q_weight = _quantize_dequantize(original_weight, high_bits, high_group_size)

        # Sensitivity = gradient alignment with (low_q - high_q) / param_size
        param_size_mb = original_weight.size / 1e6
        alignment = (gradient * (low_q_weight - high_q_weight)).sum()
        return alignment / param_size_mb

    sensitivities = tree_map(
        compute_sensitivity,
        grad_accum,
        q_model.parameters(),
        model.parameters(),
    )
    mx.eval(sensitivities)

    # Convert to dictionary (remove ".weight" suffix)
    sensitivities_flat = tree_flatten(sensitivities)
    sensitivities_dict: dict[str, float] = {}
    for k, s in sensitivities_flat:
        # Type guard: k is path string, s is sensitivity value (mx.array or float)
        if not isinstance(k, str):
            continue
        # Remove ".weight" suffix from parameter names
        key = k[:-7] if k.endswith(".weight") else k
        # Convert mx.array to Python scalar
        if isinstance(s, mx.array):
            value = float(s.item())
        else:
            value = float(s)
        sensitivities_dict[key] = value

    print(f" Sensitivity estimation complete: {len(sensitivities_dict)} layers")
    return sensitivities_dict


def estimate_threshold(
    model: nn.Module,
    sensitivities: dict[str, float],
    target_bpw: float,
    low_bits: int = 4,
    low_group_size: int = 64,
    high_bits: int = 6,
    high_group_size: int = 64,
) -> float:
    """
    Estimate sensitivity threshold for target bits-per-weight.

    Uses binary search to find the threshold such that layers with sensitivity
    above the threshold get high bits, achieving the target average bits per weight.

    Args:
        model: MLX model
        sensitivities: Dictionary of layer sensitivities
        target_bpw: Target bits per weight (e.g., 4.5)
        low_bits: Low bit width (default: 4)
        low_group_size: Group size for low bits (default: 64)
        high_bits: High bit width (default: 6)
        high_group_size: Group size for high bits (default: 64)

    Returns:
        Sensitivity threshold

    Note:
        - Layers with sensitivity > threshold receive high_bits
        - Layers with sensitivity d threshold receive low_bits
        - Binary search ensures target_bpw is met
    """
    import copy

    def predicate(path, module, threshold):
        """Predicate for nn.quantize() to decide layer quantization."""
        if not hasattr(module, "to_quantized"):
            return False
        if path in sensitivities and sensitivities[path] > threshold:
            return {"bits": high_bits, "group_size": high_group_size}
        return True

    # Binary search for threshold
    sens_vals = list(sensitivities.values())
    min_threshold = min(sens_vals)
    max_threshold = max(sens_vals)
    tolerance = 1e-3 * (max_threshold - min_threshold)

    print(f"Finding threshold for target {target_bpw:.2f} bits per weight...")

    while (max_threshold - min_threshold) > tolerance:
        mid = (max_threshold + min_threshold) / 2

        # Test this threshold
        test_model = copy.deepcopy(model)
        nn.quantize(
            test_model,
            group_size=low_group_size,
            bits=low_bits,
            class_predicate=lambda p, m, threshold=mid: predicate(p, m, threshold),
        )

        # Compute bits per weight
        bpw = _compute_bits_per_weight(test_model)

        if bpw > target_bpw:
            # Too many high bits, increase threshold (fewer high-bit layers)
            min_threshold = mid
        else:
            # Too many low bits, decrease threshold (more high-bit layers)
            max_threshold = mid

    threshold = (max_threshold + min_threshold) / 2
    print(f" Threshold found: {threshold:.6f}")
    return threshold


def _compute_bits_per_weight(model: nn.Module) -> float:
    """
    Compute average bits per weight for quantized model.

    Args:
        model: Quantized model

    Returns:
        Average bits per weight
    """
    total_bits = 0
    total_weights = 0

    for _, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if isinstance(module, nn.QuantizedLinear):
            # QuantizedLinear stores packed weights
            bits = module.bits
            # Packed weight has shape (output_dims, input_dims // (32 // bits))
            # Actual weight count is output_dims * input_dims
            num_weights = module.weight.shape[0] * module.weight.shape[1] * (32 // bits)
            total_bits += num_weights * bits
            total_weights += num_weights
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            # Unquantized layers (assume 16-bit float16)
            num_weights = module.weight.size
            total_bits += num_weights * 16
            total_weights += num_weights

    return total_bits / total_weights if total_weights > 0 else 0.0


def dynamic_quantize(
    model: nn.Module,
    calibration_data: mx.array,
    target_bpw: float = 4.5,
    low_bits: int = 4,
    high_bits: int = 6,
    group_size: int = 64,
    batch_size: int = 4,
    sensitivities_path: Optional[Path] = None,
) -> tuple[nn.Module, dict[str, float]]:
    """
    Quantize model with dynamic mixed-precision based on layer sensitivities.

    Analyzes layer sensitivities to quantization error and applies mixed-precision:
    sensitive layers get higher bits while less sensitive layers use lower bits,
    achieving better compression with minimal accuracy loss.

    Args:
        model: MLX model to quantize
        calibration_data: Calibration data tokens of shape (num_samples, seq_length)
        target_bpw: Target bits per weight (default: 4.5 for M4)
        low_bits: Low bit width for insensitive layers (default: 4)
        high_bits: High bit width for sensitive layers (default: 6)
        group_size: Group size for quantization (default: 64 for M4)
        batch_size: Batch size for sensitivity estimation (default: 4)
        sensitivities_path: Optional path to save/load sensitivity JSON

    Returns:
        Tuple of (quantized_model, sensitivities_dict)

    Note:
        - Optimized for M4 with default 4.5 target BPW (mix of 4-bit and 6-bit)
        - Sensitivities can be saved/loaded to avoid recomputation
        - Only layers with `to_quantized()` are quantized
        - Model is modified in-place

    Example:
        ```python
        import mlx.nn as nn
        from smlx.quant import dynamic_quantize, load_calibration_data

        # Load calibration data
        calibration_data = load_calibration_data(tokenizer, num_samples=128)

        # Quantize with dynamic mixed-precision (4.5 bits average)
        model, sensitivities = dynamic_quantize(
            model,
            calibration_data,
            target_bpw=4.5,  # M4-optimized
            low_bits=4,
            high_bits=6,
            group_size=64
        )

        # Save sensitivities for later reuse
        import json
        with open("sensitivities.json", "w") as f:
            json.dump(sensitivities, f)
        ```
    """
    # Load or compute sensitivities
    if sensitivities_path and Path(sensitivities_path).exists():
        print(f"Loading sensitivities from {sensitivities_path}")
        with open(sensitivities_path) as f:
            sensitivities = json.load(f)
    else:
        sensitivities = estimate_sensitivities(
            model,
            calibration_data,
            low_bits=low_bits,
            low_group_size=group_size,
            high_bits=high_bits,
            high_group_size=group_size,
            batch_size=batch_size,
        )

        # Save sensitivities if path provided
        if sensitivities_path:
            print(f"Saving sensitivities to {sensitivities_path}")
            with open(sensitivities_path, "w") as f:
                json.dump(sensitivities, f)

    # Find threshold for target BPW
    threshold = estimate_threshold(
        model,
        sensitivities,
        target_bpw=target_bpw,
        low_bits=low_bits,
        low_group_size=group_size,
        high_bits=high_bits,
        high_group_size=group_size,
    )

    # Define quantization predicate
    def quant_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if path in sensitivities and sensitivities[path] > threshold:
            return {"bits": high_bits, "group_size": group_size}
        return True

    # Apply mixed-precision quantization
    print("Applying mixed-precision quantization...")
    nn.quantize(
        model,
        group_size=group_size,
        bits=low_bits,
        class_predicate=quant_predicate,
    )

    # Report statistics
    final_bpw = _compute_bits_per_weight(model)
    num_high = sum(1 for s in sensitivities.values() if s > threshold)
    num_low = len(sensitivities) - num_high

    print(" Dynamic quantization complete:")
    print(f"  Target BPW: {target_bpw:.2f}")
    print(f"  Actual BPW: {final_bpw:.2f}")
    print(f"  High-bit ({high_bits}-bit) layers: {num_high}")
    print(f"  Low-bit ({low_bits}-bit) layers: {num_low}")

    return model, sensitivities
