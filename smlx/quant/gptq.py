"""
GPTQ (GPT Quantization) for post-training quantization.

GPTQ uses Hessian-based optimization to minimize quantization error by
iteratively quantizing weights while compensating for errors using the
inverse Hessian matrix. This results in better accuracy than naive quantization
at the same bit width.

Algorithm:
1. Capture input features and compute Hessian matrix H = X^T @ X
2. Compute inverse Hessian (Cholesky decomposition)
3. Iteratively quantize each weight column, compensating error using Hinv
4. Apply quantization with learned scales/biases

Optimized for "smol" models (<10B parameters) on Apple M4 chipsets.

Reference:
    GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
    https://arxiv.org/abs/2210.17323
    https://github.com/AutoGPTQ

Example:
    ```python
    import mlx.core as mx
    from smlx.quant import gptq_quantize, load_calibration_data
    from transformers import AutoTokenizer

    # Load model and calibration data
    model = load_your_model()
    tokenizer = AutoTokenizer.from_pretrained("model_name")
    calibration_data = load_calibration_data(tokenizer, num_samples=128)

    # Quantize with GPTQ
    quantized_model = gptq_quantize(
        model,
        calibration_data,
        bits=4,
        group_size=64,
        batch_size=8
    )
    ```
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Number of GPTQ columns to process between lazy-graph materializations inside a
# group. The inner column loop is a strict read-after-write chain on W, so the
# eval cadence is numerically irrelevant (output is bit-identical for any value)
# and only trades host-sync frequency against peak memory. Empirically (random
# 4-bit quant of representative 135M/360M projection and tied-lm_head shapes on
# the 36GB M4 target), a stride of 2 halves the per-column sync count for a
# 15-37% speedup with ZERO peak-memory regression vs. per-column eval, whereas
# stride >= 4 starts to push the deferred slice-update set past the working-set
# ceiling and regresses peak memory. See tests/quant/test_gptq.py.
_GPTQ_EVAL_COLUMN_STRIDE = 2


def _dead_feature_mask(diag_vals: mx.array) -> mx.array:
    """Boolean mask of dead input-feature channels in a Hessian diagonal.

    A channel is *dead* when its accumulated curvature — the ``H = XᵀX``
    diagonal, a sum of squared activations — is zero or negligibly small
    relative to the strongest channel. Such channels make ``H`` singular, so
    the Cholesky inverse is undefined; the GPTQ guard resets them to a clean
    diagonal so the factorization stays well-defined.

    An exact ``diag == 0`` test only catches channels whose activations are
    *exactly* zero. A channel that is effectively dead — activations of, say,
    1e-30 magnitude — leaves a tiny positive residue on the diagonal that still
    renders ``H`` singular yet slips past equality. Comparing against
    ``max(diag) * tol`` scales the cutoff to the Hessian's own magnitude and,
    unlike a mean-based cutoff, is *not* dragged toward zero by the dead
    channels it is meant to detect (the max is set by the strongest live
    channel). The 1e-8 relative floor sits far below any real channel — live
    channels carry curvature within a few orders of magnitude of the max, while
    genuinely-dead channels sit 15+ orders below — so real but weakly-activated
    channels are never misflagged. When every channel is dead (max == 0) the
    cutoff is 0 and ``<=`` flags them all, matching the all-zero degenerate
    case.
    """
    ref = mx.max(diag_vals)
    tol = ref * 1e-8
    return diag_vals <= tol


def _quantize_weights(w: mx.array, bits: int, scales: mx.array, biases: mx.array) -> mx.array:
    """
    Pack quantized weights into uint32 format.

    Args:
        w: Weight array to quantize
        bits: Bits per weight (2, 4, or 8)
        scales: Per-group scaling factors
        biases: Per-group bias offsets

    Returns:
        Packed uint32 array

    Note:
        For 4-bit: 8 values packed per uint32
        For 2-bit: 16 values packed per uint32
        For 8-bit: 4 values packed per uint32
    """
    assert bits in {2, 4, 8}, f"Unsupported bits {bits}"

    el_per_int = 32 // bits
    n_bins = 2**bits - 1

    # Reshape to group dimensions
    w = mx.unflatten(w, -1, (scales.shape[-1], -1))

    # Quantize: clip to [0, n_bins]
    w = mx.clip(mx.round((w - biases[..., None]) / scales[..., None]), 0.0, n_bins).astype(
        mx.uint32
    )

    # Pack multiple values into each uint32
    shifts = mx.power(2, mx.arange(0, 32, bits, mx.uint32))
    w = mx.unflatten(w, -1, (-1, el_per_int))
    w = mx.sum(w * shifts, axis=-1)

    return w.flatten(-2, -1)


class Catcher(nn.Module):
    """
    Wrapper module that captures input features and computes Hessian matrix.

    The Hessian is computed as H = X^T @ X where X is the flattened input features.
    This is used by GPTQ to determine how to compensate for quantization errors.

    Args:
        module: The module to wrap (typically nn.Linear)

    Attributes:
        H: Accumulated Hessian matrix (X^T @ X)
        module: The wrapped module
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.H = mx.array(0.0)

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        """
        Forward pass that accumulates Hessian.

        Args:
            x: Input tensor of shape (..., input_dims)
            *args, **kwargs: Additional arguments passed to wrapped module

        Returns:
            Output from wrapped module
        """
        # Flatten batch dimensions
        xf = x.flatten(0, -2)

        # Accumulate Hessian: H += X^T @ X
        self.H = self.H + xf.T @ xf

        return self.module(x, *args, **kwargs)


def gptq_quantize(
    model: nn.Module,
    calibration_data: mx.array,
    bits: int = 4,
    group_size: int = 64,
    batch_size: int = 8,
) -> nn.Module:
    """
    Quantize Linear layers using GPTQ algorithm.

    GPTQ applies Hessian-based quantization to nn.Linear layers for optimal accuracy
    while minimizing quantization error. Other layer types (Embedding, etc.) are left
    unquantized and can be quantized separately using mlx.nn.quantize() if needed.

    Args:
        model: MLX model to quantize
        calibration_data: Calibration data tokens of shape (num_samples, seq_length)
        bits: Bits per weight (default: 4 for M4, supports 2/4/8)
        group_size: Group size for quantization (default: 64 for M4, supports 32/64/128)
        batch_size: Batch size for Hessian computation (default: 8)

    Returns:
        Quantized model with Linear layers converted to QuantizedLinear

    Note:
        - Only nn.Linear layers are quantized with GPTQ
        - Other layers (Embedding, LayerNorm, etc.) remain unquantized
        - Model is modified in-place
        - Optimized for M4 with 4-bit, group_size=64 defaults
        - For embedding quantization, use mlx.nn.quantize() separately

    Example:
        ```python
        import mlx.nn as nn
        from smlx.quant import gptq_quantize, load_calibration_data

        # Load calibration data
        calibration_data = load_calibration_data(tokenizer, num_samples=128)

        # Quantize Linear layers with GPTQ (4-bit, optimized for M4)
        model = gptq_quantize(model, calibration_data)

        # Optionally quantize embeddings separately (if needed)
        model = nn.quantize(model, bits=6, group_size=64)

        # Or customize GPTQ settings
        model = gptq_quantize(
            model,
            calibration_data,
            bits=3,  # More aggressive
            group_size=128  # Larger groups
        )
        ```
    """
    # Wrap all Linear layers with Catcher to compute Hessians
    layers = []
    gptq_types = {nn.Linear}  # Only Linear layers get GPTQ treatment

    for key, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if isinstance(module, nn.Module) and type(module) in gptq_types:
            layers.append((key, Catcher(module)))

    model.update_modules(tree_unflatten(layers))

    # Pass calibration data through model to compute Hessians
    print(f"Computing Hessians for {len(layers)} layers...")
    for start_idx in range(0, len(calibration_data), batch_size):
        batch = calibration_data[start_idx : start_idx + batch_size]
        model(batch)
        mx.eval([catcher.H for _, catcher in layers])

    def _compute_inverse_hessian(H: mx.array) -> mx.array:
        """
        Compute inverse Hessian using Cholesky decomposition.

        Args:
            H: Hessian matrix (d x d)

        Returns:
            Upper triangular Cholesky factor of inverse Hessian
        """
        diag = mx.arange(H.shape[0])
        diag_vals = H[diag, diag]

        # Dead input features (zero — or effectively-zero — Hessian diagonal: an
        # activation channel that is always (near-)zero, common for attention
        # o_proj inputs) make H singular, so the Cholesky inverse comes back as
        # NaN and corrupts the layer's weights. Set those diagonal entries to 1
        # so the factorization stays well-defined; such columns then receive a
        # trivial, stable quantization update. This is the standard GPTQ
        # dead-feature guard, using a magnitude-relative cutoff (see
        # `_dead_feature_mask`) rather than an exact `== 0` so it does not rely
        # solely on the damping below to neutralize tiny-but-nonzero channels.
        dead = _dead_feature_mask(diag_vals)
        diag_vals = mx.where(dead, mx.ones_like(diag_vals), diag_vals)

        # Add damping to the diagonal for numerical stability.
        damp = 1e-2 * mx.mean(diag_vals)
        damped_diag = diag_vals + damp
        H[diag, diag] = damped_diag

        # Compute inverse via Cholesky decomposition (requires CPU in current MLX)
        cpu_device = mx.Device(mx.cpu)
        with mx.stream(mx.new_stream(cpu_device)):
            chol = mx.linalg.cholesky(H)
            inv = mx.linalg.cholesky_inv(chol)
            Hinv = mx.linalg.cholesky(inv, upper=True)
            mx.eval(Hinv)

        # Some real-activation Hessians are too rank-deficient for a stable
        # Cholesky even after the dead-feature guard + damping, and the factor
        # comes back non-finite. Rather than let NaN corrupt the weights, fall
        # back to a diagonal inverse: GPTQ then applies no cross-column error
        # compensation for this layer (each column is simply quantized, i.e.
        # plain per-group quantization), which is stable and never NaN.
        if not bool(mx.all(mx.isfinite(Hinv))):
            Hinv = mx.diag(1.0 / mx.sqrt(damped_diag))

        return Hinv

    @mx.compile
    def _gptq_error(w: mx.array, d: float, scales: mx.array, biases: mx.array) -> mx.array:
        """
        Compute quantization error for GPTQ.

        Args:
            w: Weight column
            d: Diagonal element of inverse Hessian
            scales: Quantization scales
            biases: Quantization biases

        Returns:
            Quantization error divided by Hessian diagonal
        """
        n_bins = 2**bits - 1
        q = mx.clip(mx.round((w - biases) / scales), 0.0, n_bins)
        q = scales * q + biases
        return (w - q) / d

    # Quantize each layer with GPTQ algorithm
    print(f"Quantizing {len(layers)} layers with GPTQ ({bits}-bit)...")
    for layer_idx, (key, catcher) in enumerate(layers):
        print(f"  [{layer_idx + 1}/{len(layers)}] {key}")

        # Compute inverse Hessian
        Hinv = _compute_inverse_hessian(catcher.H)
        del catcher.H  # Free memory
        mx.eval(Hinv)

        # Get original dtype and weights
        orig_type = catcher.module.weight.dtype
        W = catcher.module.weight.astype(mx.float32)

        # Quantize group by group
        all_scales = []
        all_biases = []

        for group_start in range(0, W.shape[-1], group_size):
            group_end = group_start + group_size
            W_group = W[..., group_start:group_end]
            err = mx.zeros_like(W_group)

            # Compute scales and biases for this group
            _, scales, biases = mx.quantize(W_group, bits=bits, group_size=group_size)
            all_scales.append(scales)
            all_biases.append(biases)

            # Iteratively quantize each column in group, compensating error
            for col_offset in range(group_size):
                col = group_start + col_offset
                if col >= W.shape[-1]:
                    break

                w_col = W[..., col : col + 1]
                d = Hinv[col, col]

                # Compute quantization error
                e = _gptq_error(w_col, d, scales, biases)

                # Compensate error in remaining columns within group
                W[..., col:group_end] -= e @ Hinv[col : col + 1, col:group_end]
                err[..., col_offset : col_offset + 1] = e

                # Materialize the lazy graph every _GPTQ_EVAL_COLUMN_STRIDE
                # columns rather than every column. The read-after-write chain
                # makes this output-identical; the stride just bounds graph depth
                # so peak memory stays flat while cutting the host-sync count
                # (see _GPTQ_EVAL_COLUMN_STRIDE for the measured trade-off).
                if (col_offset + 1) % _GPTQ_EVAL_COLUMN_STRIDE == 0:
                    mx.eval(err, W)

            # Compensate error in columns outside this group
            if group_end < W.shape[-1]:
                W[..., group_end:] -= err @ Hinv[group_start:group_end, group_end:]

            # Settle the group (including the just-applied outside-group
            # compensation) before moving on, so each group's deferred work
            # never spills into the next group's working set.
            mx.eval(W, err)

        # Pack quantized weights
        scales = mx.concatenate(all_scales, axis=-1)
        biases = mx.concatenate(all_biases, axis=-1)
        W_quantized = _quantize_weights(W, bits, scales, biases)

        # Create quantized layer
        quantized_layer = catcher.module.to_quantized(bits=bits, group_size=group_size)
        quantized_layer.weight = W_quantized
        quantized_layer.scales = scales
        quantized_layer.biases = biases
        quantized_layer.set_dtype(orig_type)
        mx.eval(quantized_layer)

        layers[layer_idx] = (key, quantized_layer)

    # Update model with quantized layers
    model.update_modules(tree_unflatten(layers))

    # Done - only Linear layers are quantized
    print(
        f"✓ GPTQ quantization complete: {len(layers)} Linear layers ({bits}-bit, group_size={group_size})"
    )
    print("  Other layers (Embedding, etc.) remain unquantized")
    print("  Use mlx.nn.quantize() to quantize embeddings if needed")

    return model
