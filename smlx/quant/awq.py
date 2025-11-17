"""
AWQ (Activation-Aware Weight Quantization) for LLM compression.

AWQ protects only 1% of the most salient weight channels during quantization,
achieving better accuracy than uniform quantization. Salient channels are
identified based on activation magnitudes rather than weight values.

The algorithm finds optimal per-channel scaling factors to minimize quantization
error, then applies weight clipping to further reduce outliers. This activation-aware
approach significantly improves quality at low bit widths (3-4 bit).

Optimized for "smol" models (<10B parameters) on Apple M4 chipsets.

Reference:
    AWQ: Activation-aware Weight Quantization for On-Device LLM Compression
    https://arxiv.org/abs/2306.00978
    Winner of MLSys 2024 Best Paper Award

Algorithm:
    1. Capture input features for each layer
    2. Search for optimal scaling factors via grid search (minimize MSE)
    3. Apply scales to adjacent layers (LayerNorm/Linear)
    4. Search for optimal weight clipping thresholds
    5. Apply final quantization with learned parameters

Example:
    ```python
    import mlx.core as mx
    from smlx.quant import awq_quantize, load_calibration_data, llama_awq
    from transformers import AutoTokenizer

    # Load model and calibration data
    model = load_your_model()
    tokenizer = AutoTokenizer.from_pretrained("model_name")
    calibration_data = load_calibration_data(tokenizer, num_samples=128)

    # Quantize with AWQ (4-bit, optimized for M4)
    quantized_model = awq_quantize(
        model,
        calibration_data,
        awq_config=llama_awq,  # Or mistral_awq, qwen_awq
        bits=4,
        group_size=64
    )
    ```
"""

from dataclasses import dataclass, field
from typing import Callable, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_map_with_path


@dataclass
class ScaleConfig:
    """
    Configuration for a single scaling operation in AWQ.

    Defines which layers to scale together and how to apply the scales
    to the previous operation (LayerNorm, Linear, etc.).

    Args:
        prev: Key path to the previous operation (e.g., "input_layernorm")
        layers: List of layer key paths to scale (e.g., ["q_proj", "k_proj"])
        block: Optional block key to scope the operation (e.g., "self_attn")
        kwargs: List of additional kwargs needed for forward pass (e.g., ["mask"])
        use_config: Optional predicate to conditionally apply this config
    """

    prev: str
    layers: list[str]
    block: Union[str, None] = None
    kwargs: list[str] = field(default_factory=list)
    use_config: Union[Callable, None] = None


@dataclass
class AWQConfig:
    """
    Model-specific AWQ configuration.

    Defines the architecture-specific layer names and scaling configurations
    for applying AWQ quantization.

    Args:
        embed: Key path to embedding layer (e.g., "embed_tokens")
        lm_head: Key path to language model head (e.g., "lm_head")
        no_clip: List of layer keys to skip clipping (e.g., ["q_proj", "k_proj"])
        scale_configs: List of ScaleConfig objects defining scaling operations
    """

    embed: str
    lm_head: str
    no_clip: list[str]
    scale_configs: list[ScaleConfig]


# ===== Model-Specific Configurations =====

llama_awq = AWQConfig(
    embed="embed_tokens",
    lm_head="lm_head",
    no_clip=["q_proj", "k_proj"],  # Don't clip attention query/key projections
    scale_configs=[
        # Scale attention projections together
        ScaleConfig(
            block="self_attn",
            prev="input_layernorm",
            layers=["q_proj", "k_proj", "v_proj"],
            kwargs=["mask"],
        ),
        # Scale MLP down projection
        ScaleConfig(
            prev="mlp.up_proj",
            layers=["mlp.down_proj"],
        ),
        # Scale MLP gate and up projections together
        ScaleConfig(
            block="mlp",
            prev="post_attention_layernorm",
            layers=["gate_proj", "up_proj"],
        ),
    ],
)

# Mistral and Qwen use same architecture as Llama
mistral_awq = llama_awq
qwen_awq = llama_awq

AWQ_MODEL_CONFIGS = {
    "llama": llama_awq,
    "mistral": mistral_awq,
    "qwen2": qwen_awq,
    "qwen3": qwen_awq,
}


# ===== Helper Functions =====


def _mse(x: mx.array, y: mx.array) -> mx.array:
    """Compute mean squared error between two arrays."""
    return ((x.astype(mx.float32) - y.astype(mx.float32)) ** 2).mean()


def _submodule_from_key(module: nn.Module, key: str) -> nn.Module:
    """Navigate to submodule using dot-separated key path."""
    keys = key.split(".")
    for k in keys:
        module = module[k]
    return module


def _run_layer(
    layer: nn.Module,
    x: mx.array,
    batch_size: int = 32,
    **kwargs,
) -> mx.array:
    """
    Run layer in batches to avoid OOM on large inputs.

    Args:
        layer: Module to run
        x: Input tensor of shape (num_samples, ...)
        batch_size: Batch size for processing
        **kwargs: Additional arguments for layer forward pass

    Returns:
        Concatenated output from all batches
    """
    y = []
    for i in range(0, x.shape[0], batch_size):
        batch_out = layer(x[i : i + batch_size], **kwargs)
        y.append(batch_out)
        mx.eval(y)
    return mx.concatenate(y, axis=0)


# ===== Core AWQ Functions =====


def search_best_scale(
    layers: list[nn.Module],
    quantize_func: Callable,
    block: Union[nn.Module, None],
    layer_kwargs: dict,
    n_grid: int = 20,
) -> mx.array:
    """
    Search for optimal per-channel scaling factors.

    AWQ key insight: protect salient channels identified by activation magnitudes.
    This function searches across different scaling ratios to minimize MSE between
    original and quantized outputs.

    Args:
        layers: List of layers to scale together
        quantize_func: Function to quantize weights (quantize + dequantize)
        block: Optional parent block containing layers
        layer_kwargs: Additional kwargs for block forward pass (e.g., mask)
        n_grid: Number of grid points to search (default: 20)

    Returns:
        Optimal per-channel scaling factors of shape (input_dims,)

    Algorithm:
        For ratio in [1/n_grid, 2/n_grid, ..., n_grid/n_grid]:
            scales = (activation_max ** ratio).normalized()
            quantize weights with scales
            compute MSE(original_output, quantized_output)
            keep scales with minimum MSE
    """
    layer_kwargs = layer_kwargs or {}

    # Get input features captured during forward pass
    if not hasattr(layers[0], "input_feat") or layers[0].input_feat is None:
        raise RuntimeError(
            "Layer input features not captured. Run forward pass with Catcher first."
        )
    x = layers[0].input_feat

    # Compute original output
    block = block or layers[0]
    out = _run_layer(block, x, **layer_kwargs)

    # Compute activation magnitudes for salient channel detection
    x_max = x.abs().mean(axis=(0, 1))

    best_error = float("inf")
    best_scales = mx.ones(x_max.shape)  # Initialize with ones instead of None

    # Save original weights - use tree_map to create a copy
    original_params = tree_map(lambda x: x, block.parameters())

    # Grid search over scaling ratios (start from 1 to avoid degenerate ratio=0)
    for ratio_idx in range(n_grid):
        ratio = (ratio_idx + 1) / n_grid  # ratios: 1/n_grid to n_grid/n_grid

        # Compute scales: x_max^ratio normalized by geometric mean
        scales = mx.maximum(x_max**ratio, 1e-4).reshape(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()

        # Apply scales to layers and quantize
        for layer in layers:
            if isinstance(layer, nn.Linear):
                layer.weight = quantize_func(layer.weight * scales) / scales

        # Compute quantized output
        out_q = _run_layer(block, x, **layer_kwargs)

        # Compute MSE loss
        loss = _mse(out, out_q)
        mx.eval(loss)

        if loss.item() < best_error:
            best_error = loss.item()
            best_scales = scales

        # Restore original weights for next iteration
        block.update(original_params)

    best_scales = best_scales.reshape(-1)
    mx.eval(best_scales)
    return best_scales


def apply_scale(
    prev_op: nn.Module,
    layers: list[nn.Module],
    scales: mx.array,
):
    """
    Fuse scaling factors into adjacent layers.

    AWQ absorbs per-channel scales into the previous operation (LayerNorm/Linear)
    and the target layers. This maintains equivalence while protecting salient channels.

    Args:
        prev_op: Previous operation (LayerNorm, RMSNorm, or Linear)
        layers: Target layers to scale
        scales: Per-channel scaling factors

    Algorithm:
        If prev_op is Linear:
            prev_op.weight /= scales (absorb inverse scale)
            layer.weight *= scales (apply scale)
        If prev_op is LayerNorm/RMSNorm:
            prev_op.weight /= scales (normalize by inverse scale)
            layer.weight *= scales (apply scale)

    Note:
        Input features are also scaled to maintain correctness during clipping search.
    """
    if isinstance(prev_op, nn.Linear):
        assert len(layers) == 1, "Linear prev_op only supports single target layer"
        assert prev_op.weight is not None, "prev_op.weight cannot be None"
        assert layers[0].weight is not None, "layer weight cannot be None"
        prev_op.weight = prev_op.weight / scales[:, mx.newaxis]
        if hasattr(prev_op, "bias") and prev_op.bias is not None:
            prev_op.bias = prev_op.bias / scales
        layers[0].weight = layers[0].weight * scales

    elif isinstance(prev_op, (nn.LayerNorm, nn.RMSNorm)):
        assert prev_op.weight is not None, "prev_op.weight cannot be None"
        prev_op.weight = prev_op.weight / scales
        if hasattr(prev_op, "bias") and prev_op.bias is not None:
            prev_op.bias = prev_op.bias / scales
        for layer in layers:
            assert layer.weight is not None, "layer weight cannot be None"
            layer.weight = layer.weight * scales

    else:
        raise NotImplementedError(f"Scale application for {type(prev_op)} not supported")

    # Update input features for subsequent clipping search
    for layer in layers:
        if hasattr(layer, "input_feat") and layer.input_feat is not None:
            layer.input_feat = layer.input_feat / scales


def scale_block(
    block: nn.Module,
    configs: list[ScaleConfig],
    quantize_func: Callable,
    layer_kwargs: dict,
    n_grid: int = 20,
):
    """
    Apply AWQ scaling to a transformer block.

    Iterates through scale configurations and applies optimal scales
    to each group of layers.

    Args:
        block: Transformer block to scale
        configs: List of ScaleConfig objects
        quantize_func: Quantization function
        layer_kwargs: Forward pass kwargs (e.g., attention mask)
        n_grid: Grid search resolution
    """
    for conf in configs:
        # Skip if config predicate fails
        if conf.use_config is not None and not conf.use_config(block):
            continue

        # Get target layers
        if conf.block is not None:
            local_block = _submodule_from_key(block, conf.block)
            layers = [_submodule_from_key(local_block, layer_key) for layer_key in conf.layers]
        else:
            local_block = None
            layers = [_submodule_from_key(block, layer_key) for layer_key in conf.layers]

        # Build local kwargs from config
        local_kwargs = {k: layer_kwargs[k] for k in conf.kwargs if k in layer_kwargs}
        for k in conf.kwargs:
            if hasattr(layers[0], k):
                local_kwargs[k] = getattr(layers[0], k)

        # Search and apply scales
        scales = search_best_scale(
            layers=layers,
            block=local_block,
            layer_kwargs=local_kwargs,
            quantize_func=quantize_func,
            n_grid=n_grid,
        )
        apply_scale(_submodule_from_key(block, conf.prev), layers, scales)


def search_best_clip(
    module: nn.Module,
    quantize_func: Callable,
    group_size: int,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    batch_size: int = 64,
    n_frames: int = 512,
) -> mx.array:
    """
    Search for optimal weight clipping thresholds per group.

    Weight clipping reduces outliers that cause large quantization errors.
    This function searches for the best clipping threshold that minimizes
    MSE between original and quantized activations.

    Args:
        module: Linear module to clip
        quantize_func: Quantization function
        group_size: Group size for quantization
        n_grid: Number of grid points for search (default: 20)
        max_shrink: Maximum shrinkage ratio (default: 0.5 = clip to 50% of max)
        batch_size: Batch size for processing weights (default: 64)
        n_frames: Number of input frames to sample (default: 512)

    Returns:
        Clipped weights with optimal thresholds per group

    Algorithm:
        For each weight group:
            For clip_ratio in [1.0, 0.95, 0.90, ..., 0.5]:
                clip_threshold = clip_ratio * max_abs_weight
                quantize clipped weights
                compute MSE on activations
                keep threshold with minimum MSE
    """
    # Subsample input features to save memory
    if not hasattr(module, "input_feat") or module.input_feat is None:
        raise RuntimeError(
            "Module input features not captured. Run forward pass with Catcher first."
        )

    input_feat: mx.array = (
        module.input_feat
    )  # Type assertion: guaranteed non-None after check above
    x = input_feat.flatten(0, 1)
    stride = (x.shape[0] + n_frames - 1) // n_frames
    x = x[::stride]

    w = module.weight
    assert w is not None, "module.weight cannot be None"
    x = x.reshape(x.shape[0], -1, group_size)

    w_init_shape = w.shape
    w_all = mx.flatten(w, 0, w.ndim - 2)
    w_max_all = []

    # Process weights in batches to save memory
    for b in range(0, w_all.shape[0], batch_size):
        w = w_all[b : b + batch_size]

        group_shape = (w.shape[0], w.shape[-1] // group_size)
        best_error = mx.full(group_shape, float("inf"))
        best_w_max = mx.zeros((*group_shape, 1), dtype=x.dtype)

        w_shape = w.shape
        w = w.reshape(*w.shape[:-1], -1, group_size)

        # Compute original activations
        out = mx.einsum("bdg,odg->bod", x, w)
        init_max = w.abs().max(axis=-1, keepdims=True)

        # Grid search over clipping ratios
        for i in range(int(max_shrink * n_grid)):
            p = 1 - i / n_grid  # Clipping ratio: 1.0 to 0.5
            w_max = p * init_max
            w_m = mx.clip(w, -w_max, w_max).reshape(w_shape)

            # Quantize clipped weights
            w_q = quantize_func(w_m)
            w_q = w_q.reshape(*w_q.shape[:-1], -1, group_size)

            # Compute quantized activations
            out_q = mx.einsum("bdg,odg->bod", x, w_q)

            # MSE loss per group
            loss = ((out - out_q) ** 2).sum(axis=0) / out.shape[0]

            # Update best thresholds
            best_indices = loss < best_error
            best_error = mx.where(best_indices, loss, best_error)
            best_w_max = mx.where(best_indices[..., mx.newaxis], w_max, best_w_max)
            mx.eval(best_w_max, best_error)

        w_max_all.append(best_w_max)

    # Concatenate and apply best clipping thresholds
    best_w_max = mx.concatenate(w_max_all, axis=0)
    w_r = w_all.reshape(*w_all.shape[:-1], -1, group_size)
    best_w = mx.clip(w_r, -best_w_max, best_w_max)
    best_w = best_w.reshape(w_init_shape)

    mx.eval(best_w)
    return best_w


def clip_block(
    block: nn.Module,
    no_clip_keys: list[str],
    quantize_func: Callable,
    group_size: int,
    n_grid: int = 20,
):
    """
    Apply weight clipping to all Linear layers in a block.

    Args:
        block: Transformer block
        no_clip_keys: Layer keys to skip clipping (e.g., ["q_proj", "k_proj"])
        quantize_func: Quantization function
        group_size: Group size for quantization
        n_grid: Grid search resolution
    """

    def apply_clip(path, module):
        # Only clip Linear layers not in no_clip_keys
        if isinstance(module, nn.Linear) and all(k not in path for k in no_clip_keys):
            best_weight = search_best_clip(
                module,
                quantize_func=quantize_func,
                group_size=group_size,
                n_grid=n_grid,
            )
            module.weight = best_weight

    tree_map_with_path(apply_clip, block.leaf_modules(), is_leaf=nn.Module.is_module)


def awq_quantize(
    model: nn.Module,
    calibration_data: mx.array,
    awq_config: AWQConfig,
    bits: int = 4,
    group_size: int = 64,
    embed_bits: int = 4,
    embed_group_size: int = 32,
    n_grid: int = 20,
) -> nn.Module:
    """
    Quantize model using AWQ (Activation-Aware Weight Quantization).

    AWQ achieves superior accuracy at low bit widths by protecting salient weight
    channels identified through activation magnitudes. This is especially effective
    for 3-4 bit quantization on "smol" models.

    Args:
        model: MLX model to quantize (must have model.model.layers structure)
        calibration_data: Calibration tokens of shape (num_samples, seq_length)
        awq_config: Model-specific AWQ configuration (e.g., llama_awq)
        bits: Bits for main layers (default: 4 for M4)
        group_size: Group size for main layers (default: 64 for M4)
        embed_bits: Bits for embeddings (default: 4)
        embed_group_size: Group size for embeddings (default: 32)
        n_grid: Grid search resolution (default: 20)

    Returns:
        Quantized model with all Linear, Embedding, and LM head layers quantized

    Algorithm:
        1. Quantize embedding layer
        2. For each transformer block:
            a. Capture input features for all Linear layers
            b. Search and apply optimal scaling factors
            c. Search and apply optimal weight clipping
            d. Quantize block with learned parameters
            e. Verify loss reduction (fallback to naive if worse)
        3. Quantize LM head

    Note:
        - Model is modified in-place
        - Optimized for M4 with 4-bit, group_size=64 defaults
        - Works best with 128+ calibration samples
        - Typical speedup: 3-4x with <5% accuracy loss

    Example:
        ```python
        from smlx.quant import awq_quantize, llama_awq, load_calibration_data

        # Load calibration data
        calibration_data = load_calibration_data(tokenizer, num_samples=128)

        # Quantize with AWQ
        model = awq_quantize(
            model,
            calibration_data,
            awq_config=llama_awq,
            bits=4,
            group_size=64
        )
        ```
    """
    print(f"Starting AWQ quantization ({bits}-bit, group_size={group_size})...")

    def quantize_func(w):
        """Quantize-dequantize function for search."""
        original_dtype = w.dtype
        wq = mx.quantize(w, bits=bits, group_size=group_size)
        return mx.dequantize(*wq, bits=bits, group_size=group_size, dtype=original_dtype)

    # Create attention mask for transformer blocks if model supports it
    mask = None
    create_mask_fn = getattr(model, "create_attention_mask", None)
    if create_mask_fn is not None:
        mask = create_mask_fn(calibration_data)

    # Get base model (either model.model or model itself)
    base_model = getattr(model, "model", model)
    if base_model is None:
        raise ValueError("Model cannot be None")
    if not hasattr(base_model, "layers"):
        raise ValueError("Model must have a 'layers' attribute containing transformer blocks")

    # Type assertion: base_model is guaranteed non-None after the check above
    assert base_model is not None

    # Quantize embedding layer
    embed_key = awq_config.embed
    print(f"Quantizing embedding: {embed_key}")
    if hasattr(base_model[embed_key], "to_quantized"):
        base_model[embed_key] = base_model[embed_key].to_quantized(
            group_size=embed_group_size, bits=embed_bits
        )
    else:
        # Fallback: use nn.quantize
        base_model[embed_key] = nn.QuantizedEmbedding.from_embedding(
            base_model[embed_key],
            group_size=embed_group_size,
            bits=embed_bits,
        )

    # Get embeddings for calibration data
    inputs = base_model[embed_key](calibration_data)

    # Catcher module to capture input features
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __call__(self, x: mx.array, *args, **kwargs):
            # Store input features on original module
            if hasattr(self.module, "input_feat"):
                self.module.input_feat = mx.concatenate([self.module.input_feat, x], axis=0)
            else:
                self.module.input_feat = x
            return self.module(x, *args, **kwargs)

    # Process each transformer block
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise ValueError("Model layers cannot be None")

    # Type assertion: layers is guaranteed non-None after the check above
    assert layers is not None

    num_layers = len(layers)
    print(f"Quantizing {num_layers} transformer blocks...")

    for layer_idx, block in enumerate(layers):
        print(f"  Block {layer_idx + 1}/{num_layers}")

        # Capture input features for all Linear layers
        orig_leaves = block.leaf_modules()
        capture_leaves = tree_map(
            lambda m: Catcher(m) if isinstance(m, nn.Linear) else m,
            orig_leaves,
            is_leaf=nn.Module.is_module,
        )
        block.update_modules(capture_leaves)

        # Run forward pass to capture features
        layer_kwargs = {"mask": mask} if mask is not None else {}
        outputs = _run_layer(block, inputs, **layer_kwargs)

        # Restore original modules
        block.update_modules(orig_leaves)
        del capture_leaves

        # Compute baseline loss (naive quantization)
        nn.quantize(block, group_size=group_size, bits=bits)
        outputs_q = _run_layer(block, inputs, **layer_kwargs)
        before_loss = _mse(outputs, outputs_q)
        mx.eval(before_loss)

        # Restore original weights
        block.update_modules(orig_leaves)
        orig_params = block.parameters()

        # Apply AWQ: scale then clip
        scale_block(
            block=block,
            configs=awq_config.scale_configs,
            quantize_func=quantize_func,
            n_grid=n_grid,
            layer_kwargs=layer_kwargs,
        )

        clip_block(
            block=block,
            no_clip_keys=awq_config.no_clip,
            quantize_func=quantize_func,
            group_size=group_size,
            n_grid=n_grid,
        )

        # Quantize with AWQ parameters
        nn.quantize(block, group_size=group_size, bits=bits)
        outputs_q = _run_layer(block, inputs, **layer_kwargs)
        after_loss = _mse(outputs, outputs_q)
        mx.eval(after_loss)

        # Check if AWQ improves quality
        loss_reduction = after_loss / before_loss
        print(f"    Loss reduction: {loss_reduction.item():.4f}")

        if after_loss > before_loss:
            # AWQ made it worse - fallback to naive quantization
            print("    Warning: AWQ increased loss, using naive quantization")
            block.update_modules(orig_leaves)
            block.update(orig_params)
            nn.quantize(block, group_size=group_size, bits=bits)

        # Update inputs for next block
        inputs = outputs

        mx.eval(block)
        mx.clear_cache()

    # Quantize LM head
    lm_head = awq_config.lm_head
    if hasattr(base_model, lm_head):
        print(f"Quantizing LM head: {lm_head}")
        lm_head_module = base_model[lm_head]
        if hasattr(lm_head_module, "to_quantized"):
            base_model[lm_head] = lm_head_module.to_quantized(
                group_size=embed_group_size, bits=embed_bits
            )
        else:
            base_model[lm_head] = nn.QuantizedLinear.from_linear(
                lm_head_module,
                group_size=embed_group_size,
                bits=embed_bits,
            )

    print("AWQ quantization complete!")
    print(f"  Layers quantized: {num_layers} blocks + embedding + LM head")
    print(f"  Bits: {bits}-bit (embeddings: {embed_bits}-bit)")
    print(f"  Group size: {group_size} (embeddings: {embed_group_size})")

    return model
