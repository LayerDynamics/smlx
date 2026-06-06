"""
Utilities for quantization in SMLX.

Provides shared functionality for calibration data loading,
layer conversion, and quantization helpers optimized for
"smol" models on Apple M4 chipsets.
"""

from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.nn import QuantizedEmbedding, QuantizedLinear


def load_calibration_data(
    tokenizer,
    num_samples: int = 128,
    sequence_length: int = 512,
    dataset: str = "default",
    verbose: bool = True,
) -> mx.array:
    """
    Load calibration dataset for quantization (GPTQ, AWQ).

    Downloads and caches calibration text data, then tokenizes it into
    random non-overlapping chunks suitable for quantization calibration.

    Args:
        tokenizer: HuggingFace tokenizer for encoding text
        num_samples: Number of calibration samples to use (default: 128)
        sequence_length: Length of each sequence (default: 512)
        dataset: Dataset to use ("default", "wikitext", or path to file)
        verbose: Print progress messages (default: True)

    Returns:
        MLX array of shape (num_samples, sequence_length) containing token IDs

    Example:
        ```python
        from transformers import AutoTokenizer
        from smlx.quant import load_calibration_data

        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        data = load_calibration_data(tokenizer, num_samples=128)
        # Returns shape: (128, 512)
        ```

    Notes:
        - Data is cached in ~/.cache/smlx/calibration/
        - First run downloads ~1MB calibration text from GitHub Gist
        - Random chunks ensure diverse calibration coverage
    """
    # Determine cache directory
    cache_dir = Path.home() / ".cache/smlx/calibration"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load or download calibration text
    if dataset == "default":
        save_path = cache_dir / "calibration_v5.txt"
        if not save_path.exists():
            from urllib import request

            if verbose:
                print(f"Downloading calibration data to {save_path}...")
            url = "https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw/571fda718462de863e5a0171078c175420c7649a/calibration_data_v5_rc.txt"
            request.urlretrieve(url, save_path)
            if verbose:
                print("Download complete!")

        with open(save_path) as f:
            texts = f.read()

    elif dataset == "wikitext":
        save_path = cache_dir / "wikitext2.txt"
        if not save_path.exists():
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "datasets package required for WikiText. "
                    "Install with: pip install datasets"
                ) from None

            if verbose:
                print(f"Downloading WikiText-2 to {save_path}...")
            dataset_obj = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = "\n\n".join(str(item["text"]) for item in dataset_obj)
            with open(save_path, "w") as f:
                f.write(texts)
            if verbose:
                print("Download complete!")
        else:
            with open(save_path) as f:
                texts = f.read()

    else:
        # Assume it's a path to a local file
        local_path = Path(dataset)
        if not local_path.exists():
            raise FileNotFoundError(
                f"Calibration dataset not found: {dataset}\n"
                f"Use 'default', 'wikitext', or provide a path to a text file"
            )
        with open(local_path) as f:
            texts = f.read()

    # Tokenize
    if verbose:
        print("Tokenizing calibration data...")

    # Handle different tokenizer interfaces
    if hasattr(tokenizer, "encode"):
        # Use standard HuggingFace tokenizer (returns list of token IDs)
        token_ids = tokenizer.encode(texts)
        # Convert to MLX array
        tokens = mx.array(token_ids, dtype=mx.int32)
    else:
        raise ValueError("Tokenizer must have an 'encode' method")

    # Select random non-overlapping chunks
    # Truncate to multiple of sequence_length
    num_tokens = (tokens.size // sequence_length) * sequence_length
    tokens = tokens[:num_tokens]

    # Reshape into chunks
    tokens = tokens.reshape(-1, sequence_length)

    # Random permutation for diverse coverage
    segments = mx.random.permutation(tokens.shape[0])

    # Limit to num_samples
    if num_samples > 0:
        if num_samples > segments.shape[0]:
            if verbose:
                print(
                    f"Warning: Requested {num_samples} samples but only "
                    f"{segments.shape[0]} available. Using all available samples."
                )
            num_samples = segments.shape[0]
        segments = segments[:num_samples]

    calibration_data = tokens[segments]

    if verbose:
        print(
            f"Loaded {calibration_data.shape[0]} calibration samples "
            f"of length {calibration_data.shape[1]}"
        )

    return calibration_data


def estimate_model_size(model, dtype=mx.float16) -> dict[str, float]:
    """
    Estimate model size in MB.

    Args:
        model: MLX model
        dtype: Data type to assume for unquantized weights

    Returns:
        Dictionary with size breakdown:
        - total_mb: Total model size in MB
        - parameters: Total number of parameters
        - quantized_mb: Size of quantized parameters
        - unquantized_mb: Size of unquantized parameters
    """
    total_params = 0
    quantized_bytes = 0
    unquantized_bytes = 0

    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            weight = module.weight

            if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
                # Quantized: bits per weight + scales + biases
                bits = module.bits
                group_size = module.group_size
                num_weights = weight.size * 32 // bits

                # Weight storage
                quantized_bytes += weight.nbytes
                # Scales and biases (float16/float32)
                num_groups = (num_weights + group_size - 1) // group_size
                if (
                    hasattr(module, "scales")
                    and module.scales is not None
                    and hasattr(module, "biases")
                    and module.biases is not None
                ):
                    quantized_bytes += num_groups * (
                        module.scales.itemsize + module.biases.itemsize
                    )

                total_params += num_weights
            else:
                # Unquantized
                unquantized_bytes += weight.nbytes
                total_params += weight.size

    return {
        "total_mb": (quantized_bytes + unquantized_bytes) / (1024 ** 2),
        "parameters": total_params,
        "quantized_mb": quantized_bytes / (1024 ** 2),
        "unquantized_mb": unquantized_bytes / (1024 ** 2),
    }


def quantize_dequantize(
    weight: mx.array,
    bits: int = 4,
    group_size: int = 64,
) -> mx.array:
    """
    Quantize and immediately dequantize weights.

    Useful for testing quantization impact without creating quantized layers.

    Args:
        weight: Weight array to quantize
        bits: Bits per weight (default: 4)
        group_size: Group size for quantization (default: 64)

    Returns:
        Dequantized weight array
    """
    w_q, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    return mx.dequantize(
        w_q, scales, biases, group_size=group_size, bits=bits, dtype=weight.dtype
    )


def check_m4_compatibility(model) -> dict[str, Any]:
    """
    Check if model is suitable for M4 with 36GB unified memory.

    Args:
        model: MLX model to check

    Returns:
        Dictionary with compatibility info:
        - is_smol: Whether model is "smol" (<10B params)
        - estimated_size_mb: Estimated model size
        - recommended_quantization: Suggested quantization strategy
        - fits_in_memory: Whether it fits in 36GB M4
    """
    size_info = estimate_model_size(model)
    total_params = size_info["parameters"]
    total_mb = size_info["total_mb"]

    # Conservative estimate: leave 10GB for system and activations
    available_mb = 26 * 1024  # 26GB available

    is_smol = total_params < 10e9  # <10B parameters
    fits_in_memory = total_mb < available_mb

    # Recommend quantization strategy
    if total_params < 3e9:  # <3B
        recommendation = "No quantization needed, or 8-bit for fine-tuning"
    elif total_params < 7e9:  # 3-7B
        recommendation = "4-bit quantization recommended for fine-tuning"
    elif total_params < 13e9:  # 7-13B
        recommendation = "4-bit quantization required, consider GPTQ or AWQ"
    else:  # >13B
        recommendation = "3-4 bit quantization required, use AWQ for best quality"

    return {
        "is_smol": is_smol,
        "estimated_size_mb": total_mb,
        "total_parameters": total_params,
        "recommended_quantization": recommendation,
        "fits_in_memory": fits_in_memory,
        "available_memory_mb": available_mb,
    }


def measure_memory_savings(original_model, quantized_model) -> dict[str, Any]:
    """
    Measure actual memory savings from quantization.

    Compares the actual memory footprint of original vs quantized models.

    Args:
        original_model: Original unquantized model
        quantized_model: Quantized model

    Returns:
        Dictionary with memory comparison:
        - original_mb: Original model size in MB
        - quantized_mb: Quantized model size in MB
        - savings_mb: Memory saved in MB
        - reduction_ratio: Compression ratio (original / quantized)
        - reduction_percent: Percentage reduction

    Example:
        ```python
        from smlx.quant import quantize_model_q4_0, measure_memory_savings
        import copy

        original = copy.deepcopy(model)
        quantize_model_q4_0(model)
        stats = measure_memory_savings(original, model)
        print(f"Saved {stats['savings_mb']:.1f} MB ({stats['reduction_percent']:.1f}%)")
        ```
    """
    original_size = estimate_model_size(original_model)
    quantized_size = estimate_model_size(quantized_model)

    original_mb = original_size["total_mb"]
    quantized_mb = quantized_size["total_mb"]
    savings_mb = original_mb - quantized_mb
    reduction_ratio = original_mb / quantized_mb if quantized_mb > 0 else 1.0
    reduction_percent = (savings_mb / original_mb * 100) if original_mb > 0 else 0.0

    return {
        "original_mb": original_mb,
        "quantized_mb": quantized_mb,
        "savings_mb": savings_mb,
        "reduction_ratio": reduction_ratio,
        "reduction_percent": reduction_percent,
        "original_params": original_size["parameters"],
        "quantized_params": quantized_size["parameters"],
    }


def get_actual_model_size(model) -> dict[str, float]:
    """
    Get the actual current memory footprint of a model.

    This function measures the actual bytes used by the model's parameters,
    including both quantized and unquantized layers.

    Args:
        model: MLX model

    Returns:
        Dictionary with actual size information:
        - total_mb: Total model size in MB
        - total_bytes: Total size in bytes
        - parameters: Total number of parameters
        - quantized_layers: Number of quantized layers
        - unquantized_layers: Number of unquantized layers

    Example:
        ```python
        from smlx.quant import get_actual_model_size

        size = get_actual_model_size(model)
        print(f"Model uses {size['total_mb']:.2f} MB")
        ```
    """
    total_bytes = 0
    total_params = 0
    quantized_layers = 0
    unquantized_layers = 0

    for _, module in model.named_modules():
        if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
            quantized_layers += 1

            # For quantized layers, sum up all components
            if hasattr(module, "weight"):
                total_bytes += module.weight.nbytes
                # Count actual parameters based on bits
                bits = module.bits if hasattr(module, "bits") else 4
                total_params += module.weight.size * 32 // bits

            if hasattr(module, "scales") and module.scales is not None:
                total_bytes += module.scales.nbytes

            if hasattr(module, "biases") and module.biases is not None:
                total_bytes += module.biases.nbytes

            if hasattr(module, "bias") and module.bias is not None:
                total_bytes += module.bias.nbytes
                total_params += module.bias.size

        elif hasattr(module, "weight"):
            unquantized_layers += 1

            # Unquantized layers
            total_bytes += module.weight.nbytes
            total_params += module.weight.size

            if hasattr(module, "bias") and module.bias is not None:
                total_bytes += module.bias.nbytes
                total_params += module.bias.size

    return {
        "total_mb": total_bytes / (1024 ** 2),
        "total_bytes": total_bytes,
        "parameters": total_params,
        "quantized_layers": quantized_layers,
        "unquantized_layers": unquantized_layers,
    }


# ============================================================================
# GGML K-Quants Bit-Packing Utilities
# ============================================================================


def pack_scales_mins_6bit(scales: mx.array, mins: mx.array) -> mx.array:
    """
    Pack 8 × 6-bit scales and 8 × 6-bit mins into 12 bytes (GGML Q4_K format).

    This implements the GGML bit-packing scheme for k-quants where 16 values
    (8 scales + 8 mins), each quantized to 6 bits, are packed into 12 bytes.

    The packing layout follows GGML convention:
    - 96 bits total (8 scales × 6 bits + 8 mins × 6 bits)
    - Packed into 12 uint8 bytes with specific bit positions

    Args:
        scales: Array of 8 scale values in range [0, 63] (6-bit)
        mins: Array of 8 min values in range [0, 63] (6-bit)

    Returns:
        Packed uint8 array of shape (..., 12) containing bit-packed data

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant.utils import pack_scales_mins_6bit

        scales = mx.array([10, 20, 30, 40, 50, 60, 62, 63], dtype=mx.uint8)
        mins = mx.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=mx.uint8)
        packed = pack_scales_mins_6bit(scales, mins)  # Shape: (12,)
        ```

    Notes:
        - Input values must be in [0, 63] range (6-bit unsigned)
        - Follows GGML bit-packing convention for compatibility
        - Used in Q4_K, Q5_K, Q6_K quantization formats
    """
    # Ensure inputs are uint8 and in valid range
    scales = mx.clip(scales, 0, 63).astype(mx.uint8)
    mins = mx.clip(mins, 0, 63).astype(mx.uint8)

    # Handle batch dimension
    original_shape = scales.shape[:-1] if len(scales.shape) > 1 else ()
    scales_flat = scales.reshape(-1, 8) if len(scales.shape) > 1 else scales.reshape(1, 8)
    mins_flat = mins.reshape(-1, 8) if len(mins.shape) > 1 else mins.reshape(1, 8)

    batch_size = scales_flat.shape[0]

    # Build packed array using list comprehension for each batch
    packed_list = []
    for b in range(batch_size):
        s = scales_flat[b]
        m = mins_flat[b]

        # Pack scales (8 × 6 bits = 48 bits = 6 bytes)
        # Byte 0: scale[0] (6 bits) + scale[1][5:4] (2 bits)
        byte0 = int(s[0]) | ((int(s[1]) & 0x30) << 2)
        # Byte 1: scale[1][3:0] (4 bits) + scale[2][5:2] (4 bits)
        byte1 = ((int(s[1]) & 0x0F) << 4) | ((int(s[2]) & 0x3C) >> 2)
        # Byte 2: scale[2][1:0] (2 bits) + scale[3] (6 bits)
        byte2 = ((int(s[2]) & 0x03) << 6) | int(s[3])
        # Byte 3: scale[4] (6 bits) + scale[5][5:4] (2 bits)
        byte3 = int(s[4]) | ((int(s[5]) & 0x30) << 2)
        # Byte 4: scale[5][3:0] (4 bits) + scale[6][5:2] (4 bits)
        byte4 = ((int(s[5]) & 0x0F) << 4) | ((int(s[6]) & 0x3C) >> 2)
        # Byte 5: scale[6][1:0] (2 bits) + scale[7] (6 bits)
        byte5 = ((int(s[6]) & 0x03) << 6) | int(s[7])

        # Pack mins (8 × 6 bits = 48 bits = 6 bytes)
        # Byte 6: min[0] (6 bits) + min[1][5:4] (2 bits)
        byte6 = int(m[0]) | ((int(m[1]) & 0x30) << 2)
        # Byte 7: min[1][3:0] (4 bits) + min[2][5:2] (4 bits)
        byte7 = ((int(m[1]) & 0x0F) << 4) | ((int(m[2]) & 0x3C) >> 2)
        # Byte 8: min[2][1:0] (2 bits) + min[3] (6 bits)
        byte8 = ((int(m[2]) & 0x03) << 6) | int(m[3])
        # Byte 9: min[4] (6 bits) + min[5][5:4] (2 bits)
        byte9 = int(m[4]) | ((int(m[5]) & 0x30) << 2)
        # Byte 10: min[5][3:0] (4 bits) + min[6][5:2] (4 bits)
        byte10 = ((int(m[5]) & 0x0F) << 4) | ((int(m[6]) & 0x3C) >> 2)
        # Byte 11: min[6][1:0] (2 bits) + min[7] (6 bits)
        byte11 = ((int(m[6]) & 0x03) << 6) | int(m[7])

        packed_list.append(
            [byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7, byte8, byte9, byte10, byte11]
        )

    packed = mx.array(packed_list, dtype=mx.uint8)

    # Reshape back to original batch shape
    if original_shape:
        packed = packed.reshape(*original_shape, 12)
    else:
        packed = packed.squeeze(0)

    return packed


def unpack_scales_mins_6bit(packed: mx.array) -> tuple[mx.array, mx.array]:
    """
    Unpack 12 bytes into 8 × 6-bit scales and 8 × 6-bit mins (GGML Q4_K format).

    Inverse of pack_scales_mins_6bit(). Extracts the bit-packed scale and min values.

    Args:
        packed: Packed uint8 array of shape (..., 12)

    Returns:
        Tuple of (scales, mins):
        - scales: Array of shape (..., 8) with values in [0, 63]
        - mins: Array of shape (..., 8) with values in [0, 63]

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant.utils import pack_scales_mins_6bit, unpack_scales_mins_6bit

        scales = mx.array([10, 20, 30, 40, 50, 60, 62, 63], dtype=mx.uint8)
        mins = mx.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=mx.uint8)
        packed = pack_scales_mins_6bit(scales, mins)
        s_unpacked, m_unpacked = unpack_scales_mins_6bit(packed)
        # s_unpacked == scales, m_unpacked == mins
        ```
    """
    # Handle batch dimension
    original_shape = packed.shape[:-1] if len(packed.shape) > 1 else ()
    packed_flat = packed.reshape(-1, 12) if len(packed.shape) > 1 else packed.reshape(1, 12)

    batch_size = packed_flat.shape[0]

    # Build unpacked arrays using list comprehension for each batch
    scales_list = []
    mins_list = []

    for b in range(batch_size):
        p = packed_flat[b]

        # Unpack scales (bytes 0-5)
        # scale[0] from byte 0[5:0]
        s0 = int(p[0]) & 0x3F
        # scale[1] from byte 0[7:6] (upper 2 bits) + byte 1[7:4] (lower 4 bits)
        s1 = ((int(p[0]) >> 6) << 4) | (int(p[1]) >> 4)
        # scale[2] from byte 1[3:0] (upper 4 bits) + byte 2[7:6] (lower 2 bits)
        s2 = ((int(p[1]) & 0x0F) << 2) | (int(p[2]) >> 6)
        # scale[3] from byte 2[5:0]
        s3 = int(p[2]) & 0x3F
        # scale[4] from byte 3[5:0]
        s4 = int(p[3]) & 0x3F
        # scale[5] from byte 3[7:6] (upper 2 bits) + byte 4[7:4] (lower 4 bits)
        s5 = ((int(p[3]) >> 6) << 4) | (int(p[4]) >> 4)
        # scale[6] from byte 4[3:0] (upper 4 bits) + byte 5[7:6] (lower 2 bits)
        s6 = ((int(p[4]) & 0x0F) << 2) | (int(p[5]) >> 6)
        # scale[7] from byte 5[5:0]
        s7 = int(p[5]) & 0x3F

        # Unpack mins (bytes 6-11)
        # min[0] from byte 6[5:0]
        m0 = int(p[6]) & 0x3F
        # min[1] from byte 6[7:6] (upper 2 bits) + byte 7[7:4] (lower 4 bits)
        m1 = ((int(p[6]) >> 6) << 4) | (int(p[7]) >> 4)
        # min[2] from byte 7[3:0] (upper 4 bits) + byte 8[7:6] (lower 2 bits)
        m2 = ((int(p[7]) & 0x0F) << 2) | (int(p[8]) >> 6)
        # min[3] from byte 8[5:0]
        m3 = int(p[8]) & 0x3F
        # min[4] from byte 9[5:0]
        m4 = int(p[9]) & 0x3F
        # min[5] from byte 9[7:6] (upper 2 bits) + byte 10[7:4] (lower 4 bits)
        m5 = ((int(p[9]) >> 6) << 4) | (int(p[10]) >> 4)
        # min[6] from byte 10[3:0] (upper 4 bits) + byte 11[7:6] (lower 2 bits)
        m6 = ((int(p[10]) & 0x0F) << 2) | (int(p[11]) >> 6)
        # min[7] from byte 11[5:0]
        m7 = int(p[11]) & 0x3F

        scales_list.append([s0, s1, s2, s3, s4, s5, s6, s7])
        mins_list.append([m0, m1, m2, m3, m4, m5, m6, m7])

    scales = mx.array(scales_list, dtype=mx.uint8)
    mins = mx.array(mins_list, dtype=mx.uint8)

    # Reshape back to original batch shape
    if original_shape:
        scales = scales.reshape(*original_shape, 8)
        mins = mins.reshape(*original_shape, 8)
    else:
        scales = scales.squeeze(0)
        mins = mins.squeeze(0)

    return scales, mins


def pack_weights_4bit(weights: mx.array) -> mx.array:
    """
    Pack 4-bit weights (2 per byte) for GGML formats.

    Packs an array of 4-bit values [0, 15] into uint8 bytes with 2 values per byte.
    First weight in lower 4 bits, second weight in upper 4 bits.

    Args:
        weights: Array of values in range [0, 15], must have even last dimension

    Returns:
        Packed uint8 array with shape (..., N//2) where N is last dimension of input

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant.utils import pack_weights_4bit

        weights = mx.array([0, 1, 2, 3, 14, 15, 0, 1], dtype=mx.uint8)
        packed = pack_weights_4bit(weights)  # Shape: (4,)
        # packed[0] = 0x10 (low=0, high=1)
        # packed[1] = 0x32 (low=2, high=3)
        ```

    Notes:
        - Input must have even number of elements in last dimension
        - Values clipped to [0, 15] range
        - Packing: byte = low_nibble | (high_nibble << 4)
    """
    # Clip to 4-bit range
    weights = mx.clip(weights, 0, 15).astype(mx.uint8)

    # Get shape info
    *batch_dims, n_weights = weights.shape

    if n_weights % 2 != 0:
        raise ValueError(f"Last dimension must be even for 4-bit packing, got {n_weights}")

    # Reshape for packing
    weights_flat = weights.reshape(-1, n_weights)
    n_packed = n_weights // 2

    # Extract low and high nibbles
    low = weights_flat[:, ::2]  # Even indices
    high = weights_flat[:, 1::2]  # Odd indices

    # Pack: low in bits [3:0], high in bits [7:4]
    packed = low | (high << 4)

    # Reshape back
    if batch_dims:
        packed = packed.reshape(*batch_dims, n_packed)
    else:
        packed = packed.squeeze(0)

    return packed


def unpack_weights_4bit(packed: mx.array, n_weights: int) -> mx.array:
    """
    Unpack 4-bit weights from packed bytes (2 per byte).

    Inverse of pack_weights_4bit(). Extracts 4-bit values from packed uint8 bytes.

    Args:
        packed: Packed uint8 array with shape (..., N//2)
        n_weights: Number of weights to extract (must be even)

    Returns:
        Unpacked uint8 array with shape (..., n_weights) containing values [0, 15]

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant.utils import pack_weights_4bit, unpack_weights_4bit

        weights = mx.array([0, 1, 2, 3, 14, 15, 0, 1], dtype=mx.uint8)
        packed = pack_weights_4bit(weights)
        unpacked = unpack_weights_4bit(packed, 8)
        # unpacked == weights
        ```

    Notes:
        - n_weights must be even and match packed array size
        - Extracts low nibble (bits [3:0]) and high nibble (bits [7:4])
    """
    if n_weights % 2 != 0:
        raise ValueError(f"n_weights must be even for 4-bit unpacking, got {n_weights}")

    # Get shape info
    *batch_dims, n_packed = packed.shape
    expected_packed = n_weights // 2

    if n_packed != expected_packed:
        raise ValueError(
            f"Packed array size {n_packed} doesn't match expected {expected_packed} "
            f"for {n_weights} weights"
        )

    # Reshape for unpacking
    packed_flat = packed.reshape(-1, n_packed)

    # Extract nibbles
    low = packed_flat & 0x0F  # Lower 4 bits
    high = (packed_flat >> 4) & 0x0F  # Upper 4 bits

    # Interleave low and high
    unpacked = mx.zeros((packed_flat.shape[0], n_weights), dtype=mx.uint8)
    unpacked = mx.put_along_axis(
        unpacked, mx.arange(0, n_weights, 2).reshape(1, -1), low, axis=1
    )
    unpacked = mx.put_along_axis(
        unpacked, mx.arange(1, n_weights, 2).reshape(1, -1), high, axis=1
    )

    # Reshape back
    if batch_dims:
        unpacked = unpacked.reshape(*batch_dims, n_weights)
    else:
        unpacked = unpacked.squeeze(0)

    return unpacked


# ============================================================================
# MXFP8 (Microscaling FP8) Utilities
# ============================================================================


def validate_mxfp8_shape(weights: mx.array, raise_error: bool = True) -> bool:
    """
    Validate that weight tensor shape is compatible with MXFP8 quantization.

    MXFP8 requires the last dimension to be divisible by 32 (OCP spec).

    Args:
        weights: Weight tensor to validate
        raise_error: If True, raise ValueError on invalid shape. If False, return bool.

    Returns:
        True if shape is valid, False otherwise (only if raise_error=False)

    Raises:
        ValueError: If shape is invalid and raise_error=True

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant.utils import validate_mxfp8_shape

        weights = mx.random.normal((768, 768))  # Valid: 768 % 32 == 0
        validate_mxfp8_shape(weights)  # Returns True

        weights = mx.random.normal((770, 770))  # Invalid: 770 % 32 != 0
        validate_mxfp8_shape(weights, raise_error=False)  # Returns False
        ```

    Notes:
        - MXFP8 uses fixed block size of 32 elements (OCP MX v1.0 spec)
        - Use pad_for_mxfp8() to auto-pad invalid shapes
    """
    last_dim = weights.shape[-1]
    is_valid = last_dim % 32 == 0

    if not is_valid and raise_error:
        raise ValueError(
            f"MXFP8 requires last dimension divisible by 32 (OCP spec), "
            f"got {last_dim}. Use pad_for_mxfp8() to auto-pad."
        )

    return is_valid


def pad_for_mxfp8(weights: mx.array) -> tuple[mx.array, int]:
    """
    Pad weight tensor to be compatible with MXFP8 quantization.

    Pads the last dimension to the nearest multiple of 32 (OCP spec requirement).
    Padding uses zeros.

    Args:
        weights: Weight tensor to pad

    Returns:
        Tuple of (padded_weights, original_size):
        - padded_weights: Padded tensor with last dim divisible by 32
        - original_size: Original size of last dimension (for unpadding)

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant.utils import pad_for_mxfp8
        from smlx.quant import quantize_to_mxfp8

        weights = mx.random.normal((770, 770))  # Not divisible by 32
        padded, orig_size = pad_for_mxfp8(weights)
        # padded.shape = (770, 800)  # 800 = ceil(770/32) * 32

        # Quantize padded tensor
        w_q, scales = quantize_to_mxfp8(padded)

        # After dequantization, remove padding:
        # restored = restored[..., :orig_size]
        ```

    Notes:
        - Padding is zero-filled
        - Remember to slice back to original size after dequantization
        - No padding added if already divisible by 32
    """
    *batch_dims, last_dim = weights.shape
    original_size = last_dim

    # Check if padding needed
    if last_dim % 32 == 0:
        return weights, original_size

    # Calculate padded size
    padded_dim = ((last_dim + 31) // 32) * 32
    pad_amount = padded_dim - last_dim

    # Create padding array
    pad_shape = (*batch_dims, pad_amount)
    padding = mx.zeros(pad_shape, dtype=weights.dtype)

    # Concatenate
    padded_weights = mx.concatenate([weights, padding], axis=-1)

    return padded_weights, original_size


def compare_mxfp8_vs_int8(
    weights: mx.array,
    group_size_int8: int = 128,
) -> dict[str, Any]:
    """
    Compare MXFP8 and INT8 quantization for a weight tensor.

    Useful for deciding which quantization method to use.

    Args:
        weights: Weight tensor to compare
        group_size_int8: Group size for INT8 quantization (default: 128)

    Returns:
        Dictionary with comparison metrics:
        - mxfp8_error: Mean absolute error for MXFP8
        - int8_error: Mean absolute error for INT8
        - mxfp8_max_error: Max absolute error for MXFP8
        - int8_max_error: Max absolute error for INT8
        - mxfp8_size_bytes: Storage size for MXFP8 (weights + scales)
        - int8_size_bytes: Storage size for INT8 (weights + scales)
        - recommendation: Which method to use ("mxfp8", "int8", or "similar")

    Example:
        ```python
        import mlx.core as mx
        from smlx.quant.utils import compare_mxfp8_vs_int8

        weights = mx.random.normal((768, 768))
        comparison = compare_mxfp8_vs_int8(weights)

        print(f"MXFP8 error: {comparison['mxfp8_error']:.6f}")
        print(f"INT8 error: {comparison['int8_error']:.6f}")
        print(f"Recommendation: {comparison['recommendation']}")
        ```

    Notes:
        - Uses mlx.quantize() for both methods
        - MXFP8: Fixed group_size=32, uint8 scales
        - INT8: Flexible group_size, float32 scales
        - Recommendation based on error and size
    """
    from smlx.quant import quantize_to_mxfp8, dequantize_from_mxfp8

    # Ensure weights are compatible with MXFP8
    orig_size = weights.shape[-1]
    if weights.shape[-1] % 32 != 0:
        weights_mxfp8, _ = pad_for_mxfp8(weights)
    else:
        weights_mxfp8 = weights

    # Quantize with MXFP8
    w_mxfp8, scales_mxfp8 = quantize_to_mxfp8(weights_mxfp8)
    w_restored_mxfp8 = dequantize_from_mxfp8(w_mxfp8, scales_mxfp8)

    # Remove padding if added
    if weights_mxfp8.shape[-1] != orig_size:
        w_restored_mxfp8 = w_restored_mxfp8[..., :orig_size]

    # Ensure weights are compatible with INT8 group_size
    if weights.shape[-1] % group_size_int8 != 0:
        # Pad to nearest multiple of group_size_int8
        pad_size = ((weights.shape[-1] + group_size_int8 - 1) // group_size_int8) * group_size_int8
        weights_int8 = mx.pad(weights, [(0, 0)] * (weights.ndim - 1) + [(0, pad_size - weights.shape[-1])])
    else:
        weights_int8 = weights

    # Quantize with INT8
    w_int8, scales_int8, biases_int8 = mx.quantize(
        weights_int8, group_size=group_size_int8, bits=8
    )
    w_restored_int8 = mx.dequantize(
        w_int8,
        scales_int8,
        biases_int8,
        group_size=group_size_int8,
        bits=8,
        dtype=weights.dtype,
    )

    # Remove padding if added
    if weights_int8.shape[-1] != orig_size:
        w_restored_int8 = w_restored_int8[..., :orig_size]

    # Calculate errors
    mxfp8_error = float(mx.mean(mx.abs(w_restored_mxfp8 - weights)))
    int8_error = float(mx.mean(mx.abs(w_restored_int8 - weights)))
    mxfp8_max_error = float(mx.max(mx.abs(w_restored_mxfp8 - weights)))
    int8_max_error = float(mx.max(mx.abs(w_restored_int8 - weights)))

    # Calculate sizes
    mxfp8_size = w_mxfp8.nbytes + scales_mxfp8.nbytes
    int8_size = w_int8.nbytes + scales_int8.nbytes + biases_int8.nbytes

    # Generate recommendation
    if mxfp8_error < int8_error * 0.9 and mxfp8_size <= int8_size:
        recommendation = "mxfp8"
        reason = "Better quality with same or less memory"
    elif int8_error < mxfp8_error * 0.9:
        recommendation = "int8"
        reason = "Better quality"
    elif mxfp8_size < int8_size * 0.9:
        recommendation = "mxfp8"
        reason = "Smaller size with similar quality"
    elif int8_size < mxfp8_size * 0.9:
        recommendation = "int8"
        reason = "Smaller size with similar quality"
    else:
        recommendation = "similar"
        reason = "Similar quality and size - use INT8 for M4 speed"

    return {
        "mxfp8_error": mxfp8_error,
        "int8_error": int8_error,
        "mxfp8_max_error": mxfp8_max_error,
        "int8_max_error": int8_max_error,
        "mxfp8_size_bytes": mxfp8_size,
        "int8_size_bytes": int8_size,
        "recommendation": recommendation,
        "reason": reason,
        "mxfp8_block_size": 32,
        "int8_group_size": group_size_int8,
    }


def estimate_mxfp8_size(model) -> dict[str, Any]:
    """
    Estimate model size after MXFP8 quantization.

    Args:
        model: MLX model to estimate

    Returns:
        Dictionary with size estimates:
        - current_mb: Current model size in MB
        - mxfp8_mb: Estimated size after MXFP8 quantization in MB
        - savings_mb: Memory saved in MB
        - reduction_ratio: Compression ratio
        - notes: Important notes about the estimate

    Example:
        ```python
        from smlx.quant.utils import estimate_mxfp8_size
        from smlx.models.SmolLM2_135M import load

        model, _ = load("mlx-community/SmolLM2-135M-Instruct")
        stats = estimate_mxfp8_size(model)

        print(f"Current: {stats['current_mb']:.1f} MB")
        print(f"MXFP8: {stats['mxfp8_mb']:.1f} MB")
        print(f"Savings: {stats['reduction_ratio']:.2f}x")
        ```

    Notes:
        - Assumes FP16 baseline
        - MXFP8: 1 byte/param + 1 byte scale/32 params
        - Actual size may vary slightly due to padding
    """
    size_info = estimate_model_size(model, dtype=mx.float16)
    current_mb = size_info["total_mb"]
    total_params = size_info["parameters"]

    # MXFP8 storage: 1 byte per parameter + 1 byte scale per 32 parameters
    mxfp8_bytes = total_params * 1.03125  # 1 + 1/32 bytes per parameter
    mxfp8_mb = mxfp8_bytes / (1024 ** 2)

    savings_mb = current_mb - mxfp8_mb
    reduction_ratio = current_mb / mxfp8_mb if mxfp8_mb > 0 else 1.0

    return {
        "current_mb": current_mb,
        "mxfp8_mb": mxfp8_mb,
        "savings_mb": savings_mb,
        "reduction_ratio": reduction_ratio,
        "parameters": total_params,
        "notes": (
            "Estimate assumes FP16 baseline. "
            "MXFP8 uses 1.03125 bytes/param (1 byte + 1/32 scale). "
            "Actual size may vary due to layer-specific padding requirements."
        ),
    }

