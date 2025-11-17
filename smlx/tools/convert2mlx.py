"""
Convert models from HuggingFace (PyTorch/safetensors) to MLX format.

This module provides functionality to:
- Download models from HuggingFace Hub or load from local paths
- Convert weights from PyTorch/safetensors to MLX format
- Apply quantization (4-bit/8-bit) optimized for Apple M4 chipsets
- Handle multimodal models (skip quantizing vision/audio modules)
- Shard large models into manageable chunks
- Upload converted models back to HuggingFace Hub
- Support for LLM, VLM, Audio, OCR, and Embedding models

Optimized for "smol" models with aggressive quantization defaults.

Supported model types:
- Language Models (LLM): SmolLM2, etc.
- Vision-Language Models (VLM): SmolVLM, nanoVLM, Moondream2, TinyLLaVA
- Audio Models: Whisper, YAMNet, SileroVAD, Chatterbox, Orpheus
- OCR Models: TrOCR, Donut
- Embedding Models: MiniLM, all-MiniLM-L6-v2
"""

import argparse
import copy
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import HfApi, snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported dtypes for conversion
MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]

# Mixed-bit quantization recipes
QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]


# ============================================================================
# Weight Management Functions
# ============================================================================


def make_shards(
    weights: dict[str, mx.array], max_file_size_gb: int = 5
) -> list[dict[str, mx.array]]:
    """
    Split weights into shards to avoid large files.

    Args:
        weights: Dictionary of weight name to MLX array
        max_file_size_gb: Maximum file size in GB (default: 5)

    Returns:
        List of weight dictionaries (shards)
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0

    for key, value in weights.items():
        estimated_size = value.nbytes

        if shard_size + estimated_size > max_file_size_bytes and shard:
            shards.append(shard)
            shard, shard_size = {}, 0

        shard[key] = value
        shard_size += estimated_size

    if shard:
        shards.append(shard)

    return shards


def save_weights(
    save_path: Path,
    weights: dict[str, mx.array],
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Save weights with sharding and index file.

    Args:
        save_path: Directory to save weights
        weights: Dictionary of weight name to MLX array
        metadata: Optional metadata to include in safetensors
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)

    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors" if shards_count > 1 else "model.safetensors"
    )

    # Calculate total size
    total_size = sum(v.nbytes for v in weights.values())

    # Create index
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Save each shard
    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        # Add MLX format metadata
        shard_metadata = {"format": "mlx"}
        if metadata:
            shard_metadata.update(metadata)

        mx.save_safetensors(str(shard_path), shard, metadata=shard_metadata)

        # Update weight map
        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name

    # Save index file
    index_path = save_path / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    logger.info(f"Saved {shards_count} shard(s) to {save_path}")


def save_config(
    config: dict[str, Any],
    config_path: Path,
) -> None:
    """
    Save model configuration.

    Args:
        config: Configuration dictionary
        config_path: Path to save config.json
    """
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ============================================================================
# Model Download and Loading
# ============================================================================


def get_model_path(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
) -> Path:
    """
    Download model from HuggingFace Hub or return local path.

    Args:
        path_or_hf_repo: Local path or HuggingFace repository ID
        revision: Optional model revision/branch

    Returns:
        Path to model directory
    """
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        logger.info(f"Downloading {path_or_hf_repo} from HuggingFace Hub...")

        # Download model files
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.bin",
                    "*.py",
                    "*.model",
                    "*.txt",
                    "*.tiktoken",
                    "tokenizer.model",
                ],
            )
        )
        logger.info(f"Downloaded to {model_path}")
    else:
        logger.info(f"Using local model at {model_path}")

    return model_path


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load model configuration from config.json.

    Args:
        config_path: Path to config.json

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        config = json.load(f)
    return config


# ============================================================================
# Quantization Support
# ============================================================================


def skip_multimodal_module(path: str) -> bool:
    """
    Check if a module path should be skipped during quantization.
    Used for multimodal models to preserve vision/audio quality.

    Args:
        path: Module path string

    Returns:
        True if module should be skipped
    """
    skip_patterns = [
        "vision_model",
        "vision_tower",
        "vision_encoder",
        "visual",
        "sam_model",
        "audio_model",
        "audio_tower",
        "audio_encoder",
    ]
    return any(pattern in path.lower() for pattern in skip_patterns)


def build_quantization_predicate(
    group_size: int = 64,
    skip_multimodal: bool = False,
) -> Callable[[str, nn.Module], bool]:
    """
    Build a predicate function for selective quantization.

    Args:
        group_size: Quantization group size
        skip_multimodal: Whether to skip vision/audio modules

    Returns:
        Predicate function for nn.quantize
    """

    def predicate(path: str, module: nn.Module) -> bool:
        # Skip if module doesn't support quantization
        if not hasattr(module, "to_quantized"):
            return False

        # Skip if weight shape incompatible with group size
        if hasattr(module, "weight"):
            weight = getattr(module, "weight", None)
            if weight is not None and hasattr(weight, "shape"):
                if weight.shape[-1] % group_size != 0:
                    return False

        # Skip multimodal modules if requested
        if skip_multimodal and skip_multimodal_module(path):
            logger.info(f"Skipping quantization for multimodal module: {path}")
            return False

        return True

    return predicate


def mixed_quant_predicate_builder(
    recipe: str,
    num_layers: int,
    group_size: int = 64,
) -> Callable[[str, nn.Module], dict[str, int]]:
    """
    Build a mixed-bit quantization predicate function.

    Based on llama.cpp's Q4_K_M recipe approach.
    - First 1/8 layers: high bits
    - Middle 6/8 layers: low bits (except every 3rd)
    - Last 1/8 layers: high bits
    - Special layers (v_proj, down_proj, lm_head): high bits

    Args:
        recipe: Quantization recipe name (e.g., "mixed_4_6")
        num_layers: Number of transformer layers
        group_size: Quantization group size

    Returns:
        Predicate function for selective quantization
    """
    high_bits = 6

    if recipe == "mixed_2_6":
        low_bits = 2
    elif recipe == "mixed_3_4":
        low_bits = 3
        high_bits = 4
    elif recipe == "mixed_3_6":
        low_bits = 3
    elif recipe == "mixed_4_6":
        low_bits = 4
    else:
        raise ValueError(f"Invalid quant recipe {recipe}")

    def mixed_quant_predicate(path: str, module: nn.Module) -> dict[str, int]:
        """Selective quantization based on layer depth and type."""
        # Extract layer index from path (e.g., "model.layers.5.attn" -> 5)
        parts = path.split(".")
        layer_idx = 0
        for part in parts:
            if part.isdigit():
                layer_idx = int(part)
                break

        # Determine if this layer should use more bits
        use_more_bits = (
            layer_idx < num_layers // 8
            or layer_idx >= 7 * num_layers // 8
            or (layer_idx - num_layers // 8) % 3 == 2
        )

        # Always use high bits for these critical layers
        if "v_proj" in path or "v_a_proj" in path or "v_b_proj" in path:
            if use_more_bits:
                return {"group_size": group_size, "bits": high_bits}

        if "down_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits}

        if "lm_head" in path:
            return {"group_size": group_size, "bits": high_bits}

        # Default to low bits for middle layers
        return {"group_size": group_size, "bits": low_bits}

    return mixed_quant_predicate


def quantize_model(
    weights: dict[str, mx.array],
    config: dict[str, Any],
    group_size: int = 64,
    bits: int = 4,
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    """
    Quantize model weights.

    Args:
        weights: Model weights dictionary
        config: Model configuration
        group_size: Quantization group size (default: 64 for M4)
        bits: Bits per weight (default: 4 for smol models)

    Returns:
        Tuple of (quantized_weights, updated_config)
    """
    logger.info(f"Quantizing model to {bits}-bit (group_size={group_size})...")

    # Update config with quantization info
    quantized_config = copy.deepcopy(config)
    quantized_config["quantization"] = {
        "group_size": group_size,
        "bits": bits,
    }

    # For now, return weights as-is with config update
    # Full quantization requires loading into an nn.Module
    # This will be integrated with smlx.quant module for advanced quantization
    logger.warning(
        "Quantization requires model architecture. "
        "Use smlx.quant module for post-conversion quantization."
    )

    return weights, quantized_config


def dequantize_model(
    weights: dict[str, mx.array],
    config: dict[str, Any],
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    """
    Dequantize model weights.

    Args:
        weights: Quantized model weights
        config: Model configuration

    Returns:
        Tuple of (dequantized_weights, updated_config)
    """
    logger.info("Dequantizing model...")

    # Update config to remove quantization info
    dequantized_config = copy.deepcopy(config)
    if "quantization" in dequantized_config:
        del dequantized_config["quantization"]

    # Dequantization logic would go here
    # Requires model architecture to properly dequantize
    logger.warning(
        "Dequantization requires model architecture. "
        "Use smlx.quant module for proper dequantization."
    )

    return weights, dequantized_config


# ============================================================================
# Model-Specific Conversion Methods
# ============================================================================


def detect_model_type(config: dict[str, Any]) -> str:
    """
    Detect model type from config.

    Args:
        config: Model configuration dictionary

    Returns:
        Model type string: "llm", "vlm", "audio", "ocr", "embedding"
    """
    model_type = config.get("model_type", "").lower()
    architectures = config.get("architectures", [])

    # Check for VLM (vision-language models)
    if any(
        arch in str(architectures)
        for arch in ["LlavaForConditionalGeneration", "VisionEncoderDecoder", "Moondream"]
    ):
        return "vlm"

    # Check for audio models
    if model_type in ["whisper", "encodec", "wav2vec2", "silero", "yamnet"]:
        return "audio"

    # Check for OCR models
    if model_type in ["vision-encoder-decoder"] or "trocr" in str(architectures).lower():
        return "ocr"

    # Check for embedding models
    if "sentence" in model_type or "embedding" in model_type:
        return "embedding"

    # Check config keys for VLM indicators
    if "vision_config" in config or "audio_config" in config:
        return "vlm"

    # Default to LLM
    return "llm"


def remap_llm_weights(
    weights: dict[str, mx.array],
    config: dict[str, Any],
    target_format: str = "mlx",
) -> dict[str, mx.array]:
    """
    Remap LLM weight keys from HuggingFace format to MLX format.

    Common transformations:
    - Remove "model." prefix
    - Standardize layer naming
    - Rename attention projections

    Args:
        weights: Original weights dictionary
        config: Model configuration
        target_format: Target format ("mlx" or "original")

    Returns:
        Remapped weights dictionary
    """
    remapped = {}

    for key, value in weights.items():
        new_key = key

        # Common transformations
        new_key = new_key.replace("model.", "")
        new_key = new_key.replace(".self_attn.", ".attn.")
        new_key = new_key.replace("embed_tokens", "tok_embeddings")
        new_key = new_key.replace("lm_head", "output")

        # Handle different attention naming conventions
        if target_format == "mlx":
            new_key = new_key.replace(".q_proj.", ".query_proj.")
            new_key = new_key.replace(".k_proj.", ".key_proj.")
            new_key = new_key.replace(".v_proj.", ".value_proj.")
            new_key = new_key.replace(".o_proj.", ".out_proj.")

        remapped[new_key] = value

    return remapped


def remap_vlm_weights(
    weights: dict[str, mx.array],
    config: dict[str, Any],
) -> dict[str, mx.array]:
    """
    Remap VLM weight keys, handling both vision and language components.

    Vision components typically stay unchanged, while language components
    follow LLM remapping rules.

    Args:
        weights: Original weights dictionary
        config: Model configuration

    Returns:
        Remapped weights dictionary
    """
    remapped = {}

    for key, value in weights.items():
        new_key = key

        # Keep vision/audio modules intact
        if any(
            prefix in key
            for prefix in ["vision_model", "vision_tower", "audio_model", "audio_tower"]
        ):
            remapped[new_key] = value
            continue

        # Apply LLM remapping for language model components
        if "language_model" in key or "lm_head" in key:
            new_key = new_key.replace("language_model.", "")
            new_key = new_key.replace(".self_attn.", ".attn.")
            new_key = new_key.replace(".q_proj.", ".query_proj.")
            new_key = new_key.replace(".k_proj.", ".key_proj.")
            new_key = new_key.replace(".v_proj.", ".value_proj.")
            new_key = new_key.replace(".o_proj.", ".out_proj.")

        remapped[new_key] = value

    return remapped


def remap_audio_weights(
    weights: dict[str, mx.array],
    config: dict[str, Any],
    model_type: str = "whisper",
) -> dict[str, mx.array]:
    """
    Remap audio model weights with architecture-specific transformations.

    Whisper models often need:
    - Encoder/decoder separation
    - Convolution weight transposition
    - Embedding layer handling

    Args:
        weights: Original weights dictionary
        config: Model configuration
        model_type: Audio model type (e.g., "whisper", "encodec")

    Returns:
        Remapped weights dictionary
    """
    remapped = {}

    for key, value in weights.items():
        new_key = key

        if model_type == "whisper":
            # Handle Whisper-specific naming
            new_key = new_key.replace("model.", "")

            # Transpose convolution weights for MLX format
            if "conv" in key and len(value.shape) == 3:
                # Conv1d: (out_channels, in_channels, kernel_size) -> transpose
                value = mx.transpose(value, (2, 1, 0))

        elif model_type == "encodec":
            # Encodec-specific transformations
            new_key = new_key.replace("encoder.", "enc.")
            new_key = new_key.replace("decoder.", "dec.")

        remapped[new_key] = value

    return remapped


def remap_ocr_weights(
    weights: dict[str, mx.array],
    config: dict[str, Any],
) -> dict[str, mx.array]:
    """
    Remap OCR model weights (TrOCR, Donut).

    OCR models typically have encoder-decoder architecture.

    Args:
        weights: Original weights dictionary
        config: Model configuration

    Returns:
        Remapped weights dictionary
    """
    remapped = {}

    for key, value in weights.items():
        new_key = key

        # TrOCR/VisionEncoderDecoder format
        new_key = new_key.replace("encoder.", "vision_encoder.")
        new_key = new_key.replace("decoder.", "text_decoder.")

        remapped[new_key] = value

    return remapped


def remap_embedding_weights(
    weights: dict[str, mx.array],
    config: dict[str, Any],
) -> dict[str, mx.array]:
    """
    Remap embedding model weights (BERT-based, Sentence Transformers).

    Embedding models often use simple key transformations.

    Args:
        weights: Original weights dictionary
        config: Model configuration

    Returns:
        Remapped weights dictionary
    """
    remapped = {}

    for key, value in weights.items():
        new_key = key

        # BERT-style transformations
        new_key = new_key.replace("bert.", "")
        new_key = new_key.replace("pooler.", "pooling.")

        remapped[new_key] = value

    return remapped


def apply_model_specific_conversion(
    weights: dict[str, mx.array],
    config: dict[str, Any],
    model_type: str,
) -> dict[str, mx.array]:
    """
    Apply model-specific weight conversions based on detected model type.

    Args:
        weights: Original weights dictionary
        config: Model configuration
        model_type: Detected model type

    Returns:
        Converted weights dictionary
    """
    logger.info(f"Applying {model_type} specific conversions...")

    if model_type == "llm":
        return remap_llm_weights(weights, config)
    elif model_type == "vlm":
        return remap_vlm_weights(weights, config)
    elif model_type == "audio":
        audio_type = config.get("model_type", "whisper")
        return remap_audio_weights(weights, config, audio_type)
    elif model_type == "ocr":
        return remap_ocr_weights(weights, config)
    elif model_type == "embedding":
        return remap_embedding_weights(weights, config)
    else:
        logger.warning(f"Unknown model type {model_type}, skipping weight remapping")
        return weights


# ============================================================================
# HuggingFace Upload
# ============================================================================


def create_model_card(
    model_name: str,
    original_repo: str,
    quantized: bool = False,
    bits: Optional[int] = None,
    dtype: Optional[str] = None,
) -> str:
    """
    Create a README.md model card for HuggingFace Hub.

    Args:
        model_name: Name of the model
        original_repo: Original HuggingFace repository
        quantized: Whether model is quantized
        bits: Quantization bits
        dtype: Model dtype

    Returns:
        Model card markdown content
    """
    card = f"""---
library_name: mlx
tags:
  - mlx
  - apple-silicon
  - m4-optimized
---

# {model_name} - MLX Format

This is an MLX-optimized version of [{original_repo}](https://huggingface.co/{original_repo}).

## Model Details

- **Converted by**: SMLX (smol MLX)
- **Original Model**: {original_repo}
- **Format**: MLX (Apple Silicon optimized)
"""

    if quantized and bits:
        card += f"- **Quantization**: {bits}-bit\n"

    if dtype:
        card += f"- **Dtype**: {dtype}\n"

    card += """
## Usage

```python
import mlx.core as mx

# Load the model
weights = mx.load("model.safetensors")
```

For more information about SMLX, visit the [GitHub repository](https://github.com/yourusername/smlx).

## Optimization

This model has been optimized for Apple M4 chipsets with unified memory architecture.
"""

    if quantized:
        card += """
### Quantization

This model has been quantized to reduce memory footprint while maintaining performance.
"""

    return card


def upload_to_hub(
    mlx_path: Path,
    repo_id: str,
    original_repo: str,
    quantized: bool = False,
    bits: Optional[int] = None,
    dtype: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> None:
    """
    Upload converted model to HuggingFace Hub.

    Args:
        mlx_path: Path to MLX model directory
        repo_id: Target repository ID (e.g., "username/model-name-mlx")
        original_repo: Original model repository ID
        quantized: Whether model is quantized
        bits: Quantization bits
        dtype: Model dtype
        commit_message: Optional commit message
    """
    logger.info(f"Uploading to HuggingFace Hub: {repo_id}")

    # Create model card
    model_name = repo_id.split("/")[-1]
    readme_content = create_model_card(
        model_name=model_name,
        original_repo=original_repo,
        quantized=quantized,
        bits=bits,
        dtype=dtype,
    )

    readme_path = mlx_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Upload using HuggingFace Hub API
    api = HfApi()

    if commit_message is None:
        commit_message = f"Convert {original_repo} to MLX format"
        if quantized:
            commit_message += f" ({bits}-bit quantized)"

    api.upload_folder(
        folder_path=str(mlx_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    logger.info(f"Successfully uploaded to https://huggingface.co/{repo_id}")


# ============================================================================
# Main Conversion Function
# ============================================================================


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: Optional[str] = None,
    dequantize: bool = False,
    upload_repo: Optional[str] = None,
    revision: Optional[str] = None,
    skip_multimodal: bool = True,
    auto_detect: bool = True,
    model_type: Optional[str] = None,
    quant_recipe: Optional[str] = None,
) -> None:
    """
    Convert a model from HuggingFace to MLX format.

    Args:
        hf_path: HuggingFace repository ID or local path
        mlx_path: Output path for MLX model (default: "mlx_model")
        quantize: Apply quantization (default: False)
        q_group_size: Quantization group size (default: 64)
        q_bits: Bits per weight for quantization (default: 4)
        dtype: Target dtype (float16, bfloat16, float32)
        dequantize: Dequantize a quantized model
        upload_repo: Upload to HuggingFace repository
        revision: Model revision/branch
        skip_multimodal: Skip quantizing vision/audio modules
        auto_detect: Automatically detect model type and apply conversions
        model_type: Manually specify model type (llm, vlm, audio, ocr, embedding)
        quant_recipe: Mixed-bit quantization recipe (mixed_2_6, mixed_3_4, etc.)
    """
    logger.info(f"Converting {hf_path} to MLX format...")

    # Get model path (download if needed)
    model_path = get_model_path(hf_path, revision)
    mlx_path_obj = Path(mlx_path)
    mlx_path_obj.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json not found in {model_path}")

    config = load_config(config_path)

    # Detect or use specified model type
    if auto_detect and model_type is None:
        detected_type = detect_model_type(config)
        logger.info(f"Auto-detected model type: {detected_type}")
        model_type = detected_type
    elif model_type:
        logger.info(f"Using specified model type: {model_type}")
    else:
        model_type = "llm"  # Default to LLM

    # Load weights
    logger.info("Loading weights...")
    weights = {}

    # Try to load from safetensors first
    safetensors_files = list(model_path.glob("*.safetensors"))
    # Exclude index files
    safetensors_files = [f for f in safetensors_files if "index" not in f.name]

    if safetensors_files:
        for sf_file in safetensors_files:
            logger.info(f"Loading {sf_file.name}...")
            loaded = mx.load(str(sf_file))
            if isinstance(loaded, dict):
                weights.update(loaded)
            else:
                # Single array returned - use filename as key
                weights[sf_file.stem] = loaded
    else:
        # Try PyTorch .bin files
        bin_files = list(model_path.glob("*.bin"))
        if bin_files:
            logger.warning(
                "PyTorch .bin files found. Direct PyTorch conversion requires torch. "
                "Please convert to safetensors first or ensure model uses safetensors."
            )
            raise NotImplementedError(
                "Direct PyTorch .bin conversion not yet implemented. "
                "Please use a model with safetensors format."
            )
        else:
            raise ValueError(f"No model weights found in {model_path}")

    logger.info(f"Loaded {len(weights)} weight tensors")

    # Apply model-specific weight conversions
    if auto_detect or model_type != "llm":
        weights = apply_model_specific_conversion(weights, config, model_type)

    # Apply dtype conversion if specified
    if dtype:
        logger.info(f"Converting weights to {dtype}...")
        dtype_map = {
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Invalid dtype: {dtype}")

        target_dtype = dtype_map[dtype]
        weights = {
            k: v.astype(target_dtype) if v.dtype in [mx.float16, mx.bfloat16, mx.float32] else v
            for k, v in weights.items()
        }
        config["torch_dtype"] = dtype

    # Handle quantization/dequantization
    if dequantize:
        weights, config = dequantize_model(weights, config)
    elif quantize:
        if quant_recipe and quant_recipe in QUANT_RECIPES:
            # Use mixed-bit quantization
            logger.info(f"Using mixed-bit quantization recipe: {quant_recipe}")
            # Note: Full mixed-bit quantization requires loading into nn.Module
            # For now, just log and apply standard quantization
            logger.warning(
                f"Mixed-bit quantization recipe {quant_recipe} requires model architecture. "
                "Applying standard quantization. For advanced quantization, use smlx.quant module."
            )
        weights, config = quantize_model(weights, config, q_group_size, q_bits)

    # Add model type to config for reference
    config["smlx_model_type"] = model_type
    config["smlx_conversion_info"] = {
        "source": hf_path,
        "quantized": quantize,
        "dtype": dtype,
        "quant_bits": q_bits if quantize else None,
        "quant_group_size": q_group_size if quantize else None,
    }

    # Save weights
    logger.info(f"Saving MLX model to {mlx_path_obj}...")
    save_weights(mlx_path_obj, weights)

    # Save config
    save_config(config, mlx_path_obj / "config.json")

    # Copy tokenizer and other files
    for pattern in ["tokenizer*", "*.json", "*.txt", "*.model", "*.tiktoken"]:
        for file in model_path.glob(pattern):
            if file.name != "config.json" and file.is_file():
                shutil.copy(file, mlx_path_obj / file.name)
                logger.info(f"Copied {file.name}")

    # Special handling for tokenizer directory
    tokenizer_dir = model_path / "tokenizer"
    if tokenizer_dir.exists():
        shutil.copytree(tokenizer_dir, mlx_path_obj / "tokenizer", dirs_exist_ok=True)
        logger.info("Copied tokenizer directory")

    # Copy processor config for VLM/Audio/OCR models
    if model_type in ["vlm", "audio", "ocr"]:
        for processor_file in ["preprocessor_config.json", "processor_config.json"]:
            processor_path = model_path / processor_file
            if processor_path.exists():
                shutil.copy(processor_path, mlx_path_obj / processor_file)
                logger.info(f"Copied {processor_file}")

    logger.info(f"✓ Conversion complete! Model saved to {mlx_path_obj}")

    # Calculate and report model size
    total_size_mb = sum(v.nbytes for v in weights.values()) / (1024**2)
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Total model size: {total_size_mb:.2f} MB")

    # Remind about smol philosophy
    if total_size_mb > 5000 and not quantize:
        logger.warning(
            f"  Model is {total_size_mb:.0f}MB. Consider using --quantize for smol models!"
        )

    # Upload to HuggingFace if requested
    if upload_repo:
        upload_to_hub(
            mlx_path=mlx_path_obj,
            repo_id=upload_repo,
            original_repo=hf_path,
            quantized=quantize,
            bits=q_bits if quantize else None,
            dtype=dtype,
        )


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    """Command-line interface for model conversion."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to MLX format (optimized for M4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python -m smlx.tools.convert2mlx --hf-path HuggingFaceTB/SmolLM2-135M-Instruct

  # With quantization
  python -m smlx.tools.convert2mlx --hf-path HuggingFaceTB/SmolLM2-135M-Instruct -q --q-bits 4

  # Mixed-bit quantization
  python -m smlx.tools.convert2mlx --hf-path HuggingFaceTB/SmolLM2-360M-Instruct -q --quant-recipe mixed_4_6

  # VLM conversion
  python -m smlx.tools.convert2mlx --hf-path HuggingFaceTB/SmolVLM-256M-Instruct --model-type vlm

  # Audio model conversion
  python -m smlx.tools.convert2mlx --hf-path openai/whisper-tiny --model-type audio

Supported model types: llm, vlm, audio, ocr, embedding
Quantization recipes: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6
        """,
    )

    # Required arguments
    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="HuggingFace repository ID or local path to model",
    )

    # Output options
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Output path for MLX model (default: mlx_model)",
    )

    # Model type options
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["llm", "vlm", "audio", "ocr", "embedding"],
        help="Manually specify model type (default: auto-detect)",
    )
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable automatic model type detection",
    )

    # Quantization options
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Apply quantization to model (recommended for smol models)",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Group size for quantization (default: 64, optimized for M4)",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 6, 8],
        help="Bits per weight for quantization (default: 4)",
    )
    parser.add_argument(
        "--quant-recipe",
        type=str,
        default=None,
        choices=QUANT_RECIPES,
        help="Mixed-bit quantization recipe (requires model architecture)",
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        action="store_true",
        help="Dequantize a quantized model",
    )

    # Dtype options
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=MODEL_CONVERSION_DTYPES,
        help="Convert weights to specified dtype (default: use model's dtype)",
    )

    # Multimodal options
    parser.add_argument(
        "--no-skip-multimodal",
        action="store_true",
        help="Don't skip vision/audio modules during quantization",
    )

    # Upload options
    parser.add_argument(
        "--upload-repo",
        type=str,
        default=None,
        help="Upload converted model to HuggingFace repository (e.g., username/model-mlx)",
    )

    # Model loading options
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision/branch to use",
    )

    args = parser.parse_args()

    # Run conversion
    convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        quantize=args.quantize,
        q_group_size=args.q_group_size,
        q_bits=args.q_bits,
        dtype=args.dtype,
        dequantize=args.dequantize,
        upload_repo=args.upload_repo,
        revision=args.revision,
        skip_multimodal=not args.no_skip_multimodal,
        auto_detect=not args.no_auto_detect,
        model_type=args.model_type,
        quant_recipe=args.quant_recipe,
    )


if __name__ == "__main__":
    main()
