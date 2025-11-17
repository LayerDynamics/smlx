#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Weight conversion utilities for Chatterbox TTS model.

Converts pre-trained TTS model weights from PyTorch/HuggingFace to MLX format.
Supports:
- Kokoro-82M (82M parameters, compact TTS)
- Spark-TTS-0.5B (500M parameters, exact size match)
- Generic HuggingFace TTS models
"""

import re
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import numpy as np


def transpose_conv1d_weight(weight: np.ndarray) -> np.ndarray:
    """
    Transpose Conv1d weight from PyTorch (O, I, K) to MLX (O, K, I) format.

    Args:
        weight: Weight array from PyTorch

    Returns:
        Transposed weight for MLX
    """
    if weight.ndim == 3:
        # Conv1d: (out_channels, in_channels, kernel_size) -> (out_channels, kernel_size, in_channels)
        return np.transpose(weight, (0, 2, 1))
    return weight


def convert_kokoro_to_chatterbox(
    kokoro_weights: Dict[str, any],
    chatterbox_config: any = None,
) -> Dict[str, mx.array]:
    """
    Convert Kokoro-82M weights to Chatterbox format.

    Kokoro Architecture:
    - 82M parameters (compact)
    - Llama-based backbone
    - Acoustic head for mel generation

    Args:
        kokoro_weights: Kokoro checkpoint weights
        chatterbox_config: Optional Chatterbox configuration

    Returns:
        Dictionary of MLX weights for Chatterbox
    """
    mlx_weights = {}

    print("Converting Kokoro-82M weights to Chatterbox format...")

    # Weight mapping: Kokoro layer names -> Chatterbox layer names
    layer_mappings = {
        # Llama backbone embeddings
        r"model\.embed_tokens\.weight": "llama_backbone.embedding.weight",

        # Llama backbone layers
        r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight": r"llama_backbone.layers.\1.self_attn.q_proj.weight",
        r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight": r"llama_backbone.layers.\1.self_attn.k_proj.weight",
        r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight": r"llama_backbone.layers.\1.self_attn.v_proj.weight",
        r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight": r"llama_backbone.layers.\1.self_attn.o_proj.weight",

        # MLP layers
        r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight": r"llama_backbone.layers.\1.mlp.gate_proj.weight",
        r"model\.layers\.(\d+)\.mlp\.up_proj\.weight": r"llama_backbone.layers.\1.mlp.up_proj.weight",
        r"model\.layers\.(\d+)\.mlp\.down_proj\.weight": r"llama_backbone.layers.\1.mlp.down_proj.weight",

        # Layer norms
        r"model\.layers\.(\d+)\.input_layernorm\.weight": r"llama_backbone.layers.\1.input_layernorm.weight",
        r"model\.layers\.(\d+)\.post_attention_layernorm\.weight": r"llama_backbone.layers.\1.post_attention_layernorm.weight",

        # Final norm
        r"model\.norm\.weight": "llama_backbone.norm.weight",

        # Acoustic head (if present)
        r"acoustic_head\.weight": "acoustic_head.weight",
        r"acoustic_head\.bias": "acoustic_head.bias",
    }

    for kokoro_key, value in kokoro_weights.items():
        # Convert to numpy
        if hasattr(value, 'cpu'):
            # PyTorch tensor
            numpy_value = value.cpu().numpy()
        else:
            numpy_value = np.array(value)

        # Apply weight mapping
        chatterbox_key = None
        for pattern, replacement in layer_mappings.items():
            if re.match(pattern, kokoro_key):
                chatterbox_key = re.sub(pattern, replacement, kokoro_key)
                break

        if chatterbox_key is None:
            # Skip unmapped keys
            print(f"  Skipping unmapped key: {kokoro_key}")
            continue

        # Transpose Conv1d weights if needed
        if "conv" in chatterbox_key.lower() and numpy_value.ndim == 3:
            numpy_value = transpose_conv1d_weight(numpy_value)

        # Convert to MLX
        mlx_weights[chatterbox_key] = mx.array(numpy_value)

    print(f"✓ Converted {len(mlx_weights)} weight tensors from Kokoro to Chatterbox")
    return mlx_weights


def convert_spark_to_chatterbox(
    spark_weights: Dict[str, any],
    chatterbox_config: any = None,
) -> Dict[str, mx.array]:
    """
    Convert Spark-TTS-0.5B weights to Chatterbox format.

    Spark Architecture:
    - 500M parameters (exact size match with Chatterbox)
    - Transformer-based TTS
    - Voice cloning support

    Args:
        spark_weights: Spark checkpoint weights
        chatterbox_config: Optional Chatterbox configuration

    Returns:
        Dictionary of MLX weights for Chatterbox
    """
    mlx_weights = {}

    print("Converting Spark-TTS-0.5B weights to Chatterbox format...")

    # Weight mapping: Spark layer names -> Chatterbox layer names
    layer_mappings = {
        # Text encoder / Llama backbone
        r"text_encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight": r"llama_backbone.layers.\1.self_attn.q_proj.weight",
        r"text_encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight": r"llama_backbone.layers.\1.self_attn.k_proj.weight",
        r"text_encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight": r"llama_backbone.layers.\1.self_attn.v_proj.weight",
        r"text_encoder\.layers\.(\d+)\.self_attn\.o_proj\.weight": r"llama_backbone.layers.\1.self_attn.o_proj.weight",

        # MLP layers
        r"text_encoder\.layers\.(\d+)\.mlp\.fc1\.weight": r"llama_backbone.layers.\1.mlp.gate_proj.weight",
        r"text_encoder\.layers\.(\d+)\.mlp\.fc2\.weight": r"llama_backbone.layers.\1.mlp.down_proj.weight",

        # Voice encoder (if present)
        r"voice_encoder\.conv\.(\d+)\.weight": r"voice_encoder.conv_layers.\1.weight",
        r"voice_encoder\.conv\.(\d+)\.bias": r"voice_encoder.conv_layers.\1.bias",

        # Acoustic head
        r"acoustic_head\.linear\.weight": "acoustic_head.weight",
        r"acoustic_head\.linear\.bias": "acoustic_head.bias",
    }

    for spark_key, value in spark_weights.items():
        # Convert to numpy
        if hasattr(value, 'cpu'):
            # PyTorch tensor
            numpy_value = value.cpu().numpy()
        else:
            numpy_value = np.array(value)

        # Apply weight mapping
        chatterbox_key = None
        for pattern, replacement in layer_mappings.items():
            if re.match(pattern, spark_key):
                chatterbox_key = re.sub(pattern, replacement, spark_key)
                break

        if chatterbox_key is None:
            # Skip unmapped keys
            print(f"  Skipping unmapped key: {spark_key}")
            continue

        # Transpose Conv1d weights if needed
        if "conv" in chatterbox_key.lower() and numpy_value.ndim == 3:
            numpy_value = transpose_conv1d_weight(numpy_value)

        # Convert to MLX
        mlx_weights[chatterbox_key] = mx.array(numpy_value)

    print(f"✓ Converted {len(mlx_weights)} weight tensors from Spark to Chatterbox")
    return mlx_weights


def convert_hf_tts_to_chatterbox(
    checkpoint_path: Path,
    model_type: str = "auto",
    chatterbox_config: any = None,
) -> Dict[str, mx.array]:
    """
    Convert HuggingFace TTS model weights to Chatterbox format.

    Automatically detects model type or uses specified type.

    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Model type ("kokoro", "spark", or "auto")
        chatterbox_config: Optional Chatterbox configuration

    Returns:
        Dictionary of MLX weights for Chatterbox

    Raises:
        ValueError: If model type cannot be determined
    """
    # Load checkpoint
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except ImportError:
        raise ImportError("PyTorch is required for weight conversion. Install with: pip install torch")

    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            weights = checkpoint['state_dict']
        elif 'model' in checkpoint:
            weights = checkpoint['model']
        else:
            weights = checkpoint
    else:
        weights = checkpoint

    # Auto-detect model type if needed
    if model_type == "auto":
        # Check for Kokoro-specific keys
        if any("model.embed_tokens" in key for key in weights.keys()):
            model_type = "kokoro"
            print("Detected Kokoro-82M model")
        # Check for Spark-specific keys
        elif any("text_encoder" in key for key in weights.keys()):
            model_type = "spark"
            print("Detected Spark-TTS-0.5B model")
        else:
            raise ValueError(
                "Could not auto-detect model type. Please specify model_type='kokoro' or 'spark'"
            )

    # Convert based on model type
    if model_type == "kokoro":
        return convert_kokoro_to_chatterbox(weights, chatterbox_config)
    elif model_type == "spark":
        return convert_spark_to_chatterbox(weights, chatterbox_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def save_mlx_weights(weights: Dict[str, mx.array], output_path: Path):
    """
    Save MLX weights to file.

    Args:
        weights: Dictionary of MLX weights
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as npz file (MLX compatible)
    mx.savez(str(output_path), **weights)

    print(f"✓ Saved {len(weights)} weight tensors to {output_path}")


def load_mlx_weights(checkpoint_path: Path) -> Dict[str, mx.array]:
    """
    Load MLX weights from file.

    Args:
        checkpoint_path: Path to MLX checkpoint

    Returns:
        Dictionary of MLX weights
    """
    weights_dict = mx.load(str(checkpoint_path))
    print(f"✓ Loaded {len(weights_dict)} weight tensors from {checkpoint_path}")
    return weights_dict


# CLI interface for weight conversion
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert TTS model weights to Chatterbox format")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "kokoro", "spark"],
        help="Model type (auto-detect by default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chatterbox_weights.npz",
        help="Output path for converted weights",
    )

    args = parser.parse_args()

    # Convert weights
    print(f"\nConverting {args.checkpoint} to Chatterbox format...")
    weights = convert_hf_tts_to_chatterbox(
        Path(args.checkpoint),
        model_type=args.model_type,
    )

    # Save converted weights
    save_mlx_weights(weights, Path(args.output))

    print(f"\n✓ Conversion complete!")
    print(f"  Input: {args.checkpoint}")
    print(f"  Output: {args.output}")
    print(f"  Weights: {len(weights)} tensors")
