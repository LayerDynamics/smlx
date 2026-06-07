#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
TrOCR Model Loading.

Handles downloading and loading TrOCR weights from HuggingFace Hub.
"""

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx

from .config import TrOCRConfig, create_config_from_dict
from .model import TrOCR
from .processor import TrOCRProcessor

# Short variant names -> canonical HuggingFace repo IDs. Used for both the
# weight download (get_model_path) and the tokenizer/processor construction in
# load(); an unknown name passes through unchanged (treated as a raw repo id).
_REPO_MAP = {
    "trocr-small-printed": "microsoft/trocr-small-printed",
    "trocr-small-handwritten": "microsoft/trocr-small-handwritten",
    "printed": "microsoft/trocr-small-printed",
    "handwritten": "microsoft/trocr-small-handwritten",
}


def get_model_path(
    model_name: str = "microsoft/trocr-small-printed",
    force_download: bool = False,
) -> Optional[Path]:
    """Get path to model files, downloading if necessary.

    Args:
        model_name: Model name or HuggingFace repo ID
        force_download: Force re-download even if cached

    Returns:
        Path to model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface-hub"
        ) from e

    repo_id = _REPO_MAP.get(model_name, model_name)

    # Download model
    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            force_download=force_download,
            allow_patterns=["*.json", "*.safetensors", "*.npz", "*.txt", "*.model"],
        )
        print(f"Downloaded TrOCR model to: {model_path}")
        return Path(model_path)
    except Exception as e:
        print(f"Warning: Could not download model: {e}")
        print("Creating model with random weights for testing.")
        return None


def load_config(model_path: Optional[Path] = None) -> TrOCRConfig:
    """Load model configuration.

    Args:
        model_path: Path to model directory

    Returns:
        TrOCR configuration
    """
    if model_path and (model_path / "config.json").exists():
        with open(model_path / "config.json") as f:
            config_dict = json.load(f)

        # Create config from HuggingFace format
        config = create_config_from_dict(config_dict)
        return config

    # Use default config
    from .config import DEFAULT_CONFIG_PRINTED

    return DEFAULT_CONFIG_PRINTED


def map_weight_names(weights: dict) -> dict:
    """Map HuggingFace weight names to our model's naming convention.

    Args:
        weights: Original weights from HuggingFace

    Returns:
        Mapped weights dictionary
    """
    mapped_weights = {}

    for key, value in weights.items():
        new_key = key

        # DeiT encoder uses CLS + distillation tokens (prepended to the patch
        # sequence). They MUST be loaded — skipping them (and leaving them random)
        # corrupts the encoder. Map them onto our encoder module.
        if "cls_token" in key:
            mapped_weights["encoder.cls_token"] = value
            continue
        if "distillation_token" in key:
            mapped_weights["encoder.distillation_token"] = value
            continue

        # Patch-embedding Conv2d: PyTorch (out, in, kH, kW) -> MLX (out, kH, kW, in).
        if "patch_embeddings.projection.weight" in key:
            mapped_weights["encoder.patch_embed.weight"] = mx.transpose(value, (0, 2, 3, 1))
            continue

        # Encoder mappings
        if key.startswith("encoder"):
            # Patch embeddings: encoder.embeddings.patch_embeddings.projection -> encoder.patch_embed
            new_key = new_key.replace(
                "encoder.embeddings.patch_embeddings.projection", "encoder.patch_embed"
            )

            # Position embeddings: encoder.embeddings.position_embeddings -> encoder.position_embeddings
            new_key = new_key.replace(
                "encoder.embeddings.position_embeddings", "encoder.position_embeddings"
            )

            # Layer mappings: encoder.encoder.layer.N -> encoder.layers.N
            new_key = new_key.replace("encoder.encoder.layer.", "encoder.layers.")

            # Attention projections
            new_key = new_key.replace(".attention.attention.query", ".attention.q_proj")
            new_key = new_key.replace(".attention.attention.key", ".attention.k_proj")
            new_key = new_key.replace(".attention.attention.value", ".attention.v_proj")
            new_key = new_key.replace(".attention.output.dense", ".attention.out_proj")

            # MLP layers: intermediate.dense -> mlp.0, output.dense -> mlp.2
            new_key = new_key.replace(".intermediate.dense", ".mlp.0")
            new_key = new_key.replace(".output.dense", ".mlp.2")

            # Layer norms
            new_key = new_key.replace(".layernorm_before", ".ln1")
            new_key = new_key.replace(".layernorm_after", ".ln2")
            new_key = new_key.replace("encoder.layernorm", "encoder.ln")

        # Decoder mappings
        elif key.startswith("decoder"):
            # decoder.model.decoder -> decoder
            new_key = new_key.replace("decoder.model.decoder.", "decoder.")

            # Token embeddings: embed_tokens -> token_embedding
            new_key = new_key.replace("decoder.embed_tokens", "decoder.token_embedding")

            # Position embeddings: embed_positions -> position_embedding
            new_key = new_key.replace("decoder.embed_positions", "decoder.position_embedding")

            # Self attention: self_attn -> self_attn (already matches)
            # Cross attention: encoder_attn -> cross_attn
            new_key = new_key.replace(".encoder_attn.", ".cross_attn.")

            # MLP layers: fc1 -> mlp.0, fc2 -> mlp.2
            new_key = new_key.replace(".fc1", ".mlp.0")
            new_key = new_key.replace(".fc2", ".mlp.2")

            # Layer norms
            new_key = new_key.replace(".self_attn_layer_norm", ".ln1")
            new_key = new_key.replace(".encoder_attn_layer_norm", ".ln2")
            new_key = new_key.replace(".final_layer_norm", ".ln3")
            new_key = new_key.replace("decoder.layernorm_embedding", "decoder.ln_embedding")

            # Final decoder layer norm (if exists)
            if new_key == "decoder.layernorm.weight" or new_key == "decoder.layernorm.bias":
                new_key = new_key.replace("decoder.layernorm", "decoder.ln")

            # Output projection: decoder.output_projection -> lm_head
            new_key = new_key.replace("decoder.output_projection", "lm_head")

        mapped_weights[new_key] = value

    return mapped_weights


def load_weights(model_path: Optional[Path] = None) -> Optional[dict]:
    """Load model weights.

    Args:
        model_path: Path to model directory

    Returns:
        Model weights dictionary or None
    """
    if not model_path:
        return None

    # Try loading safetensors
    safetensors_path = model_path / "model.safetensors"
    if safetensors_path.exists():
        try:
            weights_raw = mx.load(str(safetensors_path))
            # Ensure we have a dict (mx.load can return array or dict)
            if isinstance(weights_raw, dict):
                # Map weight names to match our architecture
                weights = map_weight_names(weights_raw)
                return weights
            else:
                print("Warning: Loaded weights are not in dictionary format")
        except Exception as e:
            print(f"Warning: Could not load safetensors: {e}")

    # Try loading NPZ format
    npz_path = model_path / "model.npz"
    if npz_path.exists():
        weights_raw = mx.load(str(npz_path))
        # Ensure we have a dict (mx.load can return array or dict)
        if isinstance(weights_raw, dict):
            # Map weight names to match our architecture
            weights = map_weight_names(weights_raw)
            return weights
        else:
            print("Warning: Loaded weights are not in dictionary format")

    return None


def load(
    model_name: str = "microsoft/trocr-small-printed",
    force_download: bool = False,
) -> tuple[TrOCR, TrOCRProcessor]:
    """Load TrOCR model and processor.

    Args:
        model_name: Model name or HuggingFace repo ID
            Options: "microsoft/trocr-small-printed",
                     "microsoft/trocr-small-handwritten",
                     "printed", "handwritten"
        force_download: Force re-download even if cached

    Returns:
        Tuple of (model, processor)

    Example:
        >>> from smlx.models.TrOCR_small import load
        >>> model, processor = load("printed")
        >>> # Use for OCR
    """
    # Get model path
    try:
        model_path = get_model_path(model_name, force_download)
    except Exception as e:
        print(f"Warning: Could not download model: {e}")
        print("Creating model with random weights for testing.")
        model_path = None

    # Load config
    config = load_config(model_path)

    # Set variant based on model name
    if "handwritten" in model_name.lower():
        config.variant = "handwritten"
    else:
        config.variant = "printed"

    # Create model
    model = TrOCR(config)

    # Load weights if available
    if model_path:
        weights = load_weights(model_path)
        if weights:
            # Load via load_weights (the API that actually applies flat dotted-key
            # weights — model.update() silently drops many of them). Filter to the
            # keys that exist as model parameters so an orphan/renamed key can't
            # abort the whole load; report any model params left uncovered.
            from mlx.utils import tree_flatten

            valid_keys = {k for k, _ in tree_flatten(model.parameters())}
            to_load = [(k, v) for k, v in weights.items() if k in valid_keys]
            model.load_weights(to_load, strict=False)

            missing = sorted(valid_keys - {k for k, _ in to_load})
            if missing:
                print(f"Warning: {len(missing)} model params not in checkpoint, e.g. {missing[:5]}")
            print(f"Loaded TrOCR weights from {model_path}")
        else:
            print("Warning: No weights found. Using random initialization.")
    else:
        print("Warning: Model not downloaded. Using random initialization.")

    # Set to eval mode
    model.eval()

    # Create processor with the *resolved* repo id so the tokenizer loads from
    # the real HF repo — a short variant name like "printed" is not a valid repo
    # id and would otherwise fail the (intentionally strict) tokenizer load.
    processor = TrOCRProcessor(config, model_name=_REPO_MAP.get(model_name, model_name))

    return model, processor


def save_model(
    model: TrOCR,
    save_path: Path,
    save_config: bool = True,
) -> None:
    """Save model weights and config.

    Args:
        model: TrOCR model to save
        save_path: Directory to save model
        save_config: Whether to save config file
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = save_path / "model.safetensors"
    # Convert model parameters to dict[str, array] explicitly
    weights: dict[str, mx.array] = dict(model.parameters())
    mx.save_safetensors(str(weights_path), weights)

    print(f"Saved model weights to {weights_path}")

    # Save config
    if save_config:
        config_path = save_path / "config.json"
        config_dict = {
            "model_type": "trocr",
            "variant": model.config.variant,
            "vision_config": {
                "hidden_size": model.config.vision_config.hidden_size,
                "num_hidden_layers": model.config.vision_config.num_hidden_layers,
                "num_attention_heads": model.config.vision_config.num_attention_heads,
                "intermediate_size": model.config.vision_config.intermediate_size,
                "image_size": model.config.vision_config.image_size,
                "patch_size": model.config.vision_config.patch_size,
            },
            "decoder_config": {
                "vocab_size": model.config.decoder_config.vocab_size,
                "hidden_size": model.config.decoder_config.hidden_size,
                "num_hidden_layers": model.config.decoder_config.num_hidden_layers,
                "num_attention_heads": model.config.decoder_config.num_attention_heads,
                "intermediate_size": model.config.decoder_config.intermediate_size,
                "max_position_embeddings": model.config.decoder_config.max_position_embeddings,
                "bos_token_id": model.config.decoder_config.bos_token_id,
                "eos_token_id": model.config.decoder_config.eos_token_id,
                "pad_token_id": model.config.decoder_config.pad_token_id,
            },
        }

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Saved model config to {config_path}")


__all__ = [
    "load",
    "save_model",
    "get_model_path",
]
