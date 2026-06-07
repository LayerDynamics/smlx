#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model loading and saving for smolGenCad.

Provides utilities to load pre-trained models from HuggingFace Hub or local paths,
and save trained models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import SmolGenCadConfig
from .model import SmolGenCad
from .tokenizer import CAD_VOCAB_SIZE, CADTokenizer


def load(
    model_path: str | None = None,
    lazy: bool = False,
) -> tuple[SmolGenCad, Any, CADTokenizer]:
    """
    Load smolGenCad model and tokenizers.

    IMPORTANT: This is a reference implementation. Pre-trained weights are
    not yet available. This function will initialize a model with random
    weights for testing the architecture.

    Args:
        model_path: Path to model checkpoint (currently unused)
        lazy: If False, eagerly load all weights

    Returns:
        Tuple of (model, text_tokenizer, cad_tokenizer)

    Example:
        >>> from smlx.models.smolGenCad import load
        >>> model, text_tokenizer, cad_tokenizer = load()
        >>> print(f"Model has {model.num_params_millions:.1f}M parameters")

    Note:
        For production use with pre-trained weights:
        1. Train the model on CAD dataset (DeepCAD, Text2CAD)
        2. Save weights with save_model()
        3. Load from HuggingFace Hub or local path
    """
    # Create default configuration
    config = SmolGenCadConfig()

    # Initialize model with random weights
    model = SmolGenCad(config)

    # Load text tokenizer (SmolLM2 tokenizer)
    try:
        from transformers import AutoTokenizer

        text_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    except Exception as e:
        print(f"Warning: Could not load text tokenizer: {e}")
        print("Note: Text tokenizer is only needed for natural language input.")
        print("CAD tokenizer will still work for CAD sequences.")
        text_tokenizer = None

    # Create CAD tokenizer with 8-bit quantization (256 bins per parameter type)
    # Following Text2CAD (NeurIPS 2024) architecture
    cad_tokenizer = CADTokenizer(config.vocabulary)

    # Eagerly evaluate if requested
    if not lazy:
        mx.eval(model.parameters())

    # No public checkpoint exists yet: this path always uses random weights.
    # Expose the signal so callers (e.g. the unified runner) report output as
    # pipeline-only rather than trained-quality. load_from_pretrained() sets
    # this True when it actually loads a checkpoint.
    model.weights_loaded = False

    print(f"Loaded smolGenCad model with {model.num_params_millions:.1f}M parameters")
    print("⚠️  Model initialized with random weights (no pre-trained weights available yet)")
    print("   For training, see docs/ModelImplementations.md")

    return model, text_tokenizer, cad_tokenizer


def load_model_from_path(
    model_path: str | Path,
    lazy: bool = False,
) -> tuple[SmolGenCad, SmolGenCadConfig]:
    """
    Load model from local path.

    Args:
        model_path: Path to model directory
        lazy: If False, eagerly load all weights

    Returns:
        Tuple of (model, config)

    Example:
        >>> model, config = load_model_from_path("./checkpoints/smolGenCad-v1")
    """
    model_path = Path(model_path)

    # Load configuration
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)
    config = SmolGenCadConfig.from_dict(config_dict)

    # Initialize model
    model = SmolGenCad(config)

    # Load weights
    weights_path = model_path / "weights.safetensors"
    if not weights_path.exists():
        weights_path = model_path / "model.safetensors"

    if weights_path.exists():
        weights = mx.load(str(weights_path))
        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))
        model.weights_loaded = True
        print(f"Loaded weights from {weights_path}")
    else:
        model.weights_loaded = False
        print(f"Warning: No weights found at {weights_path}, using random initialization")

    # Eagerly evaluate if requested
    if not lazy:
        mx.eval(model.parameters())

    return model, config


def save_model(
    model: SmolGenCad,
    output_path: str | Path,
    save_tokenizer: bool = True,
):
    """
    Save model to disk.

    Args:
        model: SmolGenCad model to save
        output_path: Output directory path
        save_tokenizer: Whether to save tokenizer config

    Example:
        >>> save_model(model, "./checkpoints/smolGenCad-v1")
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(model.config.to_dict(), f, indent=2)
    print(f"Saved config to {config_path}")

    # Save weights
    weights = dict(model.parameters())
    weights_path = output_path / "weights.safetensors"
    mx.save_safetensors(str(weights_path), weights)
    print(f"Saved weights to {weights_path}")

    # Save tokenizer config
    if save_tokenizer:
        tokenizer_config_path = output_path / "tokenizer_config.json"
        tokenizer_config = {
            "vocab_size": CAD_VOCAB_SIZE,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "sep_token_id": 3,
        }
        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        print(f"Saved tokenizer config to {tokenizer_config_path}")

    print(f"Model saved successfully to {output_path}")


def get_model_info(model: SmolGenCad) -> dict[str, Any]:
    """
    Get model information.

    Args:
        model: SmolGenCad model

    Returns:
        Dictionary with model information

    Example:
        >>> info = get_model_info(model)
        >>> print(f"Parameters: {info['num_params_millions']}M")
    """
    return {
        "model_type": "smolGenCad",
        "total_params": model.num_params,
        "num_params_millions": model.num_params_millions,
        "encoder_params": 135_000_000,  # SmolLM2-135M
        "decoder_params": model.num_params - 135_000_000,
        "encoder_layers": model.config.encoder.num_hidden_layers,
        "decoder_layers": model.config.decoder.num_hidden_layers,
        "encoder_hidden_size": model.config.encoder.hidden_size,
        "decoder_hidden_size": model.config.decoder.hidden_size,
        "max_sequence_length": model.config.vocabulary.max_sequence_length,
        "vocab_size": CAD_VOCAB_SIZE,
    }


def print_model_info(model: SmolGenCad):
    """
    Print model information to console.

    Args:
        model: SmolGenCad model

    Example:
        >>> print_model_info(model)
    """
    info = get_model_info(model)

    print("\n" + "=" * 60)
    print("smolGenCad Model Information")
    print("=" * 60)
    print(f"Total Parameters:      {info['num_params_millions']:.1f}M")
    print(f"  - Encoder:           {info['encoder_params'] / 1e6:.1f}M")
    print(f"  - Decoder:           {info['decoder_params'] / 1e6:.1f}M")
    print("\nEncoder (SmolLM2-135M):")
    print(f"  - Layers:            {info['encoder_layers']}")
    print(f"  - Hidden Size:       {info['encoder_hidden_size']}")
    print("\nDecoder (Custom):")
    print(f"  - Layers:            {info['decoder_layers']}")
    print(f"  - Hidden Size:       {info['decoder_hidden_size']}")
    print("\nVocabulary:")
    print(f"  - Size:              {info['vocab_size']}")
    print(f"  - Max Sequence:      {info['max_sequence_length']}")
    print("=" * 60 + "\n")
