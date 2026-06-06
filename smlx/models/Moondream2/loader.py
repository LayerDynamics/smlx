#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model loading utilities for Moondream2.

Handles downloading from HuggingFace Hub and loading weights.
"""

import glob
import json
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from tokenizers import Tokenizer

from .config import DEFAULT_CONFIG_05B, DEFAULT_CONFIG_2B, ModelConfig
from .model import Moondream2


class TokenizerWrapper:
    """Wrapper to make tokenizers.Tokenizer compatible with our code."""

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        # Moondream-specific special token IDs from config
        self.bos_token_id = 0
        self.eos_token_id = 0  # Will use answer_id (3) as generation stop
        self.pad_token_id = 0
        self.answer_token_id = 3
        self.vocab_size = tokenizer.get_vocab_size()

    def __len__(self) -> int:
        """Return vocabulary size for len() compatibility."""
        return self.vocab_size

    def encode(
        self, text: str, add_special_tokens: bool = True, return_tensors: str = None
    ) -> Union[list[int], list[list[int]]]:
        """Encode text to token IDs."""
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        token_ids = encoding.ids

        if return_tensors == "np":
            import numpy as np

            return np.array([token_ids])
        return token_ids

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = False
    ) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        """Stub for compatibility - starmie tokenizer already has all needed tokens."""
        return 0  # No tokens added


def get_model_path(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """Ensure model is available locally.

    Downloads from HuggingFace Hub if not found locally.

    Args:
        path_or_hf_repo: Local path or HuggingFace repo ID
        revision: Git revision (branch, tag, or commit hash)
        force_download: Force re-download even if exists

    Returns:
        Path to model directory
    """
    model_path = Path(path_or_hf_repo)

    if not model_path.exists() or force_download:
        print(f"Downloading model from HuggingFace Hub: {path_or_hf_repo}")
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.model",
                    "*.tiktoken",
                    "*.py",  # Include Python files for custom models
                ],
                force_download=force_download,
            )
        )
        print(f"Model downloaded to: {model_path}")

    return model_path


def load_config(model_path: Path, variant: str = "2b") -> ModelConfig:
    """Load model configuration.

    Args:
        model_path: Path to model directory
        variant: Model variant ("2b" or "0.5b")

    Returns:
        ModelConfig instance

    Raises:
        FileNotFoundError: If config.json not found
        ValueError: If config cannot be parsed
    """
    config_path = model_path / "config.json"
    print(f"[CONFIG] Loading config from: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Model path: {model_path}\n"
            f"Expected config.json to exist in model directory."
        )

    with open(config_path) as f:
        config_dict = json.load(f)

    print(f"[CONFIG] Loaded config.json keys: {list(config_dict.keys())}")
    print(f"[CONFIG] Full config content: {json.dumps(config_dict, indent=2)}")

    # Check if this is a HuggingFace format with nested config
    if "config" in config_dict and isinstance(config_dict["config"], dict):
        print("[CONFIG] Detected HuggingFace nested config format")
        nested_config = config_dict["config"]
        print(f"[CONFIG] Nested config keys: {list(nested_config.keys())}")

        # The official moondream2 uses a custom config where config.json has an empty
        # "config" field and the actual values are in config.py (which we've mirrored
        # in our DEFAULT_CONFIG_2B/DEFAULT_CONFIG_05B)
        if not nested_config or len(nested_config) == 0:
            print("[CONFIG] Empty nested config - this is expected for moondream2")
            print("[CONFIG] Using SMLX default configuration (matches official config.py)")

            # The official moondream2 config.py should exist for reference
            config_py_path = model_path / "config.py"
            if config_py_path.exists():
                print(f"[CONFIG] Found official config.py at: {config_py_path}")
            else:
                print("[CONFIG] WARNING: config.py not found (optional reference file)")

            print(f"[CONFIG] Loading default {variant.upper()} configuration")
            if variant == "0.5b":
                config = DEFAULT_CONFIG_05B
            else:
                config = DEFAULT_CONFIG_2B
            print(f"[CONFIG] Config values: hidden_size={config.text_config.hidden_size}, "
                  f"num_layers={config.text_config.num_hidden_layers}, "
                  f"vision_layers={config.vision_config.num_hidden_layers}")
            return config

        # If nested config has values, try to parse it
        print("[CONFIG] Nested config has values, attempting to parse")
        try:
            config = ModelConfig.from_dict(nested_config)
            print("[CONFIG] Successfully parsed nested config")
            return config
        except Exception as e:
            raise ValueError(
                f"Failed to parse nested config\n"
                f"Error: {e}\n"
                f"Config content: {json.dumps(nested_config, indent=2)}"
            ) from e

    # Try to parse as SMLX ModelConfig format
    print("[CONFIG] Attempting to parse as SMLX ModelConfig format")
    try:
        config = ModelConfig.from_dict(config_dict)
        print("[CONFIG] Successfully parsed config")
        return config
    except Exception as e:
        raise ValueError(
            f"Failed to parse config from {config_path}\n"
            f"Error: {e}\n"
            f"Config content: {json.dumps(config_dict, indent=2)}\n"
            f"Expected either HuggingFace format or SMLX ModelConfig format."
        ) from e


def load_weights(model_path: Path) -> dict:
    """Load model weights from safetensors or npz files.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary of weights

    Raises:
        FileNotFoundError: If no weight files found
        ValueError: If weights cannot be loaded
    """
    print(f"[WEIGHTS] Searching for weight files in: {model_path}")

    # Try safetensors first (HuggingFace format)
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if weight_files:
        print(f"[WEIGHTS] Found {len(weight_files)} safetensors file(s)")

    if not weight_files:
        # Try npz (MLX format)
        weight_files = glob.glob(str(model_path / "*.npz"))
        if weight_files:
            print(f"[WEIGHTS] Found {len(weight_files)} npz file(s)")

    if not weight_files:
        # Try pytorch format
        weight_files = glob.glob(str(model_path / "pytorch_model*.bin"))
        if weight_files:
            print(f"[WEIGHTS] Found {len(weight_files)} pytorch .bin file(s)")

    if not weight_files:
        all_files = list(model_path.glob("*"))
        raise FileNotFoundError(
            f"No weight files found in {model_path}\n"
            f"Expected: .safetensors, .npz, or .bin files\n"
            f"Files in directory: {[f.name for f in all_files]}"
        )

    print(f"[WEIGHTS] Loading from {len(weight_files)} file(s):")
    for wf in weight_files:
        print(f"[WEIGHTS]   - {Path(wf).name}")

    weights = {}
    for wf in weight_files:
        print(f"[WEIGHTS] Loading: {Path(wf).name}")
        try:
            loaded = mx.load(wf)
        except Exception as e:
            raise ValueError(
                f"Failed to load weights from {wf}\n"
                f"Error: {e}\n"
                f"File may be corrupted or in unsupported format."
            ) from e

        if isinstance(loaded, dict):
            print(f"[WEIGHTS]   Loaded {len(loaded)} weight tensors from {Path(wf).name}")
            # Log first few keys as sample
            sample_keys = list(loaded.keys())[:5]
            print(f"[WEIGHTS]   Sample keys: {sample_keys}")
            weights.update(loaded)
        else:
            raise ValueError(
                f"Expected dict of weights from {wf}, got {type(loaded)}\n"
                f"Weight files should contain named parameters (dict format)."
            )

    print(f"[WEIGHTS] Total weight tensors loaded: {len(weights)}")
    print(f"[WEIGHTS] All weight keys ({len(weights)} total):")
    for i, key in enumerate(sorted(weights.keys())):
        if i < 20:  # Show first 20
            print(f"[WEIGHTS]   {i+1}. {key}: shape {weights[key].shape}")
        elif i == 20:
            print(f"[WEIGHTS]   ... ({len(weights) - 20} more weights)")
            break

    return weights


def load(
    path_or_hf_repo: str = "vikhyatk/moondream2",
    variant: str = "2b",
    revision: Optional[str] = None,
    lazy: bool = False,
    force_download: bool = False,
) -> tuple[Moondream2, AutoTokenizer]:
    """Load Moondream2 model and tokenizer.

    Args:
        path_or_hf_repo: HuggingFace repo ID or local path
        variant: Model variant ("2b" or "0.5b")
        revision: Git revision (branch, tag, or commit hash)
        lazy: If False, eagerly load all weights into memory
        force_download: Force re-download from HuggingFace Hub

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> from smlx.models.Moondream2 import load
        >>> model, tokenizer = load("vikhyatk/moondream2")

        >>> # Load smaller variant
        >>> model, tokenizer = load("vikhyatk/moondream-0_5b-int8", variant="0.5b")
    """
    # Map variant names to repos
    variant_map = {
        "2b": "vikhyatk/moondream2",
        "0.5b": "vikhyatk/moondream-0_5b-int8",
    }

    # Use variant default if path not specified
    if path_or_hf_repo is None:
        path_or_hf_repo = variant_map.get(variant, "vikhyatk/moondream2")

    # Ensure model is available locally
    model_path = get_model_path(path_or_hf_repo, revision, force_download)

    # Load configuration
    config = load_config(model_path, variant)

    # Initialize model
    print(f"Initializing Moondream2-{variant.upper()} model...")
    model = Moondream2(config)

    # Load weights
    weights = load_weights(model_path)

    # Sanitize weights (HuggingFace -> MLX conversion)
    print("[SANITIZE] Converting HuggingFace weights to MLX format...")
    print(f"[SANITIZE] Input: {len(weights)} weight tensors")
    try:
        sanitized_weights = Moondream2.sanitize(weights)
    except Exception as e:
        raise ValueError(
            f"Failed to sanitize weights during HF->MLX conversion\n"
            f"Error: {e}\n"
            f"Input weight keys: {list(weights.keys())[:10]}..."
        ) from e

    print(f"[SANITIZE] Output: {len(sanitized_weights)} weight tensors")
    print(f"[SANITIZE] Sanitized weight keys ({len(sanitized_weights)} total):")
    for i, key in enumerate(sorted(sanitized_weights.keys())):
        if i < 30:  # Show first 30 to see the mapping
            print(f"[SANITIZE]   {i+1}. {key}: shape {sanitized_weights[key].shape}")
        elif i == 30:
            print(f"[SANITIZE]   ... ({len(sanitized_weights) - 30} more weights)")
            break

    # Load weights into model
    # Use strict=False to allow missing weights (e.g., detection_head which has no pretrained weights)
    print("[LOAD] Loading sanitized weights into model...")
    print("[LOAD] Model expects these parameters:")
    # MLX uses different API - get parameters via tree_flatten
    try:
        model_params = model.parameters()
        if isinstance(model_params, dict):
            param_count = len(model_params)
            print(f"[LOAD]   Total model parameters: {param_count}")
            for i, (name, param) in enumerate(sorted(model_params.items())):
                if i < 30:
                    print(f"[LOAD]   {i+1}. {name}: shape {param.shape}")
                elif i == 30:
                    print(f"[LOAD]   ... ({param_count - 30} more parameters)")
                    break
        else:
            # If parameters() doesn't return dict, just count
            print("[LOAD]   (Model parameter listing not available)")
    except Exception as e:
        print(f"[LOAD]   WARNING: Could not list model parameters: {e}")

    try:
        unmatched = model.load_weights(list(sanitized_weights.items()), strict=False)
        print(f"[LOAD] Successfully loaded {len(sanitized_weights)} weights")
        if unmatched:
            print(f"[LOAD] WARNING: {len(unmatched)} unmatched weights:")
            for i, key in enumerate(unmatched):
                if i < 15:
                    print(f"[LOAD]   - {key}")
                elif i == 15:
                    print(f"[LOAD]   ... ({len(unmatched) - 15} more unmatched)")
                    break
        else:
            print("[LOAD] All weights matched successfully!")
    except Exception as e:
        raise ValueError(
            f"Failed to load weights into model\n"
            f"Error: {e}\n"
            f"This usually means weight keys don't match model parameter names."
        ) from e

    if not lazy:
        mx.eval(model.parameters())

    model.eval()

    # Load tokenizer
    # CRITICAL FIX: Load correct starmie-v1 tokenizer
    # The starmie-v1 tokenizer has the correct special token IDs:
    #   - Token 0 = BOS/EOS/PAD
    #   - Token 3 = <|answer|> (signals model to start generating)
    #   - Token 15381 = image marker in template [1, 15381, 2]
    print("Loading Moondream starmie-v1 tokenizer...")
    try:
        raw_tokenizer = Tokenizer.from_pretrained("moondream/starmie-v1")
        tokenizer = TokenizerWrapper(raw_tokenizer)
        print("✓ Loaded starmie-v1 tokenizer successfully")
    except Exception as e:
        print(f"ERROR: Could not load starmie-v1 tokenizer: {e}")
        print("This will cause gibberish output!")
        raise RuntimeError(
            "Moondream2 requires the starmie-v1 tokenizer. "
            "Please ensure you have internet connection to download it."
        ) from e

    print("✓ Model loaded successfully!")
    return model, tokenizer


__all__ = ["load", "get_model_path", "load_config", "load_weights"]
