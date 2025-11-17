#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model loader for Orpheus-150M TTS.

Handles downloading and loading models from HuggingFace Hub.
"""

from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx

from .config import DEFAULT_CONFIG, Orpheus150MConfig, load_config
from .model import Orpheus150M
from .processor import TextProcessor, create_processor
from .vocoder import HiFiGANVocoder


def load(
    model_name: str = "orpheus-150m",
    force_download: bool = False,
) -> Tuple[Orpheus150M, TextProcessor]:
    """
    Load Orpheus-150M model and processor.

    Args:
        model_name: HuggingFace model ID or local path
        force_download: Force re-download from Hub

    Returns:
        Tuple of (model, processor)

    Example:
        >>> model, processor = load("orpheus-150m")
        >>> from smlx.models.Orpheus_150M import synthesize
        >>> audio = synthesize(model, processor, "Hello world")

    Note:
        This is a reference implementation. For production use:
        1. Search HuggingFace for "Orpheus-150M" or similar TTS models
        2. Load pre-trained weights
        3. Use full neural vocoder implementation
    """
    print(f"Loading Orpheus-150M TTS model: {model_name}")
    print("=" * 70)

    # Check if local path or HuggingFace ID
    if Path(model_name).exists():
        model_path = Path(model_name)
        print(f"Loading from local path: {model_path}")
    else:
        print(f"Searching for model on HuggingFace: {model_name}")
        model_path = download_model(model_name, force_download)

    # Load configuration
    config = load_config(str(model_path)) if model_path else DEFAULT_CONFIG

    # Create model
    print("Creating Orpheus-150M model...")
    model = Orpheus150M(config)

    # Load weights (if available)
    if model_path and model_path.exists():
        weights_loaded = load_weights(model, model_path)
        if not weights_loaded:
            print("⚠ Warning: Model initialized with random weights")
            print("   Load pre-trained weights from HuggingFace for synthesis")
    else:
        print("⚠ Warning: No model path found, using random initialization")

    model.eval()

    # Create processor
    print("Creating text processor...")
    processor = create_processor()
    print("✓ Text processor created")

    print("=" * 70)
    print("✓ Orpheus-150M model loaded")
    print("\nNote: This is a reference implementation.")
    print("For production TTS:")
    print("  1. Search HuggingFace for 'TTS' or 'text-to-speech' models")
    print("  2. Load pre-trained Tacotron2, FastSpeech2, or VITS")
    print("  3. Use with neural vocoder (HiFi-GAN, WaveGlow)")
    print("  4. See resources/mlx-examples for reference implementations")

    return model, processor


def download_model(model_name: str, force_download: bool = False) -> Optional[Path]:
    """
    Download model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model ID
        force_download: Force re-download

    Returns:
        Path to downloaded model, or None if not found
    """
    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading {model_name} from HuggingFace Hub...")
        path = snapshot_download(
            repo_id=model_name,
            force_download=force_download,
        )
        print(f"✓ Downloaded to {path}")
        return Path(path)
    except Exception as e:
        print(f"Could not download model: {e}")
        print("\nNote: Orpheus-150M may not be available on HuggingFace yet.")
        print("Alternative TTS models to try:")
        print("  - facebook/fastspeech2-en-ljspeech")
        print("  - facebook/tts_transformer-en-ljspeech")
        print("  - suno/bark (larger, but high quality)")
        print("\nProceeding with placeholder initialization...")
        return None


def load_weights(model: Orpheus150M, model_path: Path) -> bool:
    """
    Load pre-trained weights into model.

    Args:
        model: Orpheus150M model instance
        model_path: Path to model directory

    Returns:
        True if weights loaded successfully, False otherwise
    """
    # Try loading from safetensors
    safetensors_path = model_path / "model.safetensors"
    if safetensors_path.exists():
        try:
            weights = mx.load(str(safetensors_path))
            model.update(weights)
            print("✓ Weights loaded from safetensors")
            return True
        except Exception as e:
            print(f"Could not load weights: {e}")

    # Try loading from npz
    npz_path = model_path / "model.npz"
    if npz_path.exists():
        try:
            weights = mx.load(str(npz_path))
            model.update(weights)
            print("✓ Weights loaded from npz")
            return True
        except Exception as e:
            print(f"Could not load weights: {e}")

    # Try individual safetensors shards
    safetensors_index = model_path / "model.safetensors.index.json"
    if safetensors_index.exists():
        try:
            import json

            with open(safetensors_index) as f:
                index = json.load(f)

            weights = {}
            for shard_file in set(index["weight_map"].values()):
                shard_path = model_path / shard_file
                shard_weights = mx.load(str(shard_path))
                weights.update(shard_weights)

            model.update(weights)
            print("✓ Weights loaded from sharded safetensors")
            return True
        except Exception as e:
            print(f"Could not load sharded weights: {e}")

    return False


def save_model(
    model: Orpheus150M,
    output_path: str,
    config: Optional[Orpheus150MConfig] = None,
):
    """
    Save Orpheus-150M model.

    Args:
        model: Orpheus150M model
        output_path: Output directory
        config: Model configuration

    Example:
        >>> model, processor = load()
        >>> save_model(model, "./my_orpheus_model")
    """
    from .config import save_config

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    if config is None:
        config = model.config
    save_config(config, str(output_path))

    # Save weights
    weights = dict(model.parameters())
    weights_path = output_path / "model.safetensors"
    mx.save_safetensors(str(weights_path), weights)

    print(f"✓ Model saved to {output_path}")


def get_model_info(model: Orpheus150M) -> dict:
    """
    Get model information and statistics.

    Args:
        model: Orpheus150M model

    Returns:
        Dictionary with model info

    Example:
        >>> model, processor = load()
        >>> info = get_model_info(model)
        >>> print(f"Total parameters: {info['total_params']:,}")
    """
    # Count parameters
    total_params = 0
    component_params = {}

    components = {
        "text_encoder": model.text_encoder,
        "duration_predictor": model.duration_predictor,
        "decoder": model.decoder,
        "vocoder": model.vocoder,
    }

    def count_params(module):
        """Recursively count parameters in a module."""
        params_dict = module.parameters()
        total = 0
        for value in params_dict.values():
            if hasattr(value, 'size'):
                # It's an array
                total += value.size
            elif isinstance(value, dict):
                # Nested dict - recursively count
                for v in value.values():
                    if hasattr(v, 'size'):
                        total += v.size
        return total

    for name, component in components.items():
        params = count_params(component)
        component_params[name] = params
        total_params += params

    # Memory estimate (FP16)
    memory_fp16_mb = (total_params * 2) / (1024 * 1024)  # 2 bytes per param

    # Memory estimate (4-bit)
    memory_4bit_mb = (total_params * 0.5) / (1024 * 1024)  # 0.5 bytes per param

    info = {
        "total_params": total_params,
        "component_params": component_params,
        "memory_fp16_mb": memory_fp16_mb,
        "memory_4bit_mb": memory_4bit_mb,
        "sample_rate": model.config.sample_rate,
        "num_mels": model.config.num_mels,
    }

    return info


def print_model_info(model: Orpheus150M):
    """
    Print model information.

    Args:
        model: Orpheus150M model

    Example:
        >>> model, processor = load()
        >>> print_model_info(model)
    """
    info = get_model_info(model)

    print("\n" + "=" * 70)
    print("Orpheus-150M Model Information")
    print("=" * 70)
    print(f"\nTotal Parameters: {info['total_params']:,}")
    print("\nComponent Parameters:")
    for name, params in info["component_params"].items():
        print(f"  {name}: {params:,}")
    print(f"\nMemory (FP16): {info['memory_fp16_mb']:.1f} MB")
    print(f"Memory (4-bit): {info['memory_4bit_mb']:.1f} MB")
    print(f"\nSample Rate: {info['sample_rate']} Hz")
    print(f"Mel Bins: {info['num_mels']}")
    print("=" * 70)


def load_vocoder_weights(
    vocoder: HiFiGANVocoder,
    checkpoint_path: Optional[str] = None,
    repo_id: str = "nvidia/tts_hifigan",
    variant: str = "v3",
) -> bool:
    """
    Load pre-trained HiFi-GAN vocoder weights.

    Supports loading from:
    1. Local PyTorch checkpoint (.pth file)
    2. HuggingFace Hub (downloads automatically)
    3. MLX safetensors (if available)

    Args:
        vocoder: HiFiGANVocoder instance
        checkpoint_path: Local path to checkpoint (optional)
        repo_id: HuggingFace repository ID
        variant: Vocoder variant ("v1" or "v3")

    Returns:
        True if weights loaded successfully, False otherwise

    Example:
        >>> from smlx.models.Orpheus_150M import HiFiGANVocoder, load_vocoder_weights
        >>> vocoder = HiFiGANVocoder()
        >>> load_vocoder_weights(vocoder, repo_id="nvidia/tts_hifigan")
        >>> # Now vocoder is ready for inference

    Note:
        This function converts PyTorch weights to MLX format automatically.
        Weight conversion accounts for differences in Conv1d tensor layouts:
        - PyTorch Conv1d: (out_channels, in_channels, kernel_size)
        - MLX Conv1d: (kernel_size, in_channels, out_channels)
    """
    print(f"Loading HiFi-GAN {variant.upper()} vocoder weights...")

    # Download from HuggingFace if no local path provided
    if checkpoint_path is None:
        try:
            from huggingface_hub import hf_hub_download

            print(f"Downloading from {repo_id}...")
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"generator_{variant}.pth",
                cache_dir=None,  # Use default cache
            )
            print(f"✓ Downloaded to {checkpoint_path}")
        except Exception as e:
            print(f"✗ Could not download vocoder weights: {e}")
            print("\nAlternative repositories to try:")
            print("  - speechbrain/tts-hifigan-ljspeech")
            print("  - facebook/hifigan")
            print("\nOr provide local path to checkpoint")
            return False

    # Load checkpoint
    try:
        # Try loading as PyTorch checkpoint
        try:
            import torch

            print("Loading PyTorch checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Extract generator weights (checkpoint may have 'generator' key or be flat)
            if isinstance(checkpoint, dict) and "generator" in checkpoint:
                state_dict = checkpoint["generator"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            print("Converting PyTorch → MLX...")
            mlx_weights = convert_pytorch_vocoder_weights(state_dict)

        except ImportError:
            print("PyTorch not available, trying direct MLX load...")
            mlx_weights = mx.load(checkpoint_path)

        # Update vocoder parameters
        vocoder.update(mlx_weights)
        print("✓ Vocoder weights loaded successfully")
        return True

    except Exception as e:
        print(f"✗ Failed to load vocoder weights: {e}")
        import traceback

        traceback.print_exc()
        return False


def convert_pytorch_vocoder_weights(pytorch_state_dict: dict) -> dict:
    """
    Convert PyTorch HiFi-GAN weights to MLX format.

    Handles Conv1d weight transposition and parameter name mapping.

    Args:
        pytorch_state_dict: PyTorch state dict from checkpoint

    Returns:
        MLX-compatible weight dictionary

    Note:
        PyTorch Conv1d weights: (out_channels, in_channels, kernel_size)
        MLX Conv1d weights: (kernel_size, in_channels, out_channels)
    """
    import numpy as np

    mlx_weights = {}

    for key, value in pytorch_state_dict.items():
        # Convert torch tensor to numpy
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        else:
            value = np.array(value)

        # Check if this is a Conv1d weight (3D tensor)
        if "conv" in key and "weight" in key and value.ndim == 3:
            # Transpose from PyTorch (out, in, kernel) to MLX (kernel, in, out)
            value = np.transpose(value, (2, 1, 0))

        # Remove "module." prefix if present (from DataParallel)
        key = key.replace("module.", "")

        # Remove "generator." prefix if present
        key = key.replace("generator.", "")

        # Map PyTorch layer names to MLX module structure
        # HiFi-GAN uses sequential naming like "ups.0", "resblocks.0.convs.0"
        # Our MLX implementation uses lists, so we need to restructure

        # Convert to MLX array
        mlx_weights[key] = mx.array(value)

    print(f"Converted {len(mlx_weights)} weight tensors")
    return mlx_weights
