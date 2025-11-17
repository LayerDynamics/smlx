#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model loader for Chatterbox TTS.

Handles downloading and loading voice cloning TTS models.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict

import mlx.core as mx

from .config import DEFAULT_CONFIG, ChatterboxConfig, load_config
from .model import Chatterbox
from .processor import ChatterboxProcessor, create_processor
from .vocoder import HiFiGANGenerator


def load(
    model_name: str = "chatterbox-500m",
    force_download: bool = False,
) -> Tuple[Chatterbox, ChatterboxProcessor]:
    """
    Load Chatterbox model and processor.

    Args:
        model_name: HuggingFace model ID or local path
        force_download: Force re-download from Hub

    Returns:
        Tuple of (model, processor)

    Example:
        >>> model, processor = load()
        >>> from smlx.models.Chatterbox import synthesize, clone_voice
        >>>
        >>> # Clone voice
        >>> voice_emb = clone_voice(model, processor, reference_audio)
        >>>
        >>> # Synthesize with cloned voice
        >>> audio = synthesize(
        ...     model, processor,
        ...     "Hello world",
        ...     voice_embedding=voice_emb,
        ...     emotion="happy",
        ...     expressiveness=0.8
        ... )

    Note:
        This is a reference implementation. For production use:
        1. Search HuggingFace for voice cloning TTS models
        2. Try models like: facebook/tts_transformer, suno/bark
        3. Load pre-trained weights
    """
    print(f"Loading Chatterbox TTS model: {model_name}")
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
    print("Creating Chatterbox model...")
    model = Chatterbox(config)

    # Load weights (if available)
    if model_path and model_path.exists():
        weights_loaded = load_weights(model, model_path)
        if not weights_loaded:
            print("⚠ Warning: Model initialized with random weights")
            print("   Load pre-trained weights from HuggingFace for synthesis")

        # Try to load vocoder weights
        vocoder_loaded = load_vocoder_weights(model.vocoder, model_path)
        if not vocoder_loaded:
            print("⚠ Warning: Vocoder initialized with random weights")
            print("   For high-quality audio, load pre-trained HiFi-GAN weights")
    else:
        print("⚠ Warning: No model path found, using random initialization")

    model.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        from transformers import AutoTokenizer

        if model_path:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        else:
            # Fallback to SmolLM2 tokenizer (since Chatterbox is based on Llama)
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"⚠ Warning: Could not load tokenizer: {e}")
        tokenizer = None

    # Create processor
    print("Creating processor...")
    processor = create_processor(tokenizer=tokenizer, sample_rate=config.acoustic_config.sample_rate)
    print("✓ Processor created")

    print("=" * 70)
    print("✓ Chatterbox model loaded")
    print("\nNote: This is a reference implementation.")
    print("For production voice cloning TTS:")
    print("  1. Search HuggingFace for 'voice cloning' or 'multi-speaker TTS'")
    print("  2. Try models like facebook/tts_transformer, suno/bark")
    print("  3. Load pre-trained weights with voice cloning support")
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
        print("\nNote: Chatterbox may not be available on HuggingFace yet.")
        print("Alternative voice cloning TTS models to try:")
        print("  - facebook/tts_transformer")
        print("  - suno/bark (larger, high quality)")
        print("  - coqui/XTTS-v2 (multi-speaker)")
        print("\nProceeding with placeholder initialization...")
        return None


def load_weights(model: Chatterbox, model_path: Path) -> bool:
    """
    Load pre-trained weights into model.

    Args:
        model: Chatterbox model instance
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


def load_vocoder_weights(
    vocoder: HiFiGANGenerator,
    model_path: Path,
    vocoder_checkpoint: Optional[str] = None,
) -> bool:
    """
    Load pre-trained HiFi-GAN vocoder weights.

    Args:
        vocoder: HiFiGANGenerator instance
        model_path: Path to model directory
        vocoder_checkpoint: Optional specific checkpoint to load

    Returns:
        True if weights loaded successfully, False otherwise

    Note:
        If no vocoder weights are found locally, you can download pre-trained
        HiFi-GAN checkpoints from:
        - https://github.com/jik876/hifi-gan
        - HuggingFace: hifi-gan/LJSpeech-V1, etc.
    """
    # Try loading vocoder weights from model directory
    vocoder_paths = [
        model_path / "vocoder.safetensors",
        model_path / "vocoder.npz",
        model_path / "hifigan" / "generator.safetensors",
        model_path / "hifigan" / "generator.npz",
    ]

    for voc_path in vocoder_paths:
        if voc_path.exists():
            try:
                weights = mx.load(str(voc_path))
                vocoder.update(weights)
                print(f"✓ Vocoder weights loaded from {voc_path.name}")
                return True
            except Exception as e:
                print(f"Could not load vocoder weights from {voc_path}: {e}")

    # Try loading from specified checkpoint
    if vocoder_checkpoint:
        try:
            checkpoint_path = Path(vocoder_checkpoint)
            if checkpoint_path.exists():
                # Check if it's a PyTorch checkpoint
                if str(checkpoint_path).endswith(('.pt', '.pth', '.ckpt')):
                    weights = convert_pytorch_vocoder_weights(checkpoint_path)
                else:
                    weights = mx.load(str(checkpoint_path))

                vocoder.update(weights)
                print(f"✓ Vocoder weights loaded from checkpoint: {checkpoint_path}")
                return True
        except Exception as e:
            print(f"Could not load vocoder checkpoint: {e}")

    return False


def convert_pytorch_vocoder_weights(checkpoint_path: Path) -> Dict[str, mx.array]:
    """
    Convert PyTorch HiFi-GAN weights to MLX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt, .pth, .ckpt)

    Returns:
        Dictionary of MLX arrays

    Note:
        This requires PyTorch to be installed for loading the checkpoint.
        The function maps PyTorch state_dict keys to MLX format.
    """
    try:
        import torch
        import numpy as np
    except ImportError:
        raise ImportError(
            "PyTorch is required for converting checkpoints. "
            "Install with: pip install torch"
        )

    # Load PyTorch checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract generator state dict (might be nested)
    if isinstance(checkpoint, dict):
        if 'generator' in checkpoint:
            state_dict = checkpoint['generator']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Convert to MLX format
    mlx_weights = {}

    for key, value in state_dict.items():
        # Skip discriminator weights (we only need generator)
        if 'discriminator' in key.lower() or 'mpd' in key.lower() or 'msd' in key.lower():
            continue

        # Convert tensor to numpy then to MLX
        numpy_value = value.cpu().numpy()
        mlx_value = mx.array(numpy_value)

        # Map PyTorch key names to MLX format
        # PyTorch Conv1d uses (out_channels, in_channels, kernel_size)
        # MLX Conv1d uses (out_channels, kernel_size, in_channels)
        if 'weight' in key and numpy_value.ndim == 3:
            # Transpose conv weights: (O, I, K) -> (O, K, I)
            mlx_value = mx.transpose(mlx_value, (0, 2, 1))

        mlx_weights[key] = mlx_value

    print(f"✓ Converted {len(mlx_weights)} weight tensors from PyTorch to MLX")
    return mlx_weights


def download_pretrained_vocoder(
    vocoder_name: str = "hifi-gan-ljspeech",
    cache_dir: Optional[str] = None,
) -> Optional[Path]:
    """
    Download pre-trained HiFi-GAN vocoder from HuggingFace.

    Args:
        vocoder_name: Vocoder checkpoint name
        cache_dir: Optional cache directory

    Returns:
        Path to downloaded vocoder, or None if not found

    Available vocoders:
        - hifi-gan-ljspeech: 24kHz, English TTS (SpeechBrain)
        - hifi-gan-universal: Multi-language, 24kHz (NVIDIA)
    """
    try:
        from huggingface_hub import hf_hub_download

        print(f"Downloading vocoder: {vocoder_name}")

        # Map vocoder names to HuggingFace repo IDs and filenames
        vocoder_configs = {
            "hifi-gan-ljspeech": {
                "repo_id": "speechbrain/tts-hifigan-ljspeech",
                "filename": "generator.ckpt",
            },
            "hifi-gan-universal": {
                "repo_id": "nvidia/hifigan_universal_v1",
                "filename": "hifigan.pt",
            },
        }

        config = vocoder_configs.get(vocoder_name)
        if config is None:
            # Assume it's a custom repo_id
            print(f"Using custom repo: {vocoder_name}")
            repo_id = vocoder_name
            # Try common checkpoint filenames
            filenames = ["generator.ckpt", "generator.pth", "pytorch_model.bin", "model.pth"]

            for filename in filenames:
                try:
                    checkpoint_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=cache_dir,
                    )
                    print(f"✓ Vocoder downloaded to {checkpoint_path}")
                    return Path(checkpoint_path)
                except Exception:
                    continue

            raise ValueError(f"Could not find checkpoint in {repo_id} with filenames: {filenames}")

        else:
            # Download from known repo
            checkpoint_path = hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename"],
                cache_dir=cache_dir,
            )

            print(f"✓ Vocoder downloaded to {checkpoint_path}")
            return Path(checkpoint_path)

    except Exception as e:
        print(f"Could not download vocoder: {e}")
        print("\nNote: You can download HiFi-GAN manually from:")
        print("  SpeechBrain: https://huggingface.co/speechbrain/tts-hifigan-ljspeech")
        print("  NVIDIA: https://catalog.ngc.nvidia.com/orgs/nvidia/models/hifigan")
        print("  or search HuggingFace for 'hifi-gan' models")
        return None


def load_tts_model_weights(
    model: "Chatterbox",
    checkpoint_path: Path,
    model_type: str = "auto",
    strict: bool = False,
) -> bool:
    """
    Load pre-trained TTS model weights into Chatterbox.

    Supports:
    - MLX format (.npz)
    - PyTorch format (.pt, .pth, .ckpt) - auto-converts using convert_weights.py
    - Kokoro-82M weights
    - Spark-TTS-0.5B weights

    Args:
        model: Chatterbox model instance
        checkpoint_path: Path to checkpoint file
        model_type: Model type ("auto", "kokoro", "spark", "mlx")
        strict: If True, raise error on missing/unexpected keys

    Returns:
        True if weights loaded successfully, False otherwise

    Example:
        >>> model = create_model()
        >>> load_tts_model_weights(model, "kokoro_weights.pth", model_type="kokoro")
        >>> # Model is now ready for synthesis
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return False

    print(f"Loading TTS model weights from {checkpoint_path}")

    try:
        # Check file extension to determine format
        suffix = checkpoint_path.suffix.lower()

        if suffix == ".npz":
            # MLX format - load directly
            print("Loading MLX format weights...")
            weights = mx.load(str(checkpoint_path))
            model.update(weights)
            print(f"✓ Loaded {len(weights)} weight tensors (MLX format)")
            return True

        elif suffix in [".pt", ".pth", ".ckpt"]:
            # PyTorch format - convert using convert_weights.py
            print(f"Loading PyTorch checkpoint ({model_type} model)...")

            # Import conversion utilities
            from .convert_weights import convert_hf_tts_to_chatterbox

            # Convert weights
            mlx_weights = convert_hf_tts_to_chatterbox(
                checkpoint_path,
                model_type=model_type,
                chatterbox_config=model.config,
            )

            # Update model
            if strict:
                model.update(mlx_weights)
            else:
                # Non-strict mode: only load matching keys
                model_params = set(model.parameters().keys())
                checkpoint_params = set(mlx_weights.keys())

                # Find matching keys
                matching_keys = model_params & checkpoint_params
                missing_keys = model_params - checkpoint_params
                unexpected_keys = checkpoint_params - model_params

                if matching_keys:
                    matched_weights = {k: mlx_weights[k] for k in matching_keys}
                    model.update(matched_weights)
                    print(f"✓ Loaded {len(matching_keys)} matching weight tensors")

                if missing_keys:
                    print(f"⚠ Missing keys in checkpoint: {len(missing_keys)}")
                    if len(missing_keys) <= 10:
                        for key in sorted(missing_keys)[:10]:
                            print(f"    - {key}")

                if unexpected_keys:
                    print(f"⚠ Unexpected keys in checkpoint: {len(unexpected_keys)}")
                    if len(unexpected_keys) <= 10:
                        for key in sorted(unexpected_keys)[:10]:
                            print(f"    - {key}")

            return True

        else:
            print(f"Unsupported file format: {suffix}")
            print("Supported formats: .npz (MLX), .pt/.pth/.ckpt (PyTorch)")
            return False

    except Exception as e:
        print(f"Error loading TTS model weights: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_model(
    model: Chatterbox,
    output_path: str,
    config: Optional[ChatterboxConfig] = None,
):
    """
    Save Chatterbox model.

    Args:
        model: Chatterbox model
        output_path: Output directory
        config: Model configuration

    Example:
        >>> model, processor = load()
        >>> save_model(model, "./my_chatterbox_model")
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


def get_model_info(model: Chatterbox) -> dict:
    """
    Get model information and statistics.

    Args:
        model: Chatterbox model

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
        "text_embedding": model.text_embedding,
        "transformer": model.transformer_layers,
    }

    if model.config.use_voice_cloning:
        components["voice_encoder"] = model.voice_encoder

    if model.config.use_expressiveness:
        components["expressiveness"] = model.expressiveness_module

    components["acoustic_head"] = model.acoustic_head

    for name, component in components.items():
        if isinstance(component, list):
            params = sum(
                sum(p.size for p in layer.parameters().values())
                for layer in component
            )
        else:
            params = sum(p.size for p in component.parameters().values())
        component_params[name] = params
        total_params += params

    # Memory estimate (FP16)
    memory_fp16_mb = (total_params * 2) / (1024 * 1024)

    # Memory estimate (4-bit)
    memory_4bit_mb = (total_params * 0.5) / (1024 * 1024)

    info = {
        "total_params": total_params,
        "component_params": component_params,
        "memory_fp16_mb": memory_fp16_mb,
        "memory_4bit_mb": memory_4bit_mb,
        "sample_rate": model.config.acoustic_config.sample_rate,
        "num_mels": model.config.acoustic_config.num_mels,
        "voice_cloning": model.config.use_voice_cloning,
        "expressiveness": model.config.use_expressiveness,
    }

    return info


def print_model_info(model: Chatterbox):
    """
    Print model information.

    Args:
        model: Chatterbox model

    Example:
        >>> model, processor = load()
        >>> print_model_info(model)
    """
    info = get_model_info(model)

    print("\n" + "=" * 70)
    print("Chatterbox Model Information")
    print("=" * 70)
    print(f"\nTotal Parameters: {info['total_params']:,}")
    print("\nComponent Parameters:")
    for name, params in info["component_params"].items():
        print(f"  {name}: {params:,}")
    print(f"\nMemory (FP16): {info['memory_fp16_mb']:.1f} MB")
    print(f"Memory (4-bit): {info['memory_4bit_mb']:.1f} MB")
    print(f"\nSample Rate: {info['sample_rate']} Hz")
    print(f"Mel Bins: {info['num_mels']}")
    print(f"\nFeatures:")
    print(f"  Voice Cloning: {info['voice_cloning']}")
    print(f"  Expressiveness Control: {info['expressiveness']}")
    print("=" * 70)
