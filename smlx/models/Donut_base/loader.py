#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Donut model loader.

Handles loading Donut models from HuggingFace Hub.
"""

from pathlib import Path
from typing import Tuple

from .config import DEFAULT_CONFIG, DonutConfig, load_config
from .model import DonutModel
from .processor import DonutProcessor, create_processor


def load(
    model_name: str = "naver-clova-ix/donut-base",
    force_download: bool = False,
) -> Tuple[DonutModel, DonutProcessor]:
    """
    Load Donut model and processor.

    Args:
        model_name: HuggingFace model ID or local path
        force_download: Force re-download from Hub

    Returns:
        Tuple of (model, processor)

    Example:
        >>> # Load base model
        >>> model, processor = load("naver-clova-ix/donut-base")
        >>>
        >>> # Load fine-tuned for DocVQA
        >>> model, processor = load("naver-clova-ix/donut-base-finetuned-docvqa")
        >>>
        >>> # Load fine-tuned for RVL-CDIP (document classification)
        >>> model, processor = load("naver-clova-ix/donut-base-finetuned-rvlcdip")

    Note:
        This implementation provides the API structure. Full functionality
        requires loading pre-trained weights from HuggingFace Hub and
        complete Swin Transformer + BART implementations.

        See resources/mlx-examples for reference implementations of:
        - Swin Transformer vision encoder
        - BART decoder architecture
    """
    print(f"Loading Donut model: {model_name}")
    print("=" * 70)

    # Check if local path or HuggingFace ID
    if Path(model_name).exists():
        model_path = Path(model_name)
        print(f"Loading from local path: {model_path}")
    else:
        print(f"Downloading from HuggingFace Hub: {model_name}")
        model_path = download_model(model_name, force_download)

    # Load configuration
    config = load_config(str(model_path)) if model_path else DEFAULT_CONFIG

    # Create model
    print("Creating Donut model...")
    model = DonutModel(config)

    # Load weights (if available)
    if model_path and model_path.exists():
        weights_loaded = load_weights(model, model_path)
        if not weights_loaded:
            print("⚠ Warning: Model initialized with random weights")
            print("   Load pre-trained weights from HuggingFace for inference")
    else:
        print("⚠ Warning: No model path found, using random initialization")

    model.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path) if model_path else "facebook/bart-base")
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"⚠ Warning: Could not load tokenizer: {e}")
        tokenizer = None

    # Create processor
    image_size = (
        config.encoder_config.image_size
        if config and config.encoder_config
        else (224, 224)
    )
    processor = create_processor(image_size=image_size, tokenizer=tokenizer)

    print("=" * 70)
    print("✓ Donut model loaded")
    print("\nNote: This is a reference implementation.")
    print("For production use:")
    print("  1. Load pre-trained weights from HuggingFace Hub")
    print("  2. Implement full Swin Transformer encoder")
    print("  3. Implement full BART decoder")
    print("  4. See resources/mlx-examples for reference implementations")

    return model, processor


def download_model(model_name: str, force_download: bool = False) -> Path:
    """
    Download model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model ID
        force_download: Force re-download

    Returns:
        Path to downloaded model
    """
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            repo_id=model_name,
            force_download=force_download,
        )
        return Path(path)
    except Exception as e:
        print(f"Could not download model: {e}")
        return None


def load_weights(model: DonutModel, model_path: Path) -> bool:
    """
    Load pre-trained weights into model.

    Args:
        model: Donut model instance
        model_path: Path to model directory

    Returns:
        True if weights loaded successfully, False otherwise
    """
    import mlx.core as mx

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

    return False


def save_model(model: DonutModel, output_path: str, config: DonutConfig = None):
    """
    Save Donut model.

    Args:
        model: Donut model
        output_path: Output directory
        config: Model configuration
    """
    import mlx.core as mx
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
