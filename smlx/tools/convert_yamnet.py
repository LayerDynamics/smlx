#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
YAMNet Weight Conversion Tool.

Converts YAMNet weights from PyTorch (torch_audioset) format to MLX format.
Optionally uploads the converted model to HuggingFace Hub.

Usage:
    # Convert and save locally
    python -m smlx.tools.convert_yamnet --output ./yamnet_mlx

    # Convert and upload to HuggingFace
    python -m smlx.tools.convert_yamnet --upload mlx-community/yamnet

    # Force re-download of source weights
    python -m smlx.tools.convert_yamnet --force-download

Source:
    PyTorch weights from torch_audioset project:
    https://github.com/w-hc/torch_audioset

License:
    Original YAMNet: Apache 2.0 (Google)
    torch_audioset: MIT License
"""

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import numpy as np

from smlx.models.YAMNet.config import DEFAULT_CONFIG
from smlx.models.YAMNet.weights import (
    count_parameters,
    get_pytorch_to_mlx_mapping,
    validate_weight_shapes,
)


PYTORCH_WEIGHTS_URL = "https://github.com/w-hc/torch_audioset/releases/download/v0.1/yamnet.pth"


def download_pytorch_weights(
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """Download pre-converted PyTorch YAMNet weights.

    Args:
        cache_dir: Directory to cache weights (default: ~/.cache/smlx/yamnet)
        force_download: Force re-download even if cached

    Returns:
        Path to downloaded PyTorch weights file
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "smlx" / "yamnet"

    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / "yamnet.pth"

    if weights_path.exists() and not force_download:
        print(f"✓ Using cached PyTorch weights: {weights_path}")
        return weights_path

    print(f"Downloading PyTorch YAMNet weights from torch_audioset...")
    print(f"  Source: {PYTORCH_WEIGHTS_URL}")
    print(f"  Target: {weights_path}")

    try:
        urllib.request.urlretrieve(PYTORCH_WEIGHTS_URL, weights_path)
        print(f"✓ Download complete ({weights_path.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        raise RuntimeError(f"Failed to download weights: {e}")

    return weights_path


def convert_pytorch_to_mlx(pytorch_path: Path, verbose: bool = True) -> Dict:
    """Convert PyTorch YAMNet weights to MLX format.

    Args:
        pytorch_path: Path to PyTorch .pth file
        verbose: Print conversion progress

    Returns:
        Dictionary of MLX weight arrays
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for weight conversion. "
            "Install with: pip install torch"
        )

    if verbose:
        print(f"\nConverting PyTorch weights to MLX format...")
        print(f"  Loading: {pytorch_path}")

    # Load PyTorch state_dict
    state_dict = torch.load(pytorch_path, map_location='cpu')

    if verbose:
        print(f"  PyTorch weights: {len(state_dict)} tensors")

    # Get name mapping
    name_mapping = get_pytorch_to_mlx_mapping()

    # Convert weights
    mlx_weights = {}
    unmapped_keys = []

    for pytorch_key, pytorch_tensor in state_dict.items():
        # Skip PyTorch-specific tracking variables
        if 'num_batches_tracked' in pytorch_key:
            continue

        # Map to MLX key
        if pytorch_key in name_mapping:
            mlx_key = name_mapping[pytorch_key]

            # Convert to NumPy first, then MLX
            numpy_array = pytorch_tensor.numpy()
            mlx_array = mx.array(numpy_array)

            mlx_weights[mlx_key] = mlx_array

            if verbose and len(mlx_weights) % 10 == 0:
                print(f"    Converted {len(mlx_weights)} tensors...")

        else:
            unmapped_keys.append(pytorch_key)

    if verbose:
        print(f"  ✓ Converted {len(mlx_weights)} tensors to MLX format")

        if unmapped_keys:
            print(f"  ⚠ Unmapped keys ({len(unmapped_keys)}):")
            for key in unmapped_keys[:5]:
                print(f"      - {key}")
            if len(unmapped_keys) > 5:
                print(f"      ... and {len(unmapped_keys) - 5} more")

    # Validate shapes
    is_valid, errors = validate_weight_shapes(mlx_weights, strict=False)
    if not is_valid:
        print(f"\n  ⚠ Weight validation warnings:")
        for error in errors:
            print(f"      {error}")
    elif verbose:
        print(f"  ✓ Weight shapes validated")

    # Count parameters
    total_params = count_parameters(mlx_weights)
    if verbose:
        print(f"  ✓ Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    return mlx_weights


def save_mlx_weights(
    weights: Dict,
    output_dir: Path,
    save_config: bool = True,
    verbose: bool = True,
) -> None:
    """Save MLX weights and configuration.

    Args:
        weights: Dictionary of MLX weight arrays
        output_dir: Output directory
        save_config: Whether to save config.json
        verbose: Print save progress
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights as NPZ
    weights_file = output_dir / "yamnet.npz"
    if verbose:
        print(f"\nSaving MLX weights...")
        print(f"  Target: {weights_file}")

    mx.savez(str(weights_file), **weights)

    file_size_mb = weights_file.stat().st_size / 1024 / 1024
    if verbose:
        print(f"  ✓ Saved weights ({file_size_mb:.1f} MB)")

    # Save configuration
    if save_config:
        config_file = output_dir / "config.json"
        config_dict = {
            "model_type": DEFAULT_CONFIG.model_type,
            "num_classes": DEFAULT_CONFIG.num_classes,
            "embedding_size": DEFAULT_CONFIG.embedding_size,
            "sample_rate": DEFAULT_CONFIG.sample_rate,
            "num_mel_bins": DEFAULT_CONFIG.num_mel_bins,
            "patch_hop_seconds": DEFAULT_CONFIG.patch_hop_seconds,
            "patch_window_seconds": DEFAULT_CONFIG.patch_window_seconds,
        }

        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

        if verbose:
            print(f"  ✓ Saved config: {config_file}")

    # Save README
    readme_file = output_dir / "README.md"
    readme_content = f"""# YAMNet MLX

YAMNet audio event classifier converted to MLX format for Apple Silicon.

## Model Details

- **Parameters**: {count_parameters(weights):,} (~{count_parameters(weights)/1e6:.1f}M)
- **Classes**: 521 AudioSet event categories
- **Architecture**: MobileNet-v1 with depthwise-separable convolutions
- **Input**: 16kHz mono audio
- **License**: Apache 2.0

## Usage

```python
from smlx.models.YAMNet import load, classify

# Load model
model = load("mlx-community/yamnet")

# Classify audio
predictions = classify(model, "audio.wav", top_k=5)

for pred in predictions:
    print(f"{{pred.label}}: {{pred.score:.3f}}")
```

## Conversion

Converted from PyTorch (torch_audioset) to MLX format using `smlx.tools.convert_yamnet`.

**Sources**:
- Original: TensorFlow Hub (Google)
- PyTorch: w-hc/torch_audioset
- MLX: smlx/convert_yamnet.py

## License

Apache 2.0 (original YAMNet license from Google)
"""

    with open(readme_file, 'w') as f:
        f.write(readme_content)

    if verbose:
        print(f"  ✓ Saved README: {readme_file}")


def upload_to_huggingface(
    output_dir: Path,
    repo_id: str,
    verbose: bool = True,
) -> None:
    """Upload converted model to HuggingFace Hub.

    Args:
        output_dir: Directory containing converted weights
        repo_id: HuggingFace repo ID (e.g., "mlx-community/yamnet")
        verbose: Print upload progress
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for uploading. "
            "Install with: pip install huggingface-hub"
        )

    if verbose:
        print(f"\nUploading to HuggingFace Hub...")
        print(f"  Repo: {repo_id}")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        if verbose:
            print(f"  ✓ Repository ready")
    except Exception as e:
        raise RuntimeError(f"Failed to create repository: {e}")

    # Upload folder
    try:
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            repo_type="model",
        )
        if verbose:
            print(f"  ✓ Upload complete")
            print(f"  → https://huggingface.co/{repo_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload: {e}")


def main():
    """Main conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert YAMNet weights from PyTorch to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./yamnet_mlx",
        help="Output directory for converted weights (default: ./yamnet_mlx)",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloaded weights (default: ~/.cache/smlx/yamnet)",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of PyTorch weights",
    )

    parser.add_argument(
        "--upload",
        type=str,
        default=None,
        help="Upload to HuggingFace Hub (e.g., mlx-community/yamnet)",
    )

    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Don't save config.json",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    verbose = not args.quiet
    output_dir = Path(args.output)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    try:
        print("=" * 70)
        print("YAMNet PyTorch → MLX Conversion")
        print("=" * 70)

        # Step 1: Download PyTorch weights
        pytorch_path = download_pytorch_weights(
            cache_dir=cache_dir,
            force_download=args.force_download,
        )

        # Step 2: Convert to MLX
        mlx_weights = convert_pytorch_to_mlx(pytorch_path, verbose=verbose)

        # Step 3: Save MLX weights
        save_mlx_weights(
            mlx_weights,
            output_dir,
            save_config=not args.no_config,
            verbose=verbose,
        )

        # Step 4: Upload if requested
        if args.upload:
            upload_to_huggingface(output_dir, args.upload, verbose=verbose)

        print("\n" + "=" * 70)
        print("✓ Conversion complete!")
        print("=" * 70)

        if verbose:
            print(f"\nOutput directory: {output_dir.absolute()}")
            print(f"  - yamnet.npz ({(output_dir / 'yamnet.npz').stat().st_size / 1024 / 1024:.1f} MB)")
            if not args.no_config:
                print(f"  - config.json")
            print(f"  - README.md")

            if args.upload:
                print(f"\nModel available at: https://huggingface.co/{args.upload}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
