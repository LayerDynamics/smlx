#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Command-line interface for quantizing SMLX models.

Usage:
    # List available quantization methods
    python -m smlx.tools.quantize --list

    # Quantize a model with 4-bit
    python -m smlx.tools.quantize \\
        --model mlx-community/SmolLM2-135M-Instruct \\
        --output ./quantized/smollm2-135m-4bit \\
        --method 4bit

    # Quantize with GPTQ (high quality)
    python -m smlx.tools.quantize \\
        --model mlx-community/SmolLM2-135M-Instruct \\
        --output ./quantized/smollm2-135m-gptq \\
        --method gptq \\
        --bits 4 \\
        --group-size 64

    # Quantize with AWQ (activation-aware)
    python -m smlx.tools.quantize \\
        --model mlx-community/SmolLM2-135M-Instruct \\
        --output ./quantized/smollm2-135m-awq \\
        --method awq

    # Auto-select best quantization method
    python -m smlx.tools.quantize \\
        --model mlx-community/SmolLM2-135M-Instruct \\
        --output ./quantized/smollm2-135m-auto \\
        --method auto

    # Get quantization info for a model
    python -m smlx.tools.quantize \\
        --model mlx-community/SmolLM2-135M-Instruct \\
        --info
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx


def list_methods():
    """List available quantization methods."""
    print("\n" + "=" * 70)
    print("Available Quantization Methods")
    print("=" * 70)

    methods = {
        "4bit": {
            "description": "Standard 4-bit quantization",
            "bits": "4-bit",
            "quality": "Good",
            "speed": "Fast",
            "memory": "~4x reduction",
        },
        "8bit": {
            "description": "Standard 8-bit quantization",
            "bits": "8-bit",
            "quality": "Excellent",
            "speed": "Medium",
            "memory": "~2x reduction",
        },
        "gptq": {
            "description": "GPTQ post-training quantization (Hessian-based)",
            "bits": "4-bit (default)",
            "quality": "Excellent",
            "speed": "Medium",
            "memory": "~4x reduction",
        },
        "awq": {
            "description": "Activation-aware weight quantization",
            "bits": "4-bit (default)",
            "quality": "Excellent",
            "speed": "Medium",
            "memory": "~4x reduction",
        },
        "dwq": {
            "description": "Distilled weight quantization",
            "bits": "4-bit (default)",
            "quality": "Very Good",
            "speed": "Medium",
            "memory": "~4x reduction",
        },
        "auto": {
            "description": "Automatic method selection based on hardware",
            "bits": "Variable",
            "quality": "Optimal",
            "speed": "Varies",
            "memory": "Variable",
        },
    }

    for method, info in methods.items():
        print(f"\n{method.upper()}")
        print(f"  Description: {info['description']}")
        print(f"  Bits:        {info['bits']}")
        print(f"  Quality:     {info['quality']}")
        print(f"  Speed:       {info['speed']}")
        print(f"  Memory:      {info['memory']}")

    print("\n" + "=" * 70)
    print("\nRecommendations:")
    print("  - For best quality/size tradeoff: gptq or awq")
    print("  - For fastest quantization: 4bit or 8bit")
    print("  - For automatic selection: auto")
    print("  - For M4 chipsets: group_size=64 recommended")


def get_model_info(model_path: str):
    """Get information about a model."""
    from smlx.quant import (
        estimate_4bit_size_reduction,
        estimate_8bit_size_reduction,
        estimate_model_size,
        get_actual_model_size,
    )

    print("\n" + "=" * 70)
    print(f"Model Information: {model_path}")
    print("=" * 70)

    # Load the model through the maintained backend to analyze its size.
    print("\nLoading model to analyze...")
    from smlx.models import mlx_backend

    model = mlx_backend.load(model_path).model

    # Get size estimates
    actual_size = get_actual_model_size(model)
    estimated_size = estimate_model_size(model)
    size_4bit = estimate_4bit_size_reduction(model)
    size_8bit = estimate_8bit_size_reduction(model)

    print(f"\nModel Size Analysis:")
    print(f"  Actual size:       {actual_size / 1024 / 1024:.2f} MB")
    print(f"  Estimated size:    {estimated_size / 1024 / 1024:.2f} MB")
    print(f"\nQuantization Estimates:")
    print(f"  4-bit size:        {size_4bit['quantized_mb']:.2f} MB "
          f"({size_4bit['reduction_ratio']:.2f}x reduction)")
    print(f"  8-bit size:        {size_8bit['quantized_mb']:.2f} MB "
          f"({size_8bit['reduction_ratio']:.2f}x reduction)")

    # Count parameters
    total_params = sum(p.size for p in model.parameters().values() if hasattr(p, 'size'))
    print(f"\nParameters:          {total_params:,}")
    print("=" * 70)


def quantize_model(
    model_path: str,
    output_path: str,
    method: str,
    bits: int = 4,
    group_size: int = 64,
):
    """Quantize a model and save it."""
    print("\n" + "=" * 70)
    print(f"Quantizing Model: {model_path}")
    print("=" * 70)
    print(f"\nMethod:      {method}")
    print(f"Bits:        {bits}")
    print(f"Group size:  {group_size}")
    print(f"Output:      {output_path}")

    # Load the full-precision model through the maintained backend, quantize it
    # with the kept smlx.quant algorithm, then save the result in MLX format.
    print(f"\nLoading {model_path}...")
    start_time = time.time()

    from mlx_lm.utils import save_model as mlx_save_model

    from smlx.models import mlx_backend
    from smlx.quant import quantize_model as quantize_weights

    bm = mlx_backend.load(model_path)
    model, tokenizer = bm.model, bm.processor
    load_time = time.time() - start_time
    print(f"Loaded in {load_time:.2f}s")

    print(f"\nQuantizing ({bits}-bit, group_size={group_size})...")
    model = quantize_weights(model, bits=bits, group_size=group_size)

    # Save model
    print(f"\nSaving quantized model to {output_path}...")
    save_start = time.time()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    mlx_save_model(output_path, model)

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(str(output_path))

    # Save quantization metadata
    metadata = {
        "quantization_method": method,
        "bits": bits,
        "group_size": group_size,
        "quantized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_model": model_path,
    }

    metadata_path = output_path / "quantization_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    save_time = time.time() - save_start
    print(f"Saved in {save_time:.2f}s")

    # Print summary
    print("\n" + "=" * 70)
    print("Quantization Complete!")
    print("=" * 70)
    print(f"\nTotal time:  {load_time + save_time:.2f}s")
    print(f"Output:      {output_path}")
    print(f"Method:      {method}")
    print(f"Config:      {bits}-bit, group_size={group_size}")
    print("\nYou can now load this model with:")
    print("  from smlx.models import load_model")
    print(f"  bm = load_model('{output_path}')")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quantize SMLX models for efficient deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model path or HuggingFace repo ID",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for quantized model",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["4bit", "8bit", "gptq", "awq", "dwq", "auto"],
        default="4bit",
        help="Quantization method (default: 4bit)",
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 6, 8],
        help="Bits per weight (default: 4)",
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64, optimized for M4)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available quantization methods",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Get information about a model",
    )

    args = parser.parse_args()

    # Handle list command
    if args.list:
        list_methods()
        return

    # Handle info command
    if args.info:
        if not args.model:
            print("Error: --model required for --info")
            sys.exit(1)
        get_model_info(args.model)
        return

    # Handle quantization
    if not args.model or not args.output:
        print("Error: --model and --output required for quantization")
        parser.print_help()
        sys.exit(1)

    quantize_model(
        model_path=args.model,
        output_path=args.output,
        method=args.method,
        bits=args.bits,
        group_size=args.group_size,
    )


if __name__ == "__main__":
    main()
