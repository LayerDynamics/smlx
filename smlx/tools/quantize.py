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

    # Try to determine model type
    model_type = "unknown"
    if "SmolLM2-135M" in model_path or "135M" in model_path:
        model_type = "SmolLM2-135M"
    elif "SmolLM2-360M" in model_path or "360M" in model_path:
        model_type = "SmolLM2-360M"
    elif "SmolVLM-256M" in model_path or "256M" in model_path:
        model_type = "SmolVLM-256M"

    print(f"\nDetected model type: {model_type}")

    # Load model to get size information
    print("\nLoading model to analyze...")
    if model_type.startswith("SmolLM2-135M"):
        from smlx.models.SmolLM2_135M import load

        model, tokenizer = load(model_path)
    elif model_type.startswith("SmolLM2-360M"):
        from smlx.models.SmolLM2_360M import load

        model, tokenizer = load(model_path)
    else:
        print("Unknown model type - cannot load for analysis")
        return

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

    # Determine model type
    model_type = "unknown"
    if "SmolLM2-135M" in model_path or "135M" in model_path:
        model_type = "SmolLM2-135M"
    elif "SmolLM2-360M" in model_path or "360M" in model_path:
        model_type = "SmolLM2-360M"

    if model_type == "unknown":
        print(f"\nError: Unknown model type for {model_path}")
        print("Currently supported: SmolLM2-135M, SmolLM2-360M")
        return

    # Load model
    print(f"\nLoading {model_type}...")
    start_time = time.time()

    if model_type == "SmolLM2-135M":
        from smlx.models.SmolLM2_135M import load, save_model

        model, tokenizer = load(
            model_path,
            quantize=method,
            quantization_config={"bits": bits, "group_size": group_size},
        )
    elif model_type == "SmolLM2-360M":
        from smlx.models.SmolLM2_360M import load, save_model

        model, tokenizer = load(
            model_path,
            quantize=method,
            quantization_config={"bits": bits, "group_size": group_size},
        )

    load_time = time.time() - start_time
    print(f"Loaded and quantized in {load_time:.2f}s")

    # Save model
    print(f"\nSaving quantized model to {output_path}...")
    save_start = time.time()

    output_path = Path(output_path)
    save_model(model, output_path, config=model.args if hasattr(model, 'args') else None)

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
    print(f"  from smlx.models.{model_type.replace('-', '_')} import load")
    print(f"  model, tokenizer = load('{output_path}')")


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
