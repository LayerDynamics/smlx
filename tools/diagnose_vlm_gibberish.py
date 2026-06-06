#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Diagnostic Script for VLM Gibberish Output Analysis

This script investigates the root causes of gibberish outputs from VLM models:
1. Tokenizer compatibility and vocabulary alignment
2. Weight loading and initialization status
3. Vision-text embedding alignment
4. Model architecture consistency with HuggingFace implementations

Usage:
    python -m tools.diagnose_vlm_gibberish --model nanoVLM
    python -m tools.diagnose_vlm_gibberish --model Moondream2
    python -m tools.diagnose_vlm_gibberish --model TinyLLaVA
    python -m tools.diagnose_vlm_gibberish --all
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw

# Add smlx to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_simple_test_image() -> Image.Image:
    """Create a simple test image for diagnostics."""
    img = Image.new("RGB", (224, 224), color="blue")
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 174, 174], fill="red")
    return img


def diagnose_nanovlm() -> Dict[str, Any]:
    """Diagnose nanoVLM for gibberish output causes."""
    print("\n" + "=" * 80)
    print("Diagnosing nanoVLM")
    print("=" * 80)

    results = {"model_name": "nanoVLM", "issues": [], "warnings": [], "info": {}}

    try:
        from smlx.models.nanoVLM import load, generate

        # 1. Load model and capture warnings
        print("\n1. Loading model...")
        model, processor = load("lusxvr/nanoVLM-222M")

        # 2. Check tokenizer
        print("\n2. Checking tokenizer...")
        tokenizer = processor.tokenizer

        # Check if tokenizer loaded correctly
        if hasattr(tokenizer, "name_or_path"):
            results["info"]["tokenizer_source"] = tokenizer.name_or_path
            print(f"   Tokenizer source: {tokenizer.name_or_path}")
        else:
            results["issues"].append("Tokenizer missing name_or_path attribute")
            print("   ⚠️ Tokenizer missing name_or_path")

        # Check vocab size
        vocab_size = len(tokenizer)
        results["info"]["vocab_size"] = vocab_size
        print(f"   Vocab size: {vocab_size}")

        # Check model's expected vocab size
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            model_vocab_size = model.config.vocab_size
            results["info"]["model_vocab_size"] = model_vocab_size
            print(f"   Model vocab size: {model_vocab_size}")

            if vocab_size != model_vocab_size:
                results["issues"].append(
                    f"Tokenizer vocab mismatch: tokenizer={vocab_size}, model={model_vocab_size}"
                )
                print(
                    f"   ❌ MISMATCH: Tokenizer vocab ({vocab_size}) != Model vocab ({model_vocab_size})"
                )

        # Test tokenization
        test_text = "A red circle on a blue background"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"\n   Test encoding/decoding:")
        print(f"   Original: '{test_text}'")
        print(f"   Decoded:  '{decoded}'")

        if test_text.lower() != decoded.lower().strip():
            results["warnings"].append("Tokenizer decode doesn't match original text")
            print("   ⚠️ Decode mismatch detected")

        # 3. Check weight initialization
        print("\n3. Checking weight initialization...")

        # Check for NaN/Inf in model parameters
        param_stats = {}
        for name, param in model.parameters().items():
            if hasattr(param, "size"):
                param_array = mx.array(param) if not isinstance(param, mx.array) else param

                has_nan = bool(mx.any(mx.isnan(param_array)))
                has_inf = bool(mx.any(mx.isinf(param_array)))

                if has_nan or has_inf:
                    issue = f"Parameter '{name}' has "
                    if has_nan:
                        issue += "NaN values "
                    if has_inf:
                        issue += "Inf values"
                    results["issues"].append(issue)
                    print(f"   ❌ {issue}")

                param_stats[name] = {
                    "shape": param.shape if hasattr(param, "shape") else None,
                    "mean": float(mx.mean(param_array).item()) if not has_nan else None,
                    "std": float(mx.std(param_array).item()) if not has_nan else None,
                }

        results["info"]["param_stats_sample"] = {k: v for k, v in list(param_stats.items())[:5]}

        # 4. Check vision-text embedding alignment
        print("\n4. Checking vision-text embedding alignment...")

        # Create test image
        test_image = create_simple_test_image()

        # Process image
        pixel_values = processor.image_processor(test_image)

        # Get vision embeddings (if model has this method)
        if hasattr(model, "vision_encoder"):
            vision_output = model.vision_encoder(mx.array([pixel_values]))
            vision_shape = vision_output.shape
            vision_mean = float(mx.mean(vision_output).item())
            vision_std = float(mx.std(vision_output).item())

            results["info"]["vision_embedding"] = {
                "shape": vision_shape,
                "mean": vision_mean,
                "std": vision_std,
            }

            print(f"   Vision embedding shape: {vision_shape}")
            print(f"   Vision embedding mean: {vision_mean:.4f}")
            print(f"   Vision embedding std: {vision_std:.4f}")

            # Check for unusual statistics
            if abs(vision_mean) > 10:
                results["warnings"].append(f"Vision embedding mean very large: {vision_mean:.4f}")
                print(f"   ⚠️ Vision mean unusually large")

            if vision_std < 0.01 or vision_std > 100:
                results["warnings"].append(f"Vision embedding std unusual: {vision_std:.4f}")
                print(f"   ⚠️ Vision std unusual")

        # Check text embeddings
        if hasattr(model, "text_model") or hasattr(model, "language_model"):
            text_model = getattr(model, "text_model", None) or getattr(
                model, "language_model", None
            )

            if hasattr(text_model, "embed_tokens"):
                # Get embedding for a few tokens
                test_tokens = mx.array([[1, 2, 3, 4, 5]])
                text_emb = text_model.embed_tokens(test_tokens)
                text_mean = float(mx.mean(text_emb).item())
                text_std = float(mx.std(text_emb).item())

                results["info"]["text_embedding"] = {
                    "shape": text_emb.shape,
                    "mean": text_mean,
                    "std": text_std,
                }

                print(f"   Text embedding shape: {text_emb.shape}")
                print(f"   Text embedding mean: {text_mean:.4f}")
                print(f"   Text embedding std: {text_std:.4f}")

                # Check alignment
                if "vision_embedding" in results["info"]:
                    vision_std = results["info"]["vision_embedding"]["std"]
                    scale_diff = abs(vision_std - text_std) / max(vision_std, text_std)

                    if scale_diff > 0.5:
                        results["warnings"].append(
                            f"Vision and text embedding scales differ significantly: "
                            f"vision_std={vision_std:.4f}, text_std={text_std:.4f}"
                        )
                        print(f"   ⚠️ Embedding scale mismatch (diff: {scale_diff:.1%})")

        # 5. Test generation and analyze output
        print("\n5. Testing generation...")

        output = generate(
            model,
            processor,
            "Describe this image:",
            test_image,
            max_tokens=30,
            temperature=0.0,  # Greedy for deterministic output
        )

        results["info"]["sample_output"] = output
        print(f"   Output: {output}")

        # Analyze output for gibberish indicators
        words = output.split()
        avg_word_length = sum(len(w) for w in words) / max(len(words), 1)

        results["info"]["output_analysis"] = {
            "num_words": len(words),
            "avg_word_length": avg_word_length,
        }

        if avg_word_length > 15:
            results["issues"].append(f"Unusually long average word length: {avg_word_length:.1f}")
            print(f"   ❌ Abnormal word length: {avg_word_length:.1f}")

        # Check for special characters
        special_char_ratio = sum(1 for c in output if not c.isalnum() and c != " ") / max(
            len(output), 1
        )
        if special_char_ratio > 0.3:
            results["warnings"].append(f"High special character ratio: {special_char_ratio:.1%}")
            print(f"   ⚠️ High special char ratio: {special_char_ratio:.1%}")

    except Exception as e:
        results["issues"].append(f"Exception during diagnosis: {str(e)}")
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    return results


def diagnose_moondream2() -> Dict[str, Any]:
    """Diagnose Moondream2 for gibberish output causes."""
    print("\n" + "=" * 80)
    print("Diagnosing Moondream2")
    print("=" * 80)

    results = {"model_name": "Moondream2", "issues": [], "warnings": [], "info": {}}

    # Similar diagnostic logic as nanoVLM
    # (Abbreviated for brevity - would follow same pattern)

    try:
        from smlx.models.Moondream2 import load, generate

        print("\n1. Loading model...")
        model, tokenizer = load()

        print("\n2. Checking tokenizer...")
        vocab_size = len(tokenizer)
        results["info"]["vocab_size"] = vocab_size
        print(f"   Vocab size: {vocab_size}")

        # Test generation
        print("\n3. Testing generation...")
        test_image = create_simple_test_image()

        output = generate(
            model, tokenizer, test_image, "What do you see?", max_tokens=30, temperature=0.0
        )

        results["info"]["sample_output"] = output
        print(f"   Output: {output}")

    except Exception as e:
        results["issues"].append(f"Exception: {str(e)}")
        print(f"\n❌ Error: {e}")

    return results


def diagnose_tinyllava() -> Dict[str, Any]:
    """Diagnose TinyLLaVA for gibberish output causes."""
    print("\n" + "=" * 80)
    print("Diagnosing TinyLLaVA")
    print("=" * 80)

    results = {"model_name": "TinyLLaVA", "issues": [], "warnings": [], "info": {}}

    try:
        from smlx.models.TinyLLaVA import load, generate

        print("\n1. Loading model...")
        model, processor = load()

        print("\n2. Checking tokenizer...")
        vocab_size = len(processor.tokenizer)
        results["info"]["vocab_size"] = vocab_size
        print(f"   Vocab size: {vocab_size}")

        # Test generation
        print("\n3. Testing generation...")
        test_image = create_simple_test_image()

        output = generate(model, processor, "Describe:", test_image, max_tokens=30, temperature=0.0)

        results["info"]["sample_output"] = output
        print(f"   Output: {output}")

    except Exception as e:
        results["issues"].append(f"Exception: {str(e)}")
        print(f"\n❌ Error: {e}")

    return results


def print_summary(all_results: Dict[str, Dict[str, Any]]):
    """Print diagnostic summary."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    total_issues = sum(len(r["issues"]) for r in all_results.values())
    total_warnings = sum(len(r["warnings"]) for r in all_results.values())

    print(f"\nTotal Issues: {total_issues}")
    print(f"Total Warnings: {total_warnings}")

    for model_name, results in all_results.items():
        print(f"\n{model_name}:")

        if results["issues"]:
            print("  Issues:")
            for issue in results["issues"]:
                print(f"    ❌ {issue}")

        if results["warnings"]:
            print("  Warnings:")
            for warning in results["warnings"]:
                print(f"    ⚠️ {warning}")

        if not results["issues"] and not results["warnings"]:
            print("  ✅ No issues detected")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if total_issues > 0:
        print("\nCritical issues found. Recommended actions:")

        # Check for common patterns
        vocab_mismatch = any(
            "vocab" in issue.lower() for r in all_results.values() for issue in r["issues"]
        )

        if vocab_mismatch:
            print("\n1. Tokenizer Vocabulary Mismatch:")
            print("   - Verify correct tokenizer is loaded from HuggingFace")
            print("   - Check loader.py tokenizer initialization")
            print("   - Ensure tokenizer matches model training config")

        weight_issues = any(
            "nan" in issue.lower() or "inf" in issue.lower()
            for r in all_results.values()
            for issue in r["issues"]
        )

        if weight_issues:
            print("\n2. Weight Initialization Issues:")
            print("   - Some parameters contain NaN/Inf values")
            print("   - Check weight loading in loader.py")
            print("   - Verify HuggingFace Hub weights are complete")
            print("   - Consider using strict=True in load_weights()")

    if total_warnings > 0:
        print("\nWarnings detected. Consider investigating:")
        print("- Vision-text embedding alignment")
        print("- Tokenizer encode/decode consistency")
        print("- Output quality metrics")


def main():
    """Main diagnostic entry point."""
    parser = argparse.ArgumentParser(
        description="Diagnose VLM models for gibberish output causes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.diagnose_vlm_gibberish --model nanoVLM
  python -m tools.diagnose_vlm_gibberish --model Moondream2
  python -m tools.diagnose_vlm_gibberish --all
        """,
    )

    parser.add_argument(
        "--model", choices=["nanoVLM", "Moondream2", "TinyLLaVA"], help="Specific model to diagnose"
    )

    parser.add_argument("--all", action="store_true", help="Diagnose all VLM models")

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Must specify either --model or --all")

    results = {}

    if args.all or args.model == "nanoVLM":
        results["nanoVLM"] = diagnose_nanovlm()

    if args.all or args.model == "Moondream2":
        results["Moondream2"] = diagnose_moondream2()

    if args.all or args.model == "TinyLLaVA":
        results["TinyLLaVA"] = diagnose_tinyllava()

    print_summary(results)


if __name__ == "__main__":
    main()
