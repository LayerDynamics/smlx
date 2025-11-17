#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Complete Quantization Integration Example

Demonstrates all three pathways for using quantization in SMLX:
1. Direct Model Loader Integration
2. CLI Tool (shown as code comments)
3. Server Integration (shown as code comments)

This example shows the complete workflow from development to production.
"""

import time
from pathlib import Path

from smlx.models.SmolLM2_135M import generate, load


def demonstrate_loader_integration():
    """Demonstrate direct model loader quantization."""
    print("\n" + "=" * 70)
    print("1. DIRECT MODEL LOADER INTEGRATION")
    print("=" * 70)
    print("\nThis is the simplest way to use quantization in your code.")

    test_prompt = "Explain machine learning in simple terms:"

    # Example 1a: No quantization (baseline)
    print("\n1a. Loading without quantization (FP16)...")
    model_fp16, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    start = time.time()
    output_fp16 = generate(
        model=model_fp16,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=40,
        temperature=0.7,
        verbose=False,
    )
    time_fp16 = time.time() - start
    print(f"   Time: {time_fp16:.2f}s")
    print(f"   Output: {output_fp16[:60]}...")

    # Example 1b: Single-line 4-bit quantization
    print("\n1b. Loading with 4-bit quantization (single line!)...")
    model_4bit, tokenizer = load(
        "mlx-community/SmolLM2-135M-Instruct", quantize="4bit"
    )

    start = time.time()
    output_4bit = generate(
        model=model_4bit,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=40,
        temperature=0.7,
        verbose=False,
    )
    time_4bit = time.time() - start
    print(f"   Time: {time_4bit:.2f}s")
    print(f"   Output: {output_4bit[:60]}...")
    print(f"   Speedup: {time_fp16/time_4bit:.2f}x")

    # Example 1c: High-quality GPTQ quantization
    print("\n1c. Loading with GPTQ quantization (best quality)...")
    model_gptq, tokenizer = load(
        "mlx-community/SmolLM2-135M-Instruct",
        quantize="gptq",
        quantization_config={"bits": 4, "group_size": 64},
    )

    start = time.time()
    output_gptq = generate(
        model=model_gptq,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=40,
        temperature=0.7,
        verbose=False,
    )
    time_gptq = time.time() - start
    print(f"   Time: {time_gptq:.2f}s")
    print(f"   Output: {output_gptq[:60]}...")
    print(f"   Speedup: {time_fp16/time_gptq:.2f}x")

    # Example 1d: Automatic method selection
    print("\n1d. Loading with automatic quantization selection...")
    model_auto, tokenizer = load(
        "mlx-community/SmolLM2-135M-Instruct", quantize="auto"
    )

    start = time.time()
    output_auto = generate(
        model=model_auto,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=40,
        temperature=0.7,
        verbose=False,
    )
    time_auto = time.time() - start
    print(f"   Time: {time_auto:.2f}s")
    print(f"   Output: {output_auto[:60]}...")

    print("\n   ✅ Loader integration complete!")
    print("   Code: model, tokenizer = load(model_path, quantize='4bit')")


def demonstrate_cli_usage():
    """Show CLI tool usage (as documentation)."""
    print("\n" + "=" * 70)
    print("2. CLI TOOL USAGE")
    print("=" * 70)
    print("\nThe CLI tool lets you quantize models offline for deployment.")

    print("\n# List available quantization methods:")
    print("$ python -m smlx.tools.quantize --list")

    print("\n# Get model information and size estimates:")
    print("$ python -m smlx.tools.quantize \\")
    print("    --model mlx-community/SmolLM2-135M-Instruct \\")
    print("    --info")

    print("\n# Quantize a model to 4-bit:")
    print("$ python -m smlx.tools.quantize \\")
    print("    --model mlx-community/SmolLM2-135M-Instruct \\")
    print("    --output ./quantized/smollm2-135m-4bit \\")
    print("    --method 4bit")

    print("\n# Quantize with GPTQ for best quality:")
    print("$ python -m smlx.tools.quantize \\")
    print("    --model mlx-community/SmolLM2-135M-Instruct \\")
    print("    --output ./quantized/smollm2-135m-gptq \\")
    print("    --method gptq \\")
    print("    --bits 4 \\")
    print("    --group-size 64")

    print("\n# Use the quantized model:")
    print("from smlx.models.SmolLM2_135M import load")
    print("model, tokenizer = load('./quantized/smollm2-135m-gptq')")

    print("\n   ✅ CLI tool provides offline quantization workflow")


def demonstrate_server_usage():
    """Show server integration usage (as documentation)."""
    print("\n" + "=" * 70)
    print("3. SERVER INTEGRATION")
    print("=" * 70)
    print("\nThe FastAPI server supports automatic quantization via environment variables.")

    print("\n# Start server with 4-bit quantization:")
    print("$ SMLX_AUTO_QUANTIZE=4bit python -m smlx.server.app")

    print("\n# Start with GPTQ quantization:")
    print("$ SMLX_AUTO_QUANTIZE=gptq \\")
    print("  SMLX_QUANTIZE_GROUP_SIZE=64 \\")
    print("  python -m smlx.server.app")

    print("\n# Start with automatic method selection:")
    print("$ SMLX_AUTO_QUANTIZE=auto python -m smlx.server.app")

    print("\n# Test with curl:")
    print("$ curl http://localhost:8000/v1/chat/completions \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{")
    print('    "model": "SmolLM2-135M",')
    print('    "messages": [{"role": "user", "content": "Hello!"}],')
    print('    "max_tokens": 50')
    print("  }'")

    print("\n# Or use the example client:")
    print("$ python examples/server/quantization_example.py")

    print("\n   ✅ Server provides automatic quantization for production")


def show_comparison_table():
    """Show comparison of all methods."""
    print("\n" + "=" * 70)
    print("QUANTIZATION METHOD COMPARISON")
    print("=" * 70)

    print("\n| Method | Memory  | Quality    | Speed  | Use Case                    |")
    print("|--------|---------|------------|--------|-----------------------------|")
    print("| FP16   | 1x      | Perfect    | Fast   | Development, small models   |")
    print("| 8-bit  | 2x less | Excellent  | Medium | Production, quality-focused |")
    print("| 4-bit  | 4x less | Good       | Fast   | Quick prototyping           |")
    print("| GPTQ   | 4x less | Excellent  | Medium | Production, best quality    |")
    print("| AWQ    | 4x less | Excellent  | Medium | Production, activation-aware|")
    print("| DWQ    | 4x less | Very Good  | Medium | Research, distillation      |")
    print("| Auto   | Varies  | Optimal    | Varies | Automatic hardware tuning   |")


def show_workflow_recommendations():
    """Show recommended workflows."""
    print("\n" + "=" * 70)
    print("RECOMMENDED WORKFLOWS")
    print("=" * 70)

    print("\n📋 Development Workflow:")
    print("1. Start with FP16 for debugging")
    print("2. Switch to 4-bit for faster iteration")
    print("3. Test with GPTQ/AWQ before production")

    print("\n🚀 Production Workflow:")
    print("1. Quantize models offline with CLI tool")
    print("2. Test quantized models locally")
    print("3. Deploy server with auto_quantize")
    print("4. Monitor memory usage with /memory endpoint")

    print("\n🔬 Research Workflow:")
    print("1. Use loader integration for experiments")
    print("2. Compare methods with benchmark suite")
    print("3. Use DWQ for knowledge distillation")
    print("4. Document results with BENCHMARKS.md format")


def main():
    """Main example entry point."""
    print("\n" + "=" * 80)
    print(" " * 20 + "SMLX QUANTIZATION INTEGRATION")
    print(" " * 18 + "Complete Usage Guide & Examples")
    print("=" * 80)

    print("\nThis example demonstrates all three ways to use quantization in SMLX:")
    print("  1. Direct Model Loader Integration (Python API)")
    print("  2. CLI Tool (Command-line)")
    print("  3. Server Integration (FastAPI)")

    # Demonstrate each pathway
    demonstrate_loader_integration()
    demonstrate_cli_usage()
    demonstrate_server_usage()

    # Show comparison and workflows
    show_comparison_table()
    show_workflow_recommendations()

    print("\n" + "=" * 80)
    print(" " * 30 + "INTEGRATION COMPLETE!")
    print("=" * 80)

    print("\n📚 Additional Resources:")
    print("  - docs/Quant.md - Comprehensive quantization guide")
    print("  - docs/BENCHMARKS.md - Performance benchmarks")
    print("  - QUANTIZATION_RESEARCH_REPORT.md - Research findings")
    print("  - QUANTIZATION_INTEGRATION_SUMMARY.md - Integration details")

    print("\n🎯 Quick Start:")
    print("  from smlx.models.SmolLM2_135M import load")
    print("  model, tokenizer = load(model_path, quantize='4bit')")

    print("\n💡 Pro Tips:")
    print("  - Use quantize='auto' for automatic hardware optimization")
    print("  - Use group_size=64 for M4 chipsets")
    print("  - Use GPTQ or AWQ for best quality/size tradeoff")
    print("  - Monitor memory with /memory endpoint in production")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
