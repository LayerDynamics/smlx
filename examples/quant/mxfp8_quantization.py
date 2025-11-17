#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
MXFP8 Quantization Example

Demonstrates OCP Microscaling FP8 quantization for Apple Silicon.

MXFP8 is the recommended 8-bit floating-point format:
- True 8-bit storage (unlike simulated FP8)
- OCP industry standard
- 2x memory reduction vs FP16
- Software emulated on M4 (no native FP8 hardware)

This example shows:
1. Basic MXFP8 quantization usage
2. Comparison with INT8 and FP16
3. M4-specific performance characteristics
4. When to use MXFP8 vs INT8 on Apple M4
"""

import mlx.core as mx
import time
from typing import Dict, Any

from smlx.models.SmolLM2_135M import load
from smlx.models.SmolLM2_135M.generate import generate
from smlx.quant import quantize_to_mxfp8, dequantize_from_mxfp8
from smlx.utils.quantization import apply_quantization, estimate_quantized_size


def format_bytes(bytes_val: float) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def benchmark_quantized_model(
    model,
    tokenizer,
    method: str,
    test_prompts: list[str]
) -> Dict[str, Any]:
    """
    Benchmark a quantization method.

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        method: Quantization method name
        test_prompts: List of test prompts

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n📊 Benchmarking {method.upper()}...")

    # Estimate model size
    model_size = estimate_quantized_size(model, method=method)
    print(f"  Model size: {format_bytes(model_size)}")

    # Warm up
    _ = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="warm up",
        max_tokens=10,
        temperature=0.0,
        verbose=False,
    )

    # Benchmark
    total_tokens = 0
    total_time = 0

    for prompt in test_prompts:
        start = time.perf_counter()
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,
            verbose=False,
        )
        elapsed = time.perf_counter() - start

        # Count generated tokens
        output_tokens = tokenizer.encode(output)
        prompt_tokens = tokenizer.encode(prompt)
        generated = len(output_tokens) - len(prompt_tokens)

        total_tokens += generated
        total_time += elapsed

    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

    print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
    print(f"  Avg latency: {(total_time / len(test_prompts)) * 1000:.1f} ms")

    return {
        "method": method,
        "model_size_bytes": model_size,
        "tokens_per_sec": tokens_per_sec,
        "avg_latency_ms": (total_time / len(test_prompts)) * 1000,
    }


def example_basic_mxfp8():
    """Example 1: Basic MXFP8 quantization."""
    print("\n" + "=" * 70)
    print("Example 1: Basic MXFP8 Quantization")
    print("=" * 70)

    # Create sample weights
    print("\n1. Quantizing sample weight tensor...")
    weights = mx.random.normal((768, 768))
    print(f"   Original shape: {weights.shape}")
    print(f"   Original dtype: {weights.dtype}")
    print(f"   Original size: {format_bytes(weights.nbytes)}")

    # Quantize to MXFP8
    w_quantized, scales = quantize_to_mxfp8(weights)
    print(f"\n   Quantized dtype: {w_quantized.dtype} (uint8)")
    print(f"   Quantized size: {format_bytes(w_quantized.nbytes)}")
    print(f"   Scale dtype: {scales.dtype} (uint8)")
    print(f"   Scale size: {format_bytes(scales.nbytes)}")
    print(f"   Total size: {format_bytes(w_quantized.nbytes + scales.nbytes)}")
    print(f"   Compression: {weights.nbytes / (w_quantized.nbytes + scales.nbytes):.2f}x")

    # Dequantize
    print("\n2. Dequantizing...")
    w_restored = dequantize_from_mxfp8(w_quantized, scales)
    print(f"   Restored dtype: {w_restored.dtype}")
    print(f"   Restored shape: {w_restored.shape}")

    # Check error
    error = mx.mean(mx.abs(w_restored - weights))
    max_error = mx.max(mx.abs(w_restored - weights))
    print(f"\n3. Quantization error:")
    print(f"   Mean absolute error: {float(error):.6f}")
    print(f"   Max absolute error: {float(max_error):.6f}")

    # Show format details
    print("\n4. MXFP8 Format Details:")
    print("   Element format: E4M3 (4-bit exponent, 3-bit mantissa)")
    print("   Scale format: E8M0 (8-bit exponent-only)")
    print("   Block size: 32 elements (fixed by OCP spec)")
    print("   Storage: uint8 (true 8-bit, not simulated)")
    print("   Standard: OCP Microscaling Formats v1.0")


def example_model_quantization():
    """Example 2: Quantize a full model with MXFP8."""
    print("\n" + "=" * 70)
    print("Example 2: Model-Level MXFP8 Quantization")
    print("=" * 70)

    print("\n1. Loading model...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Show original size
    original_size = estimate_quantized_size(model, method="fp16")
    print(f"   Original size (FP16): {format_bytes(original_size)}")

    # Apply MXFP8 quantization
    print("\n2. Applying MXFP8 quantization...")
    model_mxfp8 = apply_quantization(model, method="mxfp8", verbose=True)

    # Show quantized size
    quantized_size = estimate_quantized_size(model_mxfp8, method="mxfp8")
    print(f"   Quantized size (MXFP8): {format_bytes(quantized_size)}")
    print(f"   Reduction: {original_size / quantized_size:.2f}x")

    # Test generation
    print("\n3. Testing generation...")
    test_prompt = "The future of AI is"
    output = generate(
        model=model_mxfp8,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=30,
        temperature=0.7,
        verbose=False,
    )
    print(f"   Prompt: {test_prompt}")
    print(f"   Output: {output}")


def example_mxfp8_vs_int8():
    """Example 3: Compare MXFP8 vs INT8 on M4."""
    print("\n" + "=" * 70)
    print("Example 3: MXFP8 vs INT8 Comparison (Apple M4)")
    print("=" * 70)

    print("\nThis example compares MXFP8 and INT8 quantization on Apple M4.")
    print("Key question: Which is better for your use case?")

    # Test prompts
    test_prompts = [
        "Explain machine learning in simple terms:",
        "Write a Python function to reverse a string:",
        "What is the capital of France?",
    ]

    # Load base model
    print("\n1. Loading base model...")
    model_fp16, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Benchmark FP16 (baseline)
    results_fp16 = benchmark_quantized_model(
        model=model_fp16,
        tokenizer=tokenizer,
        method="fp16",
        test_prompts=test_prompts,
    )

    # Load and benchmark MXFP8
    print("\n2. Testing MXFP8...")
    model_mxfp8, _ = load("mlx-community/SmolLM2-135M-Instruct")
    model_mxfp8 = apply_quantization(model_mxfp8, method="mxfp8", verbose=False)
    results_mxfp8 = benchmark_quantized_model(
        model=model_mxfp8,
        tokenizer=tokenizer,
        method="mxfp8",
        test_prompts=test_prompts,
    )

    # Load and benchmark INT8
    print("\n3. Testing INT8...")
    model_int8, _ = load("mlx-community/SmolLM2-135M-Instruct")
    model_int8 = apply_quantization(model_int8, method="int8", verbose=False)
    results_int8 = benchmark_quantized_model(
        model=model_int8,
        tokenizer=tokenizer,
        method="int8",
        test_prompts=test_prompts,
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Method':<10} {'Size':<15} {'Speed (tok/s)':<15} {'vs FP16':<12}")
    print("-" * 70)

    for result in [results_fp16, results_mxfp8, results_int8]:
        size_str = format_bytes(result["model_size_bytes"])
        speedup = result["tokens_per_sec"] / results_fp16["tokens_per_sec"]
        print(
            f"{result['method']:<10} "
            f"{size_str:<15} "
            f"{result['tokens_per_sec']:<15.1f} "
            f"{speedup:.2f}x"
        )

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS FOR APPLE M4")
    print("=" * 70)

    print("\nMXFP8 (OCP Microscaling):")
    print("  ✅ True 8-bit storage (2x memory reduction)")
    print("  ✅ Lower scale overhead (8-bit vs 32-bit for INT8)")
    print("  ✅ OCP industry standard (portable)")
    print("  ⚠️  Software emulated on M4 (no native FP8 hardware)")
    print(f"  📊 Performance: {results_mxfp8['tokens_per_sec']:.1f} tok/s")

    print("\nINT8 (Symmetric):")
    print("  ✅ Native M4 hardware acceleration (AMX + GPU)")
    print("  ✅ Proven quality with GPTQ/AWQ")
    print("  ✅ Flexible group sizes")
    print("  ⚠️  Higher scale overhead (32-bit scales)")
    print(f"  📊 Performance: {results_int8['tokens_per_sec']:.1f} tok/s")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    int8_faster = results_int8['tokens_per_sec'] > results_mxfp8['tokens_per_sec']
    speedup = results_int8['tokens_per_sec'] / results_mxfp8['tokens_per_sec']

    if int8_faster and speedup > 1.1:
        print("\n✅ Use INT8 for speed-critical inference on M4")
        print(f"   INT8 is {speedup:.1f}x faster due to native hardware acceleration")
    elif results_mxfp8["model_size_bytes"] < results_int8["model_size_bytes"] * 0.9:
        reduction = (1 - results_mxfp8["model_size_bytes"] / results_int8["model_size_bytes"]) * 100
        print("\n✅ Use MXFP8 for memory-constrained scenarios")
        print(f"   MXFP8 saves {reduction:.0f}% memory vs INT8")
    else:
        print("\n✅ Both methods perform similarly on M4")
        print("   • Use INT8 for maximum speed")
        print("   • Use MXFP8 for OCP standard compatibility")

    print("\nOther considerations:")
    print("  • Training/fine-tuning: Prefer MXFP8 (better gradient flow)")
    print("  • Export/portability: Prefer MXFP8 (OCP standard)")
    print("  • Wide dynamic range: Prefer MXFP8 (exponential spacing)")
    print("  • Maximum speed: Prefer INT8 (M4 native acceleration)")


def example_hardware_info():
    """Example 4: Display M4 hardware capabilities."""
    print("\n" + "=" * 70)
    print("Example 4: Apple M4 Hardware Capabilities")
    print("=" * 70)

    print("\nSupported Data Types on M4:")
    print("\n  Data Type    CPU    GPU    AMX    Neural Engine")
    print("  " + "-" * 56)
    print("  FP64         ✅     ❌     ❌     ❌")
    print("  FP32         ✅     ✅     ❌     ❌")
    print("  FP16         ✅     ✅     ✅     ✅")
    print("  BF16         ✅     ✅     ✅     ❌")
    print("  INT8         ✅     ✅     ✅     ✅")
    print("  FP8          ❌     ❌     ❌     ❌     (NOT supported)")
    print("  MXFP8        ⚠️      ⚠️      ❌     ❌     (Software emulated)")

    print("\nImplications for Quantization:")
    print("  • FP8 is NOT natively supported on M4")
    print("  • MXFP8 runs via Metal software emulation")
    print("  • INT8 has full hardware acceleration (fastest)")
    print("  • MXFP8 still provides true 8-bit storage (memory savings)")

    print("\nFuture Hardware:")
    print("  • Apple M5+ may include native FP8 support")
    print("  • Native FP8 would eliminate emulation overhead")
    print("  • MXFP8 code will work on future hardware (portable)")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MXFP8 QUANTIZATION EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate OCP Microscaling FP8 quantization")
    print("for Apple Silicon (M4).")

    # Run examples
    example_basic_mxfp8()
    example_model_quantization()
    example_mxfp8_vs_int8()
    example_hardware_info()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. MXFP8 provides TRUE 8-bit storage (2x memory reduction)")
    print("  2. Use MXFP8 for memory savings or OCP standard compatibility")
    print("  3. Use INT8 for maximum speed on M4 (native acceleration)")
    print("  4. Avoid simulated FP8 (deprecated, no benefits)")
    print("\nFor more details:")
    print("  • See docs/Quant.md for complete quantization guide")
    print("  • See docs/BENCHMARKS.md for M4-specific performance data")
    print("=" * 70)


if __name__ == "__main__":
    main()
