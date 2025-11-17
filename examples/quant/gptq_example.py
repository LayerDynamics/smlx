#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
GPTQ Quantization Example

Demonstrates how to use GPTQ (GPT Quantization) to compress models
with minimal quality loss. GPTQ uses Hessian-based optimization for
accurate quantization.

GPTQ is ideal for:
- Post-training quantization
- Minimal quality degradation
- Memory-constrained environments (M4 with 36GB unified memory)
"""

import mlx.core as mx
import time

from smlx.models.SmolLM2_135M import load, generate
from smlx.quant import gptq_quantize


def main():
    print("=" * 70)
    print("GPTQ Quantization Example")
    print("=" * 70)

    # Load full-precision model
    print("\n1. Loading full-precision SmolLM2-135M...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Check original size
    original_params = sum(p.size for _, p in model.parameters().items() if hasattr(p, 'size'))
    print(f"   Original parameters: {original_params:,}")

    # Test prompt
    prompt = "Explain quantum computing in simple terms:"

    # Generate with original model
    print("\n2. Generating with full-precision model...")
    start = time.time()
    original_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        verbose=False,
    )
    original_time = time.time() - start
    print(f"   Time: {original_time:.2f}s")
    print(f"   Output: {original_output[:100]}...")

    # Quantize with GPTQ
    print("\n3. Quantizing with GPTQ (4-bit, group_size=64)...")
    print("   This may take a minute...")

    quantized_model = gptq_quantize(
        model=model,
        bits=4,  # 4-bit quantization
        group_size=64,  # Optimized for M4
    )

    print("   ✅ Quantization complete!")

    # Check quantized size (approximate - actual is ~1/4 of original)
    print(f"   Expected compression: ~4x (from FP16 to 4-bit)")
    print(f"   Estimated quantized size: ~{original_params // 4:,} effective params")

    # Generate with quantized model
    print("\n4. Generating with quantized model...")
    start = time.time()
    quantized_output = generate(
        model=quantized_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        verbose=False,
    )
    quantized_time = time.time() - start
    print(f"   Time: {quantized_time:.2f}s")
    print(f"   Output: {quantized_output[:100]}...")

    # Compare results
    print("\n5. Comparison:")
    print(f"   Speed improvement: {original_time / quantized_time:.2f}x faster")
    print(f"   Memory reduction: ~4x smaller")

    # Quality check (simple comparison)
    print("\n6. Quality Check:")
    print(f"   Original length: {len(original_output)} chars")
    print(f"   Quantized length: {len(quantized_output)} chars")

    # Test different bit widths
    print("\n7. Testing different quantization levels...")

    for bits in [8, 4, 2]:
        print(f"\n   {bits}-bit quantization:")
        q_model = gptq_quantize(model, bits=bits, group_size=64)

        start = time.time()
        output = generate(
            model=q_model,
            tokenizer=tokenizer,
            prompt="Hello world",
            max_tokens=20,
            temperature=0.0,  # Greedy for consistency
            verbose=False,
        )
        elapsed = time.time() - start

        print(f"     Time: {elapsed:.2f}s")
        print(f"     Output: {output}")
        print(f"     Compression: ~{16 // bits}x")

    print("\n" + "=" * 70)
    print("✅ GPTQ Example Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- 4-bit GPTQ provides excellent quality/size tradeoff")
    print("- group_size=64 is optimized for M4 chipsets")
    print("- Minimal quality degradation with 4x compression")
    print("- Faster inference due to reduced memory bandwidth")


if __name__ == "__main__":
    main()
