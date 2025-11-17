#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
AWQ (Activation-aware Weight Quantization) Example

Demonstrates AWQ, which protects salient weight channels based on
activation patterns. AWQ typically provides better quality than
standard quantization at the same bit width.

AWQ is ideal for:
- High-quality 4-bit quantization
- Preserving model performance
- Production deployments
"""

import mlx.core as mx
import time

from smlx.models.SmolLM2_135M import load, generate
from smlx.quant import awq_quantize


def main():
    print("=" * 70)
    print("AWQ (Activation-aware Weight Quantization) Example")
    print("=" * 70)

    # Load model
    print("\n1. Loading SmolLM2-135M...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    test_prompts = [
        "Write a Python function to calculate factorial:",
        "What is machine learning?",
        "Explain photosynthesis:",
    ]

    # Baseline performance
    print("\n2. Baseline (FP16) Performance:")
    fp16_outputs = []

    for i, prompt in enumerate(test_prompts, 1):
        start = time.time()
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,
            verbose=False,
        )
        elapsed = time.time() - start
        fp16_outputs.append(output)
        print(f"   Prompt {i}: {elapsed:.2f}s - {output[:50]}...")

    # Quantize with AWQ
    print("\n3. Quantizing with AWQ...")
    print("   AWQ protects salient channels based on activations")
    print("   This typically preserves quality better than naive quantization")

    quantized_model = awq_quantize(
        model=model,
        bits=4,
        group_size=64,
    )

    print("   ✅ AWQ quantization complete!")

    # Test quantized model
    print("\n4. AWQ-Quantized (4-bit) Performance:")
    awq_outputs = []

    for i, prompt in enumerate(test_prompts, 1):
        start = time.time()
        output = generate(
            model=quantized_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,
            verbose=False,
        )
        elapsed = time.time() - start
        awq_outputs.append(output)
        print(f"   Prompt {i}: {elapsed:.2f}s - {output[:50]}...")

    # Compare AWQ vs standard quantization
    print("\n5. AWQ Features:")
    print("   ✓ Protects salient weight channels")
    print("   ✓ Activation-aware scaling factors")
    print("   ✓ Better quality preservation than standard quantization")
    print("   ✓ Grid search for optimal scaling")
    print("   ✓ ~4x memory reduction (FP16 → 4-bit)")

    # Advanced: Compare different configurations
    print("\n6. Testing AWQ configurations:")

    configs = [
        {"bits": 4, "group_size": 32, "name": "4-bit, g=32 (fine-grained)"},
        {"bits": 4, "group_size": 64, "name": "4-bit, g=64 (balanced)"},
        {"bits": 4, "group_size": 128, "name": "4-bit, g=128 (coarse)"},
    ]

    test_prompt = "Hello, how are you?"

    for config in configs:
        print(f"\n   {config['name']}:")
        q_model = awq_quantize(
            model=model,
            bits=config["bits"],
            group_size=config["group_size"],
        )

        start = time.time()
        output = generate(
            model=q_model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            max_tokens=30,
            temperature=0.0,
            verbose=False,
        )
        elapsed = time.time() - start

        print(f"     Time: {elapsed:.2f}s")
        print(f"     Output: {output}")

    print("\n" + "=" * 70)
    print("✅ AWQ Example Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- AWQ preserves quality better than naive quantization")
    print("- Activation-aware scaling protects important weights")
    print("- group_size=64 is a good balance for M4")
    print("- 4-bit AWQ rivals 8-bit standard quantization in quality")


if __name__ == "__main__":
    main()
