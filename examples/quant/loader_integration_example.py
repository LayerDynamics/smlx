#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Model Loader Quantization Integration Example

Demonstrates the new quantize= parameter in model loaders.
Shows how to load models with automatic quantization in a single line.
"""

import time

from smlx.models.SmolLM2_135M import generate, load


def main():
    print("=" * 70)
    print("Model Loader Quantization Integration Example")
    print("=" * 70)

    test_prompt = "Explain quantum computing in simple terms:"

    # Example 1: Load without quantization (baseline)
    print("\n1. Loading without quantization (FP16 baseline)...")
    model_fp16, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    start = time.time()
    output_fp16 = generate(
        model=model_fp16,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=50,
        temperature=0.7,
        verbose=False,
    )
    time_fp16 = time.time() - start
    print(f"   Time: {time_fp16:.2f}s")
    print(f"   Output: {output_fp16[:80]}...")

    # Example 2: Load with 4-bit quantization
    print("\n2. Loading with 4-bit quantization (single line)...")
    model_4bit, tokenizer = load(
        "mlx-community/SmolLM2-135M-Instruct", quantize="4bit"
    )

    start = time.time()
    output_4bit = generate(
        model=model_4bit,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=50,
        temperature=0.7,
        verbose=False,
    )
    time_4bit = time.time() - start
    print(f"   Time: {time_4bit:.2f}s")
    print(f"   Output: {output_4bit[:80]}...")

    # Example 3: Load with GPTQ quantization
    print("\n3. Loading with GPTQ quantization (high quality)...")
    model_gptq, tokenizer = load(
        "mlx-community/SmolLM2-135M-Instruct", quantize="gptq"
    )

    start = time.time()
    output_gptq = generate(
        model=model_gptq,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=50,
        temperature=0.7,
        verbose=False,
    )
    time_gptq = time.time() - start
    print(f"   Time: {time_gptq:.2f}s")
    print(f"   Output: {output_gptq[:80]}...")

    # Example 4: Load with AWQ quantization
    print("\n4. Loading with AWQ quantization (activation-aware)...")
    model_awq, tokenizer = load(
        "mlx-community/SmolLM2-135M-Instruct", quantize="awq"
    )

    start = time.time()
    output_awq = generate(
        model=model_awq,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=50,
        temperature=0.7,
        verbose=False,
    )
    time_awq = time.time() - start
    print(f"   Time: {time_awq:.2f}s")
    print(f"   Output: {output_awq[:80]}...")

    # Example 5: Load with custom quantization config
    print("\n5. Loading with custom quantization config...")
    model_custom, tokenizer = load(
        "mlx-community/SmolLM2-135M-Instruct",
        quantize="gptq",
        quantization_config={"bits": 4, "group_size": 128},
    )

    start = time.time()
    output_custom = generate(
        model=model_custom,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=50,
        temperature=0.7,
        verbose=False,
    )
    time_custom = time.time() - start
    print(f"   Time: {time_custom:.2f}s")
    print(f"   Output: {output_custom[:80]}...")

    # Example 6: Automatic quantization selection
    print("\n6. Loading with automatic quantization selection...")
    model_auto, tokenizer = load(
        "mlx-community/SmolLM2-135M-Instruct", quantize="auto"
    )

    start = time.time()
    output_auto = generate(
        model=model_auto,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=50,
        temperature=0.7,
        verbose=False,
    )
    time_auto = time.time() - start
    print(f"   Time: {time_auto:.2f}s")
    print(f"   Output: {output_auto[:80]}...")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"FP16 (baseline):      {time_fp16:.2f}s")
    print(f"4-bit:                {time_4bit:.2f}s ({time_fp16/time_4bit:.2f}x)")
    print(f"GPTQ:                 {time_gptq:.2f}s ({time_fp16/time_gptq:.2f}x)")
    print(f"AWQ:                  {time_awq:.2f}s ({time_fp16/time_awq:.2f}x)")
    print(f"Custom (GPTQ g=128):  {time_custom:.2f}s ({time_fp16/time_custom:.2f}x)")
    print(f"Auto:                 {time_auto:.2f}s ({time_fp16/time_auto:.2f}x)")

    print("\n" + "=" * 70)
    print("Key Features:")
    print("=" * 70)
    print("✓ Single-line quantization with quantize= parameter")
    print("✓ Multiple presets: '4bit', '8bit', 'gptq', 'awq', 'dwq', 'auto'")
    print("✓ Custom config with quantization_config= dict")
    print("✓ Automatic strategy selection with quantize='auto'")
    print("✓ ~4x memory reduction with minimal quality loss")


if __name__ == "__main__":
    main()
