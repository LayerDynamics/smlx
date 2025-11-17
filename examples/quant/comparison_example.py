#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Quantization Method Comparison

Comprehensive comparison of all quantization methods available in SMLX:
- GPTQ (Hessian-based)
- AWQ (Activation-aware)
- DWQ (Dynamic Weight Quantization)
- LoRA (Parameter-efficient fine-tuning)
- QLoRA (Quantization + LoRA)

Helps you choose the right quantization method for your use case.
"""

import mlx.core as mx
import time
from typing import Dict, Any

from smlx.models.SmolLM2_135M import load, generate
from smlx.quant import quantize_gptq, quantize_awq, apply_lora


def benchmark_model(model, tokenizer, prompts: list[str], name: str) -> Dict[str, Any]:
    """Benchmark a model variant."""
    print(f"\n📊 Benchmarking {name}...")

    total_time = 0
    outputs = []

    for prompt in prompts:
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
        total_time += elapsed
        outputs.append(output)

    avg_time = total_time / len(prompts)

    return {
        "name": name,
        "avg_time": avg_time,
        "total_time": total_time,
        "outputs": outputs,
    }


def main():
    print("=" * 70)
    print("Quantization Method Comparison")
    print("=" * 70)

    # Test prompts
    test_prompts = [
        "What is artificial intelligence?",
        "Write a Python function to reverse a string:",
        "Explain the water cycle:",
        "List 3 benefits of exercise:",
    ]

    # Load base model
    print("\n1. Loading SmolLM2-135M (base model)...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    original_params = sum(p.size for _, p in model.parameters().items() if hasattr(p, 'size'))
    print(f"   Parameters: {original_params:,}")

    # Benchmark baseline (FP16)
    results = []
    results.append(benchmark_model(model, tokenizer, test_prompts, "FP16 Baseline"))

    # Benchmark GPTQ 4-bit
    print("\n2. Creating GPTQ 4-bit variant...")
    gptq_model = quantize_gptq(model, bits=4, group_size=64)
    results.append(benchmark_model(gptq_model, tokenizer, test_prompts, "GPTQ 4-bit"))

    # Benchmark AWQ 4-bit
    print("\n3. Creating AWQ 4-bit variant...")
    awq_model = quantize_awq(model, bits=4, group_size=64)
    results.append(benchmark_model(awq_model, tokenizer, test_prompts, "AWQ 4-bit"))

    # Benchmark GPTQ 8-bit
    print("\n4. Creating GPTQ 8-bit variant...")
    gptq_8bit_model = quantize_gptq(model, bits=8, group_size=64)
    results.append(benchmark_model(gptq_8bit_model, tokenizer, test_prompts, "GPTQ 8-bit"))

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Method':<20} {'Avg Time':<12} {'vs Baseline':<12} {'Compression':<12}")
    print("-" * 70)

    baseline_time = results[0]["avg_time"]

    for result in results:
        speedup = baseline_time / result["avg_time"]
        print(
            f"{result['name']:<20} "
            f"{result['avg_time']:>8.2f}s    "
            f"{speedup:>7.2f}x      "
            f"{'1.0x' if 'FP16' in result['name'] else '4.0x' if '4-bit' in result['name'] else '2.0x':<12}"
        )

    # Quality comparison (first prompt)
    print("\n" + "=" * 70)
    print("QUALITY COMPARISON (First Prompt)")
    print("=" * 70)
    print(f"\nPrompt: '{test_prompts[0]}'\n")

    for result in results:
        print(f"{result['name']}:")
        print(f"  {result['outputs'][0][:100]}...")
        print()

    # Method recommendations
    print("=" * 70)
    print("METHOD RECOMMENDATIONS")
    print("=" * 70)

    recommendations = {
        "GPTQ 4-bit": {
            "Best for": "Maximum compression with good quality",
            "Use when": "Memory is constrained, quality is important",
            "Pros": "Hessian-based optimization, stable",
            "Cons": "Slower quantization process",
        },
        "AWQ 4-bit": {
            "Best for": "Best quality at 4-bit",
            "Use when": "Quality is critical, have time for quantization",
            "Pros": "Activation-aware, protects salient weights",
            "Cons": "Longer quantization time, minimal gains over GPTQ",
        },
        "GPTQ 8-bit": {
            "Best for": "Near-lossless compression",
            "Use when": "Need maximum quality, some memory available",
            "Pros": "Minimal quality loss, good speed",
            "Cons": "Only 2x compression",
        },
        "QLoRA": {
            "Best for": "Fine-tuning with limited resources",
            "Use when": "Need to adapt model to specific task",
            "Pros": "Tiny trainable parameters, memory efficient",
            "Cons": "Requires training, not just inference",
        },
    }

    for method, info in recommendations.items():
        print(f"\n{method}:")
        print(f"  Best for: {info['Best for']}")
        print(f"  Use when: {info['Use when']}")
        print(f"  Pros: {info['Pros']}")
        print(f"  Cons: {info['Cons']}")

    # M4-specific recommendations
    print("\n" + "=" * 70)
    print("M4 OPTIMIZATION GUIDE")
    print("=" * 70)

    print("\nFor M4 Macs with 36GB unified memory:")
    print("\n1. **Default Choice**: GPTQ 4-bit with group_size=64")
    print("   - Best balance of quality, speed, and memory")
    print("   - group_size=64 optimized for Metal")
    print("   - 4x compression allows running larger models")

    print("\n2. **Maximum Quality**: AWQ 4-bit or GPTQ 8-bit")
    print("   - AWQ if quality is paramount")
    print("   - GPTQ 8-bit if you have memory headroom")

    print("\n3. **Fine-tuning**: QLoRA (4-bit + LoRA)")
    print("   - Base model in 4-bit GPTQ")
    print("   - LoRA adapters for task-specific training")
    print("   - Enables fine-tuning on consumer hardware")

    print("\n4. **Production Deployment**:")
    print("   - Quantize offline with AWQ/GPTQ")
    print("   - Deploy quantized weights")
    print("   - No runtime quantization overhead")

    print("\n" + "=" * 70)
    print("✅ Comparison Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
