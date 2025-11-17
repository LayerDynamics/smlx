#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
QLoRA (Quantized Low-Rank Adaptation) Example

Combines quantization with LoRA for maximum efficiency.
Base model is quantized (4-bit) while LoRA adapters remain in full precision.

QLoRA is ideal for:
- Fine-tuning large models on consumer hardware
- Extreme memory efficiency
- Production fine-tuning on edge devices (M4 Macs)
"""

import mlx.core as mx
import time

from smlx.models.SmolLM2_135M import load, generate
from smlx.quant import quantize_gptq, apply_lora, merge_lora


def main():
    print("=" * 70)
    print("QLoRA (Quantized + LoRA) Example")
    print("=" * 70)
    print("Combines 4-bit quantization with LoRA for maximum efficiency")

    # Load model
    print("\n1. Loading SmolLM2-135M...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    original_params = sum(p.size for _, p in model.parameters().items() if hasattr(p, 'size'))
    print(f"   Original parameters: {original_params:,}")

    # Test prompt
    test_prompt = "Explain what QLoRA is:"

    # Baseline
    print("\n2. Baseline (FP16) generation:")
    start = time.time()
    fp16_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=50,
        temperature=0.7,
        verbose=False,
    )
    fp16_time = time.time() - start
    print(f"   Time: {fp16_time:.2f}s")
    print(f"   Output: {fp16_output[:80]}...")

    # Step 1: Quantize base model
    print("\n3. Step 1: Quantizing base model to 4-bit...")
    quantized_model = quantize_gptq(
        model=model,
        bits=4,
        group_size=64,
    )
    print("   ✅ Base model quantized!")
    print(f"   Memory: ~{original_params // 4:,} effective params (4x reduction)")

    # Step 2: Add LoRA adapters
    print("\n4. Step 2: Adding LoRA adapters...")
    qlora_model = apply_lora(
        model=quantized_model,
        rank=8,
        alpha=16,
        dropout=0.05,
    )

    trainable_params = sum(
        p.size for name, p in qlora_model.parameters().items()
        if 'lora' in name.lower() and hasattr(p, 'size')
    )

    print("   ✅ LoRA adapters added!")
    print(f"   Trainable LoRA params: {trainable_params:,}")
    print(f"   Trainable ratio: {trainable_params / original_params * 100:.2f}%")

    # QLoRA generation
    print("\n5. QLoRA generation:")
    start = time.time()
    qlora_output = generate(
        model=qlora_model,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=50,
        temperature=0.7,
        verbose=False,
    )
    qlora_time = time.time() - start
    print(f"   Time: {qlora_time:.2f}s")
    print(f"   Output: {qlora_output[:80]}...")

    # Compare configurations
    print("\n6. Memory Comparison:")
    print(f"   FP16 baseline: ~{original_params * 2 / 1e6:.1f} MB")
    print(f"   4-bit quantized: ~{original_params * 0.5 / 1e6:.1f} MB (4x smaller)")
    print(f"   QLoRA (4-bit + LoRA): ~{(original_params * 0.5 + trainable_params * 2) / 1e6:.1f} MB")
    print(f"   Memory savings: ~{(1 - (original_params * 0.5 + trainable_params * 2) / (original_params * 2)) * 100:.1f}%")

    # Speed comparison
    print("\n7. Speed Comparison:")
    print(f"   FP16: {fp16_time:.2f}s")
    print(f"   QLoRA: {qlora_time:.2f}s ({qlora_time / fp16_time:.2f}x)")

    # Show QLoRA advantages
    print("\n8. QLoRA Advantages:")
    print("   ✓ 4x memory reduction from quantization")
    print("   ✓ Only ~0.5% parameters trainable (LoRA)")
    print("   ✓ Base model frozen and compressed")
    print("   ✓ LoRA adapters in full precision for training stability")
    print("   ✓ Enables fine-tuning on consumer hardware")
    print("   ✓ Perfect for M4 Macs with unified memory")

    # Real-world scenario
    print("\n9. Real-world Example: Fine-tuning for Code Generation")
    print("   Scenario: Adapt SmolLM2 for Python code completion")

    code_prompt = "def fibonacci(n):"

    # Before fine-tuning (just using quantized model)
    print("\n   Before fine-tuning:")
    before_output = generate(
        model=quantized_model,
        tokenizer=tokenizer,
        prompt=code_prompt,
        max_tokens=80,
        temperature=0.3,
        verbose=False,
    )
    print(f"   {before_output}")

    # With QLoRA (simulating post-training)
    print("\n   With QLoRA adapters:")
    print("   (In practice, you would train LoRA on code dataset)")
    after_output = generate(
        model=qlora_model,
        tokenizer=tokenizer,
        prompt=code_prompt,
        max_tokens=80,
        temperature=0.3,
        verbose=False,
    )
    print(f"   {after_output}")

    print("\n" + "=" * 70)
    print("✅ QLoRA Example Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- QLoRA = 4-bit quantized base + FP16 LoRA adapters")
    print("- ~75% memory reduction vs FP16")
    print("- Only ~0.5% parameters need training")
    print("- Enables fine-tuning on 8GB+ RAM devices")
    print("- Ideal for M4 Macs (unified memory, Metal acceleration)")

    print("\n💡 Training Workflow:")
    print("1. Quantize base model to 4-bit (GPTQ or AWQ)")
    print("2. Add LoRA adapters (rank=8-16 typical)")
    print("3. Train only LoRA weights on task-specific data")
    print("4. Merge or deploy with LoRA adapters")
    print("5. Share tiny adapter files (~MB) instead of full model (~GB)")


if __name__ == "__main__":
    main()
