#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
LoRA (Low-Rank Adaptation) Example

Demonstrates parameter-efficient fine-tuning using LoRA. Instead of
updating all model weights, LoRA adds small trainable matrices that
adapt the model to new tasks.

LoRA is ideal for:
- Fine-tuning with limited resources
- Task-specific adaptations
- Quick experimentation
- Preserving base model weights
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from smlx.models.SmolLM2_135M import load, generate
from smlx.quant import apply_lora, merge_lora


def main():
    print("=" * 70)
    print("LoRA (Low-Rank Adaptation) Example")
    print("=" * 70)

    # Load base model
    print("\n1. Loading base model (SmolLM2-135M)...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Count parameters
    total_params = sum(p.size for _, p in model.parameters().items() if hasattr(p, 'size'))
    print(f"   Total parameters: {total_params:,}")

    # Test before LoRA
    test_prompt = "Complete this sentence: The quick brown fox"

    print("\n2. Generation BEFORE LoRA:")
    output_before = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=20,
        temperature=0.0,  # Greedy for consistency
        verbose=False,
    )
    print(f"   {output_before}")

    # Apply LoRA
    print("\n3. Applying LoRA adapters...")
    lora_rank = 8  # Low rank for efficiency
    lora_alpha = 16  # Scaling factor
    lora_dropout = 0.05

    lora_model = apply_lora(
        model=model,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        # Target specific layers (all linear layers by default)
    )

    # Count trainable parameters
    trainable_params = sum(
        p.size for name, p in lora_model.parameters().items()
        if 'lora' in name.lower() and hasattr(p, 'size')
    )

    print(f"   ✅ LoRA applied!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable LoRA parameters: {trainable_params:,}")
    print(f"   Trainable ratio: {trainable_params / total_params * 100:.2f}%")

    # Demonstrate LoRA efficiency
    print("\n4. LoRA Efficiency:")
    print(f"   Memory savings: {(1 - trainable_params / total_params) * 100:.1f}%")
    print(f"   Rank: {lora_rank} (controls LoRA capacity)")
    print(f"   Alpha: {lora_alpha} (scaling factor)")
    print("   Only LoRA weights need gradients during training!")

    # Show different LoRA configurations
    print("\n5. LoRA Configuration Comparison:")

    configs = [
        {"rank": 4, "alpha": 8, "name": "Low capacity (fast)"},
        {"rank": 8, "alpha": 16, "name": "Balanced"},
        {"rank": 16, "alpha": 32, "name": "High capacity"},
    ]

    for config in configs:
        lora_test = apply_lora(
            model=model,
            rank=config["rank"],
            alpha=config["alpha"],
            dropout=0.05,
        )

        trainable = sum(
            p.size for name, p in lora_test.parameters().items()
            if 'lora' in name.lower() and hasattr(p, 'size')
        )

        print(f"\n   {config['name']}:")
        print(f"     Rank: {config['rank']}, Alpha: {config['alpha']}")
        print(f"     Trainable params: {trainable:,}")
        print(f"     Ratio: {trainable / total_params * 100:.3f}%")

    # Merge LoRA weights back into base model
    print("\n6. Merging LoRA weights...")
    print("   After training, LoRA weights can be merged back into base model")
    print("   This creates a single model with no inference overhead")

    merged_model = merge_lora(lora_model)
    print("   ✅ Weights merged!")

    # Test after merge
    print("\n7. Generation AFTER LoRA merge:")
    output_after = generate(
        model=merged_model,
        tokenizer=tokenizer,
        prompt=test_prompt,
        max_tokens=20,
        temperature=0.0,
        verbose=False,
    )
    print(f"   {output_after}")

    print("\n" + "=" * 70)
    print("✅ LoRA Example Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- LoRA adds ~0.1-1% trainable parameters")
    print("- Full model weights remain frozen")
    print("- Multiple LoRA adapters can share one base model")
    print("- Rank controls capacity (higher = more expressive)")
    print("- Alpha controls scaling (typically 2x rank)")
    print("- Can merge back into base model for deployment")

    print("\n💡 Use Cases:")
    print("- Fine-tune for specific tasks (coding, summarization, etc.)")
    print("- Create domain-specific variants (medical, legal, etc.)")
    print("- Rapid experimentation with limited compute")
    print("- Multi-tenant serving (one base model, many LoRAs)")


if __name__ == "__main__":
    main()
