#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Text Evaluation Example

Demonstrates how to evaluate language models on standard benchmarks
using perplexity metrics on datasets like WikiText.
"""

import mlx.core as mx

from smlx.models.SmolLM2_135M import load
from smlx.evals.text_eval import evaluate_perplexity, load_eval_dataset, list_datasets


def main():
    print("=" * 70)
    print("Text Model Evaluation Example")
    print("=" * 70)

    # List available datasets
    print("\n1. Available evaluation datasets:")
    datasets = list_datasets()
    for ds in datasets:
        print(f"   - {ds}")

    # Load model
    print("\n2. Loading SmolLM2-135M...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
    print("   ✅ Model loaded")

    # Evaluate on WikiText-2 (smaller dataset)
    print("\n3. Evaluating on WikiText-2...")
    print("   This measures how well the model predicts text")

    results = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_name="wikitext",
        split="test",
        batch_size=4,
        num_samples=100,  # Use subset for faster evaluation
    )

    print("\n   Results:")
    print(f"   Perplexity: {results['perplexity']:.2f}")
    print(f"   Tokens/sec: {results['tokens_per_second']:.1f}")
    print(f"   Total samples: {results['num_samples']}")

    # Explanation
    print("\n4. Understanding Perplexity:")
    print("   - Lower is better (indicates better predictions)")
    print("   - Perplexity ~20-30: Excellent for small models")
    print("   - Perplexity ~30-50: Good")
    print("   - Perplexity >50: Needs improvement")

    # Load custom dataset
    print("\n5. Loading dataset directly:")
    dataset = load_eval_dataset("wikitext", tokenizer, split="test", num_samples=10)

    print(f"   Loaded {len(dataset)} samples")
    print(f"   First sample: {dataset[0]['text'][:100]}...")

    # Compare different models
    print("\n6. Comparing model variants:")
    print("   You can compare:")
    print("   - Different model sizes (135M vs 360M)")
    print("   - Quantized vs full precision")
    print("   - Fine-tuned vs base models")

    print("\n" + "=" * 70)
    print("✅ Text Evaluation Complete!")
    print("=" * 70)

    print("\n💡 Next Steps:")
    print("- Evaluate on full dataset (remove num_samples limit)")
    print("- Try different datasets (wikitext103, ptb)")
    print("- Compare quantized vs full precision")
    print("- Track perplexity during fine-tuning")


if __name__ == "__main__":
    main()
