#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
AWQ (Activation-aware Weight Quantization) Example

Demonstrates AWQ, which protects salient weight channels based on
activation patterns. AWQ typically provides better quality than
standard quantization at the same bit width.

This example includes comprehensive quality benchmarking to measure
quality degradation from quantization.

AWQ is ideal for:
- High-quality 4-bit quantization
- Preserving model performance
- Production deployments
"""

import time

import mlx.core as mx

from smlx.models.SmolLM2_135M import generate, load
from smlx.quant import awq_quantize
from smlx.quant.utils import load_calibration_data
from smlx.utils.quality_metrics import assess_quality, calculate_perplexity, compare_quality
from smlx.utils.validation import validate_text_output

# Held-out reference text scored by every model variant so AWQ's language-modeling
# impact is measured on identical tokens, independent of each model's generations.
REFERENCE_TEXT = (
    "Activation-aware weight quantization protects the most salient channels so a "
    "compressed model keeps predicting natural language with low perplexity."
)


def main():
    print("=" * 70)
    print("AWQ (Activation-aware Weight Quantization) Example")
    print("=" * 70)

    # Load model
    print("\n1. Loading SmolLM2-135M...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Load calibration data for AWQ
    print("\n2. Loading calibration data...")
    calibration_data = load_calibration_data(
        tokenizer=tokenizer,
        num_samples=128,
        sequence_length=512,
        verbose=True
    )
    print(f"   ✅ Loaded calibration data: {calibration_data.shape}")

    test_prompts = [
        "Write a Python function to calculate factorial:",
        "What is machine learning?",
        "Explain photosynthesis:",
    ]

    # Baseline performance
    print("\n3. Baseline (FP16) Performance:")
    fp16_outputs = []
    fp16_quality_metrics = []

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

        # Assess quality
        quality = assess_quality(model, tokenizer, output, context=prompt)
        fp16_quality_metrics.append(quality)

        print(f"   Prompt {i}: {elapsed:.2f}s")
        print(f"     Output: {output[:60]}...")
        print(f"     Quality: PPL={quality.perplexity:.1f}, Rep={quality.repetition_3gram:.2%}")

    # Quantize with AWQ
    print("\n4. Quantizing with AWQ...")
    print("   AWQ protects salient channels based on activations")
    print("   This typically preserves quality better than naive quantization")

    quantized_model = awq_quantize(
        model=model,
        calibration_data=calibration_data,  # Required for AWQ
        bits=4,
        group_size=64,
    )

    # MLX builds the quantized-weight graph lazily; force materialization now so
    # AWQ's compute is not folded into the first timed generation below.
    mx.eval(quantized_model.parameters())
    print("   ✅ AWQ quantization complete!")

    # Reference perplexity: score FP16 and AWQ-4bit on the SAME held-out text to
    # measure the language-modeling impact directly (lower = better).
    ref_ppl_fp16 = calculate_perplexity(model, tokenizer, REFERENCE_TEXT)
    ref_ppl_awq = calculate_perplexity(quantized_model, tokenizer, REFERENCE_TEXT)
    ref_ppl_delta = (
        (ref_ppl_awq - ref_ppl_fp16) / ref_ppl_fp16 if ref_ppl_fp16 > 0 else float("inf")
    )
    print("   Reference-text perplexity:")
    print(f"     FP16:      {ref_ppl_fp16:.2f}")
    print(f"     AWQ 4-bit: {ref_ppl_awq:.2f} ({ref_ppl_delta:+.1%})")

    # Test quantized model
    print("\n5. AWQ-Quantized (4-bit) Performance:")
    awq_outputs = []
    awq_quality_metrics = []

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

        # Assess quality
        quality = assess_quality(quantized_model, tokenizer, output, context=prompt)
        awq_quality_metrics.append(quality)

        # Validate output
        is_valid, reason = validate_text_output(
            output, min_length=5, max_repetition_ratio=0.6, check_gibberish=True
        )

        print(f"   Prompt {i}: {elapsed:.2f}s")
        print(f"     Output: {output[:60]}...")
        print(f"     Quality: PPL={quality.perplexity:.1f}, Rep={quality.repetition_3gram:.2%}")
        print(f"     Valid: {is_valid}")

    # Quality comparison
    print("\n6. Quality Degradation Analysis:")
    print("   Comparing FP16 vs AWQ-4bit quality metrics:\n")

    def _pct(value):
        return f"{value:+.1%}" if value is not None else "n/a"

    quality_degradations = []
    for i, (fp16_q, awq_q) in enumerate(zip(fp16_quality_metrics, awq_quality_metrics), 1):
        comparison = compare_quality(fp16_q, awq_q, tolerance=0.20)
        quality_degradations.append(comparison)

        print(f"   Prompt {i}:")
        print(f"     Perplexity:  {fp16_q.perplexity:.1f} → {awq_q.perplexity:.1f} "
              f"({_pct(comparison.perplexity_change)})")
        print(f"     Repetition:  {fp16_q.repetition_3gram:.2%} → {awq_q.repetition_3gram:.2%} "
              f"({_pct(comparison.repetition_change)})")
        print(f"     Diversity:   {fp16_q.diversity_score:.2f} → {awq_q.diversity_score:.2f}")
        print(f"     Acceptable:  {'✅ Yes' if comparison.acceptable else '❌ No'}")
        if not comparison.acceptable:
            print(f"     Issues: {', '.join(comparison.degradations)}")
        print()

    # Overall quality summary
    acceptable_count = sum(1 for c in quality_degradations if c.acceptable)
    print(f"   Overall: {acceptable_count}/{len(quality_degradations)} prompts passed quality threshold")

    # Compare AWQ vs standard quantization
    print("\n7. AWQ Features:")
    print("   ✓ Protects salient weight channels")
    print("   ✓ Activation-aware scaling factors")
    print("   ✓ Better quality preservation than standard quantization")
    print("   ✓ Grid search for optimal scaling")
    print("   ✓ ~4x memory reduction (FP16 → 4-bit)")

    # Advanced: Compare different configurations
    print("\n8. Testing AWQ configurations:")

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
            calibration_data=calibration_data,  # Required for AWQ
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
    print("\nQuality Metrics Summary:")
    print("- Perplexity increase typically < 20% (acceptable threshold)")
    print("- Repetition patterns remain similar to FP16")
    print("- Output validation catches any degraded generations")
    print("- Quality assessment enables automated quality assurance")


if __name__ == "__main__":
    main()
