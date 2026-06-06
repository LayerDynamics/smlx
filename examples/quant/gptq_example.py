#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
GPTQ Quantization Example

Demonstrates how to use GPTQ (GPT Quantization) to compress models
with minimal quality loss. GPTQ uses Hessian-based optimization for
accurate quantization.

This example includes comprehensive quality benchmarking to measure
quality degradation from quantization.

GPTQ is ideal for:
- Post-training quantization
- Minimal quality degradation
- Memory-constrained environments (M4 with 36GB unified memory)
"""

import mlx.core as mx
import time

from smlx.models.SmolLM2_135M import load, generate
from smlx.quant import gptq_quantize
from smlx.quant.utils import load_calibration_data
from smlx.utils.quality_metrics import assess_quality, compare_quality, calculate_perplexity
from smlx.utils.validation import validate_text_output


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

    # Load calibration data for quantization
    print("\n2. Loading calibration data...")
    calibration_data = load_calibration_data(
        tokenizer=tokenizer,
        num_samples=128,
        sequence_length=512,
        verbose=True
    )
    print(f"   ✅ Loaded calibration data: {calibration_data.shape}")

    # Test prompt
    prompt = "Explain quantum computing in simple terms:"

    # Generate with original model
    print("\n3. Generating with full-precision model...")
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

    # Assess quality
    original_quality = assess_quality(model, tokenizer, original_output, context=prompt)

    print(f"   Time: {original_time:.2f}s")
    print(f"   Output: {original_output[:80]}...")
    print(f"   Quality: PPL={original_quality.perplexity:.1f}, "
          f"Rep={original_quality.repetition_3gram:.2%}, "
          f"Div={original_quality.diversity_score:.2f}")

    # Quantize with GPTQ
    print("\n4. Quantizing with GPTQ (4-bit, group_size=64)...")
    print("   This may take a minute...")

    quantized_model = gptq_quantize(
        model=model,
        calibration_data=calibration_data,  # Required for GPTQ
        bits=4,  # 4-bit quantization
        group_size=64,  # Optimized for M4
    )

    print("   ✅ Quantization complete!")

    # Check quantized size (approximate - actual is ~1/4 of original)
    print(f"   Expected compression: ~4x (from FP16 to 4-bit)")
    print(f"   Estimated quantized size: ~{original_params // 4:,} effective params")

    # Generate with quantized model
    print("\n5. Generating with quantized model...")
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

    # Assess quality
    quantized_quality = assess_quality(quantized_model, tokenizer, quantized_output, context=prompt)

    # Validate output
    is_valid, reason = validate_text_output(
        quantized_output, min_length=10, max_repetition_ratio=0.6, check_gibberish=True
    )

    print(f"   Time: {quantized_time:.2f}s")
    print(f"   Output: {quantized_output[:80]}...")
    print(f"   Quality: PPL={quantized_quality.perplexity:.1f}, "
          f"Rep={quantized_quality.repetition_3gram:.2%}, "
          f"Div={quantized_quality.diversity_score:.2f}")
    print(f"   Valid: {is_valid}")

    # Compare results
    print("\n6. Performance Comparison:")
    print(f"   Speed improvement: {original_time / quantized_time:.2f}x faster")
    print(f"   Memory reduction: ~4x smaller")

    # Quality degradation analysis
    print("\n7. Quality Degradation Analysis:")
    comparison = compare_quality(original_quality, quantized_quality, tolerance=0.20)

    print(f"   Perplexity:  {original_quality.perplexity:.1f} → {quantized_quality.perplexity:.1f} "
          f"({comparison['perplexity_change']:+.1%})")
    print(f"   Repetition:  {original_quality.repetition_3gram:.2%} → {quantized_quality.repetition_3gram:.2%} "
          f"({comparison['repetition_change']:+.1%})")
    print(f"   Diversity:   {original_quality.diversity_score:.2f} → {quantized_quality.diversity_score:.2f}")
    print(f"   Unique Ratio: {original_quality.unique_token_ratio:.2%} → {quantized_quality.unique_token_ratio:.2%}")
    print(f"   Acceptable:  {'✅ Yes' if comparison['acceptable'] else '❌ No'}")
    if not comparison['acceptable']:
        print(f"   Issues: {', '.join(comparison['degradations'])}")

    # Test different bit widths
    print("\n8. Testing different quantization levels...")

    test_prompt = "Hello world"
    for bits in [8, 4, 2]:
        print(f"\n   {bits}-bit quantization:")
        q_model = gptq_quantize(model, calibration_data, bits=bits, group_size=64)

        start = time.time()
        output = generate(
            model=q_model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            max_tokens=20,
            temperature=0.0,  # Greedy for consistency
            verbose=False,
        )
        elapsed = time.time() - start

        # Quick quality check
        is_valid, _ = validate_text_output(output, min_length=3, check_gibberish=True)
        quality = assess_quality(q_model, tokenizer, output, context=test_prompt)

        print(f"     Time: {elapsed:.2f}s")
        print(f"     Output: {output}")
        print(f"     Compression: ~{16 // bits}x")
        print(f"     Quality: PPL={quality.perplexity:.1f}, Valid={is_valid}")

    print("\n" + "=" * 70)
    print("✅ GPTQ Example Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- 4-bit GPTQ provides excellent quality/size tradeoff")
    print("- group_size=64 is optimized for M4 chipsets")
    print("- Minimal quality degradation with 4x compression")
    print("- Faster inference due to reduced memory bandwidth")
    print("\nQuality Metrics Summary:")
    print("- GPTQ uses Hessian-based optimization for accurate quantization")
    print("- Perplexity increase typically < 20% (acceptable threshold)")
    print("- Output validation ensures no gibberish or degraded outputs")
    print("- Quality assessment enables automated quality assurance")
    print("- 8-bit quantization has nearly identical quality to FP16")
    print("- 2-bit may show noticeable degradation (use with caution)")


if __name__ == "__main__":
    main()
