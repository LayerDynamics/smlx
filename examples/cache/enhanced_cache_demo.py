#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Enhanced KV Cache Demonstration

This example demonstrates the new enhanced cache features available in SMLX models:
- Automatic cache type selection based on available memory
- Quantized caches (4-bit and 8-bit) for memory efficiency
- Memory pressure monitoring and automatic OOM prevention
- Rotating caches with automatic sizing

Supported Models:
- SmolLM2-135M / SmolLM2-360M
- Moondream2
- TinyLLaVA
- SmolVLM-256M / SmolVLM-500M

Requirements:
    pip install mlx transformers pillow
"""

import mlx.core as mx

from smlx.models.SmolLM2_135M import load, generate


def example_1_basic_enhanced_cache():
    """Example 1: Basic enhanced cache with automatic selection."""
    print("=" * 70)
    print("Example 1: Basic Enhanced Cache (Auto Mode)")
    print("=" * 70)

    # Load model
    model, tokenizer = load()

    # Create cache with automatic selection based on available memory
    # This will choose the best cache type for your system
    from smlx.models.SmolLM2_135M.cache import make_cache

    cache = make_cache(
        model,
        cache_type="auto",  # Automatically select cache type
        target_memory_gb=32.0,  # Target 32GB total memory usage
    )

    print(f"✓ Created auto cache with {len(cache)} layers")
    print(f"  Cache type: {type(cache[0]).__name__}")

    # Generate text
    prompt = "The future of AI is"
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
    )

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print()


def example_2_quantized_cache():
    """Example 2: Quantized cache for memory efficiency."""
    print("=" * 70)
    print("Example 2: Quantized Cache (4-bit)")
    print("=" * 70)

    # Load model
    model, tokenizer = load()

    # Create 4-bit quantized cache
    # This uses ~4x less memory than standard FP16 cache
    from smlx.models.SmolLM2_135M.cache import make_cache

    cache = make_cache(
        model,
        cache_type="quantized",
        quantization_bits=4,  # 4-bit quantization (~4x compression)
        max_kv_size=4096,  # Maximum sequence length
    )

    print(f"✓ Created quantized cache (4-bit)")
    print(f"  Compression: ~4x compared to FP16")
    print(f"  Max sequence length: 4096 tokens")

    # Generate text with quantized cache
    prompt = "In machine learning,"
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
    )

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print()


def example_3_memory_monitoring():
    """Example 3: Memory pressure monitoring and OOM prevention."""
    print("=" * 70)
    print("Example 3: Memory Monitoring & OOM Prevention")
    print("=" * 70)

    # Load model
    model, tokenizer = load()

    # Create cache with automatic memory monitoring
    from smlx.models.SmolLM2_135M.cache import make_cache_with_monitoring

    cache, breaker = make_cache_with_monitoring(
        model,
        cache_type="auto",
        target_memory_gb=32.0,
        warning_threshold=0.8,  # Warn at 80% memory
        critical_threshold=0.9,  # Critical at 90% memory
    )

    print(f"✓ Created cache with memory monitoring")
    print(f"  Warning threshold: 80%")
    print(f"  Critical threshold: 90%")
    print(f"  PressureBreaker enabled: {breaker.enabled}")

    # Monitor during generation
    from smlx.models.SmolLM2_135M import generate_step

    prompt = "Artificial intelligence will"
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    print(f"\nPrompt: {prompt}")
    print("Generating with memory monitoring...")

    generated_tokens = []
    for step, (token, _) in enumerate(
        generate_step(
            model=model,
            prompt_tokens=prompt_tokens,
            temperature=0.7,
            cache=cache,
        )
    ):
        # Monitor memory pressure at each step
        intervention = breaker.monitor_and_intervene(current_step=step)

        if intervention:
            print(f"\n⚠️  Memory intervention at step {step}:")
            print(f"   Action: {intervention['action']}")
            print(f"   Reason: {intervention['reason']}")

        generated_tokens.append(int(token.item()))

        # Stop after 50 tokens
        if step >= 49:
            break

    response = tokenizer.decode(generated_tokens)
    print(f"Response: {response}")

    # Get statistics
    stats = breaker.get_statistics()
    print(f"\n📊 Memory Monitoring Statistics:")
    print(f"   Total interventions: {stats['total_interventions']}")
    print(f"   Enabled: {stats['enabled']}")
    print()


def example_4_rotating_cache():
    """Example 4: Rotating cache for long sequences."""
    print("=" * 70)
    print("Example 4: Rotating Cache (Fixed Memory)")
    print("=" * 70)

    # Load model
    model, tokenizer = load()

    # Create rotating cache with fixed size
    # Old tokens are dropped when cache fills up
    from smlx.models.SmolLM2_135M.cache import make_cache

    cache = make_cache(
        model,
        cache_type="rotating",
        max_kv_size=2048,  # Keep only 2048 most recent tokens
    )

    print(f"✓ Created rotating cache")
    print(f"  Max size: 2048 tokens")
    print(f"  Behavior: Drops old tokens when full")

    # Generate text
    prompt = "The key to understanding neural networks is"
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
    )

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print()


def example_5_vision_language_model():
    """Example 5: Enhanced cache with vision-language models."""
    print("=" * 70)
    print("Example 5: Enhanced Cache with Vision-Language Model")
    print("=" * 70)

    try:
        from smlx.models.SmolVLM_256M import load as load_vlm
        from smlx.models.SmolVLM_256M import generate as generate_vlm
        from smlx.models.SmolVLM_256M.cache import make_cache_with_monitoring
        from PIL import Image
        import numpy as np

        # Load VLM model
        print("Loading SmolVLM-256M...")
        model, processor = load_vlm()

        # Create cache with monitoring
        cache, breaker = make_cache_with_monitoring(
            model.language_model,
            cache_type="auto",
            target_memory_gb=32.0,
        )

        print(f"✓ Created cache for vision-language model")
        print(f"  Language model layers: {len(cache)}")
        print(f"  Memory monitoring: enabled")

        # Create a sample image (since we may not have an actual image)
        sample_image = Image.fromarray(
            np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        )

        # Generate description
        print("\nGenerating image description...")
        response = generate_vlm(
            model=model,
            processor=processor,
            prompt="Describe this image:",
            image=sample_image,
            max_tokens=30,
            temperature=0.7,
        )

        print(f"Response: {response}")

        # Get monitoring statistics
        stats = breaker.get_statistics()
        print(f"\n📊 Cache Statistics:")
        print(f"   Interventions: {stats['total_interventions']}")

    except ImportError as e:
        print(f"⚠️  SmolVLM-256M not available: {e}")
        print("   This example requires SmolVLM-256M model")

    print()


def example_6_cache_comparison():
    """Example 6: Compare different cache configurations."""
    print("=" * 70)
    print("Example 6: Cache Configuration Comparison")
    print("=" * 70)

    import time
    from smlx.models.SmolLM2_135M import load
    from smlx.models.SmolLM2_135M.cache import make_cache

    # Load model
    model, tokenizer = load()

    prompt = "The most important aspect of machine learning is"
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    configs = [
        ("Standard Cache", {"cache_type": "standard"}),
        ("Rotating Cache (2048)", {"cache_type": "rotating", "max_kv_size": 2048}),
        ("Quantized Cache (4-bit)", {"cache_type": "quantized", "quantization_bits": 4, "max_kv_size": 2048}),
    ]

    print(f"Prompt: {prompt}")
    print(f"Generating 50 tokens with different cache configurations...\n")

    for name, cache_config in configs:
        # Create cache
        cache = make_cache(model, **cache_config)

        # Clear memory
        mx.clear_cache()

        # Measure generation time
        start_time = time.perf_counter()

        from smlx.models.SmolLM2_135M import generate_step

        tokens = []
        for i, (token, _) in enumerate(
            generate_step(model=model, prompt_tokens=prompt_tokens, temperature=0.7, cache=cache)
        ):
            tokens.append(int(token.item()))
            if i >= 49:
                break

        end_time = time.perf_counter()
        generation_time = end_time - start_time

        # Get memory usage
        peak_memory_gb = mx.metal.get_peak_memory() / 1e9 if mx.metal.is_available() else 0.0

        print(f"📊 {name}:")
        print(f"   Time: {generation_time:.3f}s")
        print(f"   Tokens/sec: {len(tokens) / generation_time:.1f}")
        print(f"   Peak memory: {peak_memory_gb:.2f} GB")
        print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "SMLX Enhanced KV Cache Examples" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    examples = [
        ("Basic Enhanced Cache", example_1_basic_enhanced_cache),
        ("Quantized Cache", example_2_quantized_cache),
        ("Memory Monitoring", example_3_memory_monitoring),
        ("Rotating Cache", example_4_rotating_cache),
        ("Vision-Language Model", example_5_vision_language_model),
        ("Cache Comparison", example_6_cache_comparison),
    ]

    for i, (name, example_fn) in enumerate(examples, 1):
        try:
            example_fn()
        except Exception as e:
            print(f"❌ Example {i} ({name}) failed: {e}")
            import traceback

            traceback.print_exc()
            print()

    print("=" * 70)
    print("✓ All examples completed!")
    print("=" * 70)
    print()
    print("For more information, see:")
    print("- smlx.kv_cache module documentation")
    print("- Model-specific cache modules (e.g., smlx.models.SmolLM2_135M.cache)")
    print()


if __name__ == "__main__":
    main()
