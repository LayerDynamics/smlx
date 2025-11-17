#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
SmolLM2-135M-Instruct Usage Examples with Memory Best Practices

This script demonstrates how to use the SmolLM2-135M-Instruct model
for various text generation tasks, with production-ready memory management.

Key memory best practices shown:
- Using memory watchdog for automatic protection
- Robust inference with automatic retry on OOM
- Graceful degradation under memory pressure
- Model-specific safe parameter profiles
- Memory monitoring and cleanup
"""

from smlx.models.SmolLM2_135M import (
    GenerationConfig,
    chat,
    generate,
    load,
    stream_generate,
)

# Memory management imports
from smlx.config.model_profiles import auto_select_params
from smlx.utils.degradation import with_graceful_degradation
from smlx.utils.memory import (
    get_active_memory_gb,
    get_cache_memory_gb,
    print_memory_state,
    smart_cleanup,
)
from smlx.utils.robust import robust_generate
from smlx.utils.watchdog import watchdog


def print_memory_info(label: str = "Memory Status"):
    """Helper to print current memory usage."""
    active_gb = get_active_memory_gb()
    cache_gb = get_cache_memory_gb()
    total_gb = active_gb + cache_gb
    print(f"\n[{label}] Active: {active_gb:.2f}GB | Cache: {cache_gb:.2f}GB | "
          f"Total: {total_gb:.2f}GB")


def example_1_basic_generation():
    """Example 1: Basic text generation."""
    print("=" * 70)
    print("Example 1: Basic Text Generation")
    print("=" * 70)

    # Load model and tokenizer
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
    print_memory_info("After model load")

    # Generate text
    prompt = "Write a Python function to calculate factorial:"
    print(f"\nPrompt: {prompt}")
    print("\nGenerating...")

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        verbose=True,
    )

    print(f"\nGenerated text:\n{response}")
    print_memory_info("After generation")


def example_2_streaming():
    """Example 2: Streaming generation."""
    print("\n" + "=" * 70)
    print("Example 2: Streaming Generation")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Stream generation
    prompt = "Explain what machine learning is in simple terms:"
    print(f"\nPrompt: {prompt}")
    print("\nStreaming output:\n")

    for text_chunk in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
    ):
        print(text_chunk, end="", flush=True)

    print("\n")


def example_3_chat():
    """Example 3: Chat-style interaction."""
    print("=" * 70)
    print("Example 3: Chat Interaction")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Chat messages
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]

    print("\nUser:", messages[0]["content"])
    print("\nAssistant:")

    response = chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=50,
        temperature=0.7,
        verbose=False,
    )

    print(response)

    # Continue conversation
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "What is it famous for?"})

    print("\nUser:", messages[2]["content"])
    print("\nAssistant:")

    response = chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        verbose=False,
    )

    print(response)


def example_4_custom_config():
    """Example 4: Using custom generation config."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Generation Config")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Create custom config
    config = GenerationConfig(
        max_tokens=200,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        stop_strings=["###", "END"],
    )

    prompt = "List three benefits of exercise:\n1."
    print(f"\nPrompt: {prompt}")
    print(f"\nConfig: temp={config.temperature}, top_p={config.top_p}, top_k={config.top_k}")
    print("\nGenerating...")

    from smlx.models.SmolLM2_135M import complete

    response = complete(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
        verbose=True,
    )

    print(f"\nGenerated:\n{response}")


def example_5_greedy_vs_sampling():
    """Example 5: Comparing greedy vs sampling."""
    print("\n" + "=" * 70)
    print("Example 5: Greedy vs Sampling")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    prompt = "The quick brown fox"

    # Greedy (deterministic)
    print(f"\nPrompt: {prompt}")
    print("\n--- Greedy (temperature=0) ---")
    response_greedy = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=20,
        temperature=0.0,  # Greedy
        verbose=False,
    )
    print(response_greedy)

    # Sampling (creative)
    print("\n--- Sampling (temperature=0.9) ---")
    response_sampling = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=20,
        temperature=0.9,  # High temperature
        verbose=False,
    )
    print(response_sampling)

    # Another sampling run (should be different)
    print("\n--- Sampling (temperature=0.9) - Run 2 ---")
    response_sampling2 = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=20,
        temperature=0.9,
        verbose=False,
    )
    print(response_sampling2)


def example_6_memory_watchdog():
    """Example 6: Using memory watchdog for protection."""
    print("\n" + "=" * 70)
    print("Example 6: Memory Watchdog Protection")
    print("=" * 70)

    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    prompt = "Write a detailed story about space exploration:"

    # Use watchdog context manager for automatic memory monitoring
    print("\nGenerating with watchdog protection...")
    print("(Watchdog will warn if memory usage exceeds thresholds)")

    with watchdog(warning_threshold=0.80, critical_threshold=0.90, auto_cleanup=True):
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=300,  # Longer generation
            temperature=0.7,
            verbose=False,
        )

    print(f"\nGenerated {len(response.split())} words")
    print(f"Preview: {response[:200]}...")


def example_7_robust_inference():
    """Example 7: Robust inference with automatic retry."""
    print("\n" + "=" * 70)
    print("Example 7: Robust Inference with Retry")
    print("=" * 70)

    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    prompt = "Explain quantum computing in detail:"

    # Use robust_generate for automatic retry on OOM or errors
    print("\nGenerating with robust wrapper...")
    print("(Will automatically retry with reduced parameters on failure)")

    result = robust_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=500,  # Aggressive parameter
        temperature=0.7,
        max_retries=3,
    )

    if result.success:
        print(f"\n✓ Success after {result.attempts} attempt(s)")
        print(f"Generated text: {result.text[:200]}...")
        print(f"Final params: max_tokens={result.final_params['max_tokens']}")
    else:
        print(f"\n✗ Failed after {result.attempts} attempts")
        print(f"Error: {result.error_message}")


def example_8_graceful_degradation():
    """Example 8: Graceful degradation under memory pressure."""
    print("\n" + "=" * 70)
    print("Example 8: Graceful Degradation")
    print("=" * 70)

    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    prompt = "Describe the history of computing:"

    # Parameters will be automatically adjusted based on available memory
    print("\nGenerating with graceful degradation...")
    print("(Parameters auto-adjust based on memory pressure)")

    params = with_graceful_degradation(
        max_tokens=400,
        temperature=0.7,
        batch_size=4,
    )

    print(f"Adjusted params: max_tokens={params['max_tokens']}, "
          f"batch_size={params['batch_size']}")

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        **params,
    )

    print(f"\nGenerated {len(response.split())} words")


def example_9_model_profiles():
    """Example 9: Using model-specific safe parameters."""
    print("\n" + "=" * 70)
    print("Example 9: Model-Specific Safe Parameters")
    print("=" * 70)

    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Auto-select safe parameters based on available memory
    safe_params = auto_select_params("SmolLM2-135M")
    print(f"\nAuto-selected safe parameters for SmolLM2-135M:")
    print(f"  max_tokens: {safe_params['max_tokens']}")
    print(f"  max_kv_size: {safe_params['max_kv_size']}")
    print(f"  batch_size: {safe_params['batch_size']}")
    print(f"  use_rotating_cache: {safe_params['use_rotating_cache']}")

    prompt = "What are the benefits of renewable energy?"

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=safe_params['max_tokens'],
        temperature=0.7,
        verbose=False,
    )

    print(f"\nGenerated: {response}")


def example_10_memory_cleanup():
    """Example 10: Manual memory cleanup."""
    print("\n" + "=" * 70)
    print("Example 10: Memory Cleanup")
    print("=" * 70)

    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    print_memory_info("Before generation")

    # Generate text
    prompt = "Write a short poem:"
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=100,
        temperature=0.8,
        verbose=False,
    )

    print(f"\nGenerated: {response}")
    print_memory_info("After generation")

    # Perform cleanup
    print("\nPerforming memory cleanup...")
    freed_gb = smart_cleanup(aggressive=False)
    print(f"Freed {freed_gb:.3f}GB from cache")

    print_memory_info("After cleanup")

    # Aggressive cleanup (includes Python GC)
    print("\nPerforming aggressive cleanup...")
    freed_gb = smart_cleanup(aggressive=True)
    print(f"Freed {freed_gb:.3f}GB from cache")

    print_memory_info("After aggressive cleanup")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "SmolLM2-135M Examples with Memory Best Practices" + " " * 9 + "║")
    print("╚" + "=" * 68 + "╝")

    examples = [
        example_1_basic_generation,
        example_2_streaming,
        example_3_chat,
        example_4_custom_config,
        example_5_greedy_vs_sampling,
        example_6_memory_watchdog,
        example_7_robust_inference,
        example_8_graceful_degradation,
        example_9_model_profiles,
        example_10_memory_cleanup,
    ]

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\n❌ Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("✓ Examples complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
