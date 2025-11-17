#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
SmolLM2-360M-Instruct Usage Examples with Memory Best Practices

This script demonstrates how to use the SmolLM2-360M-Instruct model
for various text generation tasks.

MEMORY BEST PRACTICES:
For comprehensive memory management examples, see:
    examples/models/smollm2_135m/smollm2_135m_example.py

Key utilities available:
- watchdog: Automatic memory monitoring
- robust_generate: Auto-retry on OOM
- with_graceful_degradation: Auto-adjust parameters
- auto_select_params: Model-specific safe parameters
- smart_cleanup: Manual memory cleanup
"""

from smlx.models.SmolLM2_360M import (
    GenerationConfig,
    chat,
    generate,
    load,
    stream_generate,
)

# Memory management utilities (see SmolLM2-135M example for detailed usage)
from smlx.config.model_profiles import auto_select_params
from smlx.utils.memory import smart_cleanup, get_active_memory_gb, get_cache_memory_gb
from smlx.utils.watchdog import watchdog
from smlx.utils.robust import robust_generate
from smlx.utils.degradation import with_graceful_degradation


def example_1_basic_generation():
    """Example 1: Basic text generation."""
    print("=" * 70)
    print("Example 1: Basic Text Generation")
    print("=" * 70)

    # Load model and tokenizer
    print("\nLoading SmolLM2-360M model...")
    model, tokenizer = load("mlx-community/SmolLM2-360M-Instruct")

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


def example_2_streaming():
    """Example 2: Streaming generation."""
    print("\n" + "=" * 70)
    print("Example 2: Streaming Generation")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-360M-Instruct")

    # Stream generation
    prompt = "Explain quantum computing in simple terms:"
    print(f"\nPrompt: {prompt}")
    print("\nStreaming output:")
    print("-" * 70)

    for token in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=200,
        temperature=0.8,
    ):
        print(token, end="", flush=True)

    print("\n" + "-" * 70)


def example_3_chat_interface():
    """Example 3: Chat interface."""
    print("\n" + "=" * 70)
    print("Example 3: Chat Interface")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-360M-Instruct")

    # Chat conversation
    messages = [
        {"role": "user", "content": "What is machine learning?"},
    ]

    print("\nUser: What is machine learning?")
    print("\nAssistant: ", end="")

    response = chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=150,
        temperature=0.7,
    )

    print(response)


def example_4_custom_config():
    """Example 4: Using custom generation configuration."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Generation Configuration")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-360M-Instruct")

    # Create custom configuration
    config = GenerationConfig(
        max_tokens=100,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
    )

    prompt = "Write a creative story about a robot:"
    print(f"\nPrompt: {prompt}")
    print(f"\nGeneration config: temp={config.temperature}, top_p={config.top_p}, top_k={config.top_k}")
    print("\nGenerating...")

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
    )

    print(f"\nGenerated text:\n{response}")


def example_5_greedy_vs_sampling():
    """Example 5: Compare greedy decoding vs sampling."""
    print("\n" + "=" * 70)
    print("Example 5: Greedy Decoding vs Sampling")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/SmolLM2-360M-Instruct")

    prompt = "The future of artificial intelligence is"

    # Greedy decoding (temperature=0)
    print(f"\nPrompt: {prompt}")
    print("\n--- Greedy Decoding (temperature=0) ---")
    response_greedy = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=50,
        temperature=0.0,
    )
    print(response_greedy)

    # Sampling with temperature
    print("\n--- Sampling (temperature=1.0) ---")
    response_sampling = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=50,
        temperature=1.0,
    )
    print(response_sampling)


def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 15 + "SmolLM2-360M-Instruct Examples" + " " * 23 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    try:
        example_1_basic_generation()
        example_2_streaming()
        example_3_chat_interface()
        example_4_custom_config()
        example_5_greedy_vs_sampling()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
