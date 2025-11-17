#!/usr/bin/env python3
"""
SmolVLM-256M Streaming Generation Example with Memory Best Practices

This example demonstrates:
1. Streaming text generation token-by-token
2. Real-time visual question answering
3. Chat-style interaction with images

Usage:
    python examples/vlm/smolvlm_256m/streaming_example.py

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

from PIL import Image
import numpy as np

from smlx.models.SmolVLM_256M import load, stream_generate, chat

# Memory management utilities (see SmolLM2-135M example for detailed usage)
from smlx.config.model_profiles import auto_select_params
from smlx.utils.memory import smart_cleanup, get_active_memory_gb, get_cache_memory_gb
from smlx.utils.watchdog import watchdog
from smlx.utils.robust import robust_generate
from smlx.utils.degradation import with_graceful_degradation


def create_test_image() -> Image.Image:
    """Create a sample test image."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def main():
    print("=" * 60)
    print("SmolVLM-256M - Streaming Generation Example")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    model, processor = load()
    print("✓ Model loaded!")

    # Create test image
    print("\n[2/3] Creating test image...")
    image = create_test_image()
    print(f"✓ Image created: {image.size}")

    # Example 1: Streaming generation
    print("\n[3/3] Streaming generation...")
    print("Prompt: Describe this image in detail.\n")
    print("Response: ", end="", flush=True)

    for text_chunk in stream_generate(
        model=model,
        processor=processor,
        prompt="Describe this image in detail.",
        image=image,
        max_tokens=100,
        temperature=0.8,
    ):
        print(text_chunk, end="", flush=True)

    print("\n")

    # Example 2: Chat-style interaction
    print("\n" + "-" * 60)
    print("Chat-style interaction:")
    print("-" * 60)

    messages = [{"role": "user", "content": "What do you see in this image?"}]

    print("\nUser: What do you see in this image?")
    print("Assistant: ", end="", flush=True)

    response = chat(
        model=model,
        processor=processor,
        messages=messages,
        image=image,
        max_tokens=80,
        temperature=0.7,
    )

    print(response)

    # Follow-up question
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "Can you describe it more briefly?"})

    print("\nUser: Can you describe it more briefly?")
    print("Assistant: ", end="", flush=True)

    response = chat(
        model=model,
        processor=processor,
        messages=messages,
        image=image,
        max_tokens=50,
        temperature=0.6,
    )

    print(response)

    print("\n" + "=" * 60)
    print("Streaming example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
