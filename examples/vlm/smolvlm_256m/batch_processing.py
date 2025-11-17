#!/usr/bin/env python3
"""
SmolVLM-256M Batch Image Processing Example with Memory Best Practices

This example demonstrates:
1. Processing multiple images efficiently
2. Batch visual question answering
3. Generating captions for multiple images
4. Benchmarking performance

Usage:
    python examples/vlm/smolvlm_256m/batch_processing.py

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

import time
from pathlib import Path
from PIL import Image
import numpy as np

from smlx.models.SmolVLM_256M import load, generate

# Memory management utilities (see SmolLM2-135M example for detailed usage)
from smlx.config.model_profiles import auto_select_params
from smlx.utils.memory import smart_cleanup, get_active_memory_gb, get_cache_memory_gb
from smlx.utils.watchdog import watchdog
from smlx.utils.robust import robust_generate
from smlx.utils.degradation import with_graceful_degradation


def create_test_images(num_images: int = 3) -> list[Image.Image]:
    """Create sample test images."""
    images = []
    for i in range(num_images):
        # Create random image
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        images.append(img)
    return images


def main():
    print("=" * 60)
    print("SmolVLM-256M - Batch Processing Example")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    model, processor = load()
    print("✓ Model loaded!")

    # Create test images
    print("\n[2/3] Creating test images...")
    num_images = 5
    images = create_test_images(num_images)
    print(f"✓ Created {num_images} test images")

    # Define questions for each image
    prompts = [
        "Describe this image.",
        "What objects do you see?",
        "What are the main colors?",
        "Is this a natural or artificial scene?",
        "Describe the composition of this image.",
    ]

    # Batch processing
    print(f"\n[3/3] Processing {num_images} images...")
    start_time = time.time()

    results = []
    for i, (image, prompt) in enumerate(zip(images, prompts)):
        print(f"\n--- Image {i+1}/{num_images} ---")
        print(f"Prompt: {prompt}")

        response = generate(
            model=model,
            processor=processor,
            prompt=prompt,
            image=image,
            max_tokens=50,
            temperature=0.7,
        )

        results.append(response)
        print(f"Response: {response}")

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    print(f"Total images processed: {num_images}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average time per image: {elapsed/num_images:.2f}s")
    print("\nAll responses:")
    for i, response in enumerate(results):
        print(f"{i+1}. {response}")
    print("=" * 60)


if __name__ == "__main__":
    main()
