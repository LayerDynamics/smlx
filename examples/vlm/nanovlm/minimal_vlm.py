#!/usr/bin/env python3
"""
nanoVLM - Minimal Vision-Language Model Example with Memory Best Practices

nanoVLM is the smallest VLM in SMLX (222M parameters):
- SigLIP-base vision encoder (85M, 224x224)
- MLP projection (~2M)
- SmolLM2-135M language model (135M)

This example demonstrates minimal usage for:
1. Image captioning
2. Visual question answering
3. Simple VQA queries

Usage:
    python examples/vlm/nanovlm/minimal_vlm.py

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

from smlx.models.nanoVLM import load
from smlx.models.nanoVLM.generate import generate, caption, query

# Memory management utilities (see SmolLM2-135M example for detailed usage)
from smlx.config.model_profiles import auto_select_params
from smlx.utils.memory import smart_cleanup, get_active_memory_gb, get_cache_memory_gb
from smlx.utils.watchdog import watchdog
from smlx.utils.robust import robust_generate
from smlx.utils.degradation import with_graceful_degradation


def create_sample_image() -> Image.Image:
    """Create a simple test image."""
    # Create a gradient image for testing
    arr = np.zeros((224, 224, 3), dtype=np.uint8)

    # Red to blue gradient
    for i in range(224):
        arr[i, :, 0] = int(255 * (1 - i / 224))  # Red
        arr[i, :, 2] = int(255 * (i / 224))  # Blue

    return Image.fromarray(arr)


def main():
    print("=" * 60)
    print("nanoVLM - Minimal Vision-Language Model (222M)")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading nanoVLM model...")
    try:
        model, processor = load("lusxvr/nanoVLM-222M")
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"⚠ Could not load from HuggingFace: {e}")
        print("Creating model with default config...")
        model, processor = load()

    # Create test image
    print("\n[2/4] Creating test image...")
    image = create_sample_image()
    print(f"✓ Image created: {image.size}")

    # Example 1: Image captioning
    print("\n[3/4] Generating image caption...")
    caption_text = caption(model, processor, image, max_tokens=50)
    print(f"Caption: {caption_text}")

    # Example 2: Visual question answering
    print("\n[4/4] Visual question answering...")
    questions = [
        "What colors do you see?",
        "Describe the pattern in this image.",
        "Is this a photograph or a digital creation?",
    ]

    for q in questions:
        answer = query(model, processor, image, q, max_tokens=40)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    # Example 3: Custom generation
    print("\n" + "-" * 60)
    print("Custom generation with temperature control:")
    print("-" * 60)

    prompt = "Analyze this image and describe its visual properties:"

    # Low temperature (more focused)
    print("\nLow temperature (0.3):")
    response = generate(
        model, processor, prompt, image, max_tokens=60, temperature=0.3, top_p=0.9
    )
    print(response)

    # High temperature (more creative)
    print("\nHigh temperature (1.2):")
    response = generate(
        model, processor, prompt, image, max_tokens=60, temperature=1.2, top_p=0.95
    )
    print(response)

    print("\n" + "=" * 60)
    print("nanoVLM example completed!")
    print(f"Model size: 222M parameters")
    print(f"- Vision: SigLIP-base (85M)")
    print(f"- Projection: MLP (~2M)")
    print(f"- Language: SmolLM2-135M (135M)")
    print("=" * 60)


if __name__ == "__main__":
    main()
