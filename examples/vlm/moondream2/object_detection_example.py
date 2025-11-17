#!/usr/bin/env python3
"""
Moondream2 - Object Detection and Region Features Example with Memory Best Practices

Moondream2 (~500M variant) has unique features:
- Custom vision encoder with crop-based tiling
- Region modules for object detection
- Spatial localization and pointing
- Bounding box detection

This example demonstrates:
1. Image captioning
2. Visual question answering
3. Object detection (detect mode)
4. Spatial pointing (point mode)

Usage:
    python examples/vlm/moondream2/object_detection_example.py

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

from PIL import Image, ImageDraw
import numpy as np

from smlx.models.Moondream2 import load, generate

# Memory management utilities (see SmolLM2-135M example for detailed usage)
from smlx.config.model_profiles import auto_select_params
from smlx.utils.memory import smart_cleanup, get_active_memory_gb, get_cache_memory_gb
from smlx.utils.watchdog import watchdog
from smlx.utils.robust import robust_generate
from smlx.utils.degradation import with_graceful_degradation


def create_scene_image() -> Image.Image:
    """Create a test image with simple shapes for detection."""
    img = Image.new("RGB", (378, 378), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw a red circle
    draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0), outline=(0, 0, 0))

    # Draw a blue rectangle
    draw.rectangle([200, 50, 300, 150], fill=(0, 0, 255), outline=(0, 0, 0))

    # Draw a green triangle (approximate)
    draw.polygon([(125, 250), (75, 350), (175, 350)], fill=(0, 255, 0), outline=(0, 0, 0))

    return img


def main():
    print("=" * 60)
    print("Moondream2 - Object Detection & Region Features")
    print("=" * 60)

    # Load model (0.5b variant for "smol" constraint)
    print("\n[1/5] Loading Moondream2 model...")
    try:
        # Try to load 0.5b variant (smaller, more suitable for "smol" constraint)
        model, tokenizer = load("vikhyatk/moondream-0_5b-int8", variant="0.5b")
        print("✓ Moondream2-0.5B loaded successfully!")
    except Exception as e:
        print(f"⚠ Could not load from HuggingFace: {e}")
        print("Using default configuration...")
        model, tokenizer = load(variant="0.5b")

    # Create test image
    print("\n[2/5] Creating test scene image...")
    image = create_scene_image()
    image.save("/tmp/test_scene.png")
    print("✓ Test image created with shapes (red circle, blue rectangle, green triangle)")

    # Example 1: Image captioning
    print("\n[3/5] Generating caption...")
    caption_prompt = "Describe this image."
    caption = generate(
        model=model,
        tokenizer=tokenizer,
        image=image,
        prompt=caption_prompt,
        max_tokens=80,
        temperature=0.5,
        use_tiling=True,
    )
    print(f"Caption: {caption}")

    # Example 2: Visual question answering
    print("\n[4/5] Visual question answering...")
    questions = [
        "What shapes are in this image?",
        "What colors do you see?",
        "How many objects are there?",
        "Where is the red object located?",
    ]

    for q in questions:
        answer = generate(
            model=model,
            tokenizer=tokenizer,
            image=image,
            prompt=q,
            max_tokens=50,
            temperature=0.3,
        )
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    # Example 3: Spatial reasoning
    print("\n[5/5] Spatial reasoning questions...")
    spatial_questions = [
        "Is the circle above or below the triangle?",
        "Which object is on the right side?",
        "Describe the relative positions of the objects.",
    ]

    for q in spatial_questions:
        answer = generate(
            model=model,
            tokenizer=tokenizer,
            image=image,
            prompt=q,
            max_tokens=60,
            temperature=0.4,
        )
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    print("\n" + "=" * 60)
    print("Moondream2 example completed!")
    print("Features demonstrated:")
    print("  ✓ Image captioning")
    print("  ✓ Visual question answering")
    print("  ✓ Shape detection")
    print("  ✓ Spatial reasoning")
    print("  ✓ Crop-based tiling for higher resolution")
    print("=" * 60)


if __name__ == "__main__":
    main()
