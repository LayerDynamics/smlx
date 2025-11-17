#!/usr/bin/env python3
"""
SmolVLM-256M Basic Visual Question Answering Example with Memory Best Practices

This example demonstrates:
1. Loading SmolVLM-256M model
2. Processing images from URLs, files, or PIL Images
3. Asking questions about images
4. Generating captions
5. Different generation configurations

Usage:
    python examples/vlm/smolvlm_256m/basic_vqa.py

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
import requests
from io import BytesIO

from smlx.models.SmolVLM_256M import load, generate

# Memory management utilities (see SmolLM2-135M example for detailed usage)
from smlx.config.model_profiles import auto_select_params
from smlx.utils.memory import smart_cleanup, get_active_memory_gb, get_cache_memory_gb
from smlx.utils.watchdog import watchdog
from smlx.utils.robust import robust_generate
from smlx.utils.degradation import with_graceful_degradation


def download_sample_image(url: str) -> Image.Image:
    """Download a sample image from URL."""
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def main():
    print("=" * 60)
    print("SmolVLM-256M - Visual Question Answering Example")
    print("=" * 60)

    # Load model and processor
    print("\n[1/4] Loading SmolVLM-256M model...")
    model, processor = load()
    print("✓ Model loaded successfully!")

    # Example 1: Image from URL
    print("\n[2/4] Processing image from URL...")
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

    try:
        image = download_sample_image(image_url)
        print(f"✓ Image downloaded: {image.size}")

        # Ask a question
        question = "What type of vehicle is this?"
        print(f"\nQuestion: {question}")

        response = generate(
            model=model,
            processor=processor,
            prompt=question,
            image=image,
            max_tokens=50,
            temperature=0.7,
            verbose=True,
        )
        print(f"Answer: {response}")

    except Exception as e:
        print(f"⚠ Could not download image: {e}")
        print("Skipping URL example...")

    # Example 2: Generate caption
    print("\n[3/4] Generating image caption...")
    try:
        caption_prompt = "Describe this image in detail."
        caption = generate(
            model=model,
            processor=processor,
            prompt=caption_prompt,
            image=image,
            max_tokens=100,
            temperature=0.8,
        )
        print(f"Caption: {caption}")
    except Exception as e:
        print(f"⚠ Error generating caption: {e}")

    # Example 3: Different questions
    print("\n[4/4] Asking multiple questions...")
    questions = [
        "What color is the vehicle?",
        "Is this indoors or outdoors?",
        "What is the weather like?",
    ]

    for q in questions:
        try:
            answer = generate(
                model=model,
                processor=processor,
                prompt=q,
                image=image,
                max_tokens=30,
                temperature=0.5,
            )
            print(f"\nQ: {q}")
            print(f"A: {answer}")
        except Exception as e:
            print(f"⚠ Error: {e}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
