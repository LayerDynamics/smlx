#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
TinyLLaVA Vision-Language Model Examples with Memory Best Practices

This script demonstrates the capabilities of TinyLLaVA models:
1. Basic image captioning
2. Visual question answering (VQA)
3. Multi-turn conversation with images
4. Streaming generation
5. Custom generation parameters
6. Using different variants (1.5B, 2.0B, 3.1B)

Requirements:
    - PIL (Pillow) for image loading
    - An image file or URL

Usage:
    python tinyllava_example.py [--image PATH] [--variant 1.5b|2.0b|3.1b]

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

import argparse
from pathlib import Path
import sys

try:
    from PIL import Image
    import requests
    from io import BytesIO
except ImportError:
    print("Error: PIL (Pillow) and requests are required")
    print("Install with: pip install Pillow requests")
    sys.exit(1)

# Add smlx to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from smlx.models.TinyLLaVA import load, generate, stream_generate, caption, query

# Memory management utilities (see SmolLM2-135M example for detailed usage)
from smlx.config.model_profiles import auto_select_params
from smlx.utils.memory import smart_cleanup, get_active_memory_gb, get_cache_memory_gb
from smlx.utils.watchdog import watchdog
from smlx.utils.robust import robust_generate
from smlx.utils.degradation import with_graceful_degradation


def load_sample_image(image_source: str) -> Image.Image:
    """Load an image from file path or URL."""
    if image_source.startswith(("http://", "https://")):
        print(f"Loading image from URL: {image_source}")
        response = requests.get(image_source, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        print(f"Loading image from file: {image_source}")
        image = Image.open(image_source)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def example_1_basic_captioning(model, processor, image):
    """Example 1: Basic image captioning."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Image Captioning")
    print("=" * 80)

    print("\nGenerating caption...")
    result = caption(
        model=model,
        processor=processor,
        image=image,
        prompt="Describe this image in detail.",
        max_tokens=300,
        temperature=0.7,
    )

    print(f"\nCaption: {result}")


def example_2_visual_qa(model, processor, image):
    """Example 2: Visual Question Answering."""
    print("\n" + "=" * 80)
    print("Example 2: Visual Question Answering")
    print("=" * 80)

    questions = [
        "What is the main subject of this image?",
        "What colors are prominent in this image?",
        "What is happening in this scene?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        answer = query(
            model=model,
            processor=processor,
            image=image,
            question=question,
            max_tokens=200,
            temperature=0.7,
        )
        print(f"Answer: {answer}")


def example_3_detailed_description(model, processor, image):
    """Example 3: Detailed image description with custom prompt."""
    print("\n" + "=" * 80)
    print("Example 3: Detailed Description")
    print("=" * 80)

    custom_prompt = """Please provide a comprehensive description of this image, including:
1. The main subjects or objects
2. The setting and environment
3. Colors, lighting, and mood
4. Any actions or interactions
5. Notable details"""

    print("\nGenerating detailed description...")
    result = generate(
        model=model,
        processor=processor,
        prompt=custom_prompt,
        image=image,
        max_tokens=400,
        temperature=0.7,
        top_p=0.9,
    )

    print(f"\nDetailed Description:\n{result}")


def example_4_streaming_generation(model, processor, image):
    """Example 4: Streaming generation for real-time output."""
    print("\n" + "=" * 80)
    print("Example 4: Streaming Generation")
    print("=" * 80)

    print("\nQuestion: What can you tell me about this image?")
    print("Answer (streaming): ", end="", flush=True)

    for text in stream_generate(
        model=model,
        processor=processor,
        prompt="What can you tell me about this image?",
        image=image,
        max_tokens=300,
        temperature=0.7,
    ):
        print(text, end="", flush=True)

    print("\n")


def example_5_multi_turn_conversation(model, processor, image):
    """Example 5: Multi-turn conversation about an image."""
    print("\n" + "=" * 80)
    print("Example 5: Multi-turn Conversation")
    print("=" * 80)

    conversation_turns = [
        "Describe what you see in this image.",
        "What is the mood or atmosphere of this scene?",
        "If you had to give this image a title, what would it be?",
    ]

    print("\nMulti-turn conversation:")
    for i, prompt in enumerate(conversation_turns, 1):
        print(f"\n[Turn {i}]")
        print(f"User: {prompt}")

        response = generate(
            model=model,
            processor=processor,
            prompt=prompt,
            image=image,
            max_tokens=200,
            temperature=0.7,
        )

        print(f"Assistant: {response}")


def example_6_generation_parameters(model, processor, image):
    """Example 6: Different generation parameters."""
    print("\n" + "=" * 80)
    print("Example 6: Generation Parameters")
    print("=" * 80)

    prompt = "Describe this image."

    # Low temperature (more deterministic)
    print("\n--- Low Temperature (0.1) - More Focused ---")
    result = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=image,
        max_tokens=150,
        temperature=0.1,
    )
    print(f"Result: {result}")

    # High temperature (more creative)
    print("\n--- High Temperature (1.0) - More Creative ---")
    result = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=image,
        max_tokens=150,
        temperature=1.0,
    )
    print(f"Result: {result}")

    # Greedy decoding (temperature=0)
    print("\n--- Greedy Decoding (temperature=0) - Most Deterministic ---")
    result = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=image,
        max_tokens=150,
        temperature=0.0,
    )
    print(f"Result: {result}")


def example_7_object_counting(model, processor, image):
    """Example 7: Object counting and identification."""
    print("\n" + "=" * 80)
    print("Example 7: Object Counting")
    print("=" * 80)

    questions = [
        "How many people are in this image?",
        "List all the objects you can identify in this image.",
        "What is the largest object in this image?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        answer = query(
            model=model,
            processor=processor,
            image=image,
            question=question,
            max_tokens=200,
            temperature=0.3,  # Lower temperature for factual responses
        )
        print(f"Answer: {answer}")


def main():
    """Run TinyLLaVA examples."""
    parser = argparse.ArgumentParser(
        description="TinyLLaVA Vision-Language Model Examples"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
        help="Path or URL to image (default: sample car image)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["1.5b", "2.0b", "3.1b"],
        default="1.5b",
        help="TinyLLaVA variant to use (default: 1.5b)",
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run specific example (default: run all)",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Use lazy weight loading",
    )

    args = parser.parse_args()

    # Print header
    print("=" * 80)
    print("TinyLLaVA Vision-Language Model Examples")
    print("=" * 80)
    print(f"\nVariant: {args.variant.upper()}")
    print(f"Image: {args.image}")

    # Load model
    print(f"\nLoading TinyLLaVA-{args.variant.upper()} model...")

    variant_map = {
        "1.5b": "bczhou/TinyLLaVA-1.5B",
        "2.0b": "bczhou/TinyLLaVA-2.0B",
        "3.1b": "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",
    }

    model, processor = load(
        path_or_hf_repo=variant_map[args.variant],
        variant=args.variant,
        lazy=args.lazy,
    )

    print("Model loaded successfully!")

    # Load image
    try:
        image = load_sample_image(args.image)
        print(f"Image loaded: {image.size[0]}x{image.size[1]} pixels")
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Using default sample image...")
        image = load_sample_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        )

    # Run examples
    examples = {
        1: example_1_basic_captioning,
        2: example_2_visual_qa,
        3: example_3_detailed_description,
        4: example_4_streaming_generation,
        5: example_5_multi_turn_conversation,
        6: example_6_generation_parameters,
        7: example_7_object_counting,
    }

    if args.example:
        # Run specific example
        examples[args.example](model, processor, image)
    else:
        # Run all examples
        for example_func in examples.values():
            try:
                example_func(model, processor, image)
            except KeyboardInterrupt:
                print("\n\nExamples interrupted by user.")
                break
            except Exception as e:
                print(f"\n\nError in example: {e}")
                import traceback

                traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print(f"\nModel: TinyLLaVA-{args.variant.upper()}")
    print("Architecture:")
    print("  - Vision: SigLIP-so400m (1152 hidden, 27 layers)")
    if args.variant == "1.5b":
        print("  - Language: TinyLlama (2048 hidden, 22 layers)")
        print("  - Parameters: ~1.5B")
    elif args.variant == "2.0b":
        print("  - Language: StableLM-2 (2048 hidden, 24 layers)")
        print("  - Parameters: ~2.0B")
    else:
        print("  - Language: Phi-2 (2560 hidden, 32 layers)")
        print("  - Parameters: ~3.1B")
    print("  - Connector: MLP2x-GELU")

    print("\nCapabilities demonstrated:")
    print("  ✓ Image captioning")
    print("  ✓ Visual question answering")
    print("  ✓ Multi-turn conversation")
    print("  ✓ Streaming generation")
    print("  ✓ Custom generation parameters")
    print("  ✓ Object counting and identification")

    print("\nFor more information:")
    print("  - 1.5B: https://huggingface.co/bczhou/TinyLLaVA-1.5B")
    print("  - 2.0B: https://huggingface.co/bczhou/TinyLLaVA-2.0B")
    print("  - 3.1B: https://huggingface.co/tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B")


if __name__ == "__main__":
    main()
