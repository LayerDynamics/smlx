#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
SmolVLM-500M-Instruct Examples with Memory Best Practices

Demonstrates vision-language capabilities:
- Image description
- Visual question answering
- Multi-image understanding
- Streaming generation
- Chat interface

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

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from smlx.models.SmolVLM_500M_Instruct import load, generate, stream_generate, chat

# Memory management utilities (see SmolLM2-135M example for detailed usage)
from smlx.config.model_profiles import auto_select_params
from smlx.utils.memory import smart_cleanup, get_active_memory_gb, get_cache_memory_gb
from smlx.utils.watchdog import watchdog
from smlx.utils.robust import robust_generate
from smlx.utils.degradation import with_graceful_degradation


def example_1_basic_image_description():
    """Example 1: Basic image description."""
    print("=" * 70)
    print("Example 1: Basic Image Description")
    print("=" * 70)
    print()

    # Load model
    print("Loading SmolVLM-500M model...")
    model, processor = load("HuggingFaceTB/SmolVLM-500M-Instruct")

    # Example image URL (replace with your own)
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

    print(f"Image: {image_url}")
    print()

    # Generate description
    prompt = "<image>\nDescribe this image in detail."

    print("Generating description...")
    output = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=image_url,
        max_tokens=100,
        temperature=0.7,
        verbose=True,
    )

    print()
    print("Description:")
    print(output)
    print()


def example_2_visual_question_answering():
    """Example 2: Visual question answering."""
    print("=" * 70)
    print("Example 2: Visual Question Answering")
    print("=" * 70)
    print()

    from smlx.models.SmolVLM_256M import load

    model, processor = load("HuggingFaceTB/SmolVLM-500M-Instruct")

    # Example image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

    # Ask specific questions
    questions = [
        "What color is the car?",
        "What type of vehicle is this?",
        "What's the setting or environment?",
    ]

    print(f"Image: {image_url}")
    print()

    for q in questions:
        print(f"Q: {q}")
        prompt = f"<image>\n{q}"

        output = generate(
            model=model,
            processor=processor,
            prompt=prompt,
            image=image_url,
            max_tokens=50,
            temperature=0.3,  # Lower temperature for more focused answers
        )

        print(f"A: {output}")
        print()


def example_3_streaming_generation():
    """Example 3: Streaming generation for real-time feedback."""
    print("=" * 70)
    print("Example 3: Streaming Generation")
    print("=" * 70)
    print()

    from smlx.models.SmolVLM_256M import load, stream_generate

    model, processor = load("HuggingFaceTB/SmolVLM-500M-Instruct")

    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

    print(f"Image: {image_url}")
    print("Prompt: Tell me about this image.")
    print()
    print("Response (streaming): ", end="", flush=True)

    for text in stream_generate(
        model=model,
        processor=processor,
        prompt="<image>\nTell me about this image.",
        image=image_url,
        max_tokens=80,
        temperature=0.7,
    ):
        print(text, end="", flush=True)

    print()
    print()


def example_4_chat_interface():
    """Example 4: Chat-style interaction."""
    print("=" * 70)
    print("Example 4: Chat Interface")
    print("=" * 70)
    print()

    from smlx.models.SmolVLM_256M import load, chat

    model, processor = load("HuggingFaceTB/SmolVLM-500M-Instruct")

    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

    print(f"Image: {image_url}")
    print()

    # Conversation with multiple turns
    messages = [{"role": "user", "content": "What do you see in this image?"}]

    print(f"User: {messages[0]['content']}")
    response = chat(
        model=model,
        processor=processor,
        messages=messages,
        image=image_url,
        max_tokens=60,
        temperature=0.7,
    )

    print(f"Assistant: {response}")
    print()

    # Follow-up question
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "What might this be used for?"})

    print(f"User: {messages[-1]['content']}")
    response = chat(
        model=model,
        processor=processor,
        messages=messages,
        max_tokens=60,
        temperature=0.7,
    )

    print(f"Assistant: {response}")
    print()


def example_5_local_image():
    """Example 5: Using local image file."""
    print("=" * 70)
    print("Example 5: Local Image File")
    print("=" * 70)
    print()

    from smlx.models.SmolVLM_256M import load, generate

    model, processor = load("HuggingFaceTB/SmolVLM-500M-Instruct")

    # Note: This assumes you have a local image file
    # Replace with your own image path
    print("Note: This example requires a local image file.")
    print("Update the image_path variable with your own image.")
    print()

    # image_path = "path/to/your/image.jpg"
    #
    # output = generate(
    #     model=model,
    #     processor=processor,
    #     prompt="<image>\nWhat is in this image?",
    #     image=image_path,
    #     max_tokens=100,
    # )
    #
    # print(output)

    print("Example code:")
    print(
        """
    from smlx.models.SmolVLM_256M import load, generate

    model, processor = load()

    output = generate(
        model=model,
        processor=processor,
        prompt="<image>\\nWhat is in this image?",
        image="path/to/image.jpg",
        max_tokens=100,
    )

    print(output)
    """
    )
    print()


def example_6_multi_image():
    """Example 6: Multiple images (if supported)."""
    print("=" * 70)
    print("Example 6: Multiple Images")
    print("=" * 70)
    print()

    print("SmolVLM-500M supports processing multiple images in a single prompt.")
    print()

    print("Example code:")
    print(
        """
    from smlx.models.SmolVLM_256M import load, generate

    model, processor = load()

    images = [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
    ]

    output = generate(
        model=model,
        processor=processor,
        prompt="<image>\\n<image>\\nCompare these two images.",
        image=images,
        max_tokens=150,
    )

    print(output)
    """
    )
    print()


def main():
    """Run all examples."""
    print()
    print("=" * 70)
    print("SmolVLM-500M-Instruct Examples")
    print("=" * 70)
    print()
    print("Vision-Language Model with:")
    print("- SigLIP 93M vision encoder")
    print("- SmolLM2-360M language model")
    print("- ~500M total parameters")
    print()

    try:
        # Run examples
        example_1_basic_image_description()
        example_2_visual_question_answering()
        example_3_streaming_generation()
        example_4_chat_interface()
        example_5_local_image()
        example_6_multi_image()

        print("=" * 70)
        print("✓ All examples completed!")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"Error running examples: {e}")
        print()
        print("Make sure you have:")
        print("1. Installed required dependencies (pip install -e '.[vision]')")
        print("2. Downloaded the SmolVLM-500M model")
        print("3. Have internet connection for URL images")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
