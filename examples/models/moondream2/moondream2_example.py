#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Moondream2 Vision-Language Model Examples with Memory Best Practices

Demonstrates:
- Image captioning
- Visual question answering (VQA)
- Object detection
- Spatial localization (pointing)
- Multi-turn conversations
- Streaming generation

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

from smlx.models.Moondream2 import (
    load,
    caption,
    query,
    detect,
    point,
    stream_generate,
)
from PIL import Image, ImageDraw
import tempfile
import urllib.request

# Memory management utilities (see SmolLM2-135M example for detailed usage)
from smlx.config.model_profiles import auto_select_params
from smlx.utils.memory import smart_cleanup, get_active_memory_gb, get_cache_memory_gb
from smlx.utils.watchdog import watchdog
from smlx.utils.robust import robust_generate
from smlx.utils.degradation import with_graceful_degradation


def download_sample_image() -> Image.Image:
    """Download a sample image for testing."""
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        urllib.request.urlretrieve(url, f.name)
        image = Image.open(f.name)

    return image


def example_1_image_captioning():
    """Example 1: Image captioning at different lengths."""
    print("=" * 70)
    print("Example 1: Image Captioning")
    print("=" * 70)
    print()

    # Load model
    print("Loading Moondream2 model...")
    model, tokenizer = load("vikhyatk/moondream2")

    # Load sample image
    print("Loading sample image...")
    image = download_sample_image()

    print(f"Image size: {image.size}")
    print()

    # Short caption
    print("Short Caption:")
    short_caption = caption(model, tokenizer, image, length="short", max_tokens=50)
    print(f"  {short_caption}")
    print()

    # Normal caption
    print("Normal Caption:")
    normal_caption = caption(model, tokenizer, image, length="normal", max_tokens=100)
    print(f"  {normal_caption}")
    print()

    # Long caption
    print("Long Caption:")
    long_caption = caption(model, tokenizer, image, length="long", max_tokens=200)
    print(f"  {long_caption}")
    print()


def example_2_visual_qa():
    """Example 2: Visual question answering."""
    print("=" * 70)
    print("Example 2: Visual Question Answering (VQA)")
    print("=" * 70)
    print()

    model, tokenizer = load("vikhyatk/moondream2")
    image = download_sample_image()

    # Ask various questions
    questions = [
        "What is the main object in this image?",
        "What color is the car?",
        "Where is the car located?",
        "Are there any people visible in the image?",
        "What time of day does it appear to be?",
    ]

    for question in questions:
        print(f"Q: {question}")
        answer = query(model, tokenizer, image, question, max_tokens=100)
        print(f"A: {answer}")
        print()


def example_3_object_detection():
    """Example 3: Object detection with bounding boxes."""
    print("=" * 70)
    print("Example 3: Object Detection")
    print("=" * 70)
    print()

    model, tokenizer = load("vikhyatk/moondream2")
    image = download_sample_image()

    # Detect objects
    object_types = ["car", "wheels", "windows"]

    for obj_type in object_types:
        print(f"Detecting: {obj_type}")
        detections = detect(
            model,
            tokenizer,
            image,
            obj_type,
            confidence_threshold=0.5,
            max_tokens=150,
        )

        if detections:
            print(f"  Found {len(detections)} {obj_type}(s):")
            for i, (x1, y1, x2, y2, conf) in enumerate(detections):
                print(
                    f"    {i + 1}. Box: ({x1}, {y1}, {x2}, {y2}), Confidence: {conf:.2f}"
                )
        else:
            print(f"  No {obj_type} detected")
        print()

    # Draw bounding boxes (example for first object type)
    if object_types:
        detections = detect(
            model, tokenizer, image, object_types[0], max_tokens=150
        )
        if detections:
            draw = ImageDraw.Draw(image)
            for x1, y1, x2, y2, _ in detections:
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            output_path = "output_with_detections.jpg"
            image.save(output_path)
            print(f"Saved image with bounding boxes to: {output_path}")
            print()


def example_4_pointing():
    """Example 4: Spatial localization (pointing)."""
    print("=" * 70)
    print("Example 4: Spatial Localization (Pointing)")
    print("=" * 70)
    print()

    model, tokenizer = load("vikhyatk/moondream2")
    image = download_sample_image()

    # Point to various objects
    queries = [
        "the car",
        "the front wheel",
        "the windshield",
        "the license plate",
    ]

    for obj_query in queries:
        print(f"Pointing to: {obj_query}")
        location = point(model, tokenizer, image, obj_query, max_tokens=100)

        if location:
            x, y = location
            print(f"  Location: ({x}, {y}) pixels")
        else:
            print(f"  Could not locate {obj_query}")
        print()


def example_5_streaming():
    """Example 5: Streaming generation."""
    print("=" * 70)
    print("Example 5: Streaming Generation")
    print("=" * 70)
    print()

    model, tokenizer = load("vikhyatk/moondream2")
    image = download_sample_image()

    prompt = "Provide a detailed description of this image, including colors, objects, and setting."

    print(f"Prompt: {prompt}")
    print("Response (streaming): ", end="", flush=True)

    for token in stream_generate(
        model,
        tokenizer,
        image,
        prompt,
        max_tokens=200,
        temperature=0.3,
    ):
        print(token, end="", flush=True)

    print("\n")


def example_6_multi_turn():
    """Example 6: Multi-turn conversation (simulated)."""
    print("=" * 70)
    print("Example 6: Multi-Turn Conversation")
    print("=" * 70)
    print()

    model, tokenizer = load("vikhyatk/moondream2")
    image = download_sample_image()

    # Simulate conversation by building context
    conversation = [
        "What is in this image?",
        "What color is it?",
        "Can you describe its condition?",
        "Where might this photo have been taken?",
    ]

    print("Starting multi-turn conversation about the image...")
    print()

    for i, user_query in enumerate(conversation):
        print(f"Turn {i + 1}")
        print(f"User: {user_query}")

        # For simplicity, each turn is independent
        # A true multi-turn would accumulate context
        response = query(model, tokenizer, image, user_query, max_tokens=100)

        print(f"Assistant: {response}")
        print()


def example_7_custom_prompts():
    """Example 7: Custom prompting strategies."""
    print("=" * 70)
    print("Example 7: Custom Prompting Strategies")
    print("=" * 70)
    print()

    model, tokenizer = load("vikhyatk/moondream2")
    image = download_sample_image()

    # Different prompting strategies
    prompts = [
        ("Detailed Analysis", "Analyze this image in detail, covering visual elements, composition, and context."),
        ("List Format", "List 5 key observations about this image in bullet points."),
        ("Counting", "Count and identify all distinct objects visible in this image."),
        ("Scene Understanding", "Describe the scene, including time of day, weather conditions, and setting."),
    ]

    for title, prompt in prompts:
        print(f"{title}:")
        response = query(model, tokenizer, image, prompt, max_tokens=150)
        print(f"  {response}")
        print()


def main():
    """Run all examples."""
    print()
    print("=" * 70)
    print("Moondream2 Vision-Language Model Examples")
    print("=" * 70)
    print()
    print("Model: Moondream2")
    print("Size: 1.8B parameters (~5.2GB FP16, ~1.3GB 4-bit)")
    print()
    print("Capabilities:")
    print("- Image captioning (short, normal, long)")
    print("- Visual question answering (VQA)")
    print("- Object detection with bounding boxes")
    print("- Spatial localization (pointing)")
    print("- Multi-turn conversations")
    print("- Streaming generation")
    print()

    try:
        # Run examples
        example_1_image_captioning()
        example_2_visual_qa()
        example_3_object_detection()
        example_4_pointing()
        example_5_streaming()
        example_6_multi_turn()
        example_7_custom_prompts()

        print("=" * 70)
        print("✓ All examples completed!")
        print("=" * 70)
        print()
        print("Next Steps:")
        print("- Try with your own images")
        print("- Experiment with different prompts")
        print("- Fine-tune with LoRA for custom domains")
        print("- Deploy as an API server")
        print()

    except Exception as e:
        print(f"Error running examples: {e}")
        print()
        print("Make sure you have:")
        print("1. Installed required dependencies (pip install smlx[vision])")
        print("2. Internet connection for downloading the model")
        print("3. Sufficient memory (~6GB for FP16 model)")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
