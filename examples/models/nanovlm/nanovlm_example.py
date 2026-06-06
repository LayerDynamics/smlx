#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
nanoVLM Examples with Memory Best Practices

Demonstrates minimal vision-language model capabilities:
1. Basic image captioning
2. Visual question answering
3. Streaming generation
4. Batch processing
5. Custom prompting

MEMORY BEST PRACTICES:
For comprehensive memory management examples, see:
    examples/models/smollm2_135m/smollm2_135m_example.py

Key utilities available:
- watchdog: Automatic memory monitoring
- robust_generate: Auto-retry on OOM
- with_graceful_degradation: Auto-adjust parameters
- auto_select_params: Model-specific safe parameters
- smart_cleanup: Manual memory cleanup

Usage:
    # Interactive mode (default) - pauses between examples
    python nanovlm_example.py

    # Non-interactive mode - runs all examples without pausing
    python nanovlm_example.py --non-interactive
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Auto-detect if running in non-interactive environment
def is_interactive():
    """Check if running in an interactive terminal."""
    return sys.stdin.isatty()

# Add smlx to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from smlx.models.nanoVLM import caption, generate, load, query, stream_generate

# Memory management utilities (see SmolLM2-135M example for detailed usage)
from smlx.config.model_profiles import auto_select_params
from smlx.utils.memory import smart_cleanup, get_active_memory_gb, get_cache_memory_gb
from smlx.utils.watchdog import watchdog
from smlx.utils.robust import robust_generate
from smlx.utils.degradation import with_graceful_degradation


def create_test_image(text="Test Image", color="blue"):
    """Create a simple test image with text and colored background."""
    img = Image.new("RGB", (400, 400), color="white")
    draw = ImageDraw.Draw(img)

    # Draw colored circle
    colors = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0)}
    circle_color = colors.get(color, (0, 0, 255))
    draw.ellipse([100, 100, 300, 300], fill=circle_color, outline="black", width=3)

    # Draw text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()

    draw.text((150, 350), text, fill="black", font=font)

    return img


def example_1_basic_caption():
    """Example 1: Basic image captioning."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Image Captioning")
    print("=" * 80)

    # Load model
    print("\nLoading nanoVLM model...")
    model, processor = load("lusxvr/nanoVLM-222M")
    print("✓ Model loaded!\n")

    # Create test image
    print("Creating test image (blue circle)...")
    image = create_test_image("Blue Circle", "blue")

    # Generate caption
    print("\nGenerating caption...")
    caption_text = caption(model, processor, image, max_tokens=50)

    print(f"\nImage: Blue circle on white background")
    print(f"Caption: {caption_text}")

    print("\n✓ Basic captioning complete!")


def example_2_visual_qa():
    """Example 2: Visual question answering."""
    print("\n" + "=" * 80)
    print("Example 2: Visual Question Answering (VQA)")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, processor = load("lusxvr/nanoVLM-222M")
    print("✓ Model loaded!\n")

    # Create test images
    red_image = create_test_image("Red Circle", "red")
    green_image = create_test_image("Green Square", "green")

    # Ask questions
    questions = [
        "What color is the shape?",
        "What shape do you see?",
        "Describe this image briefly.",
    ]

    for i, (image, name) in enumerate(
        [(red_image, "Red Circle"), (green_image, "Green Square")], 1
    ):
        print(f"\nImage {i}: {name}")
        for question in questions:
            answer = query(model, processor, image, question, max_tokens=30)
            print(f"  Q: {question}")
            print(f"  A: {answer}")
        print()

    print("✓ Visual QA complete!")


def example_3_streaming():
    """Example 3: Streaming generation."""
    print("\n" + "=" * 80)
    print("Example 3: Streaming Generation")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, processor = load("lusxvr/nanoVLM-222M")
    print("✓ Model loaded!\n")

    # Create test image
    image = create_test_image("Test", "blue")

    # Stream generation
    print("Streaming response:")
    print("Prompt: Describe this image in detail:\n")
    print("Response: ", end="", flush=True)

    for chunk in stream_generate(
        model, processor, prompt="Describe this image in detail:", image=image, max_tokens=80
    ):
        print(chunk, end="", flush=True)

    print("\n\n✓ Streaming generation complete!")


def example_4_batch_processing():
    """Example 4: Batch processing multiple images."""
    print("\n" + "=" * 80)
    print("Example 4: Batch Processing")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, processor = load("lusxvr/nanoVLM-222M")
    print("✓ Model loaded!\n")

    # Create multiple test images
    images = [
        ("Red Circle", create_test_image("Red", "red")),
        ("Blue Circle", create_test_image("Blue", "blue")),
        ("Green Circle", create_test_image("Green", "green")),
    ]

    print("Processing batch of images...\n")

    for i, (name, image) in enumerate(images, 1):
        print(f"Image {i}: {name}")
        caption_text = caption(model, processor, image, max_tokens=40)
        print(f"  Caption: {caption_text}\n")

    print("✓ Batch processing complete!")


def example_5_custom_prompts():
    """Example 5: Custom prompting strategies."""
    print("\n" + "=" * 80)
    print("Example 5: Custom Prompting")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, processor = load("lusxvr/nanoVLM-222M")
    print("✓ Model loaded!\n")

    # Create test image
    image = create_test_image("Sample", "red")

    # Different prompting styles
    prompts = [
        ("Simple", "What is this?"),
        ("Detailed", "Provide a detailed description of this image:"),
        ("Specific", "What shape and color do you see?"),
        ("Creative", "Describe this image as if you're writing a story:"),
    ]

    print("Testing different prompting styles:\n")

    for style, prompt in prompts:
        print(f"{style} Prompt: \"{prompt}\"")
        response = generate(model, processor, prompt, image, max_tokens=60)
        print(f"  Response: {response}\n")

    print("✓ Custom prompting complete!")


def example_6_performance_test():
    """Example 6: Performance benchmarking."""
    print("\n" + "=" * 80)
    print("Example 6: Performance Benchmarking")
    print("=" * 80)

    import time

    # Load model
    print("\nLoading model...")
    start = time.time()
    model, processor = load("lusxvr/nanoVLM-222M")
    load_time = time.time() - start
    print(f"✓ Model loaded in {load_time:.2f}s\n")

    # Create test image
    image = create_test_image("Benchmark", "blue")

    # Benchmark generation
    print("Running performance benchmark...")
    num_runs = 5
    times = []

    for i in range(num_runs):
        start = time.time()
        _ = generate(
            model, processor, prompt="Describe:", image=image, max_tokens=50
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s ({50/elapsed:.1f} tok/s)")

    avg_time = sum(times) / len(times)
    avg_tps = 50 / avg_time

    print(f"\nAverage:")
    print(f"  Time: {avg_time:.2f}s")
    print(f"  Tokens/sec: {avg_tps:.1f}")

    print("\n✓ Performance benchmark complete!")


def main(non_interactive=None):
    """Run all examples.

    Args:
        non_interactive: If True, run without pausing between examples (for automated testing).
                        If None, auto-detect based on whether stdin is a tty.
    """
    # Auto-detect non-interactive mode if not explicitly set
    if non_interactive is None:
        non_interactive = not is_interactive()

    print("=" * 80)
    print("nanoVLM - Minimal Vision-Language Model Examples")
    print("=" * 80)
    print("\nNote: nanoVLM is a 222M parameter model optimized for learning")
    print("and experimentation. For production use, consider larger models.")
    print("\nFirst-time run will download the model from HuggingFace Hub.")

    if non_interactive:
        print("\nRunning in non-interactive mode...")

    examples = [
        ("Basic Image Captioning", example_1_basic_caption),
        ("Visual Question Answering", example_2_visual_qa),
        ("Streaming Generation", example_3_streaming),
        ("Batch Processing", example_4_batch_processing),
        ("Custom Prompting", example_5_custom_prompts),
        ("Performance Benchmarking", example_6_performance_test),
    ]

    for name, example_func in examples:
        try:
            example_func()
            # Only pause for input if in interactive mode
            if not non_interactive:
                input("\nPress Enter to continue to next example...")
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"\n\nError in {name}: {e}")
            import traceback

            traceback.print_exc()

            # Only prompt to continue if in interactive mode
            if not non_interactive:
                cont = input("Continue to next example? (y/n): ").strip().lower()
                if cont != "y":
                    break
            else:
                # In non-interactive mode, continue to next example automatically
                print("Continuing to next example...")
                continue

    # Summary
    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Image captioning")
    print("  ✓ Visual question answering")
    print("  ✓ Streaming generation")
    print("  ✓ Batch processing")
    print("  ✓ Custom prompting")
    print("  ✓ Performance benchmarking")

    print("\nnanoVLM Advantages:")
    print("  - Minimal implementation (~750 lines)")
    print("  - Fast to train (6 hours on H100)")
    print("  - Easy to customize")
    print("  - Perfect for learning VLM architecture")
    print("  - Runs on <1GB RAM")
    print("  - Apache 2.0 license")

    print("\nNext Steps:")
    print("  • Fine-tune for domain-specific tasks")
    print("  • Experiment with custom prompting")
    print("  • Train from scratch on custom data")
    print("  • Quantize for even lower memory usage")
    print("  • Deploy in production applications")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="nanoVLM Examples - Minimal Vision-Language Model Demonstrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default) - pauses between examples
  python nanovlm_example.py

  # Non-interactive mode - runs all examples without pausing (useful for testing)
  python nanovlm_example.py --non-interactive
        """
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run without pausing between examples (for automated testing/CI)'
    )

    args = parser.parse_args()
    main(non_interactive=args.non_interactive)
