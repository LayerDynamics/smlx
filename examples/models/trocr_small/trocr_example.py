#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
TrOCR-small Examples.

Demonstrates optical character recognition capabilities:
1. Basic printed text recognition
2. Handwritten text recognition
3. Batch OCR processing
4. OCR with confidence scores
5. Image preprocessing for better accuracy
6. Document digitization workflow
7. Receipt parsing
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add smlx to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from smlx.models.TrOCR_small import (
    load,
    preprocess_image,
    recognize,
    recognize_batch,
    recognize_with_confidence,
)


def create_text_image(
    text: str,
    image_size: tuple = (384, 384),
    font_size: int = 32,
    background_color: str = "white",
    text_color: str = "black",
) -> Image.Image:
    """Create synthetic image with text.

    Args:
        text: Text to render
        image_size: Image size (width, height)
        font_size: Font size
        background_color: Background color
        text_color: Text color

    Returns:
        PIL Image with text
    """
    # Create blank image
    image = Image.new("RGB", image_size, background_color)
    draw = ImageDraw.Draw(image)

    # Try to use default font, fallback to basic
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()

    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2

    # Draw text
    draw.text((x, y), text, fill=text_color, font=font)

    return image


def example_1_basic_printed_text():
    """Example 1: Basic printed text recognition."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Printed Text Recognition")
    print("=" * 80)

    # Load model
    print("\nLoading TrOCR model (printed variant)...")
    model, processor = load("printed")

    # Create synthetic printed text image
    print("Creating synthetic text image...")
    text_to_recognize = "Hello World!"
    image = create_text_image(text_to_recognize)

    # Recognize text
    print("\nRecognizing text...")
    recognized_text = recognize(model, processor, image)

    print(f"\nOriginal text: {text_to_recognize}")
    print(f"Recognized text: {recognized_text}")

    # Note about synthetic images
    print("\nNote: Recognition quality depends on proper model weights.")
    print("With random weights, output will be random tokens.")


def example_2_handwritten_text():
    """Example 2: Handwritten text recognition."""
    print("\n" + "=" * 80)
    print("Example 2: Handwritten Text Recognition")
    print("=" * 80)

    # Load handwritten variant
    print("\nLoading TrOCR model (handwritten variant)...")
    model, processor = load("handwritten")

    # Create synthetic handwritten-style text
    print("Creating handwritten-style text image...")
    text = "Sample Note"
    image = create_text_image(text, font_size=28)  # Slightly smaller font

    # Recognize
    print("\nRecognizing handwritten text...")
    recognized_text = recognize(model, processor, image)

    print(f"\nOriginal text: {text}")
    print(f"Recognized text: {recognized_text}")

    print("\nNote: Handwritten accuracy depends on:")
    print("  - Training data similarity")
    print("  - Handwriting consistency")
    print("  - Image quality")


def example_3_batch_processing():
    """Example 3: Batch OCR processing."""
    print("\n" + "=" * 80)
    print("Example 3: Batch OCR Processing")
    print("=" * 80)

    # Load model
    print("\nLoading TrOCR model...")
    model, processor = load("printed")

    # Create multiple text images
    print("Creating multiple text images...")
    texts = [
        "Document 1",
        "Page 2",
        "Receipt #3",
        "Form 4",
    ]

    images = [create_text_image(text) for text in texts]

    # Batch recognize
    print("\nProcessing batch...")
    recognized_texts = recognize_batch(model, processor, images)

    print("\nResults:")
    for i, (original, recognized) in enumerate(zip(texts, recognized_texts), 1):
        print(f"  {i}. Original: '{original}' → Recognized: '{recognized}'")


def example_4_confidence_scores():
    """Example 4: OCR with confidence scores."""
    print("\n" + "=" * 80)
    print("Example 4: OCR with Confidence Scores")
    print("=" * 80)

    # Load model
    print("\nLoading TrOCR model...")
    model, processor = load("printed")

    # Create text images with varying quality
    print("Creating text images...")
    clear_text = "Clear Text"
    clear_image = create_text_image(clear_text)

    # Recognize with confidence
    print("\nRecognizing with confidence scores...")
    text, confidence = recognize_with_confidence(model, processor, clear_image)

    print(f"\nRecognized: '{text}'")
    print(f"Confidence: {confidence:.3f}")

    print("\nConfidence interpretation:")
    if confidence > 0.9:
        print("  ✓ High confidence - very likely correct")
    elif confidence > 0.7:
        print("  ⚠ Medium confidence - likely correct")
    else:
        print("  ✗ Low confidence - may need verification")


def example_5_image_preprocessing():
    """Example 5: Image preprocessing for better accuracy."""
    print("\n" + "=" * 80)
    print("Example 5: Image Preprocessing")
    print("=" * 80)

    # Load model
    print("\nLoading TrOCR model...")
    model, processor = load("printed")

    # Create low-contrast image
    print("Creating low-quality text image...")
    text = "Low Quality"
    image = create_text_image(text, background_color="lightgray", text_color="gray")

    # Recognize without preprocessing
    print("\nRecognizing without preprocessing...")
    text_no_preprocess = recognize(model, processor, image)
    print(f"  Result: '{text_no_preprocess}'")

    # Preprocess image
    print("\nPreprocessing image (contrast/sharpness enhancement)...")
    enhanced_image = preprocess_image(image, enhance=True)

    # Recognize with preprocessing
    print("Recognizing with preprocessing...")
    text_with_preprocess = recognize(model, processor, enhanced_image)
    print(f"  Result: '{text_with_preprocess}'")

    print("\nPreprocessing can improve accuracy on:")
    print("  - Low contrast images")
    print("  - Blurry images")
    print("  - Poor quality scans")


def example_6_document_digitization():
    """Example 6: Document digitization workflow."""
    print("\n" + "=" * 80)
    print("Example 6: Document Digitization Workflow")
    print("=" * 80)

    # Load model
    print("\nLoading TrOCR model...")
    model, processor = load("printed")

    # Simulate multi-line document (process line by line)
    print("\nSimulating multi-line document...")
    document_lines = [
        "Invoice #12345",
        "Date: 2024-01-15",
        "Amount: $99.99",
        "Status: Paid",
    ]

    print("\nProcessing document line by line:")
    digitized_document = []

    for i, line_text in enumerate(document_lines, 1):
        # Create image for each line
        line_image = create_text_image(line_text, font_size=28)

        # Recognize
        recognized = recognize(model, processor, line_image)

        print(f"  Line {i}: '{line_text}' → '{recognized}'")
        digitized_document.append(recognized)

    print("\nDigitized document:")
    print("\n".join(f"  {line}" for line in digitized_document))

    print("\nNote: For real documents:")
    print("  - Use layout analysis to extract lines")
    print("  - Consider line segmentation preprocessing")
    print("  - May need table/structure detection")


def example_7_receipt_parsing():
    """Example 7: Receipt parsing."""
    print("\n" + "=" * 80)
    print("Example 7: Receipt Parsing")
    print("=" * 80)

    # Load model
    print("\nLoading TrOCR model...")
    model, processor = load("printed")

    # Simulate receipt fields
    print("\nSimulating receipt fields...")
    receipt_fields = {
        "store": "COFFEE SHOP",
        "item1": "Latte - $4.50",
        "item2": "Muffin - $3.25",
        "total": "Total: $7.75",
    }

    print("\nExtracting text from receipt fields:")
    extracted_data = {}

    for field_name, field_text in receipt_fields.items():
        # Create image
        field_image = create_text_image(field_text, font_size=24)

        # Recognize
        recognized = recognize(model, processor, field_image)

        print(f"  {field_name}: '{recognized}'")
        extracted_data[field_name] = recognized

    print("\nExtracted receipt data:")
    for key, value in extracted_data.items():
        print(f"  {key}: {value}")

    print("\nUse cases:")
    print("  - Expense tracking")
    print("  - Receipt archiving")
    print("  - Automated bookkeeping")


def example_8_model_comparison():
    """Example 8: Compare printed vs handwritten models."""
    print("\n" + "=" * 80)
    print("Example 8: Model Variant Comparison")
    print("=" * 80)

    # Load both variants
    print("\nLoading both model variants...")
    printed_model, printed_proc = load("printed")
    handwritten_model, handwritten_proc = load("handwritten")

    # Create test image
    print("Creating test image...")
    text = "Test Text"
    image = create_text_image(text)

    # Test with both models
    print("\nComparing model variants:")

    print("  Printed model:")
    printed_result = recognize(printed_model, printed_proc, image)
    print(f"    Result: '{printed_result}'")

    print("  Handwritten model:")
    handwritten_result = recognize(handwritten_model, handwritten_proc, image)
    print(f"    Result: '{handwritten_result}'")

    print("\nModel selection guide:")
    print("  - Printed: Documents, receipts, forms, typed text")
    print("  - Handwritten: Notes, signatures, handwritten forms")
    print("  - Choose based on your specific use case")


def main():
    """Run all examples."""
    print("=" * 80)
    print("TrOCR-small - Optical Character Recognition Examples")
    print("=" * 80)
    print("\nNote: These examples use synthetic text images.")
    print("For real OCR, use actual scanned documents or photos.")
    print("\nWarning: TrOCR weights need to be downloaded from HuggingFace.")
    print("Examples will run with random weights for demonstration.")

    examples = [
        ("Basic Printed Text", example_1_basic_printed_text),
        ("Handwritten Text", example_2_handwritten_text),
        ("Batch Processing", example_3_batch_processing),
        ("Confidence Scores", example_4_confidence_scores),
        ("Image Preprocessing", example_5_image_preprocessing),
        ("Document Digitization", example_6_document_digitization),
        ("Receipt Parsing", example_7_receipt_parsing),
        ("Model Comparison", example_8_model_comparison),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"\n\nError in {name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Printed text recognition")
    print("  ✓ Handwritten text recognition")
    print("  ✓ Batch processing")
    print("  ✓ Confidence scoring")
    print("  ✓ Image preprocessing")
    print("  ✓ Document digitization")
    print("  ✓ Receipt parsing")

    print("\nModel Advantages:")
    print("  - Transformer-based (no external OCR engine)")
    print("  - Lightweight (~60M parameters)")
    print("  - Two variants (printed/handwritten)")
    print("  - End-to-end trainable")
    print("  - MIT license")
    print("  - Fast inference on M4")

    print("\nCommon Applications:")
    print("  - Document digitization")
    print("  - Receipt/invoice processing")
    print("  - Form field extraction")
    print("  - Handwritten note recognition")
    print("  - On-device OCR")
    print("  - Privacy-sensitive document processing")


if __name__ == "__main__":
    main()
