#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Donut-base Document Understanding Examples

This script demonstrates various document understanding tasks using Donut:
1. Document parsing (structured extraction)
2. Document VQA (visual question answering)
3. Text extraction (OCR-free)
4. Document classification
5. Batch processing
6. Performance benchmarking

IMPORTANT NOTE:
    These examples use a reference implementation with placeholder outputs.
    For production use, load pre-trained weights from HuggingFace Hub:
    - "naver-clova-ix/donut-base-finetuned-docvqa" for VQA
    - "naver-clova-ix/donut-base-finetuned-rvlcdip" for classification
    - "naver-clova-ix/donut-base-finetuned-cord-v2" for receipt parsing

Usage:
    python donut_example.py
"""

import time

from PIL import Image, ImageDraw, ImageFont


def create_sample_document(doc_type: str = "invoice") -> Image.Image:
    """
    Create a sample document image for testing.

    Args:
        doc_type: Type of document ("invoice", "receipt", "form")

    Returns:
        PIL Image of sample document
    """
    # Create white background
    img = Image.new("RGB", (800, 1000), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fallback to default
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    if doc_type == "invoice":
        # Draw invoice header
        draw.text((50, 50), "INVOICE", fill="black", font=title_font)
        draw.text((50, 120), "Invoice #: INV-2024-001", fill="black", font=text_font)
        draw.text((50, 160), "Date: January 15, 2024", fill="black", font=text_font)

        # Draw company info
        draw.text((50, 220), "ABC Company Inc.", fill="black", font=text_font)
        draw.text((50, 250), "123 Main Street", fill="black", font=small_font)
        draw.text((50, 275), "New York, NY 10001", fill="black", font=small_font)

        # Draw items table
        draw.text((50, 350), "Item", fill="black", font=text_font)
        draw.text((400, 350), "Quantity", fill="black", font=text_font)
        draw.text((600, 350), "Price", fill="black", font=text_font)
        draw.line([(50, 380), (750, 380)], fill="black", width=2)

        draw.text((50, 400), "Widget A", fill="black", font=small_font)
        draw.text((400, 400), "5", fill="black", font=small_font)
        draw.text((600, 400), "$50.00", fill="black", font=small_font)

        draw.text((50, 440), "Widget B", fill="black", font=small_font)
        draw.text((400, 440), "3", fill="black", font=small_font)
        draw.text((600, 440), "$75.00", fill="black", font=small_font)

        draw.line([(50, 480), (750, 480)], fill="black", width=2)
        draw.text((400, 500), "Total:", fill="black", font=text_font)
        draw.text((600, 500), "$325.00", fill="black", font=text_font)

    elif doc_type == "receipt":
        # Draw receipt header
        draw.text((300, 50), "RECEIPT", fill="black", font=title_font)
        draw.text((50, 120), "Store #425", fill="black", font=text_font)
        draw.text((50, 160), "Date: 2024-01-15 14:32", fill="black", font=small_font)

        # Draw items
        y = 220
        items = [
            ("Coffee", "$4.50"),
            ("Sandwich", "$8.95"),
            ("Cookie", "$2.50"),
        ]
        for item, price in items:
            draw.text((50, y), item, fill="black", font=small_font)
            draw.text((600, y), price, fill="black", font=small_font)
            y += 40

        draw.line([(50, y), (750, y)], fill="black", width=2)
        y += 20
        draw.text((50, y), "Subtotal:", fill="black", font=text_font)
        draw.text((600, y), "$15.95", fill="black", font=text_font)
        y += 40
        draw.text((50, y), "Tax:", fill="black", font=small_font)
        draw.text((600, y), "$1.28", fill="black", font=small_font)
        y += 40
        draw.text((50, y), "Total:", fill="black", font=text_font)
        draw.text((600, y), "$17.23", fill="black", font=text_font)

    elif doc_type == "form":
        # Draw form header
        draw.text((250, 50), "APPLICATION FORM", fill="black", font=title_font)

        # Draw form fields
        fields = [
            ("Name:", "John Smith"),
            ("Address:", "456 Oak Avenue"),
            ("City:", "San Francisco"),
            ("State:", "CA"),
            ("ZIP:", "94102"),
            ("Phone:", "(415) 555-1234"),
            ("Email:", "john.smith@email.com"),
        ]
        y = 150
        for label, value in fields:
            draw.text((50, y), label, fill="black", font=text_font)
            draw.text((250, y), value, fill="black", font=small_font)
            draw.line([(250, y + 30), (700, y + 30)], fill="gray", width=1)
            y += 70

    return img


def example_1_document_parsing():
    """
    Example 1: Parse document and extract structured information.

    Demonstrates:
    - Loading Donut model
    - Parsing document to structured format
    - Extracting specific fields
    """
    print("=" * 70)
    print("Example 1: Document Parsing")
    print("=" * 70)

    from smlx.models.Donut_base import load, parse_document

    # Load model
    print("\n1. Loading Donut model...")
    model, processor = load("naver-clova-ix/donut-base")

    # Create sample invoice
    print("\n2. Creating sample invoice...")
    invoice_image = create_sample_document("invoice")

    # Parse document
    print("\n3. Parsing document...")
    result = parse_document(model=model, processor=processor, image=invoice_image, task="document")

    print(f"\nParsed result: {result}")
    print("\nNote: This is a placeholder output. Load pre-trained weights for actual parsing.")


def example_2_document_vqa():
    """
    Example 2: Document Visual Question Answering.

    Demonstrates:
    - Asking questions about document content
    - Multiple questions on same document
    - Extracting specific information
    """
    print("\n" + "=" * 70)
    print("Example 2: Document VQA")
    print("=" * 70)

    from smlx.models.Donut_base import answer_question, load

    # Load model
    print("\n1. Loading Donut model for VQA...")
    model, processor = load("naver-clova-ix/donut-base-finetuned-docvqa")

    # Create sample invoice
    print("\n2. Creating sample invoice...")
    invoice_image = create_sample_document("invoice")

    # Ask questions
    questions = [
        "What is the invoice number?",
        "What is the total amount?",
        "When was the invoice issued?",
        "What is the company name?",
    ]

    print("\n3. Asking questions about the document...")
    for question in questions:
        answer = answer_question(
            model=model, processor=processor, image=invoice_image, question=question
        )
        print(f"\nQ: {question}")
        print(f"A: {answer}")

    print("\nNote: These are placeholder answers. Load pre-trained weights for actual VQA.")


def example_3_text_extraction():
    """
    Example 3: OCR-free text extraction.

    Demonstrates:
    - Extracting all text from document
    - No OCR pipeline required
    - Handling different document types
    """
    print("\n" + "=" * 70)
    print("Example 3: Text Extraction")
    print("=" * 70)

    from smlx.models.Donut_base import extract_text, load

    # Load model
    print("\n1. Loading Donut model...")
    model, processor = load("naver-clova-ix/donut-base")

    # Test different document types
    doc_types = ["invoice", "receipt", "form"]

    for doc_type in doc_types:
        print(f"\n2. Extracting text from {doc_type}...")
        document = create_sample_document(doc_type)

        text = extract_text(model=model, processor=processor, image=document)

        print(f"\nExtracted text from {doc_type}:")
        print(text)

    print("\nNote: This is placeholder output. Load pre-trained weights for actual extraction.")


def example_4_document_classification():
    """
    Example 4: Document classification.

    Demonstrates:
    - Classifying document types
    - Custom class lists
    - Handling multiple document formats
    """
    print("\n" + "=" * 70)
    print("Example 4: Document Classification")
    print("=" * 70)

    from smlx.models.Donut_base import classify_document, load

    # Load model
    print("\n1. Loading Donut model for classification...")
    model, processor = load("naver-clova-ix/donut-base-finetuned-rvlcdip")

    # Define document classes
    classes = [
        "invoice",
        "receipt",
        "form",
        "letter",
        "resume",
        "scientific_report",
        "budget",
        "presentation",
    ]

    # Test different documents
    doc_types = ["invoice", "receipt", "form"]

    print("\n2. Classifying documents...")
    for doc_type in doc_types:
        document = create_sample_document(doc_type)

        predicted_class = classify_document(
            model=model, processor=processor, image=document, classes=classes
        )

        print(f"\nDocument type: {doc_type}")
        print(f"Predicted class: {predicted_class}")

    print(
        "\nNote: This is placeholder classification. Load pre-trained weights for actual results."
    )


def example_5_batch_processing():
    """
    Example 5: Batch processing multiple documents.

    Demonstrates:
    - Processing multiple documents efficiently
    - Batch inference
    - Performance comparison
    """
    print("\n" + "=" * 70)
    print("Example 5: Batch Processing")
    print("=" * 70)

    from smlx.models.Donut_base import extract_text, load

    # Load model
    print("\n1. Loading Donut model...")
    model, processor = load("naver-clova-ix/donut-base")

    # Create batch of documents
    print("\n2. Creating batch of documents...")
    num_docs = 5
    documents = []
    for i in range(num_docs):
        doc_type = ["invoice", "receipt", "form"][i % 3]
        documents.append(create_sample_document(doc_type))

    # Sequential processing
    print("\n3. Processing documents sequentially...")
    start_time = time.time()
    results = []
    for i, doc in enumerate(documents):
        result = extract_text(model=model, processor=processor, image=doc)
        results.append(result)
        print(f"   Processed document {i + 1}/{num_docs}")
    sequential_time = time.time() - start_time

    print(f"\nSequential processing: {sequential_time:.2f}s")
    print(f"Average time per document: {sequential_time / num_docs:.2f}s")

    print("\nNote: Batch processing would be faster with actual implementation and GPU.")


def example_6_performance_benchmark():
    """
    Example 6: Performance benchmarking.

    Demonstrates:
    - Measuring inference speed
    - Memory usage tracking
    - Comparing different tasks
    """
    print("\n" + "=" * 70)
    print("Example 6: Performance Benchmark")
    print("=" * 70)

    from smlx.models.Donut_base import answer_question, extract_text, load

    # Load model
    print("\n1. Loading Donut model...")
    model, processor = load("naver-clova-ix/donut-base")

    # Create test document
    document = create_sample_document("invoice")

    # Benchmark text extraction
    print("\n2. Benchmarking text extraction...")
    num_runs = 10
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = extract_text(model=model, processor=processor, image=document)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"   Average time: {avg_time * 1000:.1f}ms")
    print(f"   Min time: {min(times) * 1000:.1f}ms")
    print(f"   Max time: {max(times) * 1000:.1f}ms")

    # Benchmark VQA
    print("\n3. Benchmarking VQA...")
    question = "What is the total amount?"
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = answer_question(model=model, processor=processor, image=document, question=question)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"   Average time: {avg_time * 1000:.1f}ms")
    print(f"   Min time: {min(times) * 1000:.1f}ms")
    print(f"   Max time: {max(times) * 1000:.1f}ms")

    print("\nNote: Performance will be significantly better with actual implementation and GPU.")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Donut Document Understanding Examples")
    print("=" * 70)
    print("\nIMPORTANT: These examples use a reference implementation.")
    print("For production use, load pre-trained weights from HuggingFace Hub:")
    print("  - naver-clova-ix/donut-base-finetuned-docvqa (VQA)")
    print("  - naver-clova-ix/donut-base-finetuned-rvlcdip (Classification)")
    print("  - naver-clova-ix/donut-base-finetuned-cord-v2 (Receipt parsing)")
    print("\nRunning 6 examples:\n")

    examples = [
        ("1. Document Parsing", example_1_document_parsing),
        ("2. Document VQA", example_2_document_vqa),
        ("3. Text Extraction", example_3_text_extraction),
        ("4. Document Classification", example_4_document_classification),
        ("5. Batch Processing", example_5_batch_processing),
        ("6. Performance Benchmark", example_6_performance_benchmark),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Load pre-trained weights from HuggingFace Hub")
    print("2. Fine-tune on your domain-specific documents")
    print("3. Integrate into your document processing pipeline")
    print("4. See docs/ModelImplementations.md for implementation details")


if __name__ == "__main__":
    main()
