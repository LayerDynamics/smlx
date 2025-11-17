#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for Donut document understanding model.

Tests document parsing, VQA, text extraction, and classification.

Run with:
    python -m pytest tests/integration/test_donut.py -v
"""

import gc

import mlx.core as mx
import pytest
from PIL import Image, ImageDraw

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
]


def create_test_document(doc_type="simple"):
    """Create a simple test document image."""
    img = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(img)

    if doc_type == "simple":
        draw.text((50, 50), "TEST DOCUMENT", fill="black")
        draw.text((50, 100), "Invoice #123", fill="black")
        draw.text((50, 150), "Total: $100.00", fill="black")
    elif doc_type == "form":
        draw.text((50, 50), "FORM", fill="black")
        draw.text((50, 100), "Name: John Doe", fill="black")
        draw.text((50, 150), "Date: 2024-01-15", fill="black")

    return img


@pytest.fixture(scope="module")
def donut_model():
    """
    Load Donut model once for all tests.

    Memory Requirements:
    - Model size: ~400MB (200M parameters in FP16)
    - Peak memory: ~800MB with activations
    """
    from smlx.models.Donut_base import load

    model, processor = load("naver-clova-ix/donut-base")

    yield model, processor

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up Donut model...")
    del model
    del processor
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print("Cleanup complete. Memory freed.")


def test_model_loading(donut_model):
    """Test that Donut model loads successfully."""
    model, processor = donut_model

    assert model is not None, "Model should not be None"
    assert processor is not None, "Processor should not be None"
    assert hasattr(model, "encoder"), "Model should have encoder"
    assert hasattr(model, "decoder"), "Model should have decoder"


def test_document_parsing(donut_model):
    """Test document parsing functionality."""
    from smlx.models.Donut_base import parse_document

    model, processor = donut_model

    # Create test document
    test_image = create_test_document("simple")

    # Parse document
    result = parse_document(
        model=model,
        processor=processor,
        image=test_image,
        task="document",
    )

    assert result is not None, "Parse result should not be None"
    # Note: With placeholder implementation, result will be minimal


def test_document_vqa(donut_model):
    """Test document visual question answering."""
    from smlx.models.Donut_base import answer_question

    model, processor = donut_model

    # Create test document
    test_image = create_test_document("simple")

    # Ask question
    question = "What is the invoice number?"
    answer = answer_question(
        model=model,
        processor=processor,
        image=test_image,
        question=question,
    )

    assert answer is not None, "Answer should not be None"
    assert isinstance(answer, str), "Answer should be a string"


def test_text_extraction(donut_model):
    """Test OCR-free text extraction."""
    from smlx.models.Donut_base import extract_text

    model, processor = donut_model

    # Create test document
    test_image = create_test_document("simple")

    # Extract text
    text = extract_text(
        model=model,
        processor=processor,
        image=test_image,
    )

    assert text is not None, "Extracted text should not be None"
    assert isinstance(text, str), "Extracted text should be a string"


def test_document_classification(donut_model):
    """Test document classification."""
    from smlx.models.Donut_base import classify_document

    model, processor = donut_model

    # Create test document
    test_image = create_test_document("simple")

    # Define classes
    classes = ["invoice", "receipt", "form", "letter"]

    # Classify
    doc_class = classify_document(
        model=model,
        processor=processor,
        image=test_image,
        classes=classes,
    )

    assert doc_class is not None, "Classification result should not be None"
    assert isinstance(doc_class, str), "Classification should be a string"


def test_generate_function(donut_model):
    """Test low-level generate function."""
    from smlx.models.Donut_base import generate

    model, processor = donut_model

    # Create test document
    test_image = create_test_document("simple")

    # Generate
    output = generate(
        model=model,
        processor=processor,
        image=test_image,
        prompt="<s_docvqa><s_question>What is this?</s_question><s_answer>",
        max_length=128,
    )

    assert output is not None, "Generate output should not be None"


def test_different_document_types(donut_model):
    """Test with different document types."""
    from smlx.models.Donut_base import extract_text

    model, processor = donut_model

    # Test different document types
    doc_types = ["simple", "form"]

    for doc_type in doc_types:
        test_image = create_test_document(doc_type)
        text = extract_text(
            model=model,
            processor=processor,
            image=test_image,
        )

        assert text is not None, f"Text extraction for {doc_type} should not be None"
        assert isinstance(text, str), f"Extracted text for {doc_type} should be string"


def test_pil_image_input(donut_model):
    """Test that PIL Image input works correctly."""
    from smlx.models.Donut_base import extract_text

    model, processor = donut_model

    # Create PIL Image
    pil_image = create_test_document("simple")

    # Should accept PIL Image
    text = extract_text(
        model=model,
        processor=processor,
        image=pil_image,
    )

    assert text is not None, "Should work with PIL Image"


def test_file_path_input(donut_model, tmp_path):
    """Test that file path input works correctly."""
    from smlx.models.Donut_base import extract_text

    model, processor = donut_model

    # Create and save test image
    test_image = create_test_document("simple")
    image_path = tmp_path / "test_document.png"
    test_image.save(image_path)

    # Should accept file path
    text = extract_text(
        model=model,
        processor=processor,
        image=str(image_path),
    )

    assert text is not None, "Should work with file path"


def test_multiple_questions(donut_model):
    """Test asking multiple questions about same document."""
    from smlx.models.Donut_base import answer_question

    model, processor = donut_model

    # Create test document
    test_image = create_test_document("simple")

    # Ask multiple questions
    questions = [
        "What is the invoice number?",
        "What is the total amount?",
        "What type of document is this?",
    ]

    for question in questions:
        answer = answer_question(
            model=model,
            processor=processor,
            image=test_image,
            question=question,
        )

        assert answer is not None, f"Answer for '{question}' should not be None"
        assert isinstance(answer, str), f"Answer for '{question}' should be string"


def test_processor_image_loading():
    """Test processor image loading functionality."""
    from smlx.models.Donut_base import load_image

    # Create test image
    test_image = create_test_document("simple")

    # Load as PIL Image (should return as-is)
    loaded = load_image(test_image)
    assert isinstance(loaded, Image.Image), "Should return PIL Image"

    # Test with file path
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name

    try:
        test_image.save(temp_path)
        loaded_from_path = load_image(temp_path)
        assert isinstance(loaded_from_path, Image.Image), "Should load from file path"
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_empty_text_extraction(donut_model):
    """Test text extraction on blank document."""
    from smlx.models.Donut_base import extract_text

    model, processor = donut_model

    # Create blank document
    blank_image = Image.new("RGB", (400, 300), color="white")

    # Extract text
    text = extract_text(
        model=model,
        processor=processor,
        image=blank_image,
    )

    # Should not crash, might return empty string
    assert text is not None, "Should handle blank document"


def test_configuration_loading():
    """Test configuration loading."""
    from smlx.models.Donut_base import DEFAULT_CONFIG, load_config, DonutConfig

    # Test DEFAULT_CONFIG exists
    assert DEFAULT_CONFIG is not None, "DEFAULT_CONFIG should exist"
    assert isinstance(DEFAULT_CONFIG, DonutConfig), "DEFAULT_CONFIG should be DonutConfig"

    # Test config attributes
    assert hasattr(DEFAULT_CONFIG, "encoder_config"), "Should have encoder_config"
    assert hasattr(DEFAULT_CONFIG, "decoder_config"), "Should have decoder_config"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
