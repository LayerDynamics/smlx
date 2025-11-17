#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for nanoVLM minimal vision-language model.

Tests image captioning, VQA, streaming, and basic multimodal capabilities.

Run with:
    python -m pytest tests/integration/test_nanovlm.py -v
"""

import gc

import mlx.core as mx
import pytest
from PIL import Image, ImageDraw

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
]


def create_test_image(image_type="simple"):
    """Create a simple test image."""
    img = Image.new("RGB", (224, 224), color="white")
    draw = ImageDraw.Draw(img)

    if image_type == "simple":
        # Draw simple shapes (suitable for 224x224)
        draw.rectangle([20, 20, 100, 100], fill="red", outline="black")
        draw.ellipse([124, 64, 204, 144], fill="blue", outline="black")
    elif image_type == "scene":
        # Simple scene
        draw.rectangle([0, 150, 224, 224], fill="green")  # Ground
        draw.ellipse([80, 20, 144, 84], fill="yellow")  # Sun
        draw.rectangle([50, 80, 100, 160], fill="brown")  # Tree trunk

    return img


@pytest.fixture(scope="module")
def nanovlm_model():
    """
    Load nanoVLM model once for all tests.

    Memory Requirements:
    - Model size: ~444MB (222M parameters in FP16)
    - Peak memory: ~800MB with activations
    """
    from smlx.models.nanoVLM import load

    model, processor = load("lusxvr/nanoVLM-222M")

    yield model, processor

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up nanoVLM model...")
    del model
    del processor
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print("Cleanup complete. Memory freed.")


def test_model_loading(nanovlm_model):
    """Test that nanoVLM model loads successfully."""
    model, processor = nanovlm_model

    assert model is not None, "Model should not be None"
    assert processor is not None, "Processor should not be None"
    assert hasattr(model, "vision_model"), "Model should have vision_model"
    assert hasattr(model, "language_model"), "Model should have language_model"
    assert hasattr(model, "projection"), "Model should have projection layer"


def test_image_loading():
    """Test image loading utilities."""
    from smlx.models.nanoVLM import load_image

    # Create test image
    test_image = create_test_image()

    # Should handle PIL Image
    loaded = load_image(test_image)
    assert isinstance(loaded, Image.Image), "Should return PIL Image"


def test_basic_generation(nanovlm_model):
    """Test basic image captioning."""
    from smlx.models.nanoVLM import generate

    model, processor = nanovlm_model

    # Create test image
    test_image = create_test_image("simple")

    # Generate caption
    output = generate(
        model=model,
        processor=processor,
        prompt="Describe this image:",
        image=test_image,
        max_tokens=30,
        temperature=0.0,
    )

    assert output is not None, "Output should not be None"
    assert isinstance(output, str), "Output should be a string"
    assert len(output) > 0, "Output should have content"


def test_caption_function(nanovlm_model):
    """Test dedicated caption function."""
    from smlx.models.nanoVLM import caption

    model, processor = nanovlm_model

    # Create test image
    test_image = create_test_image()

    # Caption
    caption_text = caption(model, processor, test_image)

    assert caption_text is not None, "Caption should not be None"
    assert isinstance(caption_text, str), "Caption should be string"


def test_query_function(nanovlm_model):
    """Test visual question answering."""
    from smlx.models.nanoVLM import query

    model, processor = nanovlm_model

    # Create test image
    test_image = create_test_image("simple")

    # Ask question
    answer = query(
        model=model,
        processor=processor,
        image=test_image,
        question="What shapes are visible?",
    )

    assert answer is not None, "Answer should not be None"
    assert isinstance(answer, str), "Answer should be string"


def test_streaming_generation(nanovlm_model):
    """Test streaming generation."""
    from smlx.models.nanoVLM import stream_generate

    model, processor = nanovlm_model

    # Create test image
    test_image = create_test_image()

    # Stream generation
    chunks = list(
        stream_generate(
            model=model,
            processor=processor,
            prompt="Describe:",
            image=test_image,
            max_tokens=30,
        )
    )

    assert len(chunks) > 0, "Should generate at least one chunk"

    for i, chunk in enumerate(chunks):
        assert chunk is not None, f"Chunk {i} should not be None"
        assert isinstance(chunk, str), f"Chunk {i} should be string"


def test_multiple_questions(nanovlm_model):
    """Test asking multiple questions about same image."""
    from smlx.models.nanoVLM import query

    model, processor = nanovlm_model

    # Create test image
    test_image = create_test_image("simple")

    # Ask multiple questions
    questions = [
        "What colors are in this image?",
        "What shapes do you see?",
        "Describe the image.",
    ]

    for question in questions:
        answer = query(
            model=model,
            processor=processor,
            image=test_image,
            question=question,
        )

        assert answer is not None, f"Answer for '{question}' should not be None"


def test_batch_processing(nanovlm_model):
    """Test processing multiple images."""
    from smlx.models.nanoVLM import caption

    model, processor = nanovlm_model

    # Create multiple test images
    images = [
        create_test_image("simple"),
        create_test_image("scene"),
        create_test_image("simple"),
    ]

    # Process each
    results = []
    for img in images:
        result = caption(model, processor, img)
        results.append(result)

    assert len(results) == len(images), "Should have result for each image"

    for i, result in enumerate(results):
        assert result is not None, f"Result {i} should not be None"


def test_prepare_inputs(nanovlm_model):
    """Test input preparation."""
    from smlx.models.nanoVLM import prepare_inputs

    model, processor = nanovlm_model

    # Create test image
    test_image = create_test_image()

    # Prepare inputs
    inputs = prepare_inputs(
        processor=processor, prompt="Test", image=test_image
    )

    assert inputs is not None, "Inputs should not be None"


def test_generation_parameters(nanovlm_model):
    """Test different generation parameters."""
    from smlx.models.nanoVLM import generate

    model, processor = nanovlm_model

    # Create test image
    test_image = create_test_image()

    # Test different temperatures
    for temp in [0.0, 0.5, 1.0]:
        output = generate(
            model=model,
            processor=processor,
            prompt="Describe:",
            image=test_image,
            max_tokens=20,
            temperature=temp,
        )

        assert output is not None, f"Should work with temperature {temp}"


def test_max_tokens(nanovlm_model):
    """Test max tokens parameter."""
    from smlx.models.nanoVLM import generate

    model, processor = nanovlm_model

    # Create test image
    test_image = create_test_image()

    # Test with small max_tokens
    output = generate(
        model=model,
        processor=processor,
        prompt="Describe:",
        image=test_image,
        max_tokens=5,
    )

    assert output is not None, "Should respect max_tokens"


def test_image_processor():
    """Test image processor."""
    from smlx.models.nanoVLM import ImageProcessor

    # Create processor
    processor = ImageProcessor()

    # Create test image
    test_image = create_test_image()

    # Process
    processed = processor(test_image)

    assert processed is not None, "Processed image should not be None"


def test_model_components(nanovlm_model):
    """Test model components."""
    model, processor = nanovlm_model

    # Check components exist
    assert hasattr(model, "vision_model"), "Should have vision_model"
    assert hasattr(model, "language_model"), "Should have language_model"
    assert hasattr(model, "projection"), "Should have projection"

    # Components should not be None
    assert model.vision_model is not None, "Vision should not be None"
    assert model.language_model is not None, "Language should not be None"
    assert model.projection is not None, "Projection should not be None"


def test_vision_encoder(nanovlm_model):
    """Test vision encoder."""
    model, processor = nanovlm_model

    vision = model.vision_model
    assert vision is not None, "Vision encoder should not be None"


def test_projection_layer(nanovlm_model):
    """Test projection layer."""
    model, processor = nanovlm_model

    projection = model.projection
    assert projection is not None, "Projection should not be None"


def test_language_model(nanovlm_model):
    """Test language model component."""
    model, processor = nanovlm_model

    language = model.language_model
    assert language is not None, "Language model should not be None"


def test_model_config(nanovlm_model):
    """Test model configuration."""
    from smlx.models.nanoVLM import NanoVLMConfig

    model, processor = nanovlm_model

    # Model should have config
    assert hasattr(model, "config"), "Model should have config"

    config = model.config
    assert config is not None, "Config should not be None"


def test_default_config():
    """Test default configuration."""
    from smlx.models.nanoVLM import DEFAULT_CONFIG

    assert DEFAULT_CONFIG is not None, "DEFAULT_CONFIG should exist"
    assert hasattr(DEFAULT_CONFIG, "vision_config"), "Should have vision_config"
    assert hasattr(DEFAULT_CONFIG, "language_config"), "Should have language_config"
    assert (
        hasattr(DEFAULT_CONFIG, "projection_config")
    ), "Should have projection_config"


def test_different_prompts(nanovlm_model):
    """Test different prompt styles."""
    from smlx.models.nanoVLM import generate

    model, processor = nanovlm_model

    # Create test image
    test_image = create_test_image()

    prompts = [
        "Describe this image:",
        "What do you see?",
        "Caption:",
    ]

    for prompt in prompts:
        output = generate(
            model=model,
            processor=processor,
            prompt=prompt,
            image=test_image,
            max_tokens=20,
        )

        assert output is not None, f"Should work with prompt: '{prompt}'"


def test_simple_scene(nanovlm_model):
    """Test with simple scene."""
    from smlx.models.nanoVLM import caption

    model, processor = nanovlm_model

    # Create scene
    test_image = create_test_image("scene")

    # Caption
    caption_text = caption(model, processor, test_image)

    assert caption_text is not None, "Scene caption should not be None"


def test_color_questions(nanovlm_model):
    """Test color-related questions."""
    from smlx.models.nanoVLM import query

    model, processor = nanovlm_model

    # Create test image with colors
    test_image = create_test_image("simple")

    # Ask about colors
    answer = query(
        model=model,
        processor=processor,
        image=test_image,
        question="What colors are present?",
    )

    assert answer is not None, "Color question should work"


def test_pil_image_input(nanovlm_model):
    """Test PIL Image input."""
    from smlx.models.nanoVLM import generate

    model, processor = nanovlm_model

    # Create PIL Image
    pil_image = create_test_image()

    # Should accept PIL Image
    output = generate(
        model=model,
        processor=processor,
        prompt="Describe:",
        image=pil_image,
        max_tokens=20,
    )

    assert output is not None, "Should work with PIL Image"


def test_file_path_input(nanovlm_model, tmp_path):
    """Test file path input."""
    from smlx.models.nanoVLM import caption

    model, processor = nanovlm_model

    # Create and save test image
    test_image = create_test_image()
    image_path = tmp_path / "test_image.png"
    test_image.save(image_path)

    # Should accept file path
    caption_text = caption(model, processor, str(image_path))

    assert caption_text is not None, "Should work with file path"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
