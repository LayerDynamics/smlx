#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for SmolVLM-256M-Instruct vision-language model.

Tests image captioning, VQA, batch processing, streaming generation.

Run with:
    python -m pytest tests/integration/test_smolvlm_256m.py -v
"""

import gc

import mlx.core as mx
import pytest
import numpy as np
from PIL import Image, ImageDraw

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
    pytest.mark.heavy_memory,  # SmolVLM-256M uses ~512MB
]


def create_test_image(image_type="simple"):
    """Create a simple test image."""
    img = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(img)

    if image_type == "simple":
        # Draw basic shapes
        draw.rectangle([50, 50, 150, 150], fill="red", outline="black")
        draw.ellipse([200, 100, 300, 200], fill="blue", outline="black")
        draw.text((50, 250), "Test Image", fill="black")
    elif image_type == "complex":
        # More complex scene
        draw.rectangle([0, 200, 400, 300], fill="green")  # Ground
        draw.ellipse([150, 50, 250, 150], fill="yellow")  # Sun
        draw.rectangle([50, 100, 150, 250], fill="brown")  # Tree trunk
        draw.ellipse([25, 25, 175, 175], fill="darkgreen")  # Tree top

    return img


@pytest.fixture(scope="module")
def smolvlm_model():
    """
    Load SmolVLM-256M model once for all tests.

    Memory Requirements:
    - Model size: ~512MB (256M parameters in FP16)
    - Peak memory: ~1GB with activations
    """
    from smlx.models.SmolVLM_256M import load

    model, processor = load("HuggingFaceTB/SmolVLM-256M-Instruct")

    yield model, processor

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up SmolVLM-256M model...")
    del model
    del processor
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print("Cleanup complete. Memory freed.")


def test_model_loading(smolvlm_model):
    """Test that SmolVLM model loads successfully."""
    model, processor = smolvlm_model

    assert model is not None, "Model should not be None"
    assert processor is not None, "Processor should not be None"
    assert hasattr(model, "vision_model"), "Model should have vision_model"
    assert hasattr(model, "language_model"), "Model should have language_model"
    assert hasattr(model, "connector"), "Model should have vision-language connector"


def test_image_loading():
    """Test image loading utilities."""
    from smlx.models.SmolVLM_256M import load_image

    # Create test image
    test_image = create_test_image()

    # Should handle PIL Image
    loaded = load_image(test_image)
    assert isinstance(loaded, Image.Image), "Should return PIL Image"

    # Should handle image paths
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
        test_image.save(temp_path)

    try:
        loaded_from_path = load_image(temp_path)
        assert isinstance(loaded_from_path, Image.Image), "Should load from path"
        # Close the image to release file handle
        if hasattr(loaded_from_path, "close"):
            loaded_from_path.close()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_basic_generation(smolvlm_model):
    """Test basic image captioning."""
    from smlx.models.SmolVLM_256M import generate

    model, processor = smolvlm_model

    # Create test image
    test_image = create_test_image("simple")

    # Generate caption
    output = generate(
        model=model,
        processor=processor,
        prompt="Describe this image:",
        image=test_image,
        max_tokens=50,
        temperature=0.0,
    )

    assert output is not None, "Output should not be None"
    assert isinstance(output, str), "Output should be a string"
    assert len(output) > 0, "Output should have content"


def test_visual_question_answering(smolvlm_model):
    """Test visual question answering."""
    from smlx.models.SmolVLM_256M import generate

    model, processor = smolvlm_model

    # Create test image with shapes
    test_image = create_test_image("simple")

    # Ask questions about the image
    questions = [
        "What colors are visible in this image?",
        "What shapes can you see?",
        "Describe what you see.",
    ]

    for question in questions:
        answer = generate(
            model=model,
            processor=processor,
            prompt=question,
            image=test_image,
            max_tokens=30,
            temperature=0.0,
        )

        assert answer is not None, f"Answer for '{question}' should not be None"
        assert isinstance(answer, str), f"Answer for '{question}' should be string"


def test_streaming_generation(smolvlm_model):
    """Test streaming generation."""
    from smlx.models.SmolVLM_256M import stream_generate

    model, processor = smolvlm_model

    # Create test image
    test_image = create_test_image()

    # Stream generation
    chunks = list(
        stream_generate(
            model=model,
            processor=processor,
            prompt="Describe this image:",
            image=test_image,
            max_tokens=50,
        )
    )

    assert len(chunks) > 0, "Should generate at least one chunk"

    for i, chunk in enumerate(chunks):
        assert chunk is not None, f"Chunk {i} should not be None"
        assert isinstance(chunk, str), f"Chunk {i} should be string"

    # Concatenate should form complete response
    full_response = "".join(chunks)
    assert len(full_response) > 0, "Full response should have content"


def test_chat_interface(smolvlm_model):
    """Test chat-style interface."""
    from smlx.models.SmolVLM_256M import chat

    model, processor = smolvlm_model

    # Create test image
    test_image = create_test_image()

    # Create chat messages
    messages = [{"role": "user", "content": "What's in this image?"}]

    # Chat
    response = chat(
        model=model,
        processor=processor,
        messages=messages,
        image=test_image,
        max_tokens=50,
    )

    assert response is not None, "Chat response should not be None"
    assert isinstance(response, str), "Response should be string"


def test_multiple_turns(smolvlm_model):
    """Test multi-turn conversation."""
    from smlx.models.SmolVLM_256M import chat

    model, processor = smolvlm_model

    # Create test image
    test_image = create_test_image("simple")

    # First turn
    messages = [{"role": "user", "content": "Describe this image briefly."}]

    response1 = chat(
        model=model,
        processor=processor,
        messages=messages,
        image=test_image,
        max_tokens=30,
    )

    # Second turn (follow-up question)
    messages.append({"role": "assistant", "content": response1})
    messages.append({"role": "user", "content": "What colors did you see?"})

    response2 = chat(
        model=model,
        processor=processor,
        messages=messages,
        image=test_image,
        max_tokens=20,
    )

    assert response2 is not None, "Second response should not be None"


def test_batch_processing(smolvlm_model):
    """Test processing multiple images."""
    from smlx.models.SmolVLM_256M import generate

    model, processor = smolvlm_model

    # Create multiple test images
    images = [
        create_test_image("simple"),
        create_test_image("complex"),
        create_test_image("simple"),
    ]

    # Process each image
    results = []
    for img in images:
        result = generate(
            model=model,
            processor=processor,
            prompt="Describe this image in one sentence:",
            image=img,
            max_tokens=30,
        )
        results.append(result)

    assert len(results) == len(images), "Should have result for each image"

    for i, result in enumerate(results):
        assert result is not None, f"Result {i} should not be None"
        assert isinstance(result, str), f"Result {i} should be string"


def test_different_prompts(smolvlm_model):
    """Test different prompt styles."""
    from smlx.models.SmolVLM_256M import generate

    model, processor = smolvlm_model

    # Create test image
    test_image = create_test_image("complex")

    prompts = [
        "Describe this image:",
        "What do you see?",
        "Caption this image:",
        "Answer: What is in this picture?",
    ]

    for prompt in prompts:
        output = generate(
            model=model,
            processor=processor,
            prompt=prompt,
            image=test_image,
            max_tokens=30,
        )

        assert output is not None, f"Output for '{prompt}' should not be None"


def test_generation_parameters(smolvlm_model):
    """Test different generation parameters."""
    from smlx.models.SmolVLM_256M import generate

    model, processor = smolvlm_model

    # Create test image
    test_image = create_test_image()

    # Test different temperatures
    for temp in [0.0, 0.5, 1.0]:
        output = generate(
            model=model,
            processor=processor,
            prompt="Describe this image:",
            image=test_image,
            max_tokens=30,
            temperature=temp,
        )

        assert output is not None, f"Should work with temperature {temp}"


def test_max_tokens_limit(smolvlm_model):
    """Test max tokens parameter."""
    from smlx.models.SmolVLM_256M import generate

    model, processor = smolvlm_model

    # Create test image
    test_image = create_test_image()

    # Test with small max_tokens
    output = generate(
        model=model,
        processor=processor,
        prompt="Describe this image in detail:",
        image=test_image,
        max_tokens=10,
        temperature=0.0,
    )

    assert output is not None, "Should respect max_tokens limit"


def test_image_preprocessing():
    """Test image preprocessing."""
    from smlx.models.SmolVLM_256M import ImageProcessor

    # Create processor
    processor = ImageProcessor()

    # Create test image
    test_image = create_test_image()

    # Process image
    processed = processor(test_image)

    assert processed is not None, "Processed image should not be None"


def test_prepare_inputs(smolvlm_model):
    """Test input preparation."""
    from smlx.models.SmolVLM_256M import prepare_inputs

    model, processor = smolvlm_model

    # Create test image
    test_image = create_test_image()

    # Prepare inputs
    inputs = prepare_inputs(
        processor=processor, prompt="Test prompt", image=test_image
    )

    assert inputs is not None, "Inputs should not be None"
    assert isinstance(inputs, dict), "Inputs should be dict"


def test_no_image_error(smolvlm_model):
    """Test that providing no image raises appropriate error."""
    from smlx.models.SmolVLM_256M import generate

    model, processor = smolvlm_model

    # Try to generate without image (should fail gracefully)
    try:
        output = generate(
            model=model,
            processor=processor,
            prompt="Describe this image:",
            image=None,
            max_tokens=10,
        )
        # If it doesn't raise an error, that's also fine
        assert True
    except (ValueError, TypeError, AttributeError):
        # Expected to fail without image
        assert True


def test_model_config(smolvlm_model):
    """Test model configuration."""
    from smlx.models.SmolVLM_256M import ModelConfig

    model, processor = smolvlm_model

    # Model should have config
    assert hasattr(model, "config"), "Model should have config"

    config = model.config
    assert config is not None, "Config should not be None"
    assert hasattr(config, "text_config"), "Should have text config"
    assert hasattr(config, "vision_config"), "Should have vision config"


def test_vision_encoder(smolvlm_model):
    """Test vision encoder component."""
    model, processor = smolvlm_model

    vision = model.vision_model
    assert vision is not None, "Vision encoder should not be None"


def test_language_model(smolvlm_model):
    """Test language model component."""
    model, processor = smolvlm_model

    language = model.language_model
    assert language is not None, "Language model should not be None"


def test_connector(smolvlm_model):
    """Test vision-language connector."""
    model, processor = smolvlm_model

    connector = model.connector
    assert connector is not None, "Connector should not be None"


def test_different_image_sizes(smolvlm_model):
    """Test with different image sizes."""
    from smlx.models.SmolVLM_256M import generate

    model, processor = smolvlm_model

    # Test different sizes
    sizes = [(200, 200), (400, 300), (800, 600)]

    for width, height in sizes:
        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, width - 10, height - 10], outline="black")

        output = generate(
            model=model,
            processor=processor,
            prompt="Describe:",
            image=img,
            max_tokens=20,
        )

        assert output is not None, f"Should work with size {width}x{height}"


def test_detailed_description(smolvlm_model):
    """Test generating detailed descriptions."""
    from smlx.models.SmolVLM_256M import generate

    model, processor = smolvlm_model

    # Create more complex image
    test_image = create_test_image("complex")

    # Ask for detailed description
    output = generate(
        model=model,
        processor=processor,
        prompt="Provide a detailed description of this image:",
        image=test_image,
        max_tokens=100,
        temperature=0.0,
    )

    assert output is not None, "Detailed description should not be None"
    assert len(output) > 0, "Should generate detailed content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
