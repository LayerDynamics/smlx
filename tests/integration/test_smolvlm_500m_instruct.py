#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for SmolVLM-500M-Instruct vision-language model.

Tests image captioning, VQA, chat, streaming generation.

Run with:
    python -m pytest tests/integration/test_smolvlm_500m_instruct.py -v
"""

import gc

import mlx.core as mx
import pytest
from PIL import Image, ImageDraw

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
    pytest.mark.heavy_memory,  # SmolVLM-500M uses ~1GB
]


def create_test_image(image_type="simple"):
    """Create a test image."""
    img = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(img)

    if image_type == "simple":
        draw.rectangle([50, 50, 150, 150], fill="red", outline="black")
        draw.ellipse([200, 100, 300, 200], fill="blue", outline="black")
    elif image_type == "complex":
        draw.rectangle([0, 200, 400, 300], fill="green")
        draw.ellipse([150, 50, 250, 150], fill="yellow")
        draw.rectangle([100, 120, 200, 250], fill="brown")

    return img


@pytest.fixture(scope="module")
def smolvlm_500m_model():
    """
    Load SmolVLM-500M model once for all tests.

    Memory Requirements:
    - Model size: ~1GB (500M parameters in FP16)
    - Peak memory: ~1.5GB with activations
    - Requires: 2GB available headroom
    """
    from smlx.utils.memory import check_memory_availability, memory_profiler
    from smlx.models.SmolVLM_500M_Instruct import load

    # Check memory before loading
    check = check_memory_availability(2.0)  # Require 2GB headroom
    if not check["available"]:
        pytest.skip(
            f"Insufficient memory for SmolVLM-500M: "
            f"{check['headroom_gb']:.1f}GB available, need 2GB"
        )

    # Load model with memory profiling
    with memory_profiler() as mem:
        model, processor = load("HuggingFaceTB/SmolVLM-500M-Instruct")
        mx.eval(model)

    print(f"\nSmolVLM-500M loaded: {mem.peak_gb:.2f}GB peak memory")

    yield model, processor

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up SmolVLM-500M model...")
    del model
    del processor
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print(f"Cleanup complete. Memory freed.")


def test_model_loading(smolvlm_500m_model):
    """Test that SmolVLM-500M model loads successfully."""
    model, processor = smolvlm_500m_model

    assert model is not None, "Model should not be None"
    assert processor is not None, "Processor should not be None"
    assert hasattr(model, "vision"), "Model should have vision encoder"
    assert hasattr(model, "language"), "Model should have language model"
    assert hasattr(model, "connector"), "Model should have connector"


def test_basic_generation(smolvlm_500m_model):
    """Test basic generation."""
    from smlx.models.SmolVLM_500M_Instruct import generate

    model, processor = smolvlm_500m_model

    test_image = create_test_image("simple")

    output = generate(
        model=model,
        processor=processor,
        prompt="Describe this image:",
        image=test_image,
        max_tokens=30,
        temperature=0.0,
    )

    assert output is not None, "Output should not be None"
    assert isinstance(output, str), "Output should be string"
    assert len(output) > 0, "Output should have content"


def test_visual_question_answering(smolvlm_500m_model):
    """Test VQA."""
    from smlx.models.SmolVLM_500M_Instruct import generate

    model, processor = smolvlm_500m_model

    test_image = create_test_image("simple")

    questions = [
        "What colors are in this image?",
        "What shapes do you see?",
        "Describe what you see.",
    ]

    for question in questions:
        answer = generate(
            model=model,
            processor=processor,
            prompt=question,
            image=test_image,
            max_tokens=30,
        )

        assert answer is not None, f"Answer for '{question}' should not be None"


def test_streaming_generation(smolvlm_500m_model):
    """Test streaming generation."""
    from smlx.models.SmolVLM_500M_Instruct import stream_generate

    model, processor = smolvlm_500m_model

    test_image = create_test_image()

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


def test_chat_interface(smolvlm_500m_model):
    """Test chat interface."""
    from smlx.models.SmolVLM_500M_Instruct import chat

    model, processor = smolvlm_500m_model

    test_image = create_test_image()

    messages = [{"role": "user", "content": "What's in this image?"}]

    response = chat(
        model=model,
        processor=processor,
        messages=messages,
        image=test_image,
        max_tokens=30,
    )

    assert response is not None, "Chat response should not be None"
    assert isinstance(response, str), "Response should be string"


def test_multi_turn_chat(smolvlm_500m_model):
    """Test multi-turn conversation."""
    from smlx.models.SmolVLM_500M_Instruct import chat

    model, processor = smolvlm_500m_model

    test_image = create_test_image("simple")

    messages = [{"role": "user", "content": "Describe this image briefly."}]

    response1 = chat(
        model=model,
        processor=processor,
        messages=messages,
        image=test_image,
        max_tokens=30,
    )

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


def test_prepare_inputs(smolvlm_500m_model):
    """Test input preparation."""
    from smlx.models.SmolVLM_500M_Instruct import prepare_inputs

    model, processor = smolvlm_500m_model

    test_image = create_test_image()

    inputs = prepare_inputs(
        processor=processor,
        prompt="Test",
        image=test_image,
    )

    assert inputs is not None, "Inputs should not be None"
    assert isinstance(inputs, dict), "Inputs should be dict"


def test_image_loading():
    """Test image loading."""
    from smlx.models.SmolVLM_500M_Instruct import load_image

    test_image = create_test_image()

    loaded = load_image(test_image)

    assert isinstance(loaded, Image.Image), "Should return PIL Image"


def test_batch_processing(smolvlm_500m_model):
    """Test processing multiple images."""
    from smlx.models.SmolVLM_500M_Instruct import generate

    model, processor = smolvlm_500m_model

    images = [
        create_test_image("simple"),
        create_test_image("complex"),
        create_test_image("simple"),
    ]

    results = []
    for img in images:
        result = generate(
            model=model,
            processor=processor,
            prompt="Describe:",
            image=img,
            max_tokens=20,
        )
        results.append(result)

    assert len(results) == len(images), "Should have result for each image"


def test_different_temperatures(smolvlm_500m_model):
    """Test different temperature settings."""
    from smlx.models.SmolVLM_500M_Instruct import generate

    model, processor = smolvlm_500m_model

    test_image = create_test_image()

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


def test_max_tokens_parameter(smolvlm_500m_model):
    """Test max tokens parameter."""
    from smlx.models.SmolVLM_500M_Instruct import generate

    model, processor = smolvlm_500m_model

    test_image = create_test_image()

    output = generate(
        model=model,
        processor=processor,
        prompt="Describe in detail:",
        image=test_image,
        max_tokens=10,
    )

    assert output is not None, "Should respect max_tokens"


def test_model_components(smolvlm_500m_model):
    """Test model components."""
    model, processor = smolvlm_500m_model

    assert hasattr(model, "vision"), "Should have vision"
    assert hasattr(model, "language"), "Should have language"
    assert hasattr(model, "connector"), "Should have connector"

    assert model.vision is not None, "Vision should not be None"
    assert model.language is not None, "Language should not be None"
    assert model.connector is not None, "Connector should not be None"


def test_vision_encoder(smolvlm_500m_model):
    """Test vision encoder."""
    model, processor = smolvlm_500m_model

    vision = model.vision
    assert vision is not None, "Vision encoder should not be None"


def test_language_model(smolvlm_500m_model):
    """Test language model."""
    model, processor = smolvlm_500m_model

    language = model.language
    assert language is not None, "Language model should not be None"


def test_connector(smolvlm_500m_model):
    """Test connector."""
    model, processor = smolvlm_500m_model

    connector = model.connector
    assert connector is not None, "Connector should not be None"


def test_model_config(smolvlm_500m_model):
    """Test model configuration."""
    from smlx.models.SmolVLM_500M_Instruct import ModelConfig

    model, processor = smolvlm_500m_model

    assert hasattr(model, "config"), "Model should have config"

    config = model.config
    assert config is not None, "Config should not be None"


def test_default_config():
    """Test default configuration."""
    from smlx.models.SmolVLM_500M_Instruct import DEFAULT_CONFIG

    assert DEFAULT_CONFIG is not None, "DEFAULT_CONFIG should exist"
    assert hasattr(DEFAULT_CONFIG, "text_config"), "Should have text_config"
    assert hasattr(DEFAULT_CONFIG, "vision_config"), "Should have vision_config"


def test_image_processor():
    """Test image processor."""
    from smlx.models.SmolVLM_500M_Instruct import ImageProcessor

    processor = ImageProcessor()
    test_image = create_test_image()

    processed = processor(test_image)

    assert processed is not None, "Processed image should not be None"


def test_file_path_input(smolvlm_500m_model, tmp_path):
    """Test file path input."""
    from smlx.models.SmolVLM_500M_Instruct import generate

    model, processor = smolvlm_500m_model

    test_image = create_test_image()
    image_path = tmp_path / "test.png"
    test_image.save(image_path)

    output = generate(
        model=model,
        processor=processor,
        prompt="Describe:",
        image=str(image_path),
        max_tokens=20,
    )

    assert output is not None, "Should work with file path"


def test_different_prompts(smolvlm_500m_model):
    """Test different prompt styles."""
    from smlx.models.SmolVLM_500M_Instruct import generate

    model, processor = smolvlm_500m_model

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
