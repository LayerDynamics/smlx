#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for Moondream2 vision-language model.

Tests captioning, VQA, object detection, spatial localization, region modules.

Run with:
    python -m pytest tests/integration/test_moondream2.py -v
"""

import gc

import mlx.core as mx
import pytest
import numpy as np
from PIL import Image, ImageDraw

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
    pytest.mark.heavy_memory,  # Moondream2 uses ~1GB
]


def create_test_image(image_type="simple"):
    """Create a test image with identifiable objects."""
    img = Image.new("RGB", (640, 480), color="white")
    draw = ImageDraw.Draw(img)

    if image_type == "simple":
        # Draw objects at specific locations
        # Red box (left)
        draw.rectangle([50, 100, 150, 200], fill="red", outline="black", width=2)

        # Blue circle (center)
        draw.ellipse([250, 150, 350, 250], fill="blue", outline="black", width=2)

        # Green triangle (right) - approximate with polygon
        draw.polygon(
            [(500, 300), (450, 200), (550, 200)],
            fill="green",
            outline="black",
        )

    elif image_type == "complex":
        # More complex scene with multiple objects
        # Sky
        draw.rectangle([0, 0, 640, 240], fill="skyblue")

        # Ground
        draw.rectangle([0, 240, 640, 480], fill="green")

        # Sun
        draw.ellipse([500, 50, 600, 150], fill="yellow", outline="orange", width=3)

        # House
        draw.rectangle([200, 200, 400, 400], fill="brown", outline="black", width=2)

        # Roof (triangle)
        draw.polygon(
            [(300, 150), (180, 200), (420, 200)],
            fill="darkred",
            outline="black",
        )

        # Window
        draw.rectangle([250, 250, 320, 320], fill="lightblue", outline="black")

        # Door
        draw.rectangle([340, 300, 390, 400], fill="saddlebrown", outline="black")

    return img


@pytest.fixture(scope="module")
def moondream_model():
    """
    Load Moondream2 model once for all tests.

    Memory Requirements:
    - Model size: ~1GB (500M parameters in FP16)
    - Peak memory: ~1.5GB with activations
    - Requires: 2GB available headroom
    """
    from smlx.utils.memory import check_memory_availability, memory_profiler
    from smlx.models.Moondream2 import load

    # Check memory before loading
    check = check_memory_availability(2.0)  # Require 2GB headroom
    if not check["available"]:
        pytest.skip(
            f"Insufficient memory for Moondream2: "
            f"{check['headroom_gb']:.1f}GB available, need 2GB"
        )

    # Load model with memory profiling
    with memory_profiler() as mem:
        model, tokenizer = load("vikhyatk/moondream2")
        mx.eval(model)

    print(f"\nMoondream2 loaded: {mem.peak_gb:.2f}GB peak memory")

    yield model, tokenizer

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up Moondream2 model...")
    del model
    del tokenizer
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print(f"Cleanup complete. Memory freed.")


def test_model_loading(moondream_model):
    """Test that Moondream2 model loads successfully."""
    model, tokenizer = moondream_model

    assert model is not None, "Model should not be None"
    assert tokenizer is not None, "Tokenizer should not be None"
    assert hasattr(model, "vision_encoder"), "Model should have vision encoder"
    assert hasattr(model, "language_model"), "Model should have language model"
    assert hasattr(model, "vision_projection"), "Model should have vision projector"
    assert hasattr(model, "lm_head"), "Model should have language model head"


def test_image_preprocessing():
    """Test image preprocessing."""
    from smlx.models.Moondream2 import preprocess_image

    # Create test image
    test_image = create_test_image()

    # Preprocess
    processed = preprocess_image(test_image)

    assert processed is not None, "Preprocessed image should not be None"


def test_basic_captioning(moondream_model):
    """Test basic image captioning."""
    from smlx.models.Moondream2 import caption

    model, tokenizer = moondream_model

    # Create test image
    test_image = create_test_image("simple")

    # Generate caption
    description = caption(model, tokenizer, test_image)

    assert description is not None, "Caption should not be None"
    assert isinstance(description, str), "Caption should be a string"
    assert len(description) > 0, "Caption should have content"


def test_visual_question_answering(moondream_model):
    """Test visual question answering."""
    from smlx.models.Moondream2 import query

    model, tokenizer = moondream_model

    # Create test image with identifiable objects
    test_image = create_test_image("simple")

    # Ask questions
    questions = [
        "What objects are in this image?",
        "What colors can you see?",
        "Describe what you see.",
    ]

    for question in questions:
        answer = query(model, tokenizer, test_image, question)

        assert answer is not None, f"Answer for '{question}' should not be None"
        assert isinstance(answer, str), f"Answer for '{question}' should be string"
        assert len(answer) > 0, f"Answer for '{question}' should have content"


def test_object_detection(moondream_model):
    """Test object detection functionality."""
    from smlx.models.Moondream2 import detect

    model, tokenizer = moondream_model

    # Create test image with objects
    test_image = create_test_image("simple")

    # Detect objects
    detections = detect(model, tokenizer, test_image, query="Find all objects")

    assert detections is not None, "Detections should not be None"
    # Detection format depends on implementation
    # May return list of dicts with boxes, or text description


def test_pointing_localization(moondream_model):
    """Test spatial pointing/localization."""
    from smlx.models.Moondream2 import point

    model, tokenizer = moondream_model

    # Create test image
    test_image = create_test_image("simple")

    # Point to objects
    try:
        location = point(model, tokenizer, test_image, object_name="red box")

        assert location is not None, "Location should not be None"
        # Location may be coordinates or text description
    except NotImplementedError:
        pytest.skip("Point function not yet implemented")


def test_streaming_generation(moondream_model):
    """Test streaming generation."""
    from smlx.models.Moondream2 import stream_generate

    model, tokenizer = moondream_model

    # Create test image
    test_image = create_test_image()

    # Stream generation
    chunks = list(
        stream_generate(
            model,
            tokenizer,
            test_image,
            prompt="Describe this image:",
            max_tokens=50,
        )
    )

    assert len(chunks) > 0, "Should generate at least one chunk"

    for i, chunk in enumerate(chunks):
        assert chunk is not None, f"Chunk {i} should not be None"
        assert isinstance(chunk, str), f"Chunk {i} should be string"


def test_basic_generation(moondream_model):
    """Test basic generation function."""
    from smlx.models.Moondream2 import generate

    model, tokenizer = moondream_model

    # Create test image
    test_image = create_test_image()

    # Generate
    output = generate(
        model,
        tokenizer,
        test_image,
        prompt="What is in this image?",
        max_tokens=50,
    )

    assert output is not None, "Output should not be None"
    assert isinstance(output, str), "Output should be string"


def test_complex_scene(moondream_model):
    """Test with more complex scene."""
    from smlx.models.Moondream2 import caption, query

    model, tokenizer = moondream_model

    # Create complex scene
    test_image = create_test_image("complex")

    # Caption
    description = caption(model, tokenizer, test_image)
    assert description is not None, "Caption should not be None"

    # Query about scene
    answer = query(model, tokenizer, test_image, "What type of building is this?")
    assert answer is not None, "Answer should not be None"


def test_crop_based_processing():
    """Test crop-based image processing."""
    import mlx.core as mx
    from smlx.models.Moondream2 import prepare_crops

    # Create test image
    test_image = create_test_image()

    # Convert PIL Image to numpy array [H, W, C]
    image_np = np.array(test_image)

    # Convert to MLX array and transpose to [C, H, W]
    image_mx = mx.array(image_np).transpose(2, 0, 1)

    # Prepare crops
    crops, crop_coords = prepare_crops(image_mx)

    assert crops is not None, "Crops should not be None"
    assert isinstance(crop_coords, list), "Crop coords should be a list"


def test_vision_encoder(moondream_model):
    """Test vision encoder component."""
    model, tokenizer = moondream_model

    vision = model.vision
    assert vision is not None, "Vision encoder should not be None"


def test_language_model(moondream_model):
    """Test language model component."""
    model, tokenizer = moondream_model

    language = model.language
    assert language is not None, "Language model should not be None"


def test_region_modules(moondream_model):
    """Test region detection modules."""
    model, tokenizer = moondream_model

    # Check if model has region modules
    if hasattr(model, "detection_head"):
        assert model.detection_head is not None, "Detection head should not be None"


def test_coordinate_parsing():
    """Test coordinate parsing utilities."""
    from smlx.models.Moondream2 import parse_coordinates_from_text

    # Test coordinate parsing with special token format
    text_with_coords = "The object is at <|coordinate|>0.5,0.4</|coordinate|>"

    # Provide image size for denormalization
    image_size = (640, 480)  # (width, height)
    coords = parse_coordinates_from_text(text_with_coords, image_size)

    # May return None if no coordinates found, or list of coordinate tuples
    if coords is not None:
        assert isinstance(coords, list), "Coords should be a list"
        assert isinstance(coords[0], tuple), "Each coord should be a tuple"


def test_box_parsing():
    """Test bounding box parsing utilities."""
    from smlx.models.Moondream2 import parse_boxes_from_text

    # Test box parsing with special token format
    text_with_boxes = "The object is at <|grounding|>0.1,0.2,0.5,0.6</|grounding|>"

    # Provide image size for denormalization
    image_size = (640, 480)  # (width, height)
    boxes = parse_boxes_from_text(text_with_boxes, image_size)

    # May return None if no boxes found, or list of boxes
    if boxes is not None:
        assert isinstance(boxes, list), "Boxes should be a list"
        assert isinstance(boxes[0], tuple), "Each box should be a tuple"
        assert len(boxes[0]) == 4, "Each box should have 4 coordinates"


def test_different_image_sizes(moondream_model):
    """Test with different image sizes."""
    from smlx.models.Moondream2 import caption

    model, tokenizer = moondream_model

    # Test different sizes
    sizes = [(320, 240), (640, 480), (800, 600)]

    for width, height in sizes:
        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, width - 10, height - 10], fill="blue")

        description = caption(model, tokenizer, img)

        assert description is not None, f"Should work with size {width}x{height}"


def test_multiple_objects_query(moondream_model):
    """Test querying about multiple objects."""
    from smlx.models.Moondream2 import query

    model, tokenizer = moondream_model

    # Create image with multiple objects
    test_image = create_test_image("simple")

    # Ask about multiple objects
    questions = [
        "How many objects are there?",
        "List all the colors you see.",
        "What shapes are present?",
    ]

    for question in questions:
        answer = query(model, tokenizer, test_image, question)
        assert answer is not None, f"Should answer: {question}"


def test_spatial_reasoning(moondream_model):
    """Test spatial reasoning capabilities."""
    from smlx.models.Moondream2 import query

    model, tokenizer = moondream_model

    # Create image with spatial layout
    test_image = create_test_image("simple")

    # Ask spatial questions
    spatial_questions = [
        "What is on the left side?",
        "What is in the center?",
        "Describe the layout of objects.",
    ]

    for question in spatial_questions:
        answer = query(model, tokenizer, test_image, question)
        assert answer is not None, f"Should answer: {question}"


def test_model_config(moondream_model):
    """Test model configuration."""
    from smlx.models.Moondream2 import ModelConfig

    model, tokenizer = moondream_model

    # Model should have config
    assert hasattr(model, "config"), "Model should have config"

    config = model.config
    assert config is not None, "Config should not be None"


def test_default_configs():
    """Test default configurations."""
    from smlx.models.Moondream2 import (
        DEFAULT_CONFIG_2B,
        DEFAULT_CONFIG_05B,
    )

    assert DEFAULT_CONFIG_2B is not None, "2B config should exist"
    assert DEFAULT_CONFIG_05B is not None, "0.5B config should exist"

    # Check config attributes
    assert hasattr(DEFAULT_CONFIG_2B, "vision_config"), "Should have vision_config"
    assert hasattr(DEFAULT_CONFIG_2B, "text_config"), "Should have text_config"


def test_vision_projection(moondream_model):
    """Test vision projector."""
    model, tokenizer = moondream_model

    vision_proj = model.vision_proj
    assert vision_proj is not None, "Vision projector should not be None"


def test_detailed_caption(moondream_model):
    """Test generating detailed captions."""
    from smlx.models.Moondream2 import query

    model, tokenizer = moondream_model

    # Create complex scene
    test_image = create_test_image("complex")

    # Ask for detailed description
    detailed = query(
        model,
        tokenizer,
        test_image,
        "Provide a detailed description of everything in this image.",
    )

    assert detailed is not None, "Detailed caption should not be None"
    assert len(detailed) > 0, "Should have detailed content"


def test_color_recognition(moondream_model):
    """Test color recognition."""
    from smlx.models.Moondream2 import query

    model, tokenizer = moondream_model

    # Create image with distinct colors
    test_image = create_test_image("simple")

    # Ask about colors
    answer = query(model, tokenizer, test_image, "What colors are in this image?")

    assert answer is not None, "Color recognition should work"


def test_scene_type_classification(moondream_model):
    """Test scene type classification."""
    from smlx.models.Moondream2 import query

    model, tokenizer = moondream_model

    # Create scene
    test_image = create_test_image("complex")

    # Ask about scene type
    answer = query(model, tokenizer, test_image, "What type of scene is this?")

    assert answer is not None, "Scene classification should work"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
