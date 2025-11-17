"""
Tests for Model Inference Runner.

This module tests the ModelRunner, InferenceConfig, and preprocessing utilities.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import mlx.core as mx

from smlx.models.smlx_runner import (
    InferenceConfig,
    ModelRunner,
    preprocess_text_input,
    preprocess_chat_input,
    preprocess_image_input,
    preprocess_audio_input,
)


# ============================================================================
# InferenceConfig Tests
# ============================================================================


@pytest.mark.unit
def test_inference_config_defaults():
    """Test default InferenceConfig values."""
    config = InferenceConfig()

    assert config.max_tokens == 100
    assert config.temperature == 0.7
    assert config.top_p == 1.0
    assert config.top_k == 0
    assert config.stop_strings is None
    assert config.stream is False
    assert config.batch_size == 1
    assert config.repetition_penalty == 1.0
    assert config.repetition_context_size == 20
    assert config.verbose is False
    assert config.use_cache is True
    assert config.cache_config == {}


@pytest.mark.unit
def test_inference_config_custom():
    """Test custom InferenceConfig."""
    config = InferenceConfig(
        max_tokens=200,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        stop_strings=["STOP", "END"],
        stream=True,
        verbose=True,
    )

    assert config.max_tokens == 200
    assert config.temperature == 0.9
    assert config.top_p == 0.95
    assert config.top_k == 50
    assert config.stop_strings == ["STOP", "END"]
    assert config.stream is True
    assert config.verbose is True


@pytest.mark.unit
def test_inference_config_validate_valid():
    """Test validation with valid config."""
    config = InferenceConfig(
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=10,
        batch_size=4,
    )

    # Should not raise
    config.validate()


@pytest.mark.unit
def test_inference_config_validate_invalid_max_tokens():
    """Test validation with invalid max_tokens."""
    config = InferenceConfig(max_tokens=0)

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        config.validate()

    config = InferenceConfig(max_tokens=-10)
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        config.validate()


@pytest.mark.unit
def test_inference_config_validate_invalid_temperature():
    """Test validation with invalid temperature."""
    config = InferenceConfig(temperature=-0.1)

    with pytest.raises(ValueError, match="temperature must be non-negative"):
        config.validate()


@pytest.mark.unit
def test_inference_config_validate_invalid_top_p():
    """Test validation with invalid top_p."""
    config = InferenceConfig(top_p=1.5)

    with pytest.raises(ValueError, match="top_p must be in"):
        config.validate()

    config = InferenceConfig(top_p=-0.1)
    with pytest.raises(ValueError, match="top_p must be in"):
        config.validate()


@pytest.mark.unit
def test_inference_config_validate_invalid_top_k():
    """Test validation with invalid top_k."""
    config = InferenceConfig(top_k=-1)

    with pytest.raises(ValueError, match="top_k must be non-negative"):
        config.validate()


@pytest.mark.unit
def test_inference_config_validate_invalid_batch_size():
    """Test validation with invalid batch_size."""
    config = InferenceConfig(batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        config.validate()


# ============================================================================
# Preprocessing Tests
# ============================================================================


@pytest.mark.unit
def test_preprocess_text_input_single():
    """Test preprocessing single text prompt."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4]

    prompts, tokenized = preprocess_text_input("Hello world", tokenizer)

    assert prompts == ["Hello world"]
    assert len(tokenized) == 1
    tokenizer.encode.assert_called_once_with("Hello world")


@pytest.mark.unit
def test_preprocess_text_input_list():
    """Test preprocessing list of prompts."""
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = [[1, 2], [3, 4, 5], [6]]

    prompts, tokenized = preprocess_text_input(
        ["Hello", "World", "Test"], tokenizer
    )

    assert prompts == ["Hello", "World", "Test"]
    assert len(tokenized) == 3
    assert tokenizer.encode.call_count == 3


@pytest.mark.unit
def test_preprocess_text_input_max_length():
    """Test preprocessing with max_length truncation."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]

    prompts, tokenized = preprocess_text_input(
        "Hello world", tokenizer, max_length=5
    )

    assert len(tokenized[0]) == 5


@pytest.mark.unit
def test_preprocess_text_input_empty():
    """Test preprocessing with empty prompt."""
    tokenizer = MagicMock()

    with pytest.raises(ValueError, match="is empty"):
        preprocess_text_input("   ", tokenizer)


@pytest.mark.unit
def test_preprocess_text_input_invalid_type():
    """Test preprocessing with invalid type."""
    tokenizer = MagicMock()

    with pytest.raises(TypeError, match="must be string"):
        preprocess_text_input([123], tokenizer)


@pytest.mark.unit
def test_preprocess_chat_input_valid():
    """Test preprocessing valid chat messages."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4]

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    prompt, tokens = preprocess_chat_input(messages, tokenizer)

    assert "User: Hello" in prompt
    assert "Assistant: Hi there" in prompt
    assert "Assistant:" in prompt  # Prefix for response
    tokenizer.encode.assert_called_once()


@pytest.mark.unit
def test_preprocess_chat_input_empty():
    """Test preprocessing empty messages."""
    tokenizer = MagicMock()

    with pytest.raises(ValueError, match="Messages list is empty"):
        preprocess_chat_input([], tokenizer)


@pytest.mark.unit
def test_preprocess_chat_input_invalid_format():
    """Test preprocessing with invalid message format."""
    tokenizer = MagicMock()

    # Not a dict
    with pytest.raises(TypeError, match="must be dict"):
        preprocess_chat_input([123], tokenizer)

    # Missing role
    with pytest.raises(ValueError, match="missing 'role' or 'content'"):
        preprocess_chat_input([{"content": "Hello"}], tokenizer)

    # Missing content
    with pytest.raises(ValueError, match="missing 'role' or 'content'"):
        preprocess_chat_input([{"role": "user"}], tokenizer)


@pytest.mark.unit
def test_preprocess_chat_input_max_length():
    """Test preprocessing chat with max_length."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]

    messages = [{"role": "user", "content": "Hello"}]

    prompt, tokens = preprocess_chat_input(messages, tokenizer, max_length=5)

    assert len(tokens) == 5


@pytest.mark.unit
def test_preprocess_image_input_pil():
    """Test preprocessing PIL image."""
    from PIL import Image
    import numpy as np

    # Create mock PIL image
    mock_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    processor = MagicMock()
    processor.process_image.return_value = mx.zeros((3, 224, 224))

    result = preprocess_image_input(mock_image, processor)

    processor.process_image.assert_called_once()


@pytest.mark.unit
def test_preprocess_image_input_path():
    """Test preprocessing image from path."""
    from PIL import Image
    import numpy as np

    # Mock image loading
    mock_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    processor = MagicMock()
    processor.process_image.return_value = mx.zeros((3, 224, 224))

    with patch("PIL.Image.open", return_value=mock_image):
        with patch.object(Path, "exists", return_value=True):
            result = preprocess_image_input("test.jpg", processor)

    processor.process_image.assert_called_once()


@pytest.mark.unit
def test_preprocess_image_input_path_not_found():
    """Test preprocessing with non-existent image path."""
    processor = MagicMock()

    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Image not found"):
            preprocess_image_input("missing.jpg", processor)


@pytest.mark.unit
def test_preprocess_image_input_callable_processor():
    """Test preprocessing with callable processor."""
    from PIL import Image
    import numpy as np

    mock_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    # Processor without process_image but is callable
    processor = MagicMock()
    processor.process_image = None
    del processor.process_image  # Remove the attribute
    processor.return_value = mx.zeros((3, 224, 224))

    result = preprocess_image_input(mock_image, processor)

    processor.assert_called_once()


@pytest.mark.unit
def test_preprocess_image_input_invalid_processor():
    """Test preprocessing with invalid processor."""
    from PIL import Image
    import numpy as np

    mock_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    # Processor that's not callable and has no process_image
    processor = MagicMock()
    del processor.process_image
    processor.__call__ = None

    # Make it not callable
    with patch("smlx.models.smlx_runner.callable", return_value=False):
        with pytest.raises(AttributeError, match="doesn't have process_image"):
            preprocess_image_input(mock_image, processor)


# ============================================================================
# ModelRunner Tests
# ============================================================================


@pytest.mark.unit
def test_runner_initialization():
    """Test ModelRunner initialization."""
    runner = ModelRunner()

    assert runner is not None
    assert runner.router is not None


@pytest.mark.unit
def test_runner_run_text_generation():
    """Test basic text generation."""
    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]

    config = InferenceConfig(max_tokens=50)

    with patch.object(runner.router, "route_text_generation", return_value="Generated text"):
        result = runner.run(
            model=model,
            tokenizer=tokenizer,
            model_type="smollm2-135m",
            prompt="Hello",
            config=config,
        )

    assert result == "Generated text"


@pytest.mark.unit
def test_runner_run_default_config():
    """Test running with default config."""
    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()

    with patch.object(runner.router, "route_text_generation", return_value="Generated"):
        result = runner.run(
            model=model,
            tokenizer=tokenizer,
            model_type="smollm2-135m",
            prompt="Test",
        )

    assert result == "Generated"


@pytest.mark.unit
def test_runner_run_streaming():
    """Test streaming generation."""
    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()

    config = InferenceConfig(stream=True)

    def mock_stream():
        yield "Hello"
        yield " "
        yield "world"

    with patch.object(runner.router, "route_streaming_generation", return_value=mock_stream()):
        result = runner.run(
            model=model,
            tokenizer=tokenizer,
            model_type="smollm2-135m",
            prompt="Test",
            config=config,
        )

    # Result should be a generator
    assert hasattr(result, "__iter__")


@pytest.mark.unit
def test_runner_run_batch():
    """Test batch generation via run()."""
    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]

    config = InferenceConfig()

    with patch.object(runner.router, "route_text_generation", side_effect=["Gen1", "Gen2", "Gen3"]):
        result = runner.run(
            model=model,
            tokenizer=tokenizer,
            model_type="smollm2-135m",
            prompt=["Test1", "Test2", "Test3"],
            config=config,
        )

    assert result == ["Gen1", "Gen2", "Gen3"]


@pytest.mark.unit
def test_runner_run_multimodal():
    """Test multimodal generation with image."""
    from PIL import Image
    import numpy as np

    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()
    processor = MagicMock()
    processor.process_image.return_value = mx.zeros((3, 224, 224))

    mock_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    config = InferenceConfig()

    with patch.object(runner.router, "route_multimodal", return_value="Image description"):
        result = runner.run(
            model=model,
            tokenizer=tokenizer,
            model_type="smolvlm-256m",
            prompt="Describe this:",
            processor=processor,
            image=mock_image,
            config=config,
        )

    assert result == "Image description"


@pytest.mark.unit
def test_runner_run_multimodal_no_processor():
    """Test multimodal without processor raises error."""
    from PIL import Image
    import numpy as np

    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()
    mock_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    config = InferenceConfig()

    with pytest.raises(ValueError, match="Image processor required"):
        runner.run(
            model=model,
            tokenizer=tokenizer,
            model_type="smolvlm-256m",
            prompt="Describe",
            image=mock_image,
            processor=None,
            config=config,
        )


@pytest.mark.unit
def test_runner_run_multimodal_unsupported():
    """Test multimodal with text-only model raises error."""
    from PIL import Image
    import numpy as np

    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()
    processor = MagicMock()
    mock_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    config = InferenceConfig()

    with pytest.raises(ValueError, match="doesn't support image input"):
        runner.run(
            model=model,
            tokenizer=tokenizer,
            model_type="smollm2-135m",  # Text-only model
            prompt="Describe",
            image=mock_image,
            processor=processor,
            config=config,
        )


@pytest.mark.unit
def test_runner_run_chat():
    """Test chat generation."""
    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]

    messages = [{"role": "user", "content": "Hello"}]
    config = InferenceConfig()

    with patch.object(runner.router, "route_chat", return_value="Hi there!"):
        result = runner.run_chat(
            model=model,
            tokenizer=tokenizer,
            model_type="smollm2-135m",
            messages=messages,
            config=config,
        )

    assert result == "Hi there!"


@pytest.mark.unit
def test_runner_run_chat_unsupported():
    """Test chat with unsupported model."""
    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()

    messages = [{"role": "user", "content": "Hello"}]
    config = InferenceConfig()

    with pytest.raises(ValueError, match="doesn't support chat"):
        runner.run_chat(
            model=model,
            tokenizer=tokenizer,
            model_type="whisper-tiny",  # Audio model
            messages=messages,
            config=config,
        )


@pytest.mark.unit
def test_runner_run_chat_invalid_config():
    """Test chat with invalid config."""
    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()

    messages = [{"role": "user", "content": "Hello"}]
    config = InferenceConfig(max_tokens=-10)

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        runner.run_chat(
            model=model,
            tokenizer=tokenizer,
            model_type="smollm2-135m",
            messages=messages,
            config=config,
        )


@pytest.mark.unit
def test_runner_run_batch_method():
    """Test run_batch method."""
    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]

    prompts = ["Hello", "World", "Test"]
    config = InferenceConfig()

    with patch.object(runner.router, "route_text_generation", side_effect=["Gen1", "Gen2", "Gen3"]) as mock_route:
        results = runner.run_batch(
            model=model,
            tokenizer=tokenizer,
            model_type="smollm2-135m",
            prompts=prompts,
            config=config,
        )

    assert results == ["Gen1", "Gen2", "Gen3"]
    assert mock_route.call_count == 3


@pytest.mark.unit
def test_runner_run_batch_invalid_config():
    """Test run_batch with invalid config."""
    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()

    prompts = ["Hello"]
    config = InferenceConfig(temperature=-1.0)

    with pytest.raises(ValueError, match="temperature must be non-negative"):
        runner.run_batch(
            model=model,
            tokenizer=tokenizer,
            model_type="smollm2-135m",
            prompts=prompts,
            config=config,
        )


@pytest.mark.unit
def test_runner_run_invalid_config():
    """Test run with invalid config."""
    runner = ModelRunner()

    model = MagicMock()
    tokenizer = MagicMock()

    config = InferenceConfig(top_p=2.0)

    with pytest.raises(ValueError, match="top_p must be in"):
        runner.run(
            model=model,
            tokenizer=tokenizer,
            model_type="smollm2-135m",
            prompt="Test",
            config=config,
        )
