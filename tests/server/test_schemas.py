# Copyright © 2025 SMLX Project

"""
Tests for server request/response schemas.
"""

import pytest
from pydantic import ValidationError

from smlx.server.schemas import (
    # Audio
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    # Chat Completion
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    # Text Completion
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    # Embeddings
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    # Errors
    ErrorDetail,
    ErrorResponse,
    # Enums
    FinishReason,
    # Messages
    Message,
    # Models
    ModelInfo,
    ModelList,
    Role,
    # Usage
    Usage,
)

# ============================================================================
# Message Schema Tests
# ============================================================================


@pytest.mark.unit
class TestMessage:
    """Tests for Message schema."""

    def test_valid_message(self):
        """Test creating a valid message."""
        msg = Message(role=Role.USER, content="Hello, world!")
        assert msg.role == Role.USER
        assert msg.content == "Hello, world!"
        assert msg.name is None

    def test_message_with_name(self):
        """Test message with optional name field."""
        msg = Message(role=Role.ASSISTANT, content="Hi!", name="Bot")
        assert msg.name == "Bot"

    def test_message_validation_error(self):
        """Test message validation errors."""
        with pytest.raises(ValidationError):
            Message(role="invalid_role", content="test")  # type: ignore[arg-type]


# ============================================================================
# Chat Completion Schema Tests
# ============================================================================


@pytest.mark.unit
class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest schema."""

    def test_minimal_request(self):
        """Test creating minimal chat completion request."""
        req = ChatCompletionRequest(
            model="mlx-community/SmolLM2-135M-Instruct",
            messages=[Message(role=Role.USER, content="Hello")],
        )
        assert req.model == "mlx-community/SmolLM2-135M-Instruct"
        assert len(req.messages) == 1
        assert req.temperature == 1.0  # default
        assert req.max_tokens == 100  # default
        assert req.stream is False  # default

    def test_full_request(self):
        """Test creating fully specified chat completion request."""
        req = ChatCompletionRequest(
            model="SmolLM2-135M",
            messages=[
                Message(role=Role.SYSTEM, content="You are helpful"),
                Message(role=Role.USER, content="Hi"),
            ],
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=200,
            stream=True,
            stop=["END"],
            n=2,
        )
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.top_k == 50
        assert req.max_tokens == 200
        assert req.stream is True
        assert req.stop == ["END"]
        assert req.n == 2

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperature
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role=Role.USER, content="test")],
            temperature=0.5,
        )
        assert req.temperature == 0.5

        # Invalid temperature (too low)
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[Message(role=Role.USER, content="test")],
                temperature=-0.1,
            )

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[Message(role=Role.USER, content="test")],
                temperature=2.1,
            )

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid max_tokens
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role=Role.USER, content="test")],
            max_tokens=50,
        )
        assert req.max_tokens == 50

        # Invalid max_tokens (too low)
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[Message(role=Role.USER, content="test")],
                max_tokens=0,
            )


@pytest.mark.unit
class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse schema."""

    def test_response_creation(self):
        """Test creating chat completion response."""
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="SmolLM2-135M",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role=Role.ASSISTANT, content="Hello!"),
                    finish_reason=FinishReason.STOP,
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert resp.id == "chatcmpl-123"
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.content == "Hello!"
        assert resp.usage.total_tokens == 15


# ============================================================================
# Text Completion Schema Tests
# ============================================================================


@pytest.mark.unit
class TestCompletionRequest:
    """Tests for CompletionRequest schema."""

    def test_single_prompt(self):
        """Test completion request with single prompt."""
        req = CompletionRequest(model="SmolLM2-135M", prompt="Hello")
        assert req.prompt == "Hello"
        assert isinstance(req.prompt, str)

    def test_multiple_prompts(self):
        """Test completion request with multiple prompts."""
        req = CompletionRequest(model="SmolLM2-135M", prompt=["Hello", "Hi"])
        assert req.prompt == ["Hello", "Hi"]
        assert isinstance(req.prompt, list)

    def test_echo_parameter(self):
        """Test echo parameter."""
        req = CompletionRequest(model="test", prompt="test", echo=True)
        assert req.echo is True

    def test_stop_sequences(self):
        """Test stop sequences."""
        # Single stop sequence
        req1 = CompletionRequest(model="test", prompt="test", stop="END")
        assert req1.stop == "END"

        # Multiple stop sequences
        req2 = CompletionRequest(model="test", prompt="test", stop=["END", "STOP"])
        assert req2.stop == ["END", "STOP"]


@pytest.mark.unit
class TestCompletionResponse:
    """Tests for CompletionResponse schema."""

    def test_response_creation(self):
        """Test creating completion response."""
        resp = CompletionResponse(
            id="cmpl-123",
            created=1234567890,
            model="SmolLM2-135M",
            choices=[
                CompletionChoice(index=0, text="Generated text", finish_reason=FinishReason.STOP)
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        assert resp.id == "cmpl-123"
        assert resp.object == "text_completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].text == "Generated text"


# ============================================================================
# Audio Transcription Schema Tests
# ============================================================================


@pytest.mark.unit
class TestAudioTranscriptionRequest:
    """Tests for AudioTranscriptionRequest schema."""

    def test_minimal_request(self):
        """Test minimal audio transcription request."""
        req = AudioTranscriptionRequest(file=b"audio_data")
        assert req.file == b"audio_data"
        assert req.model == "whisper-tiny"  # default
        assert req.temperature == 0.0  # default
        assert req.response_format == "json"  # default

    def test_full_request(self):
        """Test full audio transcription request."""
        req = AudioTranscriptionRequest(
            file=b"audio_data",
            model="whisper-tiny",
            language="en",
            prompt="Transcription prompt",
            temperature=0.2,
            response_format="srt",
        )
        assert req.language == "en"
        assert req.prompt == "Transcription prompt"
        assert req.temperature == 0.2
        assert req.response_format == "srt"


@pytest.mark.unit
class TestAudioTranscriptionResponse:
    """Tests for AudioTranscriptionResponse schema."""

    def test_response_creation(self):
        """Test creating audio transcription response."""
        resp = AudioTranscriptionResponse(
            text="Transcribed text",
            language="en",
            duration=10.5,
            segments=[{"start": 0.0, "end": 5.0, "text": "First segment"}],
        )
        assert resp.text == "Transcribed text"
        assert resp.language == "en"
        assert resp.duration == 10.5
        assert resp.segments is not None
        assert len(resp.segments) == 1


# ============================================================================
# Embedding Schema Tests
# ============================================================================


@pytest.mark.unit
class TestEmbeddingRequest:
    """Tests for EmbeddingRequest schema."""

    def test_single_input(self):
        """Test embedding request with single input."""
        req = EmbeddingRequest(model="all-MiniLM-L6-v2", input="Hello world")
        assert req.input == "Hello world"
        assert isinstance(req.input, str)

    def test_multiple_inputs(self):
        """Test embedding request with multiple inputs."""
        req = EmbeddingRequest(model="all-MiniLM-L6-v2", input=["Hello", "World"])
        assert req.input == ["Hello", "World"]
        assert isinstance(req.input, list)

    def test_encoding_format(self):
        """Test encoding format parameter."""
        req = EmbeddingRequest(model="test", input="test", encoding_format="base64")
        assert req.encoding_format == "base64"


@pytest.mark.unit
class TestEmbeddingResponse:
    """Tests for EmbeddingResponse schema."""

    def test_response_creation(self):
        """Test creating embedding response."""
        resp = EmbeddingResponse(
            data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0)],
            model="all-MiniLM-L6-v2",
            usage=Usage(prompt_tokens=5, completion_tokens=0, total_tokens=5),
        )
        assert resp.object == "list"
        assert len(resp.data) == 1
        assert resp.data[0].embedding == [0.1, 0.2, 0.3]
        assert resp.data[0].object == "embedding"


# ============================================================================
# Model Schema Tests
# ============================================================================


@pytest.mark.unit
class TestModelInfo:
    """Tests for ModelInfo schema."""

    def test_model_info_creation(self):
        """Test creating model info."""
        info = ModelInfo(id="SmolLM2-135M", created=1234567890)
        assert info.id == "SmolLM2-135M"
        assert info.object == "model"
        assert info.owned_by == "smlx"  # default
        assert info.created == 1234567890


@pytest.mark.unit
class TestModelList:
    """Tests for ModelList schema."""

    def test_model_list_creation(self):
        """Test creating model list."""
        models = [
            ModelInfo(id="SmolLM2-135M", created=123),
            ModelInfo(id="SmolLM2-360M", created=456),
        ]
        model_list = ModelList(data=models)
        assert model_list.object == "list"
        assert len(model_list.data) == 2
        assert model_list.data[0].id == "SmolLM2-135M"


# ============================================================================
# Error Schema Tests
# ============================================================================


@pytest.mark.unit
class TestErrorDetail:
    """Tests for ErrorDetail schema."""

    def test_error_detail_creation(self):
        """Test creating error detail."""
        error = ErrorDetail(
            message="Invalid request", type="validation_error", code="invalid_param"
        )
        assert error.message == "Invalid request"
        assert error.type == "validation_error"
        assert error.code == "invalid_param"


@pytest.mark.unit
class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_error_response_creation(self):
        """Test creating error response."""
        error_detail = ErrorDetail(message="Error occurred", type="server_error")
        error_resp = ErrorResponse(error=error_detail)
        assert error_resp.error.message == "Error occurred"


# ============================================================================
# Usage Schema Tests
# ============================================================================


@pytest.mark.unit
class TestUsage:
    """Tests for Usage schema."""

    def test_usage_creation(self):
        """Test creating usage statistics."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30
