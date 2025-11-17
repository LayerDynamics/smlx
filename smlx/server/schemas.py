# Copyright © 2025 SMLX Project

"""
Pydantic schemas for SMLX Server API requests and responses.

Follows OpenAI API specification for compatibility.
"""

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Enums
# ============================================================================


class Role(str, Enum):
    """Message role in chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class FinishReason(str, Enum):
    """Reason why generation stopped."""

    STOP = "stop"  # Natural stop (EOS token)
    LENGTH = "length"  # Max tokens reached
    FUNCTION_CALL = "function_call"  # Function call requested
    CONTENT_FILTER = "content_filter"  # Content filter triggered


# ============================================================================
# Chat Completion Schemas
# ============================================================================


class Message(BaseModel):
    """A single message in a chat conversation."""

    role: Role = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(default=None, description="Name of the message sender")


class ChatCompletionRequest(BaseModel):
    """Request for chat completion (OpenAI compatible)."""

    model: str = Field(..., description="Model ID to use for generation")
    messages: list[Message] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-K sampling")
    max_tokens: int = Field(default=100, ge=1, le=8192, description="Maximum tokens to generate")
    stream: bool = Field(default=False, description="Whether to stream the response")
    stop: Optional[Union[str, list[str]]] = Field(default=None, description="Stop sequences")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    n: int = Field(default=1, ge=1, le=10, description="Number of completions to generate")
    user: Optional[str] = Field(default=None, description="Unique user identifier")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "mlx-community/SmolLM2-135M-Instruct",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is machine learning?"},
                ],
                "temperature": 0.7,
                "max_tokens": 150,
            }
        }
    )


class ChatCompletionChoice(BaseModel):
    """A single completion choice from chat completion."""

    index: int = Field(..., description="Index of this choice")
    message: Message = Field(..., description="Generated message")
    finish_reason: FinishReason = Field(..., description="Why generation stopped")


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens")


class ChatCompletionResponse(BaseModel):
    """Response from chat completion."""

    id: str = Field(..., description="Unique completion ID")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., description="Unix timestamp of when completion was created")
    model: str = Field(..., description="Model used for generation")
    choices: list[ChatCompletionChoice] = Field(..., description="List of completion choices")
    usage: Usage = Field(..., description="Token usage statistics")


class ChatCompletionChunk(BaseModel):
    """A chunk from streaming chat completion."""

    id: str = Field(..., description="Unique completion ID")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: list[dict[str, Any]] = Field(..., description="Streaming choices")


# ============================================================================
# Text Completion Schemas
# ============================================================================


class CompletionRequest(BaseModel):
    """Request for text completion (OpenAI compatible)."""

    model: str = Field(..., description="Model ID to use")
    prompt: Union[str, list[str]] = Field(..., description="Prompt(s) to generate from")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-K sampling")
    max_tokens: int = Field(default=100, ge=1, le=8192, description="Max tokens to generate")
    stream: bool = Field(default=False, description="Stream the response")
    stop: Optional[Union[str, list[str]]] = Field(default=None, description="Stop sequences")
    echo: bool = Field(default=False, description="Echo the prompt in response")
    n: int = Field(default=1, ge=1, le=10, description="Number of completions")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "mlx-community/SmolLM2-135M-Instruct",
                "prompt": "Write a Python function to",
                "temperature": 0.7,
                "max_tokens": 150,
            }
        }
    )


class CompletionChoice(BaseModel):
    """A single completion choice."""

    index: int = Field(..., description="Index of this choice")
    text: str = Field(..., description="Generated text")
    finish_reason: FinishReason = Field(..., description="Why generation stopped")


class CompletionResponse(BaseModel):
    """Response from text completion."""

    id: str = Field(..., description="Unique completion ID")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: list[CompletionChoice] = Field(..., description="Completion choices")
    usage: Usage = Field(..., description="Token usage")


# ============================================================================
# Audio Transcription Schemas
# ============================================================================


class AudioTranscriptionRequest(BaseModel):
    """Request for audio transcription."""

    file: bytes = Field(..., description="Audio file to transcribe")
    model: str = Field(default="whisper-tiny", description="Model ID to use")
    language: Optional[str] = Field(default=None, description="Language code (e.g., 'en')")
    prompt: Optional[str] = Field(default=None, description="Optional prompt to guide transcription")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    response_format: Literal["json", "text", "srt", "vtt"] = Field(
        default="json", description="Response format"
    )


class AudioTranscriptionResponse(BaseModel):
    """Response from audio transcription."""

    text: str = Field(..., description="Transcribed text")
    language: Optional[str] = Field(default=None, description="Detected language")
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    segments: Optional[list[dict[str, Any]]] = Field(default=None, description="Timestamped segments")


# ============================================================================
# Embedding Schemas
# ============================================================================


class EmbeddingRequest(BaseModel):
    """Request for text embeddings."""

    model: str = Field(..., description="Model ID to use")
    input: Union[str, list[str]] = Field(..., description="Text(s) to embed")
    encoding_format: Literal["float", "base64"] = Field(default="float", description="Encoding format")
    user: Optional[str] = Field(default=None, description="User identifier")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "all-MiniLM-L6-v2",
                "input": "The quick brown fox jumps over the lazy dog",
            }
        }
    )


class Embedding(BaseModel):
    """A single embedding."""

    object: Literal["embedding"] = "embedding"
    embedding: list[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index of this embedding")


class EmbeddingResponse(BaseModel):
    """Response from embedding request."""

    object: Literal["list"] = "list"
    data: list[Embedding] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used")
    usage: Usage = Field(..., description="Token usage")


# ============================================================================
# Model Listing Schemas
# ============================================================================


class ModelInfo(BaseModel):
    """Information about a model."""

    id: str = Field(..., description="Model ID")
    object: Literal["model"] = "model"
    created: int = Field(..., description="Unix timestamp")
    owned_by: str = Field(default="smlx", description="Owner of the model")
    permission: list[dict[str, Any]] = Field(default_factory=list, description="Permissions")
    root: Optional[str] = Field(default=None, description="Root model")
    parent: Optional[str] = Field(default=None, description="Parent model")


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"] = "list"
    data: list[ModelInfo] = Field(..., description="List of models")


# ============================================================================
# Error Schemas
# ============================================================================


class ErrorDetail(BaseModel):
    """Error detail information."""

    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: Optional[str] = Field(default=None, description="Parameter that caused error")
    code: Optional[str] = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """Error response."""

    error: ErrorDetail = Field(..., description="Error details")
