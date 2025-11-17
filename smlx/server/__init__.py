# Copyright © 2025 SMLX Project

"""
SMLX Server - FastAPI-based inference server for small MLX models.

Provides OpenAI-compatible API endpoints for:
- Text generation (/v1/completions)
- Chat completions (/v1/chat/completions)
- Audio transcription (/v1/audio/transcriptions)
- Embeddings (/v1/embeddings)
- Model management (/v1/models)

Example:
    Start the server:
    >>> python -m smlx.server.app

    Or with uvicorn:
    >>> uvicorn smlx.server.app:app --host 0.0.0.0 --port 8000

Features:
- OpenAI-compatible API
- Streaming support (SSE)
- Model caching
- Rate limiting
- CORS support
- Error handling
"""

from .app import app
from .model_manager import ModelManager
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelList,
    ModelInfo,
)

__all__ = [
    "app",
    "ModelManager",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "CompletionRequest",
    "CompletionResponse",
    "AudioTranscriptionRequest",
    "AudioTranscriptionResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ModelList",
    "ModelInfo",
]

__version__ = "0.1.0"
