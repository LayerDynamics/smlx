#!/usr/bin/env python3
# Copyright � 2025 SMLX Project

"""
Model Inference Router for SMLX.

This module provides intelligent routing of inference requests to appropriate
model implementations based on model type and capabilities. It eliminates
hardcoded imports in server routes and enables dynamic model dispatch.

Architecture:
    ModelCapabilities - Defines what a model can do
    ModelRouter - Routes requests to correct model implementation
    CAPABILITY_MAP - Registry of model capabilities

Example:
    >>> from smlx.models.smlx_router import ModelRouter, get_router
    >>>
    >>> router = get_router()
    >>>
    >>> # Check what a model can do
    >>> caps = router.get_capabilities("smollm2-135m")
    >>> if caps.can_chat:
    ...     response = router.route_chat(
    ...         model_type="smollm2-135m",
    ...         model=model,
    ...         tokenizer=tokenizer,
    ...         messages=[{"role": "user", "content": "Hello"}]
    ...     )
    >>>
    >>> # Route multimodal generation
    >>> response = router.route_multimodal(
    ...     model_type="smolvlm-256m",
    ...     model=model,
    ...     processor=processor,
    ...     prompt="Describe this image:",
    ...     image=image_path
    ... )
"""

from __future__ import annotations

import importlib
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Callable

import mlx.nn as nn

from .registry import MODEL_REGISTRY, infer_model_type

# ============================================================================
# Model Capabilities
# ============================================================================


@dataclass
class ModelCapabilities:
    """
    Defines the capabilities of a model.

    Attributes:
        can_chat: Whether model supports chat/instruct format
        can_complete: Whether model supports text completion
        can_stream: Whether model supports streaming generation
        can_transcribe: Whether model can transcribe audio
        can_caption: Whether model can caption images
        can_detect: Whether model can detect objects/regions
        requires_image: Whether model requires image input
        requires_audio: Whether model requires audio input
        max_context_length: Maximum context length in tokens
        supports_batch: Whether model supports batch processing
        modality: Primary modality (text, vision-language, audio)
        category: Model category from registry
    """

    can_chat: bool = False
    can_complete: bool = True
    can_stream: bool = True
    can_transcribe: bool = False
    can_caption: bool = False
    can_detect: bool = False
    requires_image: bool = False
    requires_audio: bool = False
    max_context_length: int = 2048
    supports_batch: bool = False
    modality: str = "text"
    category: str = "language"
    extra_capabilities: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        caps = []
        if self.can_chat:
            caps.append("chat")
        if self.can_complete:
            caps.append("complete")
        if self.can_stream:
            caps.append("stream")
        if self.can_transcribe:
            caps.append("transcribe")
        if self.can_caption:
            caps.append("caption")
        if self.can_detect:
            caps.append("detect")

        return f"ModelCapabilities({', '.join(caps)}, modality={self.modality})"


# ============================================================================
# Capability Registry
# ============================================================================

# Capability mapping for all supported models
CAPABILITY_MAP: dict[str, ModelCapabilities] = {
    # Language Models
    "smollm2-135m": ModelCapabilities(
        can_chat=True,
        can_complete=True,
        can_stream=True,
        max_context_length=8192,
        supports_batch=True,
        modality="text",
        category="language",
    ),
    "smollm2-360m": ModelCapabilities(
        can_chat=True,
        can_complete=True,
        can_stream=True,
        max_context_length=8192,
        supports_batch=True,
        modality="text",
        category="language",
    ),
    "smollm2": ModelCapabilities(  # Alias for 135M
        can_chat=True,
        can_complete=True,
        can_stream=True,
        max_context_length=8192,
        supports_batch=True,
        modality="text",
        category="language",
    ),
    # Vision-Language Models
    "smolvlm-256m": ModelCapabilities(
        can_chat=True,
        can_complete=True,
        can_stream=True,
        can_caption=True,
        requires_image=True,
        max_context_length=8192,
        modality="vision-language",
        category="vision-language",
    ),
    "smolvlm-500m": ModelCapabilities(
        can_chat=True,
        can_complete=True,
        can_stream=True,
        can_caption=True,
        requires_image=True,
        max_context_length=8192,
        modality="vision-language",
        category="vision-language",
    ),
    "smolvlm": ModelCapabilities(  # Alias for 256M
        can_chat=True,
        can_complete=True,
        can_stream=True,
        can_caption=True,
        requires_image=True,
        max_context_length=8192,
        modality="vision-language",
        category="vision-language",
    ),
    "nanovlm": ModelCapabilities(
        can_chat=True,
        can_complete=True,
        can_caption=True,
        requires_image=True,
        max_context_length=8192,
        modality="vision-language",
        category="vision-language",
    ),
    "moondream2": ModelCapabilities(
        can_chat=True,
        can_complete=True,
        can_caption=True,
        can_detect=True,
        requires_image=True,
        max_context_length=2048,
        modality="vision-language",
        category="vision-language",
        extra_capabilities={
            "can_point": True,  # Spatial localization
            "can_detect_regions": True,  # Region detection
        },
    ),
    "tinyllava": ModelCapabilities(
        can_chat=True,
        can_complete=True,
        can_caption=True,
        requires_image=True,
        max_context_length=4096,
        modality="vision-language",
        category="vision-language",
    ),
    # Audio Models
    "whisper-tiny": ModelCapabilities(
        can_complete=False,
        can_stream=False,
        can_transcribe=True,
        requires_audio=True,
        max_context_length=448,  # Whisper uses fixed-length
        modality="audio",
        category="audio",
        extra_capabilities={
            "can_translate": True,  # Translate to English
            "num_languages": 99,
            "can_detect_language": True,
        },
    ),
    "whisper": ModelCapabilities(  # Alias
        can_complete=False,
        can_stream=False,
        can_transcribe=True,
        requires_audio=True,
        max_context_length=448,
        modality="audio",
        category="audio",
        extra_capabilities={
            "can_translate": True,
            "num_languages": 99,
            "can_detect_language": True,
        },
    ),
    "yamnet": ModelCapabilities(
        can_complete=False,
        can_stream=False,
        requires_audio=True,
        modality="audio",
        category="audio",
        extra_capabilities={
            "can_classify_audio": True,
            "num_classes": 521,
        },
    ),
    "silero-vad": ModelCapabilities(
        can_complete=False,
        can_stream=False,
        requires_audio=True,
        modality="audio",
        category="audio",
        extra_capabilities={
            "can_detect_voice": True,
        },
    ),
    # Document/OCR Models
    "trocr-small": ModelCapabilities(
        can_complete=False,
        can_stream=False,
        requires_image=True,
        modality="vision",
        category="document",
        extra_capabilities={
            "can_ocr": True,
            "supports_handwriting": True,
        },
    ),
    "trocr": ModelCapabilities(  # Alias
        can_complete=False,
        can_stream=False,
        requires_image=True,
        modality="vision",
        category="document",
        extra_capabilities={
            "can_ocr": True,
            "supports_handwriting": True,
        },
    ),
    "donut-base": ModelCapabilities(
        can_complete=False,
        can_stream=False,
        requires_image=True,
        modality="vision",
        category="document",
        extra_capabilities={
            "can_understand_document": True,
            "ocr_free": True,
        },
    ),
    "donut": ModelCapabilities(  # Alias
        can_complete=False,
        can_stream=False,
        requires_image=True,
        modality="vision",
        category="document",
        extra_capabilities={
            "can_understand_document": True,
            "ocr_free": True,
        },
    ),
    # Embedding Models
    "minilm": ModelCapabilities(
        can_complete=False,
        can_stream=False,
        modality="text",
        category="embedding",
        extra_capabilities={
            "can_embed": True,
            "embedding_dim": 384,
        },
    ),
    "all-minilm-l6-v2": ModelCapabilities(
        can_complete=False,
        can_stream=False,
        modality="text",
        category="embedding",
        extra_capabilities={
            "can_embed": True,
            "embedding_dim": 384,
        },
    ),
}


# ============================================================================
# Model Router
# ============================================================================


class ModelRouter:
    """
    Routes inference requests to appropriate model implementations.

    The router uses model type to determine capabilities and dispatch
    to the correct generation/inference function.

    Example:
        >>> router = ModelRouter()
        >>>
        >>> # Check capabilities
        >>> caps = router.get_capabilities("smollm2-135m")
        >>> print(caps)
        ModelCapabilities(chat, complete, stream, modality=text)
        >>>
        >>> # Route chat request
        >>> response = router.route_chat(
        ...     model_type="smollm2-135m",
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """

    def __init__(self):
        """Initialize the model router."""
        self._function_cache: dict[str, Callable] = {}

    def get_capabilities(self, model_id_or_type: str) -> ModelCapabilities:
        """
        Get capabilities for a model.

        Args:
            model_id_or_type: Model type (e.g., "smollm2-135m") or
                             HuggingFace ID (e.g., "mlx-community/SmolLM2-135M-Instruct")

        Returns:
            ModelCapabilities instance

        Raises:
            ValueError: If model type is unknown

        Example:
            >>> router = ModelRouter()
            >>> caps = router.get_capabilities("smollm2-135m")
            >>> if caps.can_chat:
            ...     print("Model supports chat!")
        """
        # Try to infer model type from ID if needed
        model_type = infer_model_type(model_id_or_type)
        if model_type is None:
            # Direct lookup
            model_type = model_id_or_type.lower()

        if model_type not in CAPABILITY_MAP:
            raise ValueError(
                f"Unknown model type: {model_type}\n"
                f"Supported types: {', '.join(CAPABILITY_MAP.keys())}"
            )

        return CAPABILITY_MAP[model_type]

    def can_handle(self, model_type: str, capability: str) -> bool:
        """
        Check if a model has a specific capability.

        Args:
            model_type: Model type key
            capability: Capability name (e.g., "can_chat", "can_transcribe")

        Returns:
            True if model has capability, False otherwise

        Example:
            >>> router = ModelRouter()
            >>> if router.can_handle("whisper-tiny", "can_transcribe"):
            ...     print("Can transcribe!")
        """
        try:
            caps = self.get_capabilities(model_type)
            return getattr(caps, capability, False)
        except ValueError:
            return False

    def _get_generation_module(self, model_type: str):
        """Get the generation module for a model type."""
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Model type {model_type} not in registry")

        module_path = MODEL_REGISTRY[model_type]

        # Import the model module
        try:
            module = importlib.import_module(module_path)
            return module
        except ImportError as e:
            raise ImportError(f"Failed to import {module_path}: {e}") from e

    def route_text_generation(
        self,
        model_type: str,
        model: nn.Module,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        stop_strings: list[str] | None = None,
        verbose: bool = False,
    ) -> str:
        """
        Route text generation request to appropriate model.

        Args:
            model_type: Model type (e.g., "smollm2-135m")
            model: Loaded model instance
            tokenizer: Model tokenizer
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            stop_strings: Optional stop strings
            verbose: Print generation progress

        Returns:
            Generated text

        Raises:
            ValueError: If model doesn't support text generation

        Example:
            >>> response = router.route_text_generation(
            ...     model_type="smollm2-135m",
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     prompt="Write a function to",
            ...     max_tokens=100
            ... )
        """
        caps = self.get_capabilities(model_type)
        if not caps.can_complete:
            raise ValueError(f"Model {model_type} does not support text completion")

        # Get the model's generation module
        module = self._get_generation_module(model_type)

        # Call the model's generate function
        if hasattr(module, "generate"):
            return module.generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_strings=stop_strings,
                verbose=verbose,
            )
        else:
            raise AttributeError(f"Module {module} does not have generate() function")

    def route_chat(
        self,
        model_type: str,
        model: nn.Module,
        tokenizer: Any,
        messages: list[dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        verbose: bool = False,
    ) -> str:
        """
        Route chat request to appropriate model.

        Args:
            model_type: Model type
            model: Loaded model instance
            tokenizer: Model tokenizer
            messages: Chat messages [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            verbose: Print generation progress

        Returns:
            Generated response text

        Raises:
            ValueError: If model doesn't support chat

        Example:
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> response = router.route_chat(
            ...     model_type="smollm2-135m",
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     messages=messages
            ... )
        """
        caps = self.get_capabilities(model_type)
        if not caps.can_chat:
            raise ValueError(f"Model {model_type} does not support chat")

        # Get the model's generation module
        module = self._get_generation_module(model_type)

        # Call the model's chat function
        if hasattr(module, "chat"):
            return module.chat(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                verbose=verbose,
            )
        else:
            raise AttributeError(f"Module {module} does not have chat() function")

    def route_streaming_generation(
        self,
        model_type: str,
        model: nn.Module,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        stop_strings: list[str] | None = None,
    ) -> Generator[str, None, None]:
        """
        Route streaming generation request.

        Args:
            model_type: Model type
            model: Loaded model instance
            tokenizer: Model tokenizer
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            stop_strings: Optional stop strings

        Yields:
            Generated text segments

        Raises:
            ValueError: If model doesn't support streaming

        Example:
            >>> for text in router.route_streaming_generation(
            ...     model_type="smollm2-135m",
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     prompt="Hello"
            ... ):
            ...     print(text, end="", flush=True)
        """
        caps = self.get_capabilities(model_type)
        if not caps.can_stream:
            raise ValueError(f"Model {model_type} does not support streaming")

        # Get the model's generation module
        module = self._get_generation_module(model_type)

        # Call the model's stream_generate function
        if hasattr(module, "stream_generate"):
            yield from module.stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_strings=stop_strings,
            )
        else:
            raise AttributeError(f"Module {module} does not have stream_generate() function")

    def route_multimodal(
        self,
        model_type: str,
        model: nn.Module,
        processor: Any,
        prompt: str,
        image: Any | None = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """
        Route multimodal generation request (vision-language).

        Args:
            model_type: Model type
            model: Loaded model instance
            processor: Image processor
            prompt: Text prompt
            image: Image (path, PIL, or array)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold

        Returns:
            Generated text

        Raises:
            ValueError: If model doesn't support multimodal generation

        Example:
            >>> response = router.route_multimodal(
            ...     model_type="smolvlm-256m",
            ...     model=model,
            ...     processor=processor,
            ...     prompt="Describe this image:",
            ...     image="photo.jpg"
            ... )
        """
        caps = self.get_capabilities(model_type)
        if not caps.can_caption and not caps.can_chat:
            raise ValueError(f"Model {model_type} does not support multimodal generation")

        if caps.requires_image and image is None:
            raise ValueError(f"Model {model_type} requires image input")

        # Get the model's generation module
        module = self._get_generation_module(model_type)

        # Call the model's generate function
        if hasattr(module, "generate"):
            return module.generate(
                model=model,
                processor=processor,
                prompt=prompt,
                image=image,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            raise AttributeError(f"Module {module} does not have generate() function")

    def route_transcription(
        self,
        model_type: str,
        model: nn.Module,
        tokenizer: Any,
        audio: Any,
        language: str | None = None,
        task: str = "transcribe",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Route audio transcription request.

        Args:
            model_type: Model type
            model: Loaded model instance
            tokenizer: Model tokenizer
            audio: Audio (path or array)
            language: Optional language code
            task: Task ("transcribe" or "translate")
            verbose: Print progress

        Returns:
            Transcription result dict with "text" and metadata

        Raises:
            ValueError: If model doesn't support transcription

        Example:
            >>> result = router.route_transcription(
            ...     model_type="whisper-tiny",
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     audio="speech.wav",
            ...     language="en"
            ... )
            >>> print(result["text"])
        """
        caps = self.get_capabilities(model_type)
        if not caps.can_transcribe:
            raise ValueError(f"Model {model_type} does not support transcription")

        # Get the model's module
        module = self._get_generation_module(model_type)

        # Call the model's transcribe function
        if hasattr(module, "transcribe"):
            return module.transcribe(
                audio,
                model,
                tokenizer,
                language=language,
                task=task,
                verbose=verbose,
            )
        else:
            raise AttributeError(f"Module {module} does not have transcribe() function")


# ============================================================================
# Global Router Instance
# ============================================================================

_global_router: ModelRouter | None = None


def get_router() -> ModelRouter:
    """
    Get the global ModelRouter instance (singleton pattern).

    Returns:
        Global ModelRouter instance

    Example:
        >>> from smlx.models.smlx_router import get_router
        >>> router = get_router()
        >>> caps = router.get_capabilities("smollm2-135m")
    """
    global _global_router
    if _global_router is None:
        _global_router = ModelRouter()
    return _global_router


__all__ = [
    "ModelCapabilities",
    "ModelRouter",
    "CAPABILITY_MAP",
    "get_router",
]
