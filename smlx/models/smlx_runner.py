#!/usr/bin/env python3
# Copyright � 2025 SMLX Project

"""
Model Inference Runner for SMLX.

This module provides unified inference execution with preprocessing, validation,
and abstraction over batch vs streaming operations.

Architecture:
    InferenceConfig - Configuration for inference runs
    ModelRunner - Unified executor for model inference
    Preprocessing utilities - Input preparation and validation

Example:
    >>> from smlx.models.smlx_runner import ModelRunner, InferenceConfig
    >>>
    >>> # Create runner
    >>> runner = ModelRunner()
    >>>
    >>> # Run inference with preprocessing
    >>> config = InferenceConfig(max_tokens=100, temperature=0.7)
    >>> result = runner.run(
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     model_type="smollm2-135m",
    ...     prompt="Hello world",
    ...     config=config
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from .smlx_router import get_router

# ============================================================================
# Inference Configuration
# ============================================================================


@dataclass
class InferenceConfig:
    """
    Configuration for model inference.

    Attributes:
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        stop_strings: Optional stop sequences
        stream: Whether to use streaming generation
        batch_size: Batch size for batch inference
        repetition_penalty: Penalty for token repetition
        repetition_context_size: Context size for repetition penalty
        verbose: Print generation progress
        use_cache: Whether to use KV cache
        cache_config: Configuration for KV cache
    """

    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    stop_strings: list[str] | None = None
    stream: bool = False
    batch_size: int = 1
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20
    verbose: bool = False
    use_cache: bool = True
    cache_config: dict[str, Any] = field(default_factory=dict)

    def validate(self):
        """Validate configuration parameters."""
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")

        if not 0 <= self.top_p <= 1:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")

        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


# ============================================================================
# Input Preprocessing
# ============================================================================


def preprocess_text_input(
    prompt: str | list[str],
    tokenizer: Any,
    max_length: int | None = None,
) -> tuple[list[str], list[mx.array]]:
    """
    Preprocess text input for generation.

    Args:
        prompt: Single prompt or list of prompts
        tokenizer: Model tokenizer
        max_length: Optional maximum input length

    Returns:
        Tuple of (prompts_list, tokenized_inputs)

    Example:
        >>> prompts, inputs = preprocess_text_input("Hello", tokenizer)
        >>> prompts
        ['Hello']
    """
    # Normalize to list
    prompts = [prompt] if isinstance(prompt, str) else prompt

    # Validate prompts
    for i, p in enumerate(prompts):
        if not isinstance(p, str):
            raise TypeError(f"Prompt {i} must be string, got {type(p)}")
        if not p.strip():
            raise ValueError(f"Prompt {i} is empty")

    # Tokenize
    tokenized = []
    for p in prompts:
        tokens = tokenizer.encode(p)

        # Truncate if needed
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized.append(mx.array(tokens))

    return prompts, tokenized


def preprocess_chat_input(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int | None = None,
) -> tuple[str, mx.array]:
    """
    Preprocess chat messages into prompt format.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: Model tokenizer
        max_length: Optional maximum input length

    Returns:
        Tuple of (formatted_prompt, tokenized_input)

    Example:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> prompt, tokens = preprocess_chat_input(messages, tokenizer)
    """
    # Validate messages
    if not messages:
        raise ValueError("Messages list is empty")

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise TypeError(f"Message {i} must be dict, got {type(msg)}")
        if "role" not in msg or "content" not in msg:
            raise ValueError(f"Message {i} missing 'role' or 'content'")

    # Format as chat prompt (simple format for now)
    prompt_parts = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        prompt_parts.append(f"{role}: {content}")

    # Add assistant prefix
    prompt_parts.append("Assistant:")
    prompt = "\n\n".join(prompt_parts)

    # Tokenize
    tokens = tokenizer.encode(prompt)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]

    return prompt, mx.array(tokens)


def preprocess_image_input(
    image: str | Path | Any,
    processor: Any,
) -> Any:
    """
    Preprocess image input for vision models.

    Args:
        image: Image path, PIL image, or array
        processor: Image processor

    Returns:
        Processed image tensor

    Example:
        >>> image_tensor = preprocess_image_input("cat.jpg", processor)
    """
    from PIL import Image

    # Handle different input types
    if isinstance(image, (str, Path)):
        # Load from path
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

    # Process image (processor-specific)
    if hasattr(processor, "process_image"):
        return processor.process_image(image)
    elif callable(processor):
        return processor(image)
    else:
        raise AttributeError("Processor doesn't have process_image or __call__")


def preprocess_audio_input(
    audio: str | Path | Any,
    processor: Any,
    sample_rate: int = 16000,
) -> Any:
    """
    Preprocess audio input for audio models.

    Args:
        audio: Audio path or array
        processor: Audio processor
        sample_rate: Expected sample rate

    Returns:
        Processed audio tensor

    Example:
        >>> audio_tensor = preprocess_audio_input("speech.wav", processor)
    """
    import numpy as np

    # Handle different input types
    if isinstance(audio, (str, Path)):
        # Load from path
        audio_path = Path(audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # Load audio (use librosa or similar)
        try:
            import soundfile as sf

            audio_data, sr = sf.read(str(audio_path))

            # Resample if needed
            if sr != sample_rate:
                import librosa

                audio_data = librosa.resample(
                    audio_data, orig_sr=sr, target_sr=sample_rate
                )
        except ImportError as e:
            raise ImportError("soundfile and librosa required for audio loading") from e

        audio = audio_data

    # Convert to appropriate format
    if isinstance(audio, np.ndarray):
        audio = mx.array(audio)

    # Process audio (processor-specific)
    if hasattr(processor, "process_audio"):
        return processor.process_audio(audio)
    elif callable(processor):
        return processor(audio)
    else:
        raise AttributeError("Processor doesn't have process_audio or __call__")


# ============================================================================
# Model Runner
# ============================================================================


class ModelRunner:
    """
    Unified runner for model inference with preprocessing.

    The runner handles:
    - Input validation and preprocessing
    - Routing to appropriate generation functions
    - Batch vs streaming abstraction
    - Cache management

    Example:
        >>> runner = ModelRunner()
        >>> config = InferenceConfig(max_tokens=50)
        >>>
        >>> # Text generation
        >>> result = runner.run(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     model_type="smollm2-135m",
        ...     prompt="Hello",
        ...     config=config
        ... )
        >>>
        >>> # Chat
        >>> result = runner.run_chat(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     model_type="smollm2-135m",
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     config=config
        ... )
    """

    def __init__(self):
        """Initialize the model runner."""
        self.router = get_router()

    def run(
        self,
        model: Any,
        tokenizer: Any,
        model_type: str,
        prompt: str | list[str],
        config: InferenceConfig | None = None,
        processor: Any | None = None,
        image: Any | None = None,
        audio: Any | None = None,
    ) -> str | list[str]:
        """
        Run inference with automatic preprocessing.

        Args:
            model: Loaded model
            tokenizer: Model tokenizer
            model_type: Model type identifier
            prompt: Text prompt(s)
            config: Optional inference configuration
            processor: Optional image/audio processor
            image: Optional image input (for VLMs)
            audio: Optional audio input (for audio models)

        Returns:
            Generated text or list of texts

        Example:
            >>> result = runner.run(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     model_type="smollm2-135m",
            ...     prompt="Write a function",
            ... )
        """
        # Use default config if not provided
        if config is None:
            config = InferenceConfig()

        # Validate config
        config.validate()

        # Get model capabilities
        caps = self.router.get_capabilities(model_type)

        # Handle multimodal input
        if image is not None:
            if not caps.requires_image and not caps.can_caption:
                raise ValueError(f"Model {model_type} doesn't support image input")

            if processor is None:
                raise ValueError("Image processor required for multimodal generation")

            # Preprocess image
            processed_image = preprocess_image_input(image, processor)

            # Use multimodal routing
            return self.router.route_multimodal(
                model_type=model_type,
                model=model,
                processor=processor,
                prompt=prompt if isinstance(prompt, str) else prompt[0],
                image=processed_image,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )

        if audio is not None:
            if not caps.requires_audio:
                raise ValueError(f"Model {model_type} doesn't support audio input")

            # Handle audio models (transcription, etc.)
            if caps.can_transcribe:
                # Preprocess audio
                processed_audio = preprocess_audio_input(audio, processor or tokenizer)

                return self.router.route_transcription(
                    model_type=model_type,
                    model=model,
                    tokenizer=tokenizer,
                    audio=processed_audio,
                    verbose=config.verbose,
                )

        # Text-only generation
        if config.stream:
            # Return generator for streaming
            return self.router.route_streaming_generation(
                model_type=model_type,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt if isinstance(prompt, str) else prompt[0],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop_strings=config.stop_strings,
            )

        # Standard text generation
        if isinstance(prompt, str):
            return self.router.route_text_generation(
                model_type=model_type,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop_strings=config.stop_strings,
                verbose=config.verbose,
            )
        else:
            # Batch generation
            return [
                self.router.route_text_generation(
                    model_type=model_type,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=p,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    stop_strings=config.stop_strings,
                    verbose=config.verbose,
                )
                for p in prompt
            ]

    def run_chat(
        self,
        model: Any,
        tokenizer: Any,
        model_type: str,
        messages: list[dict[str, str]],
        config: InferenceConfig | None = None,
    ) -> str:
        """
        Run chat inference with automatic preprocessing.

        Args:
            model: Loaded model
            tokenizer: Model tokenizer
            model_type: Model type identifier
            messages: Chat messages
            config: Optional inference configuration

        Returns:
            Generated response

        Example:
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> result = runner.run_chat(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     model_type="smollm2-135m",
            ...     messages=messages,
            ... )
        """
        # Use default config if not provided
        if config is None:
            config = InferenceConfig()

        # Validate config
        config.validate()

        # Check capability
        caps = self.router.get_capabilities(model_type)
        if not caps.can_chat:
            raise ValueError(f"Model {model_type} doesn't support chat")

        # Preprocess messages
        _, _ = preprocess_chat_input(messages, tokenizer)

        # Run chat generation
        return self.router.route_chat(
            model_type=model_type,
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            verbose=config.verbose,
        )

    def run_batch(
        self,
        model: Any,
        tokenizer: Any,
        model_type: str,
        prompts: list[str],
        config: InferenceConfig | None = None,
    ) -> list[str]:
        """
        Run batch inference.

        Args:
            model: Loaded model
            tokenizer: Model tokenizer
            model_type: Model type identifier
            prompts: List of prompts
            config: Optional inference configuration

        Returns:
            List of generated texts

        Example:
            >>> prompts = ["Hello", "Goodbye", "How are you?"]
            >>> results = runner.run_batch(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     model_type="smollm2-135m",
            ...     prompts=prompts,
            ... )
        """
        # Use default config if not provided
        if config is None:
            config = InferenceConfig()

        # Validate config
        config.validate()

        # Preprocess all prompts
        prompts_list, _ = preprocess_text_input(prompts, tokenizer)

        # Run generation for each prompt
        results = []
        for prompt in prompts_list:
            result = self.router.route_text_generation(
                model_type=model_type,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop_strings=config.stop_strings,
                verbose=config.verbose,
            )
            results.append(result)

        return results


__all__ = [
    "InferenceConfig",
    "ModelRunner",
    "preprocess_text_input",
    "preprocess_chat_input",
    "preprocess_image_input",
    "preprocess_audio_input",
]
