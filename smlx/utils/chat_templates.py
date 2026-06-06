#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Chat Template Utilities for Vision-Language Models.

Simplified implementation inspired by mlx-vlm for formatting prompts with images.
"""

from typing import Any, Dict, List, Optional, Union


class MessageFormat:
    """Message format types for different VLM models."""

    # Image tokens inserted in text
    IMAGE_TOKEN = "image_token"  # <image> in text
    IMAGE_TOKEN_NEWLINE = "image_token_newline"  # <image>\n in text

    # Structured content with images
    LIST_WITH_IMAGE = "list_with_image"  # [image, text]
    LIST_WITH_IMAGE_FIRST = "list_with_image_first"  # [text, image] or [image, text]

    # Simple prompt only
    PROMPT_ONLY = "prompt_only"  # Just text, no special formatting


# Model-specific message formats
MODEL_FORMATS = {
    "smolvlm": MessageFormat.LIST_WITH_IMAGE_FIRST,
    "nanovlm": MessageFormat.IMAGE_TOKEN,
    "llava": MessageFormat.LIST_WITH_IMAGE,
    "moondream2": MessageFormat.IMAGE_TOKEN,
    "tinyllava": MessageFormat.IMAGE_TOKEN,
    "paligemma": MessageFormat.IMAGE_TOKEN,
}


def apply_chat_template(
    model_type: str,
    prompt: Union[str, Dict[str, Any], List[Any]],
    num_images: int = 0,
    add_generation_prompt: bool = True,
) -> str:
    """
    Apply chat template formatting for VLM models.

    Args:
        model_type: Model identifier (e.g., "smolvlm", "nanovlm")
        prompt: User prompt (string or conversation history)
        num_images: Number of images in the input
        add_generation_prompt: Whether to add generation prompt marker

    Returns:
        Formatted prompt string

    Examples:
        >>> # Simple prompt with image
        >>> apply_chat_template("nanovlm", "Describe the image", num_images=1)
        'Describe the <image>'

        >>> # Multi-turn conversation
        >>> conversation = [
        ...     {"role": "user", "content": "What's in this image?"},
        ...     {"role": "assistant", "content": "A cat."},
        ...     {"role": "user", "content": "What color is it?"}
        ... ]
        >>> apply_chat_template("smolvlm", conversation)
        '<s>User: What's in this image?\\nAssistant: A cat.\\nUser: What color is it?\\nAssistant:'
    """
    message_format = MODEL_FORMATS.get(model_type.lower(), MessageFormat.IMAGE_TOKEN)

    # Handle different prompt types
    if isinstance(prompt, str):
        # Single string prompt
        return _format_single_prompt(prompt, num_images, message_format)

    elif isinstance(prompt, list):
        # Conversation history
        return _format_conversation(
            prompt, num_images, message_format, add_generation_prompt
        )

    elif isinstance(prompt, dict):
        # Single message dict
        content = prompt.get("content", "")
        return _format_single_prompt(content, num_images, message_format)

    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")


def _format_single_prompt(
    prompt: str, num_images: int, message_format: str
) -> str:
    """Format a single prompt string."""

    if message_format == MessageFormat.IMAGE_TOKEN:
        # Add <image> token if not present and image is provided
        if num_images > 0 and "<image>" not in prompt:
            # Insert image token at natural position
            if prompt.strip().startswith(("Describe", "What", "Tell", "Explain")):
                # Question format: "What is in <image>?"
                prompt = prompt.replace("image", "<image>", 1)
                if "<image>" not in prompt:
                    prompt = f"<image> {prompt}"
            else:
                # Default: prepend image token
                prompt = f"<image> {prompt}"

    elif message_format == MessageFormat.IMAGE_TOKEN_NEWLINE:
        # Add <image>\n format
        if num_images > 0 and "<image>" not in prompt:
            prompt = f"<image>\n{prompt}"

    elif message_format == MessageFormat.LIST_WITH_IMAGE_FIRST:
        # SmolVLM style: ensure proper format
        # This will be handled by the processor's chat template
        pass

    return prompt


def _format_conversation(
    messages: List[Dict[str, Any]],
    num_images: int,
    message_format: str,
    add_generation_prompt: bool,
) -> str:
    """Format a multi-turn conversation."""

    formatted_lines = []
    image_added = False

    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Add image token to first user message if needed
        if (
            not image_added
            and num_images > 0
            and role == "user"
            and "<image>" not in content
        ):
            if message_format == MessageFormat.IMAGE_TOKEN:
                content = f"<image> {content}"
            elif message_format == MessageFormat.IMAGE_TOKEN_NEWLINE:
                content = f"<image>\n{content}"
            image_added = True

        # Format role prefix
        if role == "user":
            formatted_lines.append(f"User: {content}")
        elif role == "assistant":
            formatted_lines.append(f"Assistant: {content}")
        elif role == "system":
            formatted_lines.append(f"System: {content}")

    # Add generation prompt if requested
    if add_generation_prompt:
        formatted_lines.append("Assistant:")

    return "\n".join(formatted_lines)


def get_image_token_for_model(model_type: str) -> str:
    """
    Get the image token string for a specific model.

    Args:
        model_type: Model identifier

    Returns:
        Image token string (e.g., "<image>", "<img>")
    """
    # Most models use <image>
    return "<image>"


def validate_image_count(model_type: str, num_images: int) -> None:
    """
    Validate that the model supports the number of images.

    Args:
        model_type: Model identifier
        num_images: Number of images

    Raises:
        ValueError: If model doesn't support multiple images
    """
    # Models that only support single image
    single_image_only = {"paligemma", "moondream2"}

    if model_type.lower() in single_image_only and num_images > 1:
        raise ValueError(
            f"Model {model_type} only supports single image input, got {num_images}"
        )


__all__ = [
    "apply_chat_template",
    "get_image_token_for_model",
    "validate_image_count",
    "MessageFormat",
    "MODEL_FORMATS",
]
