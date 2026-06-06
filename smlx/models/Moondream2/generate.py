#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Text generation for Moondream2.

Supports multiple task modes:
- caption: Image captioning
- query: Visual question answering
- detect: Object detection with bounding boxes
- point: Spatial localization of objects

Includes output validation to detect and prevent gibberish outputs.
"""

import logging
from typing import Optional, List, Tuple, Generator

import mlx.core as mx
from transformers import PreTrainedTokenizerBase
from PIL import Image
import numpy as np

from .model import Moondream2
from .region import parse_coordinates_from_text, parse_boxes_from_text
from .cache import make_kv_caches  # Enhanced cache with monitoring & quantization
from smlx.utils.sampling import sample_with_temperature, make_repetition_penalty
from ...utils.validation import validate_text_output

logger = logging.getLogger(__name__)


def preprocess_image(
    image: Image.Image,
    target_size: int = 378,
) -> mx.array:
    """Preprocess PIL image for model input.

    Args:
        image: PIL Image
        target_size: Target size for the image

    Returns:
        Preprocessed image tensor [1, C, H, W]
    """
    # Resize
    image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # To numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_array = (img_array - mean) / std

    # Convert to MLX and rearrange to [C, H, W]
    img_tensor = mx.array(img_array).transpose(2, 0, 1)

    # Add batch dimension [1, C, H, W]
    img_tensor = img_tensor[None, :, :, :]

    return img_tensor


def generate(
    model: Moondream2,
    tokenizer: PreTrainedTokenizerBase,
    image: Image.Image,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.5,
    top_p: float = 1.0,
    use_tiling: bool = True,
    repetition_penalty: float = 1.2,
    validate_output: bool = False,
    max_repetition_ratio: float = 0.6,
    check_gibberish: bool = True,
    retry_on_failure: bool = False,
    max_retries: int = 2,
    min_length: int = 5,
) -> str:
    """Generate text response for an image and prompt.

    Args:
        model: Moondream2 model
        tokenizer: Tokenizer
        image: PIL Image
        prompt: Text prompt/question
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling parameter
        use_tiling: Whether to use crop-based tiling
        repetition_penalty: Penalty for repeating tokens (>1 = penalize, 1.0 = disabled)
        validate_output: Enable output validation (gibberish/repetition detection)
        max_repetition_ratio: Maximum allowed repetition ratio
        check_gibberish: Check for gibberish patterns
        retry_on_failure: Retry generation if validation fails
        max_retries: Maximum number of retries
        min_length: Minimum output length

    Returns:
        Generated text response

    Example:
        >>> from smlx.models.Moondream2 import load, generate
        >>> from PIL import Image
        >>> model, tokenizer = load()
        >>> image = Image.open("photo.jpg")
        >>> response = generate(model, tokenizer, image, "Describe this image")
        >>>
        >>> # With validation
        >>> response = generate(
        ...     model, tokenizer, image,
        ...     "Describe this image",
        ...     validate_output=True,
        ...     retry_on_failure=True
        ... )
    """
    # Internal generation function for retry logic
    def _generate_internal(current_temperature: float) -> str:
        # Preprocess image
        pixel_values = preprocess_image(image, model.config.vision_config.image_size)

        # Encode image
        vision_embeddings = model.encode_image(pixel_values, use_tiling=use_tiling)
        # Evaluate to prevent graph accumulation
        mx.eval(vision_embeddings)

        # Prepend BOS embedding to vision embeddings (Phase 2 fix)
        # HF implementation caches [BOS_embedding, vision_embeddings] = 730 tokens (not 729)
        # This is critical for proper attention and generation
        bos_token_id = 0  # Will be overridden in Phase 3
        bos_embedding = model.language_model.embed_tokens(mx.array([[bos_token_id]]))  # [1, 1, hidden_size]
        vision_embeddings = mx.concatenate([bos_embedding, vision_embeddings], axis=1)  # [1, 730, hidden_size]
        mx.eval(vision_embeddings)

        # Diagnostic logging
        import os
        if os.getenv("SMLX_DEBUG"):
            print(f"[DEBUG] Vision embeddings shape: {vision_embeddings.shape}")
            print(f"[DEBUG] Vision embeddings mean: {vision_embeddings.mean().item():.4f}")
            print(f"[DEBUG] Vision embeddings std: {vision_embeddings.std().item():.4f}")
            print(f"[DEBUG] Vision embeddings min/max: {vision_embeddings.min().item():.4f} / {vision_embeddings.max().item():.4f}")

        # Tokenize prompt with HuggingFace template
        # HF Moondream2 uses: [1, 15381, 2] + question_tokens + [3]
        # Where: 1=BOS(?), 15381=<|image|>, 2=separator, 3=<|answer|> signal
        prefix_tokens = [1, 15381, 2]
        suffix_tokens = [3]  # <|answer|> token - signals model to start generating response
        question_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        full_prompt_tokens = prefix_tokens + question_tokens + suffix_tokens
        input_ids = mx.array([full_prompt_tokens])

        if os.getenv("SMLX_DEBUG"):
            print(f"[DEBUG] Input IDs shape: {input_ids.shape}")
            print(f"[DEBUG] Input IDs: {input_ids}")

        # Create KV caches
        cache = make_kv_caches(model.config.text_config)

        # CRITICAL FIX: Cache vision embeddings FIRST (matching HuggingFace flow)
        # Run vision embeddings through language model to populate KV cache
        # This caches 730 vision tokens (BOS + 729 vision) at positions 0-729
        model.cache_vision_embeddings(vision_embeddings, cache)

        if os.getenv("SMLX_DEBUG"):
            print(f"[DEBUG] Vision cached (730 tokens). Now passing text prompt...")

        # Create repetition penalty processor if enabled
        penalty_processor = None
        if repetition_penalty != 1.0:
            penalty_processor = make_repetition_penalty(repetition_penalty, context_size=20)

        # Generate - vision is now cached, pass text tokens only
        # Text tokens start at position 730 (after 730 vision tokens)
        tokens = []
        vision_token_count = vision_embeddings.shape[1]  # 730 (BOS + 729 vision)
        current_position = vision_token_count

        for _ in range(max_tokens):
            # Create position_ids for current text token(s)
            position_ids = mx.arange(current_position, current_position + input_ids.shape[1])[None, :]

            # Forward pass with text only (vision already cached at positions 0-729)
            logits, _ = model(
                input_ids,
                vision_embeddings=None,  # Already cached!
                cache=cache,
                position_ids=position_ids,
            )

            current_position += input_ids.shape[1]

            # Get next token logits
            next_token_logits = logits[:, -1, :]
            # Evaluate to prevent computation graph accumulation
            mx.eval(next_token_logits)

            # Apply repetition penalty if enabled
            if penalty_processor is not None:
                next_token_logits = penalty_processor(mx.array(tokens), next_token_logits)

            # Sample next token
            if current_temperature == 0:
                next_token = mx.argmax(next_token_logits, axis=-1)
            else:
                next_token = sample_with_temperature(
                    next_token_logits, current_temperature, top_p
                )

            next_token = next_token.item()

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break

            tokens.append(next_token)

            # Update input for next iteration
            input_ids = mx.array([[next_token]])
            # Evaluate input array
            mx.eval(input_ids)

        # Decode tokens
        response = tokenizer.decode(tokens, skip_special_tokens=True)

        return response

    # Retry loop with validation
    current_temperature = temperature
    for attempt in range(max(1, max_retries + 1 if retry_on_failure else 1)):
        generated_text = _generate_internal(current_temperature)

        # Validate output if enabled
        if validate_output:
            is_valid, reason = validate_text_output(
                generated_text,
                min_length=min_length,
                max_repetition_ratio=max_repetition_ratio,
                check_gibberish=check_gibberish,
            )

            if not is_valid:
                logger.warning(f"Moondream2 output validation failed: {reason}")

                if retry_on_failure and attempt < max_retries:
                    logger.info(f"Retrying generation (attempt {attempt + 2}/{max_retries + 1})...")
                    # Adjust temperature for retry (decrease for more deterministic output)
                    current_temperature = max(0.1, current_temperature * 0.8)
                    continue

        # Success or max retries reached
        break

    return generated_text


def stream_generate(
    model: Moondream2,
    tokenizer: PreTrainedTokenizerBase,
    image: Image.Image,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.5,
    top_p: float = 1.0,
    use_tiling: bool = True,
    repetition_penalty: float = 1.2,
) -> Generator[str, None, None]:
    """Stream generation token-by-token.

    Args:
        model: Moondream2 model
        tokenizer: Tokenizer
        image: PIL Image
        prompt: Text prompt/question
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        use_tiling: Whether to use crop-based tiling
        repetition_penalty: Penalty for repeating tokens (>1 = penalize, 1.0 = disabled)

    Yields:
        Generated text tokens

    Example:
        >>> for token in stream_generate(model, tokenizer, image, "What do you see?"):
        ...     print(token, end="", flush=True)
    """
    # Preprocess image
    pixel_values = preprocess_image(image, model.config.vision_config.image_size)

    # Encode image
    vision_embeddings = model.encode_image(pixel_values, use_tiling=use_tiling)
    # Evaluate to prevent graph accumulation
    mx.eval(vision_embeddings)

    # Prepend BOS embedding to vision embeddings (Phase 2 fix)
    bos_token_id = 0
    bos_embedding = model.language_model.embed_tokens(mx.array([[bos_token_id]]))
    vision_embeddings = mx.concatenate([bos_embedding, vision_embeddings], axis=1)
    mx.eval(vision_embeddings)

    # Tokenize prompt with HuggingFace template
    # HF Moondream2 uses: [1, 15381, 2] + question_tokens + [3]
    prefix_tokens = [1, 15381, 2]
    suffix_tokens = [3]  # <|answer|> token
    question_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    full_prompt_tokens = prefix_tokens + question_tokens + suffix_tokens
    input_ids = mx.array([full_prompt_tokens])

    # Create KV caches
    cache = make_kv_caches(model.config.text_config)

    # Cache vision embeddings FIRST (same as generate() function)
    model.cache_vision_embeddings(vision_embeddings, cache)

    # Create repetition penalty processor if enabled
    penalty_processor = None
    if repetition_penalty != 1.0:
        penalty_processor = make_repetition_penalty(repetition_penalty, context_size=20)

    # Generate - vision cached, pass text only
    # Text tokens start at position 730 (after 730 vision tokens)
    generated_tokens = []  # Track generated tokens for repetition penalty
    vision_token_count = vision_embeddings.shape[1]  # 730 (BOS + 729 vision)
    current_position = vision_token_count

    for _ in range(max_tokens):
        # Create position_ids for current text token(s)
        position_ids = mx.arange(current_position, current_position + input_ids.shape[1])[None, :]

        # Forward pass with text only (vision already cached)
        logits, _ = model(
            input_ids,
            vision_embeddings=None,
            cache=cache,
            position_ids=position_ids,
        )

        current_position += input_ids.shape[1]

        # Get next token
        next_token_logits = logits[:, -1, :]
        # Evaluate to prevent computation graph accumulation
        mx.eval(next_token_logits)

        # Apply repetition penalty if enabled
        if penalty_processor is not None:
            next_token_logits = penalty_processor(mx.array(generated_tokens), next_token_logits)

        if temperature == 0:
            next_token = mx.argmax(next_token_logits, axis=-1)
        else:
            next_token = sample_with_temperature(
                next_token_logits, temperature, top_p
            )

        next_token = next_token.item()

        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            break

        # Track generated token for mask calculation
        generated_tokens.append(next_token)

        # Decode and yield
        token_text = tokenizer.decode([next_token], skip_special_tokens=True)
        yield token_text

        # Update input
        input_ids = mx.array([[next_token]])
        # Evaluate input array
        mx.eval(input_ids)


def caption(
    model: Moondream2,
    tokenizer: PreTrainedTokenizerBase,
    image: Image.Image,
    length: str = "normal",
    **kwargs,
) -> str:
    """Generate image caption.

    Args:
        model: Moondream2 model
        tokenizer: Tokenizer
        image: PIL Image
        length: Caption length ("short", "normal", "long")
        **kwargs: Additional arguments for generate()

    Returns:
        Image caption

    Example:
        >>> caption_text = caption(model, tokenizer, image, length="long")
    """
    length_prompts = {
        "short": "Briefly describe this image.",
        "normal": "Describe this image in detail.",
        "long": "Provide a detailed and comprehensive description of this image.",
    }

    prompt = length_prompts.get(length, length_prompts["normal"])
    return generate(model, tokenizer, image, prompt, **kwargs)


def query(
    model: Moondream2,
    tokenizer: PreTrainedTokenizerBase,
    image: Image.Image,
    question: str,
    **kwargs,
) -> str:
    """Visual question answering.

    Args:
        model: Moondream2 model
        tokenizer: Tokenizer
        image: PIL Image
        question: Question about the image
        **kwargs: Additional arguments for generate()

    Returns:
        Answer to the question

    Example:
        >>> answer = query(model, tokenizer, image, "How many people are in this image?")
    """
    return generate(model, tokenizer, image, question, **kwargs)


def detect(
    model: Moondream2,
    tokenizer: PreTrainedTokenizerBase,
    image: Image.Image,
    query: str = None,
    object_name: str = None,
    confidence_threshold: float = 0.5,
    use_detection_head: bool = True,
    **kwargs,
) -> List[Tuple[int, int, int, int, Optional[float]]]:
    """Detect objects in image.

    Args:
        model: Moondream2 model
        tokenizer: Tokenizer
        image: PIL Image
        query: Detection query (alias for object_name)
        object_name: Name of object to detect (alias for query)
        confidence_threshold: Minimum confidence score (only applied when real
                             confidence scores are available, i.e. the detection head)
        use_detection_head: If True, use the model's detection head, which yields real
                           per-box confidence scores. If False, use text-based
                           generation, which yields boxes only (confidence is None —
                           the model does not emit scores in that mode).
        **kwargs: Additional arguments for generate()

    Returns:
        List of (x1, y1, x2, y2, confidence) tuples. ``confidence`` is a float in
        [0, 1] for detection-head results, or ``None`` for text-based results where
        no real score exists.

    Example:
        >>> detections = detect(model, tokenizer, image, query="person")
        >>> for x1, y1, x2, y2, conf in detections:
        ...     conf_str = f"{conf:.2f}" if conf is not None else "n/a"
        ...     print(f"Found at ({x1}, {y1}, {x2}, {y2}) with confidence {conf_str}")
    """
    # Accept either query or object_name parameter
    detection_query = query or object_name
    if detection_query is None:
        raise ValueError("Either 'query' or 'object_name' parameter must be provided")

    if use_detection_head:
        # Use model's detection head for real confidence scores
        # 1. Preprocess image
        pixel_values = preprocess_image(image, model.config.vision_config.image_size)

        # 2. Encode image to vision embeddings
        vision_embeddings = model.encode_image(pixel_values, use_tiling=True)
        mx.eval(vision_embeddings)

        # 3. Create detection prompt
        prompt = f"<|grounding|>Detect all {detection_query} in this image and provide bounding boxes."

        # 4. Tokenize prompt
        encoded = tokenizer(
            prompt,
            return_tensors="np",
            add_special_tokens=True,
        )
        input_ids = mx.array(encoded["input_ids"])

        # 5. Create KV cache
        cache = make_kv_caches(model.config.text_config)

        # 6. Run detection head to get boxes and confidence scores
        boxes, confidences = model.detect_objects(
            input_ids,
            vision_embeddings,
            cache=cache,
        )

        # 7. Convert to pixel coordinates and filter by confidence
        width, height = image.size
        detections = []

        # boxes: [B, max_detections, 4] - normalized coordinates [0, 1]
        # confidences: [B, max_detections] - confidence scores [0, 1]
        num_detections = boxes.shape[1]

        for i in range(num_detections):
            box = boxes[0, i, :]  # [4]
            conf = float(confidences[0, i])

            # Filter by confidence threshold
            if conf >= confidence_threshold:
                # Denormalize to pixel coordinates
                x1 = int(float(box[0]) * width)
                y1 = int(float(box[1]) * height)
                x2 = int(float(box[2]) * width)
                y2 = int(float(box[3]) * height)

                # Ensure valid box (x2 > x1, y2 > y1)
                if x2 > x1 and y2 > y1:
                    detections.append((x1, y1, x2, y2, conf))

        return detections

    else:
        # Fallback: text-based generation. The model emits bounding boxes only,
        # with NO per-box confidence scores, so confidence is reported as None
        # (unknown) rather than a fabricated value. The confidence_threshold is
        # not applied here because there is no real score to threshold on.
        prompt = f"<|grounding|>Detect all {detection_query} in this image and provide bounding boxes."

        response = generate(model, tokenizer, image, prompt, **kwargs)

        # Parse bounding boxes from response
        boxes = parse_boxes_from_text(response, image.size)

        if boxes is None:
            return []

        detections = [(x1, y1, x2, y2, None) for x1, y1, x2, y2 in boxes]

        return detections


def point(
    model: Moondream2,
    tokenizer: PreTrainedTokenizerBase,
    image: Image.Image,
    object_name: str = None,
    object_query: str = None,
    **kwargs,
) -> Optional[Tuple[int, int]]:
    """Point to object location in image.

    Args:
        model: Moondream2 model
        tokenizer: Tokenizer
        image: PIL Image
        object_name: Name of object to point to (alias for object_query)
        object_query: Description of object to point to (alias for object_name)
        **kwargs: Additional arguments for generate()

    Returns:
        (x, y) pixel coordinates, or None if not found

    Example:
        >>> location = point(model, tokenizer, image, object_name="the red car")
        >>> if location:
        ...     print(f"Object at pixel {location}")
    """
    # Accept either object_name or object_query parameter
    target = object_name or object_query
    if target is None:
        raise ValueError("Either 'object_name' or 'object_query' parameter must be provided")

    prompt = f"<|coordinate|>Point to {target} in this image."

    response = generate(model, tokenizer, image, prompt, **kwargs)

    # Parse coordinates from response
    coords = parse_coordinates_from_text(response, image.size)

    if coords and len(coords) > 0:
        return coords[0]  # Return first coordinate

    return None


__all__ = [
    "generate",
    "stream_generate",
    "caption",
    "query",
    "detect",
    "point",
    "preprocess_image",
]
