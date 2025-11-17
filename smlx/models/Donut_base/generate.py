#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Document parsing and generation for Donut.

Provides high-level functions for document understanding tasks.
"""

import json
import re
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
from PIL import Image

from .model import DonutModel
from .processor import DonutProcessor


# =============================================================================
# Task-Specific Prompt Templates (from Donut paper)
# =============================================================================

TASK_PROMPTS = {
    # Document parsing (generic)
    "document": "<s_document>",
    # Document VQA (question answering)
    "docvqa": "<s_docvqa><s_question>{question}</s_question><s_answer>",
    # Receipt/invoice parsing (CORD dataset)
    "cord": "<s_cord-v2>",
    "receipt": "<s_cord-v2>",
    "invoice": "<s_cord-v2>",
    # Document classification (RVL-CDIP dataset)
    "rvlcdip": "<s_rvlcdip>",
    "classification": "<s_rvlcdip>",
    # OCR (text extraction)
    "ocr": "<s_cord-v2>",
    "text": "<s_cord-v2>",
}

TASK_END_TOKENS = {
    "document": "</s_document>",
    "docvqa": "</s_answer>",
    "cord": "</s>",
    "receipt": "</s>",
    "invoice": "</s>",
    "rvlcdip": "</s>",
    "classification": "</s>",
    "ocr": "</s>",
    "text": "</s>",
}


# =============================================================================
# Core Generation Functions
# =============================================================================


def _generate_step(
    model: DonutModel,
    encoder_hidden_states: mx.array,
    decoder_input_ids: mx.array,
    cache: Optional[List] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id: int = 2,  # BART default EOS
) -> Generator[Tuple[int, mx.array, List], None, None]:
    """
    Core autoregressive generation step yielding tokens one by one.

    Args:
        model: Donut model
        encoder_hidden_states: Vision encoder output (B, seq_len, hidden_size)
        decoder_input_ids: Initial decoder input tokens (B, seq_len)
        cache: Optional KV cache from previous generation
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling threshold
        eos_token_id: End-of-sequence token ID

    Yields:
        Tuple of (token_id, logprobs, updated_cache)
    """
    current_tokens = decoder_input_ids

    for step in range(max_tokens):
        # Forward pass through decoder
        logits, cache = model.decode(
            encoder_hidden_states=encoder_hidden_states,
            decoder_input_ids=current_tokens if cache is None else current_tokens[:, -1:],
            cache=cache,
        )

        # Get logits for last position
        next_token_logits = logits[:, -1, :]  # (B, vocab_size)

        # Compute log probabilities
        logprobs = next_token_logits - mx.logsumexp(next_token_logits, axis=-1, keepdims=True)

        # Sample next token
        if temperature == 0.0:
            # Greedy sampling
            next_token = mx.argmax(next_token_logits, axis=-1)
        else:
            # Temperature sampling
            scaled_logits = next_token_logits / temperature

            # Nucleus (top-p) sampling
            if top_p < 1.0:
                sorted_logits = mx.sort(scaled_logits, axis=-1)[:, ::-1]
                sorted_probs = mx.softmax(sorted_logits, axis=-1)
                cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

                # Find cutoff index
                mask = cumsum_probs <= top_p
                # Ensure at least one token is kept
                mask = mx.concatenate(
                    [mx.ones((mask.shape[0], 1), dtype=mx.bool_), mask[:, :-1]], axis=1
                )

                # Apply mask by setting excluded tokens to -inf
                sorted_indices = mx.argsort(scaled_logits, axis=-1)[:, ::-1]
                for b in range(mask.shape[0]):
                    excluded = sorted_indices[b][~mask[b]]
                    scaled_logits[b, excluded] = float("-inf")

            # Sample from distribution
            probs = mx.softmax(scaled_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs), axis=-1)

        # Ensure we have the right shape
        next_token = next_token.reshape(-1, 1)

        yield next_token.item(), logprobs, cache

        # Check for EOS
        if next_token.item() == eos_token_id:
            break

        # Append to current tokens for next iteration
        current_tokens = mx.concatenate([current_tokens, next_token], axis=1)


def generate_text(
    model: DonutModel,
    processor: DonutProcessor,
    pixel_values: mx.array,
    prompt_ids: mx.array,
    max_length: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id: int = 2,
) -> Tuple[List[int], str]:
    """
    Generate text from image with given prompt.

    Args:
        model: Donut model
        processor: Donut processor (contains tokenizer)
        pixel_values: Preprocessed image tensor (B, H, W, C)
        prompt_ids: Prompt token IDs (B, seq_len)
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        eos_token_id: End-of-sequence token ID

    Returns:
        Tuple of (generated_token_ids, decoded_text)
    """
    # Encode image
    encoder_hidden_states = model.encode_image(pixel_values)

    # Generate tokens
    generated_ids = []
    cache = None

    for token_id, logprobs, cache in _generate_step(
        model=model,
        encoder_hidden_states=encoder_hidden_states,
        decoder_input_ids=prompt_ids,
        cache=cache,
        max_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    ):
        generated_ids.append(token_id)

    # Decode text
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)

    return generated_ids, text


# =============================================================================
# Output Parsing Utilities
# =============================================================================


def parse_json_output(text: str) -> Dict:
    """
    Parse JSON from model output, handling common formatting issues.

    Args:
        text: Generated text potentially containing JSON

    Returns:
        Parsed JSON dictionary, or error dict if parsing fails
    """
    # Remove task tokens
    for task_start in TASK_PROMPTS.values():
        text = text.replace(task_start, "")
    for task_end in TASK_END_TOKENS.values():
        text = text.replace(task_end, "")

    # Clean up text
    text = text.strip()

    # Try to find JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {e}", "raw_text": text}


def parse_answer(text: str, question: str = None) -> str:
    """
    Extract answer from DocVQA output.

    Args:
        text: Generated text
        question: Optional question for context

    Returns:
        Extracted answer string
    """
    # Remove task tokens
    text = text.replace("<s_docvqa>", "")
    text = text.replace("</s_docvqa>", "")

    if question:
        text = text.replace(f"<s_question>{question}</s_question>", "")

    # Extract answer between tags
    answer_match = re.search(r"<s_answer>(.*?)</s_answer>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # Fallback: return cleaned text
    text = text.replace("<s_question>", "")
    text = text.replace("</s_question>", "")
    text = text.replace("<s_answer>", "")
    text = text.replace("</s_answer>", "")

    return text.strip()


def parse_classification(text: str, classes: List[str] = None) -> str:
    """
    Extract classification result.

    Args:
        text: Generated text
        classes: Optional list of valid classes

    Returns:
        Predicted class
    """
    # Remove task tokens
    text = text.replace("<s_rvlcdip>", "")
    text = text.replace("</s_rvlcdip>", "")
    text = text.replace("</s>", "")
    text = text.strip()

    # If classes provided, find best match
    if classes:
        text_lower = text.lower()
        for cls in classes:
            if cls.lower() in text_lower:
                return cls

    return text


# =============================================================================
# High-Level Task Functions
# =============================================================================


def parse_document(
    model: DonutModel,
    processor: DonutProcessor,
    image: Union[str, Image.Image],
    task: str = "document",
    max_length: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Union[str, Dict]:
    """
    Parse document image and extract structured information.

    Args:
        model: Donut model
        processor: Donut processor
        image: Document image
        task: Task type ("document", "receipt", "invoice", "cord", etc.)
        max_length: Maximum output length
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling threshold

    Returns:
        Parsed document output (JSON dict for structured tasks, string otherwise)

    Example:
        >>> model, processor = load("naver-clova-ix/donut-base")
        >>> result = parse_document(model, processor, "invoice.jpg", task="invoice")
        >>> print(result)
    """
    # Process image
    inputs = processor(image=image)
    pixel_values = inputs["pixel_values"]

    # Get task-specific prompt
    task_key = task.lower()
    if task_key not in TASK_PROMPTS:
        raise ValueError(
            f"Unknown task '{task}'. Available tasks: {list(TASK_PROMPTS.keys())}"
        )

    prompt_template = TASK_PROMPTS[task_key]
    prompt_text = prompt_template.format(question="") if "{question}" in prompt_template else prompt_template

    # Encode prompt
    prompt_ids = processor.tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_ids = mx.array(prompt_ids).reshape(1, -1)  # (1, seq_len)

    # Get EOS token ID
    eos_token_id = processor.tokenizer.eos_token_id or 2

    # Generate
    generated_ids, text = generate_text(
        model=model,
        processor=processor,
        pixel_values=pixel_values,
        prompt_ids=prompt_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )

    # Parse output based on task
    if task_key in ["cord", "receipt", "invoice", "document"]:
        # Structured output - parse as JSON
        return parse_json_output(text)
    else:
        # Text output
        return text.strip()


def extract_text(
    model: DonutModel,
    processor: DonutProcessor,
    image: Union[str, Image.Image],
    max_length: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """
    Extract all text from document image (OCR task).

    Args:
        model: Donut model
        processor: Donut processor
        image: Document image
        max_length: Maximum output length
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling threshold

    Returns:
        Extracted text

    Example:
        >>> model, processor = load("naver-clova-ix/donut-base")
        >>> text = extract_text(model, processor, "document.jpg")
        >>> print(text)
    """
    result = parse_document(
        model, processor, image, task="text", max_length=max_length, temperature=temperature, top_p=top_p
    )
    # If result is dict (JSON), extract text content
    if isinstance(result, dict):
        if "error" in result:
            return result.get("raw_text", "")
        # Try to extract text from JSON structure
        return json.dumps(result, indent=2)
    return result


def answer_question(
    model: DonutModel,
    processor: DonutProcessor,
    image: Union[str, Image.Image],
    question: str,
    max_length: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """
    Answer question about document (DocVQA task).

    Args:
        model: Donut model
        processor: Donut processor
        image: Document image
        question: Question about the document
        max_length: Maximum answer length
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling threshold

    Returns:
        Answer to question

    Example:
        >>> model, processor = load("naver-clova-ix/donut-base-finetuned-docvqa")
        >>> answer = answer_question(
        ...     model, processor,
        ...     image="invoice.jpg",
        ...     question="What is the total amount?"
        ... )
        >>> print(answer)
    """
    # Process image
    inputs = processor(image=image)
    pixel_values = inputs["pixel_values"]

    # Format DocVQA prompt with question
    prompt_template = TASK_PROMPTS["docvqa"]
    prompt_text = prompt_template.format(question=question)

    # Encode prompt
    prompt_ids = processor.tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_ids = mx.array(prompt_ids).reshape(1, -1)  # (1, seq_len)

    # Get EOS token ID
    eos_token_id = processor.tokenizer.eos_token_id or 2

    # Generate
    generated_ids, text = generate_text(
        model=model,
        processor=processor,
        pixel_values=pixel_values,
        prompt_ids=prompt_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )

    # Parse and extract answer
    return parse_answer(text, question=question)


def classify_document(
    model: DonutModel,
    processor: DonutProcessor,
    image: Union[str, Image.Image],
    classes: Optional[List[str]] = None,
    max_length: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """
    Classify document type (RVL-CDIP task).

    Args:
        model: Donut model
        processor: Donut processor
        image: Document image
        classes: List of possible document classes (for validation)
        max_length: Maximum output length
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling threshold

    Returns:
        Predicted document class

    Example:
        >>> model, processor = load("naver-clova-ix/donut-base")
        >>> doc_class = classify_document(
        ...     model, processor,
        ...     image="document.jpg",
        ...     classes=["invoice", "receipt", "form", "letter"]
        ... )
        >>> print(doc_class)
    """
    if classes is None:
        # Default RVL-CDIP classes
        classes = [
            "letter",
            "form",
            "email",
            "handwritten",
            "advertisement",
            "scientific report",
            "scientific publication",
            "specification",
            "file folder",
            "news article",
            "budget",
            "invoice",
            "presentation",
            "questionnaire",
            "resume",
            "memo",
        ]

    # Process image
    inputs = processor(image=image)
    pixel_values = inputs["pixel_values"]

    # Get classification prompt
    prompt_text = TASK_PROMPTS["classification"]

    # Encode prompt
    prompt_ids = processor.tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_ids = mx.array(prompt_ids).reshape(1, -1)  # (1, seq_len)

    # Get EOS token ID
    eos_token_id = processor.tokenizer.eos_token_id or 2

    # Generate
    generated_ids, text = generate_text(
        model=model,
        processor=processor,
        pixel_values=pixel_values,
        prompt_ids=prompt_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )

    # Parse classification result
    return parse_classification(text, classes=classes)


def generate(
    model: DonutModel,
    processor: DonutProcessor,
    image: Union[str, Image.Image],
    prompt: str = "",
    max_length: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """
    General-purpose document understanding generation.

    Args:
        model: Donut model
        processor: Donut processor
        image: Document image
        prompt: Optional task prompt (defaults to generic document parsing)
        max_length: Maximum output length
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling threshold

    Returns:
        Generated output

    Example:
        >>> model, processor = load("naver-clova-ix/donut-base")
        >>> output = generate(
        ...     model, processor,
        ...     image="receipt.jpg",
        ...     prompt="<s_cord-v2>"  # Or use custom prompt
        ... )
        >>> print(output)
    """
    # Process image
    inputs = processor(image=image)
    pixel_values = inputs["pixel_values"]

    # Use provided prompt or default to document parsing
    if not prompt:
        prompt = TASK_PROMPTS["document"]

    # Encode prompt
    prompt_ids = processor.tokenizer.encode(prompt, add_special_tokens=True)
    prompt_ids = mx.array(prompt_ids).reshape(1, -1)  # (1, seq_len)

    # Get EOS token ID
    eos_token_id = processor.tokenizer.eos_token_id or 2

    # Generate
    generated_ids, text = generate_text(
        model=model,
        processor=processor,
        pixel_values=pixel_values,
        prompt_ids=prompt_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )

    return text


# =============================================================================
# Beam Search Implementation
# =============================================================================


def beam_search_generate(
    model: DonutModel,
    processor: DonutProcessor,
    pixel_values: mx.array,
    prompt_ids: mx.array,
    max_length: int = 512,
    beam_size: int = 4,
    length_penalty: float = 1.0,
    eos_token_id: int = 2,
) -> Tuple[List[int], str, float]:
    """
    Generate text using beam search with length penalty.

    Args:
        model: Donut model
        processor: Donut processor
        pixel_values: Preprocessed image tensor (B, H, W, C)
        prompt_ids: Prompt token IDs (B, seq_len)
        max_length: Maximum generation length
        beam_size: Number of beams to maintain
        length_penalty: Length penalty factor (1.0 = no penalty, >1 favors longer sequences)
        eos_token_id: End-of-sequence token ID

    Returns:
        Tuple of (best_token_ids, decoded_text, score)
    """
    # Encode image once
    encoder_hidden_states = model.encode_image(pixel_values)

    # Initialize beams: (sequence, score, cache)
    initial_sequence = prompt_ids[0].tolist()
    beams = [(initial_sequence, 0.0, None)]

    for step in range(max_length):
        all_candidates = []

        for sequence, score, cache in beams:
            # Check if this beam already ended
            if len(sequence) > 0 and sequence[-1] == eos_token_id and step > 0:
                all_candidates.append((sequence, score, cache))
                continue

            # Prepare input
            current_ids = mx.array(sequence).reshape(1, -1)
            if cache is not None:
                # Use only last token when cache exists
                current_ids = current_ids[:, -1:]

            # Forward pass
            logits, new_cache = model.decode(
                encoder_hidden_states=encoder_hidden_states,
                decoder_input_ids=current_ids,
                cache=cache,
            )

            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)

            # Compute log probabilities
            logprobs = next_token_logits - mx.logsumexp(
                next_token_logits, axis=-1, keepdims=True
            )
            logprobs = logprobs.squeeze(0)  # (vocab_size,)

            # Get top-k candidates
            top_k = beam_size * 2  # Consider more candidates than beams
            top_logprobs_indices = mx.argsort(logprobs)[::-1][:top_k]

            for idx in top_logprobs_indices:
                token_id = int(idx)
                token_logprob = float(logprobs[idx])

                new_sequence = sequence + [token_id]
                new_score = score + token_logprob

                all_candidates.append((new_sequence, new_score, new_cache))

        # Rank candidates by score with length penalty
        def compute_score_with_penalty(seq, score):
            """Apply length penalty from Google NMT paper."""
            length = len(seq)
            if length_penalty == 1.0:
                penalty = length
            else:
                # Google NMT penalty: ((5 + length) / 6) ** alpha
                penalty = ((5 + length) / 6) ** length_penalty
            return score / penalty

        scored_candidates = [
            (seq, raw_score, cache, compute_score_with_penalty(seq, raw_score))
            for seq, raw_score, cache in all_candidates
        ]

        # Sort by penalized score
        scored_candidates.sort(key=lambda x: x[3], reverse=True)

        # Keep top beam_size beams
        beams = [
            (seq, raw_score, cache) for seq, raw_score, cache, _ in scored_candidates[:beam_size]
        ]

        # Check if all beams ended
        if all(len(seq) > 0 and seq[-1] == eos_token_id for seq, _, _ in beams):
            break

    # Select best beam
    best_sequence, best_score, _ = max(
        beams, key=lambda x: compute_score_with_penalty(x[0], x[1])
    )

    # Remove prompt tokens from output
    generated_ids = best_sequence[len(initial_sequence) :]

    # Decode text
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)

    return generated_ids, text, best_score
