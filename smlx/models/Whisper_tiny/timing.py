"""
Word-level timestamp alignment for Whisper using Dynamic Time Warping.

This module provides functionality to align words with their corresponding audio
frames by analyzing cross-attention patterns from the Whisper decoder.

The alignment process:
1. Extract cross-attention weights from specific decoder heads
2. Apply median filtering to smooth attention patterns
3. Use Dynamic Time Warping (DTW) to find optimal alignment
4. Merge punctuation with adjacent words
5. Add start/end timestamps to each word

Based on the approach from:
- https://github.com/openai/whisper/discussions/1363
- https://github.com/m-bain/whisperX

Usage:
    from smlx.models.Whisper_tiny import load
    from smlx.models.Whisper_tiny.timing import add_word_timestamps

    model, tokenizer = load()
    result = transcribe("speech.wav", model, tokenizer, word_timestamps=True)

    for segment in result["segments"]:
        for word in segment["words"]:
            print(f"[{word['start']:.2f}s - {word['end']:.2f}s]: {word['word']}")
"""

import re
from typing import List, Tuple, Optional

import mlx.core as mx
import numpy as np

from .audio import N_FRAMES, HOP_LENGTH, SAMPLE_RATE


def median_filter(x: np.ndarray, filter_width: int) -> np.ndarray:
    """Apply a median filter to smooth an array.

    Args:
        x: Input array of shape (n,)
        filter_width: Width of the median filter window

    Returns:
        Filtered array of same shape as input
    """
    if filter_width <= 0:
        return x

    pad_width = filter_width // 2
    x_padded = np.pad(x, (pad_width, pad_width), mode="edge")

    result = np.zeros_like(x)
    for i in range(len(x)):
        result[i] = np.median(x_padded[i : i + filter_width])

    return result


def dtw(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Dynamic Time Warping to find optimal alignment path.

    Uses dynamic programming to find the path through the cost matrix that
    minimizes total cost. The path represents the alignment between two sequences.

    Args:
        x: Cost matrix of shape (N, M) where:
           - N is the number of text tokens
           - M is the number of audio frames
           Lower values indicate better alignment

    Returns:
        Tuple of (text_indices, time_indices) representing the alignment path
        - text_indices: Array of token indices
        - time_indices: Array of corresponding frame indices
    """
    N, M = x.shape

    # Initialize cost matrix with infinity
    cost = np.full((N + 1, M + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0

    # Initialize trace matrix to track path
    trace = -np.ones((N + 1, M + 1), dtype=np.int32)

    # Fill cost and trace matrices
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            # Consider three possible predecessors: diagonal, vertical, horizontal
            c0 = cost[i - 1, j - 1]  # Diagonal (both sequences advance)
            c1 = cost[i - 1, j]      # Vertical (only text advances)
            c2 = cost[i, j - 1]      # Horizontal (only time advances)

            # Choose minimum cost path
            if c0 <= c1 and c0 <= c2:
                cost[i, j] = x[i - 1, j - 1] + c0
                trace[i, j] = 0  # Diagonal
            elif c1 <= c2:
                cost[i, j] = x[i - 1, j - 1] + c1
                trace[i, j] = 1  # Vertical
            else:
                cost[i, j] = x[i - 1, j - 1] + c2
                trace[i, j] = 2  # Horizontal

    # Backtrack to find optimal path
    i, j = N, M
    text_indices = []
    time_indices = []

    while i > 0 and j > 0:
        text_indices.append(i - 1)
        time_indices.append(j - 1)

        direction = trace[i, j]
        if direction == 0:  # Diagonal
            i -= 1
            j -= 1
        elif direction == 1:  # Vertical
            i -= 1
        else:  # Horizontal
            j -= 1

    # Reverse to get forward path
    text_indices = np.array(text_indices[::-1], dtype=np.int32)
    time_indices = np.array(time_indices[::-1], dtype=np.int32)

    return text_indices, time_indices


def find_alignment(
    model,
    tokenizer,
    text_tokens: List[int],
    mel: mx.array,
    num_frames: int,
    *,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find word-level alignment using cross-attention patterns and DTW.

    Args:
        model: Whisper model
        tokenizer: Whisper tokenizer
        text_tokens: List of token IDs for the transcribed text
        mel: Mel-spectrogram of shape (n_mels, n_frames)
        num_frames: Number of audio frames (after encoder stride)
        medfilt_width: Width of median filter for smoothing attention weights
        qk_scale: Scale factor for attention scores

    Returns:
        Tuple of (text_indices, time_indices) representing word-to-frame alignment
    """
    if len(text_tokens) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Prepare tokens with SOT sequence
    tokens = mx.array([tokenizer.sot_sequence + tuple(text_tokens)])

    # Get logits and cross-attention weights
    logits, cross_qk = model.forward_with_cross_qk(mel[None, :], tokens)

    # Extract weights from alignment heads
    # alignment_heads is a list of (layer, head) pairs
    weights = []
    for layer_idx, head_idx in model.alignment_heads:
        # cross_qk[layer_idx] has shape (batch, n_heads, seq_len, audio_ctx)
        weight = cross_qk[layer_idx][0, head_idx]  # (seq_len, audio_ctx)
        weights.append(weight)

    # Stack and average across alignment heads
    weights = mx.stack(weights, axis=0)  # (n_alignment_heads, seq_len, audio_ctx)

    # Apply softmax to get attention probabilities
    weights = mx.softmax(weights * qk_scale, axis=-1)

    # Average across alignment heads
    weights = weights.mean(axis=0)  # (seq_len, audio_ctx)

    # Extract weights for actual text tokens (skip SOT sequence)
    sot_len = len(tokenizer.sot_sequence)
    weights = weights[sot_len : sot_len + len(text_tokens), :num_frames]

    # Convert to numpy for DTW
    weights_np = np.array(weights)

    # Apply median filter to smooth attention patterns
    if medfilt_width > 0:
        for i in range(weights_np.shape[0]):
            weights_np[i] = median_filter(weights_np[i], medfilt_width)

    # Compute cost matrix (use negative log probabilities)
    # Higher attention weight = lower cost
    cost_matrix = -np.log(weights_np + 1e-10)

    # Find alignment using DTW
    text_indices, time_indices = dtw(cost_matrix)

    return text_indices, time_indices


def merge_punctuations(words: List[dict], prepend_punctuations: str, append_punctuations: str) -> List[dict]:
    """Merge punctuation marks with adjacent words.

    Args:
        words: List of word dictionaries with 'word', 'start', 'end' keys
        prepend_punctuations: Punctuation to merge with following word (e.g., "\"'¿([{")
        append_punctuations: Punctuation to merge with preceding word (e.g., ".!?,;:%)]}\"")

    Returns:
        List of words with punctuation merged
    """
    if not words:
        return words

    # Merge punctuation that should be prepended
    i = len(words) - 2
    while i >= 0:
        word = words[i]["word"]
        if word in prepend_punctuations:
            # Merge with next word
            if i < len(words) - 1:
                words[i + 1]["word"] = word + words[i + 1]["word"]
                words[i + 1]["start"] = words[i]["start"]
                words.pop(i)
        i -= 1

    # Merge punctuation that should be appended
    i = 1
    while i < len(words):
        word = words[i]["word"]
        if word in append_punctuations:
            # Merge with previous word
            if i > 0:
                words[i - 1]["word"] = words[i - 1]["word"] + word
                words[i - 1]["end"] = words[i]["end"]
                words.pop(i)
        else:
            i += 1

    return words


def split_tokens_on_spaces(tokens: List[int], tokenizer) -> List[List[int]]:
    """Split token sequence into words based on leading/trailing spaces.

    Handles both space-separated languages (English) and non-space-separated
    languages (Chinese, Japanese, etc.) by checking decoded token text.

    Args:
        tokens: List of token IDs
        tokenizer: Whisper tokenizer

    Returns:
        List of token groups, where each group represents one word
    """
    if not tokens:
        return []

    # Decode each token to check for spaces
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]

    word_groups = []
    current_word = []

    for i, (token, decoded) in enumerate(zip(tokens, decoded_tokens)):
        # Check if this token starts with a space (new word)
        if decoded.startswith(" ") and current_word:
            # Start new word
            word_groups.append(current_word)
            current_word = [token]
        else:
            current_word.append(token)

    # Add last word
    if current_word:
        word_groups.append(current_word)

    return word_groups


def add_word_timestamps(
    *,
    segments: List[dict],
    model,
    tokenizer,
    mel: mx.array,
    num_frames: int,
    prepend_punctuations: str = "\"'¿([{-",
    append_punctuations: str = ".!?,;:%)]}\"",
    **kwargs,
) -> List[dict]:
    """Add word-level timestamps to transcription segments.

    This is the main function for adding word-level timestamps. It processes each
    segment, finds word boundaries, aligns them with audio frames using DTW,
    and adds start/end times to each word.

    Args:
        segments: List of segment dictionaries from transcribe()
        model: Whisper model
        tokenizer: Whisper tokenizer
        mel: Mel-spectrogram of shape (n_mels, n_frames)
        num_frames: Number of audio frames (after encoder stride)
        prepend_punctuations: Punctuation to merge with following word
        append_punctuations: Punctuation to merge with preceding word
        **kwargs: Additional arguments (e.g., medfilt_width, qk_scale)

    Returns:
        List of segments with added 'words' field containing word-level timestamps

    Example:
        >>> segments = [{"text": "hello world", "tokens": [15339, 1002], "start": 0.0, "end": 1.5}]
        >>> segments = add_word_timestamps(
        ...     segments=segments,
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     mel=mel,
        ...     num_frames=num_frames,
        ... )
        >>> segments[0]["words"]
        [
            {"word": "hello", "start": 0.0, "end": 0.8, "probability": 0.95},
            {"word": "world", "start": 0.8, "end": 1.5, "probability": 0.98},
        ]
    """
    if not segments:
        return segments

    # Process each segment
    for segment in segments:
        tokens = segment.get("tokens", [])
        if not tokens:
            segment["words"] = []
            continue

        # Filter out special tokens (timestamps, etc.)
        text_tokens = [
            t for t in tokens
            if t < tokenizer.timestamp_begin
        ]

        if not text_tokens:
            segment["words"] = []
            continue

        # Find alignment between tokens and audio frames
        segment_start = segment["start"]
        segment_end = segment["end"]

        # Calculate frame range for this segment
        start_frame = int(segment_start * SAMPLE_RATE / HOP_LENGTH)
        end_frame = int(segment_end * SAMPLE_RATE / HOP_LENGTH)
        segment_frames = end_frame - start_frame

        # Get alignment
        text_indices, time_indices = find_alignment(
            model=model,
            tokenizer=tokenizer,
            text_tokens=text_tokens,
            mel=mel,
            num_frames=min(segment_frames, num_frames - start_frame),
            **kwargs,
        )

        if len(text_indices) == 0:
            segment["words"] = []
            continue

        # Split tokens into words
        word_token_groups = split_tokens_on_spaces(text_tokens, tokenizer)

        # Assign timestamps to each word
        words = []
        token_idx = 0

        for word_tokens in word_token_groups:
            if not word_tokens:
                continue

            # Find frame range for this word's tokens
            word_start_idx = token_idx
            word_end_idx = token_idx + len(word_tokens) - 1

            # Find corresponding time indices
            word_time_indices = []
            for i, text_idx in enumerate(text_indices):
                if word_start_idx <= text_idx <= word_end_idx:
                    word_time_indices.append(time_indices[i])

            if word_time_indices:
                # Convert frame indices to seconds
                start_time = segment_start + (min(word_time_indices) * HOP_LENGTH / SAMPLE_RATE)
                end_time = segment_start + ((max(word_time_indices) + 1) * HOP_LENGTH / SAMPLE_RATE)

                # Decode word text
                word_text = tokenizer.decode(word_tokens)

                words.append({
                    "word": word_text,
                    "start": round(start_time, 2),
                    "end": round(min(end_time, segment_end), 2),
                    "probability": segment.get("avg_logprob", 0.0),
                })

            token_idx += len(word_tokens)

        # Merge punctuation with adjacent words
        words = merge_punctuations(words, prepend_punctuations, append_punctuations)

        # Ensure words don't exceed segment boundaries
        if words:
            words[0]["start"] = max(words[0]["start"], segment_start)
            words[-1]["end"] = min(words[-1]["end"], segment_end)

        segment["words"] = words

    return segments
