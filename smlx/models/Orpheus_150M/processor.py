#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Text processor for Orpheus-150M TTS.

Handles text normalization, tokenization, and phoneme conversion.
"""

import re
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np


class TextProcessor:
    """
    Process text for TTS synthesis.

    Handles:
    - Text normalization (lowercase, punctuation, numbers)
    - Tokenization (characters or phonemes)
    - Padding and batching

    Args:
        vocab: Vocabulary mapping (character/phoneme -> ID)
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 512,
    ):
        # Default vocabulary (basic characters + phonemes)
        if vocab is None:
            vocab = self._create_default_vocab()

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_length = max_length

        # Create reverse mapping
        self.id_to_token = {v: k for k, v in vocab.items()}

        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"

        self.pad_id = vocab.get(self.pad_token, 0)
        self.unk_id = vocab.get(self.unk_token, 1)
        self.bos_id = vocab.get(self.bos_token, 2)
        self.eos_id = vocab.get(self.eos_token, 3)

    def _create_default_vocab(self) -> Dict[str, int]:
        """
        Create default vocabulary.

        Includes:
        - Special tokens (<pad>, <unk>, <s>, </s>)
        - ASCII letters (a-z)
        - Digits (0-9)
        - Common punctuation
        - Space
        """
        vocab = {}
        idx = 0

        # Special tokens
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        for token in special_tokens:
            vocab[token] = idx
            idx += 1

        # Letters (lowercase)
        for c in "abcdefghijklmnopqrstuvwxyz":
            vocab[c] = idx
            idx += 1

        # Digits
        for c in "0123456789":
            vocab[c] = idx
            idx += 1

        # Punctuation and space
        for c in " .,!?;:-'\"()":
            vocab[c] = idx
            idx += 1

        return vocab

    def normalize_text(self, text: str) -> str:
        """
        Normalize input text.

        Args:
            text: Raw input text

        Returns:
            Normalized text

        Example:
            >>> processor = TextProcessor()
            >>> processor.normalize_text("Hello, World! 123")
            'hello, world! 123'
        """
        # Lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Trim
        text = text.strip()

        return text

    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs

        Example:
            >>> processor = TextProcessor()
            >>> processor.text_to_sequence("hello")
            [2, 7, 4, 11, 11, 14, 3]  # <s> + tokens + </s>
        """
        # Normalize text
        text = self.normalize_text(text)

        # Convert to IDs
        sequence = [self.bos_id]  # Start token

        for char in text:
            token_id = self.vocab.get(char, self.unk_id)
            sequence.append(token_id)

        sequence.append(self.eos_id)  # End token

        return sequence

    def __call__(self, text: str, padding: bool = True) -> mx.array:
        """
        Process text to model input.

        Args:
            text: Input text
            padding: Whether to pad to max_length

        Returns:
            Token IDs as MLX array
            Shape: (seq_len,) or (max_length,) if padded

        Example:
            >>> processor = TextProcessor()
            >>> tokens = processor("Hello world")
            >>> print(tokens.shape)  # (max_length,)
        """
        # Convert to sequence
        sequence = self.text_to_sequence(text)

        # Truncate if too long
        if len(sequence) > self.max_length:
            sequence = sequence[: self.max_length - 1] + [self.eos_id]

        # Pad if requested
        if padding:
            pad_length = self.max_length - len(sequence)
            sequence = sequence + [self.pad_id] * pad_length

        # Convert to MLX array
        return mx.array(sequence, dtype=mx.int32)

    def batch_process(self, texts: List[str]) -> mx.array:
        """
        Process batch of texts.

        Args:
            texts: List of input texts

        Returns:
            Batch of token IDs
            Shape: (batch_size, max_length)

        Example:
            >>> processor = TextProcessor()
            >>> batch = processor.batch_process(["Hello", "World"])
            >>> print(batch.shape)  # (2, max_length)
        """
        sequences = [self(text, padding=True) for text in texts]
        return mx.stack(sequences, axis=0)

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text

        Example:
            >>> processor = TextProcessor()
            >>> processor.decode([2, 7, 4, 11, 11, 14, 3])
            'hello'
        """
        tokens = []
        for token_id in token_ids:
            # Skip special tokens
            if token_id in [self.pad_id, self.bos_id, self.eos_id]:
                continue

            token = self.id_to_token.get(token_id, self.unk_token)
            tokens.append(token)

        return "".join(tokens)


def create_processor(vocab: Optional[Dict[str, int]] = None) -> TextProcessor:
    """
    Create text processor with default settings.

    Args:
        vocab: Optional custom vocabulary

    Returns:
        TextProcessor instance

    Example:
        >>> processor = create_processor()
        >>> tokens = processor("Hello world")
    """
    return TextProcessor(vocab=vocab)


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """
    Load vocabulary from file.

    Args:
        vocab_path: Path to vocabulary JSON file

    Returns:
        Vocabulary dictionary
    """
    import json
    from pathlib import Path

    vocab_path = Path(vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    return vocab


def save_vocab(vocab: Dict[str, int], output_path: str):
    """
    Save vocabulary to file.

    Args:
        vocab: Vocabulary dictionary
        output_path: Output path for JSON file
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"✓ Vocabulary saved to {output_path}")
