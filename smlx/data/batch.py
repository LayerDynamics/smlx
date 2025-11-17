"""
Batching and data loading utilities for SMLX.

This module provides DataLoader and collation utilities for efficient batch processing,
consolidating existing batch utilities from smlx/utils/batch.py.

Adapted from:
- smlx/utils/batch.py (existing batch utilities)
- PyTorch DataLoader patterns (simplified for MLX)
"""

import random
from typing import Any, Callable, Iterator, Optional, Protocol

import mlx.core as mx
import numpy as np
from PIL import Image


class Dataset(Protocol):
    """Protocol for dataset classes."""

    def __len__(self) -> int:
        """Return dataset length."""
        ...

    def __getitem__(self, idx: int) -> Any:
        """Get item at index."""
        ...


class DataLoader:
    """
    Simple DataLoader for batching dataset items.

    No PyTorch dependency - lightweight implementation for MLX.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data each epoch (default: False)
        collate_fn: Function to collate items into batches (default: None)
        drop_last: Drop last batch if smaller than batch_size (default: False)

    Example:
        >>> dataset = TextDataset(data, tokenizer)
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     collate_fn=collate_text
        ... )
        >>> for batch in dataloader:
        ...     outputs = model(batch)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or default_collate
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Any]:
        """Iterate over batches."""
        # Get indices
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]

            # Skip last batch if too small
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Load batch items
            batch = [self.dataset[idx] for idx in batch_indices]

            # Collate batch
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def default_collate(batch: list[Any]) -> Any:
    """
    Default collation function.

    Simply returns the batch as a list if items can't be automatically batched.

    Args:
        batch: List of items to collate

    Returns:
        Collated batch
    """
    if not batch:
        return []

    # If items are dictionaries, try to stack each key
    if isinstance(batch[0], dict):
        return {
            key: default_collate([item[key] for item in batch])
            for key in batch[0].keys()
        }

    # If items are MLX arrays, stack them
    if isinstance(batch[0], mx.array):
        return mx.stack(batch)

    # If items are numpy arrays, stack them
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)

    # Otherwise return as list
    return batch


def collate_text(batch: list[dict[str, Any]]) -> dict[str, mx.array]:
    """
    Collate text samples with padding.

    Args:
        batch: List of dictionaries with 'input_ids' and optionally 'attention_mask'

    Returns:
        Dictionary with padded tensors

    Example:
        >>> batch = [
        ...     {"input_ids": [1, 2, 3]},
        ...     {"input_ids": [4, 5]}
        ... ]
        >>> collated = collate_text(batch)
        >>> collated["input_ids"].shape
        (2, 3)
    """
    # Get input_ids
    input_ids = [item["input_ids"] for item in batch]

    # Pad sequences
    padded_ids = pad_sequences(input_ids, padding_value=0)

    result = {"input_ids": padded_ids}

    # Add attention mask if present
    if "attention_mask" in batch[0]:
        attention_masks = [item["attention_mask"] for item in batch]
        result["attention_mask"] = pad_sequences(attention_masks, padding_value=0)
    else:
        # Create attention mask based on padding
        lengths = [len(seq) for seq in input_ids]
        max_len = padded_ids.shape[1]
        attention_mask = mx.array(
            [[1] * length + [0] * (max_len - length) for length in lengths]
        )
        result["attention_mask"] = attention_mask

    return result


def collate_images(batch: list[Image.Image]) -> mx.array:
    """
    Collate PIL Images into batched tensor with padding.

    Args:
        batch: List of PIL Images

    Returns:
        Batched MLX array of shape [B, C, H, W]

    Example:
        >>> images = [Image.open(f"img{i}.jpg") for i in range(4)]
        >>> batched = collate_images(images)
        >>> batched.shape
        (4, 3, 224, 224)
    """
    # Convert images to arrays
    arrays = []
    for img in batch:
        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert to array [H, W, C]
        arr = np.array(img)

        # Transpose to [C, H, W]
        arr = arr.transpose(2, 0, 1)

        arrays.append(mx.array(arr))

    # Pad to max dimensions
    return batch_images(arrays)


def collate_audio(batch: list[mx.array], padding_value: float = 0.0) -> mx.array:
    """
    Collate audio arrays with padding.

    Args:
        batch: List of audio arrays
        padding_value: Value to use for padding (default: 0.0)

    Returns:
        Batched MLX array of shape [B, max_length]

    Example:
        >>> audios = [mx.random.randn((16000,)), mx.random.randn((12000,))]
        >>> batched = collate_audio(audios)
        >>> batched.shape
        (2, 16000)
    """
    # Get max length
    max_length = max(audio.shape[0] for audio in batch)

    # Create padded batch
    batch_size = len(batch)
    batched = mx.full((batch_size, max_length), padding_value)

    # Copy each audio into batch
    for i, audio in enumerate(batch):
        length = audio.shape[0]
        batched[i, :length] = audio

    return batched


def collate_vlm(batch: list[dict[str, Any]]) -> dict[str, mx.array]:
    """
    Collate vision-language model samples.

    Args:
        batch: List of dictionaries with 'pixel_values', 'input_ids', etc.

    Returns:
        Dictionary with batched tensors

    Example:
        >>> batch = [
        ...     {"pixel_values": img1, "input_ids": [1, 2, 3]},
        ...     {"pixel_values": img2, "input_ids": [4, 5]}
        ... ]
        >>> collated = collate_vlm(batch)
    """
    result = {}

    # Collate images if present
    if "pixel_values" in batch[0]:
        pixel_values = [item["pixel_values"] for item in batch]
        if isinstance(pixel_values[0], mx.array):
            result["pixel_values"] = batch_images(pixel_values)
        else:
            result["pixel_values"] = mx.stack(pixel_values)

    # Collate text if present
    if "input_ids" in batch[0]:
        text_batch = [
            {k: v for k, v in item.items() if k in ["input_ids", "attention_mask"]}
            for item in batch
        ]
        text_result = collate_text(text_batch)
        result.update(text_result)

    # Add any other fields as lists
    for key in batch[0].keys():
        if key not in result:
            result[key] = [item[key] for item in batch]

    return result


def pad_sequences(
    sequences: list[Any], padding_value: int = 0, max_length: Optional[int] = None
) -> mx.array:
    """
    Pad sequences to same length.

    Args:
        sequences: List of sequences (lists or arrays)
        padding_value: Value to use for padding (default: 0)
        max_length: Maximum length (default: use longest sequence)

    Returns:
        Padded MLX array of shape [batch_size, max_length]

    Example:
        >>> seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        >>> padded = pad_sequences(seqs, padding_value=0)
        >>> padded.shape
        (3, 4)
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    batch_size = len(sequences)

    # Create padded batch
    padded = mx.full((batch_size, max_length), padding_value)

    # Fill with sequences
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_length)
        if isinstance(seq, mx.array):
            padded[i, :seq_len] = seq[:seq_len]
        else:
            padded[i, :seq_len] = mx.array(seq[:seq_len])

    return padded


def batch_images(images: list[mx.array], padding_value: float = 0.0) -> mx.array:
    """
    Batch multiple images into a single tensor with padding.

    Images are padded to match the largest width and height in the batch.

    Args:
        images: List of MLX arrays with shape [C, H, W]
        padding_value: Value to use for padding (default: 0.0)

    Returns:
        MLX array of shape [B, C, H_max, W_max]

    Example:
        >>> img1 = mx.random.randn((3, 224, 224))
        >>> img2 = mx.random.randn((3, 256, 256))
        >>> batched = batch_images([img1, img2])
        >>> batched.shape
        (2, 3, 256, 256)
    """
    if not images:
        raise ValueError("Cannot batch empty list of images")

    # Get max dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    num_channels = images[0].shape[0]

    # Create batch tensor
    batch_size = len(images)
    batched = mx.full((batch_size, num_channels, max_h, max_w), padding_value)

    # Copy each image into the batch tensor
    for i, img in enumerate(images):
        c, h, w = img.shape
        batched[i, :, :h, :w] = img

    return batched


def create_batches(
    items: list[Any], batch_size: int, drop_last: bool = False
) -> Iterator[list[Any]]:
    """
    Create batches from list of items.

    Args:
        items: List of items to batch
        batch_size: Size of each batch
        drop_last: Drop last batch if smaller than batch_size (default: False)

    Yields:
        Batches of items

    Example:
        >>> items = list(range(100))
        >>> for batch in create_batches(items, batch_size=32):
        ...     process_batch(batch)
    """
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]

        # Skip last batch if too small
        if drop_last and len(batch) < batch_size:
            continue

        yield batch


def dynamic_batching(
    items: list[Any],
    get_size_fn: Callable[[Any], int],
    max_batch_tokens: int = 2048,
    max_batch_size: int = 32,
) -> Iterator[list[Any]]:
    """
    Create dynamic batches based on item sizes.

    Groups items into batches where total size <= max_batch_tokens.
    Useful for variable-length sequences.

    Args:
        items: Items to batch
        get_size_fn: Function to get size of each item
        max_batch_tokens: Maximum total tokens per batch (default: 2048)
        max_batch_size: Maximum items per batch (default: 32)

    Yields:
        Dynamic-sized batches

    Example:
        >>> texts = ["short", "medium text here", "very long text..."]
        >>> def get_length(text):
        ...     return len(text.split())
        >>> for batch in dynamic_batching(texts, get_length, max_batch_tokens=100):
        ...     process_batch(batch)
    """
    current_batch = []
    current_tokens = 0

    for item in items:
        item_size = get_size_fn(item)

        # Check if adding item would exceed limits
        would_exceed_tokens = current_tokens + item_size > max_batch_tokens
        would_exceed_size = len(current_batch) >= max_batch_size

        if current_batch and (would_exceed_tokens or would_exceed_size):
            # Yield current batch and start new one
            yield current_batch
            current_batch = [item]
            current_tokens = item_size
        else:
            # Add to current batch
            current_batch.append(item)
            current_tokens += item_size

    # Yield final batch
    if current_batch:
        yield current_batch
