#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Batch processing utilities for M4 optimization.

Provides efficient batch processing for improved throughput on Apple Silicon.
"""

from typing import Any, Callable, Iterator, List, TypeVar

import mlx.core as mx
import numpy as np

T = TypeVar("T")


def create_batches(
    items: List[T],
    batch_size: int,
    drop_last: bool = False,
) -> Iterator[List[T]]:
    """
    Create batches from list of items.

    Args:
        items: List of items to batch
        batch_size: Size of each batch
        drop_last: Drop last batch if smaller than batch_size

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


def pad_batch(
    sequences: List[mx.array],
    padding_value: int = 0,
    max_length: int = None,
) -> mx.array:
    """
    Pad sequences to same length for batching.

    Args:
        sequences: List of MLX arrays
        padding_value: Value to use for padding
        max_length: Maximum length (defaults to longest sequence)

    Returns:
        Padded batch as MLX array

    Example:
        >>> seqs = [mx.array([1, 2, 3]), mx.array([4, 5])]
        >>> batch = pad_batch(seqs, padding_value=0)
        >>> print(batch.shape)  # (2, 3)
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    batch_size = len(sequences)

    # Create padded batch
    padded = mx.full(
        (batch_size, max_length),
        padding_value,
        dtype=sequences[0].dtype,
    )

    # Fill with sequences
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded[i, :seq_len] = seq

    return padded


def batch_process(
    items: List[Any],
    process_fn: Callable,
    batch_size: int = 32,
    show_progress: bool = True,
) -> List[Any]:
    """
    Process items in batches.

    Args:
        items: Items to process
        process_fn: Function to apply to each batch
        batch_size: Batch size
        show_progress: Show progress

    Returns:
        List of processed results

    Example:
        >>> texts = ["text 1", "text 2", ...]
        >>> def process_batch(batch):
        ...     return model.encode(batch)
        >>> results = batch_process(texts, process_batch, batch_size=32)
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size

    for i, batch in enumerate(create_batches(items, batch_size)):
        if show_progress:
            print(f"Processing batch {i+1}/{total_batches}...", end="\r")

        batch_result = process_fn(batch)
        results.extend(batch_result if isinstance(batch_result, list) else [batch_result])

    if show_progress:
        print(f"Processing complete: {len(items)} items processed")

    return results


def dynamic_batching(
    items: List[Any],
    get_size_fn: Callable[[Any], int],
    max_batch_tokens: int = 2048,
    max_batch_size: int = 32,
) -> Iterator[List[Any]]:
    """
    Create dynamic batches based on item sizes.

    Groups items into batches where total size <= max_batch_tokens.
    Useful for variable-length sequences.

    Args:
        items: Items to batch
        get_size_fn: Function to get size of each item
        max_batch_tokens: Maximum total tokens per batch
        max_batch_size: Maximum items per batch

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


class BatchQueue:
    """
    Queue for batching items with automatic flushing.

    Useful for streaming applications where items arrive over time.

    Args:
        batch_size: Target batch size
        timeout: Flush batch after timeout (seconds)
        process_fn: Function to call on each batch

    Example:
        >>> queue = BatchQueue(batch_size=32, process_fn=model.encode)
        >>> queue.add("text 1")
        >>> queue.add("text 2")
        >>> # ... batch automatically processed when full
        >>> queue.flush()  # Process remaining items
    """

    def __init__(
        self,
        batch_size: int = 32,
        timeout: float = None,
        process_fn: Callable = None,
    ):
        self.batch_size = batch_size
        self.timeout = timeout
        self.process_fn = process_fn

        self.queue = []
        self.results = []
        self.last_flush = None

        if timeout is not None:
            import time

            self.last_flush = time.time()

    def add(self, item: Any) -> None:
        """Add item to queue."""
        self.queue.append(item)

        # Auto-flush if batch is full
        if len(self.queue) >= self.batch_size:
            self.flush()

        # Auto-flush if timeout exceeded
        if self.timeout is not None:
            import time

            if time.time() - self.last_flush > self.timeout:
                self.flush()

    def flush(self) -> List[Any]:
        """Process and return all queued items."""
        if not self.queue:
            return []

        # Process batch
        if self.process_fn is not None:
            results = self.process_fn(self.queue)
            self.results.extend(results if isinstance(results, list) else [results])
        else:
            self.results.extend(self.queue)

        # Clear queue
        self.queue = []

        if self.timeout is not None:
            import time

            self.last_flush = time.time()

        return self.results


def parallel_batch_process(
    items: List[Any],
    process_fn: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
) -> List[Any]:
    """
    Process batches in parallel using multiple workers.

    Args:
        items: Items to process
        process_fn: Function to apply to each batch
        batch_size: Batch size
        num_workers: Number of parallel workers

    Returns:
        List of processed results

    Example:
        >>> results = parallel_batch_process(
        ...     texts,
        ...     process_fn=embed_batch,
        ...     batch_size=32,
        ...     num_workers=4
        ... )

    Note:
        This uses threading, not multiprocessing, as MLX operations
        release the GIL during computation.
    """
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    # Create batches
    batches = list(create_batches(items, batch_size))

    # Process batches in parallel
    results = []
    results_lock = Lock()

    def process_batch_wrapper(batch):
        result = process_fn(batch)
        with results_lock:
            results.extend(result if isinstance(result, list) else [result])

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_batch_wrapper, batches)

    return results


def optimize_batch_size(
    process_fn: Callable,
    sample_items: List[Any],
    batch_sizes: List[int] = [1, 4, 8, 16, 32, 64, 128],
    metric: str = "throughput",
) -> int:
    """
    Find optimal batch size for a given operation.

    Args:
        process_fn: Function to benchmark
        sample_items: Sample items for testing
        batch_sizes: Batch sizes to test
        metric: Optimization metric ("throughput" or "latency")

    Returns:
        Optimal batch size

    Example:
        >>> optimal = optimize_batch_size(
        ...     process_fn=model.encode,
        ...     sample_items=sample_texts[:100],
        ...     metric="throughput"
        ... )
        >>> print(f"Optimal batch size: {optimal}")
    """
    import time

    print("\n" + "=" * 70)
    print("Batch Size Optimization")
    print("=" * 70)

    results = {}

    for batch_size in batch_sizes:
        if batch_size > len(sample_items):
            continue

        print(f"\nTesting batch size: {batch_size}")

        # Warmup
        batch = sample_items[:batch_size]
        process_fn(batch)

        # Benchmark
        num_runs = 5
        times = []

        for _ in range(num_runs):
            batch = sample_items[:batch_size]
            start = time.time()
            process_fn(batch)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)

        # Calculate metrics
        throughput = batch_size / avg_time  # items/sec
        latency = avg_time * 1000  # ms

        results[batch_size] = {
            "throughput": throughput,
            "latency": latency,
        }

        print(f"  Throughput: {throughput:.1f} items/sec")
        print(f"  Latency: {latency:.2f}ms")

    # Find optimal
    if metric == "throughput":
        optimal = max(results.items(), key=lambda x: x[1]["throughput"])
    else:  # latency
        optimal = min(results.items(), key=lambda x: x[1]["latency"])

    print("\n" + "=" * 70)
    print(f"Optimal batch size: {optimal[0]}")
    print(f"  {metric.capitalize()}: {optimal[1][metric]:.2f}")
    print("=" * 70)

    return optimal[0]
