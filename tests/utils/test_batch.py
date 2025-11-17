"""Tests for smlx.utils.batch module."""

import mlx.core as mx

from smlx.utils.batch import (
    BatchQueue,
    create_batches,
    dynamic_batching,
    pad_batch,
)


class TestCreateBatches:
    """Test create_batches function."""

    def test_create_batches_basic(self):
        """Test creating batches from list."""
        items = list(range(10))
        batches = list(create_batches(items, batch_size=3))

        assert len(batches) == 4  # 3, 3, 3, 1
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]

    def test_create_batches_exact_division(self):
        """Test when items divide evenly into batches."""
        items = list(range(12))
        batches = list(create_batches(items, batch_size=4))

        assert len(batches) == 3
        assert all(len(b) == 4 for b in batches)

    def test_create_batches_drop_last(self):
        """Test dropping incomplete last batch."""
        items = list(range(10))
        batches = list(create_batches(items, batch_size=3, drop_last=True))

        # Should drop the last batch with only 1 item
        assert len(batches) == 3
        assert all(len(b) == 3 for b in batches)

    def test_create_batches_single_batch(self):
        """Test when all items fit in one batch."""
        items = list(range(5))
        batches = list(create_batches(items, batch_size=10))

        assert len(batches) == 1
        assert batches[0] == items

    def test_create_batches_empty(self):
        """Test with empty list."""
        items = []
        batches = list(create_batches(items, batch_size=5))

        assert len(batches) == 0

    def test_create_batches_batch_size_one(self):
        """Test with batch size of 1."""
        items = [1, 2, 3]
        batches = list(create_batches(items, batch_size=1))

        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)


class TestPadBatch:
    """Test pad_batch function."""

    def test_pad_batch_basic(self):
        """Test basic padding of sequences."""
        sequences = [
            mx.array([1, 2, 3]),
            mx.array([4, 5]),
            mx.array([6, 7, 8, 9]),
        ]

        padded = pad_batch(sequences, padding_value=0)

        # Should pad to length 4 (longest sequence)
        assert padded.shape == (3, 4)
        # First sequence should be padded with one 0
        assert padded[0, 3].item() == 0
        # Second sequence should be padded with two 0s
        assert padded[1, 2].item() == 0
        assert padded[1, 3].item() == 0

    def test_pad_batch_custom_padding_value(self):
        """Test padding with custom value."""
        sequences = [
            mx.array([1, 2]),
            mx.array([3]),
        ]

        padded = pad_batch(sequences, padding_value=-1)

        assert padded[1, 1].item() == -1  # Should be padded with -1

    def test_pad_batch_max_length(self):
        """Test padding to specific max length."""
        sequences = [
            mx.array([1, 2]),
            mx.array([3, 4, 5]),
        ]

        padded = pad_batch(sequences, padding_value=0, max_length=5)

        assert padded.shape == (2, 5)

    def test_pad_batch_same_length(self):
        """Test padding when all sequences have same length."""
        sequences = [
            mx.array([1, 2, 3]),
            mx.array([4, 5, 6]),
        ]

        padded = pad_batch(sequences, padding_value=0)

        # No padding needed
        assert padded.shape == (2, 3)

    def test_pad_batch_single_sequence(self):
        """Test padding single sequence."""
        sequences = [mx.array([1, 2, 3])]

        padded = pad_batch(sequences, padding_value=0)

        assert padded.shape == (1, 3)

    def test_pad_batch_preserves_values(self):
        """Test that original values are preserved."""
        sequences = [
            mx.array([1, 2, 3]),
            mx.array([4, 5]),
        ]

        padded = pad_batch(sequences, padding_value=0)

        # Check original values
        assert padded[0, 0].item() == 1
        assert padded[0, 1].item() == 2
        assert padded[0, 2].item() == 3
        assert padded[1, 0].item() == 4
        assert padded[1, 1].item() == 5


class TestDynamicBatching:
    """Test dynamic_batching function."""

    def test_dynamic_batching_basic(self):
        """Test basic dynamic batching."""
        items = ["a", "bb", "ccc", "dddd", "eeeee"]

        def get_length(text):
            return len(text)

        batches = list(
            dynamic_batching(items, get_size_fn=get_length, max_batch_tokens=6)
        )

        # First batch: "a" (1) + "bb" (2) + "ccc" (3) = 6 tokens
        assert len(batches) >= 1
        # Should group items efficiently

    def test_dynamic_batching_max_batch_size(self):
        """Test max_batch_size limit."""
        items = ["a"] * 10

        def get_length(text):
            return 1

        batches = list(
            dynamic_batching(
                items,
                get_size_fn=get_length,
                max_batch_tokens=100,
                max_batch_size=3,
            )
        )

        # Should respect max_batch_size
        assert all(len(b) <= 3 for b in batches)

    def test_dynamic_batching_single_large_item(self):
        """Test when single item exceeds max_batch_tokens."""
        items = ["a" * 100, "b", "c"]

        def get_length(text):
            return len(text)

        batches = list(
            dynamic_batching(items, get_size_fn=get_length, max_batch_tokens=50)
        )

        # Large item should be in its own batch
        assert len(batches[0]) == 1

    def test_dynamic_batching_empty(self):
        """Test with empty items."""
        items = []

        batches = list(
            dynamic_batching(items, get_size_fn=lambda x: len(x), max_batch_tokens=10)
        )

        assert len(batches) == 0

    def test_dynamic_batching_variable_sizes(self):
        """Test with variable size items."""
        items = list(range(1, 11))  # 1, 2, 3, ..., 10

        def get_size(x):
            return x

        batches = list(
            dynamic_batching(items, get_size_fn=get_size, max_batch_tokens=15)
        )

        # Each batch should not exceed 15 tokens total
        for batch in batches:
            total_size = sum(get_size(item) for item in batch)
            assert total_size <= 15


class TestBatchQueue:
    """Test BatchQueue class."""

    def test_batch_queue_init(self):
        """Test BatchQueue initialization."""
        queue = BatchQueue(batch_size=32)

        assert queue.batch_size == 32
        assert len(queue.queue) == 0
        assert len(queue.results) == 0

    def test_batch_queue_add_and_auto_flush(self):
        """Test adding items and auto-flushing."""

        def process_fn(batch):
            return [x * 2 for x in batch]

        queue = BatchQueue(batch_size=3, process_fn=process_fn)

        # Add items one by one
        queue.add(1)
        queue.add(2)

        # Not yet flushed
        assert len(queue.queue) == 2

        # Adding third item should trigger auto-flush
        queue.add(3)

        # Queue should be empty, results should have processed items
        assert len(queue.queue) == 0
        assert len(queue.results) == 3
        assert queue.results == [2, 4, 6]

    def test_batch_queue_manual_flush(self):
        """Test manual flushing."""

        def process_fn(batch):
            return batch

        queue = BatchQueue(batch_size=10, process_fn=process_fn)

        queue.add(1)
        queue.add(2)

        results = queue.flush()

        assert results == [1, 2]
        assert len(queue.queue) == 0

    def test_batch_queue_no_process_fn(self):
        """Test queue without process function."""
        queue = BatchQueue(batch_size=5)

        queue.add(1)
        queue.add(2)

        results = queue.flush()

        # Without process_fn, items are just added to results
        assert 1 in results
        assert 2 in results

    def test_batch_queue_empty_flush(self):
        """Test flushing empty queue."""
        queue = BatchQueue(batch_size=5)

        results = queue.flush()

        assert results == []


class TestBatchingEdgeCases:
    """Test edge cases and special scenarios."""

    def test_create_batches_large_batch_size(self):
        """Test with batch size larger than items."""
        items = [1, 2, 3]
        batches = list(create_batches(items, batch_size=100))

        assert len(batches) == 1
        assert batches[0] == items

    def test_pad_batch_empty_sequences(self):
        """Test padding with empty sequences."""
        sequences = [mx.array([]), mx.array([1, 2])]

        padded = pad_batch(sequences, padding_value=0)

        # Empty sequence should be all padding
        assert padded.shape == (2, 2)

    def test_dynamic_batching_zero_max_tokens(self):
        """Test dynamic batching with very small max_batch_tokens."""
        items = ["a", "b", "c"]

        batches = list(
            dynamic_batching(
                items,
                get_size_fn=lambda x: len(x),
                max_batch_tokens=1,
            )
        )

        # Each item should be in its own batch
        assert len(batches) == 3

    def test_batch_queue_single_item(self):
        """Test queue with single item that doesn't fill batch."""
        queue = BatchQueue(batch_size=10)

        queue.add(1)
        results = queue.flush()

        assert results == [1]


class TestBatchingIntegration:
    """Test integration of batching utilities."""

    def test_batching_workflow(self):
        """Test complete batching workflow."""
        # Create data
        data = list(range(100))

        # Create batches
        batches = list(create_batches(data, batch_size=32))

        # Process each batch (example)
        processed = []
        for batch in batches:
            # Simulate processing
            processed.extend([x * 2 for x in batch])

        assert len(processed) == 100
        assert processed[0] == 0
        assert processed[-1] == 198

    def test_pad_and_batch_workflow(self):
        """Test padding and batching together."""
        # Variable length sequences
        sequences = [
            [1, 2],
            [3, 4, 5],
            [6],
            [7, 8, 9, 10],
        ]

        # Convert to arrays
        mx_sequences = [mx.array(seq) for seq in sequences]

        # Pad to same length
        padded = pad_batch(mx_sequences, padding_value=0)

        # Should all have length 4
        assert padded.shape[1] == 4
        assert padded.shape[0] == 4
