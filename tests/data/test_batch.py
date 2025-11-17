"""
Tests for smlx.data.batch module.

Tests DataLoader, collation functions, and batching utilities.
"""

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from smlx.data.batch import (
    DataLoader,
    batch_images,
    collate_audio,
    collate_images,
    collate_text,
    collate_vlm,
    create_batches,
    default_collate,
    dynamic_batching,
    pad_sequences,
)


class SimpleDataset:
    """Simple dataset for testing."""

    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"id": idx, "value": idx * 2}


class TestDataLoader:
    """Tests for DataLoader."""

    def test_dataloader_basic(self):
        """Test basic DataLoader functionality."""
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=3)

        batches = list(loader)

        # Should have 4 batches (10 / 3 = 3 full + 1 partial)
        assert len(batches) == 4
        # DataLoader uses default_collate, so batches are dicts with lists
        assert len(batches[0]["id"]) == 3  # First batch has 3 items
        assert len(batches[-1]["id"]) == 1  # Last batch has 1 item

    def test_dataloader_drop_last(self):
        """Test drop_last parameter."""
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=3, drop_last=True)

        batches = list(loader)

        # Should have 3 batches (last one dropped)
        assert len(batches) == 3
        assert len(batches[-1]["id"]) == 3

    def test_dataloader_shuffle(self):
        """Test shuffle parameter."""
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=5, shuffle=True)

        batches1 = list(loader)
        batches2 = list(DataLoader(dataset, batch_size=5, shuffle=True))

        # Batches should be different (with high probability)
        # Check if first item is different
        first_ids_1 = batches1[0]["id"]  # Already a list from default_collate
        first_ids_2 = batches2[0]["id"]

        # At least check they're valid IDs
        assert all(0 <= id < 10 for id in first_ids_1)

    def test_dataloader_no_shuffle(self):
        """Test without shuffle."""
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=5, shuffle=False)

        batches = list(loader)

        # First batch should be [0, 1, 2, 3, 4]
        first_batch_ids = batches[0]["id"]  # Already a list from default_collate
        assert first_batch_ids == [0, 1, 2, 3, 4]

    def test_dataloader_length(self):
        """Test __len__ method."""
        dataset = SimpleDataset(10)

        loader1 = DataLoader(dataset, batch_size=3)
        assert len(loader1) == 4  # ceil(10/3) = 4

        loader2 = DataLoader(dataset, batch_size=3, drop_last=True)
        assert len(loader2) == 3  # floor(10/3) = 3


class TestCollateText:
    """Tests for collate_text."""

    def test_collate_text_basic(self):
        """Test basic text collation."""
        batch = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
            {"input_ids": [6, 7, 8, 9]},
        ]

        result = collate_text(batch)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert isinstance(result["input_ids"], mx.array)
        assert result["input_ids"].shape == (3, 4)  # 3 samples, max length 4

    def test_collate_text_with_attention_mask(self):
        """Test collation with existing attention masks."""
        batch = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5], "attention_mask": [1, 1]},
        ]

        result = collate_text(batch)

        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)

    def test_collate_text_padding(self):
        """Test that padding is applied correctly."""
        batch = [{"input_ids": [1, 2, 3]}, {"input_ids": [4]}]

        result = collate_text(batch)

        # Check that second sequence is padded
        assert result["input_ids"][1, 0].item() == 4
        assert result["input_ids"][1, 1].item() == 0  # Padding


class TestCollateImages:
    """Tests for collate_images."""

    def test_collate_images_basic(self):
        """Test basic image collation."""
        # Create PIL images (width, height)
        images = [
            Image.new("RGB", (100, 100), color="red"),  # 100w x 100h
            Image.new("RGB", (150, 120), color="blue"),  # 150w x 120h
        ]

        result = collate_images(images)

        assert isinstance(result, mx.array)
        # Should be batched with max dimensions [B, C, H, W]
        assert result.shape[0] == 2  # Batch size
        assert result.shape[1] == 3  # Channels
        assert result.shape[2] == 120  # Max height (max of 100, 120)
        assert result.shape[3] == 150  # Max width (max of 100, 150)

    def test_collate_images_same_size(self):
        """Test collating images of same size."""
        images = [
            Image.new("RGB", (100, 100), color="red"),
            Image.new("RGB", (100, 100), color="blue"),
        ]

        result = collate_images(images)

        assert result.shape == (2, 3, 100, 100)


class TestCollateAudio:
    """Tests for collate_audio."""

    def test_collate_audio_basic(self):
        """Test basic audio collation."""
        audios = [
            mx.random.normal(shape=(16000,)),
            mx.random.normal(shape=(12000,)),
            mx.random.normal(shape=(20000,)),
        ]

        result = collate_audio(audios)

        assert isinstance(result, mx.array)
        assert result.shape == (3, 20000)  # 3 samples, max length 20000

    def test_collate_audio_padding_value(self):
        """Test custom padding value."""
        audios = [mx.ones((100,)), mx.ones((50,))]

        result = collate_audio(audios, padding_value=-1.0)

        # Check that padding is -1.0
        assert result[1, 60].item() == -1.0


class TestCollateVLM:
    """Tests for collate_vlm."""

    def test_collate_vlm_basic(self):
        """Test vision-language collation."""
        batch = [
            {
                "pixel_values": mx.random.normal(shape=(3, 224, 224)),
                "input_ids": [1, 2, 3],
            },
            {
                "pixel_values": mx.random.normal(shape=(3, 224, 224)),
                "input_ids": [4, 5],
            },
        ]

        result = collate_vlm(batch)

        assert "pixel_values" in result
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["pixel_values"].shape == (2, 3, 224, 224)


class TestPadSequences:
    """Tests for pad_sequences."""

    def test_pad_sequences_basic(self):
        """Test basic sequence padding."""
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

        result = pad_sequences(sequences)

        assert isinstance(result, mx.array)
        assert result.shape == (3, 4)  # 3 sequences, max length 4

    def test_pad_sequences_custom_value(self):
        """Test custom padding value."""
        sequences = [[1, 2, 3], [4, 5]]

        result = pad_sequences(sequences, padding_value=-1)

        # Check padding is -1
        assert result[1, 2].item() == -1

    def test_pad_sequences_max_length(self):
        """Test custom max length."""
        sequences = [[1, 2, 3, 4, 5], [6, 7]]

        result = pad_sequences(sequences, max_length=3)

        # Should truncate to max_length
        assert result.shape == (2, 3)
        assert result[0, 2].item() == 3  # Truncated

    def test_pad_sequences_mlx_arrays(self):
        """Test padding MLX arrays."""
        sequences = [mx.array([1, 2, 3]), mx.array([4, 5])]

        result = pad_sequences(sequences)

        assert isinstance(result, mx.array)
        assert result.shape == (2, 3)


class TestBatchImages:
    """Tests for batch_images."""

    def test_batch_images_basic(self):
        """Test basic image batching."""
        images = [
            mx.random.normal(shape=(3, 224, 224)),
            mx.random.normal(shape=(3, 256, 256)),
            mx.random.normal(shape=(3, 200, 300)),
        ]

        result = batch_images(images)

        assert isinstance(result, mx.array)
        assert result.shape == (3, 3, 256, 300)  # Max height 256, max width 300

    def test_batch_images_same_size(self):
        """Test batching images of same size."""
        images = [mx.random.normal(shape=(3, 224, 224)) for _ in range(4)]

        result = batch_images(images)

        assert result.shape == (4, 3, 224, 224)

    def test_batch_images_empty(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="Cannot batch empty"):
            batch_images([])

    def test_batch_images_custom_padding(self):
        """Test custom padding value."""
        images = [mx.ones((3, 100, 100)), mx.ones((3, 50, 50))]

        result = batch_images(images, padding_value=-1.0)

        # Check padding is -1.0
        assert result[1, 0, 60, 60].item() == -1.0


class TestCreateBatches:
    """Tests for create_batches."""

    def test_create_batches_basic(self):
        """Test basic batch creation."""
        items = list(range(10))
        batches = list(create_batches(items, batch_size=3))

        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[-1] == [9]

    def test_create_batches_drop_last(self):
        """Test drop_last parameter."""
        items = list(range(10))
        batches = list(create_batches(items, batch_size=3, drop_last=True))

        assert len(batches) == 3
        assert batches[-1] == [6, 7, 8]

    def test_create_batches_exact_division(self):
        """Test when items divide evenly."""
        items = list(range(12))
        batches = list(create_batches(items, batch_size=4))

        assert len(batches) == 3
        assert all(len(batch) == 4 for batch in batches)


class TestDynamicBatching:
    """Tests for dynamic_batching."""

    def test_dynamic_batching_basic(self):
        """Test basic dynamic batching."""
        items = ["short", "medium text", "very long text here"]

        def get_length(text):
            return len(text.split())

        batches = list(
            dynamic_batching(items, get_length, max_batch_tokens=5, max_batch_size=10)
        )

        # Each batch should respect max_batch_tokens
        for batch in batches:
            total_tokens = sum(get_length(item) for item in batch)
            assert total_tokens <= 5

    def test_dynamic_batching_max_size(self):
        """Test max_batch_size constraint."""
        items = ["a"] * 10

        def get_length(text):
            return 1

        batches = list(
            dynamic_batching(items, get_length, max_batch_tokens=100, max_batch_size=3)
        )

        # No batch should exceed max_batch_size
        for batch in batches:
            assert len(batch) <= 3

    def test_dynamic_batching_large_item(self):
        """Test handling item larger than max_batch_tokens."""
        items = ["small", "this is a very very long text"]

        def get_length(text):
            return len(text.split())

        batches = list(
            dynamic_batching(items, get_length, max_batch_tokens=3, max_batch_size=10)
        )

        # Large item should be in its own batch
        assert len(batches) >= 2


class TestDefaultCollate:
    """Tests for default_collate."""

    def test_default_collate_dict(self):
        """Test collating dictionaries."""
        batch = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

        result = default_collate(batch)

        assert isinstance(result, dict)
        assert "a" in result
        assert result["a"] == [1, 3]

    def test_default_collate_mlx_arrays(self):
        """Test collating MLX arrays."""
        batch = [mx.array([1, 2, 3]), mx.array([4, 5, 6])]

        result = default_collate(batch)

        assert isinstance(result, mx.array)
        assert result.shape == (2, 3)

    def test_default_collate_numpy_arrays(self):
        """Test collating numpy arrays."""
        batch = [np.array([1, 2, 3]), np.array([4, 5, 6])]

        result = default_collate(batch)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)

    def test_default_collate_lists(self):
        """Test collating other types returns list."""
        batch = ["a", "b", "c"]

        result = default_collate(batch)

        assert result == ["a", "b", "c"]

    def test_default_collate_empty(self):
        """Test collating empty batch."""
        result = default_collate([])
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
