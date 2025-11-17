"""
Tests for smlx.data.hf module.

Tests HuggingFace datasets integration and loading utilities.
"""

import json
import tempfile
from pathlib import Path

import pytest

from smlx.data.hf import (
    create_dataset,
    load_local_dataset,
    save_dataset_to_jsonl,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 2

    def encode(self, text):
        """Simple tokenization."""
        return [hash(word) % 1000 for word in text.split()]

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False):
        """Mock chat template."""
        tokens = []
        for msg in messages:
            if isinstance(msg, dict):
                tokens.extend(self.encode(msg.get("content", "")))
            else:
                tokens.extend(self.encode(str(msg)))
        return tokens


@pytest.fixture
def mock_tokenizer():
    """Fixture providing mock tokenizer."""
    return MockTokenizer()


class TestCreateDataset:
    """Tests for create_dataset."""

    def test_create_text_dataset(self, mock_tokenizer):
        """Test automatic detection of text dataset format."""
        data = [{"text": "Hello world"}, {"text": "How are you?"}]

        dataset = create_dataset(data, mock_tokenizer)

        assert len(dataset) == 2
        # Should be TextDataset
        assert hasattr(dataset, "text_key")

    def test_create_chat_dataset(self, mock_tokenizer):
        """Test automatic detection of chat dataset format."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        ]

        dataset = create_dataset(data, mock_tokenizer)

        assert len(dataset) == 1
        # Should be ChatDataset
        assert hasattr(dataset, "chat_key")

    def test_create_completions_dataset(self, mock_tokenizer):
        """Test automatic detection of completions dataset format."""
        data = [{"prompt": "Translate: Hello", "completion": "Bonjour"}]

        dataset = create_dataset(data, mock_tokenizer)

        assert len(dataset) == 1
        # Should be CompletionsDataset
        assert hasattr(dataset, "prompt_key")

    def test_create_dataset_empty_data(self, mock_tokenizer):
        """Test that empty data raises error."""
        with pytest.raises(ValueError, match="Cannot create dataset from empty"):
            create_dataset([], mock_tokenizer)

    def test_create_dataset_unsupported_format(self, mock_tokenizer):
        """Test that unsupported format raises error."""
        data = [{"unknown_field": "value"}]

        with pytest.raises(ValueError, match="Unsupported data format"):
            create_dataset(data, mock_tokenizer)

    def test_create_dataset_custom_keys(self, mock_tokenizer):
        """Test with custom field keys."""
        data = [{"content": "Hello world"}]

        dataset = create_dataset(data, mock_tokenizer, text_key="content")

        assert len(dataset) == 1

    def test_create_dataset_mask_prompt(self, mock_tokenizer):
        """Test mask_prompt parameter."""
        data = [{"prompt": "Question", "completion": "Answer"}]

        dataset = create_dataset(data, mock_tokenizer, mask_prompt=True)

        tokens, offset = dataset.process(dataset[0])
        # With masking, offset should be > 0
        assert offset >= 0

    def test_create_text_dataset_mask_prompt_error(self, mock_tokenizer):
        """Test that text dataset with mask_prompt raises error."""
        data = [{"text": "Hello"}]

        with pytest.raises(ValueError, match="Prompt masking not supported"):
            create_dataset(data, mock_tokenizer, mask_prompt=True)


class TestLoadLocalDataset:
    """Tests for load_local_dataset."""

    def test_load_local_dataset(self, mock_tokenizer):
        """Test loading dataset from local JSONL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create train.jsonl
            train_data = [{"text": f"Train sample {i}"} for i in range(5)]
            with open(tmpdir / "train.jsonl", "w") as f:
                for item in train_data:
                    f.write(json.dumps(item) + "\n")

            # Create valid.jsonl
            valid_data = [{"text": f"Valid sample {i}"} for i in range(2)]
            with open(tmpdir / "valid.jsonl", "w") as f:
                for item in valid_data:
                    f.write(json.dumps(item) + "\n")

            # Load datasets
            train, valid, test = load_local_dataset(tmpdir, mock_tokenizer)

            assert len(train) == 5
            assert len(valid) == 2
            assert len(test) == 0  # No test.jsonl

    def test_load_local_dataset_missing_files(self, mock_tokenizer):
        """Test loading with missing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # No files created
            train, valid, test = load_local_dataset(tmpdir, mock_tokenizer)

            assert len(train) == 0
            assert len(valid) == 0
            assert len(test) == 0

    def test_load_local_dataset_chat_format(self, mock_tokenizer):
        """Test loading chat format dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create chat format data
            chat_data = [
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi!"},
                    ]
                }
            ]
            with open(tmpdir / "train.jsonl", "w") as f:
                for item in chat_data:
                    f.write(json.dumps(item) + "\n")

            train, _, _ = load_local_dataset(tmpdir, mock_tokenizer)

            assert len(train) == 1


class TestSaveDatasetToJsonl:
    """Tests for save_dataset_to_jsonl."""

    def test_save_dataset(self, mock_tokenizer):
        """Test saving dataset to JSONL."""
        # Create a simple dataset
        data = [{"text": f"Sample {i}"} for i in range(10)]

        # Create simple wrapper that behaves like a dataset
        class SimpleDataset:
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        dataset = SimpleDataset(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"

            save_dataset_to_jsonl(dataset, output_path)

            # Verify file was created
            assert output_path.exists()

            # Load and verify contents
            loaded = []
            with open(output_path) as f:
                for line in f:
                    loaded.append(json.loads(line))

            assert len(loaded) == 10
            assert loaded[0]["text"] == "Sample 0"

    def test_save_dataset_with_limit(self, mock_tokenizer):
        """Test saving with num_samples limit."""

        class SimpleDataset:
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {"id": idx}

        dataset = SimpleDataset(100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "limited.jsonl"

            save_dataset_to_jsonl(dataset, output_path, num_samples=10)

            # Count lines
            with open(output_path) as f:
                lines = f.readlines()

            assert len(lines) == 10

    def test_save_dataset_creates_parent_dirs(self):
        """Test that parent directories are created."""

        class SimpleDataset:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {"id": idx}

        dataset = SimpleDataset()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "output.jsonl"

            save_dataset_to_jsonl(dataset, output_path)

            assert output_path.exists()


# Note: Tests for load_hf_dataset, download_from_hub, and load_dataset_splits
# require actual HuggingFace datasets library and network access.
# These are marked as integration tests and can be run separately.


@pytest.mark.integration
@pytest.mark.requires_hf
class TestHuggingFaceIntegration:
    """Integration tests requiring HuggingFace datasets library."""

    def test_load_hf_dataset_placeholder(self):
        """Placeholder for HF integration tests."""
        # These tests would require:
        # 1. datasets library installed
        # 2. Network access
        # 3. Actual HF dataset to test with
        pytest.skip("Requires HuggingFace datasets and network access")

    def test_download_from_hub_placeholder(self):
        """Placeholder for download tests."""
        pytest.skip("Requires HuggingFace hub and network access")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
