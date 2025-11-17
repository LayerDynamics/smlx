"""
Tests for smlx.data.datasets module.

Tests dataset classes for text, chat, completions, VLM, and audio.
"""

import pytest

from smlx.data.datasets import (
    AudioDataset,
    CacheDataset,
    ChatDataset,
    CompletionsDataset,
    ConcatenatedDataset,
    SubsetDataset,
    TextDataset,
    VisionLanguageDataset,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 2

    def encode(self, text):
        """Simple tokenization: split on spaces."""
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


class TestTextDataset:
    """Tests for TextDataset."""

    def test_text_dataset_creation(self, mock_tokenizer):
        """Test creating a text dataset."""
        data = [{"text": "Hello world"}, {"text": "How are you?"}]

        dataset = TextDataset(data, mock_tokenizer)

        assert len(dataset) == 2
        assert dataset[0] == {"text": "Hello world"}

    def test_text_dataset_process(self, mock_tokenizer):
        """Test processing text dataset items."""
        data = [{"text": "Hello world"}]
        dataset = TextDataset(data, mock_tokenizer)

        tokens, offset = dataset.process(dataset[0])

        assert isinstance(tokens, list)
        assert offset == 0
        assert tokens[-1] == mock_tokenizer.eos_token_id

    def test_text_dataset_custom_key(self, mock_tokenizer):
        """Test text dataset with custom key."""
        data = [{"content": "Hello world"}]
        dataset = TextDataset(data, mock_tokenizer, text_key="content")

        tokens, offset = dataset.process(dataset[0])
        assert len(tokens) > 0


class TestChatDataset:
    """Tests for ChatDataset."""

    def test_chat_dataset_creation(self, mock_tokenizer):
        """Test creating a chat dataset."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            }
        ]

        dataset = ChatDataset(data, mock_tokenizer)

        assert len(dataset) == 1

    def test_chat_dataset_process(self, mock_tokenizer):
        """Test processing chat dataset items."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        ]
        dataset = ChatDataset(data, mock_tokenizer, mask_prompt=False)

        tokens, offset = dataset.process(dataset[0])

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert offset == 0

    def test_chat_dataset_mask_prompt(self, mock_tokenizer):
        """Test chat dataset with prompt masking."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        ]
        dataset = ChatDataset(data, mock_tokenizer, mask_prompt=True)

        tokens, offset = dataset.process(dataset[0])

        # With masking, offset should be > 0
        assert offset >= 0


class TestCompletionsDataset:
    """Tests for CompletionsDataset."""

    def test_completions_dataset_creation(self, mock_tokenizer):
        """Test creating a completions dataset."""
        data = [
            {"prompt": "Translate to French: Hello", "completion": "Bonjour"},
            {"prompt": "What is 2+2?", "completion": "4"},
        ]

        dataset = CompletionsDataset(data, mock_tokenizer, mask_prompt=True)

        assert len(dataset) == 2

    def test_completions_dataset_process(self, mock_tokenizer):
        """Test processing completions dataset items."""
        data = [{"prompt": "Hello", "completion": "Hi"}]
        dataset = CompletionsDataset(data, mock_tokenizer, mask_prompt=False)

        tokens, offset = dataset.process(dataset[0])

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_completions_custom_keys(self, mock_tokenizer):
        """Test completions dataset with custom keys."""
        data = [{"input": "Question", "output": "Answer"}]
        dataset = CompletionsDataset(
            data, mock_tokenizer, prompt_key="input", completion_key="output"
        )

        tokens, offset = dataset.process(dataset[0])
        assert len(tokens) > 0


class TestVisionLanguageDataset:
    """Tests for VisionLanguageDataset."""

    def test_vlm_dataset_creation(self, mock_tokenizer):
        """Test creating a VLM dataset."""
        # Mock data with simple image paths
        data = [{"image": "test.jpg", "text": "A test image"}]

        dataset = VisionLanguageDataset(data, mock_tokenizer)

        assert len(dataset) == 1
        assert dataset[0]["image"] == "test.jpg"

    def test_vlm_dataset_with_qa(self, mock_tokenizer):
        """Test VLM dataset with question-answer format."""
        data = [{"image": "cat.jpg", "question": "What animal?", "answer": "A cat"}]

        dataset = VisionLanguageDataset(data, mock_tokenizer)

        assert len(dataset) == 1


class TestAudioDataset:
    """Tests for AudioDataset."""

    def test_audio_dataset_creation(self):
        """Test creating an audio dataset."""
        data = [
            {"audio": "speech1.wav", "text": "Hello world"},
            {"audio": "speech2.wav", "text": "How are you?"},
        ]

        dataset = AudioDataset(data)

        assert len(dataset) == 2
        assert dataset[0]["audio"] == "speech1.wav"


class TestConcatenatedDataset:
    """Tests for ConcatenatedDataset."""

    def test_concatenated_dataset(self, mock_tokenizer):
        """Test concatenating multiple datasets."""
        data1 = [{"text": "Hello"}, {"text": "World"}]
        data2 = [{"text": "Foo"}, {"text": "Bar"}]

        dataset1 = TextDataset(data1, mock_tokenizer)
        dataset2 = TextDataset(data2, mock_tokenizer)

        combined = ConcatenatedDataset([dataset1, dataset2])

        assert len(combined) == 4
        assert combined[0] == {"text": "Hello", "_dataset": 0}
        assert combined[2] == {"text": "Foo", "_dataset": 1}

    def test_concatenated_empty(self):
        """Test concatenating empty datasets."""
        combined = ConcatenatedDataset([])
        assert len(combined) == 0


class TestCacheDataset:
    """Tests for CacheDataset."""

    def test_cache_dataset(self, mock_tokenizer):
        """Test dataset caching."""
        data = [{"text": "Hello world"}]
        base_dataset = TextDataset(data, mock_tokenizer)
        cached = CacheDataset(base_dataset)

        # First access - processes and caches
        item1 = cached[0]
        # Second access - retrieves from cache
        item2 = cached[0]

        assert item1 == item2
        assert len(cached) == 1


class TestSubsetDataset:
    """Tests for SubsetDataset."""

    def test_subset_by_indices(self, mock_tokenizer):
        """Test creating subset by indices."""
        data = [{"text": f"Text {i}"} for i in range(10)]
        dataset = TextDataset(data, mock_tokenizer)

        subset = SubsetDataset(dataset, indices=[0, 2, 4])

        assert len(subset) == 3
        assert subset[0]["text"] == "Text 0"
        assert subset[1]["text"] == "Text 2"

    def test_subset_by_percentage(self, mock_tokenizer):
        """Test creating subset by percentage."""
        data = [{"text": f"Text {i}"} for i in range(100)]
        dataset = TextDataset(data, mock_tokenizer)

        subset = SubsetDataset(dataset, percentage=10)

        assert len(subset) == 10

    def test_subset_invalid_args(self, mock_tokenizer):
        """Test that providing both indices and percentage raises error."""
        data = [{"text": "Hello"}]
        dataset = TextDataset(data, mock_tokenizer)

        with pytest.raises(ValueError, match="Cannot specify both"):
            SubsetDataset(dataset, indices=[0], percentage=50)

    def test_subset_no_args(self, mock_tokenizer):
        """Test that providing neither indices nor percentage raises error."""
        data = [{"text": "Hello"}]
        dataset = TextDataset(data, mock_tokenizer)

        with pytest.raises(ValueError, match="Must specify either"):
            SubsetDataset(dataset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
