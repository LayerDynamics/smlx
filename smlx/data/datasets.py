"""
Dataset classes for SMLX - Text, chat, vision-language, and audio datasets.

This module provides lightweight dataset classes for different modalities and tasks,
following the patterns from MLX-LM for consistency with the MLX ecosystem.

Adapted from:
- resources/mlx-lm/mlx_lm/tuner/datasets.py (text/chat datasets)
- resources/mlx-vlm/mlx_vlm/trainer/trainer.py (VLM datasets)
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from transformers import PreTrainedTokenizer

from .loaders import load_audio, load_image


class BaseDataset(Protocol):
    """Protocol for dataset classes."""

    def __len__(self) -> int:
        """Return dataset length."""
        ...

    def __getitem__(self, idx: int) -> Any:
        """Get item at index."""
        ...

    def process(self, d: Any) -> Any:
        """Process a raw sample (e.g. tokenize) into the model-ready form."""
        ...


class TextDataset:
    """
    Dataset for plain text data.

    Each sample is a dictionary with a text field that gets tokenized.
    Suitable for language modeling or text generation tasks.

    Args:
        data: List of dictionaries containing text
        tokenizer: HuggingFace tokenizer
        text_key: Key for text field in data (default: "text")

    Example:
        >>> data = [{"text": "Hello world"}, {"text": "How are you?"}]
        >>> dataset = TextDataset(data, tokenizer, text_key="text")
        >>> tokens, offset = dataset.process(dataset[0])
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        text_key: str = "text",
    ):
        self._data = data
        self.tokenizer = tokenizer
        self.text_key = text_key

    def process(self, d: Dict[str, str]) -> tuple[List[int], int]:
        """
        Process a data sample by tokenizing text.

        Args:
            d: Data dictionary

        Returns:
            Tuple of (token_ids, offset) where offset is always 0 for text dataset
        """
        tokens = self.tokenizer.encode(d[self.text_key])
        if tokens[-1] != self.tokenizer.eos_token_id:
            tokens.append(self.tokenizer.eos_token_id)
        return (tokens, 0)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


class ChatDataset:
    """
    Dataset for chat/conversation data.

    Follows OpenAI format: {"messages": [{"role": "user", "content": "..."}, ...]}
    Supports prompt masking for training only on assistant responses.

    Args:
        data: List of dictionaries containing messages
        tokenizer: HuggingFace tokenizer
        chat_key: Key for messages field (default: "messages")
        mask_prompt: If True, only compute loss on assistant responses (default: False)

    Example:
        >>> data = [{
        ...     "messages": [
        ...         {"role": "user", "content": "Hello"},
        ...         {"role": "assistant", "content": "Hi there!"}
        ...     ]
        ... }]
        >>> dataset = ChatDataset(data, tokenizer, mask_prompt=True)
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        chat_key: str = "messages",
        mask_prompt: bool = False,
    ):
        self._data = data
        self.chat_key = chat_key
        self.mask_prompt = mask_prompt
        self.tokenizer = tokenizer

    def process(self, d: Dict[str, Any]) -> tuple[List[int], int]:
        """
        Process a chat sample by applying chat template.

        Args:
            d: Data dictionary with messages

        Returns:
            Tuple of (token_ids, offset) where offset indicates where to start loss computation
        """
        messages = d[self.chat_key]
        tools = d.get("tools", None)
        tokens = self.tokenizer.apply_chat_template(messages, tools=tools)

        if self.mask_prompt:
            # Only compute loss on assistant response
            add_generation_prompt = messages[-1].get("role") == "assistant"
            offset = len(
                self.tokenizer.apply_chat_template(
                    messages[:-1],
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                )
            )
            return (tokens, offset)
        else:
            return (tokens, 0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


class CompletionsDataset:
    """
    Dataset for prompt-completion pairs.

    Format: {"prompt": "...", "completion": "..."} or custom keys.
    Suitable for supervised fine-tuning tasks.

    Args:
        data: List of dictionaries with prompts and completions
        tokenizer: HuggingFace tokenizer
        prompt_key: Key for prompt field (default: "prompt")
        completion_key: Key for completion field (default: "completion")
        mask_prompt: If True, only compute loss on completion (default: True)

    Example:
        >>> data = [
        ...     {"prompt": "Translate to French: Hello", "completion": "Bonjour"},
        ...     {"input": "2+2=", "output": "4"}
        ... ]
        >>> dataset = CompletionsDataset(
        ...     data, tokenizer, prompt_key="input", completion_key="output"
        ... )
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        completion_key: str = "completion",
        mask_prompt: bool = True,
    ):
        self._data = data
        self.prompt_key = prompt_key
        self.completion_key = completion_key
        self.mask_prompt = mask_prompt
        self.tokenizer = tokenizer

    def process(self, d: Dict[str, str]) -> tuple[List[int], int]:
        """
        Process a prompt-completion pair by converting to chat format.

        Args:
            d: Data dictionary with prompt and completion

        Returns:
            Tuple of (token_ids, offset)
        """
        tools = d.get("tools", None)
        messages = [
            {"role": "user", "content": d[self.prompt_key]},
            {"role": "assistant", "content": d[self.completion_key]},
        ]
        tokens = self.tokenizer.apply_chat_template(messages, tools=tools)

        if self.mask_prompt:
            offset = len(
                self.tokenizer.apply_chat_template(
                    [messages[0]], tools=tools, add_generation_prompt=True
                )
            )
            return (tokens, offset)

        return (tokens, 0)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


class VisionLanguageDataset:
    """
    Dataset for vision-language tasks (VQA, captioning, etc.).

    Each sample contains an image path/URL and associated text (question, caption, etc.).

    Args:
        data: List of dictionaries with image and text fields
        tokenizer: HuggingFace tokenizer
        image_processor: Image processor or preprocessing function
        image_key: Key for image field (default: "image")
        text_key: Key for text field (default: "text")
        question_key: Key for question field (default: "question")
        answer_key: Key for answer field (default: "answer")

    Example:
        >>> data = [
        ...     {"image": "cat.jpg", "question": "What animal?", "answer": "A cat"},
        ...     {"image": "https://example.com/dog.jpg", "text": "A dog playing"}
        ... ]
        >>> dataset = VisionLanguageDataset(data, tokenizer, image_processor)
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        image_processor: Optional[Callable] = None,
        image_key: str = "image",
        text_key: str = "text",
        question_key: str = "question",
        answer_key: str = "answer",
    ):
        self._data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_key = image_key
        self.text_key = text_key
        self.question_key = question_key
        self.answer_key = answer_key

    def process(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a vision-language sample.

        Args:
            d: Data dictionary

        Returns:
            Dictionary with processed image and text
        """
        # Load image
        image = load_image(d[self.image_key])

        # Process image if processor provided
        if self.image_processor is not None:
            pixel_values = self.image_processor(image)
        else:
            pixel_values = image

        # Handle different text formats
        if self.question_key in d and self.answer_key in d:
            # Q&A format
            messages = [
                {"role": "user", "content": d[self.question_key]},
                {"role": "assistant", "content": d[self.answer_key]},
            ]
            tokens = self.tokenizer.apply_chat_template(messages)
        elif self.text_key in d:
            # Plain text format (e.g., caption)
            tokens = self.tokenizer.encode(d[self.text_key])
        else:
            raise ValueError(
                f"Data must contain either '{self.text_key}' or "
                f"'{self.question_key}' and '{self.answer_key}'"
            )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens,
            "image": image,  # Keep original for visualization
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


class AudioDataset:
    """
    Dataset for audio data (speech recognition, audio classification, etc.).

    Args:
        data: List of dictionaries with audio paths and optional transcriptions
        audio_key: Key for audio field (default: "audio")
        text_key: Key for text/transcription field (default: "text")
        sample_rate: Target sample rate (default: 16000)

    Example:
        >>> data = [
        ...     {"audio": "speech1.wav", "text": "Hello world"},
        ...     {"audio": "speech2.mp3", "text": "How are you?"}
        ... ]
        >>> dataset = AudioDataset(data, sample_rate=16000)
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        audio_key: str = "audio",
        text_key: str = "text",
        sample_rate: int = 16000,
    ):
        self._data = data
        self.audio_key = audio_key
        self.text_key = text_key
        self.sample_rate = sample_rate

    def process(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an audio sample.

        Args:
            d: Data dictionary

        Returns:
            Dictionary with audio array and optional text
        """
        # Load audio
        audio = load_audio(d[self.audio_key], sr=self.sample_rate)

        result = {"audio": audio}

        # Add text if available
        if self.text_key in d:
            result["text"] = d[self.text_key]

        return result

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


class ConcatenatedDataset:
    """
    Concatenate multiple datasets into one.

    Useful for combining multiple data sources or domains.

    Args:
        datasets: List of dataset objects to concatenate

    Example:
        >>> dataset1 = TextDataset(data1, tokenizer)
        >>> dataset2 = TextDataset(data2, tokenizer)
        >>> combined = ConcatenatedDataset([dataset1, dataset2])
        >>> len(combined)  # sum of individual lengths
    """

    def __init__(self, datasets: List[BaseDataset]):
        self._datasets = datasets
        self._len = sum(len(d) for d in datasets)

    def __getitem__(self, idx: int) -> Any:
        """Get item by finding which dataset it belongs to."""
        for dataset_idx, dataset in enumerate(self._datasets):
            j = idx - len(dataset)
            if j < 0:
                break
            idx = j

        datum = dataset[idx]

        # Track which dataset this came from
        if isinstance(datum, dict):
            datum["_dataset"] = dataset_idx

        return datum

    def process(self, d: Any) -> Any:
        """Process item using the appropriate dataset's process method."""
        if isinstance(d, dict) and "_dataset" in d:
            dataset_idx = d["_dataset"]
            return self._datasets[dataset_idx].process(d)
        else:
            # If no dataset marker, try first dataset
            return self._datasets[0].process(d)

    def __len__(self) -> int:
        return self._len


class CacheDataset:
    """
    Wrapper to cache processed dataset items in memory.

    Useful for expensive preprocessing operations that should only happen once.

    Args:
        dataset: Dataset to wrap with caching

    Example:
        >>> base_dataset = VisionLanguageDataset(data, tokenizer, image_processor)
        >>> cached_dataset = CacheDataset(base_dataset)
        >>> # First access processes and caches
        >>> item1 = cached_dataset[0]
        >>> # Second access retrieves from cache
        >>> item2 = cached_dataset[0]  # Fast!
    """

    def __init__(self, dataset: BaseDataset):
        self._dataset = dataset
        self._cache = [None] * len(dataset)

    def __getitem__(self, idx: int) -> Any:
        """Get item from cache or process and cache it."""
        if self._cache[idx] is None:
            self._cache[idx] = self._dataset.process(self._dataset[idx])
        return self._cache[idx]

    def __len__(self) -> int:
        return len(self._dataset)


class SubsetDataset:
    """
    Create a subset of a dataset by indices or percentage.

    Args:
        dataset: Base dataset
        indices: List of indices to include (mutually exclusive with percentage)
        percentage: Percentage of data to include (0-100, mutually exclusive with indices)
        shuffle: Whether to shuffle before taking subset (only used with percentage)

    Example:
        >>> dataset = TextDataset(data, tokenizer)
        >>> # Take first 100 samples
        >>> subset = SubsetDataset(dataset, indices=list(range(100)))
        >>> # Take 10% of data
        >>> subset = SubsetDataset(dataset, percentage=10, shuffle=True)
    """

    def __init__(
        self,
        dataset: BaseDataset,
        indices: Optional[List[int]] = None,
        percentage: Optional[float] = None,
        shuffle: bool = False,
    ):
        if indices is not None and percentage is not None:
            raise ValueError("Cannot specify both indices and percentage")

        self._dataset = dataset

        if indices is not None:
            self._indices = indices
        elif percentage is not None:
            if not 0 < percentage <= 100:
                raise ValueError("Percentage must be between 0 and 100")

            import random

            num_samples = int(len(dataset) * percentage / 100)
            all_indices = list(range(len(dataset)))

            if shuffle:
                random.shuffle(all_indices)

            self._indices = all_indices[:num_samples]
        else:
            raise ValueError("Must specify either indices or percentage")

    def __getitem__(self, idx: int) -> Any:
        """Get item at logical index."""
        actual_idx = self._indices[idx]
        return self._dataset[actual_idx]

    def __len__(self) -> int:
        return len(self._indices)

    def process(self, d: Any) -> Any:
        """Process using base dataset's method."""
        return self._dataset.process(d)
