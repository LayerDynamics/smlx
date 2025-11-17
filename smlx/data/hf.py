"""
HuggingFace datasets integration for SMLX.

This module provides utilities for loading and working with HuggingFace datasets,
adapted from MLX-LM patterns.

Adapted from:
- resources/mlx-lm/mlx_lm/tuner/datasets.py (HF dataset loading)
- smlx/tools/download_data.py (dataset downloading)
"""

import json
from pathlib import Path
from typing import Any, Optional

from transformers import PreTrainedTokenizer

from .datasets import ChatDataset, CompletionsDataset, TextDataset


def create_dataset(
    data: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    text_key: str = "text",
    prompt_key: str = "prompt",
    completion_key: str = "completion",
    chat_key: str = "messages",
    mask_prompt: bool = False,
):
    """
    Automatically create appropriate dataset based on data format.

    Detects format from the first sample and creates the right dataset type.

    Args:
        data: List of data dictionaries
        tokenizer: HuggingFace tokenizer
        text_key: Key for plain text field (default: "text")
        prompt_key: Key for prompt field (default: "prompt")
        completion_key: Key for completion field (default: "completion")
        chat_key: Key for messages field (default: "messages")
        mask_prompt: Whether to mask prompt during training (default: False)

    Returns:
        Appropriate dataset instance

    Raises:
        ValueError: If data format is not recognized

    Example:
        >>> data = [{"text": "Hello world"}, {"text": "How are you?"}]
        >>> dataset = create_dataset(data, tokenizer)
        >>> # Returns TextDataset
    """
    if not data:
        raise ValueError("Cannot create dataset from empty data")

    sample = data[0]

    # Check for prompt-completion format
    if prompt_key in sample and completion_key in sample:
        return CompletionsDataset(
            data, tokenizer, prompt_key, completion_key, mask_prompt
        )

    # Check for chat format
    elif chat_key in sample:
        return ChatDataset(data, tokenizer, chat_key=chat_key, mask_prompt=mask_prompt)

    # Check for plain text format
    elif text_key in sample:
        if mask_prompt:
            raise ValueError("Prompt masking not supported for text dataset.")
        return TextDataset(data, tokenizer, text_key=text_key)

    else:
        raise ValueError(
            f"Unsupported data format. Data must contain one of: "
            f"'{text_key}', '{prompt_key}'+'{completion_key}', or '{chat_key}'. "
            f"Found keys: {list(sample.keys())}"
        )


def load_local_dataset(
    data_path: Path,
    tokenizer: PreTrainedTokenizer,
    text_key: str = "text",
    prompt_key: str = "prompt",
    completion_key: str = "completion",
    chat_key: str = "messages",
    mask_prompt: bool = False,
) -> tuple[Any, Any, Any]:
    """
    Load dataset from local JSONL files.

    Expects files named train.jsonl, valid.jsonl, test.jsonl in data_path.

    Args:
        data_path: Directory containing JSONL files
        tokenizer: HuggingFace tokenizer
        text_key: Key for text field
        prompt_key: Key for prompt field
        completion_key: Key for completion field
        chat_key: Key for messages field
        mask_prompt: Whether to mask prompt during training

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)

    Example:
        >>> from pathlib import Path
        >>> train, valid, test = load_local_dataset(
        ...     Path("./data/my_dataset"),
        ...     tokenizer
        ... )
    """

    def load_subset(path: Path):
        if not path.exists():
            return []
        with open(path, "r") as fid:
            data = [json.loads(line) for line in fid]
        return create_dataset(
            data, tokenizer, text_key, prompt_key, completion_key, chat_key, mask_prompt
        )

    names = ("train", "valid", "test")
    train, valid, test = [load_subset(data_path / f"{n}.jsonl") for n in names]
    return train, valid, test


def load_hf_dataset(
    dataset_id: str,
    tokenizer: PreTrainedTokenizer,
    split: Optional[str] = None,
    text_key: str = "text",
    prompt_key: str = "prompt",
    completion_key: str = "completion",
    chat_key: str = "messages",
    mask_prompt: bool = False,
) -> tuple[Any, Any, Any]:
    """
    Load dataset from HuggingFace Hub.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "HuggingFaceTB/smoltalk")
        tokenizer: HuggingFace tokenizer
        split: Optional split specification (e.g., "train[:10%]")
        text_key: Key for text field
        prompt_key: Key for prompt field
        completion_key: Key for completion field
        chat_key: Key for messages field
        mask_prompt: Whether to mask prompt during training

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)

    Raises:
        ValueError: If dataset not found

    Example:
        >>> train, valid, test = load_hf_dataset(
        ...     "HuggingFaceTB/smoltalk",
        ...     tokenizer
        ... )
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not installed. Install with: pip install datasets"
        )

    try:
        # Load dataset
        if split is not None:
            dataset = load_dataset(dataset_id, split=split)
            # Convert to dict format expected by create_dataset
            dataset = {"train": dataset}
        else:
            dataset = load_dataset(dataset_id)

        names = ("train", "valid", "test")

        train, valid, test = [
            (
                create_dataset(
                    list(dataset[n]),
                    tokenizer,
                    text_key,
                    prompt_key,
                    completion_key,
                    chat_key,
                    mask_prompt,
                )
                if n in dataset.keys()
                else []
            )
            for n in names
        ]

    except Exception as e:
        raise ValueError(f"Failed to load HuggingFace dataset: {dataset_id}. Error: {e}")

    return train, valid, test


def download_from_hub(
    repo_id: str,
    cache_dir: Optional[Path] = None,
    revision: Optional[str] = None,
) -> Path:
    """
    Download a dataset from HuggingFace Hub to local cache.

    Args:
        repo_id: Repository ID (e.g., "username/dataset-name")
        cache_dir: Local cache directory (default: HF default cache)
        revision: Specific revision/branch (default: main)

    Returns:
        Path to downloaded dataset

    Example:
        >>> path = download_from_hub("HuggingFaceTB/smoltalk")
        >>> print(f"Downloaded to: {path}")
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )

    # Download dataset
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
        revision=revision,
    )

    return Path(local_path)


def save_dataset_to_jsonl(
    dataset: Any,
    output_path: Path,
    num_samples: Optional[int] = None,
) -> None:
    """
    Save dataset to JSONL format.

    Args:
        dataset: Dataset to save
        output_path: Output file path
        num_samples: Number of samples to save (default: all)

    Example:
        >>> save_dataset_to_jsonl(dataset, Path("train.jsonl"), num_samples=1000)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = num_samples or len(dataset)

    with open(output_path, "w") as f:
        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]
            f.write(json.dumps(item) + "\n")


def load_dataset_splits(
    data_source: str,
    tokenizer: PreTrainedTokenizer,
    train_split: str = "train[:80%]",
    valid_split: str = "train[80%:90%]",
    test_split: str = "train[90%:]",
    **kwargs,
) -> tuple[Any, Any, Any]:
    """
    Load dataset with custom split specifications.

    Args:
        data_source: Either a local path or HuggingFace dataset ID
        tokenizer: HuggingFace tokenizer
        train_split: Train split specification
        valid_split: Validation split specification
        test_split: Test split specification
        **kwargs: Additional arguments passed to create_dataset

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)

    Example:
        >>> train, valid, test = load_dataset_splits(
        ...     "HuggingFaceTB/smoltalk",
        ...     tokenizer,
        ...     train_split="train[:90%]",
        ...     valid_split="train[90%:]",
        ...     test_split=None
        ... )
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not installed. Install with: pip install datasets"
        )

    # Check if local path or HF dataset
    path = Path(data_source)
    if path.exists():
        # Local dataset
        return load_local_dataset(path, tokenizer, **kwargs)

    # HuggingFace dataset
    try:
        splits = {}

        if train_split:
            train_data = load_dataset(data_source, split=train_split)
            splits["train"] = create_dataset(list(train_data), tokenizer, **kwargs)
        else:
            splits["train"] = []

        if valid_split:
            valid_data = load_dataset(data_source, split=valid_split)
            splits["valid"] = create_dataset(list(valid_data), tokenizer, **kwargs)
        else:
            splits["valid"] = []

        if test_split:
            test_data = load_dataset(data_source, split=test_split)
            splits["test"] = create_dataset(list(test_data), tokenizer, **kwargs)
        else:
            splits["test"] = []

        return splits["train"], splits["valid"], splits["test"]

    except Exception as e:
        raise ValueError(f"Failed to load dataset from {data_source}. Error: {e}")
