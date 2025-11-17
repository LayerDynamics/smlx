"""
Pytest configuration and fixtures for evaluation tests.

Provides test models and datasets for integration testing.
"""

import os
from pathlib import Path

import pytest


def _check_downloads_enabled():
    """Check if we should attempt to download models/datasets for testing."""
    return os.getenv("SMLX_DOWNLOAD_TEST_MODELS", "0") == "1"


def _check_model_available(model_id: str) -> bool:
    """
    Check if a model is already cached locally.

    Args:
        model_id: HuggingFace model ID

    Returns:
        True if model exists in cache
    """
    from smlx.tools.download import get_cache_dir

    cache_dir = get_cache_dir() / "models"

    # HuggingFace hub uses specific directory structure
    # models--{org}--{name}
    model_dir_name = model_id.replace("/", "--")
    model_path = cache_dir / f"models--{model_dir_name}"

    return model_path.exists()


def _check_dataset_available(dataset_id: str) -> bool:
    """
    Check if a dataset is already cached locally.

    Args:
        dataset_id: HuggingFace dataset ID

    Returns:
        True if dataset exists in cache
    """
    from smlx.tools.download import get_cache_dir

    cache_dir = get_cache_dir() / "datasets"

    # HuggingFace hub uses specific directory structure
    dataset_dir_name = dataset_id.replace("/", "--")
    dataset_path = cache_dir / f"datasets--{dataset_dir_name}"

    return dataset_path.exists()


@pytest.fixture
def smolvlm_256m_model():
    """
    Load a small VLM model for testing (using LLaVA-1.5-7B-4bit as example).

    Requires SMLX_DOWNLOAD_TEST_MODELS=1 environment variable or existing cache.
    """
    # Use a small, quantized model that actually exists
    model_id = "mlx-community/llava-1.5-7b-4bit"

    # Check if model is already cached
    if _check_model_available(model_id):
        # Load from cache
        try:
            from smlx.evals.utils import load_model

            model, processor = load_model(model_id)
            return {"model": model, "processor": processor, "model_id": model_id}
        except Exception as e:
            pytest.skip(f"Model cached but failed to load: {e}")

    # Check if downloads are enabled
    if not _check_downloads_enabled():
        pytest.skip(
            f"Model {model_id} not available. "
            "Set SMLX_DOWNLOAD_TEST_MODELS=1 to enable downloads"
        )

    # Download model
    try:
        from smlx.tools.download import download_model

        print(f"\nDownloading model: {model_id}")
        download_model(
            model_id,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py", "*.model"],
        )

        # Load the downloaded model
        from smlx.evals.utils import load_model

        model, processor = load_model(model_id)
        return {"model": model, "processor": processor, "model_id": model_id}

    except Exception as e:
        pytest.skip(f"Failed to download/load model: {e}")


@pytest.fixture
def mathvista_dataset():
    """
    Load MathVista dataset for testing.

    Requires SMLX_DOWNLOAD_TEST_MODELS=1 environment variable or existing cache.
    """
    dataset_id = "AI4Math/MathVista"
    split = "testmini"

    # Try to load dataset (will use cache if available)
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_id, split=split)
        return {"dataset": dataset, "dataset_id": dataset_id, "split": split}

    except Exception as load_error:
        # Check if downloads are enabled
        if not _check_downloads_enabled():
            pytest.skip(
                f"Dataset {dataset_id} not available. "
                "Set SMLX_DOWNLOAD_TEST_MODELS=1 to enable downloads"
            )

        # Try to download
        try:
            from smlx.tools.download import download_dataset

            print(f"\nDownloading dataset: {dataset_id}")
            download_dataset(dataset_id)

            # Try loading again
            from datasets import load_dataset

            dataset = load_dataset(dataset_id, split=split)
            return {"dataset": dataset, "dataset_id": dataset_id, "split": split}

        except Exception as e:
            pytest.skip(f"Failed to download/load dataset: {e}")


@pytest.fixture
def mmmu_dataset():
    """
    Load MMMU dataset for testing.

    Requires SMLX_DOWNLOAD_TEST_MODELS=1 environment variable or existing cache.
    Loads the Math subject by default for faster testing.
    """
    dataset_id = "MMMU/MMMU"
    config_name = "Math"  # Load Math subject by default
    split = "validation"

    # Try to load dataset (will use cache if available)
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_id, config_name, split=split)
        return {
            "dataset": dataset,
            "dataset_id": dataset_id,
            "config": config_name,
            "split": split,
        }

    except Exception as load_error:
        # Check if downloads are enabled
        if not _check_downloads_enabled():
            pytest.skip(
                f"Dataset {dataset_id} not available. "
                "Set SMLX_DOWNLOAD_TEST_MODELS=1 to enable downloads"
            )

        # Try to download
        try:
            from smlx.tools.download import download_dataset

            print(f"\nDownloading dataset: {dataset_id} ({config_name})")
            download_dataset(dataset_id)

            # Try loading again
            from datasets import load_dataset

            dataset = load_dataset(dataset_id, config_name, split=split)
            return {
                "dataset": dataset,
                "dataset_id": dataset_id,
                "config": config_name,
                "split": split,
            }

        except Exception as e:
            pytest.skip(f"Failed to download/load dataset: {e}")


@pytest.fixture
def mmstar_dataset():
    """
    Load MMStar dataset for testing.

    Requires SMLX_DOWNLOAD_TEST_MODELS=1 environment variable or existing cache.
    """
    dataset_id = "Lin-Chen/MMStar"
    split = "val"

    # Try to load dataset (will use cache if available)
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_id, split=split)
        return {"dataset": dataset, "dataset_id": dataset_id, "split": split}

    except Exception as load_error:
        # Check if downloads are enabled
        if not _check_downloads_enabled():
            pytest.skip(
                f"Dataset {dataset_id} not available. "
                "Set SMLX_DOWNLOAD_TEST_MODELS=1 to enable downloads"
            )

        # Try to download
        try:
            from smlx.tools.download import download_dataset

            print(f"\nDownloading dataset: {dataset_id}")
            download_dataset(dataset_id)

            # Try loading again
            from datasets import load_dataset

            dataset = load_dataset(dataset_id, split=split)
            return {"dataset": dataset, "dataset_id": dataset_id, "split": split}

        except Exception as e:
            pytest.skip(f"Failed to download/load dataset: {e}")


@pytest.fixture
def ocrbench_dataset():
    """
    Load OCRBench dataset for testing.

    Requires SMLX_DOWNLOAD_TEST_MODELS=1 environment variable or existing cache.
    """
    dataset_id = "echo840/OCRBench"
    split = "test"

    # Try to load dataset (will use cache if available)
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_id, split=split)
        return {"dataset": dataset, "dataset_id": dataset_id, "split": split}

    except Exception as load_error:
        # Check if downloads are enabled
        if not _check_downloads_enabled():
            pytest.skip(
                f"Dataset {dataset_id} not available. "
                "Set SMLX_DOWNLOAD_TEST_MODELS=1 to enable downloads"
            )

        # Try to download
        try:
            from smlx.tools.download import download_dataset

            print(f"\nDownloading dataset: {dataset_id}")
            download_dataset(dataset_id)

            # Try loading again
            from datasets import load_dataset

            dataset = load_dataset(dataset_id, split=split)
            return {"dataset": dataset, "dataset_id": dataset_id, "split": split}

        except Exception as e:
            pytest.skip(f"Failed to download/load dataset: {e}")


@pytest.fixture
def small_test_sample():
    """
    Create a small synthetic test sample for quick unit tests.

    This doesn't require any downloads and can be used for fast testing.
    """
    return {
        "question": "What is 2 + 2?",
        "answer": "4",
        "image": None,  # No image for simple text question
        "question_type": "free_form",
    }


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_download: mark test as requiring model downloads")
