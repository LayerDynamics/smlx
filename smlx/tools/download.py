"""
Unified download utilities for SMLX models, datasets, and calibration data.

This module provides functions to download models from HuggingFace Hub,
evaluation datasets, and calibration data with caching support.
"""

import os
import urllib.request
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm


def get_cache_dir() -> Path:
    """
    Get the SMLX cache directory.

    Returns:
        Path to ~/.cache/smlx/
    """
    cache_dir = Path.home() / ".cache" / "smlx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_model(
    repo_id: str,
    *,
    allow_patterns: Optional[list[str]] = None,
    ignore_patterns: Optional[list[str]] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Download a model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace model repository ID (e.g., "mlx-community/SmolVLM-256M-Instruct")
        allow_patterns: List of file patterns to include (e.g., ["*.safetensors", "*.json"])
        ignore_patterns: List of file patterns to exclude (e.g., ["*.bin", "*.pt"])
        revision: Specific git revision/tag/branch to download
        cache_dir: Custom cache directory (defaults to ~/.cache/smlx/models/)

    Returns:
        Path to downloaded model directory

    Example:
        ```python
        from smlx.tools.download import download_model

        # Download full model
        model_path = download_model("mlx-community/SmolVLM-256M-Instruct")

        # Download only safetensors and config
        model_path = download_model(
            "mlx-community/SmolVLM-256M-Instruct",
            allow_patterns=["*.safetensors", "*.json"],
        )
        ```
    """
    if cache_dir is None:
        cache_dir = get_cache_dir() / "models"
    else:
        cache_dir = Path(cache_dir)

    print(f"Downloading model: {repo_id}")
    print(f"  Cache directory: {cache_dir}")

    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=False,
        )

        print(f"✓ Model downloaded: {model_path}")
        return Path(model_path)

    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        raise


def download_file(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Download a single file from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        filename: Name of file to download
        repo_type: Type of repository ("model", "dataset", "space")
        revision: Specific git revision/tag/branch
        cache_dir: Custom cache directory

    Returns:
        Path to downloaded file

    Example:
        ```python
        from smlx.tools.download import download_file

        # Download a config file
        config_path = download_file(
            "mlx-community/SmolVLM-256M-Instruct",
            "config.json"
        )
        ```
    """
    if cache_dir is None:
        cache_dir = get_cache_dir() / repo_type
    else:
        cache_dir = Path(cache_dir)

    print(f"Downloading file: {filename} from {repo_id}")

    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=False,
        )

        print(f"✓ File downloaded: {file_path}")
        return Path(file_path)

    except Exception as e:
        print(f"✗ Failed to download file: {e}")
        raise


def download_dataset(
    dataset_id: str,
    *,
    split: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Download an evaluation dataset from HuggingFace Hub.

    This uses snapshot_download to cache the entire dataset locally.
    For loading the dataset, use `datasets.load_dataset()` separately.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "AI4Math/MathVista")
        split: Specific split to download (e.g., "testmini", "test")
        cache_dir: Custom cache directory (defaults to ~/.cache/smlx/datasets/)

    Returns:
        Path to downloaded dataset directory

    Example:
        ```python
        from smlx.tools.download import download_dataset
        from datasets import load_dataset

        # Download dataset files
        dataset_path = download_dataset("AI4Math/MathVista")

        # Load dataset using datasets library
        dataset = load_dataset("AI4Math/MathVista", split="testmini")
        ```

    Note:
        This function downloads dataset files for caching.
        To actually load and use the dataset, use datasets.load_dataset().
    """
    if cache_dir is None:
        cache_dir = get_cache_dir() / "datasets"
    else:
        cache_dir = Path(cache_dir)

    print(f"Downloading dataset: {dataset_id}")
    if split:
        print(f"  Split: {split}")
    print(f"  Cache directory: {cache_dir}")

    try:
        dataset_path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            local_files_only=False,
        )

        print(f"✓ Dataset downloaded: {dataset_path}")
        print(f"  To load: datasets.load_dataset('{dataset_id}', split='{split or 'train'}')")
        return Path(dataset_path)

    except Exception as e:
        print(f"✗ Failed to download dataset: {e}")
        raise


class _DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None):
        """Update progress bar.

        Args:
            b: Number of blocks transferred
            bsize: Block size in bytes
            tsize: Total size in bytes
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_from_url(
    url: str,
    output_path: Union[str, Path],
    *,
    desc: Optional[str] = None,
    force: bool = False,
) -> Path:
    """
    Download a file from a URL with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save file
        desc: Description for progress bar
        force: Force re-download even if file exists

    Returns:
        Path to downloaded file

    Example:
        ```python
        from smlx.tools.download import download_from_url

        # Download calibration data
        calib_path = download_from_url(
            "https://gist.githubusercontent.com/.../calibration_v5.txt",
            "~/.cache/smlx/calibration_v5.txt",
            desc="Calibration data"
        )
        ```
    """
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if output_path.exists() and not force:
        print(f"✓ File already exists: {output_path}")
        return output_path

    print(f"Downloading from URL: {url}")
    print(f"  Output: {output_path}")

    try:
        with _DownloadProgressBar(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=desc or "Downloading",
        ) as pbar:
            urllib.request.urlretrieve(url, output_path, reporthook=pbar.update_to)

        print(f"✓ Downloaded: {output_path}")
        return output_path

    except Exception as e:
        print(f"✗ Failed to download: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        raise


def download_calibration_data(
    url: Optional[str] = None,
    *,
    output_path: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> Path:
    """
    Download calibration data for quantization.

    Args:
        url: URL to download from (defaults to standard calibration data)
        output_path: Custom output path (defaults to ~/.cache/smlx/calibration_v5.txt)
        force: Force re-download even if file exists

    Returns:
        Path to calibration data file

    Example:
        ```python
        from smlx.tools.download import download_calibration_data

        # Download default calibration data
        calib_path = download_calibration_data()

        # Use custom URL
        calib_path = download_calibration_data(
            url="https://example.com/custom_calibration.txt"
        )
        ```
    """
    # Default URL from tristandruyen's gist
    if url is None:
        url = (
            "https://gist.githubusercontent.com/tristandruyen/"
            "a6a06b02a1c5e88649ee90efde14e022/raw/"
            "2e7ffbfc0ebf964fd97f12e7a1fdb6e1be3b7b4a/calibration_v5.txt"
        )

    if output_path is None:
        output_path = get_cache_dir() / "calibration_v5.txt"
    else:
        output_path = Path(output_path)

    return download_from_url(
        url,
        output_path,
        desc="Calibration data",
        force=force,
    )


def check_download_enabled() -> bool:
    """
    Check if model/data downloads are enabled via environment variable.

    Returns:
        True if SMLX_DOWNLOAD_TEST_MODELS=1, False otherwise

    Example:
        ```python
        from smlx.tools.download import check_download_enabled

        if check_download_enabled():
            model_path = download_model("mlx-community/SmolVLM-256M-Instruct")
        else:
            print("Set SMLX_DOWNLOAD_TEST_MODELS=1 to enable downloads")
        ```
    """
    return os.getenv("SMLX_DOWNLOAD_TEST_MODELS", "0") == "1"


__all__ = [
    "get_cache_dir",
    "download_model",
    "download_file",
    "download_dataset",
    "download_from_url",
    "download_calibration_data",
    "check_download_enabled",
]
