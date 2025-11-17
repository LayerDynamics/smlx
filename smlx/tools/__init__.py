"""
Tools for SMLX - model conversion, downloading, and utilities.

This module provides command-line tools and utilities for:
- Downloading models and datasets from HuggingFace Hub
- Converting models to MLX format
- Managing cached data
"""

from .download import (
    check_download_enabled,
    download_calibration_data,
    download_dataset,
    download_file,
    download_from_url,
    download_model,
    get_cache_dir,
)

__all__ = [
    # Download utilities
    "download_model",
    "download_file",
    "download_dataset",
    "download_from_url",
    "download_calibration_data",
    "get_cache_dir",
    "check_download_enabled",
]
