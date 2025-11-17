"""
I/O utilities for reading and writing benchmark results.

Provides CSV and JSON serialization/deserialization.
"""

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2):
    """
    Save data to JSON file.

    Args:
        data: Data to save (dict, list, or dataclass)
        filepath: Output file path
        indent: JSON indentation

    Example:
        >>> save_json({"model": "SmolLM2", "tps": 123.45}, "results.json")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    if is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)
    elif isinstance(data, list) and data and is_dataclass(data[0]) and not isinstance(data[0], type):
        data = [asdict(item) for item in data]

    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data

    Example:
        >>> data = load_json("results.json")
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        return json.load(f)


def save_csv(
    data: list[Any],
    filepath: Union[str, Path],
    fieldnames: Optional[list[str]] = None,
):
    """
    Save list of dictionaries to CSV file.

    Args:
        data: list of dictionaries or dataclass instances
        filepath: Output file path
        fieldnames: Column names (if None, uses keys from first dict)

    Example:
        >>> data = [{"name": "model1", "tps": 100}, {"name": "model2", "tps": 150}]
        >>> save_csv(data, "results.csv")
    """
    if not data:
        return

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    processed_data: list[dict[str, Any]] = data
    if is_dataclass(data[0]) and not isinstance(data[0], type):
        processed_data = [asdict(item) for item in data]

    if fieldnames is None:
        fieldnames = list(processed_data[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(processed_data)


def load_csv(filepath: Union[str, Path]) -> list[dict[str, Any]]:
    """
    Load CSV file as list of dictionaries.

    Args:
        filepath: Input file path

    Returns:
        list of dictionaries

    Example:
        >>> data = load_csv("results.csv")
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        reader = csv.DictReader(f)
        return list(reader)


def append_csv(data: dict[str, Any], filepath: Union[str, Path]):
    """
    Append a single row to CSV file.

    Creates file with header if it doesn't exist.

    Args:
        data: dictionary to append
        filepath: Output file path

    Example:
        >>> append_csv({"name": "model1", "tps": 100}, "results.csv")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict
    if is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)

    file_exists = filepath.exists()
    fieldnames = list(data.keys())

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def save_jsonl(data: list[Any], filepath: Union[str, Path]):
    """
    Save data to JSON Lines format (one JSON object per line).

    Args:
        data: list of objects to save
        filepath: Output file path

    Example:
        >>> data = [{"name": "model1"}, {"name": "model2"}]
        >>> save_jsonl(data, "results.jsonl")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    if data and is_dataclass(data[0]) and not isinstance(data[0], type):
        data = [asdict(item) for item in data]

    with open(filepath, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


def load_jsonl(filepath: Union[str, Path]) -> list[dict[str, Any]]:
    """
    Load data from JSON Lines format.

    Args:
        filepath: Input file path

    Returns:
        list of dictionaries

    Example:
        >>> data = load_jsonl("results.jsonl")
    """
    filepath = Path(filepath)
    data = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def append_jsonl(data: Any, filepath: Union[str, Path]):
    """
    Append a single object to JSON Lines file.

    Args:
        data: Object to append
        filepath: Output file path

    Example:
        >>> append_jsonl({"name": "model1", "tps": 100}, "results.jsonl")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict
    if is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)

    with open(filepath, "a") as f:
        json.dump(data, f)
        f.write("\n")


def ensure_dir(filepath: Union[str, Path]) -> Path:
    """
    Ensure directory exists for filepath.

    Args:
        filepath: File or directory path

    Returns:
        Path object

    Example:
        >>> ensure_dir("results/benchmarks/output.json")
    """
    filepath = Path(filepath)
    if filepath.suffix:
        # It's a file, create parent directory
        filepath.parent.mkdir(parents=True, exist_ok=True)
    else:
        # It's a directory
        filepath.mkdir(parents=True, exist_ok=True)
    return filepath
