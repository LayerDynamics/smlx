"""
SMLX Utilities

Common utility functions for the SMLX package.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

# Import cache utilities
from .cache import (
    KVCache,
    RotatingKVCache,
    make_cache,
)

# Import chat template utilities
from .chat_templates import (
    MODEL_FORMATS,
    MessageFormat,
    apply_chat_template,
    get_image_token_for_model,
    validate_image_count,
)

# Import configuration utilities
from .config import (
    AUDIO_MODEL_DEFAULTS,
    LANGUAGE_MODEL_DEFAULTS,
    VISION_LANGUAGE_MODEL_DEFAULTS,
    BaseModelArgs,
    estimate_parameters,
    load_config,
    merge_configs,
    print_config,
    validate_config,
)
from .formatting import (
    ProgressBar,
    colorize,
    format_comparison,
    format_dict_table,
    format_header,
    format_key_value,
    format_section,
)
from .formatting import (
    format_table as format_table_new,
)

# Import generation utilities
from .generation import (
    GenerationConfig,
    chat,
    complete,
    generate,
    generate_step,
    stream_generate,
)
from .io import (
    append_csv,
    append_jsonl,
    load_jsonl,
    save_jsonl,
)
from .io import (
    ensure_dir as ensure_dir_new,
)

# Import loading utilities
from .loading import (
    check_tokenizer_compatibility,
    detect_quantization,
    get_quantized_layers,
    load_sharded_weights,
    load_tokenizer,
    load_weights,
    resolve_model_path,
    sanitize_weights,
    save_sharded_weights,
    save_weights,
    verify_model_integrity,
    verify_weights,
)
from .memory import (
    check_memory_availability,
    clear_cache,
    estimate_model_memory,
    get_active_memory_gb,
    get_cache_memory_gb,
    get_device_info,
    get_peak_memory_gb,
    memory_profiler,
    reset_peak_memory,
)

# Import quality metrics utilities
from .quality_metrics import (
    QualityComparison,
    QualityMetrics,
    analyze_repetition,
    analyze_token_distribution,
    assess_quality,
    calculate_diversity_score,
    calculate_entropy,
    calculate_perplexity,
    compare_quality,
)

# Import quantization utilities
from .quantization import (
    apply_quantization,
    count_quantizable_layers,
    create_class_predicate,
    estimate_quantized_size,
    get_quantization_config,
    get_quantization_info,
    has_quantizable_layers,
)

# Import sampling utilities
from .sampling import (
    categorical_sampling,
    make_logits_processors,
    make_repetition_penalty,
    make_sampler,
    min_p_sampling,
    sample,
    top_k_sampling,
    top_p_sampling,
)
from .stats import (
    format_duration,
    format_memory,
    mean,
    median,
    min_max,
    percentile,
    std,
    summary_stats,
    tokens_per_second,
)

# Import new benchmark-related utilities
from .timing import (
    Timer,
    benchmark,
    measure_runtime,
    timer,
)

# Import validation utilities
from .validation import (
    OutputValidator,
    ValidationResult,
    validate_audio_output,
    validate_text_output,
    validate_tokens,
)

# Import VLM diagnostics utilities
from .vlm_diagnostics import (
    compare_with_reference,
    log_attention_mask,
    log_embedding_comparison,
    log_logits_distribution,
    log_vision_features,
)

# Image Utilities


def preprocess_image(
    image: Union[str, Path, Image.Image], max_size: Optional[int] = None, mode: str = "RGB"
) -> Image.Image:
    """
    Load and preprocess an image.

    Args:
        image: Path to image file or PIL Image object
        max_size: Maximum dimension (width or height). If None, no resizing
        mode: Color mode (RGB, L, etc.)

    Returns:
        PIL Image object

    Example:
        >>> img = preprocess_image("path/to/image.jpg", max_size=512)
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    if image.mode != mode:
        image = image.convert(mode)

    if max_size is not None:
        image = resize_image(image, max_size)

    return image


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.

    Args:
        image: PIL Image object
        max_size: Maximum dimension (width or height)

    Returns:
        Resized PIL Image object
    """
    width, height = image.size

    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def image_to_base64(image: Union[str, Path, Image.Image]) -> str:
    """
    Convert image to base64 string.

    Args:
        image: Path to image or PIL Image object

    Returns:
        Base64 encoded string
    """
    import base64
    from io import BytesIO

    if isinstance(image, (str, Path)):
        image = Image.open(image)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# Text Utilities


def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    remove_punctuation: bool = False,
) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Input text
        lowercase: Convert to lowercase
        strip_whitespace: Strip leading/trailing whitespace
        remove_punctuation: Remove punctuation

    Returns:
        Normalized text

    Example:
        >>> normalize_text("  Hello, World!  ")
        'hello, world!'
    """
    if strip_whitespace:
        text = text.strip()

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = re.sub(r"[^\w\s]", "", text)

    return text


def extract_numbers(text: str) -> list[float]:
    """
    Extract all numbers from text.

    Args:
        text: Input text

    Returns:
        list of numbers found in text

    Example:
        >>> extract_numbers("The answer is 42.5 or maybe 100")
        [42.5, 100.0]
    """
    pattern = r"-?\d+\.?\d*"
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def clean_whitespace(text: str) -> str:
    """
    Clean excessive whitespace from text.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace

    Example:
        >>> clean_whitespace("Hello   world\\n\\n\\nfoo")
        'Hello world\\nfoo'
    """
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)
    # Replace multiple newlines with single newline
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


# Batch Processing Utilities


def batch_items(items: list[Any], batch_size: int) -> list[list[Any]]:
    """
    Split items into batches.

    Args:
        items: list of items
        batch_size: Size of each batch

    Returns:
        list of batches

    Example:
        >>> batch_items([1, 2, 3, 4, 5], batch_size=2)
        [[1, 2], [3, 4], [5]]
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def flatten_list(nested_list: list[list[Any]]) -> list[Any]:
    """
    Flatten a nested list.

    Args:
        nested_list: Nested list

    Returns:
        Flattened list

    Example:
        >>> flatten_list([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
    """
    return [item for sublist in nested_list for item in sublist]


# File I/O Utilities


def load_json(path: Union[str, Path]) -> dict[str, Any]:
    """
    Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_csv(path: Union[str, Path]) -> list[dict[str, Any]]:
    """
    Load CSV file.

    Args:
        path: Path to CSV file

    Returns:
        list of dictionaries (one per row)
    """
    with open(path, encoding="utf-8") as f:
        return list(csv.dictReader(f))


def save_csv(
    data: list[dict[str, Any]], path: Union[str, Path], fieldnames: Optional[list[str]] = None
) -> None:
    """
    Save data to CSV file.

    Args:
        data: list of dictionaries
        path: Output path
        fieldnames: Column names (auto-detected if None)
    """
    if not data:
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.dictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


# Result Aggregation Utilities


def aggregate_scores(results: list[dict[str, Any]], score_key: str = "score") -> dict[str, float]:
    """
    Aggregate scores from results.

    Args:
        results: list of result dictionaries
        score_key: Key containing the score

    Returns:
        dictionary with aggregated statistics

    Example:
        >>> results = [{"score": 1}, {"score": 0}, {"score": 1}]
        >>> aggregate_scores(results)
        {'mean': 0.667, 'sum': 2, 'count': 3, 'accuracy': 0.667}
    """
    scores = [r[score_key] for r in results if score_key in r]

    if not scores:
        return {"mean": 0.0, "sum": 0, "count": 0, "accuracy": 0.0}

    return {
        "mean": sum(scores) / len(scores),
        "sum": sum(scores),
        "count": len(scores),
        "accuracy": sum(scores) / len(scores),  # Alias for binary scores
    }


def group_by(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    """
    Group items by a key.

    Args:
        items: list of dictionaries
        key: Key to group by

    Returns:
        dictionary mapping key values to lists of items

    Example:
        >>> items = [{"cat": "A", "val": 1}, {"cat": "B", "val": 2}, {"cat": "A", "val": 3}]
        >>> group_by(items, "cat")
        {'A': [{'cat': 'A', 'val': 1}, {'cat': 'A', 'val': 3}], 'B': [{'cat': 'B', 'val': 2}]}
    """
    groups = {}
    for item in items:
        group_key = item.get(key)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(item)
    return groups


def compute_category_scores(
    results: list[dict[str, Any]], category_key: str = "category", score_key: str = "score"
) -> dict[str, dict[str, float]]:
    """
    Compute scores by category.

    Args:
        results: list of result dictionaries
        category_key: Key containing the category
        score_key: Key containing the score

    Returns:
        dictionary mapping categories to score statistics

    Example:
        >>> results = [
        ...     {"category": "math", "score": 1},
        ...     {"category": "math", "score": 0},
        ...     {"category": "vision", "score": 1}
        ... ]
        >>> compute_category_scores(results)
        {'math': {'accuracy': 0.5, 'count': 2}, 'vision': {'accuracy': 1.0, 'count': 1}}
    """
    grouped = group_by(results, category_key)

    category_scores = {}
    for category, items in grouped.items():
        stats = aggregate_scores(items, score_key)
        category_scores[category] = {"accuracy": stats["accuracy"], "count": stats["count"]}

    return category_scores


# Formatting Utilities


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal as a percentage string.

    Args:
        value: Decimal value (0.0 to 1.0)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string

    Example:
        >>> format_percentage(0.8572)
        '85.72%'
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """
    Format a number with thousands separators.

    Args:
        value: Number to format
        decimals: Number of decimal places (for floats)

    Returns:
        Formatted number string

    Example:
        >>> format_number(1234567.89)
        '1,234,567.89'
    """
    if isinstance(value, int):
        return f"{value:,}"
    else:
        return f"{value:,.{decimals}f}"


def format_table(
    data: list[dict[str, Any]], headers: Optional[list[str]] = None, max_col_width: int = 50
) -> str:
    """
    Format data as a simple text table.

    Args:
        data: list of dictionaries
        headers: Column headers (auto-detected if None)
        max_col_width: Maximum column width

    Returns:
        Formatted table string
    """
    if not data:
        return ""

    if headers is None:
        headers = list(data[0].keys())

    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = min(
            max(len(header), max(len(str(row.get(header, ""))) for row in data)), max_col_width
        )

    # Build table
    lines = []

    # Header
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Rows
    for row in data:
        row_line = " | ".join(
            truncate_text(str(row.get(h, "")), col_widths[h], "...").ljust(col_widths[h])
            for h in headers
        )
        lines.append(row_line)

    return "\n".join(lines)


# Path Utilities


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def get_results_dir() -> Path:
    """
    Get the results directory, creating if necessary.

    Returns:
        Path to results directory
    """
    results_dir = get_project_root() / "results"
    ensure_dir(results_dir)
    return results_dir


# Export all utilities
__all__ = [
    # Timing utilities (new)
    "timer",
    "benchmark",
    "Timer",
    "measure_runtime",
    # Sampling utilities (new)
    "sample",
    "make_sampler",
    "top_k_sampling",
    "top_p_sampling",
    "min_p_sampling",
    "categorical_sampling",
    "make_repetition_penalty",
    "make_logits_processors",
    # Generation utilities (new)
    "GenerationConfig",
    "generate_step",
    "generate",
    "stream_generate",
    "chat",
    "complete",
    # Memory utilities (new)
    "get_peak_memory_gb",
    "get_active_memory_gb",
    "get_cache_memory_gb",
    "clear_cache",
    "reset_peak_memory",
    "get_device_info",
    "memory_profiler",
    "estimate_model_memory",
    "check_memory_availability",
    # Statistical utilities (new)
    "mean",
    "median",
    "std",
    "percentile",
    "min_max",
    "summary_stats",
    "tokens_per_second",
    "format_duration",
    "format_memory",
    # Formatting utilities (new)
    "format_table_new",
    "format_dict_table",
    "format_key_value",
    "format_header",
    "format_section",
    "ProgressBar",
    "colorize",
    "format_comparison",
    # I/O utilities (new)
    "save_jsonl",
    "load_jsonl",
    "append_jsonl",
    "append_csv",
    "ensure_dir_new",
    # Image utilities
    "preprocess_image",
    "resize_image",
    "image_to_base64",
    # Text utilities
    "normalize_text",
    "extract_numbers",
    "truncate_text",
    "clean_whitespace",
    # Batch utilities
    "batch_items",
    "flatten_list",
    # File I/O utilities
    "load_json",
    "save_json",
    "load_csv",
    "save_csv",
    # Result aggregation utilities
    "aggregate_scores",
    "group_by",
    "compute_category_scores",
    # Formatting utilities
    "format_percentage",
    "format_number",
    "format_table",
    # Path utilities
    "ensure_dir",
    "get_project_root",
    "get_results_dir",
    # Configuration utilities
    "BaseModelArgs",
    "load_config",
    "merge_configs",
    "validate_config",
    "print_config",
    "estimate_parameters",
    "LANGUAGE_MODEL_DEFAULTS",
    "VISION_LANGUAGE_MODEL_DEFAULTS",
    "AUDIO_MODEL_DEFAULTS",
    # Loading utilities
    "resolve_model_path",
    "load_weights",
    "load_sharded_weights",
    "save_weights",
    "save_sharded_weights",
    "sanitize_weights",
    "load_tokenizer",
    "detect_quantization",
    "get_quantized_layers",
    "verify_weights",
    "check_tokenizer_compatibility",
    "verify_model_integrity",
    # Cache utilities
    "KVCache",
    "RotatingKVCache",
    "make_cache",
    # Quantization utilities
    "apply_quantization",
    "has_quantizable_layers",
    "count_quantizable_layers",
    "get_quantization_config",
    "create_class_predicate",
    "estimate_quantized_size",
    "get_quantization_info",
    # Validation utilities
    "ValidationResult",
    "OutputValidator",
    "validate_text_output",
    "validate_audio_output",
    "validate_tokens",
    # Quality metrics utilities
    "QualityMetrics",
    "QualityComparison",
    "calculate_perplexity",
    "calculate_entropy",
    "analyze_repetition",
    "analyze_token_distribution",
    "calculate_diversity_score",
    "assess_quality",
    "compare_quality",
    # VLM diagnostics utilities
    "log_vision_features",
    "log_embedding_comparison",
    "log_logits_distribution",
    "log_attention_mask",
    "compare_with_reference",
    # Chat template utilities
    "apply_chat_template",
    "get_image_token_for_model",
    "validate_image_count",
    "MessageFormat",
    "MODEL_FORMATS",
]
