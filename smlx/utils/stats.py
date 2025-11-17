"""
Statistical utilities for benchmark analysis.

Provides functions for computing statistics on benchmark results.
"""

import math
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    """
    Compute the arithmetic mean.

    Args:
        values: Sequence of numbers

    Returns:
        Mean value

    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values: Sequence[float]) -> float:
    """
    Compute the median value.

    Args:
        values: Sequence of numbers

    Returns:
        Median value

    Example:
        >>> median([1, 2, 3, 4, 5])
        3.0
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]


def std(values: Sequence[float], ddof: int = 1) -> float:
    """
    Compute the standard deviation.

    Args:
        values: Sequence of numbers
        ddof: Delta degrees of freedom (default: 1 for sample std)

    Returns:
        Standard deviation

    Example:
        >>> std([1, 2, 3, 4, 5])
        1.5811388300841898
    """
    if len(values) < 2:
        return 0.0

    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - ddof)
    return math.sqrt(variance)


def percentile(values: Sequence[float], p: float) -> float:
    """
    Compute the p-th percentile.

    Args:
        values: Sequence of numbers
        p: Percentile to compute (0-100)

    Returns:
        Value at the p-th percentile

    Example:
        >>> percentile([1, 2, 3, 4, 5], 50)  # Same as median
        3.0
        >>> percentile([1, 2, 3, 4, 5], 95)
        4.8
    """
    if not values:
        return 0.0

    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if p == 0:
        return sorted_values[0]
    if p == 100:
        return sorted_values[-1]

    # Linear interpolation
    k = (n - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_values[int(k)]

    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def min_max(values: Sequence[float]) -> tuple[float, float]:
    """
    Get minimum and maximum values.

    Args:
        values: Sequence of numbers

    Returns:
        Tuple of (min, max)

    Example:
        >>> min_max([1, 2, 3, 4, 5])
        (1, 5)
    """
    if not values:
        return (0.0, 0.0)
    return (min(values), max(values))


def summary_stats(values: Sequence[float]) -> dict[str, float]:
    """
    Compute summary statistics.

    Args:
        values: Sequence of numbers

    Returns:
        Dictionary with statistics:
        - count: Number of values
        - mean: Arithmetic mean
        - median: Median value
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - p25: 25th percentile
        - p75: 75th percentile
        - p95: 95th percentile
        - p99: 99th percentile

    Example:
        >>> stats = summary_stats([1, 2, 3, 4, 5])
        >>> print(f"Mean: {stats['mean']}, Std: {stats['std']:.2f}")
        Mean: 3.0, Std: 1.58
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    min_val, max_val = min_max(values)

    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "std": std(values),
        "min": min_val,
        "max": max_val,
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def tokens_per_second(num_tokens: int, time_seconds: float) -> float:
    """
    Calculate tokens per second.

    Args:
        num_tokens: Number of tokens processed
        time_seconds: Time taken in seconds

    Returns:
        Tokens per second

    Example:
        >>> tps = tokens_per_second(1000, 5.0)
        >>> print(f"Throughput: {tps:.2f} tok/s")
        Throughput: 200.00 tok/s
    """
    if time_seconds <= 0:
        return 0.0
    return num_tokens / time_seconds


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string

    Example:
        >>> format_duration(0.001)
        '1.00ms'
        >>> format_duration(1.5)
        '1.50s'
        >>> format_duration(75)
        '1m 15s'
    """
    if seconds < 0.001:
        return f"{seconds * 1e6:.2f}µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_memory(bytes_val: float) -> str:
    """
    Format memory size in human-readable format.

    Args:
        bytes_val: Memory size in bytes

    Returns:
        Formatted string

    Example:
        >>> format_memory(1024)
        '1.00 KB'
        >>> format_memory(1024 * 1024 * 1024)
        '1.00 GB'
    """
    if bytes_val < 1024:
        return f"{bytes_val:.2f} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val / 1024**2:.2f} MB"
    else:
        return f"{bytes_val / 1024**3:.2f} GB"


def format_number(num: float, precision: int = 2) -> str:
    """
    Format large numbers with thousand separators.

    Args:
        num: Number to format
        precision: Decimal precision

    Returns:
        Formatted string

    Example:
        >>> format_number(1234567.89)
        '1,234,567.89'
    """
    return f"{num:,.{precision}f}"
