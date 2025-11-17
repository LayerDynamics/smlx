"""
Memory tracking utilities for MLX.

Provides memory profiling and tracking for MLX arrays and models,
leveraging Apple's unified memory architecture.
"""

from contextlib import contextmanager
from typing import Any

import mlx.core as mx


def get_peak_memory_gb() -> float:
    """
    Get peak memory usage in gigabytes.

    Returns:
        Peak memory usage in GB

    Example:
        >>> peak_gb = get_peak_memory_gb()
        >>> print(f"Peak memory: {peak_gb:.2f} GB")
    """
    return mx.metal.get_peak_memory() / 1e9


def get_active_memory_gb() -> float:
    """
    Get current active memory usage in gigabytes.

    Returns:
        Active memory usage in GB
    """
    return mx.metal.get_active_memory() / 1e9


def get_cache_memory_gb() -> float:
    """
    Get current cache memory usage in gigabytes.

    Returns:
        Cache memory usage in GB
    """
    return mx.metal.get_cache_memory() / 1e9


def clear_cache():
    """
    Clear the MLX Metal cache.

    Useful to call between benchmark runs to ensure consistent memory usage.
    """
    mx.metal.clear_cache()


def reset_peak_memory():
    """
    Reset the peak memory counter.

    Call this before a benchmark to get accurate peak memory for that specific run.
    """
    mx.metal.reset_peak_memory()


def get_device_info() -> dict[str, Any]:
    """
    Get device information including memory limits.

    Returns:
        Dictionary with device information:
        - max_recommended_working_set_size: Maximum recommended memory in bytes
        - max_buffer_length: Maximum buffer length
        - max_recommended_working_set_size_gb: Max recommended memory in GB

    Example:
        >>> info = get_device_info()
        >>> print(f"Max memory: {info['max_recommended_working_set_size_gb']:.2f} GB")
    """
    if not mx.metal.is_available():
        return {
            "max_recommended_working_set_size": 0,
            "max_buffer_length": 0,
            "max_recommended_working_set_size_gb": 0.0,
        }

    info: dict[str, Any] = mx.metal.device_info()
    info["max_recommended_working_set_size_gb"] = (
        info["max_recommended_working_set_size"] / 1e9
    )
    return info


@contextmanager
def memory_profiler(reset_peak: bool = True, clear_cache_before: bool = True):
    """
    Context manager for profiling memory usage.

    Args:
        reset_peak: Whether to reset peak memory before profiling
        clear_cache_before: Whether to clear cache before profiling

    Yields:
        MemoryStats object with start/peak/end memory

    Example:
        >>> with memory_profiler() as mem:
        ...     result = model(input)
        ...     mx.eval(result)
        >>> print(f"Peak memory: {mem.peak_gb:.2f} GB")
        >>> print(f"Memory delta: {mem.delta_gb:.2f} GB")
    """

    class MemoryStats:
        def __init__(self):
            self.start_active_gb: float = 0.0
            self.start_cache_gb: float = 0.0
            self.start_peak_gb: float = 0.0
            self.end_active_gb: float = 0.0
            self.end_cache_gb: float = 0.0
            self.peak_gb: float = 0.0

        @property
        def delta_active_gb(self) -> float:
            """Change in active memory."""
            return self.end_active_gb - self.start_active_gb

        @property
        def delta_cache_gb(self) -> float:
            """Change in cache memory."""
            return self.end_cache_gb - self.start_cache_gb

        @property
        def delta_gb(self) -> float:
            """Total memory delta (active + cache)."""
            return self.delta_active_gb + self.delta_cache_gb

        @property
        def peak_delta_gb(self) -> float:
            """Peak memory increase from start."""
            return self.peak_gb - self.start_peak_gb

    stats = MemoryStats()

    # Setup
    if clear_cache_before:
        clear_cache()

    if reset_peak:
        reset_peak_memory()

    # Record starting state
    stats.start_active_gb = get_active_memory_gb()
    stats.start_cache_gb = get_cache_memory_gb()
    stats.start_peak_gb = get_peak_memory_gb()

    try:
        yield stats
    finally:
        # Record ending state
        stats.end_active_gb = get_active_memory_gb()
        stats.end_cache_gb = get_cache_memory_gb()
        stats.peak_gb = get_peak_memory_gb()


def estimate_model_memory(
    num_parameters: int, dtype: mx.Dtype = mx.float16
) -> dict[str, float]:
    """
    Estimate memory usage for a model.

    Args:
        num_parameters: Number of model parameters
        dtype: Data type for parameters

    Returns:
        Dictionary with memory estimates:
        - parameters: Number of parameters
        - bytes_per_param: Bytes per parameter
        - total_bytes: Total bytes
        - total_mb: Total megabytes
        - total_gb: Total gigabytes

    Example:
        >>> mem = estimate_model_memory(135_000_000, mx.float16)
        >>> print(f"Estimated size: {mem['total_mb']:.2f} MB")
    """
    dtype_sizes = {
        mx.float32: 4,
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.int8: 1,
        mx.uint8: 1,
        mx.int16: 2,
        mx.int32: 4,
    }

    bytes_per_param = dtype_sizes.get(dtype, 4)
    total_bytes = num_parameters * bytes_per_param

    return {
        "parameters": num_parameters,
        "bytes_per_param": bytes_per_param,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / 1e6,
        "total_gb": total_bytes / 1e9,
    }


def check_memory_availability(required_gb: float) -> dict[str, Any]:
    """
    Check if sufficient memory is available.

    Args:
        required_gb: Required memory in GB

    Returns:
        Dictionary with availability info:
        - available: Whether sufficient memory is available
        - required_gb: Required memory
        - max_available_gb: Maximum available memory
        - current_active_gb: Current active memory usage
        - headroom_gb: Available headroom

    Example:
        >>> check = check_memory_availability(10.0)
        >>> if not check['available']:
        ...     print("Insufficient memory!")
    """
    device_info = get_device_info()
    max_available_gb = device_info["max_recommended_working_set_size_gb"]
    current_active_gb = get_active_memory_gb()
    headroom_gb = max_available_gb - current_active_gb

    return {
        "available": headroom_gb >= required_gb,
        "required_gb": required_gb,
        "max_available_gb": max_available_gb,
        "current_active_gb": current_active_gb,
        "headroom_gb": headroom_gb,
    }


def smart_cleanup(aggressive: bool = False) -> float:
    """
    Perform intelligent memory cleanup.

    Clears MLX Metal cache and optionally forces Python garbage collection.
    Returns the amount of cache memory that was freed.

    Args:
        aggressive: If True, perform more aggressive cleanup including Python GC

    Returns:
        Amount of cache memory freed in GB

    Example:
        >>> # Standard cleanup
        >>> freed_gb = smart_cleanup()
        >>> print(f"Freed {freed_gb:.2f}GB")
        >>>
        >>> # Aggressive cleanup when memory is critical
        >>> freed_gb = smart_cleanup(aggressive=True)
    """
    import gc
    import logging

    # Measure cache before cleanup
    cache_before = get_cache_memory_gb()

    # Level 1: Clear MLX Metal cache
    clear_cache()

    if aggressive:
        # Level 2: Force Python garbage collection
        gc.collect()

        # Level 3: Clear Metal cache again (may have freed Python refs)
        clear_cache()

        logging.info("Aggressive memory cleanup performed")
    else:
        logging.info("Standard memory cleanup performed")

    # Measure cache after cleanup
    cache_after = get_cache_memory_gb()
    freed = cache_before - cache_after

    return max(0.0, freed)


class MemoryMonitor:
    """
    Real-time memory monitoring with threshold detection and trend analysis.

    Tracks memory usage over time and provides alerts when thresholds are exceeded.
    Also analyzes trends to detect memory leaks or growth patterns.

    Args:
        warning_gb: Memory usage in GB that triggers warnings (default: 28.0)
        critical_gb: Memory usage in GB that triggers critical alerts (default: 32.0)

    Example:
        >>> monitor = MemoryMonitor(warning_gb=28.0, critical_gb=32.0)
        >>>
        >>> # Check current memory status
        >>> status = monitor.check()
        >>> if status['status'] == 'warning':
        ...     print("Memory pressure detected!")
        ...     for rec in status['recommendations']:
        ...         print(f"  - {rec}")
        >>>
        >>> # Analyze trend
        >>> trend = monitor.get_trend()
        >>> if trend == 'increasing':
        ...     print("Memory usage is growing")
    """

    def __init__(self, warning_gb: float = 28.0, critical_gb: float = 32.0):
        """
        Initialize memory monitor.

        Args:
            warning_gb: Warning threshold in GB
            critical_gb: Critical threshold in GB
        """
        self.warning_gb = warning_gb
        self.critical_gb = critical_gb
        self.history: list[dict[str, Any]] = []

    def check(self) -> dict[str, Any]:
        """
        Check current memory usage against thresholds.

        Returns:
            Dictionary with status, current usage, and recommendations:
            - status: 'ok', 'warning', or 'critical'
            - active_gb: Current active memory
            - cache_gb: Current cache memory
            - total_gb: Total memory in use
            - max_gb: Maximum available memory
            - utilization: Memory utilization (0-1)
            - recommendations: List of suggested actions
        """
        active_gb = get_active_memory_gb()
        cache_gb = get_cache_memory_gb()
        total_gb = active_gb + cache_gb

        device_info = get_device_info()
        max_gb = device_info['max_recommended_working_set_size_gb']

        status = 'ok'
        recommendations = []

        if total_gb >= self.critical_gb:
            status = 'critical'
            recommendations = [
                "Immediately clear cache with clear_cache() or smart_cleanup()",
                "Reduce batch size to 1",
                "Use rotating KV cache with max_kv_size=1024",
                "Reduce max_tokens significantly",
                "Consider model quantization (4-bit or 8-bit)",
                "Close other applications to free system memory",
            ]
        elif total_gb >= self.warning_gb:
            status = 'warning'
            recommendations = [
                "Clear cache with clear_cache() or smart_cleanup()",
                "Monitor memory growth closely",
                "Consider reducing max_tokens or batch_size",
                "Use rotating KV cache if generating >1000 tokens",
            ]

        result = {
            'status': status,
            'active_gb': active_gb,
            'cache_gb': cache_gb,
            'total_gb': total_gb,
            'max_gb': max_gb,
            'utilization': total_gb / max_gb if max_gb > 0 else 0,
            'recommendations': recommendations,
        }

        self.history.append(result)
        return result

    def get_trend(self, last_n: int = 10) -> str:
        """
        Analyze memory usage trend.

        Args:
            last_n: Number of recent checks to analyze

        Returns:
            'increasing', 'stable', or 'decreasing'

        Example:
            >>> monitor = MemoryMonitor()
            >>> # ... perform multiple checks ...
            >>> trend = monitor.get_trend(last_n=10)
            >>> if trend == 'increasing':
            ...     print("Memory leak detected!")
        """
        if len(self.history) < 2:
            return 'stable'

        recent = self.history[-last_n:]
        values = [h['total_gb'] for h in recent]

        if len(values) < 2:
            return 'stable'

        # Simple linear trend analysis
        avg_change = (values[-1] - values[0]) / len(values)

        if avg_change > 0.1:  # Growing by >0.1GB per check
            return 'increasing'
        elif avg_change < -0.1:  # Decreasing by >0.1GB per check
            return 'decreasing'
        else:
            return 'stable'

    def reset(self) -> None:
        """Reset monitoring history."""
        self.history.clear()
