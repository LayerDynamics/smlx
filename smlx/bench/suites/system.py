"""
System information utilities for benchmarking.

Provides functions to detect system capabilities, especially for
Apple Silicon M4 chipsets.
"""

import platform
import subprocess
from typing import Any, Optional

import mlx.core as mx


def get_system_info() -> dict[str, Any]:
    """
    Get comprehensive system information.

    Returns:
        Dictionary with system information:
        - platform: Platform name
        - platform_version: Platform version
        - processor: Processor type
        - chip: Chip name (e.g., 'Apple M4')
        - python_version: Python version
        - mlx_version: MLX version
        - mlx_available: Whether MLX Metal is available
        - max_memory_gb: Maximum recommended working set size in GB

    Example:
        >>> info = get_system_info()
        >>> print(f"Running on {info['chip']} with {info['max_memory_gb']:.1f}GB")
    """
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "mlx_available": mx.metal.is_available(),
    }

    # Get chip information (Mac-specific)
    chip_name = get_chip_name()
    if chip_name:
        info["chip"] = chip_name

    # Get memory information
    if mx.metal.is_available():
        device_info = mx.metal.device_info()
        info["max_memory_gb"] = float(device_info["max_recommended_working_set_size"]) / 1e9
        info["max_buffer_length"] = device_info["max_buffer_length"]
    else:
        info["max_memory_gb"] = 0.0
        info["max_buffer_length"] = 0

    return info


def get_chip_name() -> Optional[str]:
    """
    Get the chip name (Apple Silicon specific).

    Returns:
        Chip name (e.g., 'Apple M4', 'Apple M3 Max') or None if not detectable

    Example:
        >>> chip = get_chip_name()
        >>> if chip and 'M4' in chip:
        ...     print("Running on M4!")
    """
    try:
        # Use sysctl to get chip name on macOS
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try to get from platform
        processor = platform.processor()
        if processor and "arm" in processor.lower():
            return f"Apple Silicon ({processor})"
        return None


def is_m4_chip() -> bool:
    """
    Check if running on Apple M4 chip.

    Returns:
        True if running on M4, False otherwise

    Example:
        >>> if is_m4_chip():
        ...     print("Optimized for M4!")
    """
    chip = get_chip_name()
    return chip is not None and "M4" in chip


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon.

    Returns:
        True if running on Apple Silicon, False otherwise
    """
    return platform.system() == "Darwin" and platform.processor() == "arm"


def get_memory_info() -> dict[str, float]:
    """
    Get memory information in GB.

    Returns:
        Dictionary with memory information:
        - max_recommended_gb: Maximum recommended working set size
        - active_gb: Currently active memory
        - cache_gb: Cache memory
        - peak_gb: Peak memory usage

    Example:
        >>> mem = get_memory_info()
        >>> print(f"Using {mem['active_gb']:.2f}GB / {mem['max_recommended_gb']:.2f}GB")
    """
    if not mx.metal.is_available():
        return {
            "max_recommended_gb": 0.0,
            "active_gb": 0.0,
            "cache_gb": 0.0,
            "peak_gb": 0.0,
        }

    device_info = mx.metal.device_info()

    return {
        "max_recommended_gb": float(device_info["max_recommended_working_set_size"]) / 1e9,
        "active_gb": mx.get_active_memory() / 1e9,
        "cache_gb": mx.get_cache_memory() / 1e9,
        "peak_gb": mx.get_peak_memory() / 1e9,
    }


def get_cpu_info() -> dict[str, Any]:
    """
    Get CPU information.

    Returns:
        Dictionary with CPU information:
        - processor: Processor name
        - physical_cores: Number of physical cores (if available)
        - logical_cores: Number of logical cores (if available)

    Example:
        >>> cpu = get_cpu_info()
        >>> print(f"CPU: {cpu['processor']} ({cpu.get('physical_cores', '?')} cores)")
    """
    info: dict[str, Any] = {
        "processor": platform.processor(),
    }

    # Try to get core count (Mac-specific)
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            capture_output=True,
            text=True,
            check=True,
        )
        info["physical_cores"] = int(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.logicalcpu"],
            capture_output=True,
            text=True,
            check=True,
        )
        info["logical_cores"] = int(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass

    return info


def print_system_info():
    """
    Print formatted system information.

    Example:
        >>> print_system_info()
        System Information
        ==================
        Platform: Darwin
        Chip: Apple M4
        ...
    """
    info = get_system_info()
    cpu_info = get_cpu_info()
    mem_info = get_memory_info()

    print("\nSystem Information")
    print("=" * 60)
    print(f"Platform: {info['platform']} {info['platform_version']}")
    if "chip" in info:
        print(f"Chip: {info['chip']}")
    print(f"Processor: {cpu_info['processor']}")
    if "physical_cores" in cpu_info:
        print(
            f"CPU Cores: {cpu_info['physical_cores']} physical, {cpu_info.get('logical_cores', '?')} logical"
        )
    print(f"Python: {info['python_version']}")
    print(f"MLX: {info['mlx_version']}")
    print(f"MLX Metal: {'Available' if info['mlx_available'] else 'Not Available'}")

    if info["mlx_available"]:
        print("\nMemory Information")
        print("-" * 60)
        print(f"Max Recommended: {mem_info['max_recommended_gb']:.2f} GB")
        print(f"Currently Active: {mem_info['active_gb']:.2f} GB")
        print(f"Cache: {mem_info['cache_gb']:.2f} GB")
        print(f"Peak: {mem_info['peak_gb']:.2f} GB")

    print("=" * 60)
