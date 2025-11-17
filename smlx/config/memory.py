#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Memory Configuration for SMLX.

Provides centralized configuration for memory management via environment
variables, allowing users to control memory behavior without code changes.

Environment Variables:
    SMLX_MAX_MEMORY_GB: Maximum memory usage in GB (default: 30.0)
    SMLX_WATCHDOG_ENABLED: Enable/disable memory watchdog (default: True)
    SMLX_AUTO_CLEANUP: Automatic cleanup on threshold (default: True)
    SMLX_MAX_KV_SIZE: Default KV cache size limit (default: 2048)
    SMLX_WARNING_THRESHOLD: Warning threshold 0-1 (default: 0.80)
    SMLX_CRITICAL_THRESHOLD: Critical threshold 0-1 (default: 0.90)
    SMLX_WATCHDOG_INTERVAL: Watchdog check interval in seconds (default: 1.0)
    SMLX_ENABLE_ROTATING_CACHE: Use rotating KV cache by default (default: True)
    SMLX_AGGRESSIVE_CLEANUP: Use aggressive cleanup strategy (default: False)
    SMLX_MEMORY_PROFILING: Enable detailed memory profiling (default: False)

Example:
    >>> import os
    >>> os.environ['SMLX_MAX_MEMORY_GB'] = '20.0'
    >>> os.environ['SMLX_WATCHDOG_ENABLED'] = 'true'
    >>>
    >>> from smlx.config.memory import MemoryConfig
    >>> config = MemoryConfig()
    >>> print(f"Max memory: {config.max_memory_gb}GB")
    >>> print(f"Watchdog enabled: {config.watchdog_enabled}")
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _get_bool_env(key: str, default: bool) -> bool:
    """Get boolean from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on', 'enabled')


def _get_float_env(key: str, default: float) -> float:
    """Get float from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Float value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_int_env(key: str, default: int) -> int:
    """Get integer from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Integer value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class MemoryConfig:
    """
    Memory configuration for SMLX.

    This class centralizes all memory-related configuration options,
    loading values from environment variables with sensible defaults.

    Attributes:
        max_memory_gb: Maximum memory usage in GB
        watchdog_enabled: Enable memory watchdog monitoring
        auto_cleanup: Automatic cleanup when thresholds exceeded
        max_kv_size: Default KV cache size limit
        warning_threshold: Memory warning threshold (0-1)
        critical_threshold: Memory critical threshold (0-1)
        watchdog_interval: Watchdog check interval in seconds
        enable_rotating_cache: Use rotating KV cache by default
        aggressive_cleanup: Use aggressive cleanup (includes Python GC)
        memory_profiling: Enable detailed memory profiling

    Example:
        >>> config = MemoryConfig()
        >>> print(f"Max memory: {config.max_memory_gb}GB")
        >>>
        >>> # Override defaults
        >>> config = MemoryConfig(max_memory_gb=20.0, watchdog_enabled=False)
        >>>
        >>> # Use environment variables
        >>> import os
        >>> os.environ['SMLX_MAX_MEMORY_GB'] = '25.0'
        >>> config = MemoryConfig.from_env()
    """

    max_memory_gb: float = 30.0
    watchdog_enabled: bool = True
    auto_cleanup: bool = True
    max_kv_size: int = 2048
    warning_threshold: float = 0.80
    critical_threshold: float = 0.90
    watchdog_interval: float = 1.0
    enable_rotating_cache: bool = True
    aggressive_cleanup: bool = False
    memory_profiling: bool = False

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """
        Create configuration from environment variables.

        Reads all SMLX_* environment variables and creates a MemoryConfig
        instance with those values, falling back to defaults if not set.

        Returns:
            MemoryConfig instance populated from environment

        Example:
            >>> import os
            >>> os.environ['SMLX_MAX_MEMORY_GB'] = '20.0'
            >>> os.environ['SMLX_WATCHDOG_ENABLED'] = 'false'
            >>> config = MemoryConfig.from_env()
            >>> assert config.max_memory_gb == 20.0
            >>> assert config.watchdog_enabled == False
        """
        return cls(
            max_memory_gb=_get_float_env('SMLX_MAX_MEMORY_GB', 30.0),
            watchdog_enabled=_get_bool_env('SMLX_WATCHDOG_ENABLED', True),
            auto_cleanup=_get_bool_env('SMLX_AUTO_CLEANUP', True),
            max_kv_size=_get_int_env('SMLX_MAX_KV_SIZE', 2048),
            warning_threshold=_get_float_env('SMLX_WARNING_THRESHOLD', 0.80),
            critical_threshold=_get_float_env('SMLX_CRITICAL_THRESHOLD', 0.90),
            watchdog_interval=_get_float_env('SMLX_WATCHDOG_INTERVAL', 1.0),
            enable_rotating_cache=_get_bool_env('SMLX_ENABLE_ROTATING_CACHE', True),
            aggressive_cleanup=_get_bool_env('SMLX_AGGRESSIVE_CLEANUP', False),
            memory_profiling=_get_bool_env('SMLX_MEMORY_PROFILING', False),
        )

    def validate(self) -> None:
        """
        Validate configuration values.

        Ensures all configuration values are within acceptable ranges.

        Raises:
            ValueError: If any configuration value is invalid

        Example:
            >>> config = MemoryConfig(max_memory_gb=-10.0)
            >>> config.validate()  # Raises ValueError
        """
        if self.max_memory_gb <= 0:
            raise ValueError(f"max_memory_gb must be positive, got {self.max_memory_gb}")

        if not 0 <= self.warning_threshold <= 1:
            raise ValueError(
                f"warning_threshold must be in [0, 1], got {self.warning_threshold}"
            )

        if not 0 <= self.critical_threshold <= 1:
            raise ValueError(
                f"critical_threshold must be in [0, 1], got {self.critical_threshold}"
            )

        if self.warning_threshold >= self.critical_threshold:
            raise ValueError(
                f"warning_threshold ({self.warning_threshold}) must be less than "
                f"critical_threshold ({self.critical_threshold})"
            )

        if self.max_kv_size <= 0:
            raise ValueError(f"max_kv_size must be positive, got {self.max_kv_size}")

        if self.watchdog_interval <= 0:
            raise ValueError(
                f"watchdog_interval must be positive, got {self.watchdog_interval}"
            )

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration

        Example:
            >>> config = MemoryConfig()
            >>> d = config.to_dict()
            >>> assert 'max_memory_gb' in d
        """
        return {
            'max_memory_gb': self.max_memory_gb,
            'watchdog_enabled': self.watchdog_enabled,
            'auto_cleanup': self.auto_cleanup,
            'max_kv_size': self.max_kv_size,
            'warning_threshold': self.warning_threshold,
            'critical_threshold': self.critical_threshold,
            'watchdog_interval': self.watchdog_interval,
            'enable_rotating_cache': self.enable_rotating_cache,
            'aggressive_cleanup': self.aggressive_cleanup,
            'memory_profiling': self.memory_profiling,
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"MemoryConfig("
            f"max_memory_gb={self.max_memory_gb}, "
            f"watchdog_enabled={self.watchdog_enabled}, "
            f"auto_cleanup={self.auto_cleanup}, "
            f"max_kv_size={self.max_kv_size}, "
            f"warning={self.warning_threshold:.0%}, "
            f"critical={self.critical_threshold:.0%})"
        )


# Global default configuration (loaded from environment)
_default_config: Optional[MemoryConfig] = None


def get_default_config() -> MemoryConfig:
    """
    Get global default memory configuration.

    Lazy-loads configuration from environment on first call,
    then caches for subsequent calls.

    Returns:
        Global default MemoryConfig instance

    Example:
        >>> from smlx.config.memory import get_default_config
        >>> config = get_default_config()
        >>> print(f"Max memory: {config.max_memory_gb}GB")
    """
    global _default_config
    if _default_config is None:
        _default_config = MemoryConfig.from_env()
        _default_config.validate()
    return _default_config


def reset_default_config() -> None:
    """
    Reset global default configuration.

    Forces re-loading from environment on next get_default_config() call.
    Useful for testing or when environment variables change at runtime.

    Example:
        >>> import os
        >>> from smlx.config.memory import reset_default_config, get_default_config
        >>>
        >>> # Change environment
        >>> os.environ['SMLX_MAX_MEMORY_GB'] = '25.0'
        >>>
        >>> # Force reload
        >>> reset_default_config()
        >>> config = get_default_config()
        >>> assert config.max_memory_gb == 25.0
    """
    global _default_config
    _default_config = None


__all__ = [
    'MemoryConfig',
    'get_default_config',
    'reset_default_config',
]
