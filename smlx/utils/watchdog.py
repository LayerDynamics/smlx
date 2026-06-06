#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Memory Watchdog for preventing crashes due to memory overflow.

This module provides background monitoring of memory usage with automatic
cleanup and configurable callbacks for warning and critical thresholds.

Features:
- Background thread monitoring with minimal overhead
- Configurable warning and critical thresholds
- Automatic cleanup when thresholds are exceeded
- Context manager for easy usage
- Apple Silicon unified memory awareness

Example:
    >>> from smlx.utils.watchdog import watchdog
    >>>
    >>> # Use as context manager
    >>> with watchdog(warning_threshold=0.80):
    ...     result = model.generate(prompt, max_tokens=1000)
    >>>
    >>> # Or start/stop manually
    >>> from smlx.utils.watchdog import MemoryWatchdog
    >>>
    >>> wd = MemoryWatchdog(warning_threshold=0.75, auto_cleanup=True)
    >>> wd.start()
    >>> # ... run inference ...
    >>> wd.stop()
"""

import logging
import threading
import time
from typing import Callable, Optional

try:
    import psutil
except ImportError:
    psutil = None
    logging.warning(
        "psutil not installed. MemoryWatchdog will not function. "
        "Install with: pip install psutil"
    )


class MemoryWatchdog:
    """
    Background thread that monitors memory usage and takes action when thresholds are exceeded.

    The watchdog monitors both system and process memory, with awareness of
    Apple Silicon's unified memory architecture. It can automatically clean up
    MLX caches and call custom callbacks when memory pressure is detected.

    Args:
        warning_threshold: Memory utilization (0-1) that triggers warnings (default: 0.80)
        critical_threshold: Memory utilization (0-1) that triggers critical actions (default: 0.90)
        check_interval: Seconds between memory checks (default: 1.0)
        on_warning: Optional callback function for warning events
        on_critical: Optional callback function for critical events
        auto_cleanup: Automatically clear caches when thresholds are exceeded (default: True)

    Example:
        >>> def handle_warning(mem_info):
        ...     print(f"Warning: Memory at {mem_info['percent']*100:.1f}%")
        >>>
        >>> watchdog = MemoryWatchdog(
        ...     warning_threshold=0.80,
        ...     critical_threshold=0.90,
        ...     on_warning=handle_warning
        ... )
        >>> watchdog.start()
        >>> # ... run inference ...
        >>> watchdog.stop()
    """

    def __init__(
        self,
        warning_threshold: float = 0.80,
        critical_threshold: float = 0.90,
        check_interval: float = 1.0,
        on_warning: Optional[Callable] = None,
        on_critical: Optional[Callable] = None,
        auto_cleanup: bool = True,
    ):
        if psutil is None:
            raise ImportError(
                "psutil is required for MemoryWatchdog. Install with: pip install psutil"
            )

        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.on_warning = on_warning
        self.on_critical = on_critical
        self.auto_cleanup = auto_cleanup

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._warning_fired = False
        self._critical_fired = False

    def start(self) -> None:
        """Start monitoring in background thread."""
        if self._thread is not None:
            logging.warning("MemoryWatchdog already running")
            return

        self._stop_event.clear()
        self._warning_fired = False
        self._critical_fired = False
        self._thread = threading.Thread(target=self._monitor, daemon=True, name="MemoryWatchdog")
        self._thread.start()
        logging.info(
            "MemoryWatchdog started (warning=%.0f%%, critical=%.0f%%)",
            self.warning_threshold * 100,
            self.critical_threshold * 100,
        )

    def stop(self) -> None:
        """Stop monitoring."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        logging.info("MemoryWatchdog stopped")

    def _monitor(self) -> None:
        """Main monitoring loop (runs in background thread)."""
        assert psutil is not None  # Guaranteed by __init__ check
        process = psutil.Process()

        while not self._stop_event.is_set():
            try:
                # Get memory info
                mem_info = self._get_memory_info(process)

                # Check thresholds
                if mem_info["percent"] >= self.critical_threshold:
                    if not self._critical_fired:
                        self._handle_critical(mem_info)
                        self._critical_fired = True

                elif mem_info["percent"] >= self.warning_threshold:
                    if not self._warning_fired:
                        self._handle_warning(mem_info)
                        self._warning_fired = True
                else:
                    # Reset flags when below warning threshold
                    self._warning_fired = False
                    self._critical_fired = False

            except Exception as e:
                logging.error("MemoryWatchdog error: %s", e)

            time.sleep(self.check_interval)

    def _get_memory_info(self, process) -> dict:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory statistics including process and system memory
        """
        assert psutil is not None  # Guaranteed by __init__ check
        # System memory
        system_mem = psutil.virtual_memory()

        # Process memory (RSS for Apple Silicon unified memory)
        proc_mem = process.memory_info()
        process_gb = proc_mem.rss / 1e9

        # System totals
        system_available_gb = system_mem.available / 1e9
        system_total_gb = system_mem.total / 1e9

        return {
            "process_gb": process_gb,
            "system_available_gb": system_available_gb,
            "system_total_gb": system_total_gb,
            "percent": system_mem.percent / 100.0,
            "timestamp": time.time(),
        }

    def _handle_warning(self, mem_info: dict) -> None:
        """Handle warning threshold exceeded."""
        logging.warning(
            "Memory warning: %.1f%% used (%.2fGB process, %.2fGB available)",
            mem_info["percent"] * 100,
            mem_info["process_gb"],
            mem_info["system_available_gb"],
        )

        if self.auto_cleanup:
            self._auto_cleanup()

        if self.on_warning:
            try:
                self.on_warning(mem_info)
            except Exception as e:
                logging.error("Error in warning callback: %s", e)

    def _handle_critical(self, mem_info: dict) -> None:
        """Handle critical threshold exceeded."""
        logging.error(
            "CRITICAL: Memory at %.1f%% (%.2fGB process, %.2fGB available)",
            mem_info["percent"] * 100,
            mem_info["process_gb"],
            mem_info["system_available_gb"],
        )

        if self.auto_cleanup:
            # Aggressive cleanup
            self._auto_cleanup()

        if self.on_critical:
            try:
                self.on_critical(mem_info)
            except Exception as e:
                logging.error("Error in critical callback: %s", e)
        else:
            # Default behavior: log error but don't raise (to avoid crashing in thread)
            logging.error(
                "Critical memory threshold exceeded. "
                "Consider reducing batch_size, max_tokens, or max_kv_size"
            )

    def _auto_cleanup(self) -> None:
        """Attempt automatic memory cleanup."""
        try:
            import gc

            import mlx.core as mx

            # Clear MLX Metal cache
            mx.clear_cache()

            # Force Python garbage collection
            gc.collect()

            logging.info("Automatic memory cleanup performed")
        except Exception as e:
            logging.error("Cleanup failed: %s", e)


class watchdog:
    """
    Context manager for memory watchdog.

    Provides easy-to-use context manager interface for memory monitoring
    during inference or other memory-intensive operations.

    Args:
        **kwargs: Arguments passed to MemoryWatchdog constructor

    Example:
        >>> from smlx.utils.watchdog import watchdog
        >>>
        >>> with watchdog(warning_threshold=0.80, auto_cleanup=True):
        ...     result = model.generate(prompt, max_tokens=1000)
    """

    def __init__(self, **kwargs):
        """
        Initialize watchdog context manager.

        Args:
            **kwargs: Arguments for MemoryWatchdog (warning_threshold, critical_threshold, etc.)
        """
        self.watchdog = MemoryWatchdog(**kwargs)

    def __enter__(self):
        """Start watchdog when entering context."""
        self.watchdog.start()
        return self.watchdog

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop watchdog when exiting context."""
        self.watchdog.stop()
        return False  # Don't suppress exceptions


__all__ = ["MemoryWatchdog", "watchdog"]
