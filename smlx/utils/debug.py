#!/usr/bin/env python3
# Copyright � 2025 SMLX Project

"""
Debug Tools for memory and performance diagnostics.

Provides utilities for diagnosing memory leaks, computation graph accumulation,
and performance bottlenecks in MLX models.

Features:
- Memory snapshot comparison
- Module-level memory tracking
- Computation graph analysis
- Layer-by-layer memory profiling
- Memory leak detection

Example:
    >>> from smlx.utils.debug import MemorySnapshot, compare_snapshots
    >>>
    >>> # Take snapshots before/after
    >>> before = MemorySnapshot.capture("Before generation")
    >>> result = model.generate(prompt)
    >>> after = MemorySnapshot.capture("After generation")
    >>>
    >>> # Compare
    >>> diff = compare_snapshots(before, after)
    >>> print(diff)
"""

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from smlx.utils.memory import (
    get_active_memory_gb,
    get_cache_memory_gb,
    get_peak_memory_gb,
)

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """
    Snapshot of memory state at a specific point in time.

    Attributes:
        timestamp: When snapshot was captured
        label: Human-readable label
        active_gb: Active memory in GB
        cache_gb: Cache memory in GB
        peak_gb: Peak memory in GB
        total_gb: Total memory (active + cache)
        python_objects: Count of Python objects (if gc tracking enabled)
        metadata: Additional metadata
    """

    timestamp: float
    label: str
    active_gb: float
    cache_gb: float
    peak_gb: float
    total_gb: float
    python_objects: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls, label: str = "", track_python: bool = False
    ) -> "MemorySnapshot":
        """
        Capture current memory state.

        Args:
            label: Human-readable label for snapshot
            track_python: Track Python object count

        Returns:
            MemorySnapshot of current state

        Example:
            >>> snapshot = MemorySnapshot.capture("After model load")
            >>> print(f"Memory: {snapshot.total_gb:.2f}GB")
        """
        active_gb = get_active_memory_gb()
        cache_gb = get_cache_memory_gb()
        peak_gb = get_peak_memory_gb()
        total_gb = active_gb + cache_gb

        python_objects = None
        if track_python:
            python_objects = len(gc.get_objects())

        return cls(
            timestamp=time.time(),
            label=label,
            active_gb=active_gb,
            cache_gb=cache_gb,
            peak_gb=peak_gb,
            total_gb=total_gb,
            python_objects=python_objects,
        )

    def __str__(self) -> str:
        """String representation of snapshot."""
        parts = [
            f"[{self.label}]" if self.label else "[Snapshot]",
            f"Active: {self.active_gb:.2f}GB",
            f"Cache: {self.cache_gb:.2f}GB",
            f"Peak: {self.peak_gb:.2f}GB",
            f"Total: {self.total_gb:.2f}GB",
        ]
        if self.python_objects is not None:
            parts.append(f"PyObjects: {self.python_objects:,}")
        return " | ".join(parts)


@dataclass
class SnapshotDiff:
    """
    Difference between two memory snapshots.

    Attributes:
        before: Earlier snapshot
        after: Later snapshot
        delta_active_gb: Change in active memory
        delta_cache_gb: Change in cache memory
        delta_total_gb: Change in total memory
        delta_python_objects: Change in Python object count
        elapsed_time: Time between snapshots
    """

    before: MemorySnapshot
    after: MemorySnapshot
    delta_active_gb: float
    delta_cache_gb: float
    delta_total_gb: float
    delta_python_objects: Optional[int] = None
    elapsed_time: float = 0.0

    def __str__(self) -> str:
        """String representation of diff."""
        parts = [
            f"Memory change: {self.before.label} � {self.after.label}",
            f"Active: {self.delta_active_gb:+.2f}GB",
            f"Cache: {self.delta_cache_gb:+.2f}GB",
            f"Total: {self.delta_total_gb:+.2f}GB",
            f"Time: {self.elapsed_time:.2f}s",
        ]
        if self.delta_python_objects is not None:
            parts.append(f"PyObjects: {self.delta_python_objects:+,}")
        return " | ".join(parts)


def compare_snapshots(
    before: MemorySnapshot, after: MemorySnapshot
) -> SnapshotDiff:
    """
    Compare two memory snapshots.

    Args:
        before: Earlier snapshot
        after: Later snapshot

    Returns:
        SnapshotDiff showing changes

    Example:
        >>> before = MemorySnapshot.capture("Start")
        >>> # ... do work ...
        >>> after = MemorySnapshot.capture("End")
        >>> diff = compare_snapshots(before, after)
        >>> if diff.delta_total_gb > 1.0:
        ...     print("Warning: Memory increased by >1GB!")
    """
    delta_python_objects = None
    if before.python_objects is not None and after.python_objects is not None:
        delta_python_objects = after.python_objects - before.python_objects

    return SnapshotDiff(
        before=before,
        after=after,
        delta_active_gb=after.active_gb - before.active_gb,
        delta_cache_gb=after.cache_gb - before.cache_gb,
        delta_total_gb=after.total_gb - before.total_gb,
        delta_python_objects=delta_python_objects,
        elapsed_time=after.timestamp - before.timestamp,
    )


@contextmanager
def memory_snapshot_context(label: str = "Operation", track_python: bool = False):
    """
    Context manager for automatic memory snapshot comparison.

    Args:
        label: Label for the operation
        track_python: Track Python object count

    Yields:
        SnapshotDiff object (populated on exit)

    Example:
        >>> with memory_snapshot_context("Model inference") as diff:
        ...     result = model.generate(prompt)
        >>> print(diff)
    """
    before = MemorySnapshot.capture(f"{label} (start)", track_python=track_python)

    # Create diff container that will be populated on exit
    diff_container: dict[str, Optional[SnapshotDiff]] = {"diff": None}

    try:
        yield diff_container
    finally:
        after = MemorySnapshot.capture(f"{label} (end)", track_python=track_python)
        diff = compare_snapshots(before, after)
        diff_container["diff"] = diff
        logger.info(str(diff))


class LayerMemoryProfiler:
    """
    Profile memory usage layer-by-layer through a model.

    Useful for identifying which layers consume the most memory
    or cause memory leaks.

    Example:
        >>> profiler = LayerMemoryProfiler()
        >>>
        >>> # Profile each layer
        >>> for i, layer in enumerate(model.layers):
        ...     with profiler.profile_layer(f"Layer {i}"):
        ...         output = layer(input)
        ...         mx.eval(output)
        >>>
        >>> # Get report
        >>> report = profiler.get_report()
        >>> print(report)
    """

    def __init__(self):
        self.snapshots: list[MemorySnapshot] = []
        self.layer_names: list[str] = []

    @contextmanager
    def profile_layer(self, layer_name: str):
        """
        Profile memory for a single layer.

        Args:
            layer_name: Name/label for the layer

        Example:
            >>> profiler = LayerMemoryProfiler()
            >>> with profiler.profile_layer("Attention"):
            ...     output = attention_layer(input)
            ...     mx.eval(output)
        """
        snapshot_before = MemorySnapshot.capture(f"{layer_name} (before)")

        try:
            yield
        finally:
            snapshot_after = MemorySnapshot.capture(f"{layer_name} (after)")
            self.snapshots.append(snapshot_before)
            self.snapshots.append(snapshot_after)
            self.layer_names.append(layer_name)

    def get_report(self) -> str:
        """
        Generate memory profile report.

        Returns:
            Formatted report string

        Example:
            >>> profiler = LayerMemoryProfiler()
            >>> # ... profile layers ...
            >>> print(profiler.get_report())
        """
        if len(self.snapshots) < 2:
            return "No profiling data available"

        lines = ["Layer Memory Profile", "=" * 80]

        for i, layer_name in enumerate(self.layer_names):
            before_idx = i * 2
            after_idx = i * 2 + 1

            if after_idx >= len(self.snapshots):
                break

            before = self.snapshots[before_idx]
            after = self.snapshots[after_idx]
            diff = compare_snapshots(before, after)

            lines.append(
                f"{layer_name:30} | "
                f"� Active: {diff.delta_active_gb:+.3f}GB | "
                f"� Cache: {diff.delta_cache_gb:+.3f}GB | "
                f"� Total: {diff.delta_total_gb:+.3f}GB"
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    def find_worst_layers(self, top_n: int = 5) -> list[tuple[str, float]]:
        """
        Find layers with highest memory increase.

        Args:
            top_n: Number of top layers to return

        Returns:
            List of (layer_name, delta_total_gb) tuples

        Example:
            >>> profiler = LayerMemoryProfiler()
            >>> # ... profile layers ...
            >>> worst = profiler.find_worst_layers(top_n=3)
            >>> for name, delta_gb in worst:
            ...     print(f"{name}: +{delta_gb:.2f}GB")
        """
        deltas = []

        for i, layer_name in enumerate(self.layer_names):
            before_idx = i * 2
            after_idx = i * 2 + 1

            if after_idx >= len(self.snapshots):
                break

            before = self.snapshots[before_idx]
            after = self.snapshots[after_idx]
            delta_total = after.total_gb - before.total_gb

            deltas.append((layer_name, delta_total))

        # Sort by delta (descending)
        deltas.sort(key=lambda x: x[1], reverse=True)

        return deltas[:top_n]


class GraphAccumulationDetector:
    """
    Detect computation graph accumulation issues.

    Monitors if memory continuously grows across iterations,
    indicating missing mx.eval() calls.

    Example:
        >>> detector = GraphAccumulationDetector(threshold=0.1)
        >>>
        >>> for i in range(100):
        ...     output = model(input)
        ...     detector.record()
        ...
        ...     if detector.is_accumulating():
        ...         print("WARNING: Graph accumulation detected!")
        ...         break
    """

    def __init__(
        self, window_size: int = 10, threshold_gb: float = 0.1, verbose: bool = True
    ):
        """
        Initialize detector.

        Args:
            window_size: Number of iterations to analyze
            threshold_gb: Memory growth threshold per iteration (GB)
            verbose: Log warnings
        """
        self.window_size = window_size
        self.threshold_gb = threshold_gb
        self.verbose = verbose
        self.measurements: list[float] = []

    def record(self) -> None:
        """
        Record current memory usage.

        Should be called once per iteration.
        """
        total_gb = get_active_memory_gb() + get_cache_memory_gb()
        self.measurements.append(total_gb)

        # Keep only recent window
        if len(self.measurements) > self.window_size * 2:
            self.measurements = self.measurements[-self.window_size :]

    def is_accumulating(self) -> bool:
        """
        Check if graph accumulation is detected.

        Returns:
            True if memory is steadily increasing

        Example:
            >>> detector = GraphAccumulationDetector()
            >>> for i in range(50):
            ...     output = model(input)
            ...     detector.record()
            ...     if detector.is_accumulating():
            ...         print("Accumulation detected at iteration", i)
            ...         break
        """
        if len(self.measurements) < self.window_size:
            return False  # Not enough data

        # Calculate average growth rate
        recent = self.measurements[-self.window_size :]
        growth_per_iteration = (recent[-1] - recent[0]) / len(recent)

        is_growing = growth_per_iteration > self.threshold_gb

        if is_growing and self.verbose:
            logger.warning(
                f"Graph accumulation detected: "
                f"{growth_per_iteration:.3f}GB/iteration "
                f"(threshold: {self.threshold_gb:.3f}GB)"
            )

        return is_growing

    def get_growth_rate(self) -> float:
        """
        Get current memory growth rate.

        Returns:
            Growth rate in GB/iteration

        Example:
            >>> detector = GraphAccumulationDetector()
            >>> # ... record measurements ...
            >>> rate = detector.get_growth_rate()
            >>> print(f"Growth rate: {rate:.3f}GB/iteration")
        """
        if len(self.measurements) < 2:
            return 0.0

        recent = self.measurements[-self.window_size :]
        if len(recent) < 2:
            return 0.0

        return (recent[-1] - recent[0]) / len(recent)


def print_memory_state(label: str = "Memory State") -> None:
    """
    Print current memory state to console.

    Args:
        label: Label for the printout

    Example:
        >>> print_memory_state("After model load")
        >>> model.generate(prompt)
        >>> print_memory_state("After generation")
    """
    snapshot = MemorySnapshot.capture(label)
    print(snapshot)


def detect_leaking_modules() -> list[tuple[str, int]]:
    """
    Detect Python modules that may be leaking objects.

    Uses garbage collector to find objects that aren't being freed.

    Returns:
        List of (module_name, object_count) tuples

    Example:
        >>> # After running inference
        >>> leaking = detect_leaking_modules()
        >>> for module, count in leaking[:5]:
        ...     print(f"{module}: {count} objects")
    """
    gc.collect()  # Force collection first

    # Count objects by type
    type_counts: dict[str, int] = {}

    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        module = type(obj).__module__
        key = f"{module}.{obj_type}"
        type_counts[key] = type_counts.get(key, 0) + 1

    # Sort by count (descending)
    sorted_counts = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

    return sorted_counts


__all__ = [
    'MemorySnapshot',
    'SnapshotDiff',
    'compare_snapshots',
    'memory_snapshot_context',
    'LayerMemoryProfiler',
    'GraphAccumulationDetector',
    'print_memory_state',
    'detect_leaking_modules',
]
