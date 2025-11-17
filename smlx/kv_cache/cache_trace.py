# Copyright � 2025 SMLX Project

"""
Cache tracing and profiling utilities for debugging KV cache behavior.

Provides detailed logging and analysis tools for understanding cache operations,
memory usage patterns, and performance characteristics during generation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from smlx.utils.memory import get_active_memory_gb, get_cache_memory_gb, get_peak_memory_gb

if TYPE_CHECKING:
    from smlx.kv_cache.kv_manager import KVCacheManager


class CacheTracer:
    """
    Trace and profile KV cache operations.

    Records detailed information about cache updates, memory usage, and timing
    for debugging and performance analysis.

    Attributes:
        enabled: Whether tracing is currently active
        events: List of recorded trace events
        start_time: Timestamp when tracing started

    Example:
        >>> tracer = CacheTracer(enabled=True)
        >>>
        >>> # Record cache update
        >>> tracer.record_update(layer_idx=0, old_offset=0, new_offset=10, keys_shape=(1, 12, 10, 64))
        >>>
        >>> # Record memory snapshot
        >>> tracer.record_memory_snapshot()
        >>>
        >>> # Get summary
        >>> summary = tracer.get_summary()
        >>> print(f"Total updates: {summary['total_updates']}")
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize cache tracer.

        Args:
            enabled: Whether to enable tracing immediately (default: True)
        """
        self.enabled = enabled
        self.events: list[dict[str, Any]] = []
        self.start_time = time.time()

    def record_update(
        self,
        layer_idx: int,
        old_offset: int,
        new_offset: int,
        keys_shape: tuple,
        values_shape: tuple | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        """
        Record a cache update event.

        Args:
            layer_idx: Layer index
            old_offset: Cache offset before update
            new_offset: Cache offset after update
            keys_shape: Shape of keys tensor
            values_shape: Optional shape of values tensor
            elapsed_ms: Optional elapsed time in milliseconds

        Example:
            >>> tracer.record_update(
            ...     layer_idx=0,
            ...     old_offset=0,
            ...     new_offset=10,
            ...     keys_shape=(1, 12, 10, 64)
            ... )
        """
        if not self.enabled:
            return

        event = {
            "type": "update",
            "layer_idx": layer_idx,
            "old_offset": old_offset,
            "new_offset": new_offset,
            "tokens_added": new_offset - old_offset,
            "keys_shape": keys_shape,
            "timestamp": time.time(),
            "elapsed_since_start": time.time() - self.start_time,
        }

        if values_shape is not None:
            event["values_shape"] = values_shape

        if elapsed_ms is not None:
            event["elapsed_ms"] = elapsed_ms

        self.events.append(event)

    def record_memory_snapshot(self, label: str | None = None) -> None:
        """
        Record current memory usage snapshot.

        Args:
            label: Optional label for this snapshot

        Example:
            >>> tracer.record_memory_snapshot(label="after_prefill")
        """
        if not self.enabled:
            return

        event = {
            "type": "memory",
            "active_gb": get_active_memory_gb(),
            "cache_gb": get_cache_memory_gb(),
            "peak_gb": get_peak_memory_gb(),
            "timestamp": time.time(),
            "elapsed_since_start": time.time() - self.start_time,
        }

        if label is not None:
            event["label"] = label

        self.events.append(event)

    def record_rotation(self, layer_idx: int, old_idx: int, new_idx: int) -> None:
        """
        Record a cache rotation event.

        Args:
            layer_idx: Layer index
            old_idx: Index before rotation
            new_idx: Index after rotation

        Example:
            >>> tracer.record_rotation(layer_idx=0, old_idx=2048, new_idx=256)
        """
        if not self.enabled:
            return

        event = {
            "type": "rotation",
            "layer_idx": layer_idx,
            "old_idx": old_idx,
            "new_idx": new_idx,
            "rotated": new_idx < old_idx,
            "timestamp": time.time(),
            "elapsed_since_start": time.time() - self.start_time,
        }

        self.events.append(event)

    def record_custom(self, event_type: str, **kwargs) -> None:
        """
        Record a custom event.

        Args:
            event_type: Type of event
            **kwargs: Additional event data

        Example:
            >>> tracer.record_custom(
            ...     "quantization",
            ...     layer_idx=0,
            ...     bits=4,
            ...     group_size=64
            ... )
        """
        if not self.enabled:
            return

        event = {
            "type": event_type,
            "timestamp": time.time(),
            "elapsed_since_start": time.time() - self.start_time,
            **kwargs,
        }

        self.events.append(event)

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of trace data.

        Returns:
            Dictionary with trace statistics:
            - enabled: Whether tracing is enabled
            - total_events: Total number of events
            - total_updates: Number of update events
            - total_memory_snapshots: Number of memory snapshots
            - total_rotations: Number of rotation events
            - peak_memory_gb: Peak memory during trace
            - duration_seconds: Total trace duration
            - events_by_type: Count of events by type

        Example:
            >>> summary = tracer.get_summary()
            >>> print(f"Total updates: {summary['total_updates']}")
            >>> print(f"Peak memory: {summary['peak_memory_gb']:.2f} GB")
        """
        if not self.enabled:
            return {"enabled": False}

        if not self.events:
            # Return empty summary with enabled flag
            return {
                "enabled": True,
                "total_events": 0,
                "total_updates": 0,
                "total_memory_snapshots": 0,
                "total_rotations": 0,
                "peak_memory_gb": 0.0,
                "duration_seconds": 0.0,
                "events_by_type": {},
            }

        # Count events by type
        events_by_type: dict[str, int] = {}
        for event in self.events:
            event_type = event.get("type", "unknown")
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1

        # Find peak memory
        memory_events = [e for e in self.events if e.get("type") == "memory"]
        peak_memory_gb = max((e.get("peak_gb", 0) for e in memory_events), default=0)

        # Calculate duration
        if self.events:
            duration_seconds = self.events[-1]["elapsed_since_start"]
        else:
            duration_seconds = 0

        return {
            "enabled": True,
            "total_events": len(self.events),
            "total_updates": events_by_type.get("update", 0),
            "total_memory_snapshots": events_by_type.get("memory", 0),
            "total_rotations": events_by_type.get("rotation", 0),
            "peak_memory_gb": peak_memory_gb,
            "duration_seconds": duration_seconds,
            "events_by_type": events_by_type,
        }

    def get_layer_summary(self, layer_idx: int) -> dict[str, Any]:
        """
        Get summary for a specific layer.

        Args:
            layer_idx: Layer index to summarize

        Returns:
            Dictionary with layer-specific statistics

        Example:
            >>> layer_stats = tracer.get_layer_summary(layer_idx=0)
            >>> print(f"Layer 0 updates: {layer_stats['total_updates']}")
        """
        layer_events = [e for e in self.events if e.get("layer_idx") == layer_idx]

        if not layer_events:
            return {"layer_idx": layer_idx, "total_events": 0}

        updates = [e for e in layer_events if e.get("type") == "update"]
        rotations = [e for e in layer_events if e.get("type") == "rotation"]

        total_tokens_added = sum(e.get("tokens_added", 0) for e in updates)

        return {
            "layer_idx": layer_idx,
            "total_events": len(layer_events),
            "total_updates": len(updates),
            "total_rotations": len(rotations),
            "total_tokens_added": total_tokens_added,
        }

    def get_memory_timeline(self) -> list[dict]:
        """
        Get timeline of memory usage.

        Returns:
            List of memory snapshots in chronological order

        Example:
            >>> timeline = tracer.get_memory_timeline()
            >>> for snapshot in timeline:
            ...     print(f"t={snapshot['elapsed_since_start']:.2f}s: "
            ...           f"{snapshot['active_gb']:.2f} GB")
        """
        return [e for e in self.events if e.get("type") == "memory"]

    def export_json(self, path: str | Path) -> None:
        """
        Export trace to JSON file.

        Args:
            path: Path to output JSON file

        Example:
            >>> tracer.export_json("cache_trace.json")
        """
        path = Path(path)

        trace_data = {
            "summary": self.get_summary(),
            "events": self.events,
            "start_time": self.start_time,
        }

        with path.open("w") as f:
            json.dump(trace_data, f, indent=2, default=str)

    def import_json(self, path: str | Path) -> None:
        """
        Import trace from JSON file.

        Args:
            path: Path to input JSON file

        Example:
            >>> tracer.import_json("cache_trace.json")
        """
        path = Path(path)

        with path.open("r") as f:
            trace_data = json.load(f)

        self.events = trace_data.get("events", [])
        self.start_time = trace_data.get("start_time", time.time())

    def clear(self) -> None:
        """
        Clear all trace events.

        Example:
            >>> tracer.clear()
        """
        self.events.clear()
        self.start_time = time.time()

    def enable(self) -> None:
        """Enable tracing."""
        self.enabled = True

    def disable(self) -> None:
        """Disable tracing."""
        self.enabled = False


def trace_cache_manager(
    cache_manager: KVCacheManager,
    tracer: CacheTracer | None = None,
) -> CacheTracer:
    """
    Attach tracer to a cache manager's caches.

    Enables tracing on all caches in the manager and returns the tracer.

    Args:
        cache_manager: Cache manager to trace
        tracer: Optional existing tracer (creates new one if None)

    Returns:
        CacheTracer instance

    Example:
        >>> manager = KVCacheManager.create_standard(num_layers=24, enable_tracing=True)
        >>> tracer = trace_cache_manager(manager)
        >>>
        >>> # ... run generation ...
        >>>
        >>> summary = tracer.get_summary()
    """
    if tracer is None:
        tracer = CacheTracer(enabled=True)

    # Enable tracing on all caches
    for cache in cache_manager.caches:
        if hasattr(cache, "enable_tracing"):
            cache.enable_tracing = True

    return tracer


__all__ = ["CacheTracer", "trace_cache_manager"]
