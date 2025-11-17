"""Tests for smlx.kv_cache.cache_trace module."""

import json
import tempfile
from pathlib import Path

import pytest

from smlx.kv_cache.cache_trace import CacheTracer, trace_cache_manager
from smlx.kv_cache.kv_manager import KVCacheManager


class TestCacheTracer:
    """Test CacheTracer class."""

    def test_cache_tracer_init(self):
        """Test CacheTracer initialization."""
        tracer = CacheTracer(enabled=True)

        assert tracer.enabled is True
        assert isinstance(tracer.events, list)
        assert len(tracer.events) == 0
        assert tracer.start_time > 0

    def test_cache_tracer_init_disabled(self):
        """Test CacheTracer initialization with tracing disabled."""
        tracer = CacheTracer(enabled=False)

        assert tracer.enabled is False

    @pytest.mark.unit
    def test_record_update(self):
        """Test recording cache update event."""
        tracer = CacheTracer(enabled=True)

        tracer.record_update(
            layer_idx=0,
            old_offset=0,
            new_offset=10,
            keys_shape=(1, 12, 10, 64),
            values_shape=(1, 12, 10, 64),
        )

        assert len(tracer.events) == 1
        event = tracer.events[0]

        assert event["type"] == "update"
        assert event["layer_idx"] == 0
        assert event["old_offset"] == 0
        assert event["new_offset"] == 10
        assert event["tokens_added"] == 10
        assert event["keys_shape"] == (1, 12, 10, 64)
        assert event["values_shape"] == (1, 12, 10, 64)
        assert "timestamp" in event
        assert "elapsed_since_start" in event

    @pytest.mark.unit
    def test_record_update_with_elapsed(self):
        """Test recording update with elapsed time."""
        tracer = CacheTracer(enabled=True)

        tracer.record_update(
            layer_idx=0,
            old_offset=0,
            new_offset=10,
            keys_shape=(1, 12, 10, 64),
            elapsed_ms=1.5,
        )

        event = tracer.events[0]
        assert event["elapsed_ms"] == 1.5

    @pytest.mark.unit
    def test_record_update_disabled(self):
        """Test that recording does nothing when disabled."""
        tracer = CacheTracer(enabled=False)

        tracer.record_update(
            layer_idx=0,
            old_offset=0,
            new_offset=10,
            keys_shape=(1, 12, 10, 64),
        )

        # No events should be recorded
        assert len(tracer.events) == 0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_record_memory_snapshot(self):
        """Test recording memory snapshot."""
        tracer = CacheTracer(enabled=True)

        tracer.record_memory_snapshot(label="test_snapshot")

        assert len(tracer.events) == 1
        event = tracer.events[0]

        assert event["type"] == "memory"
        assert event["label"] == "test_snapshot"
        assert "active_gb" in event
        assert "cache_gb" in event
        assert "peak_gb" in event
        assert "timestamp" in event

    @pytest.mark.unit
    def test_record_rotation(self):
        """Test recording rotation event."""
        tracer = CacheTracer(enabled=True)

        tracer.record_rotation(layer_idx=0, old_idx=2048, new_idx=256)

        assert len(tracer.events) == 1
        event = tracer.events[0]

        assert event["type"] == "rotation"
        assert event["layer_idx"] == 0
        assert event["old_idx"] == 2048
        assert event["new_idx"] == 256
        assert event["rotated"] is True

    @pytest.mark.unit
    def test_record_custom(self):
        """Test recording custom event."""
        tracer = CacheTracer(enabled=True)

        tracer.record_custom(
            "quantization",
            layer_idx=0,
            bits=4,
            group_size=64,
        )

        assert len(tracer.events) == 1
        event = tracer.events[0]

        assert event["type"] == "quantization"
        assert event["layer_idx"] == 0
        assert event["bits"] == 4
        assert event["group_size"] == 64

    @pytest.mark.unit
    def test_get_summary_empty(self):
        """Test getting summary with no events."""
        tracer = CacheTracer(enabled=True)

        summary = tracer.get_summary()

        assert summary["enabled"] is True
        assert summary["total_events"] == 0

    @pytest.mark.unit
    def test_get_summary_with_events(self):
        """Test getting summary with events."""
        tracer = CacheTracer(enabled=True)

        # Add various events
        tracer.record_update(0, 0, 10, (1, 12, 10, 64))
        tracer.record_update(1, 0, 10, (1, 12, 10, 64))
        tracer.record_memory_snapshot()
        tracer.record_rotation(0, 100, 50)

        summary = tracer.get_summary()

        assert summary["enabled"] is True
        assert summary["total_events"] == 4
        assert summary["total_updates"] == 2
        assert summary["total_memory_snapshots"] == 1
        assert summary["total_rotations"] == 1
        assert "duration_seconds" in summary
        assert "events_by_type" in summary

    @pytest.mark.unit
    def test_get_summary_disabled(self):
        """Test getting summary when tracing is disabled."""
        tracer = CacheTracer(enabled=False)

        summary = tracer.get_summary()

        assert summary["enabled"] is False

    @pytest.mark.unit
    def test_get_layer_summary(self):
        """Test getting summary for specific layer."""
        tracer = CacheTracer(enabled=True)

        # Add events for different layers
        tracer.record_update(0, 0, 10, (1, 12, 10, 64))
        tracer.record_update(0, 10, 15, (1, 12, 5, 64))
        tracer.record_update(1, 0, 10, (1, 12, 10, 64))
        tracer.record_rotation(0, 100, 50)

        layer_0_summary = tracer.get_layer_summary(layer_idx=0)

        assert layer_0_summary["layer_idx"] == 0
        assert layer_0_summary["total_events"] == 3  # 2 updates + 1 rotation
        assert layer_0_summary["total_updates"] == 2
        assert layer_0_summary["total_rotations"] == 1
        assert layer_0_summary["total_tokens_added"] == 15

    @pytest.mark.unit
    def test_get_layer_summary_empty(self):
        """Test getting summary for layer with no events."""
        tracer = CacheTracer(enabled=True)

        summary = tracer.get_layer_summary(layer_idx=5)

        assert summary["layer_idx"] == 5
        assert summary["total_events"] == 0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_memory_timeline(self):
        """Test getting memory timeline."""
        tracer = CacheTracer(enabled=True)

        # Add memory snapshots
        tracer.record_memory_snapshot(label="start")
        tracer.record_update(0, 0, 10, (1, 12, 10, 64))
        tracer.record_memory_snapshot(label="mid")
        tracer.record_update(0, 10, 20, (1, 12, 10, 64))
        tracer.record_memory_snapshot(label="end")

        timeline = tracer.get_memory_timeline()

        assert len(timeline) == 3
        assert all(e["type"] == "memory" for e in timeline)
        assert timeline[0]["label"] == "start"
        assert timeline[1]["label"] == "mid"
        assert timeline[2]["label"] == "end"

    @pytest.mark.unit
    def test_export_json(self):
        """Test exporting trace to JSON."""
        tracer = CacheTracer(enabled=True)

        tracer.record_update(0, 0, 10, (1, 12, 10, 64))
        tracer.record_memory_snapshot(label="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "trace.json"
            tracer.export_json(json_path)

            # Verify file exists and is valid JSON
            assert json_path.exists()

            with json_path.open("r") as f:
                data = json.load(f)

            assert "summary" in data
            assert "events" in data
            assert "start_time" in data
            assert len(data["events"]) == 2

    @pytest.mark.unit
    def test_import_json(self):
        """Test importing trace from JSON."""
        tracer1 = CacheTracer(enabled=True)
        tracer1.record_update(0, 0, 10, (1, 12, 10, 64))

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "trace.json"
            tracer1.export_json(json_path)

            # Import into new tracer
            tracer2 = CacheTracer(enabled=True)
            tracer2.import_json(json_path)

            assert len(tracer2.events) == len(tracer1.events)

    @pytest.mark.unit
    def test_clear(self):
        """Test clearing trace events."""
        tracer = CacheTracer(enabled=True)

        tracer.record_update(0, 0, 10, (1, 12, 10, 64))
        tracer.record_memory_snapshot()

        assert len(tracer.events) > 0

        tracer.clear()

        assert len(tracer.events) == 0

    @pytest.mark.unit
    def test_enable_disable(self):
        """Test enabling and disabling tracer."""
        tracer = CacheTracer(enabled=False)

        tracer.record_update(0, 0, 10, (1, 12, 10, 64))
        assert len(tracer.events) == 0

        tracer.enable()
        assert tracer.enabled is True

        tracer.record_update(0, 0, 10, (1, 12, 10, 64))
        assert len(tracer.events) == 1

        tracer.disable()
        assert tracer.enabled is False

        tracer.record_update(1, 0, 10, (1, 12, 10, 64))
        assert len(tracer.events) == 1  # No new event


class TestTraceCacheManager:
    """Test trace_cache_manager function."""

    @pytest.mark.unit
    def test_trace_cache_manager_new_tracer(self):
        """Test attaching new tracer to cache manager."""
        manager = KVCacheManager.create_standard(
            num_layers=6,
            enable_tracing=False,  # Start with tracing disabled
        )

        tracer = trace_cache_manager(manager)

        assert isinstance(tracer, CacheTracer)
        assert tracer.enabled is True

        # All caches should have tracing enabled
        for cache in manager.caches:
            assert cache.enable_tracing is True

    @pytest.mark.unit
    def test_trace_cache_manager_existing_tracer(self):
        """Test attaching existing tracer to cache manager."""
        manager = KVCacheManager.create_standard(
            num_layers=6,
            enable_tracing=False,
        )

        existing_tracer = CacheTracer(enabled=True)
        tracer = trace_cache_manager(manager, existing_tracer)

        assert tracer is existing_tracer

        # All caches should have tracing enabled
        for cache in manager.caches:
            assert cache.enable_tracing is True

    @pytest.mark.unit
    def test_trace_cache_manager_already_enabled(self):
        """Test attaching tracer when tracing already enabled."""
        manager = KVCacheManager.create_standard(
            num_layers=6,
            enable_tracing=True,  # Already enabled
        )

        tracer = trace_cache_manager(manager)

        # Should still work
        assert tracer.enabled is True


class TestCacheTracerIntegration:
    """Test integration scenarios for cache tracer."""

    @pytest.mark.unit
    def test_trace_multi_layer_generation(self):
        """Test tracing multi-layer generation."""
        tracer = CacheTracer(enabled=True)

        # Simulate multi-layer forward pass
        for layer_idx in range(6):
            tracer.record_update(
                layer_idx=layer_idx,
                old_offset=0,
                new_offset=10,
                keys_shape=(1, 12, 10, 64),
                values_shape=(1, 12, 10, 64),
            )

        summary = tracer.get_summary()

        assert summary["total_updates"] == 6

        # Check each layer
        for layer_idx in range(6):
            layer_summary = tracer.get_layer_summary(layer_idx)
            assert layer_summary["total_updates"] == 1

    @pytest.mark.unit
    def test_trace_autoregressive_generation(self):
        """Test tracing autoregressive generation."""
        tracer = CacheTracer(enabled=True)

        # Initial prompt
        tracer.record_update(0, 0, 10, (1, 12, 10, 64))
        tracer.record_memory_snapshot(label="after_prompt")

        # Generate tokens
        for i in range(10):
            tracer.record_update(0, 10 + i, 11 + i, (1, 12, 1, 64))

        tracer.record_memory_snapshot(label="after_generation")

        summary = tracer.get_summary()

        assert summary["total_updates"] == 11
        assert summary["total_memory_snapshots"] == 2

        layer_summary = tracer.get_layer_summary(0)
        assert layer_summary["total_tokens_added"] == 20

    @pytest.mark.unit
    def test_trace_with_rotation(self):
        """Test tracing with cache rotation."""
        tracer = CacheTracer(enabled=True)

        # Add tokens until rotation
        for i in range(100):
            tracer.record_update(0, i, i + 1, (1, 12, 1, 64))

            # Simulate rotation at 50 tokens
            if i == 49:
                tracer.record_rotation(0, 50, 10)

        summary = tracer.get_summary()

        assert summary["total_updates"] == 100
        assert summary["total_rotations"] == 1


class TestCacheTracerEdgeCases:
    """Test edge cases for cache tracer."""

    @pytest.mark.unit
    def test_export_import_empty_trace(self):
        """Test exporting and importing empty trace."""
        tracer1 = CacheTracer(enabled=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "empty_trace.json"
            tracer1.export_json(json_path)

            tracer2 = CacheTracer(enabled=True)
            tracer2.import_json(json_path)

            assert len(tracer2.events) == 0

    @pytest.mark.unit
    def test_get_memory_timeline_empty(self):
        """Test getting memory timeline with no memory events."""
        tracer = CacheTracer(enabled=True)

        tracer.record_update(0, 0, 10, (1, 12, 10, 64))

        timeline = tracer.get_memory_timeline()

        assert len(timeline) == 0

    @pytest.mark.unit
    def test_record_rotation_no_rotation(self):
        """Test recording rotation when no rotation occurred."""
        tracer = CacheTracer(enabled=True)

        # new_idx >= old_idx means no rotation
        tracer.record_rotation(0, 50, 60)

        event = tracer.events[0]
        assert event["rotated"] is False

    @pytest.mark.unit
    def test_multiple_clear_operations(self):
        """Test multiple clear operations."""
        tracer = CacheTracer(enabled=True)

        tracer.record_update(0, 0, 10, (1, 12, 10, 64))
        tracer.clear()

        assert len(tracer.events) == 0

        tracer.record_update(0, 0, 10, (1, 12, 10, 64))
        tracer.clear()

        assert len(tracer.events) == 0

    @pytest.mark.unit
    def test_events_by_type_count(self):
        """Test event count by type."""
        tracer = CacheTracer(enabled=True)

        # Add various event types
        tracer.record_update(0, 0, 10, (1, 12, 10, 64))
        tracer.record_update(0, 10, 20, (1, 12, 10, 64))
        tracer.record_memory_snapshot()
        tracer.record_rotation(0, 100, 50)
        tracer.record_custom("test_event", data="test")

        summary = tracer.get_summary()

        events_by_type = summary["events_by_type"]
        assert events_by_type.get("update", 0) == 2
        assert events_by_type.get("memory", 0) == 1
        assert events_by_type.get("rotation", 0) == 1
        assert events_by_type.get("test_event", 0) == 1
