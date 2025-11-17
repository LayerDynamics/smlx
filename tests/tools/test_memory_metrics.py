"""
Tests for memory metrics tool.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the Python interpreter in the smlx conda environment
PYTHON_PATH = "/Users/ryanoboyle/miniforge3/envs/smlx/bin/python"


@pytest.mark.unit
def test_memory_metrics_basic():
    """Test basic memory metrics collection."""
    result = subprocess.run(
        [PYTHON_PATH, "-m", "smlx.tools.memory_metrics", "--top", "5"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "SYSTEM MEMORY OVERVIEW" in result.stdout
    assert "PROCESS MEMORY USAGE" in result.stdout
    assert "Total Memory:" in result.stdout


@pytest.mark.unit
def test_memory_metrics_json_output():
    """Test JSON output format."""
    result = subprocess.run(
        [PYTHON_PATH, "-m", "smlx.tools.memory_metrics", "--top", "3", "--json"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0

    # Verify it's valid JSON
    data = json.loads(result.stdout)

    # Check structure
    assert "timestamp" in data
    assert "memory_pressure" in data
    assert "total_processes" in data
    assert "system" in data
    assert "processes" in data

    # Check system info
    system = data["system"]
    assert "total_mb" in system
    assert "used_mb" in system
    assert "available_mb" in system
    assert "percent" in system

    # Check processes
    assert isinstance(data["processes"], list)
    assert len(data["processes"]) >= 0


@pytest.mark.unit
def test_memory_metrics_csv_output():
    """Test CSV output format."""
    result = subprocess.run(
        [PYTHON_PATH, "-m", "smlx.tools.memory_metrics", "--top", "3", "--csv"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0

    # Check CSV header
    lines = result.stdout.strip().split("\n")
    assert len(lines) > 0
    header = lines[0]
    assert "PID" in header
    assert "Name" in header
    assert "RSS_MB" in header
    assert "Memory_Percent" in header


@pytest.mark.unit
def test_memory_metrics_threshold():
    """Test threshold filtering."""
    result = subprocess.run(
        [PYTHON_PATH, "-m", "smlx.tools.memory_metrics", "--threshold", "1000"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    # Should have some output even with high threshold
    assert "SYSTEM MEMORY OVERVIEW" in result.stdout


@pytest.mark.unit
def test_memory_metrics_help():
    """Test help text."""
    result = subprocess.run(
        [PYTHON_PATH, "-m", "smlx.tools.memory_metrics", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "Monitor system memory usage" in result.stdout
    assert "--top" in result.stdout
    assert "--threshold" in result.stdout
    assert "--json" in result.stdout
    assert "--csv" in result.stdout


@pytest.mark.unit
def test_memory_metrics_module_import():
    """Test that the module can be imported and used programmatically."""
    from smlx.tools.memory_metrics import (
        MemoryMetricsCollector,
        MemoryMetricsFormatter,
    )

    # Test collector
    collector = MemoryMetricsCollector()
    metrics = collector.collect_metrics(threshold_mb=100.0)

    assert metrics is not None
    assert metrics.system is not None
    assert metrics.processes is not None
    assert metrics.timestamp is not None
    assert metrics.memory_pressure in ["normal", "warn", "critical", "unknown"]

    # Test formatter
    formatter = MemoryMetricsFormatter()

    # Test text output
    text_output = formatter.format_metrics(metrics, top_n=5)
    assert "SYSTEM MEMORY OVERVIEW" in text_output

    # Test JSON output
    json_output = formatter.to_json(metrics)
    data = json.loads(json_output)
    assert "system" in data
    assert "processes" in data

    # Test CSV output
    csv_output = formatter.to_csv(metrics)
    assert "PID,Name" in csv_output


@pytest.mark.unit
def test_memory_pressure_detection():
    """Test memory pressure detection."""
    from smlx.tools.memory_metrics import MemoryMetricsCollector

    collector = MemoryMetricsCollector()
    pressure = collector.get_memory_pressure()

    assert pressure in ["normal", "warn", "critical", "unknown"]


@pytest.mark.unit
def test_system_memory_info():
    """Test system memory info collection."""
    from smlx.tools.memory_metrics import MemoryMetricsCollector

    collector = MemoryMetricsCollector()
    system_info = collector.get_system_memory()

    assert system_info is not None
    assert system_info.total_mb > 0
    assert system_info.used_mb >= 0
    assert system_info.available_mb >= 0
    assert 0 <= system_info.percent <= 100


@pytest.mark.unit
def test_process_memory_collection():
    """Test process memory collection."""
    from smlx.tools.memory_metrics import MemoryMetricsCollector

    collector = MemoryMetricsCollector()
    processes = collector.get_process_memory(threshold_mb=0.0)

    assert processes is not None
    assert isinstance(processes, list)
    assert len(processes) > 0

    # Check first process has expected fields
    if processes:
        proc = processes[0]
        assert hasattr(proc, "pid")
        assert hasattr(proc, "name")
        assert hasattr(proc, "rss_mb")
        assert hasattr(proc, "vms_mb")
        assert hasattr(proc, "percent")
        assert proc.rss_mb >= 0
        assert proc.vms_mb >= 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
