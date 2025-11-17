# Memory Metrics Tool

Comprehensive system memory monitoring tool for tracking memory usage across the system and per-process consumption.

## Overview

The Memory Metrics tool provides detailed insights into:
- **System-wide memory statistics** (total, used, available, active, inactive, wired, swap)
- **Per-process memory consumption** (RSS, VMS, percentage, CPU usage)
- **Memory pressure indicators** (normal, warn, critical)
- **Flexible output formats** (text, JSON, CSV)
- **Real-time monitoring** with watch mode

## Quick Start

```bash
# Show top 10 memory-consuming processes
python -m smlx.tools.memory_metrics --top 10

# Show only processes using > 100MB
python -m smlx.tools.memory_metrics --threshold 100

# Watch mode (refresh every 2 seconds)
python -m smlx.tools.memory_metrics --watch 2 --top 20

# Export to JSON
python -m smlx.tools.memory_metrics --json --output memory.json

# Verbose output with detailed process information
python -m smlx.tools.memory_metrics --verbose --top 5
```

## Installation

The tool requires `psutil` for cross-platform memory information:

```bash
pip install psutil
```

Or install with the full SMLX package:

```bash
pip install -e ".[dev]"
```

## Command-Line Usage

### Basic Options

```bash
# Show all processes (sorted by memory usage)
python -m smlx.tools.memory_metrics

# Show top N processes
python -m smlx.tools.memory_metrics --top 20

# Filter by memory threshold (MB)
python -m smlx.tools.memory_metrics --threshold 100
```

### Sorting

```bash
# Sort by RSS memory (default)
python -m smlx.tools.memory_metrics --sort rss

# Sort by virtual memory
python -m smlx.tools.memory_metrics --sort vms

# Sort by memory percentage
python -m smlx.tools.memory_metrics --sort percent

# Sort by CPU usage
python -m smlx.tools.memory_metrics --sort cpu

# Sort by process name
python -m smlx.tools.memory_metrics --sort name
```

### Output Formats

#### Text Format (Default)

```bash
python -m smlx.tools.memory_metrics --top 10
```

Output:
```
Memory Metrics Snapshot - 2025-11-16T19:00:00.000000
Memory Pressure: NORMAL

================================================================================
SYSTEM MEMORY OVERVIEW
================================================================================
Total Memory:        36864.00 MB
Used Memory:         12579.00 MB (74.8%)
Available Memory:     9271.95 MB
Free Memory:           410.25 MB

Memory Breakdown:
  Active:             8706.89 MB
  Inactive:           8808.39 MB
  Wired:              3872.11 MB

Swap Memory:
  Total:              6144.00 MB
  Used:               4778.38 MB (77.8%)
  Free:               1365.62 MB
================================================================================

PROCESS MEMORY USAGE (Top 10)
============================================================================================
PID      USER           RSS (MB)   VMS (MB)   MEM%   CPU%  THREADS NAME
============================================================================================
56100    ryanoboyle      3948.89 1828736.64  10.71   0.00       28 Browser Helper (Renderer)
1934     ryanoboyle      1645.09 1826588.50   4.46   0.00       17 Code Helper (Plugin)
...
============================================================================================
```

#### JSON Format

```bash
python -m smlx.tools.memory_metrics --json --output memory.json
```

Output structure:
```json
{
  "timestamp": "2025-11-16T19:00:00.000000",
  "memory_pressure": "normal",
  "total_processes": 429,
  "system": {
    "total_mb": 36864.0,
    "available_mb": 9297.52,
    "used_mb": 12594.05,
    "free_mb": 207.77,
    "percent": 74.8,
    "active_mb": 9082.98,
    "inactive_mb": 9025.81,
    "wired_mb": 3511.06,
    "swap_total_mb": 6144.0,
    "swap_used_mb": 4706.38,
    "swap_free_mb": 1437.62,
    "swap_percent": 76.6
  },
  "processes": [
    {
      "pid": 56100,
      "name": "Browser Helper (Renderer)",
      "status": "running",
      "rss_mb": 3948.89,
      "vms_mb": 1828736.64,
      "percent": 10.71,
      "num_threads": 28,
      "cpu_percent": 0.0,
      "username": "ryanoboyle",
      "cmdline": "/Applications/Arc.app/..."
    }
  ]
}
```

#### CSV Format

```bash
python -m smlx.tools.memory_metrics --csv --output memory.csv
```

Output:
```csv
PID,Name,User,Status,RSS_MB,VMS_MB,Memory_Percent,CPU_Percent,Threads,Command
56100,"Browser Helper (Renderer)",ryanoboyle,running,3948.89,1828736.64,10.71,0.0,28,"/Applications/Arc.app/..."
1934,"Code Helper (Plugin)",ryanoboyle,running,1645.09,1826588.50,4.46,0.0,17,"/Applications/Visual Studio Code.app/..."
```

### Advanced Options

#### Verbose Mode

Show detailed process information including full command lines:

```bash
python -m smlx.tools.memory_metrics --verbose --top 5
```

#### Include Kernel Processes

By default, kernel processes are filtered out. To include them:

```bash
python -m smlx.tools.memory_metrics --include-kernel
```

#### Watch Mode

Continuously monitor and refresh display:

```bash
# Refresh every 2 seconds
python -m smlx.tools.memory_metrics --watch 2 --top 20

# Stop with Ctrl+C
```

#### Output to File

```bash
# JSON to file
python -m smlx.tools.memory_metrics --json --output /path/to/memory.json

# CSV to file
python -m smlx.tools.memory_metrics --csv --output /path/to/memory.csv
```

## Programmatic Usage

### Basic Collection

```python
from smlx.tools.memory_metrics import MemoryMetricsCollector, MemoryMetricsFormatter

# Create collector
collector = MemoryMetricsCollector()

# Collect metrics
metrics = collector.collect_metrics()

# Format and display
formatter = MemoryMetricsFormatter()
print(formatter.format_metrics(metrics, top_n=10))
```

### System Memory Only

```python
from smlx.tools.memory_metrics import MemoryMetricsCollector

collector = MemoryMetricsCollector()
system_info = collector.get_system_memory()

print(f"Total: {system_info.total_mb} MB")
print(f"Used: {system_info.used_mb} MB ({system_info.percent}%)")
print(f"Available: {system_info.available_mb} MB")
```

### Process Memory with Filtering

```python
from smlx.tools.memory_metrics import MemoryMetricsCollector

collector = MemoryMetricsCollector()

# Get only processes using > 100MB
processes = collector.get_process_memory(threshold_mb=100.0)

for proc in processes[:10]:
    print(f"{proc.name}: {proc.rss_mb} MB")
```

### Memory Pressure Detection

```python
from smlx.tools.memory_metrics import MemoryMetricsCollector

collector = MemoryMetricsCollector()
pressure = collector.get_memory_pressure()

if pressure == "critical":
    print("⚠️ Critical memory pressure detected!")
elif pressure == "warn":
    print("⚠️ Memory pressure warning!")
else:
    print("✅ Memory pressure normal")
```

### Export to Different Formats

```python
from smlx.tools.memory_metrics import MemoryMetricsCollector, MemoryMetricsFormatter
import json

collector = MemoryMetricsCollector()
metrics = collector.collect_metrics()
formatter = MemoryMetricsFormatter()

# Export to JSON
json_str = formatter.to_json(metrics)
data = json.loads(json_str)

# Export to CSV
csv_str = formatter.to_csv(metrics)

# Save to files
with open("memory.json", "w") as f:
    f.write(json_str)

with open("memory.csv", "w") as f:
    f.write(csv_str)
```

## Data Structures

### SystemMemoryInfo

```python
@dataclass
class SystemMemoryInfo:
    total_mb: float          # Total physical memory
    available_mb: float      # Available for allocation
    used_mb: float          # Currently in use
    free_mb: float          # Completely free
    percent: float          # Percentage used
    active_mb: float        # Active (recently accessed)
    inactive_mb: float      # Inactive (can be freed)
    wired_mb: float         # Wired (cannot be swapped)
    cached_mb: float        # Cached (Linux only)
    swap_total_mb: float    # Total swap
    swap_used_mb: float     # Used swap
    swap_free_mb: float     # Free swap
    swap_percent: float     # Swap usage percentage
```

### ProcessMemoryInfo

```python
@dataclass
class ProcessMemoryInfo:
    pid: int                # Process ID
    name: str              # Process name
    status: str            # Process status (running, sleeping, etc.)
    rss_mb: float          # Resident Set Size (physical memory)
    vms_mb: float          # Virtual Memory Size
    percent: float         # % of total system memory
    num_threads: int       # Number of threads
    cpu_percent: float     # CPU usage percentage
    username: str          # Process owner
    cmdline: str           # Full command line
```

## Memory Pressure Levels

The tool reports memory pressure as one of:

- **normal** - System has adequate free memory
- **warn** - Memory is running low (>70% used)
- **critical** - System is under severe memory pressure (>85% used)
- **unknown** - Unable to determine (fallback to percentage-based estimate)

On macOS, the tool uses the native `memory_pressure` command for accurate reporting.

## Use Cases

### Development

Monitor memory usage during development and testing:

```bash
# Watch memory while running tests
python -m smlx.tools.memory_metrics --watch 1 --top 20 --threshold 100
```

### Debugging Memory Leaks

Track specific processes over time:

```bash
# Export snapshots at intervals
while true; do
    python -m smlx.tools.memory_metrics --json --output "snapshot_$(date +%s).json"
    sleep 60
done
```

### System Monitoring

Include in monitoring scripts or dashboards:

```python
from smlx.tools.memory_metrics import MemoryMetricsCollector

collector = MemoryMetricsCollector()
metrics = collector.collect_metrics()

# Alert if memory pressure is high
if metrics.memory_pressure in ["warn", "critical"]:
    send_alert(f"Memory pressure: {metrics.memory_pressure}")

# Alert if any process uses > 2GB
for proc in metrics.processes:
    if proc.rss_mb > 2000:
        send_alert(f"High memory: {proc.name} using {proc.rss_mb} MB")
```

### CI/CD Integration

Monitor memory during automated testing:

```bash
# Before tests
python -m smlx.tools.memory_metrics --json --output before.json

# Run tests
pytest

# After tests
python -m smlx.tools.memory_metrics --json --output after.json

# Compare memory usage
python compare_memory.py before.json after.json
```

## Performance

The tool is designed to be lightweight and fast:

- **Collection time**: ~0.1-0.2 seconds for 400+ processes
- **Memory overhead**: ~20-30 MB during execution
- **CPU usage**: Minimal (< 1% on modern systems)

## Platform Support

- **macOS** - Full support (native memory_pressure detection)
- **Linux** - Full support (uses /proc for memory info)
- **Windows** - Full support via psutil

## Troubleshooting

### Permission Errors

Some processes may be inaccessible due to permissions. The tool silently skips these processes. To see all processes including kernel processes:

```bash
python -m smlx.tools.memory_metrics --include-kernel
```

### Memory Pressure Unknown

If memory pressure shows as "unknown", the native `memory_pressure` command is unavailable. The tool falls back to percentage-based estimation.

### High Memory Usage

If the tool itself uses significant memory, try:

```bash
# Limit to top processes only
python -m smlx.tools.memory_metrics --top 20 --threshold 50
```

## Examples

See [examples/tools/memory_metrics_example.py](../../examples/tools/memory_metrics_example.py) for comprehensive usage examples.

## Related Tools

- `psutil` - Cross-platform process and system utilities
- `htop` / `top` - Interactive process viewers
- `vmstat` - Virtual memory statistics
- `memory_pressure` (macOS) - Native memory pressure detection

## License

This tool is part of the SMLX project and follows the same license.
