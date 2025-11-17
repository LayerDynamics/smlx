"""
Example usage of the memory metrics tool.

This demonstrates both CLI usage and programmatic usage of the memory metrics
system to monitor memory consumption.
"""

import json
from smlx.tools.memory_metrics import (
    MemoryMetricsCollector,
    MemoryMetricsFormatter,
)


def example_basic_usage():
    """Basic example: collect and display memory metrics."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Memory Metrics Collection")
    print("=" * 80)

    # Create collector
    collector = MemoryMetricsCollector()

    # Collect metrics
    metrics = collector.collect_metrics()

    # Format and display
    formatter = MemoryMetricsFormatter()
    output = formatter.format_metrics(metrics, top_n=10)
    print(output)


def example_threshold_filtering():
    """Example: filter processes by memory threshold."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Filtering Processes Over 100MB")
    print("=" * 80)

    collector = MemoryMetricsCollector()

    # Only show processes using more than 100MB
    metrics = collector.collect_metrics(threshold_mb=100.0)

    formatter = MemoryMetricsFormatter()
    output = formatter.format_metrics(metrics, verbose=True)
    print(output)


def example_json_export():
    """Example: export metrics as JSON."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: JSON Export")
    print("=" * 80)

    collector = MemoryMetricsCollector()
    metrics = collector.collect_metrics()

    formatter = MemoryMetricsFormatter()
    json_output = formatter.to_json(metrics)

    # Parse and display a subset
    data = json.loads(json_output)
    print(f"Timestamp: {data['timestamp']}")
    print(f"Memory Pressure: {data['memory_pressure']}")
    print(f"Total Processes: {data['total_processes']}")
    print(f"\nSystem Memory:")
    print(f"  Total: {data['system']['total_mb']:.2f} MB")
    print(f"  Used: {data['system']['used_mb']:.2f} MB ({data['system']['percent']:.1f}%)")
    print(f"  Available: {data['system']['available_mb']:.2f} MB")

    print(f"\nTop 5 Memory Consumers:")
    for i, proc in enumerate(data["processes"][:5], 1):
        print(
            f"  {i}. {proc['name']} (PID {proc['pid']}): "
            f"{proc['rss_mb']:.2f} MB ({proc['percent']:.2f}%)"
        )


def example_csv_export():
    """Example: export metrics as CSV."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: CSV Export")
    print("=" * 80)

    collector = MemoryMetricsCollector()
    metrics = collector.collect_metrics(threshold_mb=50.0)

    formatter = MemoryMetricsFormatter()
    csv_output = formatter.to_csv(metrics)

    # Display first 10 lines
    lines = csv_output.split("\n")
    for line in lines[:10]:
        print(line)

    print(f"\n... ({len(lines)} total lines)")


def example_system_info_only():
    """Example: get only system memory information."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: System Memory Info Only")
    print("=" * 80)

    collector = MemoryMetricsCollector()
    system_info = collector.get_system_memory()

    print(f"Total Memory:     {system_info.total_mb:>10.2f} MB")
    print(f"Used Memory:      {system_info.used_mb:>10.2f} MB ({system_info.percent:.1f}%)")
    print(f"Available Memory: {system_info.available_mb:>10.2f} MB")
    print(f"Free Memory:      {system_info.free_mb:>10.2f} MB")
    print(f"\nMemory Breakdown:")
    print(f"  Active:         {system_info.active_mb:>10.2f} MB")
    print(f"  Inactive:       {system_info.inactive_mb:>10.2f} MB")
    print(f"  Wired:          {system_info.wired_mb:>10.2f} MB")
    print(f"\nSwap:")
    print(
        f"  Total:          {system_info.swap_total_mb:>10.2f} MB"
    )
    print(
        f"  Used:           {system_info.swap_used_mb:>10.2f} MB "
        f"({system_info.swap_percent:.1f}%)"
    )

    # Get memory pressure
    pressure = collector.get_memory_pressure()
    print(f"\nMemory Pressure: {pressure.upper()}")


def example_process_analysis():
    """Example: analyze specific process types."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Process Type Analysis")
    print("=" * 80)

    collector = MemoryMetricsCollector()
    processes = collector.get_process_memory()

    # Group processes by type
    process_types = {}
    for proc in processes:
        # Simple categorization based on name
        name = proc.name.lower()
        if "python" in name:
            category = "Python"
        elif "helper" in name or "renderer" in name:
            category = "Helper/Renderer"
        elif "code" in name or "vscode" in name:
            category = "VS Code"
        elif "arc" in name or "browser" in name:
            category = "Browser"
        else:
            category = "Other"

        if category not in process_types:
            process_types[category] = {"count": 0, "total_mb": 0.0}

        process_types[category]["count"] += 1
        process_types[category]["total_mb"] += proc.rss_mb

    # Display results
    print(f"{'Category':<20} {'Count':>10} {'Total Memory':>15}")
    print("=" * 50)
    for category, stats in sorted(
        process_types.items(), key=lambda x: x[1]["total_mb"], reverse=True
    ):
        print(f"{category:<20} {stats['count']:>10} {stats['total_mb']:>12.2f} MB")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("MEMORY METRICS TOOL - USAGE EXAMPLES")
    print("=" * 80)

    print("\n📋 CLI Usage Examples:")
    print("-" * 80)
    print("# Show top 10 memory consumers")
    print("python -m smlx.tools.memory_metrics --top 10")
    print("\n# Show processes using > 100MB")
    print("python -m smlx.tools.memory_metrics --threshold 100")
    print("\n# Export to JSON")
    print("python -m smlx.tools.memory_metrics --json --output memory.json")
    print("\n# Export to CSV")
    print("python -m smlx.tools.memory_metrics --csv --output memory.csv")
    print("\n# Watch mode (refresh every 2 seconds)")
    print("python -m smlx.tools.memory_metrics --watch 2 --top 20")
    print("\n# Verbose output with process details")
    print("python -m smlx.tools.memory_metrics --verbose --top 5")
    print("\n# Sort by CPU usage")
    print("python -m smlx.tools.memory_metrics --sort cpu --top 10")
    print("-" * 80)

    print("\n\n📊 Programmatic Usage Examples:")
    print("-" * 80)

    try:
        example_basic_usage()
        example_threshold_filtering()
        example_json_export()
        example_csv_export()
        example_system_info_only()
        example_process_analysis()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
