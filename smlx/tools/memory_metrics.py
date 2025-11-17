"""
Memory Metrics Tool

Comprehensive tool for monitoring system memory usage, including per-process
memory consumption, system-wide statistics, and memory pressure indicators.

Usage:
    python -m smlx.tools.memory_metrics              # Show all processes
    python -m smlx.tools.memory_metrics --top 10     # Show top 10 consumers
    python -m smlx.tools.memory_metrics --sort rss   # Sort by RSS memory
    python -m smlx.tools.memory_metrics --json       # Output as JSON
    python -m smlx.tools.memory_metrics --threshold 100  # Show only >100MB processes
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    print("Error: psutil is required. Install with: pip install psutil")
    sys.exit(1)


@dataclass
class ProcessMemoryInfo:
    """Memory information for a single process."""

    pid: int
    name: str
    status: str
    rss_mb: float  # Resident Set Size (actual physical memory)
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percentage of total memory
    num_threads: int
    cpu_percent: float
    username: str
    cmdline: str


@dataclass
class SystemMemoryInfo:
    """System-wide memory statistics."""

    total_mb: float
    available_mb: float
    used_mb: float
    free_mb: float
    percent: float
    active_mb: float
    inactive_mb: float
    wired_mb: float
    cached_mb: Optional[float] = None
    swap_total_mb: Optional[float] = None
    swap_used_mb: Optional[float] = None
    swap_free_mb: Optional[float] = None
    swap_percent: Optional[float] = None


@dataclass
class MemoryMetrics:
    """Complete memory metrics snapshot."""

    timestamp: str
    system: SystemMemoryInfo
    processes: list[ProcessMemoryInfo]
    total_processes: int
    memory_pressure: str


class MemoryMetricsCollector:
    """Collects comprehensive memory metrics."""

    def __init__(self):
        """Initialize the memory metrics collector."""
        self.page_size = 4096  # Typical page size in bytes

    def get_system_memory(self) -> SystemMemoryInfo:
        """Get system-wide memory statistics."""
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Convert bytes to megabytes
        def to_mb(bytes_val):
            return round(bytes_val / (1024 * 1024), 2)

        system_info = SystemMemoryInfo(
            total_mb=to_mb(vm.total),
            available_mb=to_mb(vm.available),
            used_mb=to_mb(vm.used),
            free_mb=to_mb(vm.free),
            percent=vm.percent,
            active_mb=to_mb(vm.active) if hasattr(vm, "active") else 0,
            inactive_mb=to_mb(vm.inactive) if hasattr(vm, "inactive") else 0,
            wired_mb=to_mb(vm.wired) if hasattr(vm, "wired") else 0,
            cached_mb=to_mb(vm.cached) if hasattr(vm, "cached") else None,
            swap_total_mb=to_mb(swap.total),
            swap_used_mb=to_mb(swap.used),
            swap_free_mb=to_mb(swap.free),
            swap_percent=swap.percent,
        )

        return system_info

    def get_memory_pressure(self) -> str:
        """
        Get memory pressure level (macOS specific).

        Returns:
            Memory pressure level: 'normal', 'warn', 'critical', or 'unknown'
        """
        try:
            result = subprocess.run(
                ["memory_pressure"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout.lower()

            if "normal" in output:
                return "normal"
            elif "warn" in output:
                return "warn"
            elif "critical" in output:
                return "critical"
            else:
                return "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # Fallback based on memory percentage
            vm = psutil.virtual_memory()
            if vm.percent < 70:
                return "normal"
            elif vm.percent < 85:
                return "warn"
            else:
                return "critical"

    def get_process_memory(
        self, threshold_mb: float = 0.0, include_kernel: bool = False
    ) -> list[ProcessMemoryInfo]:
        """
        Get memory information for all processes.

        Args:
            threshold_mb: Only include processes using more than this amount of memory
            include_kernel: Include kernel processes (typically low-level system processes)

        Returns:
            List of ProcessMemoryInfo objects
        """
        processes = []

        for proc in psutil.process_iter(
            [
                "pid",
                "name",
                "status",
                "memory_info",
                "memory_percent",
                "num_threads",
                "cpu_percent",
                "username",
                "cmdline",
            ]
        ):
            try:
                pinfo = proc.info
                mem_info = pinfo.get("memory_info")

                # Skip processes without memory info (common for system processes)
                if mem_info is None:
                    continue

                # Convert bytes to MB
                rss_mb = round(mem_info.rss / (1024 * 1024), 2)
                vms_mb = round(mem_info.vms / (1024 * 1024), 2)

                # Apply threshold filter
                if rss_mb < threshold_mb:
                    continue

                # Filter kernel processes if requested
                username = pinfo.get("username", "")
                if not include_kernel and username in ["root", "_windowserver"]:
                    continue

                # Get command line (truncate if too long)
                cmdline = pinfo.get("cmdline", [])
                cmdline_str = " ".join(cmdline) if cmdline else pinfo.get("name", "")
                if len(cmdline_str) > 100:
                    cmdline_str = cmdline_str[:97] + "..."

                process_info = ProcessMemoryInfo(
                    pid=pinfo["pid"],
                    name=pinfo.get("name", "Unknown"),
                    status=pinfo.get("status", "unknown"),
                    rss_mb=rss_mb,
                    vms_mb=vms_mb,
                    percent=round(pinfo.get("memory_percent", 0.0), 2),
                    num_threads=pinfo.get("num_threads", 0),
                    cpu_percent=round(pinfo.get("cpu_percent", 0.0), 2),
                    username=username or "unknown",
                    cmdline=cmdline_str,
                )

                processes.append(process_info)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process terminated or access denied - silently skip
                continue
            except Exception:
                # Unexpected error - silently skip this process
                continue

        return processes

    def collect_metrics(
        self, threshold_mb: float = 0.0, include_kernel: bool = False
    ) -> MemoryMetrics:
        """
        Collect complete memory metrics.

        Args:
            threshold_mb: Only include processes using more than this amount of memory
            include_kernel: Include kernel processes

        Returns:
            MemoryMetrics object with complete snapshot
        """
        timestamp = datetime.now().isoformat()
        system = self.get_system_memory()
        processes = self.get_process_memory(threshold_mb, include_kernel)
        memory_pressure = self.get_memory_pressure()

        metrics = MemoryMetrics(
            timestamp=timestamp,
            system=system,
            processes=processes,
            total_processes=len(processes),
            memory_pressure=memory_pressure,
        )

        return metrics


class MemoryMetricsFormatter:
    """Formats memory metrics for display."""

    @staticmethod
    def format_bytes(bytes_val: float) -> str:
        """Format bytes to human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} PB"

    @staticmethod
    def format_system_info(system: SystemMemoryInfo) -> str:
        """Format system memory information."""
        lines = [
            "=" * 80,
            "SYSTEM MEMORY OVERVIEW",
            "=" * 80,
            f"Total Memory:      {system.total_mb:>10.2f} MB",
            f"Used Memory:       {system.used_mb:>10.2f} MB ({system.percent:.1f}%)",
            f"Available Memory:  {system.available_mb:>10.2f} MB",
            f"Free Memory:       {system.free_mb:>10.2f} MB",
            "",
            "Memory Breakdown:",
            f"  Active:          {system.active_mb:>10.2f} MB",
            f"  Inactive:        {system.inactive_mb:>10.2f} MB",
            f"  Wired:           {system.wired_mb:>10.2f} MB",
        ]

        if system.cached_mb is not None:
            lines.append(f"  Cached:          {system.cached_mb:>10.2f} MB")

        lines.extend(
            [
                "",
                "Swap Memory:",
                f"  Total:           {system.swap_total_mb:>10.2f} MB",
                f"  Used:            {system.swap_used_mb:>10.2f} MB ({system.swap_percent:.1f}%)",
                f"  Free:            {system.swap_free_mb:>10.2f} MB",
                "=" * 80,
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def format_process_table(
        processes: list[ProcessMemoryInfo], top_n: Optional[int] = None
    ) -> str:
        """Format process memory information as a table."""
        if not processes:
            return "No processes found matching criteria."

        # Sort by RSS memory (descending)
        sorted_procs = sorted(processes, key=lambda p: p.rss_mb, reverse=True)

        if top_n:
            sorted_procs = sorted_procs[:top_n]

        # Table header
        header = (
            f"{'PID':<8} {'USER':<12} {'RSS (MB)':>10} {'VMS (MB)':>10} "
            f"{'MEM%':>6} {'CPU%':>6} {'THREADS':>8} {'NAME':<25}"
        )
        separator = "=" * len(header)

        lines = [separator, header, separator]

        # Table rows
        for proc in sorted_procs:
            line = (
                f"{proc.pid:<8} {proc.username[:12]:<12} {proc.rss_mb:>10.2f} "
                f"{proc.vms_mb:>10.2f} {proc.percent:>6.2f} {proc.cpu_percent:>6.2f} "
                f"{proc.num_threads:>8} {proc.name[:25]:<25}"
            )
            lines.append(line)

        lines.append(separator)
        lines.append(f"Total processes shown: {len(sorted_procs)}")

        return "\n".join(lines)

    @staticmethod
    def format_metrics(
        metrics: MemoryMetrics, top_n: Optional[int] = None, verbose: bool = False
    ) -> str:
        """Format complete memory metrics."""
        lines = [
            f"Memory Metrics Snapshot - {metrics.timestamp}",
            f"Memory Pressure: {metrics.memory_pressure.upper()}",
            "",
            MemoryMetricsFormatter.format_system_info(metrics.system),
            "",
            f"PROCESS MEMORY USAGE (Top {top_n if top_n else 'All'})",
            MemoryMetricsFormatter.format_process_table(metrics.processes, top_n),
        ]

        if verbose:
            lines.extend(
                [
                    "",
                    "=" * 80,
                    "DETAILED PROCESS INFORMATION",
                    "=" * 80,
                ]
            )

            sorted_procs = sorted(metrics.processes, key=lambda p: p.rss_mb, reverse=True)
            if top_n:
                sorted_procs = sorted_procs[:top_n]

            for proc in sorted_procs:
                lines.extend(
                    [
                        f"\nPID: {proc.pid} | {proc.name}",
                        f"  User:        {proc.username}",
                        f"  Status:      {proc.status}",
                        f"  RSS Memory:  {proc.rss_mb:.2f} MB",
                        f"  VMS Memory:  {proc.vms_mb:.2f} MB",
                        f"  Memory %:    {proc.percent:.2f}%",
                        f"  CPU %:       {proc.cpu_percent:.2f}%",
                        f"  Threads:     {proc.num_threads}",
                        f"  Command:     {proc.cmdline}",
                        "-" * 80,
                    ]
                )

        return "\n".join(lines)

    @staticmethod
    def to_json(metrics: MemoryMetrics) -> str:
        """Convert metrics to JSON string."""
        data = {
            "timestamp": metrics.timestamp,
            "memory_pressure": metrics.memory_pressure,
            "total_processes": metrics.total_processes,
            "system": asdict(metrics.system),
            "processes": [asdict(p) for p in metrics.processes],
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def to_csv(metrics: MemoryMetrics) -> str:
        """Convert process metrics to CSV format."""
        lines = [
            "PID,Name,User,Status,RSS_MB,VMS_MB,Memory_Percent,CPU_Percent,Threads,Command"
        ]

        for proc in sorted(metrics.processes, key=lambda p: p.rss_mb, reverse=True):
            # Escape commas in command
            cmdline = proc.cmdline.replace(",", ";")
            lines.append(
                f'{proc.pid},"{proc.name}",{proc.username},{proc.status},'
                f"{proc.rss_mb},{proc.vms_mb},{proc.percent},{proc.cpu_percent},"
                f'{proc.num_threads},"{cmdline}"'
            )

        return "\n".join(lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor system memory usage and per-process consumption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Show all processes
  %(prog)s --top 20                 # Show top 20 memory consumers
  %(prog)s --threshold 100          # Show only processes using >100MB
  %(prog)s --sort rss --desc        # Sort by RSS memory (descending)
  %(prog)s --json                   # Output as JSON
  %(prog)s --csv --output mem.csv   # Export to CSV file
  %(prog)s --verbose                # Show detailed process information
  %(prog)s --include-kernel         # Include kernel processes
        """,
    )

    parser.add_argument(
        "--top", "-t", type=int, metavar="N", help="Show only top N processes by memory usage"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        metavar="MB",
        help="Only show processes using more than MB megabytes (default: 0)",
    )

    parser.add_argument(
        "--sort",
        choices=["rss", "vms", "percent", "cpu", "pid", "name"],
        default="rss",
        help="Sort processes by field (default: rss)",
    )

    parser.add_argument(
        "--desc", action="store_true", help="Sort in descending order (default: ascending)"
    )

    parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    parser.add_argument(
        "--csv", action="store_true", help="Output in CSV format"
    )

    parser.add_argument(
        "--output", "-o", type=Path, metavar="FILE", help="Write output to file"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed process information"
    )

    parser.add_argument(
        "--include-kernel", action="store_true", help="Include kernel processes"
    )

    parser.add_argument(
        "--watch",
        "-w",
        type=int,
        metavar="SECONDS",
        help="Continuously monitor and refresh every SECONDS",
    )

    args = parser.parse_args()

    # Collect metrics
    collector = MemoryMetricsCollector()

    def collect_and_display():
        """Collect metrics and display/save based on arguments."""
        metrics = collector.collect_metrics(
            threshold_mb=args.threshold, include_kernel=args.include_kernel
        )

        # Sort processes if requested
        if args.sort != "rss" or args.desc:
            reverse = not args.desc  # Default is descending for memory, so invert
            sort_key_map = {
                "rss": lambda p: p.rss_mb,
                "vms": lambda p: p.vms_mb,
                "percent": lambda p: p.percent,
                "cpu": lambda p: p.cpu_percent,
                "pid": lambda p: p.pid,
                "name": lambda p: p.name,
            }
            metrics.processes.sort(key=sort_key_map[args.sort], reverse=reverse)

        # Format output
        formatter = MemoryMetricsFormatter()

        if args.json:
            output = formatter.to_json(metrics)
        elif args.csv:
            output = formatter.to_csv(metrics)
        else:
            output = formatter.format_metrics(metrics, top_n=args.top, verbose=args.verbose)

        # Write to file or stdout
        if args.output:
            args.output.write_text(output)
            print(f"Output written to {args.output}")
        else:
            print(output)

    # Handle watch mode
    if args.watch:
        import time

        try:
            while True:
                # Clear screen (platform-independent)
                print("\033[2J\033[H", end="")
                collect_and_display()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        collect_and_display()


if __name__ == "__main__":
    main()
