"""
Reporting utilities for benchmark results.

Provides functions for formatting and displaying benchmark results
in various formats (console, CSV, JSON, markdown).
"""

from pathlib import Path
from typing import Union

from smlx.utils.formatting import (
    format_dict_table,
    format_header,
)
from smlx.utils.io import save_csv, save_json, save_jsonl

from .stats import (
    BenchmarkStats,
    BenchmarkSuite,
    ComparisonStats,
    ModelBenchmarkStats,
    OperationBenchmarkStats,
)


def print_benchmark_stats(stats: BenchmarkStats):
    """
    Print benchmark statistics to console.

    Args:
        stats: Benchmark statistics to print

    Example:
        >>> stats = quick_benchmark(lambda x: x @ x.T, mx.random.normal((1000, 1000)))
        >>> print_benchmark_stats(stats)
    """
    print(format_header(stats.name))
    print(f"Duration: {stats.duration_ms:.2f}ms")
    print(f"Iterations: {stats.iterations}")
    print(f"Per iteration: {stats.duration_per_iter_ms:.2f}ms")
    print(f"Peak memory: {stats.peak_memory_gb:.2f}GB")

    if isinstance(stats, ModelBenchmarkStats):
        print_model_stats(stats)
    elif isinstance(stats, OperationBenchmarkStats):
        print_operation_stats(stats)

    if stats.metadata:
        print("\nMetadata:")
        for key, value in stats.metadata.items():
            print(f"  {key}: {value}")


def print_model_stats(stats: ModelBenchmarkStats):
    """Print model-specific statistics."""
    print(f"\nModel: {stats.model_name}")
    if stats.quantization:
        print(f"Quantization: {stats.quantization}")
    print(f"Batch size: {stats.batch_size}")

    print("\nTokens:")
    print(f"  Prompt: {stats.prompt_tokens}")
    print(f"  Generation: {stats.generation_tokens}")
    print(f"  Total: {stats.total_tokens}")

    print("\nTiming:")
    print(f"  Prompt time: {stats.prompt_time:.3f}s")
    print(f"  Generation time: {stats.generation_time:.3f}s")
    print(f"  Total time: {stats.total_time:.3f}s")
    print(f"  Time to first token: {stats.time_to_first_token:.3f}s")

    print("\nThroughput:")
    print(f"  Prompt: {stats.prompt_tps:.2f} tok/s")
    print(f"  Generation: {stats.generation_tps:.2f} tok/s")
    print(f"  Overall: {stats.overall_tps:.2f} tok/s")


def print_operation_stats(stats: OperationBenchmarkStats):
    """Print operation-specific statistics."""
    print(f"\nOperation: {stats.operation}")
    print(f"Device: {stats.device}")
    print(f"Dtype: {stats.dtype}")

    if stats.input_shapes:
        print("\nInput shapes:")
        for i, shape in enumerate(stats.input_shapes):
            print(f"  [{i}]: {shape}")

    if stats.output_shape:
        print(f"Output shape: {stats.output_shape}")

    if stats.gflops > 0:
        print("\nPerformance:")
        print(f"  GFLOPS: {stats.gflops:.2f}")


def print_comparison(comp: ComparisonStats):
    """
    Print comparison statistics.

    Args:
        comp: Comparison statistics

    Example:
        >>> comp = ComparisonStats(baseline_stats, optimized_stats)
        >>> print_comparison(comp)
    """
    print(format_header("Comparison"))

    print(f"\nBaseline: {comp.baseline.name}")
    print(f"  Duration: {comp.baseline.duration_ms:.2f}ms")
    print(f"  Memory: {comp.baseline.peak_memory_gb:.2f}GB")

    print(f"\nComparison: {comp.comparison.name}")
    print(f"  Duration: {comp.comparison.duration_ms:.2f}ms")
    print(f"  Memory: {comp.comparison.peak_memory_gb:.2f}GB")

    print("\nResults:")
    print(f"  Speedup: {comp.speedup:.2f}x")
    print(f"  Memory reduction: {comp.memory_reduction:.2f}GB ({comp.memory_reduction_percent:.1f}%)")


def create_benchmark_table(benchmarks: list[BenchmarkStats]) -> str:
    """
    Create a table from benchmark results.

    Args:
        benchmarks: list of benchmark statistics

    Returns:
        Formatted table string

    Example:
        >>> benchmarks = [stats1, stats2, stats3]
        >>> table = create_benchmark_table(benchmarks)
        >>> print(table)
    """
    if not benchmarks:
        return ""

    # Convert to list of dicts for table formatting
    data = []
    for bench in benchmarks:
        row = {
            "Name": bench.name,
            "Duration (ms)": f"{bench.duration_ms:.2f}",
            "Per Iter (ms)": f"{bench.duration_per_iter_ms:.2f}",
            "Memory (GB)": f"{bench.peak_memory_gb:.2f}",
        }

        # Add model-specific columns
        if isinstance(bench, ModelBenchmarkStats):
            row["Prompt TPS"] = f"{bench.prompt_tps:.2f}"
            row["Gen TPS"] = f"{bench.generation_tps:.2f}"

        # Add operation-specific columns
        elif isinstance(bench, OperationBenchmarkStats):
            row["Operation"] = bench.operation
            if bench.gflops > 0:
                row["GFLOPS"] = f"{bench.gflops:.2f}"

        data.append(row)

    return format_dict_table(data)


def save_benchmark_results(
    benchmarks: Union[BenchmarkStats, list[BenchmarkStats], BenchmarkSuite],
    filepath: Union[str, Path],
    format: str = "auto",
):
    """
    Save benchmark results to file.

    Args:
        benchmarks: Benchmark statistics or suite
        filepath: Output file path
        format: Output format ('json', 'csv', 'jsonl', 'auto')
                'auto' detects from file extension

    Example:
        >>> save_benchmark_results(stats, "results.json")
        >>> save_benchmark_results([stats1, stats2], "results.csv")
        >>> save_benchmark_results(suite, "results/suite.json")
    """
    filepath = Path(filepath)

    # Detect format from extension
    if format == "auto":
        ext = filepath.suffix.lower()
        if ext == ".json":
            format = "json"
        elif ext == ".csv":
            format = "csv"
        elif ext == ".jsonl":
            format = "jsonl"
        else:
            format = "json"  # Default

    # Convert to list if single stats
    if isinstance(benchmarks, BenchmarkStats):
        benchmarks = [benchmarks]

    # Convert suite to list
    if isinstance(benchmarks, BenchmarkSuite):
        data = benchmarks.to_dict()
        if format == "json":
            save_json(data, filepath)
        return

    # Save in requested format
    if format == "json":
        data = [b.to_dict() for b in benchmarks]
        save_json(data, filepath)
    elif format == "csv":
        data = [b.to_dict() for b in benchmarks]
        save_csv(data, filepath)
    elif format == "jsonl":
        data = [b.to_dict() for b in benchmarks]
        save_jsonl(data, filepath)


def generate_markdown_report(
    benchmarks: Union[list[BenchmarkStats], BenchmarkSuite],
    title: str = "Benchmark Results",
    include_system_info: bool = True,
) -> str:
    """
    Generate a markdown report from benchmark results.

    Args:
        benchmarks: list of benchmarks or suite
        title: Report title
        include_system_info: Whether to include system information

    Returns:
        Markdown-formatted report

    Example:
        >>> report = generate_markdown_report([stats1, stats2], "My Benchmarks")
        >>> print(report)
        >>> # Or save to file:
        >>> Path("report.md").write_text(report)
    """
    lines = []

    # Title
    lines.append(f"# {title}\n")

    # System info
    if include_system_info:
        from .suites.system import get_cpu_info, get_memory_info, get_system_info

        sys_info = get_system_info()
        cpu_info = get_cpu_info()
        mem_info = get_memory_info()

        lines.append("## System Information\n")
        lines.append(f"- **Platform**: {sys_info['platform']}")
        if "chip" in sys_info:
            lines.append(f"- **Chip**: {sys_info['chip']}")
        if "physical_cores" in cpu_info:
            lines.append(
                f"- **CPU Cores**: {cpu_info['physical_cores']} physical, {cpu_info.get('logical_cores', '?')} logical"
            )
        lines.append(f"- **MLX Version**: {sys_info['mlx_version']}")
        if sys_info['mlx_available']:
            lines.append(f"- **Max Memory**: {mem_info['max_recommended_gb']:.2f} GB")
        lines.append("")

    # Extract benchmarks list
    if isinstance(benchmarks, BenchmarkSuite):
        lines.append(f"## Suite: {benchmarks.name}\n")
        bench_list = benchmarks.benchmarks
    else:
        bench_list = benchmarks

    # Summary table
    lines.append("## Results\n")

    # Create markdown table
    if bench_list:
        # Table header
        lines.append("| Name | Duration (ms) | Memory (GB) | Details |")
        lines.append("|------|---------------|-------------|---------|")

        # Table rows
        for bench in bench_list:
            details = ""
            if isinstance(bench, ModelBenchmarkStats):
                details = f"Gen: {bench.generation_tps:.1f} tok/s"
            elif isinstance(bench, OperationBenchmarkStats):
                details = bench.operation

            lines.append(
                f"| {bench.name} | {bench.duration_ms:.2f} | {bench.peak_memory_gb:.2f} | {details} |"
            )

        lines.append("")

    # Detailed results
    lines.append("## Detailed Results\n")
    for i, bench in enumerate(bench_list, 1):
        lines.append(f"### {i}. {bench.name}\n")
        lines.append(f"- **Duration**: {bench.duration_ms:.2f}ms")
        lines.append(f"- **Per Iteration**: {bench.duration_per_iter_ms:.2f}ms")
        lines.append(f"- **Peak Memory**: {bench.peak_memory_gb:.2f}GB")

        if isinstance(bench, ModelBenchmarkStats):
            lines.append(f"- **Model**: {bench.model_name}")
            lines.append(f"- **Prompt TPS**: {bench.prompt_tps:.2f} tok/s")
            lines.append(f"- **Generation TPS**: {bench.generation_tps:.2f} tok/s")
            lines.append(f"- **Total Tokens**: {bench.total_tokens}")

        elif isinstance(bench, OperationBenchmarkStats):
            lines.append(f"- **Operation**: {bench.operation}")
            lines.append(f"- **Device**: {bench.device}")
            if bench.gflops > 0:
                lines.append(f"- **GFLOPS**: {bench.gflops:.2f}")

        lines.append("")

    return "\n".join(lines)
