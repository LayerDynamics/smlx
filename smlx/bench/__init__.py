"""
SMLX Benchmarking Module

Performance benchmarking for MLX models and operations, optimized for
Apple Silicon M4 chipsets.

Main components:
- stats: Dataclasses for tracking benchmark statistics
- system: System information and capability detection
- runners: Benchmark execution framework
- suites: Model-specific and operation-level benchmark suites
- report: Result formatting and reporting
- cli: Command-line interface

Example usage:
    >>> from smlx.bench import quick_benchmark, print_benchmark_stats
    >>> import mlx.core as mx
    >>>
    >>> # Quick benchmark
    >>> stats = quick_benchmark(
    ...     lambda x: x @ x.T,
    ...     mx.random.normal((1000, 1000)),
    ...     name="matmul_1000x1000"
    ... )
    >>> print_benchmark_stats(stats)
    >>>
    >>> # Operation benchmarks
    >>> from smlx.bench.suites import benchmark_matmul
    >>> stats = benchmark_matmul((1000, 1000), (1000, 1000))
    >>>
    >>> # Model benchmarks (when models are implemented)
    >>> from smlx.bench.suites import benchmark_llm
    >>> # stats = benchmark_llm(model, tokenizer)
    >>>
    >>> # System information
    >>> from smlx.bench import print_system_info
    >>> print_system_info()
"""

# Stats
from .stats import (
    BenchmarkStats,
    ModelBenchmarkStats,
    OperationBenchmarkStats,
    ComparisonStats,
    BenchmarkSuite,
    create_model_stats,
)

# System
from .suites.system import (
    get_system_info,
    get_chip_name,
    is_m4_chip,
    is_apple_silicon,
    get_memory_info,
    get_cpu_info,
    print_system_info,
)

# Runners
from .runners import (
    BenchmarkRunner,
    FunctionBenchmarkRunner,
    OperationBenchmarkRunner,
    quick_benchmark,
    compare_implementations,
)

# Suites
from .suites import (
    # LLM
    benchmark_llm,
    LLMBenchmarkConfig,
    # Operations
    benchmark_operation,
    benchmark_matmul,
    benchmark_attention,
)

# Report
from .report import (
    print_benchmark_stats,
    print_model_stats,
    print_operation_stats,
    print_comparison,
    create_benchmark_table,
    save_benchmark_results,
    generate_markdown_report,
)

__all__ = [
    # Stats
    "BenchmarkStats",
    "ModelBenchmarkStats",
    "OperationBenchmarkStats",
    "ComparisonStats",
    "BenchmarkSuite",
    "create_model_stats",
    # System
    "get_system_info",
    "get_chip_name",
    "is_m4_chip",
    "is_apple_silicon",
    "get_memory_info",
    "get_cpu_info",
    "print_system_info",
    # Runners
    "BenchmarkRunner",
    "FunctionBenchmarkRunner",
    "OperationBenchmarkRunner",
    "quick_benchmark",
    "compare_implementations",
    # Suites
    "benchmark_llm",
    "LLMBenchmarkConfig",
    "benchmark_operation",
    "benchmark_matmul",
    "benchmark_attention",
    # Report
    "print_benchmark_stats",
    "print_model_stats",
    "print_operation_stats",
    "print_comparison",
    "create_benchmark_table",
    "save_benchmark_results",
    "generate_markdown_report",
]
