"""
Benchmark suites for different model types and operations.

This module contains specialized benchmark suites for:
- Language models (LLMs)
- Vision-language models (VLMs)
- Text generation (context scaling, generation length, etc.)
- Individual operations (matmul, attention, etc.)
"""

from .llm import benchmark_llm, LLMBenchmarkConfig
from .ops import benchmark_operation, benchmark_matmul, benchmark_attention, run_ops_suite
from .text_generation import (
    benchmark_single_generation,
    benchmark_context_scaling,
    benchmark_generation_length,
    benchmark_temperature_effects,
    run_comprehensive_suite,
    TextGenerationBenchmarkResult,
)

__all__ = [
    # LLM benchmarks
    "benchmark_llm",
    "LLMBenchmarkConfig",
    # Text generation benchmarks
    "benchmark_single_generation",
    "benchmark_context_scaling",
    "benchmark_generation_length",
    "benchmark_temperature_effects",
    "run_comprehensive_suite",
    "TextGenerationBenchmarkResult",
    # Operation benchmarks
    "benchmark_operation",
    "benchmark_matmul",
    "benchmark_attention",
    "run_ops_suite",
]
