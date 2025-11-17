"""
Benchmark statistics dataclasses.

Provides standard dataclasses for tracking performance metrics,
following patterns from MLX-LM and MLX-VLM.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class BenchmarkStats:
    """
    Core benchmark statistics for any operation.

    This is the base class for all benchmark statistics.
    """

    name: str
    """Name of the benchmark"""

    duration_ms: float = 0.0
    """Total duration in milliseconds"""

    iterations: int = 1
    """Number of iterations"""

    peak_memory_gb: float = 0.0
    """Peak memory usage in gigabytes"""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    """ISO timestamp of benchmark"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    @property
    def duration_per_iter_ms(self) -> float:
        """Average duration per iteration in milliseconds."""
        if self.iterations == 0:
            return 0.0
        return self.duration_ms / self.iterations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelBenchmarkStats(BenchmarkStats):
    """
    Benchmark statistics for model inference.

    Similar to MLX-LM's BatchStats and MLX-VLM's GenerationResult.
    """

    # Token counts
    prompt_tokens: int = 0
    """Number of prompt tokens"""

    generation_tokens: int = 0
    """Number of generated tokens"""

    # Timing
    prompt_time: float = 0.0
    """Time spent processing prompt (seconds)"""

    generation_time: float = 0.0
    """Time spent generating tokens (seconds)"""

    # Throughput
    prompt_tps: float = 0.0
    """Prompt tokens per second"""

    generation_tps: float = 0.0
    """Generation tokens per second"""

    # Memory (inherited from BenchmarkStats)

    # Model info
    model_name: str = ""
    """Model name"""

    quantization: Optional[str] = None
    """Quantization method (e.g., '4bit', '8bit', 'none')"""

    batch_size: int = 1
    """Batch size used"""

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + generation)."""
        return self.prompt_tokens + self.generation_tokens

    @property
    def total_time(self) -> float:
        """Total time (prompt + generation) in seconds."""
        return self.prompt_time + self.generation_time

    @property
    def time_to_first_token(self) -> float:
        """Time to first token in seconds (same as prompt_time)."""
        return self.prompt_time

    @property
    def overall_tps(self) -> float:
        """Overall tokens per second (total tokens / total time)."""
        if self.total_time == 0:
            return 0.0
        return self.total_tokens / self.total_time


@dataclass
class OperationBenchmarkStats(BenchmarkStats):
    """
    Benchmark statistics for individual operations (e.g., matmul, attention).

    Similar to MLX core benchmarks.
    """

    operation: str = ""
    """Operation name (e.g., 'matmul', 'attention', 'layernorm')"""

    input_shapes: list[tuple] = field(default_factory=list)
    """Input tensor shapes"""

    output_shape: Optional[tuple] = None
    """Output tensor shape"""

    dtype: str = "float32"
    """Data type used"""

    device: str = "gpu"
    """Device used ('cpu' or 'gpu')"""

    flops: Optional[float] = None
    """Floating point operations (if applicable)"""

    @property
    def gflops(self) -> float:
        """GFLOPS (billions of FLOPs per second)."""
        if self.flops is None or self.duration_ms == 0:
            return 0.0
        return (self.flops / 1e9) / (self.duration_ms / 1000)

    @property
    def throughput(self) -> float:
        """Operations per second."""
        if self.duration_ms == 0:
            return 0.0
        return self.iterations / (self.duration_ms / 1000)


@dataclass
class ComparisonStats:
    """
    Statistics comparing two benchmark results.

    Useful for comparing before/after quantization or different models.
    """

    baseline: BenchmarkStats
    """Baseline benchmark"""

    comparison: BenchmarkStats
    """Comparison benchmark"""

    @property
    def speedup(self) -> float:
        """Speedup factor (baseline_time / comparison_time)."""
        if self.comparison.duration_ms == 0:
            return 0.0
        return self.baseline.duration_ms / self.comparison.duration_ms

    @property
    def memory_reduction(self) -> float:
        """Memory reduction in GB (baseline - comparison)."""
        return self.baseline.peak_memory_gb - self.comparison.peak_memory_gb

    @property
    def memory_reduction_percent(self) -> float:
        """Memory reduction as percentage."""
        if self.baseline.peak_memory_gb == 0:
            return 0.0
        return (self.memory_reduction / self.baseline.peak_memory_gb) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline": self.baseline.to_dict(),
            "comparison": self.comparison.to_dict(),
            "speedup": self.speedup,
            "memory_reduction_gb": self.memory_reduction,
            "memory_reduction_percent": self.memory_reduction_percent,
        }


@dataclass
class BenchmarkSuite:
    """
    Collection of benchmark results.

    Useful for running multiple benchmarks and aggregating results.
    """

    name: str
    """Suite name"""

    benchmarks: list[BenchmarkStats] = field(default_factory=list)
    """list of benchmark results"""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    """ISO timestamp of suite"""

    def add(self, benchmark: BenchmarkStats):
        """Add a benchmark result."""
        self.benchmarks.append(benchmark)

    @property
    def total_duration_ms(self) -> float:
        """Total duration of all benchmarks."""
        return sum(b.duration_ms for b in self.benchmarks)

    @property
    def mean_duration_ms(self) -> float:
        """Mean duration of benchmarks."""
        if not self.benchmarks:
            return 0.0
        return self.total_duration_ms / len(self.benchmarks)

    @property
    def peak_memory_gb(self) -> float:
        """Peak memory usage across all benchmarks."""
        if not self.benchmarks:
            return 0.0
        return max(b.peak_memory_gb for b in self.benchmarks)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "summary": {
                "total_duration_ms": self.total_duration_ms,
                "mean_duration_ms": self.mean_duration_ms,
                "peak_memory_gb": self.peak_memory_gb,
                "count": len(self.benchmarks),
            },
        }


def create_model_stats(
    model_name: str,
    prompt_tokens: int,
    prompt_time: float,
    generation_tokens: int,
    generation_time: float,
    peak_memory_gb: float,
    quantization: Optional[str] = None,
    batch_size: int = 1,
    **kwargs,
) -> ModelBenchmarkStats:
    """
    Helper function to create ModelBenchmarkStats with computed metrics.

    Args:
        model_name: Name of the model
        prompt_tokens: Number of prompt tokens
        prompt_time: Time spent on prompt (seconds)
        generation_tokens: Number of generated tokens
        generation_time: Time spent on generation (seconds)
        peak_memory_gb: Peak memory usage (GB)
        quantization: Quantization method
        batch_size: Batch size
        **kwargs: Additional metadata

    Returns:
        ModelBenchmarkStats with computed TPS metrics

    Example:
        >>> stats = create_model_stats(
        ...     "SmolLM2-135M",
        ...     prompt_tokens=100,
        ...     prompt_time=0.5,
        ...     generation_tokens=50,
        ...     generation_time=2.0,
        ...     peak_memory_gb=2.3,
        ... )
    """
    # Calculate throughput
    prompt_tps = prompt_tokens / prompt_time if prompt_time > 0 else 0.0
    generation_tps = generation_tokens / generation_time if generation_time > 0 else 0.0

    return ModelBenchmarkStats(
        name=f"{model_name}_benchmark",
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        prompt_time=prompt_time,
        generation_tokens=generation_tokens,
        generation_time=generation_time,
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        peak_memory_gb=peak_memory_gb,
        quantization=quantization,
        batch_size=batch_size,
        duration_ms=(prompt_time + generation_time) * 1000,
        metadata=kwargs,
    )
