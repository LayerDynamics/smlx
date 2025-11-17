"""
Benchmark suite for individual MLX operations.

Provides functions for benchmarking core operations like matmul,
attention, layer normalization, etc.
"""

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from ..runners import OperationBenchmarkRunner
from ..stats import BenchmarkSuite, OperationBenchmarkStats


def benchmark_operation(
    name: str,
    fn: Callable,
    *args,
    input_shapes: Optional[list[tuple]] = None,
    dtype: str = "float32",
    num_warmup: int = 5,
    num_iterations: int = 100,
    **kwargs,
) -> OperationBenchmarkStats:
    """
    Benchmark an arbitrary operation.

    Args:
        name: Operation name
        fn: Function to benchmark
        *args: Positional arguments for fn
        input_shapes: List of input tensor shapes
        dtype: Data type
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        **kwargs: Keyword arguments for fn

    Returns:
        OperationBenchmarkStats

    Example:
        >>> def custom_op(x, y):
        ...     return mx.exp(x) + mx.log(y)
        >>> stats = benchmark_operation(
        ...     "custom",
        ...     custom_op,
        ...     mx.random.normal((1000, 1000)),
        ...     mx.random.normal((1000, 1000)),
        ... )
    """
    # Infer input shapes if not provided
    if input_shapes is None:
        input_shapes = []
        for arg in args:
            if hasattr(arg, "shape"):
                input_shapes.append(tuple(arg.shape))

    runner = OperationBenchmarkRunner(
        name=name,
        operation=name,
        fn=fn,
        args=args,
        kwargs=kwargs,
        input_shapes=input_shapes,
        dtype=dtype,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )

    return runner.run()


def benchmark_matmul(
    shape_a: tuple[int, int],
    shape_b: tuple[int, int],
    dtype: mx.Dtype = mx.float32,
    num_iterations: int = 100,
) -> OperationBenchmarkStats:
    """
    Benchmark matrix multiplication.

    Args:
        shape_a: Shape of first matrix (M, K)
        shape_b: Shape of second matrix (K, N)
        dtype: Data type
        num_iterations: Number of iterations

    Returns:
        OperationBenchmarkStats

    Example:
        >>> stats = benchmark_matmul((1000, 1000), (1000, 1000))
        >>> print(f"Matmul: {stats.duration_ms:.2f}ms")
    """
    a = mx.random.normal(shape_a).astype(dtype)
    b = mx.random.normal(shape_b).astype(dtype)

    def matmul_fn(x, y):
        return mx.matmul(x, y)

    return benchmark_operation(
        f"matmul_{shape_a}x{shape_b}",
        matmul_fn,
        a,
        b,
        input_shapes=[shape_a, shape_b],
        dtype=str(dtype),
        num_iterations=num_iterations,
    )


def benchmark_attention(
    batch_size: int = 1,
    seq_len: int = 128,
    num_heads: int = 8,
    head_dim: int = 64,
    dtype: mx.Dtype = mx.float16,
    num_iterations: int = 50,
) -> OperationBenchmarkStats:
    """
    Benchmark scaled dot-product attention.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Data type
        num_iterations: Number of iterations

    Returns:
        OperationBenchmarkStats

    Example:
        >>> stats = benchmark_attention(batch_size=2, seq_len=256)
        >>> print(f"Attention: {stats.duration_ms:.2f}ms")
    """
    # Create Q, K, V tensors
    shape = (batch_size, seq_len, num_heads, head_dim)
    q = mx.random.normal(shape).astype(dtype)
    k = mx.random.normal(shape).astype(dtype)
    v = mx.random.normal(shape).astype(dtype)

    def attention_fn(q, k, v):
        # Scaled dot-product attention
        # Shape: (B, L, H, D) -> (B, H, L, D)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Attention scores
        scale = 1.0 / (head_dim**0.5)
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale

        # Softmax
        attn = mx.softmax(scores, axis=-1)

        # Apply attention to values
        out = mx.matmul(attn, v)

        # Transpose back
        out = mx.transpose(out, (0, 2, 1, 3))
        return out

    return benchmark_operation(
        f"attention_b{batch_size}_l{seq_len}_h{num_heads}",
        attention_fn,
        q,
        k,
        v,
        input_shapes=[shape, shape, shape],
        dtype=str(dtype),
        num_iterations=num_iterations,
    )


def benchmark_layernorm(
    shape: tuple[int, ...],
    dtype: mx.Dtype = mx.float32,
    num_iterations: int = 100,
) -> OperationBenchmarkStats:
    """
    Benchmark layer normalization.

    Args:
        shape: Input tensor shape
        dtype: Data type
        num_iterations: Number of iterations

    Returns:
        OperationBenchmarkStats

    Example:
        >>> stats = benchmark_layernorm((32, 512, 768))
        >>> print(f"LayerNorm: {stats.duration_ms:.2f}ms")
    """
    x = mx.random.normal(shape).astype(dtype)

    # Create layer norm
    norm = nn.LayerNorm(shape[-1])

    def layernorm_fn(x):
        return norm(x)

    return benchmark_operation(
        f"layernorm_{shape}",
        layernorm_fn,
        x,
        input_shapes=[shape],
        dtype=str(dtype),
        num_iterations=num_iterations,
    )


def benchmark_gelu(
    shape: tuple[int, ...],
    dtype: mx.Dtype = mx.float32,
    num_iterations: int = 100,
) -> OperationBenchmarkStats:
    """
    Benchmark GELU activation.

    Args:
        shape: Input tensor shape
        dtype: Data type
        num_iterations: Number of iterations

    Returns:
        OperationBenchmarkStats
    """
    x = mx.random.normal(shape).astype(dtype)

    def gelu_fn(x):
        return nn.gelu(x)

    return benchmark_operation(
        f"gelu_{shape}",
        gelu_fn,
        x,
        input_shapes=[shape],
        dtype=str(dtype),
        num_iterations=num_iterations,
    )


def run_ops_suite(
    operation: Optional[str] = "all",
    shape: Optional[str] = None,
    num_iterations: int = 100,
    **kwargs,
) -> BenchmarkSuite:
    """
    Run operation benchmark suite.

    Args:
        operation: Operation to benchmark ("matmul", "attention", "layernorm", "all")
        shape: Shape string for matmul (e.g., "1000,1000")
        num_iterations: Number of iterations
        **kwargs: Additional arguments

    Returns:
        BenchmarkSuite with all operation benchmark results

    Example:
        >>> suite = run_ops_suite(operation="matmul", shape="1000,1000")
        >>> suite = run_ops_suite(operation="all")
    """
    suite = BenchmarkSuite(name="MLX Operations")

    if operation in ["matmul", "all"]:
        print("\nBenchmarking matmul...")
        if shape:
            parts = shape.split(",")
            if len(parts) == 2:
                m, n = int(parts[0]), int(parts[1])
                shape_a = (m, n)
                shape_b = (n, n)
            else:
                # Default shape
                shape_a = (1000, 1000)
                shape_b = (1000, 1000)
        else:
            shape_a = (1000, 1000)
            shape_b = (1000, 1000)

        stats = benchmark_matmul(shape_a, shape_b, num_iterations=num_iterations)
        suite.add(stats)

    if operation in ["attention", "all"]:
        print("\nBenchmarking attention...")
        stats = benchmark_attention(num_iterations=num_iterations)
        suite.add(stats)

    if operation in ["layernorm", "all"]:
        print("\nBenchmarking layernorm...")
        stats = benchmark_layernorm((32, 512, 768), num_iterations=num_iterations)
        suite.add(stats)

    return suite
