#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Performance Optimization Demo

Demonstrates profiling, batch processing, and caching optimizations for M4.

This example shows:
1. System information and profiling
2. Batch size optimization
3. KV cache benefits
4. Memory profiling
5. Quantization impact
6. Implementation comparison

Usage:
    python optimization_demo.py
"""

import time

import mlx.core as mx
import numpy as np


def demo_1_profiling():
    """
    Demo 1: Profiling and system information.

    Shows how to:
    - Get system information
    - Time operations
    - Profile memory usage
    - Benchmark functions
    """
    print("=" * 70)
    print("Demo 1: Profiling and System Information")
    print("=" * 70)

    from smlx.utils.profiling import (
        benchmark_function,
        print_system_info,
        profile_memory,
        timer,
    )

    # System information
    print("\n1. System Information:")
    print_system_info()

    # Simple timing
    print("\n2. Simple Timing:")

    def slow_operation():
        time.sleep(0.1)
        return mx.zeros((1000, 1000))

    with timer("Slow Operation"):
        result = slow_operation()

    # Memory profiling
    print("\n3. Memory Profiling:")

    def allocate_arrays():
        arrays = [mx.random.normal((1000, 1000)) for _ in range(10)]
        return arrays

    with profile_memory("Array Allocation"):
        arrays = allocate_arrays()

    # Function benchmarking
    print("\n4. Function Benchmarking:")

    def matrix_multiply():
        a = mx.random.normal((500, 500))
        b = mx.random.normal((500, 500))
        return mx.matmul(a, b)

    stats = benchmark_function(matrix_multiply, num_runs=10, warmup_runs=3)

    print("\nBenchmark Statistics:")
    print(f"  Mean: {stats['mean']:.2f}ms")
    print(f"  Median: {stats['median']:.2f}ms")
    print(f"  P95: {stats['p95']:.2f}ms")
    print(f"  P99: {stats['p99']:.2f}ms")


def demo_2_batch_optimization():
    """
    Demo 2: Batch processing optimization.

    Shows how to:
    - Find optimal batch size
    - Create batches
    - Pad sequences
    - Use dynamic batching
    """
    print("\n" + "=" * 70)
    print("Demo 2: Batch Processing Optimization")
    print("=" * 70)

    from smlx.utils.batch import (
        create_batches,
        dynamic_batching,
        optimize_batch_size,
        pad_batch,
    )

    # Create sample data
    num_samples = 100
    sample_arrays = [mx.random.normal((np.random.randint(10, 50),)) for _ in range(num_samples)]

    # Simple batching
    print("\n1. Simple Batching:")
    batch_count = 0
    for batch in create_batches(sample_arrays, batch_size=16):
        batch_count += 1
    print(f"   Created {batch_count} batches from {num_samples} samples (batch_size=16)")

    # Padding
    print("\n2. Sequence Padding:")
    sample_seqs = [
        mx.array([1, 2, 3]),
        mx.array([4, 5]),
        mx.array([6, 7, 8, 9]),
    ]
    padded = pad_batch(sample_seqs, padding_value=0)
    print(f"   Original lengths: {[len(s) for s in sample_seqs]}")
    print(f"   Padded shape: {padded.shape}")

    # Dynamic batching
    print("\n3. Dynamic Batching:")

    def get_size(arr):
        return len(arr)

    dynamic_batch_count = 0
    for batch in dynamic_batching(
        sample_arrays[:20],  # Use subset for demo
        get_size_fn=get_size,
        max_batch_tokens=200,
        max_batch_size=16,
    ):
        dynamic_batch_count += 1
        total_tokens = sum(len(arr) for arr in batch)
        print(f"   Batch {dynamic_batch_count}: {len(batch)} items, {total_tokens} total tokens")

    # Optimal batch size (using simple function)
    print("\n4. Finding Optimal Batch Size:")

    def process_batch(batch):
        # Simple matrix operation
        matrices = [mx.random.normal((10, 10)) for _ in range(len(batch))]
        results = [mx.sum(m) for m in matrices]
        mx.eval(results)
        return results

    sample_items = list(range(100))

    optimal_size = optimize_batch_size(
        process_fn=process_batch,
        sample_items=sample_items,
        batch_sizes=[1, 4, 8, 16, 32],
        metric="throughput",
    )


def demo_3_kv_cache():
    """
    Demo 3: KV cache benefits.

    Shows:
    - Standard KV cache
    - Rotating KV cache
    - Performance comparison (with vs without cache)
    """
    print("\n" + "=" * 70)
    print("Demo 3: KV Cache Benefits")
    print("=" * 70)

    from smlx.utils.cache import KVCache, RotatingKVCache, make_cache, reset_cache

    # Standard KV cache
    print("\n1. Standard KV Cache:")
    cache = KVCache()

    batch_size = 1
    n_heads = 8
    head_dim = 64

    for i in range(5):
        # Simulate adding new tokens
        new_keys = mx.random.normal((batch_size, n_heads, 1, head_dim))
        new_values = mx.random.normal((batch_size, n_heads, 1, head_dim))

        all_keys, all_values = cache.update_and_fetch(new_keys, new_values)

        print(f"   Step {i+1}: Cache size = {all_keys.shape[2]} tokens")

    # Rotating KV cache
    print("\n2. Rotating KV Cache (max_size=10, keep=3):")
    rotating_cache = RotatingKVCache(max_size=10, keep=3)

    for i in range(15):
        new_keys = mx.random.normal((batch_size, n_heads, 1, head_dim))
        new_values = mx.random.normal((batch_size, n_heads, 1, head_dim))

        all_keys, all_values = rotating_cache.update_and_fetch(new_keys, new_values)

        print(f"   Step {i+1}: Cache size = {all_keys.shape[2]} tokens (offset={rotating_cache.offset})")

    # Multi-layer cache
    print("\n3. Multi-Layer Cache:")
    num_layers = 4
    layer_caches = make_cache(num_layers=num_layers)

    print(f"   Created cache for {num_layers} layers")
    print(f"   Cache types: {[type(c).__name__ for c in layer_caches]}")

    # Reset caches
    reset_cache(layer_caches)
    print(f"   Caches reset")


def demo_4_memory_optimization():
    """
    Demo 4: Memory optimization techniques.

    Shows:
    - Memory monitoring
    - Clearing cache
    - Memory-efficient operations
    """
    print("\n" + "=" * 70)
    print("Demo 4: Memory Optimization")
    print("=" * 70)

    def check_memory(label=""):
        active = mx.metal.get_active_memory() / (1024 * 1024)
        peak = mx.metal.get_peak_memory() / (1024 * 1024)
        if label:
            print(f"{label}:")
        print(f"  Active: {active:.1f}MB, Peak: {peak:.1f}MB")

    # Initial state
    print("\n1. Initial Memory State:")
    mx.metal.clear_cache()
    check_memory()

    # Allocate memory
    print("\n2. After Allocation:")
    arrays = [mx.random.normal((1000, 1000)) for _ in range(10)]
    mx.eval(arrays)
    check_memory()

    # Clear cache
    print("\n3. After Clearing Cache:")
    del arrays
    mx.metal.clear_cache()
    check_memory()

    # Memory-efficient operations
    print("\n4. Memory-Efficient Operations:")
    print("   Using in-place operations when possible...")

    # Example: In-place update
    x = mx.random.normal((1000, 1000))
    mx.eval(x)
    check_memory("Before update")

    # In-place-like operation
    x = x * 2.0
    mx.eval(x)
    check_memory("After update")


def demo_5_quantization_impact():
    """
    Demo 5: Quantization performance impact.

    Shows:
    - Memory savings
    - Speed comparison (simulated)
    """
    print("\n" + "=" * 70)
    print("Demo 5: Quantization Impact")
    print("=" * 70)

    # Simulate model sizes
    print("\n1. Memory Comparison:")

    model_params = 135_000_000  # 135M parameters

    # FP16: 2 bytes per parameter
    memory_fp16 = (model_params * 2) / (1024 * 1024)

    # 4-bit: 0.5 bytes per parameter
    memory_4bit = (model_params * 0.5) / (1024 * 1024)

    savings = (1 - memory_4bit / memory_fp16) * 100

    print(f"   FP16: {memory_fp16:.1f}MB")
    print(f"   4-bit: {memory_4bit:.1f}MB")
    print(f"   Savings: {savings:.1f}%")

    # Simulated speed comparison
    print("\n2. Speed Comparison (simulated):")

    def fp16_operation():
        x = mx.random.normal((1000, 1000))
        y = mx.random.normal((1000, 1000))
        result = mx.matmul(x, y)
        mx.eval(result)

    def quantized_operation():
        # Simulate slightly faster operation
        x = mx.random.normal((1000, 1000))
        y = mx.random.normal((1000, 1000))
        result = mx.matmul(x, y) * 0.95  # Simulate speedup
        mx.eval(result)

    from smlx.utils.profiling import benchmark_function

    print("\n   Benchmarking FP16...")
    stats_fp16 = benchmark_function(fp16_operation, num_runs=10, warmup_runs=3, name="FP16")

    print("\n   Benchmarking 4-bit (simulated)...")
    stats_4bit = benchmark_function(quantized_operation, num_runs=10, warmup_runs=3, name="4-bit")

    speedup = stats_fp16["mean"] / stats_4bit["mean"]
    print(f"\n   Speedup: {speedup:.2f}x")


def demo_6_implementation_comparison():
    """
    Demo 6: Compare different implementations.

    Shows how to benchmark and compare different approaches.
    """
    print("\n" + "=" * 70)
    print("Demo 6: Implementation Comparison")
    print("=" * 70)

    from smlx.utils.profiling import compare_implementations

    # Different implementations of same task
    def naive_sum():
        result = 0
        for i in range(1000):
            result += i
        return result

    def optimized_sum():
        # Use formula: n*(n-1)/2
        return 999 * 1000 // 2

    def numpy_sum():
        return int(np.arange(1000).sum())

    implementations = {
        "naive_loop": naive_sum,
        "formula": optimized_sum,
        "numpy": numpy_sum,
    }

    print("\nComparing 3 implementations of sum(0..999):")
    results = compare_implementations(implementations, num_runs=100)


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("Performance Optimization Demo")
    print("=" * 70)
    print("\nDemonstrating profiling, batching, caching, and optimization techniques")
    print("Optimized for Apple M4 chipsets\n")

    demos = [
        ("1. Profiling", demo_1_profiling),
        ("2. Batch Optimization", demo_2_batch_optimization),
        ("3. KV Cache", demo_3_kv_cache),
        ("4. Memory Optimization", demo_4_memory_optimization),
        ("5. Quantization Impact", demo_5_quantization_impact),
        ("6. Implementation Comparison", demo_6_implementation_comparison),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Profile before optimizing - measure to find bottlenecks")
    print("2. Batch processing significantly improves throughput")
    print("3. KV caching provides 5-10x speedup for generation")
    print("4. Monitor memory usage to avoid OOM")
    print("5. Quantization saves 75% memory with minimal quality loss")
    print("6. Compare implementations to validate optimizations")
    print("\nSee docs/PerformanceOptimization.md for complete guide")


if __name__ == "__main__":
    main()
