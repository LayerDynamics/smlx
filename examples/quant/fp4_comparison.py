#!/usr/bin/env python3
"""
FP4 Quantization Comparison Example

This example demonstrates all FP4 quantization modes and compares their:
- Quantization quality (error metrics)
- Performance (quantization and dequantization speed)
- Memory usage
- Use cases

Modes compared:
1. E2M1 - Standard FP4 simulation (flexible group sizes)
2. MXFP4 - MLX native (hardware accelerated, group_size=32)
3. NVFP4 - MLX native (NVIDIA optimized, group_size=16)
4. NF4 - QLoRA Normal Float 4 (optimal for normal distributions)
5. INT4 - Standard integer quantization (baseline)
"""

import time
import mlx.core as mx
import mlx.nn as nn

from smlx.quant.fp4 import (
    quantize_fp4,
    dequantize_fp4,
    quantize_model_fp4,
    estimate_fp4_size,
    compare_fp4_vs_int4,
    FP4_E2M1_VALUES,
    NF4_VALUES,
)


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80 + "\n")


def benchmark_quantization(weights, mode, group_size=64, n_runs=10):
    """
    Benchmark quantization for a given mode.

    Args:
        weights: Weight array to quantize
        mode: FP4 mode ("e2m1", "mxfp4", "nvfp4", "nf4", "int4")
        group_size: Group size (ignored for mxfp4/nvfp4/int4)
        n_runs: Number of runs for timing

    Returns:
        Dictionary with benchmark results
    """
    # Warm-up
    if mode == "int4":
        w_q, scales, biases = mx.quantize(weights, group_size=group_size, bits=4)
        mx.eval(w_q, scales, biases)
    else:
        q, scales = quantize_fp4(weights, mode=mode, group_size=group_size)
        mx.eval(q, scales)

    # Benchmark quantization
    quant_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        if mode == "int4":
            w_q, scales, biases = mx.quantize(weights, group_size=group_size, bits=4)
            mx.eval(w_q, scales, biases)
        else:
            q, scales = quantize_fp4(weights, mode=mode, group_size=group_size)
            mx.eval(q, scales)
        end = time.perf_counter()
        quant_times.append((end - start) * 1000)  # Convert to ms

    # Benchmark dequantization
    dequant_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        if mode == "int4":
            restored = mx.dequantize(w_q, scales, biases, group_size=group_size, bits=4)
            mx.eval(restored)
        else:
            # Don't pass group_size for MXFP4/NVFP4 (auto-set by mode)
            if mode in ["mxfp4", "nvfp4"]:
                restored = dequantize_fp4(q, scales, mode=mode)
            else:
                restored = dequantize_fp4(q, scales, mode=mode, group_size=group_size)
            mx.eval(restored)
        end = time.perf_counter()
        dequant_times.append((end - start) * 1000)

    # Calculate error metrics
    if mode == "int4":
        restored_final = mx.dequantize(w_q, scales, biases, group_size=group_size, bits=4)
    else:
        # Don't pass group_size for MXFP4/NVFP4 (auto-set by mode)
        if mode in ["mxfp4", "nvfp4"]:
            restored_final = dequantize_fp4(q, scales, mode=mode)
        else:
            restored_final = dequantize_fp4(q, scales, mode=mode, group_size=group_size)

    mean_error = float(mx.mean(mx.abs(restored_final - weights)))
    max_error = float(mx.max(mx.abs(restored_final - weights)))
    mse = float(mx.mean((restored_final - weights) ** 2))

    # Calculate memory usage
    if mode == "int4":
        # INT4: 4 bits per weight + scales (float32) + biases (float32)
        n_groups = (weights.size + group_size - 1) // group_size
        memory_bytes = (weights.size // 2) + (n_groups * 8)  # 4 bits + 2*float32
    else:
        # FP4: indices (uint8 or quantized) + scales (float16)
        if mode in ["mxfp4", "nvfp4"]:
            # Native quantized format
            memory_bytes = q.nbytes + scales.nbytes
        else:
            # E2M1/NF4: indices (uint8) + scales (float16)
            memory_bytes = q.nbytes + scales.nbytes

    return {
        "mode": mode,
        "quant_time_ms": sum(quant_times) / n_runs,
        "dequant_time_ms": sum(dequant_times) / n_runs,
        "mean_error": mean_error,
        "max_error": max_error,
        "mse": mse,
        "memory_mb": memory_bytes / (1024**2),
        "compression_ratio": weights.nbytes / memory_bytes,
    }


def demo_lookup_tables():
    """Demonstrate FP4 lookup tables."""
    print_header("FP4 Lookup Tables")

    print("E2M1 Values (Standard FP4):")
    print("-" * 40)
    print(f"Values: {FP4_E2M1_VALUES.tolist()}")
    print(f"Range: [{float(mx.min(FP4_E2M1_VALUES)):.2f}, {float(mx.max(FP4_E2M1_VALUES)):.2f}]")
    print(f"Spacing: Exponential (powers of 2)")
    print()

    print("NF4 Values (Normal Float 4 - QLoRA):")
    print("-" * 40)
    print(f"Values: {[f'{v:.4f}' for v in NF4_VALUES.tolist()]}")
    print(f"Range: [{float(mx.min(NF4_VALUES)):.2f}, {float(mx.max(NF4_VALUES)):.2f}]")
    print(f"Spacing: Non-uniform (optimized for N(0,1))")


def demo_quantization_quality():
    """Demonstrate quantization quality for different distributions."""
    print_header("Quantization Quality Analysis")

    # Test 1: Normal distribution (NF4 should excel)
    print("Test 1: Normal Distribution N(0, 1)")
    print("-" * 40)
    mx.random.seed(42)
    normal_weights = mx.random.normal((512, 512))

    for mode in ["e2m1", "nf4", "mxfp4", "int4"]:
        result = benchmark_quantization(normal_weights, mode, group_size=64, n_runs=5)
        print(f"{mode:8s}: Mean Error = {result['mean_error']:.6f}, "
              f"Max Error = {result['max_error']:.6f}")
    print()

    # Test 2: Uniform distribution
    print("Test 2: Uniform Distribution [-5, 5]")
    print("-" * 40)
    uniform_weights = mx.random.uniform(-5.0, 5.0, (512, 512))

    for mode in ["e2m1", "nf4", "mxfp4", "int4"]:
        result = benchmark_quantization(uniform_weights, mode, group_size=64, n_runs=5)
        print(f"{mode:8s}: Mean Error = {result['mean_error']:.6f}, "
              f"Max Error = {result['max_error']:.6f}")
    print()

    # Test 3: Wide dynamic range (E2M1/FP4 should excel)
    print("Test 3: Wide Dynamic Range [0.001, 100]")
    print("-" * 40)
    wide_weights = mx.exp(mx.random.normal((512, 512)) * 2)

    for mode in ["e2m1", "nf4", "mxfp4", "int4"]:
        result = benchmark_quantization(wide_weights, mode, group_size=64, n_runs=5)
        print(f"{mode:8s}: Mean Error = {result['mean_error']:.6f}, "
              f"Max Error = {result['max_error']:.6f}")


def demo_performance_comparison():
    """Benchmark performance of all FP4 modes."""
    print_header("Performance Comparison")

    # Generate test weights
    mx.random.seed(123)
    weights = mx.random.normal((1024, 1024))

    modes = [
        ("e2m1", 64),
        ("mxfp4", 32),
        ("nvfp4", 16),
        ("nf4", 64),
        ("int4", 64),
    ]

    results = []
    for mode, group_size in modes:
        result = benchmark_quantization(weights, mode, group_size, n_runs=20)
        results.append(result)

    # Print results table
    print(f"{'Mode':<10} {'Quant (ms)':<12} {'Dequant (ms)':<13} "
          f"{'Memory (MB)':<12} {'Compression':<12}")
    print("-" * 70)

    for r in results:
        print(f"{r['mode']:<10} {r['quant_time_ms']:>10.3f}   "
              f"{r['dequant_time_ms']:>11.3f}   "
              f"{r['memory_mb']:>10.4f}   {r['compression_ratio']:>10.2f}x")

    print()
    print("Observations:")
    print("- MXFP4/NVFP4 are typically fastest (hardware accelerated)")
    print("- E2M1 is most flexible (custom group sizes)")
    print("- NF4 is optimal for normally distributed weights")
    print("- All FP4 modes provide ~4x compression vs FP16")


def demo_group_size_effects():
    """Demonstrate effects of different group sizes."""
    print_header("Group Size Effects (E2M1 Mode)")

    mx.random.seed(42)
    weights = mx.random.normal((512, 512))

    group_sizes = [16, 32, 64, 128, 256]

    print(f"{'Group Size':<12} {'Mean Error':<12} {'Memory (MB)':<12} "
          f"{'Quant Time (ms)':<18}")
    print("-" * 60)

    for gs in group_sizes:
        result = benchmark_quantization(weights, "e2m1", group_size=gs, n_runs=10)
        print(f"{gs:<12} {result['mean_error']:<12.6f} "
              f"{result['memory_mb']:<12.4f} {result['quant_time_ms']:<18.3f}")

    print()
    print("Observations:")
    print("- Larger group sizes: Lower memory, higher error")
    print("- Smaller group sizes: Higher memory, lower error")
    print("- Typical choice: 64 (good balance)")


def demo_model_quantization():
    """Demonstrate model-level quantization."""
    print_header("Model-Level Quantization")

    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(512, 256)
            self.layer2 = nn.Linear(256, 128)
            self.layer3 = nn.Linear(128, 64)

        def __call__(self, x):
            x = nn.relu(self.layer1(x))
            x = nn.relu(self.layer2(x))
            return self.layer3(x)

    model = TestModel()

    # Estimate size reduction
    print("Model Size Estimation:")
    print("-" * 40)
    stats = estimate_fp4_size(model, group_size=64)
    print(f"Original size: {stats['current_mb']:.2f} MB")
    print(f"FP4 size (E2M1): {stats['fp4_mb']:.2f} MB")
    print(f"Reduction ratio: {stats['reduction_ratio']:.2f}x")
    print(f"Space saved: {stats['saved_mb']:.2f} MB")
    print()

    # Quantize with different modes
    print("Quantizing model with different modes:")
    print("-" * 40)

    for mode in ["e2m1", "mxfp4", "nf4"]:
        start = time.perf_counter()
        quantized_weights = quantize_model_fp4(model, mode=mode)
        end = time.perf_counter()

        print(f"{mode:8s}: {len(quantized_weights)} layers quantized "
              f"in {(end - start) * 1000:.2f} ms")


def demo_fp4_vs_int4():
    """Compare FP4 vs INT4 using built-in comparison."""
    print_header("FP4 vs INT4 Detailed Comparison")

    # Test different weight distributions
    test_cases = [
        ("Normal N(0,1)", mx.random.normal((512, 512))),
        ("Uniform [-1, 1]", mx.random.uniform(-1.0, 1.0, (512, 512))),
        ("Heavy-tailed", mx.random.normal((512, 512)) * 3),
    ]

    for name, weights in test_cases:
        print(f"\n{name}:")
        print("-" * 40)
        comparison = compare_fp4_vs_int4(weights, group_size=64)

        print(f"FP4 Error (mean): {comparison['fp4_error']:.6f}")
        print(f"INT4 Error (mean): {comparison['int4_error']:.6f}")
        print(f"FP4 Error (max): {comparison['fp4_max_error']:.6f}")
        print(f"INT4 Error (max): {comparison['int4_max_error']:.6f}")
        print(f"Improvement Ratio: {comparison['improvement_ratio']:.2f}x")
        print(f"Recommendation: {comparison['recommendation']}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("FP4 Quantization Comprehensive Comparison".center(80))
    print("=" * 80)

    # Set random seed for reproducibility
    mx.random.seed(42)

    # Run all demonstrations
    demo_lookup_tables()
    demo_quantization_quality()
    demo_performance_comparison()
    demo_group_size_effects()
    demo_model_quantization()
    demo_fp4_vs_int4()

    print_header("Summary")
    print("""
Key Takeaways:

1. **E2M1 (Standard FP4)**:
   - Best for: Research, custom experiments, flexible group sizes
   - Pros: Flexible, wide dynamic range
   - Cons: Requires dequantization, not hardware accelerated

2. **MXFP4 (MLX Native)**:
   - Best for: Production inference on Apple Silicon
   - Pros: Hardware accelerated, same E2M1 format
   - Cons: Fixed group_size=32

3. **NVFP4 (NVIDIA Optimized)**:
   - Best for: NVIDIA GPU inference
   - Pros: Hardware optimized, same E2M1 format
   - Cons: Fixed group_size=16

4. **NF4 (QLoRA)**:
   - Best for: QLoRA fine-tuning, normally distributed weights
   - Pros: Information-theoretically optimal for N(0,1)
   - Cons: Requires dequantization

5. **INT4 (Baseline)**:
   - Best for: Broad hardware support
   - Pros: Widely supported, good quantization libraries (GPTQ, AWQ)
   - Cons: Narrower dynamic range than FP4

Recommendations:
- For production inference: Use MXFP4 or NVFP4
- For QLoRA fine-tuning: Use NF4
- For research/analysis: Use E2M1
- For best compatibility: Use INT4 (GPTQ/AWQ)
    """)


if __name__ == "__main__":
    main()
