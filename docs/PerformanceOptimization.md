# Performance Optimization for M4

Guide for optimizing SMLX models on Apple M4 chipsets with unified memory.

## Table of Contents

- [Overview](#overview)
- [Profiling](#profiling)
- [Batch Processing](#batch-processing)
- [Caching](#caching)
- [Memory Optimization](#memory-optimization)
- [Quantization](#quantization)
- [Best Practices](#best-practices)

## Overview

SMLX is optimized for Apple M4 chipsets with:
- **Unified Memory**: Shared between CPU and GPU
- **Metal Acceleration**: GPU compute via MLX
- **Efficient Memory Access**: Optimized for Apple Silicon architecture

### Performance Targets (M4 Pro)

| Model Type | Memory (FP16) | Memory (4-bit) | Speed (tokens/sec) |
|-----------|---------------|----------------|-------------------|
| SmolLM2-135M | ~500MB | ~125MB | ~150 |
| SmolLM2-360M | ~1.5GB | ~375MB | ~100 |
| SmolVLM-256M | ~1GB | ~250MB | ~80 |
| Whisper-tiny | ~150MB | ~40MB | ~10x RT |

## Profiling

Use profiling utilities to identify bottlenecks:

### Basic Timing

```python
from smlx.utils.profiling import timer

with timer("Generation"):
    output = generate(model, tokenizer, prompt, max_tokens=100)

# Output: Generation: 123.45ms
```

### Memory Profiling

```python
from smlx.utils.profiling import profile_memory

with profile_memory("Model Loading"):
    model = load("smollm2-135m").model

# Output: Model Loading: 512.3MB
```

### Function Benchmarking

```python
from smlx.utils.profiling import benchmark_function

def my_operation():
    return model.generate(prompt)

stats = benchmark_function(my_operation, num_runs=20, warmup_runs=5)
print(f"Average: {stats['mean']:.2f}ms")
print(f"P95: {stats['p95']:.2f}ms")
```

### Generation Profiling

```python
from smlx.utils.profiling import profile_generation

result = profile_generation(
    model, tokenizer,
    prompt="Hello world",
    max_tokens=100,
    num_runs=10
)

print(result)
# Output:
# Text Generation:
#   Time: 123.45ms
#   Memory: 45.2MB
#   Speed: 152.3 tokens/sec
```

### System Information

```python
from smlx.utils.profiling import print_system_info

print_system_info()
# Output:
# ==================================================
# System Information
# ==================================================
# Platform: Darwin 24.5.0
# Processor: arm
# Python: 3.11.5
# Device: Apple Silicon (Metal)
# Memory: 36.0GB
# ==================================================
```

## Batch Processing

Batch processing significantly improves throughput:

### Basic Batching

```python
from smlx.utils.batch import create_batches

texts = ["text 1", "text 2", ..., "text 1000"]

for batch in create_batches(texts, batch_size=32):
    embeddings = model.encode(batch)
    # Process batch...
```

### Padding Sequences

```python
from smlx.utils.batch import pad_batch
import mlx.core as mx

sequences = [
    mx.array([1, 2, 3]),
    mx.array([4, 5]),
    mx.array([6, 7, 8, 9])
]

# Pad to same length
batch = pad_batch(sequences, padding_value=0)
# Shape: (3, 4) - all padded to length 4
```

### Dynamic Batching

For variable-length sequences, use dynamic batching to maximize GPU utilization:

```python
from smlx.utils.batch import dynamic_batching

texts = ["short text", "this is a much longer text...", ...]

def get_length(text):
    return len(text.split())

for batch in dynamic_batching(
    texts,
    get_size_fn=get_length,
    max_batch_tokens=2048,  # Total tokens per batch
    max_batch_size=32       # Max items per batch
):
    process_batch(batch)
```

### Batch Queue (Streaming)

For streaming applications:

```python
from smlx.utils.batch import BatchQueue

queue = BatchQueue(
    batch_size=32,
    process_fn=model.encode
)

# Add items as they arrive
for item in streaming_source:
    queue.add(item)
    # Automatically processes when batch is full

# Process remaining items
queue.flush()
```

### Finding Optimal Batch Size

```python
from smlx.utils.batch import optimize_batch_size

optimal_size = optimize_batch_size(
    process_fn=model.encode,
    sample_items=sample_texts[:100],
    batch_sizes=[1, 4, 8, 16, 32, 64],
    metric="throughput"  # or "latency"
)

print(f"Optimal batch size: {optimal_size}")
```

**Typical Optimal Batch Sizes:**
- **SmolLM2-135M**: 16-32 for generation, 64-128 for embeddings
- **SmolVLM-256M**: 8-16 for image+text
- **Whisper-tiny**: 4-8 for audio transcription

## Caching

KV caching dramatically speeds up autoregressive generation:

### Standard KV Cache

```python
from smlx.utils.cache import make_cache, reset_cache

# Create cache (one per layer)
cache = make_cache(num_layers=30)

# Use during generation
output = generate(
    model, tokenizer,
    prompt="Hello",
    max_tokens=100,
    cache=cache  # Reuse keys/values
)

# Reset for next generation
reset_cache(cache)
```

### Rotating KV Cache (Long Context)

For very long sequences, use rotating cache to limit memory:

```python
from smlx.utils.cache import make_cache

# Limit cache to 2048 tokens, keep first 256
cache = make_cache(
    num_layers=30,
    max_kv_size=2048,
    keep=256  # Always keep first 256 tokens (prompt)
)

# Generate very long sequences without OOM
output = generate(model, tokenizer, long_prompt, max_tokens=5000, cache=cache)
```

**Memory Savings:**
- **Without cache**: Recomputes all previous tokens (~N² operations)
- **With cache**: Only computes new token (~N operations)
- **Speedup**: 5-10x for typical generation lengths

## Memory Optimization

### Quantization

Reduce memory by 75% with 4-bit quantization:

```python
from smlx.quant import quantize_model

# Load model
model = load("smollm2-135m").model

# Quantize to 4-bit
model_4bit = quantize_model(model, bits=4)

# Memory: ~500MB → ~125MB
# Speed: Often 10-20% faster due to reduced memory bandwidth
```

See [Quantization Guide](Quant.md) for details.

### Memory-Efficient Generation

```python
# Limit max_tokens to control peak memory
output = generate(
    model, tokenizer,
    prompt="Hello",
    max_tokens=512,  # Instead of 2048
)

# Use streaming for long outputs
for chunk in stream_generate(model, tokenizer, prompt):
    print(chunk, end="", flush=True)
    # Process/save chunks immediately
```

### Clear Metal Cache

```python
import mlx.core as mx

# Clear MLX metal cache
mx.metal.clear_cache()

# Check memory usage
active_mb = mx.metal.get_active_memory() / (1024 * 1024)
peak_mb = mx.metal.get_peak_memory() / (1024 * 1024)

print(f"Active: {active_mb:.1f}MB, Peak: {peak_mb:.1f}MB")
```

## Best Practices

### 1. Choose Right Model Size

Start with smallest model that meets quality requirements:

```python
# For simple chat: SmolLM2-135M (~500MB)
model = load("smollm2-135m").model

# For better quality: SmolLM2-360M (~1.5GB)
model = load("smollm2-360m").model
```

### 2. Use Quantization by Default

4-bit quantization usually has minimal quality impact:

```python
from smlx.quant import quantize_model

model_4bit = quantize_model(model, bits=4)
# 75% memory reduction, often faster
```

### 3. Batch When Possible

Process multiple items together:

```python
# Inefficient
for text in texts:
    embedding = model.encode([text])

# Efficient
from smlx.utils.batch import batch_process

embeddings = batch_process(
    texts,
    process_fn=model.encode,
    batch_size=32
)
```

### 4. Reuse KV Cache

Don't recreate cache for each generation:

```python
# Create cache once
cache = make_cache(num_layers=30)

# Reuse across generations
for prompt in prompts:
    output = generate(model, tokenizer, prompt, cache=cache)
    reset_cache(cache)  # Reset between prompts
```

### 5. Profile Before Optimizing

Measure first to find actual bottlenecks:

```python
from smlx.utils.profiling import benchmark_function

# Benchmark current implementation
stats_v1 = benchmark_function(implementation_v1, num_runs=20)

# Benchmark optimized version
stats_v2 = benchmark_function(implementation_v2, num_runs=20)

# Compare
speedup = stats_v1['mean'] / stats_v2['mean']
print(f"Speedup: {speedup:.2f}x")
```

### 6. Monitor Memory

Keep an eye on memory usage:

```python
import mlx.core as mx

def check_memory():
    active = mx.metal.get_active_memory() / (1024**3)
    peak = mx.metal.get_peak_memory() / (1024**3)
    print(f"Memory - Active: {active:.2f}GB, Peak: {peak:.2f}GB")

# Check before/after operations
check_memory()
model = load("smollm2-360m").model
check_memory()
```

### 7. Optimize Data Loading

Minimize time spent loading data:

```python
# Bad: Load in loop
for file in files:
    data = load_file(file)
    process(data)

# Good: Preload and batch
data_batch = [load_file(f) for f in files]
for batch in create_batches(data_batch, 32):
    process(batch)
```

## M4-Specific Optimizations

### Unified Memory Benefits

M4's unified memory allows efficient CPU-GPU data sharing:

```python
# No explicit copy needed - unified memory!
import mlx.core as mx
import numpy as np

# NumPy array
np_array = np.random.randn(1000, 1000)

# Convert to MLX (fast, minimal copy)
mlx_array = mx.array(np_array)

# Compute on GPU
result = mx.matmul(mlx_array, mlx_array.T)

# Convert back (fast)
np_result = np.array(result)
```

### Metal Optimization

MLX automatically uses Metal for GPU acceleration:

```python
import mlx.core as mx

# All MLX operations use Metal automatically
x = mx.random.normal((1000, 1000))
y = mx.matmul(x, x.T)  # Runs on GPU via Metal

# No manual device management needed!
```

### Multi-Core CPU Utilization

For CPU-bound operations:

```python
from concurrent.futures import ThreadPoolExecutor

def process_item(item):
    # CPU-bound preprocessing
    return preprocess(item)

# Utilize multiple cores
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_item, items))
```

## Performance Checklist

Before deploying, verify:

- [ ] **Model size** - Smallest model that meets quality bar
- [ ] **Quantization** - Using 4-bit for production (75% memory savings)
- [ ] **Batch size** - Optimized via benchmarking
- [ ] **KV cache** - Enabled for generation tasks
- [ ] **Memory limits** - Stays within available RAM
- [ ] **Profiling** - Identified and addressed bottlenecks
- [ ] **Benchmarks** - Meets latency/throughput requirements

## Troubleshooting

### Out of Memory (OOM)

1. **Reduce batch size**
   ```python
   batch_size = 16  # Instead of 64
   ```

2. **Use quantization**
   ```python
   model_4bit = quantize_model(model, bits=4)
   ```

3. **Use rotating cache for long sequences**
   ```python
   cache = make_cache(num_layers=30, max_kv_size=2048)
   ```

4. **Clear cache regularly**
   ```python
   mx.metal.clear_cache()
   ```

### Slow Performance

1. **Profile to find bottleneck**
   ```python
   from smlx.utils.profiling import profile_generation
   result = profile_generation(model, tokenizer, prompt)
   ```

2. **Increase batch size** (if memory allows)
   ```python
   batch_size = 32  # Instead of 8
   ```

3. **Use quantization** (often faster)
   ```python
   model_4bit = quantize_model(model, bits=4)
   ```

4. **Ensure Metal is active**
   ```python
   import mlx.core as mx
   assert mx.metal.is_available()
   ```

### High Latency

1. **Reduce max_tokens**
   ```python
   max_tokens = 128  # Instead of 512
   ```

2. **Use streaming**
   ```python
   for chunk in stream_generate(model, tokenizer, prompt):
       print(chunk, end="")  # Show partial results
   ```

3. **Optimize batch size for latency**
   ```python
   optimal = optimize_batch_size(..., metric="latency")
   ```

## Additional Resources

- **Benchmarks**: See [BENCHMARKS.md](../BENCHMARKS.md) for reference performance
- **Quantization**: See [Quant.md](Quant.md) for quantization techniques
- **Examples**: See `examples/performance/` for optimization examples

## Summary

Key takeaways for M4 optimization:

1. **Profile first** - Measure before optimizing
2. **Use quantization** - 4-bit saves 75% memory with minimal quality loss
3. **Batch processing** - Dramatically improves throughput
4. **KV caching** - Essential for generation tasks (5-10x speedup)
5. **Right model size** - Smallest that meets requirements
6. **Monitor memory** - Stay within unified memory limits
7. **Leverage Metal** - MLX automatically optimizes for Apple Silicon

With these optimizations, SMLX models run efficiently on M4:
- **Low latency**: <100ms for most tasks
- **High throughput**: 100+ tokens/sec
- **Small memory**: <1GB for typical models
- **Great quality**: Competitive with much larger models
