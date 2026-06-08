# Enhanced KV Cache System

**SMLX Enhanced KV Cache** provides advanced memory management, quantization, and automatic OOM prevention for transformer models.

> **API note:** the basic cache factory now lives in `smlx.utils.cache`
> (`make_cache`, `make_kv_caches`, `reset_cache`, `KVCache`, `RotatingKVCache`).
> The combined cache-plus-pressure-monitoring helper that earlier examples call
> `make_cache_with_monitoring()` (returning a `(cache, breaker)` pair) was replaced
> by `smlx.kv_cache.KVCacheManager` — create a monitored cache with
> `KVCacheManager.create_auto(num_layers=..., model_size_gb=..., enable_monitoring=True)`
> (also `create_standard` / `create_rotating` / `create_quantized`). The
> `make_cache_with_monitoring(...)` snippets below are retained as a conceptual
> reference for the monitoring behaviour; use the `KVCacheManager` factory in code.

## Overview

The enhanced cache system builds on the standard KV cache implementation with:

- **Automatic cache sizing** based on available system memory
- **4-bit and 8-bit quantized caches** for memory efficiency
- **Memory pressure monitoring** with real-time alerts
- **Automatic OOM prevention** via PressureBreaker
- **Rotating caches** for long-context generation
- **Backward compatibility** with existing code

## Supported Models

Caching is **model-agnostic** — there are no longer per-model cache modules. The
shared factories in `smlx.utils.cache` (`make_cache`, `make_kv_caches`) build a
correct per-layer KV cache for any MLX model loaded through `smlx.models.load`, and
`smlx.kv_cache.KVCacheManager` adds memory-pressure monitoring on top. The number
of cache layers is read from the model's own config (`model.args.num_hidden_layers`),
so the same code works for SmolLM2, SmolVLM, moondream3, and any other curated model.

## Quick Start

### Basic Usage (Automatic Selection)

```python
from smlx.models import load, generate
from smlx.utils.cache import make_cache

# Load model
model, tokenizer = load()

# Create cache with automatic configuration
cache = make_cache(
    model,
    cache_type="auto",  # Automatically select best cache type
    target_memory_gb=32.0  # Target total memory usage
)

# Generate text
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="The future of AI is",
    max_tokens=100
)
```

### Quantized Cache (Memory Efficient)

```python
from smlx.utils.cache import make_cache

# Create 4-bit quantized cache (~4x memory reduction)
cache = make_cache(
    model,
    cache_type="quantized",
    quantization_bits=4,  # 4-bit or 8-bit
    max_kv_size=4096  # Maximum sequence length
)
```

**Memory Savings:**
- 4-bit quantization: ~4x reduction vs FP16
- 8-bit quantization: ~2x reduction vs FP16

### Memory Monitoring & OOM Prevention

```python
from smlx.utils.cache import make_cache
from smlx.utils.generation import generate_step

# Standard per-layer KV cache. For built-in memory-pressure monitoring, create
# the cache through smlx.kv_cache.KVCacheManager instead:
#   from smlx.kv_cache import KVCacheManager
#   cache = KVCacheManager.create_auto(
#       num_layers=model.args.num_hidden_layers,
#       model_size_gb=0.5,
#       enable_monitoring=True,
#   )
cache = make_cache(model)

# Generate against the cache
for step, (token, _) in enumerate(generate_step(model, prompt_tokens, cache=cache)):
    # ... process token ...
    pass
```

### Rotating Cache (Fixed Memory)

```python
from smlx.utils.cache import make_cache

# Create rotating cache (drops old tokens when full)
cache = make_cache(
    model,
    cache_type="rotating",
    max_kv_size=2048  # Keep only 2048 most recent tokens
)
```

**Use case:** Long-context generation with limited memory.

## Cache Types

### 1. Standard Cache (`cache_type="standard"`)

- **Unlimited growth** - cache grows with sequence length
- **Best for:** Short to medium sequences with sufficient memory
- **Memory:** Grows linearly with sequence length

```python
cache = make_cache(model, cache_type="standard")
```

### 2. Auto Cache (`cache_type="auto"`)

- **Automatic selection** based on available memory
- **Best for:** Production use, uncertain memory constraints
- **Behavior:** Selects optimal cache type for your system

```python
cache = make_cache(model, cache_type="auto", target_memory_gb=32.0)
```

### 3. Rotating Cache (`cache_type="rotating"`)

- **Fixed size** - drops old tokens when full
- **Best for:** Very long sequences, limited memory
- **Memory:** Constant (based on `max_kv_size`)

```python
cache = make_cache(model, cache_type="rotating", max_kv_size=2048)
```

### 4. Quantized Cache (`cache_type="quantized"`)

- **Compressed storage** - 4-bit or 8-bit quantization
- **Best for:** Memory-constrained environments
- **Memory:** ~2-4x reduction vs FP16

```python
cache = make_cache(
    model,
    cache_type="quantized",
    quantization_bits=4,  # 4 or 8
    max_kv_size=4096
)
```

## Advanced Features

### Memory Pressure Monitoring

The `PressureBreaker` automatically monitors memory usage and intervenes when thresholds are exceeded:

```python
cache, breaker = make_cache_with_monitoring(
    model,
    warning_threshold=0.8,  # Warn at 80%
    critical_threshold=0.9   # Critical at 90%
)

# During generation
intervention = breaker.monitor_and_intervene(current_step=step)

if intervention:
    print(f"Action: {intervention['action']}")
    print(f"Reason: {intervention['reason']}")
    print(f"Memory: {intervention['current_memory']:.2f} GB")
```

**Intervention Actions:**
- `reset_cache` - Clear KV cache
- `gc_collect` - Run garbage collection
- `reduce_batch` - Suggest batch size reduction
- `quantize_cache` - Suggest quantization

### Cache Statistics

```python
# Get monitoring statistics
stats = breaker.get_statistics()

print(f"Total interventions: {stats['total_interventions']}")
print(f"Enabled: {stats['enabled']}")
print(f"Thresholds: warning={stats['warning_threshold']}, "
      f"critical={stats['critical_threshold']}")
```

### Custom Cache Configuration

```python
from smlx.kv_cache import KVCacheManager

# Create custom cache manager
manager = KVCacheManager.create_quantized(
    num_layers=30,
    bits=4,
    max_size=2048,
    enable_monitoring=True
)

# Convert to list for model compatibility
cache = list(manager)
```

## Vision-Language Models

Enhanced cache works with VLM models' language components:

```python
from smlx.models import load
from smlx.kv_cache import KVCacheManager

bm = load("smolvlm-256m")
model, processor = bm.model, bm.processor

# Create a monitored cache for the VLM's language model (see the API note above).
cache = KVCacheManager.create_auto(
    num_layers=model.language_model.args.num_hidden_layers,
    model_size_gb=0.5,
    enable_monitoring=True,
)

# Use with generation (cache is automatically managed)
response = generate(
    model=model,
    processor=processor,
    prompt="Describe this image:",
    image=image,
    max_tokens=100
)
```

## Backward Compatibility

The enhanced cache is fully backward compatible with existing code:

```python
# Old API - still works!
from smlx.utils.cache import make_cache

cache = make_cache(num_layers=30, max_kv_size=2048)

# New API - enhanced features
from smlx.utils.cache import make_cache

cache = make_cache(model, cache_type="quantized", quantization_bits=4)
```

**Migration:** Existing code continues to work unchanged. New features are opt-in.

## Performance Considerations

### Memory vs Quality Trade-offs

| Cache Type | Memory Usage | Quality | Speed |
|------------|--------------|---------|-------|
| Standard | High | ✓✓✓ | ✓✓✓ |
| Rotating (2048) | Medium | ✓✓ | ✓✓✓ |
| Quantized (8-bit) | Medium | ✓✓✓ | ✓✓ |
| Quantized (4-bit) | Low | ✓✓ | ✓✓ |

### Best Practices

1. **Start with `cache_type="auto"`** - Let the system choose optimal configuration
2. **Use quantization for long sequences** - 4-bit or 8-bit reduces memory significantly
3. **Enable monitoring in production** - Prevents OOM errors in deployed systems
4. **Profile your workload** - Measure memory usage for your specific use case
5. **Adjust `target_memory_gb`** - Set based on your system's available memory

### Benchmarking

```python
from smlx.bench.suites.llm import benchmark_llm, LLMBenchmarkConfig

# Benchmark with enhanced cache
config = LLMBenchmarkConfig(
    generation_tokens=200,
    cache_type="quantized",
    quantization_bits=4,
    enable_monitoring=True
)

stats = benchmark_llm(model, tokenizer, config=config)
print(f"Tokens/sec: {stats.tokens_per_second:.1f}")
print(f"Peak memory: {stats.peak_memory_gb:.2f} GB")
```

## Troubleshooting

### OOM Errors

If you encounter out-of-memory errors:

1. **Enable monitoring:**
   ```python
   cache, breaker = make_cache_with_monitoring(model, target_memory_gb=32.0)
   ```

2. **Use quantized cache:**
   ```python
   cache = make_cache(model, cache_type="quantized", quantization_bits=4)
   ```

3. **Use rotating cache:**
   ```python
   cache = make_cache(model, cache_type="rotating", max_kv_size=1024)
   ```

4. **Reduce batch size** or sequence length

### Quality Degradation

If generation quality decreases with quantized/rotating cache:

1. **Increase quantization bits:** 4-bit → 8-bit
2. **Increase `max_kv_size`** for rotating cache
3. **Use standard cache** if memory allows

### Performance Issues

If generation is slow:

1. **Disable monitoring** if not needed in production
2. **Use standard cache** (fastest)
3. **Profile with benchmarks** to identify bottlenecks

## API Reference

### `make_cache()`

```python
def make_cache(
    model: Model,
    max_kv_size: int | None = None,
    cache_type: CacheType | Literal["auto"] = "auto",
    enable_quantization: bool = False,
    quantization_bits: int = 4,
    enable_monitoring: bool = False,
    target_memory_gb: float = 32.0,
) -> list
```

**Parameters:**
- `model` - Language model instance
- `max_kv_size` - Maximum cache size (tokens)
- `cache_type` - Cache type: "auto", "standard", "rotating", "quantized"
- `enable_quantization` - Enable quantization
- `quantization_bits` - 4 or 8
- `enable_monitoring` - Enable memory monitoring
- `target_memory_gb` - Target total memory usage

**Returns:** List of cache instances (one per layer)

### `make_cache_with_monitoring()`

```python
def make_cache_with_monitoring(
    model: Model,
    max_kv_size: int | None = None,
    cache_type: CacheType | Literal["auto"] = "auto",
    enable_quantization: bool = False,
    quantization_bits: int = 4,
    target_memory_gb: float = 32.0,
    warning_threshold: float = 0.8,
    critical_threshold: float = 0.9,
) -> tuple[list, PressureBreaker]
```

**Parameters:** Same as `make_cache()` plus:
- `warning_threshold` - Memory warning threshold (0.0-1.0)
- `critical_threshold` - Memory critical threshold (0.0-1.0)

**Returns:** Tuple of (cache_list, pressure_breaker)

## Examples

See [`examples/cache/enhanced_cache_demo.py`](../examples/cache/enhanced_cache_demo.py) for comprehensive examples:

1. Basic enhanced cache with auto mode
2. Quantized cache (4-bit)
3. Memory monitoring and OOM prevention
4. Rotating cache for long sequences
5. Vision-language model integration
6. Cache configuration comparison

Run the examples:

```bash
python examples/cache/enhanced_cache_demo.py
```

## Core Infrastructure

The enhanced cache system is built on the `smlx.kv_cache` module:

- `KVCacheManager` - Central cache management
- `MemoryPressureGauge` - Memory monitoring
- `PressureBreaker` - Automatic intervention
- `CacheLimitManager` - Automatic size computation
- `QuantizedKVCache` - 4-bit/8-bit quantized storage

See `smlx/kv_cache/` for implementation details.

## Related Documentation

- [Benchmarking Guide](BENCHMARKS.md) - Performance benchmarking with enhanced caches
- [Model Implementation Guide](ModelImplementations.md) - Implementing cache-aware models
- [Performance Optimization](PerformanceOptimization.md) - General optimization strategies

## Migration Guide

### From Standard Cache

**Before:**
```python
from smlx.utils.cache import make_cache

cache = make_cache(num_layers=30)
```

**After:**
```python
from smlx.utils.cache import make_cache

cache = make_cache(model, cache_type="auto")
```

### From Manual Cache Management

**Before:**
```python
# Manual cache size management
max_size = compute_safe_cache_size(model, memory_gb=32.0)
cache = make_cache(num_layers=30, max_kv_size=max_size)
```

**After:**
```python
# Automatic sizing and monitoring
cache, breaker = make_cache_with_monitoring(
    model,
    cache_type="auto",
    target_memory_gb=32.0
)
```

## Contributing

Caching is **model-agnostic** — there are no per-model cache modules. The shared factories live in smlx/utils/cache.py (make_cache, make_kv_caches, reset_cache, KVCache, RotatingKVCache) and expect the layer count as an integer (num_layers). Memory-pressure monitoring and advanced cache configurations (such as quantized or rotating caches) are provided by smlx/kv_cache/kv_manager.py (KVCacheManager.create_*(..., enable_monitoring=True)). To extend caching behaviour, work in those two modules — they apply to every curated model automatically.

## License

Copyright © 2025 SMLX Project. All rights reserved.
