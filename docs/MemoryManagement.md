# Memory Management in SMLX

Comprehensive guide to memory management utilities and best practices for running MLX models efficiently on Apple Silicon.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Memory Utilities](#memory-utilities)
  - [Memory Watchdog](#memory-watchdog)
  - [Robust Inference](#robust-inference)
  - [Graceful Degradation](#graceful-degradation)
  - [Model Profiles](#model-profiles)
  - [Memory Monitoring](#memory-monitoring)
  - [Debug Tools](#debug-tools)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

---

## Overview

SMLX provides a comprehensive memory management system designed to prevent crashes, handle out-of-memory (OOM) errors gracefully, and optimize performance on Apple Silicon devices with unified memory architecture.

### Key Features

- **Automatic Memory Monitoring**: Background watchdog process with configurable thresholds
- **Robust Inference**: Automatic retry with exponential backoff on OOM errors
- **Graceful Degradation**: Progressive parameter reduction under memory pressure
- **Model Profiles**: Pre-configured safe parameters for each model
- **Debug Tools**: Memory snapshots, leak detection, and profiling utilities
- **Smart Cleanup**: Intelligent cache management with optional Python GC

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (generate, chat, stream_generate, etc.)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Watchdog │   │  Robust  │   │Degradation│
│ (Monitor)│   │ (Retry)  │   │ (Adjust) │
└────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │
     └──────────────┼──────────────┘
                    ▼
        ┌──────────────────────┐
        │  Memory Utilities    │
        │ (Cleanup, Monitor)   │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │   MLX Metal Memory   │
        │  (Unified Memory)    │
        └──────────────────────┘
```

---

## Quick Start

### Basic Usage

```python
from smlx.models import load
from smlx.utils.generation import generate
from smlx.utils.watchdog import watchdog

# Load model
bm = load("smollm2-135m")
model, tokenizer = bm.model, bm.processor

# Use watchdog for automatic protection
with watchdog(warning_threshold=0.80, auto_cleanup=True):
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="Explain quantum computing:",
        max_tokens=200,
        temperature=0.7
    )
```

### With Automatic Retry

```python
from smlx.utils.robust import robust_generate

result = robust_generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Write a story:",
    max_tokens=500,
    temperature=0.8,
    max_retries=3
)

if result.success:
    print(result.text)
else:
    print(f"Failed: {result.error_message}")
```

### With Graceful Degradation

```python
from smlx.utils.degradation import with_graceful_degradation

params = with_graceful_degradation(
    max_tokens=500,
    temperature=0.7,
    batch_size=4
)

response = generate(model, tokenizer, prompt="Hello", **params)
```

---

## Memory Utilities

### Memory Watchdog

Automatic background monitoring with configurable thresholds and cleanup.

#### Features

- Real-time memory monitoring in background thread
- Configurable warning and critical thresholds
- Automatic cleanup on memory pressure
- Exception raising on critical threshold
- Graceful shutdown

#### Usage

```python
from smlx.utils.watchdog import watchdog

# Context manager (recommended)
with watchdog(
    warning_threshold=0.80,      # Warn at 80% memory
    critical_threshold=0.90,     # Raise exception at 90%
    auto_cleanup=True,           # Automatic cleanup on warning
    check_interval=1.0,          # Check every 1 second
    verbose=True                 # Log warnings
):
    # Your inference code here
    result = generate(...)

# Manual control
from smlx.utils.watchdog import MemoryWatchdog

watchdog = MemoryWatchdog(warning_threshold=0.80)
watchdog.start()
try:
    result = generate(...)
finally:
    watchdog.stop()
```

#### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warning_threshold` | float | 0.85 | Memory utilization to trigger warning (0-1) |
| `critical_threshold` | float | 0.95 | Memory utilization to raise exception (0-1) |
| `auto_cleanup` | bool | True | Automatically cleanup on warning |
| `check_interval` | float | 1.0 | Seconds between memory checks |
| `verbose` | bool | True | Log warnings and actions |

#### Environment Variables

```bash
# Disable watchdog globally
export SMLX_MEMORY_WATCHDOG_ENABLED=false

# Set default thresholds
export SMLX_MEMORY_WARNING_THRESHOLD=0.80
export SMLX_MEMORY_CRITICAL_THRESHOLD=0.90
```

---

### Robust Inference

Automatic retry with exponential backoff on OOM and other errors.

#### Features

- Automatic retry on OOM errors
- Exponential backoff between retries
- Progressive parameter reduction
- Detailed result tracking
- Configurable retry limits

#### Usage

```python
from smlx.utils.robust import robust_generate, RobustInferenceResult

result = robust_generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Explain machine learning:",
    max_tokens=500,
    temperature=0.7,
    max_retries=3,
    backoff_factor=1.5,
    min_tokens=50  # Minimum tokens to attempt
)

# Check result
if result.success:
    print(f"Success after {result.attempts} attempt(s)")
    print(f"Generated text: {result.text}")
    print(f"Final params: {result.final_params}")
else:
    print(f"Failed after {result.attempts} attempts")
    print(f"Error: {result.error_message}")
```

#### Retry Strategy

1. **First attempt**: Use provided parameters
2. **On OOM**: Reduce max_tokens by 50%, wait with exponential backoff
3. **Continue**: Repeat until success or max_retries reached
4. **Final attempt**: Try with min_tokens if all retries fail

#### Result Object

```python
@dataclass
class RobustInferenceResult:
    success: bool                    # Whether inference succeeded
    text: Optional[str]              # Generated text (if success)
    error_message: Optional[str]     # Error message (if failed)
    attempts: int                    # Number of attempts made
    final_params: Dict[str, Any]     # Final parameters used
    retry_history: List[Dict]        # History of all attempts
```

---

### Graceful Degradation

Progressive parameter reduction based on memory pressure.

#### Features

- Four degradation levels (NORMAL → REDUCED → MINIMAL → CRITICAL)
- Automatic parameter adjustment
- KV cache management
- Batch size reduction
- Temperature override

#### Usage

```python
from smlx.utils.degradation import GracefulDegradation, with_graceful_degradation

# Simple usage
params = with_graceful_degradation(
    max_tokens=500,
    temperature=0.7,
    batch_size=4
)
response = generate(model, tokenizer, prompt="Hello", **params)

# Advanced usage
degradation = GracefulDegradation(
    auto_cleanup=True,
    min_level=DegradationLevel.CRITICAL,
    verbose=True
)

level = degradation.get_current_level()
print(f"Current level: {level.value}")

adjusted_params = degradation.adjust_params(
    max_tokens=500,
    temperature=0.7,
    batch_size=4
)
```

#### Degradation Levels

| Level | Memory | max_tokens | temperature | batch_size | KV cache | Description |
|-------|--------|------------|-------------|------------|----------|-------------|
| **NORMAL** | <75% | 100% | As specified | 1x | Full | No degradation |
| **REDUCED** | 75-85% | 75% | As specified | 2x reduction | 2048, rotating | Moderate reduction |
| **MINIMAL** | 85-95% | 50% | Max 0.3 | 4x reduction | 1024, rotating | Heavy reduction |
| **CRITICAL** | >95% | 25% | 0.0 (greedy) | 8x reduction | 512, rotating | Emergency mode |

#### Auto-Cleanup

When `auto_cleanup=True`, the system automatically performs memory cleanup when degrading:
- **REDUCED/MINIMAL**: Aggressive cleanup (includes Python GC)
- **CRITICAL**: Emergency cleanup

---

### Model Profiles

Pre-configured safe parameters for each model based on available memory.

#### Features

- Memory tier detection (LOW <16GB, MEDIUM 16-32GB, HIGH >32GB)
- Model-specific safe parameters
- Automatic parameter selection
- Override support

#### Usage

```python
from smlx.config.model_profiles import auto_select_params, get_model_profile

# Auto-select parameters
safe_params = auto_select_params("SmolLM2-135M")
print(safe_params)
# {
#     'max_tokens': 512,
#     'max_kv_size': 4096,
#     'batch_size': 4,
#     'use_rotating_cache': False
# }

# Get full model profile
profile = get_model_profile("SmolVLM-256M")
print(profile.memory_estimate_gb)  # 1.2GB
print(profile.safe_params)         # Tier-specific params

# Use in generation
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Hello",
    **auto_select_params("SmolLM2-135M")
)
```

#### Available Profiles

**Language Models:**
- SmolLM2-135M (0.5GB)
- SmolLM2-360M (1.2GB)

**Vision-Language Models:**
- SmolVLM-256M (1.2GB)
- SmolVLM-500M-Instruct (2.0GB)
- nanoVLM (0.9GB)
- Moondream2 (2.0GB)
- TinyLLaVA-1.5B (5.5GB)
- TinyLLaVA-2.0B (7.5GB)
- TinyLLaVA-3.1B (11.5GB)

#### Memory Tiers

```python
# Automatic detection
tier = detect_memory_tier()  # LOW, MEDIUM, or HIGH

# Manual override
from smlx.config.model_profiles import MemoryTier

params = auto_select_params("SmolLM2-135M", tier=MemoryTier.LOW)
```

---

### Memory Monitoring

Real-time memory tracking and reporting.

#### Features

- Active memory tracking
- Cache memory tracking
- Peak memory tracking
- Utilization percentage
- Trend analysis

#### Usage

```python
from smlx.utils.memory import (
    get_active_memory_gb,
    get_cache_memory_gb,
    get_peak_memory_gb,
    print_memory_state,
    smart_cleanup,
    MemoryMonitor
)

# Simple queries
active_gb = get_active_memory_gb()
cache_gb = get_cache_memory_gb()
peak_gb = get_peak_memory_gb()
print(f"Active: {active_gb:.2f}GB, Cache: {cache_gb:.2f}GB, Peak: {peak_gb:.2f}GB")

# Print formatted state
print_memory_state("After model load")
# [After model load] Active: 2.34GB | Cache: 0.56GB | Total: 2.90GB

# Smart cleanup
freed_gb = smart_cleanup(aggressive=False)
print(f"Freed {freed_gb:.2f}GB")

freed_gb = smart_cleanup(aggressive=True)  # Includes Python GC
print(f"Freed {freed_gb:.2f}GB")

# Monitor with history
monitor = MemoryMonitor(history_size=100)
status = monitor.check()
print(f"Utilization: {status['utilization']:.1%}")
print(f"Trend: {status['trend']}")  # 'increasing', 'stable', 'decreasing'
```

#### MemoryMonitor API

```python
monitor = MemoryMonitor(history_size=100)

# Check status
status = monitor.check()
# {
#     'active_gb': 2.5,
#     'cache_gb': 0.5,
#     'total_gb': 3.0,
#     'peak_gb': 4.0,
#     'available_gb': 32.0,
#     'utilization': 0.09375,
#     'trend': 'stable'
# }

# Get recent history
history = monitor.get_history(n=10)

# Reset tracking
monitor.reset()
```

---

### Debug Tools

Advanced memory debugging and profiling utilities.

#### Memory Snapshots

```python
from smlx.utils.debug import MemorySnapshot, compare_snapshots

# Capture snapshots
before = MemorySnapshot.capture("Before generation")
result = generate(...)
after = MemorySnapshot.capture("After generation")

# Compare
diff = compare_snapshots(before, after)
print(diff)
# Memory change: Before generation → After generation
# Active: +0.45GB | Cache: +0.12GB | Total: +0.57GB | Time: 3.21s

# With context manager
from smlx.utils.debug import memory_snapshot_context

with memory_snapshot_context("Model inference") as diff_container:
    result = generate(...)

print(diff_container["diff"])
```

#### Layer-by-Layer Profiling

```python
from smlx.utils.debug import LayerMemoryProfiler

profiler = LayerMemoryProfiler()

# Profile each layer
for i, layer in enumerate(model.layers):
    with profiler.profile_layer(f"Layer {i}"):
        output = layer(input)
        mx.eval(output)

# Get report
print(profiler.get_report())

# Find worst layers
worst = profiler.find_worst_layers(top_n=5)
for name, delta_gb in worst:
    print(f"{name}: +{delta_gb:.2f}GB")
```

#### Graph Accumulation Detection

```python
from smlx.utils.debug import GraphAccumulationDetector

detector = GraphAccumulationDetector(
    window_size=10,
    threshold_gb=0.1
)

for i in range(100):
    output = model(input)
    detector.record()

    if detector.is_accumulating():
        print(f"WARNING: Graph accumulation at iteration {i}")
        mx.eval(output)  # Force evaluation
        break

rate = detector.get_growth_rate()
print(f"Memory growth rate: {rate:.3f}GB/iteration")
```

#### Leak Detection

```python
from smlx.utils.debug import detect_leaking_modules

# Run after inference
leaking = detect_leaking_modules()

# Show top 10 modules with most objects
for module, count in leaking[:10]:
    print(f"{module}: {count:,} objects")
```

---

## Configuration

### Environment Variables

Configure memory management globally via environment variables:

```bash
# Watchdog
export SMLX_MEMORY_WATCHDOG_ENABLED=true
export SMLX_MEMORY_WARNING_THRESHOLD=0.80
export SMLX_MEMORY_CRITICAL_THRESHOLD=0.90
export SMLX_MEMORY_CHECK_INTERVAL=1.0

# Cleanup
export SMLX_MEMORY_CLEANUP_THRESHOLD=0.75
export SMLX_MEMORY_AGGRESSIVE_CLEANUP=false

# Profiling
export SMLX_MEMORY_PROFILING_ENABLED=true
export SMLX_MEMORY_SNAPSHOT_INTERVAL=10
```

### Config File

Create `~/.smlx/memory.yaml`:

```yaml
memory:
  watchdog:
    enabled: true
    warning_threshold: 0.80
    critical_threshold: 0.90
    auto_cleanup: true
    check_interval: 1.0

  degradation:
    enabled: true
    min_level: "critical"
    auto_adjust: true

  cleanup:
    threshold: 0.75
    aggressive: false
    on_warning: true

  profiling:
    enabled: false
    snapshot_interval: 10
```

Load configuration:

```python
from smlx.config.memory import load_config, get_default_config

# Load custom config
config = load_config("~/.smlx/memory.yaml")

# Use default config
config = get_default_config()
```

---

## Best Practices

### 1. Always Use Watchdog in Production

```python
# Good
with watchdog(warning_threshold=0.80):
    result = generate(...)

# Bad - No protection
result = generate(...)
```

### 2. Use Robust Inference for Long Generations

```python
# Good - Automatic retry
result = robust_generate(
    model, tokenizer, prompt,
    max_tokens=1000,
    max_retries=3
)

# Risky - May OOM without retry
result = generate(model, tokenizer, prompt, max_tokens=1000)
```

### 3. Apply Model Profiles

```python
# Good - Use safe parameters
safe_params = auto_select_params("SmolVLM-256M")
result = generate(model, processor, prompt, **safe_params)

# Risky - May use too much memory
result = generate(model, processor, prompt, max_tokens=2048)
```

### 4. Cleanup Between Batches

```python
# Good
for batch in batches:
    results = process_batch(batch)
    smart_cleanup()  # Clear cache between batches

# Bad - Memory accumulates
for batch in batches:
    results = process_batch(batch)
```

### 5. Monitor Trends

```python
# Good - Track memory trends
monitor = MemoryMonitor()
for i in range(num_iterations):
    result = generate(...)
    status = monitor.check()
    if status['trend'] == 'increasing':
        print(f"Warning: Memory growing at iteration {i}")
        smart_cleanup(aggressive=True)
```

### 6. Use mx.eval() Correctly

```python
# Good - Evaluate regularly
for i, layer in enumerate(layers):
    output = layer(input)
    if i % 5 == 0:  # Every 5 layers
        mx.eval(output)

# Bad - Computation graph accumulates
for layer in layers:
    output = layer(input)
# Only evaluated at the end
mx.eval(output)
```

### 7. Profile Memory-Intensive Operations

```python
# Good - Profile during development
profiler = LayerMemoryProfiler()
for i, layer in enumerate(model.layers):
    with profiler.profile_layer(f"Layer {i}"):
        output = layer(input)
        mx.eval(output)

worst = profiler.find_worst_layers(top_n=3)
print("Most memory-intensive layers:", worst)
```

---

## Troubleshooting

### Out of Memory Errors

**Symptom**: `RuntimeError: Out of memory`

**Solutions**:

1. Use robust inference with automatic retry:
```python
result = robust_generate(model, tokenizer, prompt, max_retries=3)
```

2. Reduce max_tokens:
```python
result = generate(model, tokenizer, prompt, max_tokens=256)  # Instead of 512
```

3. Use graceful degradation:
```python
params = with_graceful_degradation(max_tokens=500)
result = generate(model, tokenizer, prompt, **params)
```

4. Enable aggressive cleanup:
```python
smart_cleanup(aggressive=True)
```

### Memory Leaks

**Symptom**: Memory usage grows over time

**Solutions**:

1. Check for graph accumulation:
```python
detector = GraphAccumulationDetector()
for i in range(100):
    output = model(input)
    detector.record()
    if detector.is_accumulating():
        print("Missing mx.eval() calls!")
        break
```

2. Find leaking modules:
```python
leaking = detect_leaking_modules()
for module, count in leaking[:10]:
    print(f"{module}: {count} objects")
```

3. Cleanup between iterations:
```python
for item in items:
    result = process(item)
    smart_cleanup()
```

### Slow Generation

**Symptom**: Generation slower than expected

**Solutions**:

1. Check memory pressure:
```python
status = monitor.check()
if status['utilization'] > 0.85:
    print("Memory pressure causing slowdown")
    smart_cleanup(aggressive=True)
```

2. Reduce KV cache size:
```python
result = generate(model, tokenizer, prompt, max_kv_size=2048)
```

3. Use rotating cache:
```python
result = generate(model, tokenizer, prompt, use_rotating_cache=True)
```

### Watchdog False Alarms

**Symptom**: Watchdog warnings during normal operation

**Solutions**:

1. Adjust thresholds:
```python
with watchdog(warning_threshold=0.90, critical_threshold=0.95):
    result = generate(...)
```

2. Disable auto-cleanup:
```python
with watchdog(auto_cleanup=False):
    result = generate(...)
```

3. Increase check interval:
```python
with watchdog(check_interval=2.0):  # Check less frequently
    result = generate(...)
```

---

## Examples

### Complete Production Example

```python
from smlx.models import load, generate
from smlx.utils.watchdog import watchdog
from smlx.utils.robust import robust_generate
from smlx.utils.degradation import with_graceful_degradation
from smlx.config.model_profiles import auto_select_params
from smlx.utils.memory import smart_cleanup, MemoryMonitor

# Initialize
bm = load("smolvlm-256m")
model, processor = bm.model, bm.processor
monitor = MemoryMonitor()

# Get safe parameters for this model
safe_params = auto_select_params("SmolVLM-256M")

# Process items with full protection
with watchdog(warning_threshold=0.80, auto_cleanup=True):
    for i, (image, prompt) in enumerate(items):
        # Check memory status
        status = monitor.check()
        if status['utilization'] > 0.75:
            print(f"High memory usage: {status['utilization']:.1%}")
            smart_cleanup(aggressive=True)

        # Apply graceful degradation if needed
        params = with_graceful_degradation(**safe_params)

        # Robust inference with retry
        result = robust_generate(
            model=model,
            tokenizer=processor,
            prompt=prompt,
            image=image,
            **params,
            max_retries=3
        )

        if result.success:
            print(f"Item {i}: Success after {result.attempts} attempt(s)")
            save_result(result.text)
        else:
            print(f"Item {i}: Failed - {result.error_message}")

        # Periodic cleanup
        if i % 10 == 0:
            smart_cleanup()
```

### Debugging Memory Issues

```python
from smlx.utils.debug import (
    MemorySnapshot,
    compare_snapshots,
    LayerMemoryProfiler,
    GraphAccumulationDetector,
    detect_leaking_modules
)
from smlx.models import load
from smlx.utils.generation import generate

# Load model with snapshot
before_load = MemorySnapshot.capture("Before load")
bm = load("smollm2-135m")
model, tokenizer = bm.model, bm.processor
after_load = MemorySnapshot.capture("After load")

diff = compare_snapshots(before_load, after_load)
print(f"Model loading: {diff}")

# Profile layers
profiler = LayerMemoryProfiler()
for i, layer in enumerate(model.layers):
    with profiler.profile_layer(f"Layer {i}"):
        output = layer(input_data)
        mx.eval(output)

print(profiler.get_report())
worst_layers = profiler.find_worst_layers(top_n=3)
print(f"\nWorst layers: {worst_layers}")

# Detect graph accumulation
detector = GraphAccumulationDetector()
for i in range(50):
    output = generate(model, tokenizer, "Test", max_tokens=10)
    detector.record()

    if detector.is_accumulating():
        rate = detector.get_growth_rate()
        print(f"Graph accumulation detected: {rate:.3f}GB/iteration")
        break

# Find memory leaks
leaking = detect_leaking_modules()
print("\nTop 10 modules with most objects:")
for module, count in leaking[:10]:
    print(f"  {module}: {count:,} objects")
```

### Server Integration

```python
from fastapi import FastAPI, HTTPException
from smlx.utils.watchdog import MemoryWatchdog
from smlx.utils.memory import smart_cleanup, MemoryMonitor

app = FastAPI()

# Global watchdog
watchdog = MemoryWatchdog(
    warning_threshold=0.85,
    critical_threshold=0.95,
    auto_cleanup=True
)
monitor = MemoryMonitor()

@app.on_event("startup")
async def startup():
    watchdog.start()
    print("Memory watchdog started")

@app.on_event("shutdown")
async def shutdown():
    watchdog.stop()
    print("Memory watchdog stopped")

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    # Check memory before processing
    status = monitor.check()
    if status['utilization'] > 0.90:
        smart_cleanup(aggressive=True)
        raise HTTPException(
            status_code=503,
            detail="Server under memory pressure"
        )

    # Apply graceful degradation
    params = with_graceful_degradation(
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )

    # Robust generation
    result = robust_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=request.prompt,
        **params,
        max_retries=2
    )

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {result.error_message}"
        )

    return {"text": result.text, "attempts": result.attempts}

@app.get("/health")
async def health():
    status = monitor.check()
    return {
        "status": "healthy" if status['utilization'] < 0.85 else "degraded",
        "memory_usage": f"{status['utilization']:.1%}",
        "trend": status['trend']
    }
```

---

## Reference

### Module Summary

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `smlx.utils.watchdog` | Background memory monitoring | `MemoryWatchdog`, `watchdog()` |
| `smlx.utils.robust` | Automatic retry on errors | `robust_generate()`, `RobustInferenceResult` |
| `smlx.utils.degradation` | Graceful parameter reduction | `GracefulDegradation`, `with_graceful_degradation()` |
| `smlx.config.model_profiles` | Model-specific parameters | `auto_select_params()`, `get_model_profile()` |
| `smlx.utils.memory` | Memory utilities | `smart_cleanup()`, `MemoryMonitor` |
| `smlx.utils.debug` | Debugging tools | `MemorySnapshot`, `LayerMemoryProfiler` |
| `smlx.config.memory` | Configuration | `load_config()`, `get_default_config()` |

### API Reference

Full API documentation available at: [API Docs](../README.md#api-reference)

---

## Contributing

To contribute memory management improvements:

1. Add tests in `tests/utils/test_memory.py`
2. Update documentation in this file
3. Follow existing patterns for consistency
4. Ensure backward compatibility

---

## License

Copyright © 2025 SMLX Project
