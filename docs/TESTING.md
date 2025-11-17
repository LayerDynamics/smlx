# Testing Guide - Memory Safety for Integration Tests

## ⚠️ Critical: Kernel Panic Prevention

Integration tests load multiple large models (16 test files × 100MB-3GB each = 10-12GB+ memory). **As of January 2025, all integration test fixtures have been fixed with explicit memory cleanup to prevent kernel panics.**

### ✅ FIXED: All Test Fixtures Now Have Proper Cleanup

Every integration test fixture now includes:

- **Explicit teardown** with `yield` instead of `return`
- **Mandatory cleanup**: `del model`, `mx.clear_cache()`, `gc.collect()`
- **Memory guards** for heavy models (>500MB)
- **Memory profiling** for the 5 largest models

**You can now safely run integration tests sequentially** without kernel panics. However, running all 16 files at once is still discouraged due to cumulative memory pressure.

## Problem

The kernel panic you experienced was caused by:

1. **Module-scoped fixtures** - Each test file keeps a model in memory for all its tests
2. **16 integration test files** - Loading different VLMs/LLMs simultaneously
3. **Memory exhaustion** - Swap space filled to 100%, causing watchdog timeout

```
Compressor Info: 35% of compressed pages limit (OK) and 100% of segments limit (BAD)
panic(cpu 5): watchdog timeout: no checkins from watchdogd in 94 seconds
```

## Solutions

### ✅ Solution 1: Run Tests in Batches (Recommended)

Use the batching script to run integration tests in safe groups:

```bash
# Run all integration tests in safe batches (3 files at a time)
python scripts/run_integration_tests.py

# Smaller batches (2 files at a time) for safer memory usage
python scripts/run_integration_tests.py --batch-size 2

# Run specific groups only
python scripts/run_integration_tests.py --group llm    # Language models only
python scripts/run_integration_tests.py --group vlm    # Vision-language models only
python scripts/run_integration_tests.py --group audio  # Audio models only
python scripts/run_integration_tests.py --group ocr    # OCR models only
```

### ✅ Solution 2: Run Individual Test Files

Run one test file at a time:

```bash
# Safest option - run one file at a time
python -m pytest tests/integration/test_smollm2_generation.py -v
python -m pytest tests/integration/test_smolvlm_256m.py -v
python -m pytest tests/integration/test_moondream2.py -v
```

### ✅ Solution 3: Run by Group with Wildcards

```bash
# LLM tests only
python -m pytest tests/integration/test_smollm2*.py -v

# VLM tests (careful - these are larger)
python -m pytest tests/integration/test_*vlm*.py tests/integration/test_moondream2.py -v

# Audio tests
python -m pytest tests/integration/test_whisper*.py tests/integration/test_*vad.py -v
```

### ❌ DO NOT: Run All Integration Tests at Once

```bash
# ⛔ DANGER: Will cause kernel panic!
python -m pytest tests/integration/ -v

# ⛔ DANGER: Will cause kernel panic!
python -m pytest
```

## Memory Monitoring

The test suite now monitors memory usage:

- **At session start**: Shows available memory and warns if < 10GB
- **Per module**: Tracks memory before/after each test file
- **Auto-cleanup**: Aggressive garbage collection and MLX cache clearing between files

Example output:

```
======================================================================
🚀 SMLX Test Session Starting
======================================================================
Initial memory: 2847.3 MB
Available system memory: 28.4 GB / 36.0 GB

📊 Module starting - Memory: 2912.5 MB
[... tests run ...]
📊 Module complete - Memory: 3421.8 MB (Δ +509.3 MB)
```

## Test Groups

Tests are organized by model type:

- **llm** (4 files): SmolLM2-135M, SmolLM2-360M, MiniLM, all-MiniLM-L6-v2
- **vlm** (5 files): SmolVLM-256M, SmolVLM-500M, nanoVLM, TinyLLaVA, Moondream2
- **audio** (5 files): Whisper-tiny, Chatterbox, Orpheus, YAMNet, SileroVAD
- **ocr** (2 files): TrOCR-small, Donut

## Unit Tests (Safe to Run in Parallel)

Unit tests don't load large models and are safe to run in parallel:

```bash
# Run unit tests with parallelization
python -m pytest tests/unit/ -n auto -v

# Run all quant tests
python -m pytest tests/quant/ -v
```

## Best Practices

1. **Close other applications** before running integration tests
2. **Use the batching script** for full test runs
3. **Monitor memory** during tests (watch the session output)
4. **Run individually** when debugging specific tests
5. **Use markers** to skip slow/heavy tests:

```bash
# Skip slow tests
python -m pytest tests/integration/ -m "not slow" -v

# Skip tests requiring model downloads
python -m pytest -m "not requires_model" -v
```

## Automatic Cleanup

The test suite now includes automatic cleanup:

- **Per-function cleanup**: After every test function
  - Clear MLX cache
  - Run garbage collection

- **Per-module cleanup**: After every test file
  - Clear MLX cache (twice for thoroughness)
  - Run 3 rounds of garbage collection
  - Report memory delta

- **Session-level**: At start and end of test session
  - Report memory usage
  - Warn if available memory is low

## Configuration Changes

### pytest.ini

- Parallel execution disabled by default (prevents memory exhaustion)
- Shows 10 slowest tests for performance monitoring

### tests/conftest.py

- Added psutil-based memory monitoring
- Module-scoped cleanup fixture (auto-runs after each test file)
- Session hooks for startup/shutdown monitoring
- Removed 8GB memory limit (was too restrictive)

### pyproject.toml

- Added `psutil>=5.9.0` to dev dependencies

## Troubleshooting

**Q: I'm still getting memory errors**

- A: Use smaller batch sizes: `python scripts/run_integration_tests.py --batch-size 1`
- A: Close all other applications
- A: Run individual test files

**Q: How do I know if a test is causing memory leaks?**

- A: Check the per-module memory delta in test output
- A: Increases > 500MB after module cleanup may indicate leaks

**Q: Can I run tests faster?**

- A: Unit tests can use `-n auto` for parallelization
- A: Integration tests MUST run sequentially for safety

**Q: Tests are timing out**

- A: Some VLM tests can be slow on first run (model download)
- A: Use `--timeout=300` to increase timeout if needed

## Model Memory Requirements

All integration tests now document their memory requirements in fixture docstrings. Here's a summary:

### Heavy Memory Models (marked with `@pytest.mark.heavy_memory`)

| Model | Parameters | FP16 Size | Peak Memory | Headroom Required | Notes |
|-------|-----------|-----------|-------------|-------------------|-------|
| **TinyLLaVA** | 1.5B | ~3GB | ~4-5GB | 5GB | **Largest model** - use caution |
| SmolVLM-500M | 500M | ~1GB | ~1.5GB | 2GB | VLM with larger capacity |
| Moondream2 | ~500M | ~1GB | ~1.5GB | 2GB | VLM with region detection |
| Chatterbox | 500M | ~1GB | ~1.5GB | 2GB | TTS model |
| SmolLM2-360M | 360M | ~720MB | ~1.2GB | 1GB | Language model |

### Medium Memory Models

| Model | Parameters | FP16 Size | Peak Memory | Notes |
|-------|-----------|-----------|-------------|-------|
| SmolVLM-256M | 256M | ~512MB | ~1GB | VLM base model |
| nanoVLM | 222M | ~444MB | ~800MB | Minimal VLM |
| Donut | ~200M | ~400MB | ~800MB | Document understanding |
| Orpheus | 150M | ~300MB | ~600MB | TTS model |
| TrOCR (x2) | ~100M each | ~200MB each | ~400MB total | **Two fixtures** |

### Light Memory Models

| Model | Size | Peak Memory | Notes |
|-------|------|-------------|-------|
| Whisper-tiny | ~150MB | ~300MB | Audio transcription |
| SmolLM2-135M | ~270MB | ~500MB | Smallest language model |
| MiniLM | ~100MB | ~200MB | Text embeddings |
| all-MiniLM-L6-v2 | ~100MB | ~200MB | Sentence embeddings |
| YAMNet | ~100MB | ~200MB | Audio classification |
| SileroVAD (x2) | ~50MB each | ~100MB each | **Two fixtures** |

### Memory Guard Behavior

Tests with **heavy_memory** marker automatically check available memory before loading:

```python
# Automatically skips if insufficient memory
check = check_memory_availability(5.0)  # Require 5GB headroom
if not check["available"]:
    pytest.skip(f"Insufficient memory: {check['headroom_gb']:.1f}GB available")
```

### Running Heavy Tests Selectively

```bash
# Skip heavy memory tests (>500MB)
pytest tests/integration/ -m "not heavy_memory" -v

# Run ONLY heavy memory tests
pytest tests/integration/ -m "heavy_memory" -v

# Run specific heavy test individually
pytest tests/integration/test_tinyllava.py -v  # Safest for 3GB model
```

## Recent Fixes (January 2025)

### What Was Fixed

1. **All 16 integration test fixtures** now use proper teardown:
   - Changed from `return model` to `yield model` + explicit cleanup
   - Added `del model`, `mx.clear_cache()`, `gc.collect()` to all fixtures

2. **TrOCR test** fixed double-fixture issue:
   - Previously loaded 2 models simultaneously (~400MB total)
   - Now loads them separately with cleanup between

3. **Heavy models** (TinyLLaVA, SmolVLM-500M, Moondream2, Chatterbox) added:
   - Memory availability guards
   - Memory profiling with `memory_profiler()` context manager
   - Automatic test skipping if insufficient memory

4. **All tests** marked appropriately:
   - `@pytest.mark.heavy_memory` for models >500MB
   - Documented memory requirements in fixture docstrings

### Testing the Fixes

```bash
# Test the largest model first (most likely to cause issues)
pytest tests/integration/test_tinyllava.py -v

# Test a heavy model
pytest tests/integration/test_smolvlm_500m_instruct.py -v

# Run 3 tests sequentially (safe batch)
pytest tests/integration/test_smollm2_135m.py \
       tests/integration/test_whisper_tiny.py \
       tests/integration/test_minilm.py -v

# Monitor memory during full run (with batching script)
python scripts/run_integration_tests.py --batch-size 3
```

### Expected Behavior After Fixes

- ✅ Tests run sequentially without kernel panics
- ✅ Memory properly freed between test modules
- ✅ Heavy tests skip if insufficient memory (graceful degradation)
- ✅ Memory profiling shows actual usage for heavy models
- ⚠️ Running all 16 tests at once still not recommended (10-12GB cumulative)
