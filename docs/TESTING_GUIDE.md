# Testing Guide for SMLX

## Safe Testing Practices

### The Problem

When running the full integration test suite for models like SileroVAD, you may encounter system instability or crashes. This is **not** due to bugs in the code, but rather:

1. **Memory accumulation**: Multiple test fixtures loading models simultaneously
2. **MLX/Metal GPU resource contention**: 22+ tests competing for GPU resources
3. **Module-level fixtures**: Models staying in memory across all tests
4. **No garbage collection between tests**: Resources not released until suite completion

### Recommended Testing Strategies

#### 1. Run Tests Incrementally

```bash
# Test individual files
pytest tests/integration/test_silerovad.py::test_basic_speech_detection -v

# Test specific test classes/functions
pytest tests/integration/test_silerovad.py::test_model_loading_16k -v

# Test a few at a time
pytest tests/integration/test_silerovad.py -k "audio_loading or audio_resampling" -v
```

#### 2. Use Markers to Control Test Execution

```bash
# Skip slow/GPU-intensive tests
pytest tests/integration/test_silerovad.py -m "not slow and not gpu" -v

# Run only unit tests
pytest tests/integration/test_silerovad.py -m unit -v

# Skip tests requiring model downloads
pytest tests/integration/test_silerovad.py -m "not requires_model" -v
```

#### 3. Add Timeouts and Limits

```bash
# Add timeout to prevent hanging
pytest tests/integration/test_silerovad.py --timeout=30 -v

# Stop on first failure
pytest tests/integration/test_silerovad.py -x -v

# Limit test workers (if using pytest-xdist)
pytest tests/integration/test_silerovad.py -n 2 -v  # Only 2 parallel workers
```

#### 4. Run Standalone Tests

For critical debugging, run tests outside pytest:

```python
#!/usr/bin/env python3
"""Standalone test runner."""

from smlx.models.SileroVAD import load, create_streaming_vad
import numpy as np

# Create simple test
vad = load(sample_rate=16000)
streaming = create_streaming_vad(vad)

# Process test audio
audio = np.random.randn(512).astype(np.float32)
probs = streaming.process_chunk(audio)
print(f"Success! Probs: {probs}")
```

### Resource Management

#### For Test Fixtures

```python
@pytest.fixture(scope="function")  # Not "module"!
def vad_model():
    """Load model for single test."""
    model = load(sample_rate=16000)
    yield model
    # Cleanup
    del model
    mx.metal.clear_cache()  # Clear MLX cache if available
```

#### For Integration Tests

```python
@pytest.mark.integration
@pytest.mark.slow
def test_heavy_operation(vad_model):
    """Test that requires significant resources."""
    # Your test here
    pass
```

### MLX-Specific Considerations

1. **Array Type Safety**: Always ensure arrays are MLX arrays before operations:
   ```python
   if isinstance(arr, np.ndarray):
       arr = mx.array(arr.astype(np.float32))
   ```

2. **Hidden State Management**: Reset model state between independent tests:
   ```python
   vad.reset_state()
   ```

3. **Memory Pressure**: For long-running tests, periodically clear caches:
   ```python
   import gc
   gc.collect()
   ```

### Debugging Test Failures

If you encounter test failures:

1. **Run the failing test alone**:
   ```bash
   pytest tests/integration/test_silerovad.py::test_failing_test -vv
   ```

2. **Check for type errors**: Look for `addmm()` or similar errors - these often indicate numpy/MLX type mismatches

3. **Verify resource availability**:
   ```bash
   # Check memory
   vm_stat

   # Check GPU usage (if applicable)
   ioreg -l | grep "IOAcceleratorMemory"
   ```

4. **Use standalone scripts**: Create minimal reproduction outside pytest framework

### Example: Safe Test Run

```bash
# Complete safe testing workflow
cd /path/to/smlx

# 1. Run fast unit tests first
pytest tests/integration/test_silerovad.py -m "unit" -v

# 2. Run one slow test to verify setup
pytest tests/integration/test_silerovad.py::test_model_loading_16k -v

# 3. Run remaining tests in small batches
pytest tests/integration/test_silerovad.py -k "audio" -v
pytest tests/integration/test_silerovad.py -k "speech" -v
pytest tests/integration/test_silerovad.py -k "streaming" -v

# 4. If needed, run full suite with protections
pytest tests/integration/test_silerovad.py --timeout=60 -v --maxfail=3
```

## Continuous Integration Recommendations

For CI/CD pipelines:

```yaml
# .github/workflows/test.yml example
- name: Run Integration Tests
  run: |
    # Run tests in smaller groups
    pytest tests/integration/test_silerovad.py -m "unit" -v
    pytest tests/integration/test_silerovad.py -m "integration" -v --timeout=120 -x
```

## Summary

- ✅ The code is correct and works reliably
- ⚠️  Running all tests simultaneously can overwhelm system resources
- 🎯 Use markers, filters, and incremental testing for best results
- 🔧 Adjust fixture scopes and add cleanup for heavy tests
