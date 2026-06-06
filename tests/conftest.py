"""Pytest configuration and fixtures for SMLX tests."""

import gc
import os

# Force a non-interactive matplotlib backend for the whole test session BEFORE
# pyplot is imported anywhere. Several gym visualization helpers call
# ``plt.show()``; on an interactive backend (e.g. macOS) that opens a GUI window
# and blocks the test runner until it is closed, which manifests as a hang/
# timeout in CI and headless runs. "Agg" makes ``plt.show()`` a safe no-op.
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except ImportError:
    pass

import mlx.core as mx
import psutil
import pytest


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def pytest_configure(config):
    """Configure pytest and MLX before running tests."""
    # Set SMLX_TESTING environment variable to disable compilation
    # This prevents issues with random state in compiled functions during testing
    os.environ["SMLX_TESTING"] = "1"

    # REMOVED: Memory limit was too restrictive for integration tests
    # Integration tests with multiple VLMs can easily exceed 8GB
    # Let MLX use system memory as needed (up to 36GB on M4)

    # NOTE: Delay MLX Metal initialization to avoid sentencepiece segfault
    # Some models need to load tokenizers before MLX is initialized
    # See: https://github.com/huggingface/transformers/issues/XXX
    # Tokenizers will be loaded first, then MLX will be initialized in fixtures


@pytest.fixture(scope="function", autouse=True)
def cleanup_mlx_memory():
    """Cleanup MLX memory before and after each test to prevent memory leaks."""
    # Clear before test
    if mx.metal.is_available():
        mx.clear_cache()
    gc.collect()

    yield

    # Clear after test (more aggressive cleanup)
    if mx.metal.is_available():
        mx.clear_cache()
    gc.collect()


@pytest.fixture(scope="module", autouse=True)
def cleanup_after_module():
    """
    Cleanup after each test module (file).

    CRITICAL for integration tests with module-scoped model fixtures.
    Ensures models are freed from memory between test files.
    """
    initial_memory = get_memory_usage()
    print(f"\n📊 Module starting - Memory: {initial_memory:.1f} MB")

    # Run all tests in module
    yield

    # Aggressive multi-stage cleanup after module completes
    # Stage 1: Clear MLX caches
    if mx.metal.is_available():
        mx.clear_cache()

    # Stage 2: Multiple garbage collection passes
    # (Required to break circular references in large models)
    for _ in range(3):
        gc.collect()

    # Report final memory
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    print(f"📊 Module complete - Memory: {final_memory:.1f} MB (Δ {memory_increase:+.1f} MB)")


def pytest_sessionstart(session):
    """Called at start of test session."""
    print("\n" + "="*70)
    print("🚀 SMLX Test Session Starting")
    print("="*70)

    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.1f} MB")

    # Check available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    total_gb = memory.total / (1024**3)
    print(f"Available system memory: {available_gb:.1f} GB / {total_gb:.1f} GB")

    if available_gb < 10:
        print("\n⚠️  WARNING: Less than 10GB available memory!")
        print("    Integration tests may cause memory pressure.")
        print("    Consider:")
        print("      - Closing other applications")
        print("      - Running tests in smaller groups:")
        print("        python -m pytest tests/integration/test_smollm2_generation.py -v")
        print("")


def pytest_sessionfinish(session, exitstatus):
    """Called at end of test session."""
    print("\n" + "="*70)
    print("🏁 SMLX Test Session Complete")
    print("="*70)

    final_memory = get_memory_usage()
    print(f"Final memory: {final_memory:.1f} MB")

    # Final cleanup
    if mx.metal.is_available():
        mx.clear_cache()

    for _ in range(3):
        gc.collect()
