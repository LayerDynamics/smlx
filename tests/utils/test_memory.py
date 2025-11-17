"""Tests for smlx.utils.memory module."""

import mlx.core as mx
import pytest

from smlx.utils.memory import (
    check_memory_availability,
    clear_cache,
    estimate_model_memory,
    get_active_memory_gb,
    get_cache_memory_gb,
    get_device_info,
    get_peak_memory_gb,
    memory_profiler,
    reset_peak_memory,
)


class TestMemoryQueries:
    """Test memory query functions."""

    @pytest.mark.gpu
    def test_get_peak_memory_gb(self):
        """Test getting peak memory."""
        peak = get_peak_memory_gb()

        assert isinstance(peak, float)
        assert peak >= 0

    @pytest.mark.gpu
    def test_get_active_memory_gb(self):
        """Test getting active memory."""
        active = get_active_memory_gb()

        assert isinstance(active, float)
        assert active >= 0

    @pytest.mark.gpu
    def test_get_cache_memory_gb(self):
        """Test getting cache memory."""
        cache = get_cache_memory_gb()

        assert isinstance(cache, float)
        assert cache >= 0

    @pytest.mark.gpu
    def test_memory_changes_with_allocation(self):
        """Test that memory increases with allocation."""
        clear_cache()
        reset_peak_memory()

        initial_active = get_active_memory_gb()

        # Allocate some memory
        large_array = mx.random.normal((1000, 1000))
        mx.eval(large_array)

        final_active = get_active_memory_gb()

        # Memory should increase (or at least not decrease)
        assert final_active >= initial_active


class TestMemoryOperations:
    """Test memory management operations."""

    @pytest.mark.gpu
    def test_clear_cache(self):
        """Test clearing cache."""
        # Allocate and then clear
        arr = mx.random.normal((100, 100))
        mx.eval(arr)

        clear_cache()

        # Should not raise an error
        cache_after = get_cache_memory_gb()
        assert cache_after >= 0

    @pytest.mark.gpu
    def test_reset_peak_memory(self):
        """Test resetting peak memory."""
        # Allocate something
        arr = mx.random.normal((100, 100))
        mx.eval(arr)

        # Reset peak
        reset_peak_memory()
        peak_after_reset = get_peak_memory_gb()

        # Peak should be reset (close to 0 or current usage)
        assert peak_after_reset >= 0


class TestDeviceInfo:
    """Test device information."""

    @pytest.mark.gpu
    def test_get_device_info(self):
        """Test getting device information."""
        info = get_device_info()

        assert isinstance(info, dict)
        assert "max_recommended_working_set_size" in info
        assert "max_buffer_length" in info
        assert "max_recommended_working_set_size_gb" in info

        # Values should be non-negative
        assert info["max_recommended_working_set_size"] >= 0
        assert info["max_recommended_working_set_size_gb"] >= 0

    def test_get_device_info_no_metal(self):
        """Test device info when Metal is not available (simulated)."""
        # This test may not work on all systems
        # Just check that function returns dict with expected structure
        info = get_device_info()

        assert isinstance(info, dict)
        assert "max_recommended_working_set_size_gb" in info


class TestMemoryProfiler:
    """Test memory profiler context manager."""

    @pytest.mark.gpu
    def test_memory_profiler_basic(self):
        """Test basic memory profiling."""
        with memory_profiler() as mem:
            # Allocate some memory
            arr = mx.random.normal((500, 500))
            mx.eval(arr)

        # Should have memory stats
        assert hasattr(mem, "start_active_gb")
        assert hasattr(mem, "end_active_gb")
        assert hasattr(mem, "peak_gb")

        # Values should be non-negative
        assert mem.start_active_gb >= 0
        assert mem.end_active_gb >= 0
        assert mem.peak_gb >= 0

    @pytest.mark.gpu
    def test_memory_profiler_delta(self):
        """Test memory delta calculation."""
        with memory_profiler() as mem:
            # Allocate memory
            arr = mx.random.normal((1000, 1000))
            mx.eval(arr)

        # Delta should be positive (we allocated memory)
        assert mem.delta_active_gb >= 0
        assert mem.delta_gb >= 0

    @pytest.mark.gpu
    def test_memory_profiler_peak_delta(self):
        """Test peak memory delta."""
        with memory_profiler(reset_peak=True) as mem:
            # Allocate memory
            arr = mx.random.normal((1000, 1000))
            mx.eval(arr)

        # Peak delta should be non-negative
        assert mem.peak_delta_gb >= 0

    @pytest.mark.gpu
    def test_memory_profiler_no_clear_cache(self):
        """Test memory profiler without clearing cache."""
        with memory_profiler(clear_cache_before=False) as mem:
            arr = mx.ones((100, 100))
            mx.eval(arr)

        # Should still work
        assert mem.end_active_gb >= 0

    @pytest.mark.gpu
    def test_memory_profiler_multiple_allocations(self):
        """Test memory profiler with multiple allocations."""
        with memory_profiler() as mem:
            # Multiple allocations
            for _ in range(5):
                arr = mx.random.normal((200, 200))
                mx.eval(arr)

        # Should track total memory change
        assert mem.peak_gb >= mem.start_peak_gb


class TestEstimateModelMemory:
    """Test model memory estimation."""

    def test_estimate_model_memory_float16(self):
        """Test estimating memory for float16 model."""
        mem = estimate_model_memory(135_000_000, dtype=mx.float16)

        assert mem["parameters"] == 135_000_000
        assert mem["bytes_per_param"] == 2  # float16 is 2 bytes
        assert mem["total_bytes"] == 135_000_000 * 2
        assert mem["total_mb"] > 0
        assert mem["total_gb"] > 0

    def test_estimate_model_memory_float32(self):
        """Test estimating memory for float32 model."""
        mem = estimate_model_memory(100_000_000, dtype=mx.float32)

        assert mem["bytes_per_param"] == 4  # float32 is 4 bytes
        assert mem["total_bytes"] == 100_000_000 * 4

    def test_estimate_model_memory_int8(self):
        """Test estimating memory for quantized (int8) model."""
        mem = estimate_model_memory(135_000_000, dtype=mx.int8)

        assert mem["bytes_per_param"] == 1  # int8 is 1 byte
        assert mem["total_bytes"] == 135_000_000

    def test_estimate_model_memory_sizes(self):
        """Test that size conversions are correct."""
        mem = estimate_model_memory(1_000_000, dtype=mx.float32)

        # 1M params * 4 bytes = 4MB
        assert mem["total_mb"] == 4.0
        assert mem["total_gb"] == 0.004

    def test_estimate_model_memory_small(self):
        """Test estimating memory for small model."""
        mem = estimate_model_memory(1_000, dtype=mx.float16)

        assert mem["total_mb"] < 1.0


class TestCheckMemoryAvailability:
    """Test memory availability checking."""

    @pytest.mark.gpu
    def test_check_memory_availability_small(self):
        """Test checking availability for small requirement."""
        check = check_memory_availability(0.1)  # 100 MB

        assert isinstance(check, dict)
        assert "available" in check
        assert "required_gb" in check
        assert "max_available_gb" in check
        assert "current_active_gb" in check
        assert "headroom_gb" in check

        # Should be available
        assert check["available"] is True
        assert check["required_gb"] == 0.1

    @pytest.mark.gpu
    def test_check_memory_availability_large(self):
        """Test checking availability for very large requirement."""
        check = check_memory_availability(1000.0)  # 1TB

        # Likely not available
        assert "available" in check
        assert isinstance(check["available"], bool)

    @pytest.mark.gpu
    def test_check_memory_availability_fields(self):
        """Test that all expected fields are present."""
        check = check_memory_availability(1.0)

        assert check["max_available_gb"] >= 0
        assert check["current_active_gb"] >= 0
        assert isinstance(check["headroom_gb"], float)


class TestMemoryEdgeCases:
    """Test edge cases and error handling."""

    def test_estimate_model_memory_zero_params(self):
        """Test estimating memory with zero parameters."""
        mem = estimate_model_memory(0, dtype=mx.float16)

        assert mem["total_bytes"] == 0
        assert mem["total_mb"] == 0.0
        assert mem["total_gb"] == 0.0

    def test_estimate_model_memory_unsupported_dtype(self):
        """Test with unsupported dtype (should use default)."""
        # Using a dtype not in the map should default to 4 bytes
        mem = estimate_model_memory(1000, dtype=mx.complex64)

        # Should use default of 4 bytes
        assert mem["bytes_per_param"] == 4

    @pytest.mark.gpu
    def test_memory_profiler_with_exception(self):
        """Test that memory profiler handles exceptions."""
        try:
            with memory_profiler() as mem:
                arr = mx.ones((10, 10))
                mx.eval(arr)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Memory stats should still be recorded
        assert mem.end_active_gb >= 0

    def test_check_memory_availability_negative(self):
        """Test checking availability with negative requirement."""
        check = check_memory_availability(-1.0)

        # Should handle gracefully
        assert isinstance(check, dict)


class TestMemoryIntegration:
    """Test integration of memory utilities."""

    @pytest.mark.gpu
    def test_memory_workflow(self):
        """Test typical memory profiling workflow."""
        # Clear cache and reset peak
        clear_cache()
        reset_peak_memory()

        # Check availability
        check = check_memory_availability(0.5)
        initial_available = check["available"]

        # Profile an operation
        with memory_profiler() as mem:
            # Allocate 100x100 array
            arr = mx.random.normal((100, 100))
            mx.eval(arr)

        # Verify memory was tracked
        assert mem.peak_gb >= 0
        assert mem.delta_active_gb >= 0

        # Clear cache again
        clear_cache()

    @pytest.mark.gpu
    def test_multiple_profilers_nested(self):
        """Test nested memory profilers."""
        with memory_profiler() as outer_mem:
            arr1 = mx.ones((100, 100))
            mx.eval(arr1)

            with memory_profiler() as inner_mem:
                arr2 = mx.ones((50, 50))
                mx.eval(arr2)

            # Inner profiler should have stats
            assert inner_mem.end_active_gb >= 0

        # Outer profiler should have stats
        assert outer_mem.end_active_gb >= 0
