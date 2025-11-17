"""Tests for smlx.utils.profiling module."""

import mlx.core as mx
import pytest

from smlx.utils.profiling import (
    ProfileResult,
    get_system_info,
)


class TestProfileResult:
    """Test ProfileResult dataclass."""

    def test_profile_result_creation(self):
        """Test creating ProfileResult."""
        result = ProfileResult(
            operation="test_op",
            elapsed_time=0.1,
            memory_used=100.0,
            tokens_per_second=50.0,
        )

        assert result.operation == "test_op"
        assert result.elapsed_time == 0.1
        assert result.memory_used == 100.0
        assert result.tokens_per_second == 50.0

    def test_profile_result_optional_fields(self):
        """Test ProfileResult with optional fields."""
        result = ProfileResult(
            operation="test_op",
            elapsed_time=0.1,
        )

        assert result.memory_used is None
        assert result.tokens_per_second is None
        assert result.throughput is None

    def test_profile_result_str(self):
        """Test ProfileResult string representation."""
        result = ProfileResult(
            operation="Test Operation",
            elapsed_time=0.5,
            memory_used=256.0,
            tokens_per_second=100.0,
        )

        str_repr = str(result)

        assert "Test Operation" in str_repr
        assert "500" in str_repr or "500.00" in str_repr  # 0.5s = 500ms
        assert "256" in str_repr  # Memory
        assert "100" in str_repr  # TPS

    def test_profile_result_str_minimal(self):
        """Test ProfileResult string with minimal fields."""
        result = ProfileResult(
            operation="Minimal",
            elapsed_time=0.01,
        )

        str_repr = str(result)

        assert "Minimal" in str_repr
        assert "ms" in str_repr  # Time should be shown

    def test_profile_result_throughput(self):
        """Test ProfileResult with throughput."""
        result = ProfileResult(
            operation="Batch Processing",
            elapsed_time=1.0,
            throughput=1000.0,
        )

        str_repr = str(result)

        assert "1000" in str_repr  # Throughput


class TestGetSystemInfo:
    """Test get_system_info function."""

    def test_get_system_info_returns_dict(self):
        """Test that get_system_info returns a dictionary."""
        info = get_system_info()

        assert isinstance(info, dict)

    def test_get_system_info_required_fields(self):
        """Test that system info has required fields."""
        info = get_system_info()

        assert "platform" in info
        assert "processor" in info
        assert "python_version" in info
        assert "mlx_device" in info
        assert "metal_available" in info

    def test_get_system_info_platform(self):
        """Test platform information."""
        info = get_system_info()

        # Should return a valid platform
        assert isinstance(info["platform"], str)
        assert len(info["platform"]) > 0

    def test_get_system_info_python_version(self):
        """Test Python version information."""
        info = get_system_info()

        # Should have Python version
        assert isinstance(info["python_version"], str)
        # Format like "3.9.1" or "3.10.0"
        assert "." in info["python_version"]

    @pytest.mark.gpu
    def test_get_system_info_metal(self):
        """Test Metal availability information."""
        info = get_system_info()

        # Should report Metal status
        assert isinstance(info["metal_available"], bool)

    def test_get_system_info_device(self):
        """Test device information."""
        info = get_system_info()

        assert "mlx_device" in info
        # Should mention Metal or Apple Silicon
        assert "Metal" in info["mlx_device"] or "Apple" in info["mlx_device"]


class TestProfilingEdgeCases:
    """Test edge cases in profiling utilities."""

    def test_profile_result_zero_time(self):
        """Test ProfileResult with zero elapsed time."""
        result = ProfileResult(
            operation="Instant",
            elapsed_time=0.0,
        )

        assert result.elapsed_time == 0.0
        str_repr = str(result)
        assert "Instant" in str_repr

    def test_profile_result_large_values(self):
        """Test ProfileResult with large values."""
        result = ProfileResult(
            operation="Large Operation",
            elapsed_time=100.0,
            memory_used=10000.0,
            tokens_per_second=10000.0,
        )

        assert result.elapsed_time == 100.0
        assert result.memory_used == 10000.0

    def test_get_system_info_consistency(self):
        """Test that get_system_info returns consistent results."""
        info1 = get_system_info()
        info2 = get_system_info()

        # Platform and Python version should be consistent
        assert info1["platform"] == info2["platform"]
        assert info1["python_version"] == info2["python_version"]


class TestProfilingIntegration:
    """Test integration scenarios."""

    def test_profile_simple_operation(self):
        """Test profiling a simple operation."""
        # Create a simple MLX operation
        a = mx.random.normal((10, 10))
        b = mx.random.normal((10, 10))

        import time

        start = time.time()
        c = a @ b
        mx.eval(c)
        elapsed = time.time() - start

        # Create profile result
        result = ProfileResult(
            operation="Matrix Multiply",
            elapsed_time=elapsed,
        )

        assert result.elapsed_time > 0
        assert result.operation == "Matrix Multiply"

    def test_system_info_in_profiling(self):
        """Test using system info in profiling workflow."""
        info = get_system_info()

        # Check if Metal is available
        if info["metal_available"]:
            # Can run GPU operations
            arr = mx.ones((100, 100))
            mx.eval(arr)

        # Should not raise errors
        assert True
