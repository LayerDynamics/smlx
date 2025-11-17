"""
Tests for system information utilities.

Tests functions for detecting system capabilities, especially for
Apple Silicon chipsets.
"""

import platform
import pytest

from smlx.bench.suites.system import (
    get_chip_name,
    get_cpu_info,
    get_memory_info,
    get_system_info,
    is_apple_silicon,
    is_m4_chip,
    print_system_info,
)


@pytest.mark.unit
class TestGetSystemInfo:
    """Test get_system_info function."""

    def test_basic_info(self):
        """Test that basic system info is returned."""
        info = get_system_info()

        assert isinstance(info, dict)
        assert "platform" in info
        assert "platform_version" in info
        assert "processor" in info
        assert "python_version" in info
        assert "mlx_version" in info
        assert "mlx_available" in info

    def test_platform_info(self):
        """Test platform information."""
        info = get_system_info()

        # Platform should be a string
        assert isinstance(info["platform"], str)
        assert isinstance(info["platform_version"], str)
        assert isinstance(info["processor"], str)

    def test_mlx_info(self):
        """Test MLX information."""
        info = get_system_info()

        assert isinstance(info["mlx_available"], bool)
        assert isinstance(info["mlx_version"], str)

    @pytest.mark.gpu
    def test_memory_info_when_mlx_available(self):
        """Test memory info when MLX is available."""
        info = get_system_info()

        if info["mlx_available"]:
            assert "max_memory_gb" in info
            assert "max_buffer_length" in info
            assert info["max_memory_gb"] > 0

    def test_chip_info_on_mac(self):
        """Test chip info on macOS."""
        info = get_system_info()

        if platform.system() == "Darwin":
            # On macOS, should have chip info
            if platform.processor() == "arm":
                assert "chip" in info


@pytest.mark.unit
class TestGetChipName:
    """Test get_chip_name function."""

    def test_returns_string_or_none(self):
        """Test that function returns string or None."""
        chip = get_chip_name()

        assert chip is None or isinstance(chip, str)

    @pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="Chip detection only works on macOS"
    )
    def test_mac_chip_detection(self):
        """Test chip detection on macOS."""
        chip = get_chip_name()

        # On macOS, should return something
        assert chip is not None

    @pytest.mark.skipif(
        platform.system() != "Darwin" or platform.processor() != "arm",
        reason="Only applicable on Apple Silicon"
    )
    def test_apple_silicon_chip_name(self):
        """Test that Apple Silicon chip name contains expected keywords."""
        chip = get_chip_name()

        if chip:
            # Should contain Apple or M-series
            assert "Apple" in chip or "M" in chip


@pytest.mark.unit
class TestIsM4Chip:
    """Test is_m4_chip function."""

    def test_returns_bool(self):
        """Test that function returns boolean."""
        result = is_m4_chip()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        platform.system() != "Darwin" or platform.processor() != "arm",
        reason="Only applicable on Apple Silicon"
    )
    def test_m4_detection(self):
        """Test M4 chip detection."""
        result = is_m4_chip()

        # Result should be consistent with chip name
        chip = get_chip_name()
        if chip and "M4" in chip:
            assert result is True


@pytest.mark.unit
class TestIsAppleSilicon:
    """Test is_apple_silicon function."""

    def test_returns_bool(self):
        """Test that function returns boolean."""
        result = is_apple_silicon()
        assert isinstance(result, bool)

    def test_darwin_arm_detection(self):
        """Test Apple Silicon detection."""
        result = is_apple_silicon()

        # Should return True on Darwin + ARM
        if platform.system() == "Darwin" and platform.processor() == "arm":
            assert result is True
        else:
            assert result is False


@pytest.mark.unit
class TestGetMemoryInfo:
    """Test get_memory_info function."""

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        info = get_memory_info()

        assert isinstance(info, dict)
        assert "max_recommended_gb" in info
        assert "active_gb" in info
        assert "cache_gb" in info
        assert "peak_gb" in info

    def test_values_are_floats(self):
        """Test that memory values are floats."""
        info = get_memory_info()

        assert isinstance(info["max_recommended_gb"], float)
        assert isinstance(info["active_gb"], float)
        assert isinstance(info["cache_gb"], float)
        assert isinstance(info["peak_gb"], float)

    def test_values_are_non_negative(self):
        """Test that memory values are non-negative."""
        info = get_memory_info()

        assert info["max_recommended_gb"] >= 0
        assert info["active_gb"] >= 0
        assert info["cache_gb"] >= 0
        assert info["peak_gb"] >= 0

    @pytest.mark.gpu
    def test_memory_values_when_mlx_available(self):
        """Test memory values when MLX Metal is available."""
        info = get_memory_info()

        # When MLX is available, should have positive values
        import mlx.core as mx
        if mx.metal.is_available():
            assert info["max_recommended_gb"] > 0


@pytest.mark.unit
class TestGetCPUInfo:
    """Test get_cpu_info function."""

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        info = get_cpu_info()

        assert isinstance(info, dict)
        assert "processor" in info

    def test_processor_is_string(self):
        """Test that processor is a string."""
        info = get_cpu_info()
        assert isinstance(info["processor"], str)

    @pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="Core count detection only works on macOS"
    )
    def test_core_count_on_mac(self):
        """Test core count detection on macOS."""
        info = get_cpu_info()

        # On macOS, should have core counts
        if "physical_cores" in info:
            assert isinstance(info["physical_cores"], int)
            assert info["physical_cores"] > 0

        if "logical_cores" in info:
            assert isinstance(info["logical_cores"], int)
            assert info["logical_cores"] > 0
            # Logical cores should be >= physical cores
            if "physical_cores" in info:
                assert info["logical_cores"] >= info["physical_cores"]


@pytest.mark.unit
class TestPrintSystemInfo:
    """Test print_system_info function."""

    def test_prints_without_error(self, capsys):
        """Test that function prints without error."""
        print_system_info()

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_output_contains_expected_info(self, capsys):
        """Test that output contains expected information."""
        print_system_info()

        captured = capsys.readouterr()

        assert "System Information" in captured.out
        assert "Platform:" in captured.out
        assert "Python:" in captured.out
        assert "MLX:" in captured.out

    @pytest.mark.gpu
    def test_output_contains_memory_info_when_mlx_available(self, capsys):
        """Test that memory info is shown when MLX is available."""
        import mlx.core as mx

        if mx.metal.is_available():
            print_system_info()

            captured = capsys.readouterr()

            assert "Memory Information" in captured.out
            assert "Max Recommended:" in captured.out

    @pytest.mark.skipif(
        platform.system() != "Darwin" or platform.processor() != "arm",
        reason="Only applicable on Apple Silicon"
    )
    def test_output_contains_chip_info_on_apple_silicon(self, capsys):
        """Test that chip info is shown on Apple Silicon."""
        print_system_info()

        captured = capsys.readouterr()

        assert "Chip:" in captured.out


@pytest.mark.integration
class TestSystemInfoIntegration:
    """Integration tests for system info functions."""

    def test_complete_system_detection(self):
        """Test complete system detection workflow."""
        # Get all system info
        system_info = get_system_info()
        cpu_info = get_cpu_info()
        memory_info = get_memory_info()

        # All should return valid data
        assert system_info["platform"]
        assert system_info["python_version"]
        assert cpu_info["processor"]

        # Verify consistency
        assert system_info["processor"] == cpu_info["processor"]

    @pytest.mark.gpu
    def test_mlx_detection_consistency(self):
        """Test MLX detection is consistent."""
        import mlx.core as mx

        system_info = get_system_info()
        mlx_available = mx.metal.is_available()

        assert system_info["mlx_available"] == mlx_available

        if mlx_available:
            memory_info = get_memory_info()
            assert memory_info["max_recommended_gb"] > 0

    def test_apple_silicon_detection_consistency(self):
        """Test Apple Silicon detection is consistent."""
        is_as = is_apple_silicon()
        chip = get_chip_name()

        if is_as:
            # If we detect Apple Silicon, platform should be Darwin + ARM
            assert platform.system() == "Darwin"
            assert platform.processor() == "arm"

            # Should have chip name
            if chip:
                assert "Apple" in chip or "M" in chip

    def test_m4_detection_consistency(self):
        """Test M4 detection is consistent."""
        is_m4 = is_m4_chip()
        chip = get_chip_name()

        if is_m4:
            # If M4 detected, should be Apple Silicon
            assert is_apple_silicon()

            # Chip name should contain M4
            if chip:
                assert "M4" in chip

    @pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="Only applicable on macOS"
    )
    def test_mac_specific_features(self):
        """Test macOS-specific features."""
        system_info = get_system_info()
        cpu_info = get_cpu_info()

        # On macOS, should detect platform correctly
        assert system_info["platform"] == "Darwin"

        # Should have chip info on ARM
        if platform.processor() == "arm":
            assert "chip" in system_info


@pytest.mark.benchmark
class TestSystemInfoPerformance:
    """Performance tests for system info functions."""

    def test_system_info_is_fast(self):
        """Test that system info retrieval is fast."""
        import time

        start = time.perf_counter()
        _ = get_system_info()
        elapsed = time.perf_counter() - start

        # Should complete in less than 100ms
        assert elapsed < 0.1

    def test_multiple_calls_are_consistent(self):
        """Test that multiple calls return consistent results."""
        info1 = get_system_info()
        info2 = get_system_info()

        # Core attributes should be the same
        assert info1["platform"] == info2["platform"]
        assert info1["processor"] == info2["processor"]
        assert info1["mlx_available"] == info2["mlx_available"]

    def test_memory_info_updates(self):
        """Test that memory info reflects current state."""
        import mlx.core as mx

        if not mx.metal.is_available():
            pytest.skip("MLX Metal not available")

        # Get initial memory
        mem1 = get_memory_info()

        # Allocate some memory
        x = mx.random.normal((1000, 1000))
        mx.eval(x)

        # Get updated memory
        mem2 = get_memory_info()

        # Active or peak memory should have increased
        # (or at least not decreased)
        assert mem2["active_gb"] >= mem1["active_gb"] or mem2["peak_gb"] >= mem1["peak_gb"]
