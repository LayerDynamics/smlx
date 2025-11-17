"""Tests for smlx.utils.stats module."""

import pytest

from smlx.utils.stats import (
    format_duration,
    format_memory,
    format_number,
    mean,
    median,
    min_max,
    percentile,
    std,
    summary_stats,
    tokens_per_second,
)


class TestBasicStatistics:
    """Test basic statistical functions."""

    def test_mean_basic(self):
        """Test mean calculation."""
        assert mean([1, 2, 3, 4, 5]) == 3.0
        assert mean([10, 20, 30]) == 20.0
        assert mean([5]) == 5.0

    def test_mean_empty(self):
        """Test mean with empty list."""
        assert mean([]) == 0.0

    def test_mean_floats(self):
        """Test mean with floats."""
        result = mean([1.5, 2.5, 3.5])
        assert abs(result - 2.5) < 1e-10

    def test_median_odd(self):
        """Test median with odd number of elements."""
        assert median([1, 2, 3, 4, 5]) == 3.0
        assert median([1, 3, 5]) == 3.0

    def test_median_even(self):
        """Test median with even number of elements."""
        assert median([1, 2, 3, 4]) == 2.5
        assert median([1, 2]) == 1.5

    def test_median_empty(self):
        """Test median with empty list."""
        assert median([]) == 0.0

    def test_median_unsorted(self):
        """Test median with unsorted data."""
        assert median([5, 1, 3, 2, 4]) == 3.0
        assert median([10, 1, 5, 2]) == 3.5

    def test_std_basic(self):
        """Test standard deviation."""
        # For [1, 2, 3, 4, 5], sample std H 1.58
        result = std([1, 2, 3, 4, 5])
        assert 1.5 < result < 1.7

    def test_std_population(self):
        """Test population standard deviation (ddof=0)."""
        result = std([1, 2, 3, 4, 5], ddof=0)
        assert 1.3 < result < 1.5

    def test_std_empty_or_single(self):
        """Test std with empty or single element."""
        assert std([]) == 0.0
        assert std([5]) == 0.0

    def test_std_zero_variance(self):
        """Test std with zero variance."""
        assert std([5, 5, 5, 5]) == 0.0


class TestPercentile:
    """Test percentile calculations."""

    def test_percentile_median(self):
        """Test that 50th percentile equals median."""
        data = [1, 2, 3, 4, 5]
        assert percentile(data, 50) == median(data)

    def test_percentile_boundaries(self):
        """Test percentile at boundaries."""
        data = [1, 2, 3, 4, 5]
        assert percentile(data, 0) == 1
        assert percentile(data, 100) == 5

    def test_percentile_95(self):
        """Test 95th percentile."""
        data = [1, 2, 3, 4, 5]
        p95 = percentile(data, 95)
        assert 4.5 <= p95 <= 5.0

    def test_percentile_interpolation(self):
        """Test linear interpolation in percentile."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p25 = percentile(data, 25)
        assert 2.5 <= p25 <= 3.5

    def test_percentile_empty(self):
        """Test percentile with empty list."""
        assert percentile([], 50) == 0.0

    def test_percentile_invalid_range(self):
        """Test percentile with invalid range."""
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            percentile([1, 2, 3], -10)
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            percentile([1, 2, 3], 150)


class TestMinMax:
    """Test min_max function."""

    def test_min_max_basic(self):
        """Test basic min_max."""
        min_val, max_val = min_max([1, 2, 3, 4, 5])
        assert min_val == 1
        assert max_val == 5

    def test_min_max_single(self):
        """Test min_max with single element."""
        min_val, max_val = min_max([5])
        assert min_val == 5
        assert max_val == 5

    def test_min_max_negative(self):
        """Test min_max with negative numbers."""
        min_val, max_val = min_max([-5, -1, 0, 3, 10])
        assert min_val == -5
        assert max_val == 10

    def test_min_max_empty(self):
        """Test min_max with empty list."""
        min_val, max_val = min_max([])
        assert min_val == 0.0
        assert max_val == 0.0


class TestSummaryStats:
    """Test summary statistics."""

    def test_summary_stats_basic(self):
        """Test summary statistics with normal data."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        stats = summary_stats(data)

        assert stats["count"] == 10
        assert stats["mean"] == 5.5
        assert stats["median"] == 5.5
        assert stats["min"] == 1
        assert stats["max"] == 10
        assert stats["std"] > 0
        assert 0 < stats["p25"] < stats["median"] < stats["p75"] < stats["p95"] < stats["p99"]

    def test_summary_stats_empty(self):
        """Test summary statistics with empty data."""
        stats = summary_stats([])

        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["std"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0

    def test_summary_stats_keys(self):
        """Test that summary_stats returns all expected keys."""
        stats = summary_stats([1, 2, 3])

        expected_keys = ["count", "mean", "median", "std", "min", "max", "p25", "p75", "p95", "p99"]
        for key in expected_keys:
            assert key in stats


class TestTokensPerSecond:
    """Test tokens_per_second calculation."""

    def test_tokens_per_second_basic(self):
        """Test basic tokens per second calculation."""
        tps = tokens_per_second(1000, 5.0)
        assert tps == 200.0

    def test_tokens_per_second_fast(self):
        """Test with fast generation."""
        tps = tokens_per_second(100, 0.1)
        assert tps == 1000.0

    def test_tokens_per_second_zero_time(self):
        """Test with zero time (should return 0)."""
        tps = tokens_per_second(100, 0.0)
        assert tps == 0.0

    def test_tokens_per_second_negative_time(self):
        """Test with negative time (should return 0)."""
        tps = tokens_per_second(100, -1.0)
        assert tps == 0.0


class TestFormatDuration:
    """Test duration formatting."""

    def test_format_duration_microseconds(self):
        """Test formatting microseconds."""
        assert "µs" in format_duration(0.0000001)
        assert "µs" in format_duration(0.0005)

    def test_format_duration_milliseconds(self):
        """Test formatting milliseconds."""
        result = format_duration(0.001)
        assert "ms" in result
        assert "1.00" in result

        result = format_duration(0.5)
        assert "ms" in result

    def test_format_duration_seconds(self):
        """Test formatting seconds."""
        result = format_duration(1.5)
        assert "s" in result
        assert "1.50" in result

        result = format_duration(45.0)
        assert "s" in result

    def test_format_duration_minutes(self):
        """Test formatting minutes."""
        result = format_duration(75)
        assert "1m 15s" in result

        result = format_duration(120)
        assert "2m 0s" in result

    def test_format_duration_hours(self):
        """Test formatting hours."""
        result = format_duration(3600)
        assert "1h 0m" in result

        result = format_duration(3661)
        assert "1h 1m" in result


class TestFormatMemory:
    """Test memory formatting."""

    def test_format_memory_bytes(self):
        """Test formatting bytes."""
        result = format_memory(512)
        assert "B" in result
        assert "512" in result

    def test_format_memory_kilobytes(self):
        """Test formatting kilobytes."""
        result = format_memory(1024)
        assert "KB" in result
        assert "1.00" in result

        result = format_memory(2048)
        assert "2.00 KB" in result

    def test_format_memory_megabytes(self):
        """Test formatting megabytes."""
        result = format_memory(1024 * 1024)
        assert "MB" in result
        assert "1.00" in result

        result = format_memory(5 * 1024 * 1024)
        assert "5.00 MB" in result

    def test_format_memory_gigabytes(self):
        """Test formatting gigabytes."""
        result = format_memory(1024 * 1024 * 1024)
        assert "GB" in result
        assert "1.00" in result

        result = format_memory(10 * 1024 * 1024 * 1024)
        assert "10.00 GB" in result


class TestFormatNumber:
    """Test number formatting."""

    def test_format_number_basic(self):
        """Test basic number formatting."""
        assert format_number(1234567.89) == "1,234,567.89"
        assert format_number(1000) == "1,000.00"

    def test_format_number_precision(self):
        """Test number formatting with custom precision."""
        assert format_number(1234.5678, precision=0) == "1,235"
        assert format_number(1234.5678, precision=3) == "1,234.568"

    def test_format_number_small(self):
        """Test formatting small numbers."""
        result = format_number(123.45, precision=2)
        assert "123.45" in result

    def test_format_number_large(self):
        """Test formatting large numbers."""
        result = format_number(1000000000.0, precision=0)
        assert "1,000,000,000" in result
