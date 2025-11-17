"""
Tests for benchmark reporting utilities.

Tests the functions for formatting and displaying benchmark results.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from smlx.bench.report import (
    create_benchmark_table,
    generate_markdown_report,
    print_benchmark_stats,
    print_comparison,
    print_model_stats,
    print_operation_stats,
    save_benchmark_results,
)
from smlx.bench.stats import (
    BenchmarkStats,
    BenchmarkSuite,
    ComparisonStats,
    ModelBenchmarkStats,
    OperationBenchmarkStats,
)


@pytest.mark.unit
class TestPrintBenchmarkStats:
    """Test print_benchmark_stats function."""

    def test_print_basic_stats(self, capsys):
        """Test printing basic benchmark stats."""
        stats = BenchmarkStats(
            name="test_benchmark",
            duration_ms=123.45,
            iterations=10,
            peak_memory_gb=2.5,
        )

        print_benchmark_stats(stats)

        captured = capsys.readouterr()
        assert "test_benchmark" in captured.out
        assert "123.45" in captured.out
        assert "10" in captured.out
        assert "2.5" in captured.out or "2.50" in captured.out

    def test_print_with_metadata(self, capsys):
        """Test printing stats with metadata."""
        stats = BenchmarkStats(
            name="test",
            duration_ms=100.0,
            metadata={"foo": "bar", "num": 42},
        )

        print_benchmark_stats(stats)

        captured = capsys.readouterr()
        assert "Metadata" in captured.out
        assert "foo" in captured.out
        assert "bar" in captured.out

    def test_print_model_stats(self, capsys):
        """Test printing model-specific stats."""
        stats = ModelBenchmarkStats(
            name="llm_test",
            model_name="SmolLM2-135M",
            prompt_tokens=100,
            generation_tokens=50,
            prompt_time=0.5,
            generation_time=2.0,
            prompt_tps=200.0,
            generation_tps=25.0,
            peak_memory_gb=2.3,
            quantization="4bit",
        )

        print_benchmark_stats(stats)

        captured = capsys.readouterr()
        assert "SmolLM2-135M" in captured.out
        assert "4bit" in captured.out
        assert "100" in captured.out  # prompt tokens
        assert "50" in captured.out   # generation tokens

    def test_print_operation_stats(self, capsys):
        """Test printing operation-specific stats."""
        stats = OperationBenchmarkStats(
            name="matmul_test",
            operation="matmul",
            duration_ms=10.5,
            input_shapes=[(100, 100), (100, 100)],
            output_shape=(100, 100),
            dtype="float32",
            device="gpu",
        )

        print_benchmark_stats(stats)

        captured = capsys.readouterr()
        assert "matmul" in captured.out
        assert "gpu" in captured.out
        assert "float32" in captured.out


@pytest.mark.unit
class TestPrintModelStats:
    """Test print_model_stats function."""

    def test_print_model_details(self, capsys):
        """Test printing model details."""
        stats = ModelBenchmarkStats(
            name="test",
            model_name="SmolLM2-135M",
            prompt_tokens=100,
            generation_tokens=50,
            prompt_time=0.5,
            generation_time=2.0,
            prompt_tps=200.0,
            generation_tps=25.0,
            quantization="4bit",
            batch_size=2,
        )

        print_model_stats(stats)

        captured = capsys.readouterr()
        assert "SmolLM2-135M" in captured.out
        assert "4bit" in captured.out
        assert "Batch size: 2" in captured.out
        assert "Prompt: 100" in captured.out
        assert "Generation: 50" in captured.out


@pytest.mark.unit
class TestPrintOperationStats:
    """Test print_operation_stats function."""

    def test_print_operation_details(self, capsys):
        """Test printing operation details."""
        stats = OperationBenchmarkStats(
            name="test",
            operation="matmul",
            input_shapes=[(100, 100), (100, 100)],
            output_shape=(100, 100),
            dtype="float32",
            device="gpu",
        )

        print_operation_stats(stats)

        captured = capsys.readouterr()
        assert "matmul" in captured.out
        assert "gpu" in captured.out
        assert "float32" in captured.out
        assert "(100, 100)" in captured.out


@pytest.mark.unit
class TestPrintComparison:
    """Test print_comparison function."""

    def test_print_comparison_stats(self, capsys):
        """Test printing comparison statistics."""
        baseline = BenchmarkStats(
            name="baseline",
            duration_ms=100.0,
            peak_memory_gb=4.0,
        )
        comparison = BenchmarkStats(
            name="optimized",
            duration_ms=50.0,
            peak_memory_gb=2.0,
        )
        comp = ComparisonStats(baseline=baseline, comparison=comparison)

        print_comparison(comp)

        captured = capsys.readouterr()
        assert "baseline" in captured.out
        assert "optimized" in captured.out
        assert "Speedup" in captured.out
        assert "2.00" in captured.out  # 2x speedup
        assert "Memory reduction" in captured.out


@pytest.mark.unit
class TestCreateBenchmarkTable:
    """Test create_benchmark_table function."""

    def test_empty_list(self):
        """Test with empty benchmark list."""
        table = create_benchmark_table([])
        assert table == ""

    def test_basic_benchmarks(self):
        """Test table creation with basic benchmarks."""
        benchmarks = [
            BenchmarkStats(
                name="test1",
                duration_ms=100.0,
                iterations=10,
                peak_memory_gb=2.0,
            ),
            BenchmarkStats(
                name="test2",
                duration_ms=200.0,
                iterations=10,
                peak_memory_gb=3.0,
            ),
        ]

        table = create_benchmark_table(benchmarks)

        assert isinstance(table, str)
        assert "test1" in table
        assert "test2" in table
        assert "100.00" in table
        assert "200.00" in table

    def test_model_benchmarks(self):
        """Test table with model benchmarks."""
        benchmarks = [
            ModelBenchmarkStats(
                name="llm1",
                model_name="SmolLM2-135M",
                prompt_tps=200.0,
                generation_tps=25.0,
                duration_ms=100.0,
                peak_memory_gb=2.0,
            ),
        ]

        table = create_benchmark_table(benchmarks)

        assert "llm1" in table
        assert "Prompt TPS" in table or "200.00" in table
        assert "Gen TPS" in table or "25.00" in table

    def test_operation_benchmarks(self):
        """Test table with operation benchmarks."""
        benchmarks = [
            OperationBenchmarkStats(
                name="matmul",
                operation="matmul",
                duration_ms=10.0,
                peak_memory_gb=1.0,
            ),
        ]

        table = create_benchmark_table(benchmarks)

        assert "matmul" in table


@pytest.mark.unit
class TestSaveBenchmarkResults:
    """Test save_benchmark_results function."""

    def test_save_single_stats_json(self):
        """Test saving single stats to JSON."""
        with TemporaryDirectory() as tmpdir:
            stats = BenchmarkStats(
                name="test",
                duration_ms=100.0,
                peak_memory_gb=2.0,
            )

            filepath = Path(tmpdir) / "results.json"
            save_benchmark_results(stats, filepath)

            assert filepath.exists()

            # Load and verify
            import json
            with open(filepath) as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "test"

    def test_save_list_json(self):
        """Test saving list of stats to JSON."""
        with TemporaryDirectory() as tmpdir:
            benchmarks = [
                BenchmarkStats(name="test1", duration_ms=100.0),
                BenchmarkStats(name="test2", duration_ms=200.0),
            ]

            filepath = Path(tmpdir) / "results.json"
            save_benchmark_results(benchmarks, filepath)

            assert filepath.exists()

            import json
            with open(filepath) as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["name"] == "test1"
            assert data[1]["name"] == "test2"

    def test_save_suite_json(self):
        """Test saving suite to JSON."""
        with TemporaryDirectory() as tmpdir:
            suite = BenchmarkSuite(name="test_suite")
            suite.add(BenchmarkStats(name="test1", duration_ms=100.0))
            suite.add(BenchmarkStats(name="test2", duration_ms=200.0))

            filepath = Path(tmpdir) / "suite.json"
            save_benchmark_results(suite, filepath)

            assert filepath.exists()

            import json
            with open(filepath) as f:
                data = json.load(f)

            assert data["name"] == "test_suite"
            assert "benchmarks" in data
            assert len(data["benchmarks"]) == 2
            assert "summary" in data

    def test_auto_format_detection(self):
        """Test automatic format detection from extension."""
        with TemporaryDirectory() as tmpdir:
            stats = BenchmarkStats(name="test", duration_ms=100.0)

            # JSON
            json_path = Path(tmpdir) / "results.json"
            save_benchmark_results(stats, json_path, format="auto")
            assert json_path.exists()

            # CSV
            csv_path = Path(tmpdir) / "results.csv"
            save_benchmark_results(stats, csv_path, format="auto")
            assert csv_path.exists()

    def test_explicit_json_format(self):
        """Test explicit JSON format."""
        with TemporaryDirectory() as tmpdir:
            stats = BenchmarkStats(name="test", duration_ms=100.0)

            filepath = Path(tmpdir) / "output.txt"
            save_benchmark_results(stats, filepath, format="json")

            # Should save as JSON despite .txt extension
            assert filepath.exists()

            import json
            with open(filepath) as f:
                data = json.load(f)

            assert isinstance(data, list)


@pytest.mark.unit
class TestGenerateMarkdownReport:
    """Test generate_markdown_report function."""

    def test_basic_report(self):
        """Test generating basic markdown report."""
        benchmarks = [
            BenchmarkStats(
                name="test1",
                duration_ms=100.0,
                peak_memory_gb=2.0,
            ),
        ]

        report = generate_markdown_report(
            benchmarks,
            title="Test Report",
            include_system_info=False,
        )

        assert isinstance(report, str)
        assert "# Test Report" in report
        assert "test1" in report
        assert "100.00" in report

    def test_report_with_multiple_benchmarks(self):
        """Test report with multiple benchmarks."""
        benchmarks = [
            BenchmarkStats(name="test1", duration_ms=100.0, peak_memory_gb=2.0),
            BenchmarkStats(name="test2", duration_ms=200.0, peak_memory_gb=3.0),
            BenchmarkStats(name="test3", duration_ms=150.0, peak_memory_gb=2.5),
        ]

        report = generate_markdown_report(
            benchmarks,
            include_system_info=False,
        )

        assert "test1" in report
        assert "test2" in report
        assert "test3" in report

    def test_report_with_model_stats(self):
        """Test report with model benchmarks."""
        benchmarks = [
            ModelBenchmarkStats(
                name="llm_test",
                model_name="SmolLM2-135M",
                prompt_tokens=100,
                generation_tokens=50,
                prompt_tps=200.0,
                generation_tps=25.0,
                duration_ms=100.0,
                peak_memory_gb=2.0,
            ),
        ]

        report = generate_markdown_report(
            benchmarks,
            include_system_info=False,
        )

        assert "SmolLM2-135M" in report
        assert "tok/s" in report

    def test_report_with_suite(self):
        """Test report with benchmark suite."""
        suite = BenchmarkSuite(name="Performance Tests")
        suite.add(BenchmarkStats(name="test1", duration_ms=100.0))
        suite.add(BenchmarkStats(name="test2", duration_ms=200.0))

        report = generate_markdown_report(
            suite,
            include_system_info=False,
        )

        assert "Performance Tests" in report
        assert "test1" in report
        assert "test2" in report

    def test_markdown_table_format(self):
        """Test that report contains markdown table."""
        benchmarks = [
            BenchmarkStats(name="test", duration_ms=100.0, peak_memory_gb=2.0),
        ]

        report = generate_markdown_report(
            benchmarks,
            include_system_info=False,
        )

        # Check for markdown table syntax
        assert "|" in report
        assert "---" in report
        assert "Name" in report
        assert "Duration" in report
        assert "Memory" in report

    def test_detailed_results_section(self):
        """Test detailed results section."""
        benchmarks = [
            BenchmarkStats(name="test", duration_ms=100.0, peak_memory_gb=2.0),
        ]

        report = generate_markdown_report(
            benchmarks,
            include_system_info=False,
        )

        assert "## Detailed Results" in report
        assert "### 1. test" in report
        assert "**Duration**" in report
        assert "**Peak Memory**" in report

    @pytest.mark.gpu
    def test_report_with_system_info(self):
        """Test report with system information."""
        benchmarks = [
            BenchmarkStats(name="test", duration_ms=100.0),
        ]

        report = generate_markdown_report(
            benchmarks,
            include_system_info=True,
        )

        assert "## System Information" in report
        assert "Platform" in report or "**Platform**" in report


@pytest.mark.integration
class TestReportingIntegration:
    """Integration tests for reporting functionality."""

    def test_complete_workflow(self):
        """Test complete reporting workflow."""
        with TemporaryDirectory() as tmpdir:
            # Create benchmarks
            suite = BenchmarkSuite(name="Integration Test Suite")
            suite.add(BenchmarkStats(
                name="benchmark1",
                duration_ms=100.0,
                iterations=10,
                peak_memory_gb=2.0,
            ))
            suite.add(ModelBenchmarkStats(
                name="llm_benchmark",
                model_name="SmolLM2-135M",
                prompt_tokens=100,
                generation_tokens=50,
                prompt_tps=200.0,
                generation_tps=25.0,
                duration_ms=150.0,
                peak_memory_gb=2.5,
            ))

            # Save as JSON
            json_path = Path(tmpdir) / "results.json"
            save_benchmark_results(suite, json_path)
            assert json_path.exists()

            # Generate markdown report
            report = generate_markdown_report(
                suite,
                title="Integration Test Report",
                include_system_info=False,
            )

            # Save markdown report
            md_path = Path(tmpdir) / "report.md"
            md_path.write_text(report)
            assert md_path.exists()

            # Verify content
            assert "Integration Test Suite" in report
            assert "benchmark1" in report
            assert "llm_benchmark" in report

    def test_save_and_load_round_trip(self):
        """Test saving and loading benchmark results."""
        with TemporaryDirectory() as tmpdir:
            original_stats = [
                BenchmarkStats(
                    name="test1",
                    duration_ms=100.0,
                    iterations=10,
                    peak_memory_gb=2.0,
                ),
                BenchmarkStats(
                    name="test2",
                    duration_ms=200.0,
                    iterations=20,
                    peak_memory_gb=3.0,
                ),
            ]

            # Save
            filepath = Path(tmpdir) / "results.json"
            save_benchmark_results(original_stats, filepath)

            # Load
            import json
            with open(filepath) as f:
                loaded_data = json.load(f)

            # Verify
            assert len(loaded_data) == 2
            assert loaded_data[0]["name"] == "test1"
            assert loaded_data[0]["duration_ms"] == 100.0
            assert loaded_data[1]["name"] == "test2"
            assert loaded_data[1]["duration_ms"] == 200.0
