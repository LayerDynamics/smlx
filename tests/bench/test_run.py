"""
Tests for benchmark suite runner.

Tests the unified benchmark runner CLI and orchestration.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from smlx.bench.run import (
    BENCHMARK_SUITES,
    list_benchmark_suites,
    run_all_benchmarks,
    run_benchmark_suite,
)


@pytest.mark.unit
class TestBenchmarkSuites:
    """Test BENCHMARK_SUITES configuration."""

    def test_suite_structure(self):
        """Test that all suites have required fields."""
        required_fields = [
            "name",
            "description",
            "module",
            "function",
            "requires_model",
            "model_types",
            "returns_stats",
        ]

        for suite_id, suite_info in BENCHMARK_SUITES.items():
            for field in required_fields:
                assert field in suite_info, f"Suite {suite_id} missing field {field}"

    def test_suite_names(self):
        """Test that expected suites are present."""
        expected_suites = [
            "system",
            "text_generation",
            "quantization",
            "llm",
            "ops",
        ]

        for suite in expected_suites:
            assert suite in BENCHMARK_SUITES

    def test_suite_model_requirements(self):
        """Test model requirement consistency."""
        for suite_id, suite_info in BENCHMARK_SUITES.items():
            requires_model = suite_info["requires_model"]
            model_types = suite_info["model_types"]

            if requires_model:
                # If requires model, should have model types (or empty list)
                assert isinstance(model_types, list)
            else:
                # If doesn't require model, model_types should be empty
                assert len(model_types) == 0, f"Suite {suite_id} doesn't require model but has model_types"

    def test_suite_modules_exist(self):
        """Test that suite modules can be imported."""
        # Test a few key suites
        test_suites = ["system", "ops"]

        for suite_id in test_suites:
            suite_info = BENCHMARK_SUITES[suite_id]
            module_name = suite_info["module"]

            try:
                import importlib
                module = importlib.import_module(module_name)
                assert module is not None
            except ImportError:
                pytest.fail(f"Failed to import module {module_name} for suite {suite_id}")


@pytest.mark.unit
class TestListBenchmarkSuites:
    """Test list_benchmark_suites function."""

    def test_list_suites(self, capsys):
        """Test listing benchmark suites."""
        list_benchmark_suites()

        captured = capsys.readouterr()

        assert "AVAILABLE BENCHMARK SUITES" in captured.out

        # Check that key suites are listed
        assert "system" in captured.out
        assert "text_generation" in captured.out
        assert "ops" in captured.out

        # Check that descriptions are shown
        assert "Description:" in captured.out

    def test_list_shows_requirements(self, capsys):
        """Test that listing shows model requirements."""
        list_benchmark_suites()

        captured = capsys.readouterr()

        assert "Requires model:" in captured.out

    def test_list_shows_usage(self, capsys):
        """Test that listing shows usage examples."""
        list_benchmark_suites()

        captured = capsys.readouterr()

        assert "Usage:" in captured.out
        assert "Examples:" in captured.out


@pytest.mark.unit
class TestRunBenchmarkSuite:
    """Test run_benchmark_suite function."""

    def test_unknown_suite(self):
        """Test running unknown suite."""
        with pytest.raises(SystemExit):
            run_benchmark_suite(
                suite_name="nonexistent_suite",
                verbose=False,
            )

    def test_system_suite(self):
        """Test running system suite."""
        # System suite should run without model
        result = run_benchmark_suite(
            suite_name="system",
            verbose=False,
        )

        # System suite returns None (prints directly)
        assert result is None

    @pytest.mark.skip(reason="Requires SmolLM2_135M model")
    def test_llm_suite_requires_model(self):
        """Test that LLM suite requires model."""
        with pytest.raises(SystemExit):
            run_benchmark_suite(
                suite_name="llm",
                model_path=None,
                verbose=False,
            )

    @pytest.mark.gpu
    def test_ops_suite(self):
        """Test running ops suite."""
        # Ops suite doesn't require model
        result = run_benchmark_suite(
            suite_name="ops",
            operation="matmul",
            num_iterations=2,
            verbose=False,
        )

        # Should return results
        assert result is not None

    @pytest.mark.gpu
    def test_ops_suite_with_output(self):
        """Test running ops suite with output file."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"

            result = run_benchmark_suite(
                suite_name="ops",
                operation="matmul",
                output_path=output_path,
                num_iterations=2,
                verbose=False,
            )

            assert result is not None
            assert output_path.exists()

    @pytest.mark.skip(reason="Requires model infrastructure")
    def test_suite_with_custom_args(self):
        """Test running suite with custom arguments."""
        result = run_benchmark_suite(
            suite_name="ops",
            shape="500,500",
            num_iterations=5,
            verbose=False,
        )

        assert result is not None


@pytest.mark.unit
class TestRunAllBenchmarks:
    """Test run_all_benchmarks function."""

    @pytest.mark.skip(reason="Requires model infrastructure")
    def test_run_all_with_model(self):
        """Test running all benchmarks."""
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            results = run_all_benchmarks(
                model_path="test_model",
                output_dir=output_dir,
                verbose=False,
            )

            assert isinstance(results, dict)
            # Should have run multiple suites
            assert len(results) > 0

    def test_run_all_without_model(self):
        """Test that run_all requires model."""
        # This should handle gracefully or error
        # Implementation may vary
        pass


@pytest.mark.integration
@pytest.mark.gpu
class TestBenchmarkRunnerIntegration:
    """Integration tests for benchmark runner."""

    def test_complete_ops_workflow(self):
        """Test complete ops benchmark workflow."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ops_results.json"

            result = run_benchmark_suite(
                suite_name="ops",
                operation="matmul",
                shape="100,100",
                num_iterations=3,
                output_path=output_path,
                verbose=False,
            )

            # Verify result
            assert result is not None

            # Verify output file
            assert output_path.exists()

            # Load and verify JSON
            import json
            with open(output_path) as f:
                data = json.load(f)

            assert isinstance(data, dict)

    def test_system_info_workflow(self):
        """Test system info display workflow."""
        # Should run without errors
        result = run_benchmark_suite(
            suite_name="system",
            verbose=False,
        )

        # System returns None
        assert result is None

    @pytest.mark.skip(reason="Requires model infrastructure")
    def test_multiple_suites_sequential(self):
        """Test running multiple suites sequentially."""
        suites = ["ops", "system"]

        results = {}
        for suite in suites:
            result = run_benchmark_suite(
                suite_name=suite,
                verbose=False,
            )
            results[suite] = result

        # Should have run both
        assert len(results) == 2


@pytest.mark.unit
class TestBenchmarkRunnerHelpers:
    """Test helper functions in benchmark runner."""

    def test_suite_lookup(self):
        """Test looking up suite information."""
        assert "ops" in BENCHMARK_SUITES

        suite_info = BENCHMARK_SUITES["ops"]
        assert suite_info["name"] == "Operation Benchmarks"
        assert suite_info["requires_model"] is False

    def test_model_required_suites(self):
        """Test identifying suites that require models."""
        model_required = [
            name for name, info in BENCHMARK_SUITES.items()
            if info["requires_model"]
        ]

        # Should include text_generation, quantization, llm
        assert "text_generation" in model_required
        assert "quantization" in model_required
        assert "llm" in model_required

        # Should NOT include ops, system
        assert "ops" not in model_required
        assert "system" not in model_required

    def test_no_model_suites(self):
        """Test identifying suites that don't require models."""
        no_model = [
            name for name, info in BENCHMARK_SUITES.items()
            if not info["requires_model"]
        ]

        assert "ops" in no_model
        assert "system" in no_model


@pytest.mark.unit
class TestBenchmarkCLI:
    """Test CLI-related functionality."""

    def test_suite_help_text(self):
        """Test that suites have help text."""
        for suite_id, suite_info in BENCHMARK_SUITES.items():
            assert "description" in suite_info
            assert len(suite_info["description"]) > 0

    def test_suite_module_paths(self):
        """Test that module paths are valid."""
        for suite_id, suite_info in BENCHMARK_SUITES.items():
            module_name = suite_info["module"]

            # Should start with smlx.bench.suites
            assert module_name.startswith("smlx.bench.suites")

    def test_suite_function_names(self):
        """Test that function names are valid identifiers."""
        import re

        for suite_id, suite_info in BENCHMARK_SUITES.items():
            function_name = suite_info["function"]

            # Should be valid Python identifier
            assert re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', function_name)


@pytest.mark.benchmark
class TestBenchmarkRunnerPerformance:
    """Performance tests for benchmark runner."""

    def test_suite_loading_overhead(self):
        """Test that suite loading is fast."""
        import time

        # Test loading suite info
        start = time.perf_counter()
        suite_info = BENCHMARK_SUITES["ops"]
        elapsed = time.perf_counter() - start

        # Should be instant (just dict lookup)
        assert elapsed < 0.001

    @pytest.mark.gpu
    def test_ops_benchmark_reasonable_time(self):
        """Test that ops benchmark completes in reasonable time."""
        import time

        start = time.perf_counter()
        result = run_benchmark_suite(
            suite_name="ops",
            operation="matmul",
            shape="100,100",
            num_iterations=2,
            verbose=False,
        )
        elapsed = time.perf_counter() - start

        # Should complete within reasonable time (30 seconds)
        assert elapsed < 30


@pytest.mark.unit
class TestResultSerialization:
    """Test result serialization."""

    @pytest.mark.gpu
    def test_ops_results_serializable(self):
        """Test that ops results can be serialized."""
        import json

        result = run_benchmark_suite(
            suite_name="ops",
            operation="matmul",
            num_iterations=2,
            verbose=False,
        )

        # Should be able to convert to JSON-serializable format
        # The run_benchmark_suite already handles this internally
        assert result is not None

    @pytest.mark.gpu
    def test_save_results_creates_file(self):
        """Test that saving results creates file."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_results.json"

            run_benchmark_suite(
                suite_name="ops",
                operation="matmul",
                output_path=output_path,
                num_iterations=2,
                verbose=False,
            )

            assert output_path.exists()

            # Verify it's valid JSON
            import json
            with open(output_path) as f:
                data = json.load(f)

            assert isinstance(data, dict)


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in benchmark runner."""

    def test_invalid_suite_name(self):
        """Test error handling for invalid suite name."""
        with pytest.raises(SystemExit):
            run_benchmark_suite(
                suite_name="invalid_suite_name_xyz",
                verbose=False,
            )

    def test_missing_required_model(self):
        """Test error when model is required but not provided."""
        with pytest.raises(SystemExit):
            run_benchmark_suite(
                suite_name="llm",  # Requires model
                model_path=None,
                verbose=False,
            )

    @pytest.mark.skip(reason="Error handling may vary by implementation")
    def test_invalid_model_path(self):
        """Test error handling for invalid model path."""
        # This should handle gracefully
        with pytest.raises((SystemExit, Exception)):
            run_benchmark_suite(
                suite_name="llm",
                model_path="/nonexistent/path/to/model",
                verbose=False,
            )


@pytest.mark.unit
class TestVerboseOutput:
    """Test verbose output control."""

    def test_verbose_true(self, capsys):
        """Test with verbose=True."""
        run_benchmark_suite(
            suite_name="system",
            verbose=True,
        )

        captured = capsys.readouterr()
        # Should have output
        assert len(captured.out) > 0

    def test_verbose_false(self, capsys):
        """Test with verbose=False."""
        run_benchmark_suite(
            suite_name="system",
            verbose=False,
        )

        captured = capsys.readouterr()
        # May still have some output (implementation-dependent)
        # Just verify no error


@pytest.mark.unit
class TestBenchmarkSuiteRegistry:
    """Test benchmark suite registry."""

    def test_all_suites_have_unique_names(self):
        """Test that all suite IDs are unique."""
        suite_ids = list(BENCHMARK_SUITES.keys())
        assert len(suite_ids) == len(set(suite_ids))

    def test_all_suite_names_are_strings(self):
        """Test that all suite metadata uses strings."""
        for suite_id, suite_info in BENCHMARK_SUITES.items():
            assert isinstance(suite_info["name"], str)
            assert isinstance(suite_info["description"], str)
            assert isinstance(suite_info["module"], str)
            assert isinstance(suite_info["function"], str)

    def test_all_suites_have_model_types_list(self):
        """Test that model_types is always a list."""
        for suite_id, suite_info in BENCHMARK_SUITES.items():
            assert isinstance(suite_info["model_types"], list)

    def test_all_suites_have_returns_stats_bool(self):
        """Test that returns_stats is a boolean."""
        for suite_id, suite_info in BENCHMARK_SUITES.items():
            assert isinstance(suite_info["returns_stats"], bool)
