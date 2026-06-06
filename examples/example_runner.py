#!/usr/bin/env python3
# Copyright � 2025 SMLX Project

"""
Comprehensive Example Runner for SMLX

Discovers, filters, executes, and reports on all examples in the project.
Provides robust error handling, timeout management, and resource tracking.

Usage:
    # List all examples
    python -m examples.example_runner --list

    # Run all examples
    python -m examples.example_runner

    # Run examples from a specific category
    python -m examples.example_runner --category model

    # Run a specific example
    python -m examples.example_runner --run smollm2_135m_example

    # Run with custom timeout and verbose output
    python -m examples.example_runner --timeout 600 --verbose

    # Export results to JSON
    python -m examples.example_runner --json-output results.json

    # Check dependencies only
    python -m examples.example_runner --check-deps
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Category-specific timeouts (seconds)
CATEGORY_TIMEOUTS = {
    "model": 300,  # Model examples may take time to load
    "quant": 600,  # Quantization can be slow
    "agent": 300,  # Agents may iterate multiple times
    "server": 60,  # Server examples should be quick
    "eval": 120,  # Evaluation utilities
    "performance": 180,  # Performance demos
}

DEFAULT_TIMEOUT = 300


@dataclass
class ExampleMetadata:
    """Metadata for a discovered example."""

    name: str
    path: Path
    category: str
    description: str = ""
    requires_model: Optional[str] = None
    requires_server: bool = False
    timeout: int = DEFAULT_TIMEOUT
    dependencies: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Set category-specific timeout if not already set."""
        if self.timeout == DEFAULT_TIMEOUT and self.category in CATEGORY_TIMEOUTS:
            self.timeout = CATEGORY_TIMEOUTS[self.category]


@dataclass
class ExampleResult:
    """Result of running a single example."""

    example_name: str
    status: str  # 'passed', 'failed', 'skipped', 'timeout'
    duration: float
    error: Optional[str] = None
    output: str = ""
    returncode: Optional[int] = None


@dataclass
class ExampleReport:
    """Aggregate report of all example runs."""

    total: int
    passed: int
    failed: int
    skipped: int
    timeout: int
    results: list[ExampleResult] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate (passed / (passed + failed))."""
        attempted = self.passed + self.failed + self.timeout
        return (self.passed / attempted * 100) if attempted > 0 else 0.0


class ExampleDiscovery:
    """Discovers and categorizes examples in the examples/ directory."""

    def __init__(self, examples_dir: Path):
        self.examples_dir = examples_dir

    def discover_all(self) -> list[ExampleMetadata]:
        """Discover all example files."""
        examples = []

        # Find all *_example.py files (excluding __init__.py)
        example_files = sorted(
            self.examples_dir.rglob("*_example.py"),
            key=lambda p: (p.parent.name, p.name),
        )

        for example_path in example_files:
            metadata = self._extract_metadata(example_path)
            if metadata:
                examples.append(metadata)

        return examples

    def _extract_metadata(self, path: Path) -> Optional[ExampleMetadata]:
        """Extract metadata from an example file."""
        # Determine category from directory structure
        relative_path = path.relative_to(self.examples_dir)
        parts = relative_path.parts

        if len(parts) < 2:
            category = "general"
        else:
            category = parts[0]  # First directory level (models, quant, agents, etc.)

        # Extract name from filename
        name = path.stem  # Remove .py extension

        # Try to extract docstring for description
        description = ""
        requires_model = None
        requires_server = False
        dependencies = []

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

                # Extract module docstring
                lines = content.split("\n")
                in_docstring = False
                docstring_lines = []

                for line in lines:
                    stripped = line.strip()
                    if '"""' in stripped or "'''" in stripped:
                        if not in_docstring:
                            in_docstring = True
                            # Check if docstring starts and ends on same line
                            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                                docstring_lines.append(
                                    stripped.replace('"""', "").replace("'''", "").strip()
                                )
                                break
                            continue
                        else:
                            in_docstring = False
                            break
                    elif in_docstring:
                        docstring_lines.append(stripped)

                description = " ".join(docstring_lines).strip()
                if len(description) > 200:
                    description = description[:197] + "..."

                # Check for server dependency
                if "server" in content.lower() and "requests" in content:
                    requires_server = True

                # Check for model requirements (heuristic)
                if "load(" in content or "load_model(" in content:
                    # Try to find model name in load calls
                    if 'load("' in content or "load('" in content:
                        # Simple heuristic: check for common model patterns
                        if "SmolLM2-135M" in content:
                            requires_model = "SmolLM2-135M"
                        elif "SmolLM2-360M" in content:
                            requires_model = "SmolLM2-360M"
                        elif "whisper-tiny" in content:
                            requires_model = "whisper-tiny"
                        elif "SmolVLM" in content:
                            requires_model = "SmolVLM"

                # Extract dependencies from imports
                for line in lines:
                    if line.strip().startswith(("import ", "from ")):
                        # Simplified dependency extraction
                        if "smlx.models" in line:
                            dependencies.append("models")
                        if "smlx.quant" in line:
                            dependencies.append("quantization")
                        if "smlx.agents" in line:
                            dependencies.append("agents")
                        if "requests" in line:
                            dependencies.append("requests")

        except Exception as e:
            # If we can't read the file, still create metadata with basic info
            description = f"Example file (error reading metadata: {e})"

        return ExampleMetadata(
            name=name,
            path=path,
            category=category,
            description=description,
            requires_model=requires_model,
            requires_server=requires_server,
            dependencies=list(set(dependencies)),  # Remove duplicates
        )


class ExampleRunner:
    """Executes examples with timeout and output capture."""

    def __init__(self, python_path: str = sys.executable, verbose: bool = False):
        self.python_path = python_path
        self.verbose = verbose

    def run_example(
        self, metadata: ExampleMetadata, timeout: Optional[int] = None
    ) -> ExampleResult:
        """Run a single example in a subprocess."""
        start_time = time.time()
        timeout_seconds = timeout or metadata.timeout

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Running: {metadata.name}")
            print(f"Category: {metadata.category}")
            print(f"Path: {metadata.path}")
            print(f"Timeout: {timeout_seconds}s")
            print(f"{'=' * 70}\n")

        try:
            # Run example in subprocess
            result = subprocess.run(
                [self.python_path, str(metadata.path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=metadata.path.parent,  # Run from example directory
            )

            duration = time.time() - start_time

            # Determine status
            if result.returncode == 0:
                status = "passed"
                error = None
            else:
                status = "failed"
                error = f"Exit code: {result.returncode}"
                if result.stderr:
                    error += f"\n{result.stderr}"

            output = result.stdout if result.stdout else ""
            if result.stderr and self.verbose:
                output += f"\n[STDERR]\n{result.stderr}"

            return ExampleResult(
                example_name=metadata.name,
                status=status,
                duration=duration,
                error=error,
                output=output,
                returncode=result.returncode,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return ExampleResult(
                example_name=metadata.name,
                status="timeout",
                duration=duration,
                error=f"Timeout after {timeout_seconds}s",
                output="",
                returncode=None,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ExampleResult(
                example_name=metadata.name,
                status="failed",
                duration=duration,
                error=f"Exception: {str(e)}\n{traceback.format_exc()}",
                output="",
                returncode=None,
            )

    def run_multiple(
        self,
        examples: list[ExampleMetadata],
        timeout: Optional[int] = None,
        stop_on_failure: bool = False,
    ) -> ExampleReport:
        """Run multiple examples and generate a report."""
        results = []
        total_duration = 0.0

        for i, metadata in enumerate(examples, 1):
            # Print header with example info
            print(f"\n{'=' * 80}")
            print(f"[{i}/{len(examples)}] {metadata.name}")
            print(f"{'=' * 80}")

            # Print what the example does
            if metadata.description:
                print(f"Description: {metadata.description}")
            print(f"Category: {metadata.category}")
            if metadata.requires_model:
                print(f"Model: {metadata.requires_model}")
            print("\nRunning...")
            print("-" * 80)

            result = self.run_example(metadata, timeout)
            results.append(result)
            total_duration += result.duration

            # Print result output
            print("-" * 80)
            if result.output and result.output.strip():
                # Show the actual output from the example
                print("Output:")
                print(result.output.strip())
                print("-" * 80)

            # Print status
            status_icon = {
                "passed": "",
                "failed": "",
                "timeout": "�",
                "skipped": "�",
            }.get(result.status, "?")

            print(f"\n{status_icon} {result.status.upper()} ({result.duration:.2f}s)")

            if result.status == "failed":
                if result.error:
                    print(f"\nError: {result.error}")

            if stop_on_failure and result.status in ("failed", "timeout"):
                print("\nStopping due to failure (--stop-on-failure)")
                break

        # Generate report
        report = ExampleReport(
            total=len(results),
            passed=sum(1 for r in results if r.status == "passed"),
            failed=sum(1 for r in results if r.status == "failed"),
            skipped=sum(1 for r in results if r.status == "skipped"),
            timeout=sum(1 for r in results if r.status == "timeout"),
            results=results,
            total_duration=total_duration,
        )

        return report


class ExampleFilter:
    """Filters examples based on various criteria."""

    @staticmethod
    def by_category(
        examples: list[ExampleMetadata], categories: list[str]
    ) -> list[ExampleMetadata]:
        """Filter examples by category."""
        if not categories:
            return examples
        return [e for e in examples if e.category in categories]

    @staticmethod
    def by_name(
        examples: list[ExampleMetadata], names: list[str]
    ) -> list[ExampleMetadata]:
        """Filter examples by name (partial match)."""
        if not names:
            return examples
        return [
            e for e in examples if any(name.lower() in e.name.lower() for name in names)
        ]

    @staticmethod
    def by_model(
        examples: list[ExampleMetadata], model: str
    ) -> list[ExampleMetadata]:
        """Filter examples by required model."""
        if not model:
            return examples
        return [
            e
            for e in examples
            if e.requires_model and model.lower() in e.requires_model.lower()
        ]

    @staticmethod
    def exclude_server(examples: list[ExampleMetadata]) -> list[ExampleMetadata]:
        """Exclude examples that require a server."""
        return [e for e in examples if not e.requires_server]


class ExampleReporter:
    """Formats and displays example results."""

    @staticmethod
    def print_summary_table(report: ExampleReport):
        """Print a summary table of results."""
        print("\n" + "=" * 80)
        print(" " * 25 + "EXAMPLE RUNNER SUMMARY")
        print("=" * 80)

        # Status breakdown
        print(f"\nTotal Examples: {report.total}")
        print(f"   Passed:  {report.passed}")
        print(f"   Failed:  {report.failed}")
        print(f"  � Timeout: {report.timeout}")
        print(f"  � Skipped: {report.skipped}")
        print(f"\nSuccess Rate: {report.success_rate:.1f}%")
        print(f"Total Duration: {report.total_duration:.2f}s")

        # Results table
        print("\n" + "-" * 80)
        print(f"{'Example':<40} {'Status':<10} {'Duration':<12} {'Category':<15}")
        print("-" * 80)

        for result in report.results:
            status_icon = {
                "passed": "",
                "failed": "",
                "timeout": "�",
                "skipped": "�",
            }.get(result.status, "?")

            # Truncate long names
            name = result.example_name
            if len(name) > 38:
                name = name[:35] + "..."

            # Extract category from result (need to store it)
            category = ""

            print(
                f"{name:<40} {status_icon} {result.status:<9} "
                f"{result.duration:>6.2f}s     {category:<15}"
            )

        print("-" * 80)

    @staticmethod
    def print_list(examples: list[ExampleMetadata]):
        """Print a list of discovered examples."""
        # Group by category
        by_category: dict[str, list[ExampleMetadata]] = {}
        for example in examples:
            if example.category not in by_category:
                by_category[example.category] = []
            by_category[example.category].append(example)

        print("\n" + "=" * 80)
        print(" " * 25 + "AVAILABLE EXAMPLES")
        print("=" * 80)

        for category, examples_in_cat in sorted(by_category.items()):
            print(f"\n{category.upper()}:")
            print("-" * 80)

            for example in sorted(examples_in_cat, key=lambda e: e.name):
                # Truncate description
                desc = example.description or "No description available"
                if len(desc) > 60:
                    desc = desc[:57] + "..."

                print(f"  • {example.name}")
                print(f"    {desc}")

                if example.requires_model:
                    print(f"    Requires: {example.requires_model}")
                if example.requires_server:
                    print("    Requires: Server running")

                print()

        print("-" * 80)
        print(f"Total: {len(examples)} examples")
        print("=" * 80)

    @staticmethod
    def export_json(report: ExampleReport, output_path: Path):
        """Export report to JSON file."""
        # Convert dataclasses to dicts
        report_dict = {
            "summary": {
                "total": report.total,
                "passed": report.passed,
                "failed": report.failed,
                "skipped": report.skipped,
                "timeout": report.timeout,
                "success_rate": report.success_rate,
                "total_duration": report.total_duration,
            },
            "results": [asdict(r) for r in report.results],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        print(f"\nResults exported to: {output_path}")


def check_server_running() -> bool:
    """Check if the SMLX server is running."""
    try:
        import requests

        response = requests.get("http://localhost:8000/v1/models", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def main():
    """Main entry point for the example runner."""
    parser = argparse.ArgumentParser(
        description="Run SMLX examples with filtering and reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all examples
  python -m examples.example_runner --list

  # Run all examples
  python -m examples.example_runner

  # Run model examples only
  python -m examples.example_runner --category model

  # Run specific example
  python -m examples.example_runner --run smollm2_135m_example

  # Run with custom timeout
  python -m examples.example_runner --timeout 600 --verbose

  # Export results to JSON
  python -m examples.example_runner --json-output results.json
        """,
    )

    parser.add_argument(
        "--list", action="store_true", help="List all available examples without running"
    )

    parser.add_argument(
        "--category",
        type=str,
        nargs="+",
        help="Filter by category (model, quant, agent, server, eval, performance)",
    )

    parser.add_argument(
        "--run",
        type=str,
        nargs="+",
        help="Run specific example(s) by name (supports partial matches)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Filter examples that use a specific model",
    )

    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip examples that require a running server",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds (overrides category-specific defaults)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output from examples"
    )

    parser.add_argument(
        "--json-output", type=str, help="Export results to JSON file"
    )

    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop running examples after first failure",
    )

    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies only (don't run examples)",
    )

    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help=f"Python interpreter to use (default: {sys.executable})",
    )

    args = parser.parse_args()

    # Find examples directory
    script_dir = Path(__file__).parent
    examples_dir = script_dir

    # Discover examples
    print("Discovering examples...")
    discovery = ExampleDiscovery(examples_dir)
    all_examples = discovery.discover_all()
    print(f"Found {len(all_examples)} examples")

    # Apply filters
    filtered_examples = all_examples

    if args.category:
        filtered_examples = ExampleFilter.by_category(filtered_examples, args.category)
        print(f"Filtered to {len(filtered_examples)} examples by category")

    if args.run:
        filtered_examples = ExampleFilter.by_name(filtered_examples, args.run)
        print(f"Filtered to {len(filtered_examples)} examples by name")

    if args.model:
        filtered_examples = ExampleFilter.by_model(filtered_examples, args.model)
        print(f"Filtered to {len(filtered_examples)} examples by model")

    if args.skip_server:
        filtered_examples = ExampleFilter.exclude_server(filtered_examples)
        print(f"Filtered to {len(filtered_examples)} examples (excluding server)")

    # List mode
    if args.list:
        ExampleReporter.print_list(filtered_examples)
        return 0

    # Check dependencies mode
    if args.check_deps:
        print("\nChecking dependencies...")
        has_issues = False

        # Check if server is required but not running
        server_required = any(e.requires_server for e in filtered_examples)
        if server_required:
            if check_server_running():
                print(" Server is running")
            else:
                print(" Server required but not running (start with: python -m smlx.server.app)")
                has_issues = True

        # Check for model requirements
        models_required = {e.requires_model for e in filtered_examples if e.requires_model}
        if models_required:
            print(f"\nModels required: {', '.join(models_required)}")
            print("  (Ensure models are downloaded before running examples)")

        return 1 if has_issues else 0

    # No examples to run
    if not filtered_examples:
        print("\n No examples match the specified filters")
        return 1

    # Run examples
    print(f"\nRunning {len(filtered_examples)} example(s)...\n")

    runner = ExampleRunner(python_path=args.python, verbose=args.verbose)

    try:
        report = runner.run_multiple(
            filtered_examples,
            timeout=args.timeout,
            stop_on_failure=args.stop_on_failure,
        )
    except KeyboardInterrupt:
        print("\n\n� Interrupted by user")
        return 130

    # Display results
    ExampleReporter.print_summary_table(report)

    # Export to JSON if requested
    if args.json_output:
        output_path = Path(args.json_output)
        ExampleReporter.export_json(report, output_path)

    # Return exit code based on results
    return 0 if report.failed == 0 and report.timeout == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
