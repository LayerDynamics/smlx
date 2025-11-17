"""
Compare evaluation results from multiple models.

This tool compares results from SMLX evaluation benchmarks across different models,
providing detailed comparisons and statistical analysis.

Usage:
    # Compare two model runs on MathVista
    python -m smlx.tools.compare_results \\
        results/model1_mathvista.json \\
        results/model2_mathvista.json

    # Compare multiple models with custom output
    python -m smlx.tools.compare_results \\
        results/*.json \\
        --output comparison_report.json \\
        --format both

    # Compare specific benchmark
    python -m smlx.tools.compare_results \\
        results/*_mmmu.json \\
        --benchmark mmmu
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from smlx.utils import (
    ensure_dir,
    format_number,
    format_percentage,
    format_table,
    load_json,
    save_json,
)


class ResultComparator:
    """Compare evaluation results from multiple models."""

    def __init__(self, result_files: list[Path], benchmark: Optional[str] = None):
        """
        Initialize comparator.

        Args:
            result_files: list of result JSON file paths
            benchmark: Benchmark type (auto-detected if None)
        """
        self.result_files = result_files
        self.benchmark = benchmark
        self.results = []
        self.model_names = []

        self._load_results()

    def _load_results(self) -> None:
        """Load all result files."""
        for path in self.result_files:
            try:
                data = load_json(path)
                self.results.append(data)

                # Extract model name from filename or data
                if "model_name" in data:
                    model_name = data["model_name"]
                else:
                    model_name = path.stem

                self.model_names.append(model_name)

                # Auto-detect benchmark if not specified
                if self.benchmark is None:
                    self.benchmark = self._detect_benchmark(data)

            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)

    def _detect_benchmark(self, data: dict[str, Any]) -> str:
        """
        Auto-detect benchmark type from result structure.

        Args:
            data: Result data

        Returns:
            Benchmark name
        """
        # Check for benchmark-specific keys
        if "overall_accuracy" in data:
            if "category_scores" in data:
                if any(
                    "coarse_perception" in str(k).lower() for k in data.get("category_scores", {})
                ):
                    return "mmstar"
                elif any(
                    "question_type" in str(k).lower() for k in data.get("category_scores", {})
                ):
                    return "mathvista"
            elif "subject_scores" in data:
                return "mmmu"
            elif "type_scores" in data or "task_scores" in data:
                return "ocrbench"

        return "unknown"

    def compare_overall(self) -> dict[str, Any]:
        """
        Compare overall accuracy across models.

        Returns:
            dictionary with overall comparison
        """
        comparison = {"benchmark": self.benchmark, "num_models": len(self.results), "models": []}

        for model_name, result in zip(self.model_names, self.results):
            model_data = {
                "name": model_name,
                "accuracy": result.get("overall_accuracy", 0.0),
                "total_samples": result.get("total_samples", 0),
                "correct": result.get("correct_samples", 0),
            }
            comparison["models"].append(model_data)

        # Sort by accuracy (descending)
        comparison["models"].sort(key=lambda x: x["accuracy"], reverse=True)

        # Add rank
        for i, model in enumerate(comparison["models"], 1):
            model["rank"] = i

        # Calculate statistics
        accuracies = [m["accuracy"] for m in comparison["models"]]
        if accuracies:
            comparison["statistics"] = {
                "best": max(accuracies),
                "worst": min(accuracies),
                "mean": sum(accuracies) / len(accuracies),
                "range": max(accuracies) - min(accuracies),
            }

        return comparison

    def compare_categories(self) -> dict[str, Any]:
        """
        Compare category-wise performance.

        Returns:
            dictionary with category comparison
        """
        if self.benchmark == "mathvista":
            return self._compare_mathvista_categories()
        elif self.benchmark == "mmmu":
            return self._compare_mmmu_subjects()
        elif self.benchmark == "mmstar":
            return self._compare_mmstar_capabilities()
        elif self.benchmark == "ocrbench":
            return self._compare_ocrbench_types()
        else:
            return {}

    def _compare_mathvista_categories(self) -> dict[str, Any]:
        """Compare MathVista category scores."""
        comparison = {"categories": {}}

        # Collect all categories
        all_categories = set()
        for result in self.results:
            if "category_scores" in result:
                all_categories.update(result["category_scores"].keys())

        # Compare each category
        for category in sorted(all_categories):
            category_data = {"category": category, "models": []}

            for model_name, result in zip(self.model_names, self.results):
                if "category_scores" in result and category in result["category_scores"]:
                    score_data = result["category_scores"][category]
                    category_data["models"].append(
                        {
                            "name": model_name,
                            "accuracy": score_data.get("accuracy", 0.0),
                            "count": score_data.get("count", 0),
                        }
                    )

            # Sort by accuracy
            category_data["models"].sort(key=lambda x: x["accuracy"], reverse=True)

            # Calculate best/worst
            if category_data["models"]:
                accuracies = [m["accuracy"] for m in category_data["models"]]
                category_data["best"] = max(accuracies)
                category_data["worst"] = min(accuracies)
                category_data["range"] = max(accuracies) - min(accuracies)

            comparison["categories"][category] = category_data

        return comparison

    def _compare_mmmu_subjects(self) -> dict[str, Any]:
        """Compare MMMU subject scores."""
        comparison = {"subjects": {}}

        # Collect all subjects
        all_subjects = set()
        for result in self.results:
            if "subject_scores" in result:
                all_subjects.update(result["subject_scores"].keys())

        # Compare each subject
        for subject in sorted(all_subjects):
            subject_data = {"subject": subject, "models": []}

            for model_name, result in zip(self.model_names, self.results):
                if "subject_scores" in result and subject in result["subject_scores"]:
                    score_data = result["subject_scores"][subject]
                    subject_data["models"].append(
                        {
                            "name": model_name,
                            "accuracy": score_data.get("accuracy", 0.0),
                            "count": score_data.get("count", 0),
                        }
                    )

            # Sort by accuracy
            subject_data["models"].sort(key=lambda x: x["accuracy"], reverse=True)

            # Calculate best/worst
            if subject_data["models"]:
                accuracies = [m["accuracy"] for m in subject_data["models"]]
                subject_data["best"] = max(accuracies)
                subject_data["worst"] = min(accuracies)
                subject_data["range"] = max(accuracies) - min(accuracies)

            comparison["subjects"][subject] = subject_data

        return comparison

    def _compare_mmstar_capabilities(self) -> dict[str, Any]:
        """Compare MMStar capability scores."""
        comparison = {"capabilities": {}}

        # Collect all capabilities
        all_capabilities = set()
        for result in self.results:
            if "category_scores" in result:
                all_capabilities.update(result["category_scores"].keys())

        # Compare each capability
        for capability in sorted(all_capabilities):
            capability_data = {"capability": capability, "models": []}

            for model_name, result in zip(self.model_names, self.results):
                if "category_scores" in result and capability in result["category_scores"]:
                    score_data = result["category_scores"][capability]
                    capability_data["models"].append(
                        {
                            "name": model_name,
                            "accuracy": score_data.get("accuracy", 0.0),
                            "count": score_data.get("count", 0),
                        }
                    )

            # Sort by accuracy
            capability_data["models"].sort(key=lambda x: x["accuracy"], reverse=True)

            # Calculate best/worst
            if capability_data["models"]:
                accuracies = [m["accuracy"] for m in capability_data["models"]]
                capability_data["best"] = max(accuracies)
                capability_data["worst"] = min(accuracies)
                capability_data["range"] = max(accuracies) - min(accuracies)

            comparison["capabilities"][capability] = capability_data

        return comparison

    def _compare_ocrbench_types(self) -> dict[str, Any]:
        """Compare OCRBench task type scores."""
        comparison = {"task_types": {}}

        # Collect all task types
        all_types = set()
        for result in self.results:
            if "type_scores" in result:
                all_types.update(result["type_scores"].keys())
            elif "task_scores" in result:
                all_types.update(result["task_scores"].keys())

        # Compare each type
        for task_type in sorted(all_types):
            type_data = {"task_type": task_type, "models": []}

            for model_name, result in zip(self.model_names, self.results):
                scores_key = "type_scores" if "type_scores" in result else "task_scores"
                if scores_key in result and task_type in result[scores_key]:
                    score_data = result[scores_key][task_type]
                    type_data["models"].append(
                        {
                            "name": model_name,
                            "accuracy": score_data.get("accuracy", 0.0),
                            "count": score_data.get("count", 0),
                        }
                    )

            # Sort by accuracy
            type_data["models"].sort(key=lambda x: x["accuracy"], reverse=True)

            # Calculate best/worst
            if type_data["models"]:
                accuracies = [m["accuracy"] for m in type_data["models"]]
                type_data["best"] = max(accuracies)
                type_data["worst"] = min(accuracies)
                type_data["range"] = max(accuracies) - min(accuracies)

            comparison["task_types"][task_type] = type_data

        return comparison

    def generate_comparison(self) -> dict[str, Any]:
        """
        Generate complete comparison.

        Returns:
            Complete comparison dictionary
        """
        comparison = {
            "benchmark": self.benchmark,
            "overall": self.compare_overall(),
            "detailed": self.compare_categories(),
        }

        return comparison

    def format_text_report(self, comparison: dict[str, Any]) -> str:
        """
        Format comparison as text report.

        Args:
            comparison: Comparison data

        Returns:
            Formatted text report
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("SMLX Evaluation Comparison Report")
        lines.append(f"Benchmark: {comparison['benchmark'].upper()}")
        lines.append("=" * 80)
        lines.append("")

        # Overall comparison
        lines.append("Overall Performance")
        lines.append("-" * 80)

        overall = comparison["overall"]
        table_data = []
        for model in overall["models"]:
            table_data.append(
                {
                    "Rank": model["rank"],
                    "Model": model["name"],
                    "Accuracy": format_percentage(model["accuracy"]),
                    "Correct": f"{model['correct']}/{model['total_samples']}",
                    "Samples": format_number(model["total_samples"]),
                }
            )

        lines.append(format_table(table_data))
        lines.append("")

        # Statistics
        if "statistics" in overall:
            stats = overall["statistics"]
            lines.append("Statistics:")
            lines.append(f"  Best:  {format_percentage(stats['best'])}")
            lines.append(f"  Worst: {format_percentage(stats['worst'])}")
            lines.append(f"  Mean:  {format_percentage(stats['mean'])}")
            lines.append(f"  Range: {format_percentage(stats['range'])}")
            lines.append("")

        # Detailed comparison
        detailed = comparison["detailed"]

        if "categories" in detailed:
            lines.append("Category-wise Comparison (MathVista)")
            lines.append("-" * 80)
            for category, data in detailed["categories"].items():
                lines.append(f"\n{category}:")
                for model in data["models"]:
                    lines.append(
                        f"  {model['name']}: {format_percentage(model['accuracy'])} ({model['count']} samples)"
                    )

        elif "subjects" in detailed:
            lines.append("Subject-wise Comparison (MMMU)")
            lines.append("-" * 80)
            for subject, data in detailed["subjects"].items():
                lines.append(f"\n{subject}:")
                for model in data["models"]:
                    lines.append(
                        f"  {model['name']}: {format_percentage(model['accuracy'])} ({model['count']} samples)"
                    )

        elif "capabilities" in detailed:
            lines.append("Capability-wise Comparison (MMStar)")
            lines.append("-" * 80)
            for capability, data in detailed["capabilities"].items():
                lines.append(f"\n{capability}:")
                for model in data["models"]:
                    lines.append(
                        f"  {model['name']}: {format_percentage(model['accuracy'])} ({model['count']} samples)"
                    )

        elif "task_types" in detailed:
            lines.append("Task Type Comparison (OCRBench)")
            lines.append("-" * 80)
            for task_type, data in detailed["task_types"].items():
                lines.append(f"\n{task_type}:")
                for model in data["models"]:
                    lines.append(
                        f"  {model['name']}: {format_percentage(model['accuracy'])} ({model['count']} samples)"
                    )

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    """Main entry point for result comparison."""
    parser = argparse.ArgumentParser(
        description="Compare SMLX evaluation results from multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two model runs
  python -m smlx.tools.compare_results results/model1.json results/model2.json

  # Compare all MathVista results
  python -m smlx.tools.compare_results results/*_mathvista.json

  # Compare with custom output
  python -m smlx.tools.compare_results results/*.json --output comparison.json --format both

  # Compare specific benchmark
  python -m smlx.tools.compare_results results/*.json --benchmark mmmu
        """,
    )

    parser.add_argument("result_files", nargs="+", type=Path, help="Result JSON files to compare")

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["mathvista", "mmmu", "mmstar", "ocrbench"],
        help="Benchmark type (auto-detected if not specified)",
    )

    parser.add_argument("--output", type=Path, help="Output file path (default: print to stdout)")

    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "both"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    # Validate input files
    valid_files = []
    for file in args.result_files:
        if file.exists():
            valid_files.append(file)
        else:
            print(f"Warning: File not found: {file}", file=sys.stderr)

    if not valid_files:
        print("Error: No valid result files found", file=sys.stderr)
        sys.exit(1)

    # Create comparator
    comparator = ResultComparator(valid_files, benchmark=args.benchmark)

    if not comparator.results:
        print("Error: Failed to load any results", file=sys.stderr)
        sys.exit(1)

    # Generate comparison
    comparison = comparator.generate_comparison()

    # Output results
    if args.format in ["text", "both"]:
        text_report = comparator.format_text_report(comparison)

        if args.output and args.format == "text":
            ensure_dir(args.output.parent)
            args.output.write_text(text_report)
            print(f"Text report saved to {args.output}")
        else:
            print(text_report)

    if args.format in ["json", "both"]:
        if args.output:
            output_path = args.output
            if args.format == "both":
                # Change extension to .json for JSON output
                output_path = args.output.with_suffix(".json")

            save_json(comparison, output_path)
            print(f"JSON report saved to {output_path}")
        else:
            print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
