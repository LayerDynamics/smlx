"""
SMLX Benchmark Suite Runner

Unified CLI for running all benchmark suites.

Available Suites:
- system: Display system information (chip, memory, MLX)
- text_generation: Text generation benchmarks (context scaling, generation length, etc.)
- quantization: Compare quantization methods (FP16, 8-bit, 4-bit)
- llm: Language model benchmarks (prompt processing, generation)
- vlm: Vision-language model benchmarks
- ops: Low-level operation benchmarks (matmul, attention)

Usage:
    # List available benchmark suites
    python -m smlx.bench.run --list

    # Display system information
    python -m smlx.bench.run system

    # Run specific suite
    python -m smlx.bench.run text_generation --model SmolLM2-135M

    # Run all benchmarks
    python -m smlx.bench.run --all --model SmolLM2-135M

    # Save results
    python -m smlx.bench.run text_generation \\
        --model SmolLM2-135M \\
        --output results/benchmarks.json
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from smlx.utils import format_duration, save_json

# Available benchmark suites
BENCHMARK_SUITES = {
    "system": {
        "name": "System Information",
        "description": "Display system capabilities (chip, memory, MLX)",
        "module": "smlx.bench.suites.system",
        "function": "print_system_info",
        "requires_model": False,
        "model_types": [],
        "returns_stats": False,  # Special case: prints directly, no stats returned
    },
    "text_generation": {
        "name": "Text Generation Benchmarks",
        "description": "Context scaling, generation length, temperature effects",
        "module": "smlx.bench.suites.text_generation",
        "function": "run_comprehensive_suite",
        "requires_model": True,
        "model_types": ["text"],
        "returns_stats": True,
    },
    "quantization": {
        "name": "Quantization Comparison",
        "description": "Compare FP16, 8-bit, 4-bit quantization methods",
        "module": "smlx.bench.suites.quantization",
        "function": "compare_quantization_methods",
        "requires_model": True,
        "model_types": ["text", "vlm"],
        "returns_stats": True,
    },
    "llm": {
        "name": "Language Model Benchmarks",
        "description": "Basic LLM performance benchmarks",
        "module": "smlx.bench.suites.llm",
        "function": "benchmark_llm",
        "requires_model": True,
        "model_types": ["text"],
        "returns_stats": True,
    },
    "ops": {
        "name": "Operation Benchmarks",
        "description": "Low-level MLX operation benchmarks (matmul, attention)",
        "module": "smlx.bench.suites.ops",
        "function": "run_ops_suite",
        "requires_model": False,
        "model_types": [],
        "returns_stats": True,
    },
}


def list_benchmark_suites():
    """List all available benchmark suites."""
    print("\n" + "=" * 70)
    print("AVAILABLE BENCHMARK SUITES")
    print("=" * 70)

    for suite_id, suite_info in BENCHMARK_SUITES.items():
        print(f"\n{suite_id}:")
        print(f"  Name: {suite_info['name']}")
        print(f"  Description: {suite_info['description']}")
        print(f"  Requires model: {suite_info['requires_model']}")
        if suite_info['model_types']:
            print(f"  Model types: {', '.join(suite_info['model_types'])}")

    print("\n" + "=" * 70)
    print("\nUsage:")
    print("  python -m smlx.bench.run <suite_name> --model <model_path>")
    print("\nExamples:")
    print("  python -m smlx.bench.run system")
    print("  python -m smlx.bench.run text_generation --model mlx-community/SmolLM2-135M-Instruct")
    print("  python -m smlx.bench.run quantization --model SmolLM2-135M --methods fp16 8bit 4bit")
    print("  python -m smlx.bench.run ops --operation matmul --shape 1000,1000")
    print("=" * 70)


def run_benchmark_suite(
    suite_name: str,
    model_path: Optional[str] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
    **kwargs,
) -> Optional[dict[str, Any]]:
    """
    Run a specific benchmark suite.

    Args:
        suite_name: Name of the suite to run
        model_path: Path to model (if required)
        output_path: Path to save results
        verbose: Print progress
        **kwargs: Additional arguments for the benchmark

    Returns:
        Dictionary of benchmark results
    """
    if suite_name not in BENCHMARK_SUITES:
        print(f"Error: Unknown benchmark suite '{suite_name}'", file=sys.stderr)
        print(f"Available suites: {', '.join(BENCHMARK_SUITES.keys())}", file=sys.stderr)
        sys.exit(1)

    suite_info = BENCHMARK_SUITES[suite_name]

    # Check if model is required
    if suite_info['requires_model'] and not model_path:
        print(f"Error: Benchmark '{suite_name}' requires --model argument", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print("\n" + "=" * 70)
        print(f"RUNNING: {suite_info['name']}")
        print("=" * 70)
        if model_path:
            print(f"Model: {model_path}")
        print()

    # Import and run the benchmark
    try:
        module_name, function_name = suite_info['module'], suite_info['function']

        # Import module
        import importlib
        module = importlib.import_module(module_name)

        # Get function
        benchmark_function = getattr(module, function_name)

        # Special case: system info just prints and returns None
        if suite_name == "system":
            benchmark_function()
            return None

        # Prepare arguments
        benchmark_args = {}
        if suite_info['requires_model']:
            # At this point model_path cannot be None due to check at line 144-146
            assert model_path is not None, "model_path should not be None for model-based suites"

            if suite_name == "llm":
                # LLM benchmark needs model and tokenizer loaded
                from smlx.models.SmolLM2_135M import load
                model, tokenizer = load(model_path)
                benchmark_args['model'] = model
                benchmark_args['tokenizer'] = tokenizer
            else:
                benchmark_args['model_path'] = model_path

        # Add custom arguments
        benchmark_args.update(kwargs)
        if 'verbose' in benchmark_args:
            benchmark_args['verbose'] = verbose

        # Run benchmark
        import time
        start_time = time.time()

        results = benchmark_function(**benchmark_args)

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"\nBenchmark completed in {format_duration(elapsed_time)}")

        # Save results if requested
        if output_path:
            # Convert results to serializable format
            if hasattr(results, 'to_dict'):
                output_data = results.to_dict()
            elif isinstance(results, dict):
                output_data = {}
                for key, value in results.items():
                    if hasattr(value, 'to_dict'):
                        output_data[key] = value.to_dict()
                    elif isinstance(value, list):
                        output_data[key] = [
                            item.to_dict() if hasattr(item, 'to_dict') else item
                            for item in value
                        ]
                    else:
                        output_data[key] = value
            else:
                output_data = results

            save_json(output_data, output_path)
            if verbose:
                print(f"Results saved to {output_path}")

        return results

    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_all_benchmarks(
    model_path: str,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run all applicable benchmark suites.

    Args:
        model_path: Path to model
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        Dictionary mapping suite names to results
    """
    all_results = {}

    # Determine which suites to run based on model type
    # For now, run all model-based suites
    suites_to_run = [
        name for name, info in BENCHMARK_SUITES.items()
        if info['requires_model']
    ]

    if verbose:
        print("\n" + "=" * 70)
        print("RUNNING ALL BENCHMARK SUITES")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Suites: {', '.join(suites_to_run)}")
        print("=" * 70)

    for suite_name in suites_to_run:
        try:
            # Determine output path
            if output_dir:
                output_path = output_dir / f"{suite_name}_results.json"
            else:
                output_path = None

            # Run suite
            results = run_benchmark_suite(
                suite_name=suite_name,
                model_path=model_path,
                output_path=output_path,
                verbose=verbose,
            )

            all_results[suite_name] = results

        except Exception as e:
            print(f"Error running {suite_name}: {e}", file=sys.stderr)
            continue

    if verbose:
        print("\n" + "=" * 70)
        print("ALL BENCHMARKS COMPLETE")
        print("=" * 70)

    return all_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SMLX Benchmark Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available benchmark suites
  python -m smlx.bench.run --list

  # Run text generation benchmarks
  python -m smlx.bench.run text_generation --model mlx-community/SmolLM2-135M-Instruct

  # Run quantization comparison
  python -m smlx.bench.run quantization \\
      --model mlx-community/SmolLM2-135M-Instruct \\
      --methods fp16 8bit 4bit

  # Run all benchmarks
  python -m smlx.bench.run --all \\
      --model mlx-community/SmolLM2-135M-Instruct \\
      --output-dir results/

  # Run with custom parameters
  python -m smlx.bench.run text_generation \\
      --model mlx-community/SmolLM2-135M-Instruct \\
      --output results/text_gen.json \\
      --verbose
        """,
    )

    parser.add_argument(
        "suite",
        nargs="?",
        help="Benchmark suite to run (see --list for available suites)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmark suites and exit",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all applicable benchmark suites",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model path or HuggingFace ID",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results (used with --all)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    # Additional arguments for specific benchmarks
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Quantization methods for quantization benchmark (fp16, 8bit, 4bit)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Test prompt for benchmarks",
    )

    parser.add_argument(
        "--generation-tokens",
        type=int,
        help="Number of tokens to generate",
    )

    # Operation benchmark arguments
    parser.add_argument(
        "--operation",
        type=str,
        choices=["matmul", "attention", "layernorm", "all"],
        help="Operation to benchmark (for ops suite)",
    )

    parser.add_argument(
        "--shape",
        type=str,
        help="Shape for operation (e.g., '1000,1000' for matmul)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of iterations for benchmarks",
    )

    args = parser.parse_args()

    # Handle verbose/quiet
    verbose = args.verbose and not args.quiet

    # List suites if requested
    if args.list:
        list_benchmark_suites()
        return

    # Run all benchmarks if requested
    if args.all:
        if not args.model:
            parser.error("--all requires --model argument")

        # Create output directory if specified
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)

        run_all_benchmarks(
            model_path=args.model,
            output_dir=args.output_dir,
            verbose=verbose,
        )
        return

    # Check that a suite was specified
    if not args.suite:
        parser.error("Please specify a benchmark suite or use --list or --all")

    # Prepare additional arguments
    kwargs = {}
    if args.methods:
        kwargs['quantization_methods'] = args.methods
    if args.prompt:
        kwargs['prompt'] = args.prompt
        kwargs['test_prompt'] = args.prompt
    if args.generation_tokens:
        kwargs['generation_tokens'] = args.generation_tokens
    if args.operation:
        kwargs['operation'] = args.operation
    if args.shape:
        kwargs['shape'] = args.shape
    if args.iterations:
        kwargs['num_iterations'] = args.iterations

    # Run the specified suite
    run_benchmark_suite(
        suite_name=args.suite,
        model_path=args.model,
        output_path=args.output,
        verbose=verbose,
        **kwargs,
    )


if __name__ == "__main__":
    main()
