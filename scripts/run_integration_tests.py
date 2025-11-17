#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Run integration tests in memory-safe batches.

Integration tests load large models (200-500MB each). Running all 16 test files
at once can exhaust system memory and cause kernel panics.

This script runs tests in small batches with cleanup between each batch.

Usage:
    python scripts/run_integration_tests.py              # Run all in batches
    python scripts/run_integration_tests.py --batch-size 2   # Smaller batches
    python scripts/run_integration_tests.py --group llm      # Run specific group
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Group integration tests by model type to reduce model loading overhead
TEST_GROUPS = {
    "llm": [
        "tests/integration/test_smollm2_generation.py",
        "tests/integration/test_smollm2_360m.py",
        "tests/integration/test_minilm.py",
        "tests/integration/test_all_minilm_l6_v2.py",
    ],
    "vlm": [
        "tests/integration/test_smolvlm_256m.py",
        "tests/integration/test_smolvlm_500m_instruct.py",
        "tests/integration/test_nanovlm.py",
        "tests/integration/test_tinyllava.py",
        "tests/integration/test_moondream2.py",
    ],
    "audio": [
        "tests/integration/test_whisper_tiny.py",
        "tests/integration/test_chatterbox.py",
        "tests/integration/test_orpheus.py",
        "tests/integration/test_yamnet.py",
        "tests/integration/test_silerovad.py",
    ],
    "ocr": [
        "tests/integration/test_trocr_small.py",
        "tests/integration/test_donut.py",
    ],
}


def run_test_batch(test_files, pytest_args=None):
    """Run a batch of test files."""
    pytest_args = pytest_args or []

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *test_files,
        "-v",
        *pytest_args,
    ]

    print(f"\n{'='*70}")
    print(f"Running batch: {', '.join(Path(f).name for f in test_files)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run integration tests in memory-safe batches"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of test files per batch (default: 3)",
    )
    parser.add_argument(
        "--group",
        choices=list(TEST_GROUPS.keys()) + ["all"],
        default="all",
        help="Run specific test group (default: all)",
    )
    parser.add_argument(
        "--pytest-args",
        nargs="*",
        default=[],
        help="Additional arguments to pass to pytest",
    )

    args = parser.parse_args()

    # Gather test files
    if args.group == "all":
        test_files = []
        for group_files in TEST_GROUPS.values():
            test_files.extend(group_files)
    else:
        test_files = TEST_GROUPS[args.group]

    # Split into batches
    batches = []
    for i in range(0, len(test_files), args.batch_size):
        batches.append(test_files[i:i + args.batch_size])

    print(f"\n{'='*70}")
    print(f"Integration Test Batching")
    print(f"{'='*70}")
    print(f"Total test files: {len(test_files)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(batches)}")
    print(f"{'='*70}\n")

    # Run each batch
    failed_batches = []
    for i, batch in enumerate(batches, 1):
        print(f"\n{'='*70}")
        print(f"BATCH {i}/{len(batches)}")
        print(f"{'='*70}")

        returncode = run_test_batch(batch, args.pytest_args)

        if returncode != 0:
            failed_batches.append((i, batch))
            print(f"\n❌ Batch {i} FAILED")
        else:
            print(f"\n✅ Batch {i} PASSED")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total batches: {len(batches)}")
    print(f"Passed: {len(batches) - len(failed_batches)}")
    print(f"Failed: {len(failed_batches)}")

    if failed_batches:
        print(f"\n❌ Failed batches:")
        for i, batch in failed_batches:
            print(f"  Batch {i}: {', '.join(Path(f).name for f in batch)}")
        sys.exit(1)
    else:
        print(f"\n✅ All batches passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
