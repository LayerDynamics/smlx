#!/usr/bin/env python3
"""Validate and inventory the bundled datasets under ``data/``.

Probes every dataset registered in :mod:`smlx.tools.download_data` (and present
on disk), reporting layout, split availability, example/shard counts, on-disk
size, and whether a single sample loads cleanly. Also surfaces *orphans*:
dataset directories on disk that no registry entry points at (drift between the
registry and what was actually downloaded).

The probe is lightweight -- it reads ``dataset_info.json`` and loads exactly one
sample per dataset; it never materialises a full dataset into memory (important
on the 36 GB M4 target where ``data/datasets`` alone is ~12 GB).

Usage::

    python scripts/validate_data.py                 # full report
    python scripts/validate_data.py --category benchmark
    python scripts/validate_data.py --no-size       # skip on-disk size (faster)
    python scripts/validate_data.py --json          # machine-readable output

Exit code is non-zero if any *present* dataset fails its single-sample load
(i.e. is corrupt / unreadable). Datasets that are simply not downloaded are
reported but do not fail the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from smlx.data import local, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--category",
        choices=["benchmark", "training"],
        help="Only report datasets in this category",
    )
    parser.add_argument(
        "--no-size",
        action="store_true",
        help="Skip on-disk size computation (faster for large dirs)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = parser.parse_args()

    results = local.inventory(compute_size=not args.no_size)
    if args.category:
        results = [r for r in results if r.category == args.category]
    orphans = local.find_orphans()

    if args.json:
        payload = {
            "data_dir": str(local.data_dir()),
            "datasets": [{**asdict(r), "layout": r.layout.value} for r in results],
            "orphans": orphans,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"SMLX data directory: {local.data_dir()}\n")
        for line in report.inventory_lines(results, orphans=orphans, show_size=not args.no_size):
            print(line)

    # Non-zero exit only when a *present* dataset is corrupt/unreadable.
    failed = [r for r in results if r.available and r.sample_ok is False]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
