#!/usr/bin/env python3
"""Preview samples from any bundled dataset under ``data/``.

Loads a dataset through :mod:`smlx.data.local` and prints a handful of samples
with each field rendered compactly (long text truncated; images shown as
``WxH mode``; audio shown as ``Ns @ rateHz``). No model is loaded -- this is a
fast "what's actually in here" inspector that works across all on-disk layouts
(HuggingFace ``save_to_disk``, parquet snapshots, image folders, JSON indexes).

Usage::

    python scripts/preview_data.py --list                 # list datasets
    python scripts/preview_data.py wikitext               # preview default split
    python scripts/preview_data.py glue --split mrpc_validation -n 3
    python scripts/preview_data.py coco8 --split val
    python scripts/preview_data.py mathvista --json
"""

from __future__ import annotations

import argparse
import json
import sys

from smlx.data import local, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("dataset", nargs="?", help="Dataset name (see --list)")
    parser.add_argument("--split", help="Split to preview (default: dataset's default)")
    parser.add_argument("-n", "--limit", type=int, default=5, help="Number of samples to show")
    parser.add_argument("--list", action="store_true", help="List datasets and exit")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    if args.list or not args.dataset:
        for line in report.list_lines(local.inventory(compute_size=False)):
            print(line)
        return 0

    name = args.dataset
    if name not in local.registry():
        print(f"Unknown dataset '{name}'. Use --list to see available datasets.", file=sys.stderr)
        return 2
    if not local.is_available(name):
        print(
            f"Dataset '{name}' is not present on disk. Download it with:\n"
            f"  python -m smlx.tools.download_data --dataset {name}",
            file=sys.stderr,
        )
        return 1

    try:
        samples = list(local.iter_samples(name, split=args.split, limit=args.limit))
    except Exception as exc:
        print(f"Failed to load samples from '{name}': {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"dataset": name, "split": args.split, "samples": samples}, indent=2))
        return 0

    for line in report.preview_lines(name, args.split, samples):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
