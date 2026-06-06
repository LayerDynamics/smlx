"""Shared, surface-agnostic rendering of local-dataset reports.

Pure formatting helpers that turn :mod:`smlx.data.local` probe/preview results
into plain text lines. Used by ``scripts/validate_data.py``,
``scripts/preview_data.py`` and the ``smlx data`` CLI so the table / preview
formatting lives in exactly one place.
"""

from __future__ import annotations

from smlx.data import local


def human_size(n: int) -> str:
    """Render a byte count as a compact human-readable size."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def list_lines(results: list[local.ProbeResult]) -> list[str]:
    """Compact dataset listing (name / category / present / layout / splits)."""
    lines = [f"{'DATASET':<20} {'CAT':<9} {'PRESENT':<7} {'LAYOUT':<15} SPLITS"]
    lines.append("-" * 72)
    for r in results:
        present = "yes" if r.available else "no"
        splits = ", ".join(r.splits) if r.splits else ""
        lines.append(f"{r.name:<20} {r.category:<9} {present:<7} {r.layout.value:<15} {splits}")
    return lines


def inventory_lines(
    results: list[local.ProbeResult],
    orphans: list[str] | None = None,
    show_size: bool = True,
) -> list[str]:
    """Full validation report: per-dataset table + summary + orphans."""
    header = (
        f"{'DATASET':<20} {'CAT':<9} {'PRESENT':<7} {'LAYOUT':<15} "
        f"{'EXAMPLES':>9} {'SHARDS':>6}"
    )
    if show_size:
        header += f" {'SIZE':>8}"
    header += f" {'SAMPLE':<6}"

    lines = [header, "-" * len(header)]
    for r in results:
        present = "yes" if r.available else "NO"
        examples = "" if r.num_examples is None else f"{r.num_examples:,}"
        if not r.available:
            sample = "-"
        elif r.sample_ok is True:
            sample = "ok"
        elif r.sample_ok is False:
            sample = "FAIL"
        else:
            sample = "?"
        row = (
            f"{r.name:<20} {r.category:<9} {present:<7} {r.layout.value:<15} "
            f"{examples:>9} {r.num_shards:>6}"
        )
        if show_size:
            row += f" {human_size(r.size_bytes) if r.size_bytes else '':>8}"
        row += f" {sample:<6}"
        lines.append(row)
        if r.error:
            lines.append(f"    └─ error: {r.error}")

    present_ds = [r for r in results if r.available]
    missing = [r for r in results if not r.available]
    failed = [r for r in results if r.available and r.sample_ok is False]
    total_size = sum(r.size_bytes for r in present_ds)

    lines.append("")
    summary = (
        f"Summary: {len(present_ds)} present, {len(missing)} not downloaded, "
        f"{len(failed)} failed to load."
    )
    if show_size:
        summary += f" On-disk total: {human_size(total_size)}."
    lines.append(summary)

    if missing:
        lines.append("  Not downloaded: " + ", ".join(r.name for r in missing))
        lines.append("  (fetch with: python -m smlx.tools.download_data --dataset <name>)")
    if orphans:
        lines.append("")
        lines.append(
            "  Orphans (data on disk not matched to a registry path -- " "registry/disk drift):"
        )
        lines.extend(f"    - {o}" for o in orphans)
    if failed:
        lines.append("")
        lines.append("  FAILED datasets:")
        lines.extend(f"    - {r.name}: {r.error}" for r in failed)
    return lines


def preview_lines(name: str, split: str | None, samples: list[dict[str, str]]) -> list[str]:
    """Header + per-sample field dump for a dataset preview."""
    entry = local.get(name)
    splits = local.available_splits(name)
    lines = [
        f"Dataset:  {name}  ({entry.category})",
        f"Path:     {local.local_path(name).relative_to(local.data_dir())}",
        f"Layout:   {local.detect_layout(entry.path).value}",
        f"Splits:   {', '.join(splits)}",
    ]
    if entry.description:
        lines.append(f"About:    {entry.description}")
    suffix = f" from split '{split}'" if split else ""
    lines.append(f"Showing {len(samples)} sample(s){suffix}")
    lines.append("=" * 72)
    for i, sample in enumerate(samples):
        lines.append(f"[{i}]")
        width = max((len(k) for k in sample), default=0)
        for key, value in sample.items():
            lines.append(f"  {key:<{width}} : {value}")
        lines.append("-" * 72)
    return lines
