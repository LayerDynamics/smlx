"""Local dataset access layer for the repo-root ``data/`` tree.

This is the single source of truth for *where* SMLX's bundled datasets live on
disk and *how* to load them. It is built directly on top of the dataset
registry declared in :mod:`smlx.tools.download_data` (``BENCHMARK_DATASETS`` and
``TRAINING_DATASETS``) so dataset paths are never hardcoded in more than one
place. Every consumer of ``data/`` -- the ``smlx data`` CLI, the
``scripts/validate_data.py`` / ``scripts/preview_data.py`` utilities, and the
showcase examples -- goes through this module.

The bundled datasets are stored in five distinct on-disk layouts; this module
detects and loads each transparently:

==================  ==========================================================
Layout              Shape on disk
==================  ==========================================================
``HF_DISK``         A single ``datasets.save_to_disk`` directory
                    (``dataset_info.json`` + ``*.arrow`` shards). e.g. esc50.
``HF_DISK_SPLITS``  A parent whose subdirectories are each an ``HF_DISK`` split
                    (e.g. ``wikitext/{test,train}``, ``glue/sst2_validation``).
``PARQUET``         An HF repo snapshot with ``data/*.parquet`` shards grouped
                    by split prefix (e.g. ``mathvista/data/testmini-*``).
``IMAGE_TREE``      An image folder tree ``images/{train,val}/*.jpg`` with an
                    optional parallel ``labels/`` tree (e.g. coco8, coco128).
``JSON_INDEX``      A JSON list of records referencing image files by path
                    (e.g. ``ocrbench/OCRBench_v2/OCRBench_v2.json``).
==================  ==========================================================
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from smlx.tools.download_data import BENCHMARK_DATASETS, TRAINING_DATASETS

# Repo root is three levels up from smlx/data/local.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Files that are never dataset content -- ignored when descending / classifying.
_IGNORE_NAMES = {
    "README.md",
    "readme.md",
    "LICENSE",
    "LICENSE.txt",
    ".gitkeep",
    ".gitattributes",
}
# JSON sidecars that are metadata, not a dataset index.
_META_JSON = {"dataset_info.json", "state.json", "metadata.json"}

# Image extensions recognised by the image-tree / json-index loaders.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


class Layout(str, Enum):
    """How a dataset is stored on disk (see module docstring)."""

    HF_DISK = "hf_disk"
    HF_DISK_SPLITS = "hf_disk_splits"
    PARQUET = "parquet"
    IMAGE_TREE = "image_tree"
    JSON_INDEX = "json_index"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass
class DatasetEntry:
    """A registered dataset and its location under ``data/``."""

    name: str
    category: str  # "benchmark" | "training"
    output_dir: str  # relative to DATA_DIR
    description: str = ""
    repo_id: str | None = None
    url: str | None = None
    config: str | None = None
    declared_splits: list[str] = field(default_factory=list)
    declared_size_mb: int | None = None
    note: str | None = None

    @property
    def path(self) -> Path:
        """Absolute path to the dataset's declared output directory."""
        return DATA_DIR / self.output_dir


# ---------------------------------------------------------------------------
# Registry (built once from the download_data tool's declarations)
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, DatasetEntry]:
    reg: dict[str, DatasetEntry] = {}
    for name, info in BENCHMARK_DATASETS.items():
        reg[name] = DatasetEntry(
            name=name,
            category="benchmark",
            output_dir=info["output_dir"],
            description=info.get("description", ""),
            repo_id=info.get("repo_id"),
            config=info.get("config"),
            declared_splits=list(info.get("splits", {}).keys()),
            note=info.get("note"),
        )
    for name, info in TRAINING_DATASETS.items():
        split = info.get("split")
        reg[name] = DatasetEntry(
            name=name,
            category="training",
            output_dir=info["output_dir"],
            description=info.get("description", ""),
            repo_id=info.get("repo_id"),
            url=info.get("url"),
            config=info.get("config"),
            declared_splits=[split] if split else [],
            declared_size_mb=info.get("size_mb"),
            note=info.get("note"),
        )
    return reg


REGISTRY: dict[str, DatasetEntry] = _build_registry()


def registry() -> dict[str, DatasetEntry]:
    """Return the full dataset registry (name -> :class:`DatasetEntry`)."""
    return REGISTRY


def data_dir() -> Path:
    """Return the absolute path to the repo-root ``data/`` directory."""
    return DATA_DIR


def get(name: str) -> DatasetEntry:
    """Look up a dataset entry by name, raising ``KeyError`` if unknown."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset '{name}'. Known datasets: {', '.join(sorted(REGISTRY))}"
        ) from None


def local_path(name: str) -> Path:
    """Return the resolved on-disk data root for a dataset (after descent)."""
    return _resolve_data_root(get(name).path)


def is_available(name: str) -> bool:
    """True if the dataset's data is present and recognisable on disk."""
    return detect_layout(get(name).path) not in (Layout.MISSING, Layout.UNKNOWN)


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------


def _has_markers(d: Path) -> bool:
    """True if ``d`` itself looks like a loadable dataset root."""
    if (d / "dataset_info.json").exists():
        return True
    if (d / "images").is_dir():
        return True
    if any(d.glob("*.parquet")):
        return True
    if (d / "data").is_dir() and any((d / "data").glob("*.parquet")):
        return True
    if any(j.name not in _META_JSON for j in d.glob("*.json")):
        return True
    if any((sd / "dataset_info.json").exists() for sd in d.iterdir() if sd.is_dir()):
        return True
    return False


def _resolve_data_root(path: Path) -> Path:
    """Descend through redundant single-subdir nesting (e.g. ``coco8/coco8``,
    ``ocrbench/OCRBench_v2``) until the real dataset root is reached."""
    cur = path
    for _ in range(4):
        if not cur.is_dir():
            return cur
        if _has_markers(cur):
            return cur
        meaningful = [e for e in cur.iterdir() if e.name not in _IGNORE_NAMES]
        dirs = [e for e in meaningful if e.is_dir()]
        files = [e for e in meaningful if e.is_file()]
        # Only descend when there is exactly one candidate subdir and no
        # competing data files at this level.
        if len(dirs) == 1 and not files:
            cur = dirs[0]
            continue
        return cur
    return cur


def detect_layout(path: Path) -> Layout:
    """Classify the on-disk layout of a dataset directory."""
    if not path.exists():
        return Layout.MISSING
    root = _resolve_data_root(path)
    if not root.is_dir():
        return Layout.UNKNOWN
    if (root / "dataset_info.json").exists():
        return Layout.HF_DISK
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if subdirs and any((d / "dataset_info.json").exists() for d in subdirs):
        return Layout.HF_DISK_SPLITS
    if any(root.glob("*.parquet")) or (
        (root / "data").is_dir() and any((root / "data").glob("*.parquet"))
    ):
        return Layout.PARQUET
    if (root / "images").is_dir():
        return Layout.IMAGE_TREE
    if any(j.name not in _META_JSON for j in root.glob("*.json")):
        return Layout.JSON_INDEX
    return Layout.UNKNOWN


# ---------------------------------------------------------------------------
# Split discovery
# ---------------------------------------------------------------------------


def _disk_splits(root: Path) -> list[str]:
    """Split subdirectory names for an ``HF_DISK_SPLITS`` root."""
    return sorted(
        d.name for d in root.iterdir() if d.is_dir() and (d / "dataset_info.json").exists()
    )


def _parquet_groups(root: Path) -> dict[str, list[Path]]:
    """Group parquet shards by split prefix (text before the first '-')."""
    pq_dir = root / "data" if (root / "data").is_dir() else root
    groups: dict[str, list[Path]] = {}
    for f in sorted(pq_dir.glob("*.parquet")):
        split = f.stem.split("-", 1)[0]
        groups.setdefault(split, []).append(f)
    return groups


def available_splits(name: str) -> list[str]:
    """Return the splits that are actually present on disk for a dataset."""
    entry = get(name)
    root = _resolve_data_root(entry.path)
    layout = detect_layout(entry.path)
    if layout == Layout.HF_DISK_SPLITS:
        return _disk_splits(root)
    if layout == Layout.PARQUET:
        return sorted(_parquet_groups(root).keys())
    if layout == Layout.IMAGE_TREE:
        img = root / "images"
        sub = [d.name for d in img.iterdir() if d.is_dir()] if img.is_dir() else []
        return sorted(sub) if sub else ["all"]
    if layout in (Layout.HF_DISK, Layout.JSON_INDEX):
        return ["train"] if layout == Layout.HF_DISK else ["all"]
    return []


def _default_split(name: str, splits: list[str]) -> str:
    """Pick a sensible default split, preferring the registry's declared one."""
    entry = get(name)
    for declared in entry.declared_splits:
        if declared in splits:
            return declared
    # Common preference order for benchmarks.
    for pref in ("testmini", "test", "validation", "val", "dev", "train", "all"):
        if pref in splits:
            return pref
    return splits[0] if splits else "train"


# ---------------------------------------------------------------------------
# Non-HF dataset wrappers
# ---------------------------------------------------------------------------


@dataclass
class ImageTree:
    """A coco-style ``images/{split}/*.jpg`` tree with optional YOLO labels."""

    root: Path
    split: str
    images: list[Path]
    labels_dir: Path | None

    @classmethod
    def from_dir(cls, root: Path, split: str | None = None) -> ImageTree:
        img_root = root / "images"
        subdirs = [d for d in img_root.iterdir() if d.is_dir()] if img_root.is_dir() else []
        if subdirs:
            names = {d.name: d for d in subdirs}
            chosen = split or ("train" if "train" in names else sorted(names)[0])
            if chosen not in names:
                raise ValueError(f"Split '{chosen}' not found; available: {sorted(names)}")
            img_dir = names[chosen]
        else:
            chosen = "all"
            img_dir = img_root
        images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
        labels_dir = root / "labels"
        return cls(
            root=root,
            split=chosen,
            images=images,
            labels_dir=labels_dir if labels_dir.is_dir() else None,
        )

    def __len__(self) -> int:
        return len(self.images)

    def label_for(self, image_path: Path) -> Path | None:
        if not self.labels_dir:
            return None
        rel = image_path.relative_to(self.root / "images")
        cand = (self.labels_dir / rel).with_suffix(".txt")
        return cand if cand.exists() else None


@dataclass
class JsonIndex:
    """A JSON list of records that reference image files by relative path."""

    root: Path
    json_path: Path
    records: list[dict[str, Any]]

    @classmethod
    def from_dir(cls, root: Path) -> JsonIndex:
        candidates = [j for j in root.glob("*.json") if j.name not in _META_JSON]
        if not candidates:
            raise FileNotFoundError(f"No index JSON found under {root}")
        json_path = candidates[0]
        with open(json_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Some indexes wrap the list under a top-level key.
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                data = [data]
        return cls(root=root, json_path=json_path, records=list(data))

    def __len__(self) -> int:
        return len(self.records)

    def resolve_image(self, record: dict[str, Any]) -> Path | None:
        for key in ("image_path", "image", "img_path", "file_name", "filename"):
            if key in record and isinstance(record[key], str):
                cand = self.root / record[key]
                return cand if cand.exists() else cand
        return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(name: str, split: str | None = None, streaming: bool = False) -> Any:
    """Load a dataset from disk, returning the natural object for its layout.

    Returns a ``datasets.Dataset`` for HF / parquet layouts, an
    :class:`ImageTree` for image folders, or a :class:`JsonIndex` for JSON
    record indexes. Raises ``FileNotFoundError`` if the dataset is not present.
    """
    entry = get(name)
    layout = detect_layout(entry.path)
    root = _resolve_data_root(entry.path)

    if layout == Layout.MISSING:
        raise FileNotFoundError(
            f"Dataset '{name}' is not present at {entry.path}. "
            f"Download it with: python -m smlx.tools.download_data --dataset {name}"
        )
    if layout == Layout.UNKNOWN:
        raise ValueError(f"Could not recognise the on-disk layout of '{name}' at {root}")

    if layout == Layout.HF_DISK:
        from datasets import load_from_disk

        return load_from_disk(str(root))

    if layout == Layout.HF_DISK_SPLITS:
        from datasets import load_from_disk

        splits = _disk_splits(root)
        chosen = split or _default_split(name, splits)
        if chosen not in splits:
            raise ValueError(f"Split '{chosen}' not found for '{name}'; have {splits}")
        return load_from_disk(str(root / chosen))

    if layout == Layout.PARQUET:
        from datasets import load_dataset

        groups = _parquet_groups(root)
        chosen = split or _default_split(name, sorted(groups))
        if chosen not in groups:
            raise ValueError(f"Split '{chosen}' not found for '{name}'; have {sorted(groups)}")
        return load_dataset(
            "parquet",
            data_files=[str(f) for f in groups[chosen]],
            split="train",
            streaming=streaming,
        )

    if layout == Layout.IMAGE_TREE:
        return ImageTree.from_dir(root, split=split)

    if layout == Layout.JSON_INDEX:
        return JsonIndex.from_dir(root)

    raise ValueError(f"Unhandled layout {layout} for '{name}'")


# ---------------------------------------------------------------------------
# Inspection / preview
# ---------------------------------------------------------------------------


def _describe_value(value: Any, max_str: int = 80) -> str:
    """Render a single field value compactly for preview output."""
    # HF audio column: {"array": np.ndarray, "sampling_rate": int}
    if isinstance(value, dict) and "array" in value and "sampling_rate" in value:
        arr = value["array"]
        sr = value["sampling_rate"]
        try:
            n = len(arr)
            return f"<audio {n / sr:.2f}s @ {sr}Hz, {n} samples>"
        except TypeError:
            return f"<audio @ {sr}Hz>"
    # PIL image
    if value.__class__.__name__ == "Image" or hasattr(value, "size") and hasattr(value, "mode"):
        try:
            return f"<image {value.size[0]}x{value.size[1]} {value.mode}>"
        except Exception:
            return "<image>"
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, str):
        s = value.replace("\n", " ⏎ ")
        return s if len(s) <= max_str else s[: max_str - 1] + "…"
    if isinstance(value, (list, tuple)):
        return f"<{type(value).__name__} len={len(value)}>"
    return repr(value)


def iter_samples(name: str, split: str | None = None, limit: int = 5) -> Iterator[dict[str, str]]:
    """Yield up to ``limit`` normalised, preview-ready sample dicts.

    Works across every layout: each yielded dict maps field name -> a compact
    string description of that field's value, suitable for printing.
    """
    yield from _iter_obj(load(name, split=split), limit)


def _iter_obj(obj: Any, limit: int) -> Iterator[dict[str, str]]:
    """Yield normalised preview dicts from an already-loaded dataset object."""
    if isinstance(obj, ImageTree):
        for img_path in obj.images[:limit]:
            row: dict[str, str] = {
                "image": str(img_path.relative_to(obj.root)),
            }
            try:
                from PIL import Image

                with Image.open(img_path) as im:
                    row["size"] = f"{im.size[0]}x{im.size[1]} {im.mode}"
            except Exception as exc:  # pragma: no cover - corrupt image edge case
                row["size"] = f"<unreadable: {exc}>"
            label = obj.label_for(img_path)
            if label is not None:
                row["label_file"] = str(label.relative_to(obj.root))
            yield row
        return

    if isinstance(obj, JsonIndex):
        for rec in obj.records[:limit]:
            yield {k: _describe_value(v) for k, v in rec.items()}
        return

    # HF datasets.Dataset (map-style) or IterableDataset (streaming)
    count = 0
    for row in obj:
        yield {k: _describe_value(v) for k, v in row.items()}
        count += 1
        if count >= limit:
            break


# ---------------------------------------------------------------------------
# Probing / inventory (lightweight -- never materialises a full dataset)
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """A lightweight health report for one registered dataset."""

    name: str
    category: str
    output_dir: str
    available: bool
    layout: Layout
    resolved_path: str | None = None
    splits: list[str] = field(default_factory=list)
    num_shards: int = 0
    num_examples: int | None = None
    fields: list[str] = field(default_factory=list)
    size_bytes: int = 0
    sample_ok: bool | None = None
    error: str | None = None


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def probe(name: str, compute_size: bool = True) -> ProbeResult:
    """Lightweight health probe for a dataset: existence, layout, shard count,
    declared field/example counts, and a single-sample load check. Never loads
    a full dataset into memory."""
    entry = get(name)
    layout = detect_layout(entry.path)
    result = ProbeResult(
        name=name,
        category=entry.category,
        output_dir=entry.output_dir,
        available=layout not in (Layout.MISSING, Layout.UNKNOWN),
        layout=layout,
    )
    if layout == Layout.MISSING:
        return result

    root = _resolve_data_root(entry.path)
    result.resolved_path = str(root.relative_to(DATA_DIR))
    if compute_size and root.exists():
        result.size_bytes = _dir_size(root)

    try:
        result.splits = available_splits(name)
    except Exception as exc:  # pragma: no cover - defensive
        result.error = f"split discovery: {exc}"

    # Shard counts (cheap, file-listing only). Reflect the default split for
    # split-style layouts so the count matches the reported example total.
    if layout == Layout.HF_DISK:
        result.num_shards = len(list(root.glob("*.arrow")))
    elif layout == Layout.HF_DISK_SPLITS:
        default = _default_split(name, result.splits) if result.splits else None
        if default:
            result.num_shards = len(list((root / default).glob("*.arrow")))
    elif layout == Layout.PARQUET:
        default = _default_split(name, result.splits) if result.splits else None
        groups = _parquet_groups(root)
        result.num_shards = len(groups.get(default, [])) if default else 0

    # Single load of the default split: gives the *actual* on-disk example
    # count (len() is O(1) for memory-mapped HF / Arrow datasets) and the real
    # field names -- more honest than the (often stale) dataset_info splits dict.
    try:
        obj = load(name)
        try:
            result.num_examples = len(obj)
        except TypeError:
            result.num_examples = None
        sample = next(_iter_obj(obj, 1), None)
        result.sample_ok = sample is not None
        if sample:
            result.fields = sorted(sample.keys())
    except Exception as exc:
        result.sample_ok = False
        result.error = (result.error + "; " if result.error else "") + f"load: {exc}"

    return result


def inventory(compute_size: bool = True) -> list[ProbeResult]:
    """Probe every registered dataset and return the results sorted by
    category then name. Present datasets first within each category."""
    results = [probe(name, compute_size=compute_size) for name in REGISTRY]
    results.sort(key=lambda r: (r.category, not r.available, r.name))
    return results


def find_orphans() -> list[str]:
    """Scan ``data/`` for dataset directories (``dataset_info.json`` or an
    ``images/`` tree) whose path is not covered by any registry entry.

    This surfaces drift between the registry and what is actually on disk --
    e.g. a full dataset downloaded to a different folder than the registry
    declares.
    """
    if not DATA_DIR.is_dir():
        return []
    registered_roots = {
        str(_resolve_data_root(e.path).resolve()) for e in REGISTRY.values() if e.path.exists()
    }
    orphans: list[str] = []
    seen: set[str] = set()
    for info in DATA_DIR.rglob("dataset_info.json"):
        root = info.parent
        # The split subdir of a registered HF_DISK_SPLITS parent is not orphan.
        resolved = str(root.resolve())
        parent_resolved = str(root.parent.resolve())
        if resolved in registered_roots or parent_resolved in registered_roots:
            continue
        # Avoid double-reporting nested splits of the same orphan parent.
        if any(resolved.startswith(s + "/") for s in seen):
            continue
        rel = root.relative_to(DATA_DIR)
        orphans.append(str(rel))
        seen.add(resolved)
    return sorted(orphans)
