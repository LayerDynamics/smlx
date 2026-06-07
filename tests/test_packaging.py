"""Packaging regression tests (WS-1).

Guards the install story that a fresh `pip install smlx` must satisfy:
- runtime deps that are imported at module-import time live in CORE
  dependencies, not optional extras (a clean install of `smlx.models` must not
  raise ModuleNotFoundError);
- the `smlx` console script is declared;
- the PEP 561 `py.typed` marker exists so it ships in the wheel.

Regression context: `smlx_manager.py` imports `psutil` at module top, and core
loaders use `safetensors`; both were previously only in `[dev]`/`[tools]`, so a
clean install crashed on `import smlx.models`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - py<3.11
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None  # type: ignore

pytestmark = pytest.mark.skipif(tomllib is None, reason="needs tomllib (py>=3.11) or tomli")

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _pyproject() -> dict:
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


@pytest.mark.unit
def test_import_time_deps_are_core():
    """Deps imported at module-import time must be in core dependencies."""
    core = _pyproject()["project"]["dependencies"]
    names = {re_split(d) for d in core}
    for required in ("psutil", "safetensors"):
        assert required in names, (
            f"{required!r} is imported at import time but missing from core "
            f"[project.dependencies]; a clean `pip install smlx` would crash. Core: {sorted(names)}"
        )


@pytest.mark.unit
def test_console_script_declared():
    scripts = _pyproject()["project"].get("scripts", {})
    assert scripts.get("smlx") == "smlx.main:main", f"smlx console script missing/wrong: {scripts}"


@pytest.mark.unit
def test_py_typed_marker_exists():
    assert (REPO_ROOT / "smlx" / "py.typed").is_file(), "smlx/py.typed missing — types won't ship"


@pytest.mark.unit
def test_models_package_imports_with_core_only():
    """smlx.models must import (it pulls psutil/safetensors transitively)."""
    import importlib

    importlib.import_module("smlx.models")
    assert "psutil" in sys.modules  # proves the import-time dep is satisfiable


def re_split(dep: str) -> str:
    """Extract the distribution name from a PEP 508 dependency string."""
    import re

    return re.split(r"[<>=!~ \[]", dep.strip(), maxsplit=1)[0].lower()
