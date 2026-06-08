"""Deterministic text -> CAD generation.

The smolGenCad neural model has no public trained checkpoint (it emits random
output). This module is a **real, correct** alternative: it parses a natural-
language CAD spec into a valid CadQuery program and solid for the common
parametric primitives. It is rule-based, not ML — but the output is genuine
correct CAD (verified by executing the emitted CadQuery), not a placeholder.

Supported primitives: cylinder, box/cube, sphere, cone — with dimensions in
mm/cm/m/in and an optional fillet.

    >>> from smlx.models.smolGenCad.text_to_cad import generate
    >>> r = generate("a cylinder with radius 5mm and height 10mm")
    >>> r["primitive"], r["params"], r["bbox"]
    ('cylinder', {'radius': 5.0, 'height': 10.0}, (10.0, 10.0, 10.0))
"""

from __future__ import annotations

import re

# Length units -> millimetres.
_UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, '"': 25.4, "inch": 25.4}
_NUM = r"([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|m|inch|in|\")?"


def _to_mm(value: str, unit: str | None) -> float:
    return float(value) * _UNITS.get((unit or "mm").lower(), 1.0)


def _after(text: str, *keys: str) -> float | None:
    """First number following any keyword, e.g. 'radius 5mm' / 'r=5' / 'height: 10'."""
    for k in keys:
        m = re.search(rf"\b{k}\b\s*[:=]?\s*{_NUM}", text)
        if m:
            return _to_mm(m.group(1), m.group(2))
    return None


def _all_numbers(text: str) -> list[float]:
    return [_to_mm(v, u) for v, u in re.findall(_NUM, text)]


class CADParseError(ValueError):
    """Raised when a text spec cannot be parsed into a supported primitive."""


def parse_spec(text: str) -> dict:
    """Parse a text CAD spec into {'primitive': str, 'params': dict}."""
    t = text.lower().strip()
    nums = _all_numbers(t)

    def has(*keys: str) -> bool:
        return any(re.search(rf"\b{k}\b", t) for k in keys)

    def radius() -> float | None:
        """Radius, keyword-first and diameter-aware (never grabs a diameter as r)."""
        r = _after(t, "radius", "r")
        if r is not None:
            return r
        d = _after(t, "diameter", "dia")
        if d is not None:
            return d / 2
        # Positional fallback only when NO radius/diameter keyword is present.
        if not has("radius", "diameter", "dia", "r") and nums:
            return nums[0]
        return None

    def height() -> float | None:
        return _after(t, "height", "tall", "long", "length", "thickness", "h")

    # --- sphere -----------------------------------------------------------
    if "sphere" in t or "ball" in t:
        r = radius()
        if r is None:
            raise CADParseError("sphere needs a radius or diameter")
        return {"primitive": "sphere", "params": {"radius": r}}

    # --- cone -------------------------------------------------------------
    if "cone" in t or "tapered" in t or "frustum" in t:
        r1 = radius()
        r2 = _after(t, "top radius", "top") or 0.0
        h = height()
        if h is None and len(nums) >= 2:
            h = nums[1]
        if r1 is None or h is None:
            raise CADParseError("cone needs a (base) radius and a height")
        return {"primitive": "cone", "params": {"radius": r1, "top_radius": r2, "height": h}}

    # --- cylinder / rod / tube / disc ------------------------------------
    if any(w in t for w in ("cylinder", "cylindrical", "rod", "tube", "pipe", "disc", "disk")):
        r = radius()
        h = height()
        if h is None:
            # A leftover positional number that isn't the radius/diameter value.
            d = _after(t, "diameter", "dia")
            used = {x for x in (r, d, _after(t, "radius", "r")) if x is not None}
            leftover = [n for n in nums if n not in used]
            h = leftover[0] if leftover else (nums[1] if len(nums) > 1 else None)
        if r is None or h is None:
            raise CADParseError("cylinder needs a radius (or diameter) and a height")
        return {"primitive": "cylinder", "params": {"radius": r, "height": h}}

    # --- cube / box / block ----------------------------------------------
    if any(w in t for w in ("cube", "box", "block", "cuboid", "rectangular")):
        side = _after(t, "side", "size", "edge")
        if "cube" in t:
            if side is None:
                side = nums[0] if nums else None
            if side is None:
                raise CADParseError("cube needs a side length")
            return {"primitive": "box", "params": {"width": side, "depth": side, "height": side}}
        w = _after(t, "width", "w")
        d = _after(t, "depth", "length", "long", "d")
        h = _after(t, "height", "tall", "h")
        # Bare "box 10 20 30": fill missing dims positionally.
        if (w, d, h) == (None, None, None) and len(nums) >= 3:
            w, d, h = nums[0], nums[1], nums[2]
        if None in (w, d, h):
            raise CADParseError("box needs width, depth and height (or a cube 'side')")
        return {"primitive": "box", "params": {"width": w, "depth": d, "height": h}}

    raise CADParseError(
        f"unsupported CAD spec: {text!r} (supported: cylinder, box/cube, sphere, cone)"
    )


def _fillet_radius(text: str) -> float | None:
    if "fillet" in text.lower() or "rounded" in text.lower():
        return _after(text.lower(), "fillet", "radius", "round") or 1.0
    return None


def to_cadquery_code(spec: dict, fillet: float | None = None) -> str:
    """Emit a valid CadQuery program for a parsed spec."""
    prim, p = spec["primitive"], spec["params"]
    lines = ["import cadquery as cq", ""]
    if prim == "cylinder":
        lines.append(f"result = cq.Workplane('XY').cylinder({p['height']}, {p['radius']})")
    elif prim == "box":
        lines.append(f"result = cq.Workplane('XY').box({p['width']}, {p['depth']}, {p['height']})")
    elif prim == "sphere":
        lines.append(f"result = cq.Workplane('XY').sphere({p['radius']})")
    elif prim == "cone":
        # A (frustum) cone via a revolved profile / built-in solid.
        lines.append(
            "result = cq.Workplane('XY').add("
            f"cq.Solid.makeCone({p['radius']}, {p['top_radius']}, {p['height']}))"
        )
    else:  # pragma: no cover - parse_spec only emits the above
        raise CADParseError(f"no CadQuery emitter for {prim!r}")
    if fillet:
        lines.append(f"result = result.edges().fillet({fillet})")
    return "\n".join(lines)


def build_solid(code: str):
    """Execute the emitted CadQuery code and return the `result` object."""
    import cadquery as cq  # noqa: F401  (used by the exec'd code)

    ns: dict = {}
    exec(code, {"cq": cq}, ns)  # noqa: S102 - code is generated by to_cadquery_code
    return ns["result"]


def generate(text: str, validate: bool = True) -> dict:
    """Parse a text CAD spec and produce a real CadQuery program (+ bbox).

    Returns a dict: primitive, params, python (CadQuery code), and — when
    ``validate`` and cadquery is installed — the executed solid's bounding box.
    """
    spec = parse_spec(text)
    fillet = _fillet_radius(text)
    code = to_cadquery_code(spec, fillet=fillet)
    out = {"primitive": spec["primitive"], "params": spec["params"], "python": code, "bbox": None}
    if validate:
        bb = build_solid(code).val().BoundingBox()
        out["bbox"] = (round(bb.xlen, 3), round(bb.ylen, 3), round(bb.zlen, 3))
    return out
