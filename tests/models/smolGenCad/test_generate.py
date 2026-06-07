#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""End-to-end generation tests for smolGenCad.

These verify that the text->CAD generation *pipeline produces output* — it runs
front to back without error and emits a structurally-valid CAD sequence plus
well-formed JSON/Python exports. This is deliberately distinct from a
trained-quality check: smolGenCad currently ships with random-initialised
weights (no public checkpoint), so the *content* of the CAD is not meaningful,
but the generation path itself must work and yield real, well-formed output —
never a crash, an empty/None result, or a placeholder.

If a trained checkpoint is added later, layer a semantic-correctness test on top;
these output-happens guarantees should keep passing unchanged.
"""

from __future__ import annotations

import ast
import json

import pytest

from smlx.models.smolGenCad import generate, load
from smlx.models.smolGenCad.commands import CADCommandType
from smlx.models.smolGenCad.generate import (
    sequence_to_dict,
    sequence_to_json,
    sequence_to_python,
)

# load() pulls the SmolLM2 text tokenizer from the HF Hub (cached after first
# run); the model weights themselves are random-initialised locally.
pytestmark = [pytest.mark.requires_model]


@pytest.fixture(scope="module")
def cad_model():
    """Load the model + both tokenizers once for the module."""
    model, text_tok, cad_tok = load()
    return model, text_tok, cad_tok


def test_generation_pipeline_produces_output(cad_model):
    """generate() runs end to end and returns a real, well-formed CAD sequence."""
    model, text_tok, cad_tok = cad_model

    sequence = generate(
        model,
        text_tok,
        cad_tok,
        prompt="Create a cylinder with radius 5cm and height 10cm",
        max_new_tokens=64,
        temperature=0.0,  # greedy: deterministic, pure pipeline check
    )

    # Output happened: a real list (not None, not an exception) of well-typed
    # (command, params) tuples.
    assert isinstance(sequence, list)
    for item in sequence:
        assert isinstance(item, tuple) and len(item) == 2
        command, params = item
        assert isinstance(command, CADCommandType)
        assert isinstance(params, dict)


def test_python_export_is_valid_code(cad_model):
    """The generated sequence exports to syntactically-valid Python (compiles)."""
    model, text_tok, cad_tok = cad_model
    sequence = generate(
        model, text_tok, cad_tok, prompt="Create a 10mm cube", max_new_tokens=64, temperature=0.0
    )

    code = sequence_to_python(sequence)
    assert isinstance(code, str) and code.strip()
    assert "import cadquery as cq" in code
    # Must parse as real Python — guards against malformed interpolation
    # (e.g. an out-of-spec sketch plane breaking the emitted source).
    ast.parse(code)


def test_json_export_is_valid(cad_model):
    """The generated sequence exports to parseable JSON with the right schema."""
    model, text_tok, cad_tok = cad_model
    sequence = generate(
        model, text_tok, cad_tok, prompt="Draw a circle", max_new_tokens=64, temperature=0.0
    )

    payload = json.loads(sequence_to_json(sequence))
    assert isinstance(payload, list)
    for entry in payload:
        assert set(entry.keys()) == {"command", "parameters"}
        assert isinstance(entry["command"], str)
        assert isinstance(entry["parameters"], dict)
    # dict export agrees with the json export
    assert sequence_to_dict(sequence) == payload
