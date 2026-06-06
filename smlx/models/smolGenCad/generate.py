#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
CAD generation interface.

High-level API for generating CAD sequences from text descriptions.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .cache import make_cache
from .commands import CADCommandType
from .model import SmolGenCad
from .tokenizer import CADTokenizer
from .validator import auto_fix_sequence, validate_sequence

# Canonical CadQuery sketch planes (matches the SKETCH_START parameter spec in
# commands.py). Used to guard the plane string that is interpolated into emitted
# CadQuery code in sequence_to_python().
_VALID_SKETCH_PLANES = frozenset({"XY", "XZ", "YZ"})


def generate(
    model: SmolGenCad,
    text_tokenizer: Any,  # HuggingFace tokenizer for text
    cad_tokenizer: CADTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int | None = 50,
    auto_fix: bool = True,
    validate: bool = True,
) -> list[tuple[CADCommandType, dict[str, Any]]]:
    """
    Generate CAD sequence from text description.

    Args:
        model: SmolGenCad model
        text_tokenizer: Tokenizer for input text (HuggingFace tokenizer)
        cad_tokenizer: Tokenizer for CAD sequences
        prompt: Natural language description of CAD model
        max_new_tokens: Maximum CAD tokens to generate
        temperature: Sampling temperature (0 = greedy, higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling
        auto_fix: Automatically fix common sequence errors
        validate: Validate generated sequence

    Returns:
        List of (command, parameters) tuples

    Example:
        >>> from smlx.models.smolGenCad import load, generate
        >>> model, text_tokenizer, cad_tokenizer = load()
        >>> cad_sequence = generate(
        ...     model,
        ...     text_tokenizer,
        ...     cad_tokenizer,
        ...     prompt="Create a cylinder with radius 5cm and height 10cm",
        ...     temperature=0.7
        ... )
        >>> print(f"Generated {len(cad_sequence)} CAD commands")
    """
    # Tokenize text input
    text_inputs = text_tokenizer(
        prompt,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512,
    )
    text_ids = mx.array(text_inputs["input_ids"])

    # Encode text
    encoder_outputs = model.encode(text_ids)

    # Initialize CAD sequence with BOS token
    cad_ids = mx.array([[cad_tokenizer.bos_token_id]], dtype=mx.int32)

    # Create cache
    cache = make_cache(model)

    # Generate tokens autoregressively
    generated_tokens = []

    for _ in range(max_new_tokens):
        # Generate next token
        next_token, cache = model.generate_step(
            cad_ids,
            encoder_outputs,
            cache=cache,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Convert to list for appending
        next_token = int(next_token.item())
        generated_tokens.append(next_token)

        # Check for EOS
        if next_token == cad_tokenizer.eos_token_id:
            break

        # Append to sequence
        cad_ids = mx.concatenate([cad_ids, mx.array([[next_token]], dtype=mx.int32)], axis=1)

    # Decode tokens to CAD sequence
    cad_sequence = cad_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Auto-fix if requested
    if auto_fix:
        cad_sequence = auto_fix_sequence(cad_sequence)

    # Validate if requested
    if validate:
        is_valid, errors = validate_sequence(cad_sequence)
        if not is_valid:
            print(f"Warning: Generated sequence has validation errors: {errors}")

    return cad_sequence


def generate_batch(
    model: SmolGenCad,
    text_tokenizer: Any,
    cad_tokenizer: CADTokenizer,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int | None = 50,
    auto_fix: bool = True,
    validate: bool = True,
) -> list[list[tuple[CADCommandType, dict[str, Any]]]]:
    """
    Generate CAD sequences for multiple prompts.

    Args:
        model: SmolGenCad model
        text_tokenizer: Tokenizer for input text
        cad_tokenizer: Tokenizer for CAD sequences
        prompts: List of text descriptions
        max_new_tokens: Maximum tokens per sequence
        temperature: Sampling temperature
        top_p: Nucleus sampling
        top_k: Top-k sampling

    Returns:
        List of CAD sequences (one per prompt)

    Example:
        >>> prompts = [
        ...     "Create a cube with side 10cm",
        ...     "Create a cylinder with radius 5cm and height 15cm"
        ... ]
        >>> sequences = generate_batch(model, text_tokenizer, cad_tokenizer, prompts)
    """
    sequences = []
    for prompt in prompts:
        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            auto_fix=auto_fix,
            validate=validate,
        )
        sequences.append(sequence)

    return sequences


def sequence_to_dict(sequence: list[tuple[CADCommandType, dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Convert CAD sequence to dictionary format for serialization.

    Args:
        sequence: CAD sequence

    Returns:
        List of dictionaries with 'command' and 'parameters' keys

    Example:
        >>> sequence = [(CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 50})]
        >>> dict_seq = sequence_to_dict(sequence)
        >>> print(dict_seq)
        [{'command': 'CIRCLE', 'parameters': {'cx': 0, 'cy': 0, 'r': 50}}]
    """
    return [{"command": command.name, "parameters": parameters} for command, parameters in sequence]


def sequence_to_json(sequence: list[tuple[CADCommandType, dict[str, Any]]]) -> str:
    """
    Convert CAD sequence to JSON string.

    Args:
        sequence: CAD sequence

    Returns:
        JSON string representation

    Example:
        >>> import json
        >>> sequence = [(CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 50})]
        >>> json_str = sequence_to_json(sequence)
        >>> parsed = json.loads(json_str)
    """
    import json

    return json.dumps(sequence_to_dict(sequence), indent=2)


def sequence_to_python(sequence: list[tuple[CADCommandType, dict[str, Any]]]) -> str:
    """
    Convert CAD sequence to Python code representation.

    Generates executable Python code using CadQuery library.

    Args:
        sequence: CAD sequence

    Returns:
        Python code string

    Example:
        >>> sequence = [
        ...     (CADCommandType.SKETCH_START, {"plane": "XY"}),
        ...     (CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 50}),
        ...     (CADCommandType.SKETCH_END, {}),
        ...     (CADCommandType.EXTRUDE, {"distance": 100}),
        ... ]
        >>> code = sequence_to_python(sequence)
        >>> print(code)
    """
    lines = ["import cadquery as cq", "", "result = cq.Workplane('XY')"]

    in_sketch = False
    sketch_commands: list[str] = []  # geometry drawn in the currently-open sketch

    for command, params in sequence:
        if command == CADCommandType.SKETCH_START:
            in_sketch = True
            sketch_commands = []
            plane = params.get("plane", "XY")
            # The plane string is interpolated verbatim into emitted code, so an
            # out-of-spec value (or a stray quote) would produce broken Python.
            # Restrict to the canonical CadQuery sketch planes (see commands.py),
            # falling back to XY and flagging the substitution.
            if plane not in _VALID_SKETCH_PLANES:
                lines.append(f"# warning: invalid sketch plane {plane!r}, defaulting to 'XY'")
                plane = "XY"
            lines.append(f"result = result.workplane('{plane}')")

        elif command == CADCommandType.SKETCH_END:
            # Document the closed sketch's contents; flag empty sketches since a
            # following extrude would then be a no-op.
            summary = ", ".join(sketch_commands) if sketch_commands else "no geometry"
            lines.append(f"# sketch closed ({summary})")
            in_sketch = False
            sketch_commands = []

        elif command == CADCommandType.CIRCLE:
            cx, cy, r = params.get("cx", 0), params.get("cy", 0), params.get("r", 1)
            if not in_sketch:
                lines.append("# warning: circle drawn outside a sketch")
            # Place the circle at its center — a bare .circle(r) draws at the
            # workplane origin and ignores cx/cy.
            lines.append(f"result = result.moveTo({cx}, {cy}).circle({r})")
            sketch_commands.append(f"circle(r={r}) @ ({cx}, {cy})")

        elif command == CADCommandType.RECTANGLE:
            cx, cy = params.get("cx", 0), params.get("cy", 0)
            w, h = params.get("width", 1), params.get("height", 1)
            if not in_sketch:
                lines.append("# warning: rectangle drawn outside a sketch")
            # Place the rectangle at its center (.rect is centered on the point).
            lines.append(f"result = result.moveTo({cx}, {cy}).rect({w}, {h})")
            sketch_commands.append(f"rect({w}x{h}) @ ({cx}, {cy})")

        elif command == CADCommandType.EXTRUDE:
            distance = params.get("distance", 1)
            lines.append(f"result = result.extrude({distance})")

        elif command == CADCommandType.FILLET:
            radius = params.get("radius", 1)
            lines.append(f"result = result.edges().fillet({radius})")

        elif command == CADCommandType.CHAMFER:
            distance = params.get("distance", 1)
            lines.append(f"result = result.edges().chamfer({distance})")

    return "\n".join(lines)


def generate_and_export(
    model: SmolGenCad,
    text_tokenizer: Any,
    cad_tokenizer: CADTokenizer,
    prompt: str,
    output_format: str = "json",
    **generate_kwargs,
) -> str:
    """
    Generate CAD sequence and export to specified format.

    Args:
        model: SmolGenCad model
        text_tokenizer: Text tokenizer
        cad_tokenizer: CAD tokenizer
        prompt: Text description
        output_format: Output format ('json', 'python', 'dict')
        **generate_kwargs: Additional arguments for generate()

    Returns:
        Formatted output string

    Example:
        >>> output = generate_and_export(
        ...     model, text_tokenizer, cad_tokenizer,
        ...     "Create a cylinder",
        ...     output_format="python"
        ... )
    """
    # Generate sequence
    sequence = generate(model, text_tokenizer, cad_tokenizer, prompt, **generate_kwargs)

    # Export to format
    if output_format == "json":
        return sequence_to_json(sequence)
    elif output_format == "python":
        return sequence_to_python(sequence)
    elif output_format == "dict":
        return str(sequence_to_dict(sequence))
    else:
        raise ValueError(f"Unknown output format: {output_format}. Use 'json', 'python', or 'dict'")
