#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
CAD sequence validator.

Validates CAD command sequences for semantic correctness and geometric validity.
"""

from __future__ import annotations

from typing import Any

from .commands import (
    SEQUENCE_RULES,
    CADCommandType,
    validate_parameters,
)


class ValidationError(Exception):
    """Exception raised when CAD sequence validation fails."""

    pass


class CADSequenceValidator:
    """
    Validator for CAD command sequences.

    Checks semantic correctness, parameter validity, and geometric constraints.

    Example:
        >>> validator = CADSequenceValidator()
        >>> sequence = [
        ...     (CADCommandType.SKETCH_START, {"plane": "XY"}),
        ...     (CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 50}),
        ...     (CADCommandType.SKETCH_END, {}),
        ...     (CADCommandType.EXTRUDE, {"distance": 100}),
        ... ]
        >>> is_valid, errors = validator.validate(sequence)
        >>> if not is_valid:
        ...     print(f"Validation errors: {errors}")
    """

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, raise exceptions on validation failures
        """
        self.strict = strict

    def validate(
        self, sequence: list[tuple[CADCommandType, dict[str, Any]]]
    ) -> tuple[bool, list[str]]:
        """
        Validate complete CAD sequence.

        Args:
            sequence: List of (command, parameters) tuples

        Returns:
            Tuple of (is_valid, error_messages)

        Raises:
            ValidationError: If strict mode and validation fails
        """
        errors = []

        # 1. Validate sequence structure
        struct_errors = self._validate_structure(sequence)
        errors.extend(struct_errors)

        # 2. Validate individual commands
        for i, (command, parameters) in enumerate(sequence):
            param_errors = self._validate_command(i, command, parameters)
            errors.extend(param_errors)

        # 3. Validate sequence constraints
        constraint_errors = self._validate_constraints(sequence)
        errors.extend(constraint_errors)

        # 4. Validate geometric properties
        geom_errors = self._validate_geometry(sequence)
        errors.extend(geom_errors)

        is_valid = len(errors) == 0

        if self.strict and not is_valid:
            raise ValidationError(f"Validation failed: {errors}")

        return is_valid, errors

    def _validate_structure(
        self, sequence: list[tuple[CADCommandType, dict[str, Any]]]
    ) -> list[str]:
        """Validate overall sequence structure."""
        errors = []

        if len(sequence) == 0:
            errors.append("Empty CAD sequence")
            return errors

        # Check for START/END commands
        first_cmd = sequence[0][0]
        last_cmd = sequence[-1][0]

        if first_cmd != CADCommandType.START and first_cmd != CADCommandType.SKETCH_START:
            errors.append(
                f"Sequence should start with START or SKETCH_START, got {first_cmd.name}"
            )

        if last_cmd != CADCommandType.END:
            errors.append(f"Sequence should end with END, got {last_cmd.name}")

        return errors

    def _validate_command(
        self,
        index: int,
        command: CADCommandType,
        parameters: dict[str, Any],
    ) -> list[str]:
        """Validate individual command and its parameters."""
        errors = []

        # Validate parameters
        is_valid, error = validate_parameters(command, parameters)
        if not is_valid:
            errors.append(f"Step {index} ({command.name}): {error}")

        return errors

    def _validate_constraints(
        self, sequence: list[tuple[CADCommandType, dict[str, Any]]]
    ) -> list[str]:
        """Validate sequence constraints (sketch enclosure, dependencies, etc.)."""
        errors = []

        # Track state
        in_sketch = False
        has_sketch = False
        has_feature = False
        sketch_start_idx = -1

        for i, (command, _) in enumerate(sequence):
            # Check sketch enclosure
            if command == CADCommandType.SKETCH_START:
                if in_sketch:
                    errors.append(f"Step {i}: Nested sketch not allowed")
                in_sketch = True
                sketch_start_idx = i

            elif command == CADCommandType.SKETCH_END:
                if not in_sketch:
                    errors.append(f"Step {i}: SKETCH_END without SKETCH_START")
                in_sketch = False
                has_sketch = True

            # Sketch commands must be inside sketch block
            elif command in SEQUENCE_RULES["sketch_enclosed"]:
                if not in_sketch:
                    errors.append(
                        f"Step {i}: {command.name} must be inside SKETCH_START/SKETCH_END"
                    )

            # Feature commands require sketch
            elif command in SEQUENCE_RULES["requires_sketch"]:
                if not has_sketch:
                    errors.append(
                        f"Step {i}: {command.name} requires at least one sketch"
                    )
                has_feature = True

            # Refinement commands require feature
            elif command in SEQUENCE_RULES["requires_feature"]:
                if not has_feature:
                    errors.append(
                        f"Step {i}: {command.name} requires at least one 3D feature"
                    )

        # Check if sketch was closed
        if in_sketch:
            errors.append(
                f"Step {sketch_start_idx}: SKETCH_START not closed with SKETCH_END"
            )

        return errors

    def _validate_geometry(
        self, sequence: list[tuple[CADCommandType, dict[str, Any]]]
    ) -> list[str]:
        """Validate geometric properties (non-degenerate shapes, etc.)."""
        errors = []

        for i, (command, parameters) in enumerate(sequence):
            # Check for degenerate circles
            if command == CADCommandType.CIRCLE:
                radius = parameters.get("r", 0)
                if radius < 0.1:
                    errors.append(f"Step {i}: Circle radius too small ({radius}mm)")

            # Check for degenerate rectangles
            elif command == CADCommandType.RECTANGLE:
                width = parameters.get("width", 0)
                height = parameters.get("height", 0)
                if width < 0.1:
                    errors.append(f"Step {i}: Rectangle width too small ({width}mm)")
                if height < 0.1:
                    errors.append(f"Step {i}: Rectangle height too small ({height}mm)")

            # Check for zero-distance extrusions
            elif command in (
                CADCommandType.EXTRUDE,
                CADCommandType.CUT_EXTRUDE,
            ):
                distance = parameters.get("distance", 0)
                if distance < 0.1:
                    errors.append(
                        f"Step {i}: Extrude distance too small ({distance}mm)"
                    )

            # Check for invalid revolve angles
            elif command == CADCommandType.REVOLVE:
                angle = parameters.get("angle", 0)
                if angle <= 0 or angle > 360:
                    errors.append(
                        f"Step {i}: Invalid revolve angle ({angle} degrees)"
                    )

            # Check for negative fillet/chamfer radii
            elif command in (CADCommandType.FILLET, CADCommandType.CHAMFER):
                value = parameters.get("radius") or parameters.get("distance", 0)
                if value < 0.1:
                    errors.append(
                        f"Step {i}: {command.name} value too small ({value}mm)"
                    )

        return errors

    def auto_fix(
        self, sequence: list[tuple[CADCommandType, dict[str, Any]]]
    ) -> list[tuple[CADCommandType, dict[str, Any]]]:
        """
        Attempt to automatically fix common issues.

        Args:
            sequence: CAD sequence to fix

        Returns:
            Fixed sequence (best effort)
        """
        if len(sequence) == 0:
            return sequence

        fixed = []

        # Add START if missing (only if first command is not START or SKETCH_START)
        if sequence[0][0] not in (CADCommandType.START, CADCommandType.SKETCH_START):
            fixed.append((CADCommandType.START, {}))

        # Copy commands with sketch enclosure logic
        in_sketch = False
        for command, parameters in sequence:
            # Track if we're entering a sketch
            if command == CADCommandType.SKETCH_START:
                in_sketch = True
                fixed.append((command, parameters))
            elif command == CADCommandType.SKETCH_END:
                in_sketch = False
                fixed.append((command, parameters))
            # Ensure sketch commands are enclosed
            elif command in SEQUENCE_RULES["sketch_enclosed"] and not in_sketch:
                fixed.append((CADCommandType.SKETCH_START, {"plane": "XY"}))
                in_sketch = True
                fixed.append((command, parameters))
            # Auto-close sketch before features
            elif in_sketch and command in SEQUENCE_RULES["requires_sketch"]:
                fixed.append((CADCommandType.SKETCH_END, {}))
                in_sketch = False
                fixed.append((command, parameters))
            else:
                fixed.append((command, parameters))

        # Close unclosed sketch
        if in_sketch:
            fixed.append((CADCommandType.SKETCH_END, {}))

        # Add END if missing
        if fixed[-1][0] != CADCommandType.END:
            fixed.append((CADCommandType.END, {}))

        return fixed


def validate_sequence(
    sequence: list[tuple[CADCommandType, dict[str, Any]]], strict: bool = False
) -> tuple[bool, list[str]]:
    """
    Convenience function to validate CAD sequence.

    Args:
        sequence: CAD sequence
        strict: Raise exception on validation failure

    Returns:
        Tuple of (is_valid, error_messages)

    Example:
        >>> sequence = [(CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 50})]
        >>> is_valid, errors = validate_sequence(sequence)
    """
    validator = CADSequenceValidator(strict=strict)
    return validator.validate(sequence)


def auto_fix_sequence(
    sequence: list[tuple[CADCommandType, dict[str, Any]]]
) -> list[tuple[CADCommandType, dict[str, Any]]]:
    """
    Convenience function to auto-fix CAD sequence.

    Args:
        sequence: CAD sequence to fix

    Returns:
        Fixed sequence

    Example:
        >>> sequence = [(CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 50})]
        >>> fixed = auto_fix_sequence(sequence)
        >>> # fixed now includes SKETCH_START, SKETCH_END, etc.
    """
    validator = CADSequenceValidator()
    return validator.auto_fix(sequence)
