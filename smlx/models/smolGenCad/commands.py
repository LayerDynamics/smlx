#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
CAD command definitions and parameter schemas.

This module defines the vocabulary of CAD commands supported by smolGenCad,
along with their parameter schemas and validation logic.

Based on the SSR (Sketch, Sketch-based feature, Refinements) paradigm from
DeepCAD and Text2CAD research.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, NamedTuple


class CADCommandType(Enum):
    """
    CAD command types.

    Commands are organized into categories following the SSR paradigm:
    - Sketch: 2D drawing operations
    - Sketch-based features: 3D operations that extrude/revolve sketches
    - Refinements: Operations that modify existing geometry
    - Control: Sequence control (start, end, etc.)
    """

    # Control commands (0-9)
    START = 0  # Start of CAD sequence
    END = 1  # End of CAD sequence
    SKETCH_START = 2  # Start a new sketch
    SKETCH_END = 3  # End current sketch
    END_CURVE = 4  # End of curve in sketch (hierarchical marker)
    END_LOOP = 5  # End of loop in sketch (hierarchical marker)
    END_FACE = 6  # End of face in sketch (hierarchical marker)
    END_EXTRUDE = 7  # End of extrusion operation (hierarchical marker)

    # 2D Sketch commands (10-29)
    LINE = 10  # Draw line from point A to point B
    CIRCLE = 11  # Draw circle with center and radius
    ARC = 12  # Draw arc (center, start angle, end angle, radius)
    RECTANGLE = 13  # Draw rectangle (corner, width, height)
    POLYGON = 14  # Draw regular polygon (center, radius, num_sides)
    POINT = 15  # Define a point
    SPLINE = 16  # Draw spline through points
    ELLIPSE = 17  # Draw ellipse (center, major axis, minor axis)

    # 3D Feature commands (30-49)
    EXTRUDE = 30  # Extrude sketch to create 3D solid
    REVOLVE = 31  # Revolve sketch around axis
    LOFT = 32  # Loft between multiple sketches
    SWEEP = 33  # Sweep sketch along path
    CUT_EXTRUDE = 34  # Extrude and cut material
    CUT_REVOLVE = 35  # Revolve and cut material

    # Refinement commands (50-69)
    FILLET = 50  # Round edges with specified radius
    CHAMFER = 51  # Chamfer edges with specified distance
    SHELL = 52  # Hollow out solid with wall thickness
    MIRROR = 53  # Mirror features across plane
    PATTERN_LINEAR = 54  # Linear pattern of features
    PATTERN_CIRCULAR = 55  # Circular pattern of features
    DRAFT = 56  # Add draft angle to faces
    HOLE = 57  # Create hole (simple, counterbore, countersink)
    THREAD = 58  # Add thread to cylindrical face
    RIB = 59  # Create thin-walled rib

    @property
    def category(self) -> str:
        """Get command category."""
        value = self.value
        if value < 10:
            return "control"
        elif value < 30:
            return "sketch"
        elif value < 50:
            return "feature"
        else:
            return "refinement"

    @property
    def is_sketch_command(self) -> bool:
        """Check if this is a sketch command."""
        return self.category == "sketch"

    @property
    def is_feature_command(self) -> bool:
        """Check if this is a feature command."""
        return self.category == "feature"

    @property
    def is_refinement_command(self) -> bool:
        """Check if this is a refinement command."""
        return self.category == "refinement"


class ParameterSpec(NamedTuple):
    """Parameter specification for a CAD command."""

    name: str  # Parameter name
    type: type  # Parameter type (float, int, str)
    required: bool  # Whether parameter is required
    default: Any = None  # Default value
    min_value: float | None = None  # Minimum allowed value
    max_value: float | None = None  # Maximum allowed value
    description: str = ""  # Parameter description


# Parameter schemas for each command
COMMAND_PARAMETERS: dict[CADCommandType, list[ParameterSpec]] = {
    # Control commands
    CADCommandType.START: [],
    CADCommandType.END: [],
    CADCommandType.SKETCH_START: [
        ParameterSpec("plane", str, True, "XY", description="Sketch plane (XY, XZ, YZ)")
    ],
    CADCommandType.SKETCH_END: [],
    # Sketch commands
    CADCommandType.LINE: [
        ParameterSpec("x1", float, True, description="Start X coordinate"),
        ParameterSpec("y1", float, True, description="Start Y coordinate"),
        ParameterSpec("x2", float, True, description="End X coordinate"),
        ParameterSpec("y2", float, True, description="End Y coordinate"),
    ],
    CADCommandType.CIRCLE: [
        ParameterSpec("cx", float, True, description="Center X coordinate"),
        ParameterSpec("cy", float, True, description="Center Y coordinate"),
        ParameterSpec(
            "r", float, True, min_value=0.1, description="Radius"
        ),
    ],
    CADCommandType.ARC: [
        ParameterSpec("cx", float, True, description="Center X coordinate"),
        ParameterSpec("cy", float, True, description="Center Y coordinate"),
        ParameterSpec(
            "r", float, True, min_value=0.1, description="Radius"
        ),
        ParameterSpec(
            "start_angle", float, True, min_value=0, max_value=360, description="Start angle (degrees)"
        ),
        ParameterSpec(
            "end_angle", float, True, min_value=0, max_value=360, description="End angle (degrees)"
        ),
    ],
    CADCommandType.RECTANGLE: [
        ParameterSpec("x", float, True, description="Corner X coordinate"),
        ParameterSpec("y", float, True, description="Corner Y coordinate"),
        ParameterSpec(
            "width", float, True, min_value=0.1, description="Width"
        ),
        ParameterSpec(
            "height", float, True, min_value=0.1, description="Height"
        ),
    ],
    CADCommandType.POLYGON: [
        ParameterSpec("cx", float, True, description="Center X coordinate"),
        ParameterSpec("cy", float, True, description="Center Y coordinate"),
        ParameterSpec(
            "r", float, True, min_value=0.1, description="Radius"
        ),
        ParameterSpec(
            "sides", int, True, min_value=3, max_value=12, description="Number of sides"
        ),
    ],
    # 3D Feature commands
    CADCommandType.EXTRUDE: [
        ParameterSpec(
            "distance",
            float,
            True,
            min_value=0.1,
            description="Extrusion distance",
        ),
        ParameterSpec(
            "direction",
            str,
            False,
            "normal",
            description="Extrusion direction (normal, reverse)",
        ),
        ParameterSpec(
            "operation",
            str,
            False,
            "new_body",
            description="Operation type (new_body, join, cut)",
        ),
    ],
    CADCommandType.REVOLVE: [
        ParameterSpec(
            "angle",
            float,
            True,
            min_value=0,
            max_value=360,
            description="Revolve angle (degrees)",
        ),
        ParameterSpec(
            "axis", str, False, "Y", description="Revolve axis (X, Y, Z)"
        ),
    ],
    CADCommandType.CUT_EXTRUDE: [
        ParameterSpec(
            "distance",
            float,
            True,
            min_value=0.1,
            description="Cut depth",
        ),
        ParameterSpec(
            "direction",
            str,
            False,
            "normal",
            description="Cut direction (normal, reverse)",
        ),
    ],
    # Refinement commands
    CADCommandType.FILLET: [
        ParameterSpec(
            "radius",
            float,
            True,
            min_value=0.1,
            description="Fillet radius",
        ),
        ParameterSpec(
            "edge_indices",
            str,
            False,
            "all",
            description="Edge indices to fillet (comma-separated or 'all')",
        ),
    ],
    CADCommandType.CHAMFER: [
        ParameterSpec(
            "distance",
            float,
            True,
            min_value=0.1,
            description="Chamfer distance",
        ),
        ParameterSpec(
            "edge_indices",
            str,
            False,
            "all",
            description="Edge indices to chamfer",
        ),
    ],
    CADCommandType.SHELL: [
        ParameterSpec(
            "thickness",
            float,
            True,
            min_value=0.1,
            description="Wall thickness",
        ),
        ParameterSpec(
            "faces_to_remove",
            str,
            False,
            "",
            description="Face indices to remove (comma-separated)",
        ),
    ],
    CADCommandType.MIRROR: [
        ParameterSpec(
            "plane", str, True, "XY", description="Mirror plane (XY, XZ, YZ)"
        ),
    ],
    CADCommandType.PATTERN_LINEAR: [
        ParameterSpec("axis", str, True, "X", description="Pattern axis (X, Y, Z)"),
        ParameterSpec("count", int, True, min_value=1, description="Number of instances"),
        ParameterSpec("spacing", float, True, min_value=0.1, description="Spacing between instances"),
    ],
    CADCommandType.PATTERN_CIRCULAR: [
        ParameterSpec("axis", str, True, "Z", description="Rotation axis (X, Y, Z)"),
        ParameterSpec("count", int, True, min_value=1, description="Number of instances"),
        ParameterSpec("angle", float, False, 360.0, min_value=0.0, max_value=360.0, description="Total angle"),
    ],
    CADCommandType.HOLE: [
        ParameterSpec("cx", float, True, description="Hole center X"),
        ParameterSpec("cy", float, True, description="Hole center Y"),
        ParameterSpec(
            "diameter", float, True, min_value=0.1, description="Hole diameter"
        ),
        ParameterSpec(
            "depth", float, True, min_value=0.1, description="Hole depth"
        ),
    ],
}


def get_command_parameters(command: CADCommandType) -> list[ParameterSpec]:
    """
    Get parameter specifications for a command.

    Args:
        command: CAD command type

    Returns:
        List of parameter specifications
    """
    return COMMAND_PARAMETERS.get(command, [])


def validate_parameters(
    command: CADCommandType, parameters: dict[str, Any]
) -> tuple[bool, str]:
    """
    Validate parameters for a command.

    Args:
        command: CAD command type
        parameters: Parameter dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    specs = get_command_parameters(command)

    # Check required parameters
    for spec in specs:
        if spec.required and spec.name not in parameters:
            return False, f"Missing required parameter: {spec.name}"

    # Validate parameter types and ranges
    for param_name, param_value in parameters.items():
        # Find spec for this parameter
        spec = next((s for s in specs if s.name == param_name), None)
        if spec is None:
            return False, f"Unknown parameter: {param_name}"

        # Type check
        if not isinstance(param_value, spec.type):
            return (
                False,
                f"Parameter {param_name} must be {spec.type.__name__}, got {type(param_value).__name__}",
            )

        # Range check for numeric values
        if spec.type in (int, float):
            if spec.min_value is not None and param_value < spec.min_value:
                return (
                    False,
                    f"Parameter {param_name} must be >= {spec.min_value}, got {param_value}",
                )
            if spec.max_value is not None and param_value > spec.max_value:
                return (
                    False,
                    f"Parameter {param_name} must be <= {spec.max_value}, got {param_value}",
                )

    return True, ""


# Command sequence constraints
SEQUENCE_RULES = {
    # Sketch commands must be between SKETCH_START and SKETCH_END
    "sketch_enclosed": [
        CADCommandType.LINE,
        CADCommandType.CIRCLE,
        CADCommandType.ARC,
        CADCommandType.RECTANGLE,
        CADCommandType.POLYGON,
    ],
    # Feature commands require at least one sketch
    "requires_sketch": [
        CADCommandType.EXTRUDE,
        CADCommandType.REVOLVE,
        CADCommandType.CUT_EXTRUDE,
        CADCommandType.CUT_REVOLVE,
    ],
    # Refinement commands require at least one feature
    "requires_feature": [
        CADCommandType.FILLET,
        CADCommandType.CHAMFER,
        CADCommandType.SHELL,
        CADCommandType.MIRROR,
    ],
}
