#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Unit tests for CAD command definitions and validation.

Tests command vocabulary, parameter schemas, and validation logic.
"""

import pytest

from smlx.models.smolGenCad.commands import (
    CADCommandType,
    COMMAND_PARAMETERS,
    SEQUENCE_RULES,
    get_command_parameters,
    validate_parameters,
)


class TestCADCommandType:
    """Test CAD command type enum."""

    def test_command_categories(self):
        """Test command category classification."""
        assert CADCommandType.START.category == "control"
        assert CADCommandType.LINE.category == "sketch"
        assert CADCommandType.EXTRUDE.category == "feature"
        assert CADCommandType.FILLET.category == "refinement"

    def test_is_sketch_command(self):
        """Test sketch command identification."""
        assert CADCommandType.CIRCLE.is_sketch_command
        assert CADCommandType.LINE.is_sketch_command
        assert not CADCommandType.EXTRUDE.is_sketch_command
        assert not CADCommandType.FILLET.is_sketch_command

    def test_is_feature_command(self):
        """Test feature command identification."""
        assert CADCommandType.EXTRUDE.is_feature_command
        assert CADCommandType.REVOLVE.is_feature_command
        assert not CADCommandType.LINE.is_feature_command
        assert not CADCommandType.FILLET.is_feature_command

    def test_is_refinement_command(self):
        """Test refinement command identification."""
        assert CADCommandType.FILLET.is_refinement_command
        assert CADCommandType.CHAMFER.is_refinement_command
        assert not CADCommandType.EXTRUDE.is_refinement_command
        assert not CADCommandType.LINE.is_refinement_command

    def test_all_commands_have_category(self):
        """Test all commands are categorized."""
        for cmd in CADCommandType:
            assert cmd.category in ["control", "sketch", "feature", "refinement"]


class TestCommandParameters:
    """Test command parameter schemas."""

    def test_get_command_parameters(self):
        """Test retrieving parameter specs for commands."""
        # Circle should have cx, cy, r parameters
        params = get_command_parameters(CADCommandType.CIRCLE)
        assert len(params) >= 3
        param_names = [p.name for p in params]
        assert "cx" in param_names
        assert "cy" in param_names
        assert "r" in param_names

    def test_line_parameters(self):
        """Test line command parameters."""
        params = get_command_parameters(CADCommandType.LINE)
        param_names = [p.name for p in params]
        assert "x1" in param_names
        assert "y1" in param_names
        assert "x2" in param_names
        assert "y2" in param_names

    def test_extrude_parameters(self):
        """Test extrude command parameters."""
        params = get_command_parameters(CADCommandType.EXTRUDE)
        param_names = [p.name for p in params]
        assert "distance" in param_names

    def test_fillet_parameters(self):
        """Test fillet command parameters."""
        params = get_command_parameters(CADCommandType.FILLET)
        param_names = [p.name for p in params]
        assert "radius" in param_names

    def test_all_commands_have_parameters_defined(self):
        """Test all commands have parameter definitions."""
        for cmd in CADCommandType:
            # Should not raise KeyError
            params = get_command_parameters(cmd)
            assert isinstance(params, list)


class TestParameterValidation:
    """Test parameter validation logic."""

    def test_valid_circle_parameters(self):
        """Test validating valid circle parameters."""
        is_valid, error = validate_parameters(
            CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}
        )
        assert is_valid
        assert error == ""

    def test_missing_required_parameter(self):
        """Test validation fails for missing required parameter."""
        is_valid, error = validate_parameters(
            CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0}  # Missing 'r'
        )
        assert not is_valid
        assert "radius" in error.lower() or "r" in error.lower()

    def test_invalid_parameter_type(self):
        """Test validation fails for wrong parameter type."""
        is_valid, error = validate_parameters(
            CADCommandType.CIRCLE,
            {"cx": 0.0, "cy": 0.0, "r": "not_a_number"},  # Wrong type
        )
        assert not is_valid
        assert "type" in error.lower() or "float" in error.lower()

    def test_parameter_range_validation_min(self):
        """Test validation fails for values below minimum."""
        is_valid, error = validate_parameters(
            CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 0.05}  # Below min (0.1)
        )
        assert not is_valid
        assert ">=" in error

    def test_parameter_range_validation_max(self):
        """Test validation fails for values above maximum."""
        is_valid, error = validate_parameters(
            CADCommandType.ARC,
            {
                "cx": 0.0,
                "cy": 0.0,
                "r": 10.0,
                "start_angle": 0.0,
                "end_angle": 400.0,  # Above max (360)
            },
        )
        assert not is_valid
        assert "<=" in error

    def test_unknown_parameter(self):
        """Test validation fails for unknown parameter."""
        is_valid, error = validate_parameters(
            CADCommandType.CIRCLE,
            {
                "cx": 0.0,
                "cy": 0.0,
                "r": 50.0,
                "unknown_param": 123,  # Unknown parameter
            },
        )
        assert not is_valid
        assert "unknown" in error.lower()

    def test_valid_extrude_parameters(self):
        """Test validating valid extrude parameters."""
        is_valid, error = validate_parameters(
            CADCommandType.EXTRUDE, {"distance": 100.0}
        )
        assert is_valid
        assert error == ""

    def test_valid_fillet_parameters(self):
        """Test validating valid fillet parameters."""
        is_valid, error = validate_parameters(
            CADCommandType.FILLET, {"radius": 5.0}
        )
        assert is_valid
        assert error == ""


class TestSequenceRules:
    """Test sequence constraint rules."""

    def test_sketch_enclosed_rule(self):
        """Test sketch commands must be enclosed."""
        sketch_commands = SEQUENCE_RULES["sketch_enclosed"]
        assert CADCommandType.LINE in sketch_commands
        assert CADCommandType.CIRCLE in sketch_commands
        assert CADCommandType.ARC in sketch_commands

    def test_requires_sketch_rule(self):
        """Test commands that require a sketch."""
        requires_sketch = SEQUENCE_RULES["requires_sketch"]
        assert CADCommandType.EXTRUDE in requires_sketch
        assert CADCommandType.REVOLVE in requires_sketch

    def test_requires_feature_rule(self):
        """Test commands that require a feature."""
        requires_feature = SEQUENCE_RULES["requires_feature"]
        assert CADCommandType.FILLET in requires_feature
        assert CADCommandType.CHAMFER in requires_feature


@pytest.mark.unit
class TestCommandParameterSpecs:
    """Test parameter specification details."""

    def test_parameter_spec_has_required_fields(self):
        """Test parameter specs have all required fields."""
        params = get_command_parameters(CADCommandType.CIRCLE)
        for spec in params:
            assert hasattr(spec, "name")
            assert hasattr(spec, "type")
            assert hasattr(spec, "required")
            assert hasattr(spec, "default")
            assert hasattr(spec, "min_value")
            assert hasattr(spec, "max_value")
            assert hasattr(spec, "description")

    def test_required_parameters_marked_correctly(self):
        """Test required parameters are marked as required."""
        params = get_command_parameters(CADCommandType.CIRCLE)
        radius_param = next(p for p in params if p.name == "r")
        assert radius_param.required is True

    def test_parameter_types_are_valid(self):
        """Test all parameter types are Python types."""
        for cmd in CADCommandType:
            params = get_command_parameters(cmd)
            for spec in params:
                assert spec.type in (int, float, str, bool)
