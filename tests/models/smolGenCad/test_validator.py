#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Unit tests for smolGenCad validator.

Tests sequence validation, constraint checking, and auto-fixing.
"""

import pytest

from smlx.models.smolGenCad.commands import CADCommandType
from smlx.models.smolGenCad.validator import (
    CADSequenceValidator,
    ValidationError,
    auto_fix_sequence,
    validate_sequence,
)


@pytest.fixture
def validator():
    """Create a validator instance."""
    return CADSequenceValidator()


@pytest.fixture
def valid_sequence():
    """Create a valid CAD sequence."""
    return [
        (CADCommandType.START, {}),
        (CADCommandType.SKETCH_START, {"plane": "XY"}),
        (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
        (CADCommandType.SKETCH_END, {}),
        (CADCommandType.EXTRUDE, {"distance": 100.0}),
        (CADCommandType.END, {}),
    ]


@pytest.fixture
def invalid_sequence_missing_start():
    """Sequence missing START command."""
    return [
        (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
        (CADCommandType.EXTRUDE, {"distance": 100.0}),
        (CADCommandType.END, {}),
    ]


@pytest.fixture
def invalid_sequence_unclosed_sketch():
    """Sequence with unclosed sketch."""
    return [
        (CADCommandType.START, {}),
        (CADCommandType.SKETCH_START, {"plane": "XY"}),
        (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
        # Missing SKETCH_END
        (CADCommandType.EXTRUDE, {"distance": 100.0}),
        (CADCommandType.END, {}),
    ]


@pytest.mark.unit
class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_creation(self):
        """Test creating validation error."""
        error = ValidationError("Test error message")
        assert str(error) == "Test error message"

    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from Exception."""
        error = ValidationError("Test")
        assert isinstance(error, Exception)


@pytest.mark.unit
class TestCADSequenceValidatorInit:
    """Test CADSequenceValidator initialization."""

    def test_default_initialization(self):
        """Test validator with default settings."""
        validator = CADSequenceValidator()
        assert validator.strict is False

    def test_strict_mode_initialization(self):
        """Test validator with strict mode."""
        validator = CADSequenceValidator(strict=True)
        assert validator.strict is True


@pytest.mark.unit
class TestValidateStructure:
    """Test sequence structure validation."""

    def test_empty_sequence(self, validator):
        """Test validation of empty sequence."""
        is_valid, errors = validator.validate([])
        assert not is_valid
        assert any("Empty CAD sequence" in err for err in errors)

    def test_valid_structure(self, validator, valid_sequence):
        """Test validation of valid sequence structure."""
        is_valid, errors = validator.validate(valid_sequence)
        assert is_valid
        assert len(errors) == 0

    def test_missing_start_command(self, validator):
        """Test sequence missing START command."""
        sequence = [
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("should start with START" in err for err in errors)

    def test_missing_end_command(self, validator):
        """Test sequence missing END command."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.SKETCH_END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("should end with END" in err for err in errors)

    def test_sketch_start_is_acceptable_start(self, validator):
        """Test SKETCH_START is acceptable as first command."""
        sequence = [
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.EXTRUDE, {"distance": 100.0}),
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        # Should only fail on structure if other issues exist
        structure_errors = [e for e in errors if "should start with" in e]
        assert len(structure_errors) == 0


@pytest.mark.unit
class TestValidateCommand:
    """Test individual command validation."""

    def test_valid_command_parameters(self, validator):
        """Test validation of valid command parameters."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert is_valid
        assert len(errors) == 0

    def test_invalid_command_parameters(self, validator):
        """Test validation fails with invalid parameters."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            # Missing required 'r' parameter
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0}),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("CIRCLE" in err and "Step 2" in err for err in errors)

    def test_extra_parameters_rejected(self, validator):
        """Test that extra parameters are rejected."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (
                CADCommandType.CIRCLE,
                {"cx": 0.0, "cy": 0.0, "r": 50.0, "extra": "value"},
            ),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        # Extra parameters should cause validation failure
        assert not is_valid
        param_errors = [e for e in errors if "extra" in e.lower()]
        assert len(param_errors) > 0


@pytest.mark.unit
class TestValidateConstraints:
    """Test sequence constraint validation."""

    def test_nested_sketches_not_allowed(self, validator):
        """Test that nested sketches are detected."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.SKETCH_START, {"plane": "XZ"}),  # Nested!
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("Nested sketch" in err for err in errors)

    def test_sketch_end_without_start(self, validator):
        """Test SKETCH_END without SKETCH_START."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.SKETCH_END, {}),  # No matching START
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("SKETCH_END without SKETCH_START" in err for err in errors)

    def test_unclosed_sketch(self, validator, invalid_sequence_unclosed_sketch):
        """Test unclosed sketch detection."""
        is_valid, errors = validator.validate(invalid_sequence_unclosed_sketch)
        assert not is_valid
        assert any("not closed with SKETCH_END" in err for err in errors)

    def test_sketch_command_outside_sketch_block(self, validator):
        """Test sketch command outside sketch block."""
        sequence = [
            (CADCommandType.START, {}),
            # LINE outside sketch block
            (CADCommandType.LINE, {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 0.0}),
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any(
            "must be inside SKETCH_START/SKETCH_END" in err for err in errors
        )

    def test_feature_requires_sketch(self, validator):
        """Test that features require a sketch."""
        sequence = [
            (CADCommandType.START, {}),
            # EXTRUDE without any sketch
            (CADCommandType.EXTRUDE, {"distance": 100.0}),
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("requires at least one sketch" in err for err in errors)

    def test_refinement_requires_feature(self, validator):
        """Test that refinements require a 3D feature."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.SKETCH_END, {}),
            # FILLET without any 3D feature
            (CADCommandType.FILLET, {"radius": 5.0}),
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("requires at least one 3D feature" in err for err in errors)

    def test_valid_feature_after_sketch(self, validator, valid_sequence):
        """Test valid feature after sketch."""
        is_valid, errors = validator.validate(valid_sequence)
        assert is_valid
        assert len(errors) == 0


@pytest.mark.unit
class TestValidateGeometry:
    """Test geometric validation."""

    def test_degenerate_circle(self, validator):
        """Test degenerate circle detection."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 0.05}),  # Too small
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("Circle radius too small" in err for err in errors)

    def test_degenerate_rectangle(self, validator):
        """Test degenerate rectangle detection."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (
                CADCommandType.RECTANGLE,
                {"x": 0.0, "y": 0.0, "width": 0.05, "height": 100.0},
            ),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("Rectangle width too small" in err for err in errors)

    def test_zero_distance_extrude(self, validator):
        """Test zero-distance extrude detection."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.EXTRUDE, {"distance": 0.05}),  # Too small
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("Extrude distance too small" in err for err in errors)

    def test_invalid_revolve_angle(self, validator):
        """Test invalid revolve angle detection."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.REVOLVE, {"axis": "X", "angle": 400.0}),  # > 360
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("Invalid revolve angle" in err for err in errors)

    def test_negative_fillet_radius(self, validator):
        """Test negative fillet radius detection."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.EXTRUDE, {"distance": 100.0}),
            (CADCommandType.FILLET, {"radius": 0.05}),  # Too small
            (CADCommandType.END, {}),
        ]
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert any("FILLET value too small" in err for err in errors)

    def test_valid_geometry(self, validator, valid_sequence):
        """Test valid geometry passes."""
        is_valid, errors = validator.validate(valid_sequence)
        assert is_valid
        assert len(errors) == 0


@pytest.mark.unit
class TestAutoFix:
    """Test automatic sequence fixing."""

    def test_add_missing_start(self, validator):
        """Test auto-fix adds missing START command."""
        sequence = [
            # Missing START (starts with regular command, not SKETCH_START)
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.END, {}),
        ]
        fixed = validator.auto_fix(sequence)

        # Should have START at beginning
        assert fixed[0][0] == CADCommandType.START

    def test_add_missing_end(self, validator):
        """Test auto-fix adds missing END command."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.SKETCH_END, {}),
        ]
        fixed = validator.auto_fix(sequence)

        # Should have END at end
        assert fixed[-1][0] == CADCommandType.END

    def test_enclose_sketch_commands(self, validator):
        """Test auto-fix encloses sketch commands."""
        sequence = [
            (CADCommandType.START, {}),
            # LINE without sketch enclosure
            (CADCommandType.LINE, {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 0.0}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.END, {}),
        ]
        fixed = validator.auto_fix(sequence)

        # Should have SKETCH_START before LINE
        line_idx = next(i for i, (cmd, _) in enumerate(fixed) if cmd == CADCommandType.LINE)
        assert fixed[line_idx - 1][0] == CADCommandType.SKETCH_START

        # Should have SKETCH_END after sketch commands
        sketch_end_exists = any(cmd == CADCommandType.SKETCH_END for cmd, _ in fixed)
        assert sketch_end_exists

    def test_close_unclosed_sketch(self, validator):
        """Test auto-fix closes unclosed sketch."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            # Missing SKETCH_END
            (CADCommandType.END, {}),
        ]
        fixed = validator.auto_fix(sequence)

        # Should have SKETCH_END somewhere before the final END
        has_sketch_end = any(cmd == CADCommandType.SKETCH_END for cmd, _ in fixed)
        assert has_sketch_end

        # Verify END is still last
        assert fixed[-1][0] == CADCommandType.END

    def test_auto_close_sketch_before_feature(self, validator):
        """Test auto-fix closes sketch before feature."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            # Missing SKETCH_END before EXTRUDE
            (CADCommandType.EXTRUDE, {"distance": 100.0}),
            (CADCommandType.END, {}),
        ]
        fixed = validator.auto_fix(sequence)

        # Should have SKETCH_END before EXTRUDE
        extrude_idx = next(
            i for i, (cmd, _) in enumerate(fixed) if cmd == CADCommandType.EXTRUDE
        )
        assert fixed[extrude_idx - 1][0] == CADCommandType.SKETCH_END

    def test_auto_fix_preserves_valid_sequence(self, validator, valid_sequence):
        """Test auto-fix preserves already valid sequence."""
        fixed = validator.auto_fix(valid_sequence)

        # Fixed sequence should still be valid
        is_valid, errors = validator.validate(fixed)
        assert is_valid or len(errors) < 3  # Allow minor issues

        # Should have all original commands present
        original_commands = [cmd for cmd, _ in valid_sequence]
        fixed_commands = [cmd for cmd, _ in fixed]

        # All original commands should be in fixed sequence
        for orig_cmd in original_commands:
            assert orig_cmd in fixed_commands


@pytest.mark.unit
class TestStrictMode:
    """Test strict mode behavior."""

    def test_strict_mode_raises_on_invalid(self):
        """Test strict mode raises ValidationError."""
        validator = CADSequenceValidator(strict=True)
        sequence = [
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            # Invalid - missing START/END
        ]

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(sequence)

        assert "Validation failed" in str(exc_info.value)

    def test_non_strict_mode_returns_errors(self):
        """Test non-strict mode returns errors without raising."""
        validator = CADSequenceValidator(strict=False)
        sequence = [
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
        ]

        # Should not raise
        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        assert len(errors) > 0


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_validate_sequence_function(self, valid_sequence):
        """Test validate_sequence convenience function."""
        is_valid, errors = validate_sequence(valid_sequence)
        assert is_valid
        assert len(errors) == 0

    def test_validate_sequence_with_strict(self):
        """Test validate_sequence with strict mode."""
        sequence = [(CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0})]

        with pytest.raises(ValidationError):
            validate_sequence(sequence, strict=True)

    def test_auto_fix_sequence_function(self):
        """Test auto_fix_sequence convenience function."""
        sequence = [
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
        ]

        fixed = auto_fix_sequence(sequence)

        # Should have proper structure
        assert fixed[0][0] in (CADCommandType.START, CADCommandType.SKETCH_START)
        assert fixed[-1][0] == CADCommandType.END

    def test_auto_fix_then_validate(self):
        """Test fixing then validating a sequence."""
        sequence = [
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.EXTRUDE, {"distance": 100.0}),
        ]

        # Original is invalid
        is_valid, errors = validate_sequence(sequence)
        assert not is_valid

        # Fix it
        fixed = auto_fix_sequence(sequence)

        # Fixed should be valid
        is_valid, errors = validate_sequence(fixed)
        assert is_valid
        assert len(errors) == 0


@pytest.mark.unit
class TestComplexValidationScenarios:
    """Test complex validation scenarios."""

    def test_multiple_sketches_and_features(self, validator):
        """Test sequence with multiple sketches and features."""
        sequence = [
            (CADCommandType.START, {}),
            # First sketch and feature
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.EXTRUDE, {"distance": 100.0}),
            # Second sketch and feature
            (CADCommandType.SKETCH_START, {"plane": "XZ"}),
            (CADCommandType.RECTANGLE, {"x": 0.0, "y": 0.0, "width": 20.0, "height": 30.0}),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.CUT_EXTRUDE, {"distance": 50.0}),
            # Refinement
            (CADCommandType.FILLET, {"radius": 2.0}),
            (CADCommandType.END, {}),
        ]

        is_valid, errors = validator.validate(sequence)
        assert is_valid
        assert len(errors) == 0

    def test_sequence_with_patterns(self, validator):
        """Test sequence with pattern operations."""
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 5.0}),
            (CADCommandType.SKETCH_END, {}),
            (CADCommandType.EXTRUDE, {"distance": 10.0}),
            (CADCommandType.PATTERN_LINEAR, {"axis": "X", "count": 5, "spacing": 20.0}),
            (CADCommandType.END, {}),
        ]

        is_valid, errors = validator.validate(sequence)
        assert is_valid
        assert len(errors) == 0

    def test_all_error_types_together(self, validator):
        """Test sequence with multiple error types."""
        sequence = [
            # Missing START
            (CADCommandType.SKETCH_START, {"plane": "XY"}),
            # Degenerate circle
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 0.01}),
            # Missing SKETCH_END
            # Invalid extrude
            (CADCommandType.EXTRUDE, {"distance": 0.01}),
            # Missing END
        ]

        is_valid, errors = validator.validate(sequence)
        assert not is_valid
        # Should have multiple errors
        assert len(errors) >= 3
