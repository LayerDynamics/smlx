#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for smolGenCad.

Tests complete workflows from text input to CAD sequence generation.
"""

import mlx.core as mx
import pytest

from smlx.models.smolGenCad import (
    CADCommandType,
    CADTokenizer,
    SmolGenCad,
    SmolGenCadConfig,
    auto_fix_sequence,
    generate,
    generate_batch,
    sequence_to_dict,
    sequence_to_json,
    sequence_to_python,
    validate_sequence,
)
from smlx.models.smolGenCad.cache import make_cache


class MockTextTokenizer:
    """Mock text tokenizer for testing (replaces HuggingFace tokenizer)."""

    def __init__(self):
        self.vocab_size = 49152

    def __call__(self, text, return_tensors=None, padding=True, truncation=True, max_length=512):
        """Tokenize text (mock implementation)."""
        # Simple mock: return random token IDs
        # In real use, this would be HuggingFace tokenizer
        num_tokens = min(len(text.split()), max_length)
        token_ids = [1] + list(range(2, num_tokens + 2))  # BOS + tokens

        if return_tensors == "np":
            import numpy as np

            return {"input_ids": np.array([token_ids])}
        return {"input_ids": token_ids}


@pytest.fixture
def model_config():
    """Create model configuration."""
    return SmolGenCadConfig()


@pytest.fixture
def model(model_config):
    """Create model instance."""
    return SmolGenCad(model_config)


@pytest.fixture
def text_tokenizer():
    """Create mock text tokenizer."""
    return MockTextTokenizer()


@pytest.fixture
def cad_tokenizer(model_config):
    """Create CAD tokenizer."""
    return CADTokenizer(model_config.vocabulary)


@pytest.fixture
def valid_cad_sequence():
    """Create a valid CAD sequence for testing."""
    return [
        (CADCommandType.START, {}),
        (CADCommandType.SKETCH_START, {"plane": "XY"}),
        (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
        (CADCommandType.SKETCH_END, {}),
        (CADCommandType.EXTRUDE, {"distance": 100.0}),
        (CADCommandType.END, {}),
    ]


@pytest.mark.integration
class TestModelLoading:
    """Test model loading and initialization."""

    def test_model_initialization(self, model_config):
        """Test model can be initialized."""
        model = SmolGenCad(model_config)

        assert model is not None
        assert isinstance(model, SmolGenCad)
        assert model.config == model_config

    def test_model_has_required_components(self, model):
        """Test model has all required components."""
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "cad_head")

        # Every leaf of the parameter tree must be a materialized MLX array — this
        # is what makes the model runnable on Metal and catches a component that
        # was left as a plain Python value instead of an nn-registered weight.
        from mlx.utils import tree_flatten

        params = tree_flatten(model.parameters())
        assert params, "model exposes no parameters"
        assert all(isinstance(p, mx.array) for _, p in params)

    def test_tokenizer_initialization(self, cad_tokenizer):
        """Test CAD tokenizer can be initialized."""
        assert cad_tokenizer is not None
        assert isinstance(cad_tokenizer, CADTokenizer)
        # The tokenizer's vocabulary is 10 special + 70 commands + 4*256 param
        # bins (offset base 80) = 1104 addressable tokens (see CADTokenizer
        # ._build_vocabulary). 1100 was a stale approximation.
        assert cad_tokenizer.vocab_size == 1104

    def test_cache_creation(self, model):
        """Test KV cache can be created for model."""
        cache = make_cache(model)
        assert cache is not None


@pytest.mark.integration
class TestTextToCADGeneration:
    """Test complete text-to-CAD generation pipeline."""

    def test_simple_generation(self, model, text_tokenizer, cad_tokenizer):
        """Test generating CAD sequence from text."""
        prompt = "Create a cylinder with radius 5cm and height 10cm"

        # Generate (with validation disabled to avoid errors from random model)
        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            validate=False,
        )

        # Should return a list of (command, params) tuples
        assert isinstance(sequence, list)

        # Each item should be a tuple
        if len(sequence) > 0:
            assert isinstance(sequence[0], tuple)
            assert len(sequence[0]) == 2

    def test_generation_with_different_temperatures(self, model, text_tokenizer, cad_tokenizer):
        """Test generation with different temperature settings."""
        prompt = "Create a cube"

        for temp in [0.5, 1.0, 1.5]:
            sequence = generate(
                model,
                text_tokenizer,
                cad_tokenizer,
                prompt,
                max_new_tokens=10,
                temperature=temp,
                validate=False,
            )

            assert isinstance(sequence, list)

    def test_generation_with_top_k(self, model, text_tokenizer, cad_tokenizer):
        """Test generation with top-k sampling."""
        prompt = "Create a sphere"

        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            max_new_tokens=10,
            top_k=50,
            validate=False,
        )

        assert isinstance(sequence, list)

    def test_generation_with_top_p(self, model, text_tokenizer, cad_tokenizer):
        """Test generation with nucleus (top-p) sampling."""
        prompt = "Create a rectangular box"

        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            max_new_tokens=10,
            top_p=0.9,
            validate=False,
        )

        assert isinstance(sequence, list)

    def test_generation_max_tokens_limit(self, model, text_tokenizer, cad_tokenizer):
        """Test generation respects max_new_tokens limit."""
        prompt = "Create a complex mechanical part"

        # With EOS detection disabled by setting high max tokens
        # the sequence length should be capped
        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            max_new_tokens=5,  # Very short
            validate=False,
        )

        # Note: Actual length may be less if EOS is generated
        assert isinstance(sequence, list)


@pytest.mark.integration
class TestBatchGeneration:
    """Test batch generation workflows."""

    def test_batch_generation_basic(self, model, text_tokenizer, cad_tokenizer):
        """Test generating sequences for multiple prompts."""
        prompts = [
            "Create a cylinder",
            "Create a cube",
            "Create a sphere",
        ]

        sequences = generate_batch(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompts,
            max_new_tokens=10,
            validate=False,
        )

        # Should return one sequence per prompt
        assert len(sequences) == len(prompts)

        # Each sequence should be a list
        for seq in sequences:
            assert isinstance(seq, list)

    def test_batch_generation_empty_prompts(self, model, text_tokenizer, cad_tokenizer):
        """Test batch generation with empty prompt list."""
        prompts = []

        sequences = generate_batch(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompts,
            max_new_tokens=10,
        )

        assert len(sequences) == 0

    def test_batch_generation_single_prompt(self, model, text_tokenizer, cad_tokenizer):
        """Test batch generation with single prompt."""
        prompts = ["Create a cylinder"]

        sequences = generate_batch(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompts,
            max_new_tokens=10,
            validate=False,
        )

        assert len(sequences) == 1


@pytest.mark.integration
class TestExportFormats:
    """Test exporting CAD sequences to different formats."""

    def test_export_to_dict(self, valid_cad_sequence):
        """Test exporting sequence to dictionary format."""
        dict_seq = sequence_to_dict(valid_cad_sequence)

        assert isinstance(dict_seq, list)
        assert len(dict_seq) == len(valid_cad_sequence)

        # Check first item structure
        assert "command" in dict_seq[0]
        assert "parameters" in dict_seq[0]
        assert dict_seq[0]["command"] == "START"

    def test_export_to_json(self, valid_cad_sequence):
        """Test exporting sequence to JSON format."""
        json_str = sequence_to_json(valid_cad_sequence)

        assert isinstance(json_str, str)
        assert "START" in json_str
        assert "CIRCLE" in json_str

        # Should be valid JSON
        import json

        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_export_to_python(self, valid_cad_sequence):
        """Test exporting sequence to Python (CadQuery) code."""
        python_code = sequence_to_python(valid_cad_sequence)

        assert isinstance(python_code, str)
        assert "import cadquery as cq" in python_code
        assert "Workplane" in python_code

        # Should contain CAD operations
        assert "circle" in python_code.lower() or "extrude" in python_code.lower()

    def test_export_empty_sequence(self):
        """Test exporting empty sequence."""
        empty_seq = []

        dict_seq = sequence_to_dict(empty_seq)
        assert dict_seq == []

        json_str = sequence_to_json(empty_seq)
        assert json_str == "[]"

        python_code = sequence_to_python(empty_seq)
        # Should still have import statement
        assert "import cadquery" in python_code

    @pytest.mark.parametrize("plane", ["XY", "XZ", "YZ"])
    def test_export_python_valid_planes(self, plane):
        """A canonical sketch plane is emitted verbatim into the CadQuery code."""
        sequence = [
            (CADCommandType.SKETCH_START, {"plane": plane}),
            (CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 10}),
            (CADCommandType.SKETCH_END, {}),
        ]
        python_code = sequence_to_python(sequence)

        assert f"result = result.workplane('{plane}')" in python_code
        assert "invalid sketch plane" not in python_code

    def test_export_python_invalid_plane_defaults_to_xy(self):
        """An out-of-spec plane falls back to XY and is flagged, not used raw."""
        sequence = [
            (CADCommandType.SKETCH_START, {"plane": "BOGUS"}),
            (CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 10}),
            (CADCommandType.SKETCH_END, {}),
        ]
        python_code = sequence_to_python(sequence)

        # The only workplane() call uses the safe fallback, never the bad value.
        assert "result = result.workplane('XY')" in python_code
        assert "workplane('BOGUS')" not in python_code
        # The rejected value is surfaced in a diagnostic comment.
        assert "# warning: invalid sketch plane 'BOGUS', defaulting to 'XY'" in python_code

    def test_export_python_plane_with_stray_quote_cannot_break_codegen(self):
        """A stray quote in the plane string must not break the emitted Python.

        The plane is interpolated into ``workplane('{plane}')``; an injected quote
        would otherwise produce syntactically invalid (or injected) code. The
        validation guard substitutes XY and repr-quotes the rejected value into a
        single-line comment, so the output must remain compilable.
        """
        import ast

        malicious_plane = "XY') or __import__('os"
        sequence = [
            (CADCommandType.SKETCH_START, {"plane": malicious_plane}),
            (CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 10}),
            (CADCommandType.SKETCH_END, {}),
        ]
        python_code = sequence_to_python(sequence)

        # The injected expression never lands in an executable statement.
        assert "result = result.workplane('XY')" in python_code
        assert "or __import__" not in "\n".join(
            line for line in python_code.splitlines() if not line.lstrip().startswith("#")
        )
        # Emitted code must still parse as valid Python.
        ast.parse(python_code)


@pytest.mark.integration
class TestValidationWorkflow:
    """Test validation and auto-fixing in generation workflow."""

    def test_auto_fix_in_generation(self, model, text_tokenizer, cad_tokenizer):
        """Test auto-fix is applied during generation."""
        prompt = "Create a simple shape"

        # Generate with auto_fix enabled
        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            max_new_tokens=15,
            auto_fix=True,
            validate=False,
        )

        # Sequence should be auto-fixed
        assert isinstance(sequence, list)

    def test_validation_in_generation(self, model, text_tokenizer, cad_tokenizer):
        """Test validation during generation."""
        prompt = "Create a cylinder"

        # Generate with validation enabled
        # (may produce warnings but shouldn't fail)
        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            max_new_tokens=10,
            validate=True,
            auto_fix=True,
        )

        assert isinstance(sequence, list)

    def test_manual_validation_workflow(self, valid_cad_sequence):
        """Test manual validation workflow."""
        # Validate valid sequence
        is_valid, errors = validate_sequence(valid_cad_sequence)
        assert is_valid
        assert len(errors) == 0

    def test_manual_auto_fix_workflow(self):
        """Test manual auto-fix workflow."""
        # Create invalid sequence
        invalid_seq = [
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.EXTRUDE, {"distance": 100.0}),
        ]

        # Auto-fix
        fixed_seq = auto_fix_sequence(invalid_seq)

        # Fixed should be valid
        is_valid, errors = validate_sequence(fixed_seq)
        assert is_valid or len(errors) < len(validate_sequence(invalid_seq)[1])


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_workflow_cylinder(self, model, text_tokenizer, cad_tokenizer):
        """Test complete workflow: text → CAD → export → validate."""
        prompt = "Create a cylinder with radius 10mm and height 20mm"

        # 1. Generate CAD sequence
        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            max_new_tokens=20,
            temperature=0.8,
            auto_fix=True,
            validate=False,
        )

        # 2. Export to different formats
        dict_format = sequence_to_dict(sequence)
        json_format = sequence_to_json(sequence)
        python_format = sequence_to_python(sequence)

        # 3. Validate all exports produced output
        assert isinstance(dict_format, list)
        assert isinstance(json_format, str)
        assert isinstance(python_format, str)

        # 4. Validate sequence
        is_valid, errors = validate_sequence(sequence)
        # May not be valid (random weights) but shouldn't crash
        assert isinstance(is_valid, bool)

    def test_complete_workflow_batch(self, model, text_tokenizer, cad_tokenizer):
        """Test complete batch workflow."""
        prompts = [
            "Create a cube with side 15mm",
            "Create a hollow cylinder",
        ]

        # 1. Generate batch
        sequences = generate_batch(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompts,
            max_new_tokens=15,
            temperature=0.7,
        )

        # 2. Export each sequence
        for seq in sequences:
            dict_format = sequence_to_dict(seq)
            assert isinstance(dict_format, list)

    def test_workflow_with_different_prompts(self, model, text_tokenizer, cad_tokenizer):
        """Test workflow with various CAD descriptions."""
        prompts = [
            "Create a simple cylinder",
            "Make a rectangular box",
            "Design a sphere",
            "Create a part with filleted edges",
        ]

        for prompt in prompts:
            sequence = generate(
                model,
                text_tokenizer,
                cad_tokenizer,
                prompt,
                max_new_tokens=10,
                validate=False,
            )

            # Should generate something
            assert isinstance(sequence, list)


@pytest.mark.integration
class TestModelPersistence:
    """Test model saving and loading (basic tests)."""

    def test_model_config_serialization(self, model_config):
        """Test model configuration can be serialized."""
        config_dict = model_config.to_dict()

        assert isinstance(config_dict, dict)
        assert "model_type" in config_dict
        assert config_dict["model_type"] == "smolGenCad"

    def test_model_config_deserialization(self, model_config):
        """Test model configuration can be deserialized."""
        config_dict = model_config.to_dict()
        restored_config = SmolGenCadConfig.from_dict(config_dict)

        assert restored_config.model_type == model_config.model_type
        assert restored_config.encoder.hidden_size == model_config.encoder.hidden_size
        assert restored_config.decoder.hidden_size == model_config.decoder.hidden_size

    def test_model_weights_extraction(self, model):
        """Test model weights can be extracted."""
        weights = model.parameters()

        assert isinstance(weights, dict)
        # Should have encoder, decoder, and head weights
        assert len(weights) > 0


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in integration workflows."""

    def test_generation_with_empty_prompt(self, model, text_tokenizer, cad_tokenizer):
        """Test generation with empty prompt."""
        prompt = ""

        # Should handle gracefully
        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            max_new_tokens=5,
            validate=False,
        )

        assert isinstance(sequence, list)

    def test_generation_with_very_long_prompt(self, model, text_tokenizer, cad_tokenizer):
        """Test generation with very long prompt."""
        prompt = " ".join(["Create a cylinder"] * 100)  # Very long

        # Should truncate and handle
        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            max_new_tokens=5,
            validate=False,
        )

        assert isinstance(sequence, list)

    def test_export_with_invalid_commands(self):
        """Test exporting sequence with potentially invalid commands."""
        # Sequence with minimal parameters
        sequence = [
            (CADCommandType.START, {}),
            (CADCommandType.END, {}),
        ]

        # Should export without crashing
        dict_format = sequence_to_dict(sequence)
        json_format = sequence_to_json(sequence)
        python_format = sequence_to_python(sequence)

        assert isinstance(dict_format, list)
        assert isinstance(json_format, str)
        assert isinstance(python_format, str)


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Test performance characteristics (marked as slow)."""

    def test_generation_speed(self, model, text_tokenizer, cad_tokenizer):
        """Test generation completes in reasonable time."""
        import time

        prompt = "Create a simple cylinder"

        start_time = time.time()

        sequence = generate(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompt,
            max_new_tokens=10,
            validate=False,
        )

        elapsed = time.time() - start_time

        # Should complete quickly (even with random weights)
        # Actual time depends on hardware, but shouldn't take minutes
        assert elapsed < 60.0  # Very generous timeout

        assert isinstance(sequence, list)

    def test_batch_generation_speed(self, model, text_tokenizer, cad_tokenizer):
        """Test batch generation completes in reasonable time."""
        import time

        prompts = ["Create a cylinder"] * 5

        start_time = time.time()

        sequences = generate_batch(
            model,
            text_tokenizer,
            cad_tokenizer,
            prompts,
            max_new_tokens=10,
        )

        elapsed = time.time() - start_time

        # Batch should complete in reasonable time
        assert elapsed < 120.0  # 2 minutes for 5 sequences

        assert len(sequences) == len(prompts)
