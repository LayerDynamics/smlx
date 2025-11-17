"""
Tests for quantization utilities.

Tests the quantization utilities in smlx/utils/quantization.py including:
- apply_quantization() for different methods
- Quantization configuration
- Helper functions for checking quantizable layers
- Size estimation

Run with:
    pytest tests/utils/test_quantization.py -v
"""

import pytest

import mlx.core as mx
import mlx.nn as nn

from smlx.utils.quantization import (
    apply_quantization,
    count_quantizable_layers,
    create_class_predicate,
    estimate_quantized_size,
    get_quantization_config,
    get_quantization_info,
    has_quantizable_layers,
)


class SimpleModel(nn.Module):
    """Simple test model with quantizable layers."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.fc3 = nn.Linear(dim, dim)

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.relu(x)
        x = self.fc3(x)
        return x


class ModelWithoutQuantizableLayers(nn.Module):
    """Model without quantizable layers (for negative testing)."""

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()

    def __call__(self, x):
        return self.activation(x)


# Test Fixtures


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(dim=32)


@pytest.fixture
def non_quantizable_model():
    """Create a model without quantizable layers."""
    return ModelWithoutQuantizableLayers()


# Test Quantization Config


def test_get_quantization_config_fp16():
    """Test FP16 config (no quantization)."""
    config = get_quantization_config("fp16")
    assert config is None


def test_get_quantization_config_4bit():
    """Test 4-bit quantization config."""
    config = get_quantization_config("4bit")
    assert config is not None
    assert config["bits"] == 4
    assert config["group_size"] == 64
    assert config["mode"] == "affine"


def test_get_quantization_config_8bit():
    """Test 8-bit quantization config."""
    config = get_quantization_config("8bit")
    assert config is not None
    assert config["bits"] == 8
    assert config["group_size"] == 64
    assert config["mode"] == "affine"


def test_get_quantization_config_invalid():
    """Test invalid quantization method."""
    with pytest.raises(ValueError, match="Unknown quantization method"):
        get_quantization_config("invalid_method")


def test_get_quantization_config_gptq():
    """Test GPTQ config exists (even if not implemented)."""
    config = get_quantization_config("gptq")
    assert config is not None
    assert config["bits"] == 4
    assert config.get("requires_calibration") == True


# Test Layer Detection


def test_has_quantizable_layers_simple_model(simple_model):
    """Test detecting quantizable layers in simple model."""
    assert has_quantizable_layers(simple_model) is True


def test_has_quantizable_layers_non_quantizable(non_quantizable_model):
    """Test detecting no quantizable layers."""
    assert has_quantizable_layers(non_quantizable_model) is False


def test_count_quantizable_layers_simple_model(simple_model):
    """Test counting quantizable layers."""
    count = count_quantizable_layers(simple_model)
    assert count == 3  # fc1, fc2, fc3


def test_count_quantizable_layers_non_quantizable(non_quantizable_model):
    """Test counting layers in non-quantizable model."""
    count = count_quantizable_layers(non_quantizable_model)
    assert count == 0


# Test Class Predicate


def test_create_class_predicate_no_weights():
    """Test creating class predicate without weights."""
    predicate = create_class_predicate(weights=None)

    # Test with quantizable module (nn.Linear has to_quantized)
    linear = nn.Linear(32, 64)
    assert predicate("layer.fc1", linear) is True

    # Test with non-quantizable module
    relu = nn.ReLU()
    assert predicate("layer.relu", relu) is False


def test_create_class_predicate_with_weights():
    """Test creating class predicate with weights dict."""
    # Simulate pre-quantized weights
    weights = {
        "layer1.weight": mx.zeros((64, 32)),
        "layer1.scales": mx.ones((64, 1)),  # Indicates quantized
        "layer2.weight": mx.zeros((32, 64)),
        # layer2 has no scales, so not pre-quantized
    }

    predicate = create_class_predicate(weights=weights)

    # Mock modules
    linear1 = nn.Linear(32, 64)
    linear2 = nn.Linear(64, 32)

    # layer1 has scales, should be quantized
    assert predicate("layer1", linear1) is True

    # layer2 has no scales, should not be quantized
    assert predicate("layer2", linear2) is False


# Test apply_quantization


def test_apply_quantization_fp16(simple_model):
    """Test FP16 (no quantization)."""
    # Flatten nested parameter structure
    def count_params(params):
        count = 0
        for v in params.values():
            if isinstance(v, dict):
                count += count_params(v)
            else:
                count += v.size
        return count

    original_params = count_params(simple_model.parameters())

    result = apply_quantization(simple_model, method="fp16", verbose=False)

    assert result is simple_model  # Same object returned
    # Parameters should be unchanged
    new_params = count_params(simple_model.parameters())
    assert new_params == original_params


@pytest.mark.slow
def test_apply_quantization_4bit(simple_model):
    """Test 4-bit quantization."""
    # Initialize model weights
    mx.eval(simple_model.parameters())

    # Use group_size=32 since model has input_dims=32 (must be divisible)
    result = apply_quantization(simple_model, method="4bit", group_size=32, verbose=True)

    assert result is simple_model  # Modified in-place

    # Check that model has quantized layers
    info = get_quantization_info(simple_model)
    assert info["is_quantized"] is True
    assert info["num_quantized"] > 0


@pytest.mark.slow
def test_apply_quantization_8bit(simple_model):
    """Test 8-bit quantization."""
    # Initialize model weights
    mx.eval(simple_model.parameters())

    # Use group_size=32 since model has input_dims=32 (must be divisible)
    result = apply_quantization(simple_model, method="8bit", group_size=32, verbose=False)

    assert result is simple_model

    # Check that model has quantized layers
    info = get_quantization_info(simple_model)
    assert info["is_quantized"] is True
    assert info["num_quantized"] > 0


def test_apply_quantization_invalid_method(simple_model):
    """Test applying invalid quantization method."""
    with pytest.raises(ValueError, match="Unknown quantization method"):
        apply_quantization(simple_model, method="invalid_method")


def test_apply_quantization_requires_calibration(simple_model):
    """Test that calibration-requiring methods raise error."""
    with pytest.raises(ValueError, match="requires calibration"):
        apply_quantization(simple_model, method="gptq")

    with pytest.raises(ValueError, match="requires calibration"):
        apply_quantization(simple_model, method="awq")


def test_apply_quantization_no_quantizable_layers(non_quantizable_model):
    """Test applying quantization to model without quantizable layers."""
    with pytest.raises(RuntimeError, match="no quantizable layers"):
        apply_quantization(non_quantizable_model, method="4bit")


@pytest.mark.slow
def test_apply_quantization_custom_params(simple_model):
    """Test applying quantization with custom parameters."""
    mx.eval(simple_model.parameters())

    result = apply_quantization(
        simple_model,
        method="4bit",
        group_size=32,  # Must be divisible by input_dims (32)
        bits=4,
        mode="affine",
        verbose=False,
    )

    assert result is simple_model

    info = get_quantization_info(simple_model)
    assert info["is_quantized"] is True


# Test Model Size Estimation


def test_estimate_quantized_size_fp16(simple_model):
    """Test size estimation for FP16."""
    size_gb = estimate_quantized_size(simple_model, method="fp16")
    assert size_gb > 0
    # Should be roughly param_count * 2 bytes / 1e9


def test_estimate_quantized_size_4bit(simple_model):
    """Test size estimation for 4-bit."""
    size_fp16 = estimate_quantized_size(simple_model, method="fp16")
    size_4bit = estimate_quantized_size(simple_model, method="4bit")

    # 4-bit should be roughly 1/4 of FP16
    assert size_4bit < size_fp16
    assert size_4bit * 3 < size_fp16  # Should be significantly smaller


def test_estimate_quantized_size_8bit(simple_model):
    """Test size estimation for 8-bit."""
    size_fp16 = estimate_quantized_size(simple_model, method="fp16")
    size_8bit = estimate_quantized_size(simple_model, method="8bit")

    # 8-bit should be roughly 1/2 of FP16
    assert size_8bit < size_fp16
    assert size_8bit * 1.5 < size_fp16


# Test Quantization Info


def test_get_quantization_info_unquantized(simple_model):
    """Test getting info for unquantized model."""
    info = get_quantization_info(simple_model)

    assert info["is_quantized"] is False
    assert info["num_quantized"] == 0
    assert info["num_quantizable"] == 3  # fc1, fc2, fc3
    assert len(info["quantizable_layers"]) == 3
    assert len(info["quantized_layers"]) == 0


@pytest.mark.slow
def test_get_quantization_info_quantized(simple_model):
    """Test getting info for quantized model."""
    # Initialize and quantize
    mx.eval(simple_model.parameters())
    apply_quantization(simple_model, method="4bit", group_size=32, verbose=False)

    info = get_quantization_info(simple_model)

    assert info["is_quantized"] is True
    assert info["num_quantized"] > 0
    assert len(info["quantized_layers"]) > 0


def test_get_quantization_info_non_quantizable(non_quantizable_model):
    """Test getting info for model without quantizable layers."""
    info = get_quantization_info(non_quantizable_model)

    assert info["is_quantized"] is False
    assert info["num_quantized"] == 0
    assert info["num_quantizable"] == 0


# Integration Tests


@pytest.mark.slow
@pytest.mark.integration
def test_quantization_forward_pass(simple_model):
    """Test that quantized model can perform forward pass."""
    # Create input
    x = mx.random.normal((2, 32))

    # Get output from original model
    mx.eval(simple_model.parameters())
    output_original = simple_model(x)
    mx.eval(output_original)

    # Quantize model
    apply_quantization(simple_model, method="4bit", group_size=32, verbose=False)

    # Get output from quantized model
    output_quantized = simple_model(x)
    mx.eval(output_quantized)

    # Both should produce outputs of the same shape
    assert output_quantized.shape == output_original.shape
    assert output_quantized.shape == (2, 32)


@pytest.mark.slow
@pytest.mark.integration
def test_quantization_reduces_model_size(simple_model):
    """Test that quantization actually reduces model parameter size."""
    # Count bytes in nested parameter structure
    def count_bytes(params):
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += count_bytes(v)
            else:
                total += v.nbytes
        return total

    # Get original size
    mx.eval(simple_model.parameters())
    original_size = count_bytes(simple_model.parameters())

    # Quantize
    apply_quantization(simple_model, method="4bit", group_size=32, verbose=False)

    # Get quantized size
    quantized_size = count_bytes(simple_model.parameters())

    # Quantized should be smaller (though not exactly 1/4 due to scales/biases)
    assert quantized_size < original_size


@pytest.mark.slow
@pytest.mark.integration
def test_multiple_quantization_methods(simple_model):
    """Test applying different quantization methods."""
    methods = ["fp16", "8bit", "4bit"]

    for method in methods:
        # Create fresh model for each method
        model = SimpleModel(dim=32)
        mx.eval(model.parameters())

        # Apply quantization
        if method == "fp16":
            result = apply_quantization(model, method=method, verbose=False)
            assert result is model
        else:
            result = apply_quantization(model, method=method, group_size=32, verbose=False)
            assert result is model

            # Verify quantization was applied
            info = get_quantization_info(model)
            assert info["is_quantized"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
