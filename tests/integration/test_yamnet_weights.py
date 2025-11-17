#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for YAMNet weight loading and conversion.

Tests the complete weight loading pipeline:
- Downloading PyTorch weights
- Converting to MLX format
- Loading into model
- Verifying weight shapes
- Testing inference with real weights
"""

import pytest
import mlx.core as mx
import numpy as np
from pathlib import Path

from smlx.models.YAMNet import load, classify, extract_embeddings
from smlx.models.YAMNet.loader import (
    download_pytorch_weights,
    convert_pytorch_to_mlx,
    load_weights,
)
from smlx.models.YAMNet.weights import (
    validate_weight_shapes,
    count_parameters,
    get_expected_weight_shapes,
)
from smlx.models.YAMNet.model import YAMNet, count_parameters as model_count_parameters


@pytest.fixture(scope="module")
def temp_cache_dir(tmp_path_factory):
    """Create temporary cache directory for tests."""
    return tmp_path_factory.mktemp("yamnet_test_cache")


@pytest.mark.integration
@pytest.mark.requires_model
@pytest.mark.slow
class TestYAMNetWeightDownload:
    """Test weight downloading functionality."""

    def test_download_pytorch_weights(self, temp_cache_dir):
        """Test downloading PyTorch weights from torch_audioset."""
        pytorch_path = download_pytorch_weights(
            cache_dir=temp_cache_dir,
            force_download=False,
        )

        assert pytorch_path.exists(), "PyTorch weights file should exist"
        assert pytorch_path.suffix == ".pth", "Should be a .pth file"
        assert pytorch_path.stat().st_size > 0, "File should not be empty"

        # Verify it's cached (second call should be instant)
        pytorch_path_2 = download_pytorch_weights(
            cache_dir=temp_cache_dir,
            force_download=False,
        )
        assert pytorch_path == pytorch_path_2, "Should return same cached file"

    def test_download_force(self, temp_cache_dir):
        """Test force re-download of weights."""
        # First download
        pytorch_path_1 = download_pytorch_weights(
            cache_dir=temp_cache_dir,
            force_download=False,
        )

        # Force re-download
        pytorch_path_2 = download_pytorch_weights(
            cache_dir=temp_cache_dir,
            force_download=True,
        )

        assert pytorch_path_1 == pytorch_path_2, "Paths should be same"
        assert pytorch_path_2.exists(), "File should exist after re-download"


@pytest.mark.integration
@pytest.mark.requires_model
@pytest.mark.slow
class TestYAMNetWeightConversion:
    """Test PyTorch → MLX weight conversion."""

    @pytest.fixture(scope="class")
    def pytorch_weights_path(self, temp_cache_dir):
        """Download PyTorch weights for conversion tests."""
        return download_pytorch_weights(cache_dir=temp_cache_dir)

    def test_convert_pytorch_to_mlx(self, pytorch_weights_path):
        """Test converting PyTorch weights to MLX format."""
        pytest.importorskip("torch", reason="PyTorch required for conversion")

        mlx_weights = convert_pytorch_to_mlx(pytorch_weights_path)

        assert isinstance(mlx_weights, dict), "Should return dictionary"
        assert len(mlx_weights) > 0, "Should have converted weights"

        # Check some expected keys
        assert "conv1.weight" in mlx_weights, "Should have conv1 weights"
        assert "bn1.weight" in mlx_weights, "Should have bn1 weights"
        assert "classifier.weight" in mlx_weights, "Should have classifier weights"

    def test_converted_weight_shapes(self, pytorch_weights_path):
        """Test that converted weights have expected shapes."""
        pytest.importorskip("torch", reason="PyTorch required for conversion")

        mlx_weights = convert_pytorch_to_mlx(pytorch_weights_path)

        # Validate shapes
        is_valid, errors = validate_weight_shapes(mlx_weights, strict=False)

        assert is_valid or len(errors) < 5, \
            f"Weight shapes should be mostly valid. Errors: {errors}"

    def test_converted_weight_types(self, pytorch_weights_path):
        """Test that all converted weights are MLX arrays."""
        pytest.importorskip("torch", reason="PyTorch required for conversion")

        mlx_weights = convert_pytorch_to_mlx(pytorch_weights_path)

        for key, value in mlx_weights.items():
            assert isinstance(value, mx.array), \
                f"Weight {key} should be MLX array, got {type(value)}"

    def test_parameter_count(self, pytorch_weights_path):
        """Test that parameter count matches expected (3.7M)."""
        pytest.importorskip("torch", reason="PyTorch required for conversion")

        mlx_weights = convert_pytorch_to_mlx(pytorch_weights_path)
        param_count = count_parameters(mlx_weights)

        # YAMNet should have ~3.7M parameters
        assert 3_500_000 < param_count < 4_000_000, \
            f"Expected ~3.7M parameters, got {param_count:,}"


@pytest.mark.integration
@pytest.mark.requires_model
@pytest.mark.slow
class TestYAMNetWeightLoading:
    """Test loading weights into YAMNet model."""

    def test_load_weights_function(self, temp_cache_dir):
        """Test load_weights function."""
        pytest.importorskip("torch", reason="PyTorch required for first-time conversion")

        weights = load_weights(cache_dir=temp_cache_dir, force_download=False)

        assert isinstance(weights, dict), "Should return weight dictionary"
        assert len(weights) > 0, "Should have weights"

        # Check weights are cached
        mlx_cache_path = temp_cache_dir / "yamnet_mlx.npz"
        assert mlx_cache_path.exists(), "Converted weights should be cached"

    def test_load_model_with_weights(self, temp_cache_dir):
        """Test loading complete YAMNet model with weights."""
        pytest.importorskip("torch", reason="PyTorch required for first-time conversion")

        model = load(cache_dir=temp_cache_dir)

        assert isinstance(model, YAMNet), "Should return YAMNet model"

        # Check model has weights loaded
        param_count = model_count_parameters(model)
        assert param_count > 3_000_000, \
            f"Model should have ~3.7M parameters, got {param_count:,}"

    def test_model_in_eval_mode(self, temp_cache_dir):
        """Test that loaded model is in eval mode."""
        pytest.importorskip("torch", reason="PyTorch required for first-time conversion")

        model = load(cache_dir=temp_cache_dir)

        # Model should be in eval mode (no dropout, batch norm uses running stats)
        assert not model.training, "Model should be in eval mode"


@pytest.mark.integration
@pytest.mark.requires_model
@pytest.mark.slow
class TestYAMNetInference:
    """Test inference with loaded weights."""

    @pytest.fixture(scope="class")
    def loaded_model(self, temp_cache_dir):
        """Load YAMNet model once for inference tests."""
        pytest.importorskip("torch", reason="PyTorch required for first-time conversion")
        return load(cache_dir=temp_cache_dir)

    def test_model_forward_pass(self, loaded_model):
        """Test forward pass produces expected output shape."""
        # Create dummy mel spectrogram patch (96 x 64)
        batch_size = 2
        patch = mx.random.normal((batch_size, 96, 64, 1))

        logits = loaded_model(patch)

        assert logits.shape == (batch_size, 521), \
            f"Expected shape (2, 521), got {logits.shape}"
        assert not mx.isnan(logits).any(), "Logits should not contain NaN"
        assert not mx.isinf(logits).any(), "Logits should not contain Inf"

    def test_extract_embeddings(self, loaded_model):
        """Test embedding extraction."""
        # Create dummy patch
        patch = mx.random.normal((1, 96, 64, 1))

        embeddings = loaded_model.extract_embeddings(patch)

        assert embeddings.shape == (1, 1024), \
            f"Expected embeddings shape (1, 1024), got {embeddings.shape}"
        assert not mx.isnan(embeddings).any(), "Embeddings should not contain NaN"

    def test_predict_proba(self, loaded_model):
        """Test probability prediction."""
        # Create dummy patch
        patch = mx.random.normal((1, 96, 64, 1))

        probs = loaded_model.predict_proba(patch)

        assert probs.shape == (1, 521), "Should have probabilities for 521 classes"
        assert mx.allclose(mx.sum(probs, axis=-1), mx.array([1.0]), atol=1e-5), \
            "Probabilities should sum to 1"
        assert (probs >= 0).all() and (probs <= 1).all(), \
            "All probabilities should be between 0 and 1"

    def test_classify_synthetic_audio(self, loaded_model):
        """Test classification with synthetic audio."""
        # Create synthetic audio (3 seconds at 16kHz)
        sample_rate = 16000
        duration = 3
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, sample_rate * duration))
        audio = audio.astype(np.float32)

        # Classify
        predictions = classify(loaded_model, audio, sample_rate=sample_rate, top_k=5)

        assert len(predictions) == 5, "Should return top-5 predictions"

        # Check predictions structure
        for pred in predictions:
            assert hasattr(pred, "label"), "Prediction should have label"
            assert hasattr(pred, "score"), "Prediction should have score"
            assert hasattr(pred, "class_id"), "Prediction should have class_id"
            assert 0 <= pred.score <= 1, "Score should be probability"
            assert 0 <= pred.class_id < 521, "Class ID should be valid"

        # Scores should be in descending order
        scores = [pred.score for pred in predictions]
        assert scores == sorted(scores, reverse=True), \
            "Predictions should be sorted by score"

    def test_extract_embeddings_api(self, loaded_model):
        """Test extract_embeddings API function."""
        # Create synthetic audio
        audio = np.random.randn(16000 * 2).astype(np.float32)

        embeddings = extract_embeddings(loaded_model, audio, sample_rate=16000)

        assert embeddings.ndim == 2, "Embeddings should be 2D (patches, embedding_size)"
        assert embeddings.shape[1] == 1024, "Embedding dimension should be 1024"
        assert embeddings.shape[0] > 0, "Should have at least one patch"


@pytest.mark.integration
class TestYAMNetWeightShapes:
    """Test weight shape expectations."""

    def test_expected_weight_shapes(self):
        """Test that get_expected_weight_shapes returns valid shapes."""
        expected_shapes = get_expected_weight_shapes()

        assert isinstance(expected_shapes, dict), "Should return dictionary"
        assert len(expected_shapes) > 100, "Should have many weight tensors"

        # Check some key shapes
        assert expected_shapes["conv1.weight"] == (32, 1, 3, 3)
        assert expected_shapes["bn1.weight"] == (32,)
        assert expected_shapes["classifier.weight"] == (521, 1024)
        assert expected_shapes["classifier.bias"] == (521,)

    def test_validate_weight_shapes_with_correct_shapes(self):
        """Test validation with correct shapes."""
        expected_shapes = get_expected_weight_shapes()

        # Create dummy weights with correct shapes
        dummy_weights = {}
        for key, shape in expected_shapes.items():
            dummy_weights[key] = mx.zeros(shape)

        is_valid, errors = validate_weight_shapes(dummy_weights, strict=True)

        assert is_valid, f"Validation should pass. Errors: {errors}"
        assert len(errors) == 0, "Should have no errors"

    def test_validate_weight_shapes_with_wrong_shapes(self):
        """Test validation with incorrect shapes."""
        wrong_weights = {
            "conv1.weight": mx.zeros((16, 1, 3, 3)),  # Wrong: should be (32, 1, 3, 3)
            "bn1.weight": mx.zeros((32,)),
            "classifier.weight": mx.zeros((521, 512)),  # Wrong: should be (521, 1024)
        }

        is_valid, errors = validate_weight_shapes(wrong_weights, strict=False)

        assert not is_valid, "Validation should fail with wrong shapes"
        assert len(errors) > 0, "Should report errors"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
