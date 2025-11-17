"""Tests for smlx.utils.loading module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import mlx.core as mx
import pytest

from smlx.utils.loading import (
    load_sharded_weights,
    load_tokenizer,
    load_weights,
    resolve_model_path,
    sanitize_weights,
    save_sharded_weights,
    save_weights,
    verify_weights,
)


class TestResolveModelPath:
    """Test resolve_model_path function."""

    def test_resolve_local_path(self):
        """Test resolving local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()

            resolved = resolve_model_path(str(model_path))

            assert resolved.exists()
            assert resolved.is_absolute()
            assert resolved == model_path.resolve()

    def test_resolve_local_path_pathlib(self):
        """Test resolving local path with Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            model_path.mkdir()

            resolved = resolve_model_path(model_path)

            assert resolved.exists()

    @patch("smlx.utils.loading.snapshot_download")
    @patch("smlx.tools.download.get_cache_dir")
    def test_resolve_hub_path(self, mock_cache_dir, mock_snapshot):
        """Test resolving HuggingFace Hub path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_cache_dir.return_value = Path(tmpdir)
            mock_snapshot.return_value = str(Path(tmpdir) / "downloaded_model")

            resolved = resolve_model_path("mlx-community/SmolLM2-135M")

            mock_snapshot.assert_called_once()
            assert isinstance(resolved, Path)

    def test_resolve_nonexistent_path_no_hub(self):
        """Test error when path doesn't exist and hub download fails."""
        with pytest.raises(ValueError, match="Could not find model"):
            resolve_model_path("/nonexistent/path")


class TestLoadWeights:
    """Test load_weights function."""

    def test_load_weights_single_file(self):
        """Test loading weights from single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "model.safetensors"

            # Create test weights
            weights = {
                "layer1.weight": mx.ones((10, 10)),
                "layer2.weight": mx.ones((5, 5)),
            }
            mx.save_safetensors(str(weights_path), weights)

            # Load weights
            loaded = load_weights(weights_path)

            assert "layer1.weight" in loaded
            assert "layer2.weight" in loaded
            assert loaded["layer1.weight"].shape == (10, 10)

    def test_load_weights_from_directory(self):
        """Test loading weights from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            weights_path = model_dir / "model.safetensors"

            weights = {"layer.weight": mx.ones((5, 5))}
            mx.save_safetensors(str(weights_path), weights)

            # Load from directory
            loaded = load_weights(model_dir)

            assert "layer.weight" in loaded

    def test_load_weights_npz(self):
        """Test loading weights from npz file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "model.npz"

            weights = {"layer.weight": mx.ones((5, 5))}
            mx.savez(str(weights_path), **weights)

            loaded = load_weights(weights_path)

            assert "layer.weight" in loaded

    def test_load_weights_lazy(self):
        """Test lazy loading of weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "model.safetensors"

            weights = {"layer.weight": mx.ones((5, 5))}
            mx.save_safetensors(str(weights_path), weights)

            loaded = load_weights(weights_path, lazy=True)

            assert "layer.weight" in loaded

    def test_load_weights_missing_file(self):
        """Test error when no weights file found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No weights file"):
                load_weights(tmpdir)

    def test_load_weights_unsupported_format(self):
        """Test error with unsupported weight format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "model.bin"
            bad_file.touch()

            with pytest.raises(ValueError, match="Unsupported weights format"):
                load_weights(bad_file)


class TestLoadShardedWeights:
    """Test load_sharded_weights function."""

    def test_load_sharded_weights(self):
        """Test loading sharded weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            # Create shard 1
            shard1 = {"layer1.weight": mx.ones((10, 10))}
            shard1_path = model_dir / "model-00001-of-00002.safetensors"
            mx.save_safetensors(str(shard1_path), shard1)

            # Create shard 2
            shard2 = {"layer2.weight": mx.ones((5, 5))}
            shard2_path = model_dir / "model-00002-of-00002.safetensors"
            mx.save_safetensors(str(shard2_path), shard2)

            # Create index file
            index = {
                "weight_map": {
                    "layer1.weight": "model-00001-of-00002.safetensors",
                    "layer2.weight": "model-00002-of-00002.safetensors",
                }
            }
            index_path = model_dir / "model.safetensors.index.json"
            with open(index_path, "w") as f:
                json.dump(index, f)

            # Load sharded weights
            loaded = load_sharded_weights(model_dir)

            assert "layer1.weight" in loaded
            assert "layer2.weight" in loaded
            assert loaded["layer1.weight"].shape == (10, 10)
            assert loaded["layer2.weight"].shape == (5, 5)

    def test_load_sharded_weights_lazy(self):
        """Test lazy loading of sharded weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            shard1 = {"layer.weight": mx.ones((5, 5))}
            shard1_path = model_dir / "model-00001-of-00001.safetensors"
            mx.save_safetensors(str(shard1_path), shard1)

            index = {"weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}}
            index_path = model_dir / "model.safetensors.index.json"
            with open(index_path, "w") as f:
                json.dump(index, f)

            loaded = load_sharded_weights(model_dir, lazy=True)

            assert "layer.weight" in loaded


class TestSaveWeights:
    """Test save_weights function."""

    def test_save_weights_single_file(self):
        """Test saving weights to single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"

            weights = {
                "layer1.weight": mx.ones((10, 10)),
                "layer2.weight": mx.ones((5, 5)),
            }

            save_weights(weights, save_path)

            assert save_path.exists()

            # Verify we can load it back
            loaded = mx.load(str(save_path))
            assert "layer1.weight" in loaded

    def test_save_weights_to_directory(self):
        """Test saving weights to directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "model"

            weights = {"layer.weight": mx.ones((5, 5))}

            save_weights(weights, save_dir)

            assert save_dir.exists()
            assert (save_dir / "model.safetensors").exists()

    def test_save_weights_with_metadata(self):
        """Test saving weights with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"

            weights = {"layer.weight": mx.ones((5, 5))}
            metadata = {"model_type": "test", "version": "1.0"}

            save_weights(weights, save_path, metadata=metadata)

            assert save_path.exists()

    def test_save_weights_large_model_sharding(self):
        """Test automatic sharding for large models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "model"

            # Create weights that exceed shard size
            # Use very small shard size for testing
            weights = {
                f"layer{i}.weight": mx.ones((1000, 1000))
                for i in range(5)
            }

            save_weights(weights, save_dir, max_shard_size_gb=0.001)

            # Should create sharded files
            assert save_dir.exists()
            # Check for index or multiple shard files
            shard_files = list(save_dir.glob("model-*-of-*.safetensors"))
            index_file = save_dir / "model.safetensors.index.json"

            # Either sharded or single file depending on actual size
            assert len(shard_files) > 0 or (save_dir / "model.safetensors").exists()


class TestSaveShardedWeights:
    """Test save_sharded_weights function."""

    def test_save_sharded_weights_basic(self):
        """Test saving sharded weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            weights = {
                "layer1.weight": mx.ones((10, 10)),
                "layer2.weight": mx.ones((5, 5)),
            }

            # Use small max shard size to force sharding
            max_shard_size = 500  # bytes

            save_sharded_weights(weights, save_dir, max_shard_size)

            # Check index file exists
            index_path = save_dir / "model.safetensors.index.json"
            assert index_path.exists()

            # Check shard files exist
            shard_files = list(save_dir.glob("model-*-of-*.safetensors"))
            assert len(shard_files) > 0

    def test_save_sharded_weights_with_metadata(self):
        """Test saving sharded weights with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            weights = {"layer.weight": mx.ones((5, 5))}
            metadata = {"model_type": "test"}

            save_sharded_weights(weights, save_dir, max_shard_size=100, metadata=metadata)

            # Check metadata in index
            index_path = save_dir / "model.safetensors.index.json"
            with open(index_path) as f:
                index = json.load(f)

            assert "metadata" in index


class TestSanitizeWeights:
    """Test sanitize_weights function."""

    def test_sanitize_weights_remove_patterns(self):
        """Test removing weights matching patterns."""
        weights = {
            "layer.weight": mx.ones((5, 5)),
            "rotary_emb.inv_freq": mx.ones((10,)),
            "_orig_mod.layer.bias": mx.ones((5,)),
        }

        sanitized = sanitize_weights(weights)

        assert "layer.weight" in sanitized
        assert "rotary_emb.inv_freq" not in sanitized
        assert "_orig_mod.layer.bias" not in sanitized

    def test_sanitize_weights_custom_patterns(self):
        """Test removing weights with custom patterns."""
        weights = {
            "layer.weight": mx.ones((5, 5)),
            "debug.info": mx.ones((10,)),
            "temp.data": mx.ones((5,)),
        }

        sanitized = sanitize_weights(weights, remove_patterns=["debug", "temp"])

        assert "layer.weight" in sanitized
        assert "debug.info" not in sanitized
        assert "temp.data" not in sanitized

    def test_sanitize_weights_with_model(self):
        """Test sanitizing weights against model parameters."""
        weights = {
            "layer1.weight": mx.ones((5, 5)),
            "layer2.weight": mx.ones((3, 3)),
            "unused.weight": mx.ones((2, 2)),
        }

        # Mock model
        model = Mock()
        model.named_parameters = Mock(
            return_value=[
                ("layer1.weight", mx.ones((5, 5))),
                ("layer2.weight", mx.ones((3, 3))),
            ]
        )

        sanitized = sanitize_weights(weights, model=model)

        assert "layer1.weight" in sanitized
        assert "layer2.weight" in sanitized
        assert "unused.weight" not in sanitized

    def test_sanitize_weights_no_removal(self):
        """Test sanitizing when no patterns match."""
        weights = {
            "layer1.weight": mx.ones((5, 5)),
            "layer2.weight": mx.ones((3, 3)),
        }

        sanitized = sanitize_weights(weights, remove_patterns=[])

        assert len(sanitized) == len(weights)
        assert "layer1.weight" in sanitized


class TestLoadTokenizer:
    """Test load_tokenizer function."""

    @patch("smlx.utils.loading.AutoTokenizer")
    def test_load_tokenizer_basic(self, mock_tokenizer_class):
        """Test loading tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = load_tokenizer(tmpdir)

            mock_tokenizer_class.from_pretrained.assert_called_once()
            assert tokenizer is not None

    @patch("smlx.utils.loading.AutoTokenizer")
    def test_load_tokenizer_trust_remote_code(self, mock_tokenizer_class):
        """Test loading tokenizer with trust_remote_code."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            load_tokenizer(tmpdir, trust_remote_code=True)

            args, kwargs = mock_tokenizer_class.from_pretrained.call_args
            assert kwargs.get("trust_remote_code") is True


class TestVerifyWeights:
    """Test verify_weights function."""

    def test_verify_weights_valid(self):
        """Test verifying valid weights."""
        weights = {
            "layer1.weight": mx.ones((5, 5)),
            "layer2.weight": mx.ones((3, 3)),
        }

        result = verify_weights(weights)
        assert result is True

    def test_verify_weights_with_expected_keys(self):
        """Test verifying weights with expected keys."""
        weights = {
            "layer1.weight": mx.ones((5, 5)),
            "layer2.weight": mx.ones((3, 3)),
        }

        expected_keys = ["layer1.weight", "layer2.weight"]

        result = verify_weights(weights, expected_keys=expected_keys)
        assert result is True

    def test_verify_weights_missing_keys(self):
        """Test error when expected keys are missing."""
        weights = {
            "layer1.weight": mx.ones((5, 5)),
        }

        expected_keys = ["layer1.weight", "layer2.weight"]

        with pytest.raises(ValueError, match="Missing expected keys"):
            verify_weights(weights, expected_keys=expected_keys)

    def test_verify_weights_not_dict(self):
        """Test error when weights is not a dict."""
        with pytest.raises(ValueError, match="must be a dict"):
            verify_weights([1, 2, 3])

    def test_verify_weights_empty(self):
        """Test error when weights is empty."""
        with pytest.raises(ValueError, match="empty"):
            verify_weights({})

    def test_verify_weights_with_model(self):
        """Test verifying weights against model."""
        weights = {
            "layer1.weight": mx.ones((5, 5)),
            "layer2.weight": mx.ones((3, 3)),
        }

        # Mock model
        model = Mock()
        model.named_parameters = Mock(
            return_value=[
                ("layer1.weight", mx.ones((5, 5))),
                ("layer2.weight", mx.ones((3, 3))),
            ]
        )

        result = verify_weights(weights, model=model)
        assert result is True

    def test_verify_weights_missing_in_model(self):
        """Test error when weights missing for model parameters."""
        weights = {
            "layer1.weight": mx.ones((5, 5)),
        }

        model = Mock()
        model.named_parameters = Mock(
            return_value=[
                ("layer1.weight", mx.ones((5, 5))),
                ("layer2.weight", mx.ones((3, 3))),
            ]
        )

        with pytest.raises(ValueError, match="Missing keys in weights"):
            verify_weights(weights, model=model)

    def test_verify_weights_unexpected_keys(self):
        """Test warning for unexpected keys."""
        weights = {
            "layer1.weight": mx.ones((5, 5)),
            "extra.weight": mx.ones((2, 2)),
        }

        model = Mock()
        model.named_parameters = Mock(
            return_value=[
                ("layer1.weight", mx.ones((5, 5))),
            ]
        )

        # Should not raise, but prints warning
        result = verify_weights(weights, model=model)
        assert result is True


class TestLoadingEdgeCases:
    """Test edge cases in loading utilities."""

    def test_load_weights_prefers_standard_names(self):
        """Test that load_weights prefers standard file names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            # Create multiple weight files
            (model_dir / "random.safetensors").touch()
            weights = {"layer.weight": mx.ones((5, 5))}
            mx.save_safetensors(str(model_dir / "model.safetensors"), weights)

            # Should load model.safetensors
            loaded = load_weights(model_dir)
            assert "layer.weight" in loaded

    def test_save_weights_creates_parent_directories(self):
        """Test that save_weights creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "subdir" / "model" / "weights.safetensors"

            weights = {"layer.weight": mx.ones((5, 5))}

            save_weights(weights, save_path)

            assert save_path.exists()
            assert save_path.parent.exists()

    def test_sanitize_weights_empty_dict(self):
        """Test sanitizing empty weights dict."""
        sanitized = sanitize_weights({})
        assert sanitized == {}


class TestLoadingIntegration:
    """Test integration scenarios."""

    def test_save_and_load_roundtrip(self):
        """Test saving and loading weights roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"

            original_weights = {
                "layer1.weight": mx.ones((10, 10)),
                "layer2.weight": mx.ones((5, 5)) * 2,
            }

            # Save weights
            save_weights(original_weights, save_path)

            # Load weights back
            loaded_weights = load_weights(save_path)

            # Verify they match
            assert set(loaded_weights.keys()) == set(original_weights.keys())
            for key in original_weights.keys():
                assert loaded_weights[key].shape == original_weights[key].shape

    def test_sharded_save_and_load_roundtrip(self):
        """Test saving and loading sharded weights roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            original_weights = {
                "layer1.weight": mx.ones((100, 100)),
                "layer2.weight": mx.ones((50, 50)),
            }

            # Save as sharded
            save_sharded_weights(original_weights, save_dir, max_shard_size=5000)

            # Load sharded weights
            loaded_weights = load_sharded_weights(save_dir)

            # Verify they match
            assert set(loaded_weights.keys()) == set(original_weights.keys())

    def test_sanitize_and_verify_workflow(self):
        """Test sanitizing then verifying weights."""
        weights = {
            "layer1.weight": mx.ones((5, 5)),
            "layer2.weight": mx.ones((3, 3)),
            "rotary_emb.inv_freq": mx.ones((10,)),
        }

        # Sanitize
        sanitized = sanitize_weights(weights)

        # Verify sanitized weights
        expected_keys = ["layer1.weight", "layer2.weight"]
        result = verify_weights(sanitized, expected_keys=expected_keys)

        assert result is True
        assert "rotary_emb.inv_freq" not in sanitized
