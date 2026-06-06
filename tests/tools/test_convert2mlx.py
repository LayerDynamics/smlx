"""
Tests for convert2mlx tool.

Tests cover:
- Weight sharding logic
- Save/load weights with sharding
- Model path resolution (local and HuggingFace)
- Multimodal module detection
- Quantization predicate building
- Full conversion pipeline
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.tools.convert2mlx import (
    build_quantization_predicate,
    get_model_path,
    make_shards,
    save_weights,
    skip_multimodal_module,
)


@pytest.mark.unit
class TestMakeShards:
    """Tests for weight sharding logic."""

    def test_no_sharding_for_small_weights(self):
        """Test that small weights don't get sharded."""
        # Create small weights (< 5GB)
        weights = {
            "layer1.weight": mx.random.normal((100, 100)),
            "layer2.weight": mx.random.normal((100, 100)),
        }

        shards = make_shards(weights, max_file_size_gb=5)

        # Should return single shard
        assert len(shards) == 1
        assert "layer1.weight" in shards[0]
        assert "layer2.weight" in shards[0]

    def test_sharding_for_large_weights(self):
        """Test that large weights get sharded properly."""
        # Create weights that exceed max_file_size
        # Note: max_file_size_gb must be int (used with bit shift <<)
        # Each 1000x1000 float32 = 4MB, so 5 layers = 20MB total
        max_file_size_gb = 1  # 1GB - but we'll pack them so only 1 shard
        # Create larger weights to force multiple shards (250MB each = 1.25GB total)
        weights = {f"layer{i}.weight": mx.random.normal((8000, 8000)) for i in range(5)}

        shards = make_shards(weights, max_file_size_gb=max_file_size_gb)

        # Should create multiple shards (each layer is ~250MB, max is 1GB)
        assert len(shards) > 1

        # All weights should be distributed across shards
        all_keys = set()
        for shard in shards:
            all_keys.update(shard.keys())
        assert all_keys == set(weights.keys())

    def test_empty_weights(self):
        """Test handling of empty weights dictionary."""
        weights = {}
        shards = make_shards(weights, max_file_size_gb=5)

        # Empty weights return empty list (no shard created)
        assert len(shards) == 0


@pytest.mark.unit
class TestSaveWeights:
    """Tests for save_weights function."""

    def test_save_single_shard(self):
        """Test saving weights without sharding."""
        weights = {
            "layer1.weight": mx.random.normal((10, 10)),
            "layer2.weight": mx.random.normal((10, 10)),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            save_weights(save_path, weights)

            # Should create model.safetensors
            assert (save_path / "model.safetensors").exists()

            # Load and verify
            loaded = mx.load(str(save_path / "model.safetensors"))
            assert set(loaded.keys()) == set(weights.keys())

    def test_save_with_metadata(self):
        """Test saving weights with metadata."""
        weights = {"layer.weight": mx.random.normal((10, 10))}
        metadata = {"model_type": "test", "version": "1.0"}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            save_weights(save_path, weights, metadata=metadata)

            # Metadata should be embedded in safetensors file
            assert (save_path / "model.safetensors").exists()

    def test_save_with_sharding(self):
        """Test saving weights with multiple shards."""
        # Force sharding with a tiny per-shard cap instead of materializing
        # multi-gigabyte tensors (the old version allocated ~5GB and wrote it to
        # disk, which is slow and fails in disk-constrained/CI environments).
        # Each 256x256 float32 array is ~256KB; with a ~512KB cap, 10 arrays
        # span several shards.
        weights = {f"layer{i}.weight": mx.random.normal((256, 256)) for i in range(10)}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            save_weights(save_path, weights, max_file_size_gb=0.0005)

            # The tiny threshold must force multiple shard files
            shard_files = list(save_path.glob("model-*.safetensors"))
            total_files = list(save_path.glob("model*.safetensors"))
            assert len(total_files) > 0
            assert len(shard_files) > 1, "tiny shard cap should produce >1 shard"

            # Sharding must produce an index file with a complete weight map
            assert (save_path / "model.safetensors.index.json").exists()
            with open(save_path / "model.safetensors.index.json") as f:
                index = json.load(f)
            assert "weight_map" in index
            assert "metadata" in index
            # Every weight must be mapped to some shard
            assert set(index["weight_map"].keys()) == set(weights.keys())


@pytest.mark.unit
class TestGetModelPath:
    """Tests for get_model_path function."""

    def test_local_path(self):
        """Test that local paths are returned as-is."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "model"
            test_path.mkdir()

            result = get_model_path(str(test_path))
            assert result == test_path

    def test_relative_path(self):
        """Test that relative paths are resolved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "model"
            test_path.mkdir()

            # Change to temp directory and use relative path
            import os

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = get_model_path("model")
                assert result.name == "model"
            finally:
                os.chdir(old_cwd)

    @patch("smlx.tools.convert2mlx.snapshot_download")
    def test_huggingface_download(self, mock_snapshot):
        """Test that HuggingFace models are downloaded."""
        mock_path = Path("/tmp/cached/model")
        mock_snapshot.return_value = str(mock_path)

        get_model_path("org/model-name")

        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs["repo_id"] == "org/model-name"
        assert call_kwargs["allow_patterns"] is not None

    @patch("smlx.tools.convert2mlx.snapshot_download")
    def test_huggingface_with_revision(self, mock_snapshot):
        """Test HuggingFace download with specific revision."""
        mock_path = Path("/tmp/cached/model")
        mock_snapshot.return_value = str(mock_path)

        get_model_path("org/model-name", revision="v1.0")

        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs["revision"] == "v1.0"


@pytest.mark.unit
class TestSkipMultimodalModule:
    """Tests for skip_multimodal_module function."""

    def test_skip_vision_modules(self):
        """Test that vision modules are detected."""
        assert skip_multimodal_module("model.vision_model.layer1") is True
        assert skip_multimodal_module("vision_tower.encoder") is True
        assert skip_multimodal_module("model.vision_encoder.block") is True
        assert skip_multimodal_module("model.visual.patch_embed") is True

    def test_skip_audio_modules(self):
        """Test that audio modules are detected."""
        assert skip_multimodal_module("model.audio_encoder.layer1") is True
        assert skip_multimodal_module("audio_tower.conv1") is True
        assert skip_multimodal_module("model.audio_model.block") is True

    def test_skip_sam_modules(self):
        """Test that SAM (Segment Anything Model) modules are detected."""
        assert skip_multimodal_module("model.sam_model.encoder") is True

    def test_allow_language_modules(self):
        """Test that language modules are not skipped."""
        assert skip_multimodal_module("model.layers.0.self_attn") is False
        assert skip_multimodal_module("lm_head.weight") is False
        assert skip_multimodal_module("model.embed_tokens") is False

    def test_allow_text_modules(self):
        """Test that text modules are not skipped."""
        assert skip_multimodal_module("text_encoder.layer.0") is False
        assert skip_multimodal_module("transformer.h.0.mlp") is False


@pytest.mark.unit
class TestBuildQuantizationPredicate:
    """Tests for build_quantization_predicate function."""

    def test_predicate_quantizes_linear(self):
        """Test that predicate allows Linear layer quantization."""
        predicate = build_quantization_predicate(group_size=64)

        linear = nn.Linear(128, 128)
        result = predicate("model.layer.linear", linear)

        # Should return True (quantize with default settings)
        assert result is True

    def test_predicate_quantizes_embedding(self):
        """Test that predicate allows Embedding quantization."""
        predicate = build_quantization_predicate(group_size=64)

        embedding = nn.Embedding(1000, 128)
        result = predicate("model.embed_tokens", embedding)

        assert result is True

    def test_predicate_skips_non_quantizable(self):
        """Test that predicate skips non-quantizable layers."""
        predicate = build_quantization_predicate(group_size=64)

        layernorm = nn.LayerNorm(128)
        result = predicate("model.layer.norm", layernorm)

        # Should return False (no to_quantized method)
        assert result is False

    def test_predicate_skips_multimodal_when_enabled(self):
        """Test that predicate skips multimodal modules when flag is set."""
        predicate = build_quantization_predicate(group_size=64, skip_multimodal=True)

        linear = nn.Linear(128, 128)
        result = predicate("model.vision_model.encoder.linear", linear)

        # Should return False (multimodal module)
        assert result is False

    def test_predicate_allows_multimodal_when_disabled(self):
        """Test that predicate allows multimodal modules when flag is False."""
        predicate = build_quantization_predicate(group_size=64, skip_multimodal=False)

        linear = nn.Linear(128, 128)
        result = predicate("model.vision_model.encoder.linear", linear)

        # Should return True (skip_multimodal=False)
        assert result is True

    def test_predicate_returns_custom_config(self):
        """Test that predicate can return custom quantization config."""
        predicate = build_quantization_predicate(group_size=128, skip_multimodal=False)

        linear = nn.Linear(256, 256)
        # The predicate returns True for default quantization settings
        # or a dict with custom settings
        result = predicate("model.layer.linear", linear)

        assert result is True or (isinstance(result, dict) and "group_size" in result)


@pytest.mark.integration
@pytest.mark.slow
class TestConvert:
    """Integration tests for convert function."""

    @patch("smlx.tools.convert2mlx.get_model_path")
    def test_convert_requires_config(self, mock_get_path):
        """
        Test that convert() requires config.json.

        Note: This is a placeholder test. Full testing of convert()
        requires actual model files or extensive mocking.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock get_model_path to return a directory without config.json
            fake_path = Path(tmpdir) / "model_without_config"
            fake_path.mkdir()
            mock_get_path.return_value = fake_path

            from smlx.tools.convert2mlx import convert

            with pytest.raises(ValueError, match="config.json not found"):
                convert(
                    hf_path="nonexistent/model",
                    mlx_path=str(Path(tmpdir) / "output"),
                    quantize=False,
                )

    @patch("smlx.tools.convert2mlx.get_model_path")
    def test_convert_loads_config(self, mock_get_path):
        """Test that convert() loads config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()

            # Create a config.json
            config_path = model_path / "config.json"
            config_path.write_text('{"model_type": "test"}')

            mock_get_path.return_value = model_path

            from smlx.tools.convert2mlx import convert

            # Should fail later due to missing weights, but config loading succeeds
            # We can't test full conversion without a real model
            with pytest.raises((FileNotFoundError, ValueError)):
                convert(
                    hf_path="test/model",
                    mlx_path=str(Path(tmpdir) / "output"),
                    quantize=False,
                )

            # At minimum, convert() was called successfully
            mock_get_path.assert_called_once()


@pytest.mark.unit
class TestQuantizeModel:
    """Tests for quantize_model function."""

    def test_quantize_model_updates_config(self):
        """Test that quantize_model updates config with quantization info."""
        from smlx.tools.convert2mlx import quantize_model

        weights = {"layer.weight": mx.random.normal((128, 128)).astype(mx.float32)}
        config = {"model_type": "test"}

        result_weights, result_config = quantize_model(weights, config, group_size=64, bits=4)

        # Weights must be REALLY quantized: the weight becomes packed uint32 with
        # scales/biases siblings (not returned as-is, which was the old no-op bug).
        assert result_weights["layer.weight"].dtype == mx.uint32
        assert "layer.scales" in result_weights
        assert "layer.biases" in result_weights

        # Config should have quantization info
        assert "quantization" in result_config
        assert result_config["quantization"]["bits"] == 4
        assert result_config["quantization"]["group_size"] == 64

    def test_quantize_model_returns_tuple(self):
        """Test that quantize_model returns (weights, config) tuple."""
        from smlx.tools.convert2mlx import quantize_model

        weights = {"layer.weight": mx.random.normal((64, 64))}
        config = {"model_type": "test"}

        result = quantize_model(weights, config)

        assert isinstance(result, tuple)
        assert len(result) == 2
        result_weights, result_config = result
        assert isinstance(result_weights, dict)
        assert isinstance(result_config, dict)
