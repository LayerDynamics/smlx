# Copyright © 2025 SMLX Project

"""
Tests for ModelManager.
"""

from unittest.mock import Mock, patch

import pytest

from smlx.server.model_manager import ModelManager


@pytest.mark.unit
class TestModelManagerInit:
    """Tests for ModelManager initialization."""

    def test_init_default_cache_size(self):
        """Test ModelManager initialization with default cache size."""
        manager = ModelManager()
        assert manager.cache_size == 3
        assert manager.loaded_models == {}
        assert manager.model_types == {}

    def test_init_custom_cache_size(self):
        """Test ModelManager initialization with custom cache size."""
        manager = ModelManager(cache_size=5)
        assert manager.cache_size == 5


@pytest.mark.unit
class TestModelTypeInference:
    """Tests for model type inference."""

    def test_infer_smollm_from_registry(self):
        """Test inferring SmolLM from supported models registry."""
        manager = ModelManager()
        model_type = manager._infer_model_type("mlx-community/SmolLM2-135M-Instruct")
        assert model_type == "smollm"

    def test_infer_whisper_from_registry(self):
        """Test inferring Whisper from supported models registry."""
        manager = ModelManager()
        model_type = manager._infer_model_type("whisper-tiny")
        assert model_type == "whisper"

    def test_infer_embedding_from_registry(self):
        """Test inferring embedding model from registry."""
        manager = ModelManager()
        model_type = manager._infer_model_type("all-MiniLM-L6-v2")
        assert model_type == "embedding"

    def test_infer_vlm_from_registry(self):
        """Test inferring VLM from registry."""
        manager = ModelManager()
        model_type = manager._infer_model_type("SmolVLM-256M")
        assert model_type == "vlm"

    def test_infer_smollm_from_name(self):
        """Test inferring SmolLM from model name patterns."""
        manager = ModelManager()
        # Test various patterns
        assert manager._infer_model_type("custom-smollm-model") == "smollm"
        assert manager._infer_model_type("model-135m") == "smollm"
        assert manager._infer_model_type("model-360M") == "smollm"

    def test_infer_whisper_from_name(self):
        """Test inferring Whisper from model name patterns."""
        manager = ModelManager()
        assert manager._infer_model_type("custom-whisper-model") == "whisper"

    def test_infer_unsupported_model(self):
        """Test error when model type cannot be inferred."""
        manager = ModelManager()
        with pytest.raises(ValueError, match="Cannot infer model type"):
            manager._infer_model_type("unknown-model-xyz")


@pytest.mark.unit
@pytest.mark.asyncio
class TestModelLoading:
    """Tests for model loading functionality."""

    @patch("smlx.server.model_manager.ModelManager._load_smollm")
    async def test_load_smollm_model(self, mock_load_smollm):
        """Test loading a SmolLM model."""
        manager = ModelManager()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_smollm.return_value = (mock_model, mock_tokenizer)

        model, tokenizer = await manager.load_model("mlx-community/SmolLM2-135M-Instruct")

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_load_smollm.assert_called_once_with("mlx-community/SmolLM2-135M-Instruct")
        assert "mlx-community/SmolLM2-135M-Instruct" in manager.loaded_models
        assert manager.model_types["mlx-community/SmolLM2-135M-Instruct"] == "smollm"

    @patch("smlx.server.model_manager.ModelManager._load_whisper")
    async def test_load_whisper_model(self, mock_load_whisper):
        """Test loading a Whisper model."""
        manager = ModelManager()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_whisper.return_value = (mock_model, mock_tokenizer)

        model, tokenizer = await manager.load_model("whisper-tiny")

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_load_whisper.assert_called_once_with("whisper-tiny")
        assert "whisper-tiny" in manager.loaded_models

    @patch("smlx.server.model_manager.ModelManager._load_embedding")
    async def test_load_embedding_model_not_implemented(self, mock_load_embedding):
        """Test loading an embedding model raises NotImplementedError."""
        manager = ModelManager()
        mock_load_embedding.side_effect = NotImplementedError(
            "Embedding models not yet implemented"
        )

        with pytest.raises(NotImplementedError, match="Embedding models not yet implemented"):
            await manager.load_model("all-MiniLM-L6-v2")

    @patch("smlx.server.model_manager.ModelManager._load_vlm")
    async def test_load_vlm_model_not_implemented(self, mock_load_vlm):
        """Test loading a VLM raises NotImplementedError."""
        manager = ModelManager()
        mock_load_vlm.side_effect = NotImplementedError(
            "Vision-language models not yet implemented"
        )

        with pytest.raises(NotImplementedError, match="Vision-language models not yet implemented"):
            await manager.load_model("SmolVLM-256M")

    async def test_load_unsupported_model_type(self):
        """Test loading a model with unsupported type."""
        manager = ModelManager()

        with pytest.raises(ValueError, match="Cannot infer model type"):
            await manager.load_model("unsupported-model-xyz")

    @patch("smlx.server.model_manager.ModelManager._load_smollm")
    async def test_load_model_caching(self, mock_load_smollm):
        """Test that models are cached and not reloaded."""
        manager = ModelManager()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_smollm.return_value = (mock_model, mock_tokenizer)

        # Load model first time
        model1, tokenizer1 = await manager.load_model("mlx-community/SmolLM2-135M-Instruct")

        # Load same model again
        model2, tokenizer2 = await manager.load_model("mlx-community/SmolLM2-135M-Instruct")

        # Should return cached model
        assert model1 == model2
        assert tokenizer1 == tokenizer2
        # Should only call load once
        mock_load_smollm.assert_called_once()

    @patch("smlx.server.model_manager.ModelManager._load_smollm")
    async def test_explicit_model_type(self, mock_load_smollm):
        """Test loading model with explicit model type."""
        manager = ModelManager()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_smollm.return_value = (mock_model, mock_tokenizer)

        model, tokenizer = await manager.load_model("custom-model", model_type="smollm")

        assert model == mock_model
        mock_load_smollm.assert_called_once_with("custom-model")


@pytest.mark.unit
class TestCacheManagement:
    """Tests for model cache management."""

    @pytest.mark.asyncio
    @patch("smlx.server.model_manager.ModelManager._load_smollm")
    async def test_cache_eviction(self, mock_load_smollm):
        """Test that LRU model is evicted when cache is full."""
        manager = ModelManager(cache_size=2)

        # Create unique mock objects for each model
        mock_model1, mock_tokenizer1 = Mock(), Mock()
        mock_model2, mock_tokenizer2 = Mock(), Mock()
        mock_model3, mock_tokenizer3 = Mock(), Mock()

        # Return different mocks for each call
        mock_load_smollm.side_effect = [
            (mock_model1, mock_tokenizer1),
            (mock_model2, mock_tokenizer2),
            (mock_model3, mock_tokenizer3),
        ]

        # Load 2 models (fill cache) - use names that will be recognized as smollm
        await manager.load_model("smollm-1")
        await manager.load_model("smollm-2")

        assert len(manager.loaded_models) == 2
        assert "smollm-1" in manager.loaded_models
        assert "smollm-2" in manager.loaded_models

        # Load third model (should evict first)
        await manager.load_model("smollm-3")

        assert len(manager.loaded_models) == 2
        assert "smollm-1" not in manager.loaded_models  # Evicted
        assert "smollm-2" in manager.loaded_models
        assert "smollm-3" in manager.loaded_models

    def test_get_cached_model(self):
        """Test getting a cached model."""
        manager = ModelManager()
        mock_model = Mock()
        mock_tokenizer = Mock()

        # Manually add to cache
        manager.loaded_models["test-model"] = (mock_model, mock_tokenizer)

        result = manager.get_model("test-model")

        assert result == (mock_model, mock_tokenizer)

    def test_get_uncached_model(self):
        """Test getting a model that's not cached."""
        manager = ModelManager()

        result = manager.get_model("nonexistent-model")

        assert result is None

    @pytest.mark.asyncio
    async def test_unload_model(self):
        """Test unloading a model from cache."""
        manager = ModelManager()

        # Manually add to cache
        manager.loaded_models["test-model"] = (Mock(), Mock())
        manager.model_types["test-model"] = "smollm"

        success = await manager.unload_model("test-model")

        assert success is True
        assert "test-model" not in manager.loaded_models
        assert "test-model" not in manager.model_types

    @pytest.mark.asyncio
    async def test_unload_nonexistent_model(self):
        """Test unloading a model that doesn't exist."""
        manager = ModelManager()

        success = await manager.unload_model("nonexistent-model")

        assert success is False


@pytest.mark.unit
class TestModelListings:
    """Tests for model listing functions."""

    def test_list_loaded_models_empty(self):
        """Test listing loaded models when none are loaded."""
        manager = ModelManager()

        models = manager.list_loaded_models()

        assert models == []

    def test_list_loaded_models(self):
        """Test listing loaded models."""
        manager = ModelManager()

        # Manually add models to cache
        manager.loaded_models["model1"] = (Mock(), Mock())
        manager.loaded_models["model2"] = (Mock(), Mock())

        models = manager.list_loaded_models()

        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models

    def test_list_supported_models(self):
        """Test listing all supported models."""
        manager = ModelManager()

        supported = manager.list_supported_models()

        # Should return a copy of supported_models dict
        assert isinstance(supported, dict)
        assert "mlx-community/SmolLM2-135M-Instruct" in supported
        assert "whisper-tiny" in supported
        assert supported["mlx-community/SmolLM2-135M-Instruct"] == "smollm"
        assert supported["whisper-tiny"] == "whisper"


@pytest.mark.unit
@pytest.mark.asyncio
class TestCleanup:
    """Tests for cleanup functionality."""

    @patch("mlx.core.metal.clear_cache")
    async def test_cleanup(self, mock_clear_cache):
        """Test cleanup clears all models and caches."""
        manager = ModelManager()

        # Add some models
        manager.loaded_models["model1"] = (Mock(), Mock())
        manager.loaded_models["model2"] = (Mock(), Mock())
        manager.model_types["model1"] = "smollm"
        manager.model_types["model2"] = "whisper"

        await manager.cleanup()

        assert len(manager.loaded_models) == 0
        assert len(manager.model_types) == 0
        mock_clear_cache.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
class TestSpecificModelLoaders:
    """Tests for specific model loader methods."""

    @patch("smlx.models.SmolLM2_135M.load")
    async def test_load_smollm_135m(self, mock_load):
        """Test loading SmolLM2-135M model."""
        manager = ModelManager()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        model, tokenizer = await manager._load_smollm("mlx-community/SmolLM2-135M-Instruct")

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_load.assert_called_once_with("mlx-community/SmolLM2-135M-Instruct")

    @patch("smlx.models.SmolLM2_135M.load")
    async def test_load_smollm_135m_alias(self, mock_load):
        """Test loading SmolLM2-135M using alias."""
        manager = ModelManager()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Use alias "SmolLM2-135M" which should be normalized
        model, tokenizer = await manager._load_smollm("SmolLM2-135M")

        assert model == mock_model
        mock_load.assert_called_once_with("mlx-community/SmolLM2-135M-Instruct")

    @patch("smlx.models.SmolLM2_360M.load")
    async def test_load_smollm_360m(self, mock_load):
        """Test loading SmolLM2-360M model."""
        manager = ModelManager()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        model, tokenizer = await manager._load_smollm("mlx-community/SmolLM2-360M-Instruct")

        assert model == mock_model
        mock_load.assert_called_once_with("mlx-community/SmolLM2-360M-Instruct")

    async def test_load_smollm_unknown_variant(self):
        """Test loading unknown SmolLM variant."""
        manager = ModelManager()

        with pytest.raises(ValueError, match="Unknown SmolLM variant"):
            await manager._load_smollm("SmolLM2-999M")

    @patch("smlx.models.Whisper_tiny.load")
    async def test_load_whisper_tiny(self, mock_load):
        """Test loading Whisper-tiny model."""
        manager = ModelManager()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        model, tokenizer = await manager._load_whisper("whisper-tiny")

        assert model == mock_model
        mock_load.assert_called_once_with("whisper-tiny")

    async def test_load_embedding_not_implemented(self):
        """Test loading embedding model raises NotImplementedError."""
        manager = ModelManager()

        with pytest.raises(NotImplementedError, match="Embedding models not yet implemented"):
            await manager._load_embedding("all-MiniLM-L6-v2")

    async def test_load_vlm_not_implemented(self):
        """Test loading VLM raises NotImplementedError."""
        manager = ModelManager()

        with pytest.raises(NotImplementedError, match="Vision-language models not yet implemented"):
            await manager._load_vlm("SmolVLM-256M")
