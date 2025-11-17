# Copyright © 2025 SMLX Project

"""
Unit tests for MiniLM sentence embedding model implementation.

Tests the BERT-based sentence transformer for text embeddings.
"""

import mlx.core as mx
import pytest

from smlx.models.MiniLM import (
    DEFAULT_CONFIG_L6,
    DEFAULT_CONFIG_L12,
    MiniLM,
    ModelConfig,
    cosine_similarity,
    mean_pooling,
    normalize_embeddings,
)

# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.unit
class TestModelConfiguration:
    """Test model configuration and validation."""

    def test_default_config_l6(self):
        """Test that default L6 configuration is valid."""
        config = DEFAULT_CONFIG_L6

        assert config.model_type == "bert"
        assert config.hidden_size == 384
        assert config.num_hidden_layers == 6
        assert config.num_attention_heads == 12
        assert config.intermediate_size == 1536
        assert config.vocab_size == 30522

    def test_default_config_l12(self):
        """Test that default L12 configuration is valid."""
        config = DEFAULT_CONFIG_L12

        assert config.model_type == "bert"
        assert config.hidden_size == 384
        assert config.num_hidden_layers == 12  # More layers than L6
        assert config.num_attention_heads == 12
        assert config.vocab_size == 30522

    def test_config_pooling_settings(self):
        """Test pooling configuration settings."""
        config = DEFAULT_CONFIG_L6

        assert config.pooling_mode_mean_tokens is True
        assert config.pooling_mode_cls_token is False
        assert config.pooling_mode_max_tokens is False
        assert config.normalize_embeddings is True

    def test_config_bert_settings(self):
        """Test BERT-specific settings."""
        config = ModelConfig()

        assert config.max_position_embeddings == 512
        assert config.type_vocab_size == 2
        assert config.pad_token_id == 0
        assert config.hidden_act == "gelu"
        assert config.layer_norm_eps == 1e-12

    def test_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "model_type": "bert",
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "vocab_size": 30522,
            "pooling_mode_mean_tokens": True,
            "normalize_embeddings": True,
        }

        config = ModelConfig.from_dict(config_dict)
        assert config.hidden_size == 384
        assert config.num_hidden_layers == 6
        assert config.pooling_mode_mean_tokens is True


# ============================================================================
# Pooling Tests
# ============================================================================


@pytest.mark.unit
class TestPoolingFunctions:
    """Test pooling functions for embeddings."""

    def test_mean_pooling(self):
        """Test mean pooling over sequence."""
        batch_size = 2
        seq_len = 10
        hidden_size = 384

        # Create test embeddings
        embeddings = mx.random.normal((batch_size, seq_len, hidden_size))

        # Create attention mask
        attention_mask = mx.ones((batch_size, seq_len))

        # Apply mean pooling
        pooled = mean_pooling(embeddings, attention_mask)

        # Check output shape
        assert pooled.shape == (batch_size, hidden_size)

    def test_normalize_embeddings(self):
        """Test embedding normalization."""
        batch_size = 2
        hidden_size = 384

        # Create test embeddings
        embeddings = mx.random.normal((batch_size, hidden_size))

        # Normalize
        normalized = normalize_embeddings(embeddings)

        # Check shape
        assert normalized.shape == embeddings.shape

        # Check that vectors have unit length (approximately)
        norms = mx.sqrt(mx.sum(normalized * normalized, axis=-1))
        assert mx.allclose(norms, mx.ones(batch_size), atol=1e-5)


# ============================================================================
# Similarity Tests
# ============================================================================


@pytest.mark.unit
class TestSimilarityFunctions:
    """Test similarity computation functions."""

    def test_cosine_similarity_shape(self):
        """Test cosine similarity output shape."""
        batch_size_a = 2
        batch_size_b = 3
        hidden_size = 384

        embeddings_a = mx.random.normal((batch_size_a, hidden_size))
        embeddings_b = mx.random.normal((batch_size_b, hidden_size))

        # Compute similarity
        similarity = cosine_similarity(embeddings_a, embeddings_b)  # type: ignore[arg-type]

        # Check shape
        assert similarity.shape == (batch_size_a, batch_size_b)

    def test_cosine_similarity_range(self):
        """Test cosine similarity value range."""
        hidden_size = 384

        # Create normalized embeddings
        embeddings_a = mx.random.normal((2, hidden_size))
        embeddings_b = mx.random.normal((2, hidden_size))

        embeddings_a = normalize_embeddings(embeddings_a)
        embeddings_b = normalize_embeddings(embeddings_b)

        # Compute similarity
        similarity = cosine_similarity(embeddings_a, embeddings_b)  # type: ignore[arg-type]

        # Cosine similarity should be in [-1, 1]
        assert mx.all(similarity >= -1.0)  # type: ignore[arg-type]
        assert mx.all(similarity <= 1.0)  # type: ignore[arg-type]

    def test_cosine_similarity_identity(self):
        """Test cosine similarity of identical vectors."""
        hidden_size = 384

        embeddings = mx.random.normal((2, hidden_size))
        embeddings = normalize_embeddings(embeddings)

        # Similarity with itself should be ~1
        similarity = cosine_similarity(embeddings, embeddings)  # type: ignore[arg-type]

        # Diagonal should be close to 1
        assert mx.allclose(mx.diag(similarity), mx.ones(2), atol=1e-5)  # type: ignore[arg-type]


# ============================================================================
# Model Component Tests
# ============================================================================


@pytest.mark.unit
class TestModelComponents:
    """Test individual model components."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        return ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            vocab_size=1000,
            max_position_embeddings=128,
        )

    def test_model_config_dimensions(self, small_config):
        """Test model configuration dimensions."""
        assert small_config.hidden_size == 128
        assert small_config.num_hidden_layers == 2
        assert small_config.num_attention_heads == 4

        # Check that hidden size is divisible by num heads
        assert small_config.hidden_size % small_config.num_attention_heads == 0


# ============================================================================
# Full Model Tests
# ============================================================================


@pytest.mark.unit
class TestMiniLMModel:
    """Test the complete MiniLM model."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        return ModelConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            vocab_size=1000,
            max_position_embeddings=128,
        )

    @pytest.fixture
    def model(self, small_config):
        """Provide test model instance."""
        return MiniLM(small_config)

    def test_model_creation(self, model, small_config):
        """Test model instantiation."""
        assert isinstance(model, MiniLM)
        assert hasattr(model, "config")
        assert model.config == small_config

    def test_model_creation_with_default_config(self):
        """Test model creation with default configs."""
        # Test L6 variant
        config_l6 = DEFAULT_CONFIG_L6
        model_l6 = MiniLM(config_l6)
        assert isinstance(model_l6, MiniLM)
        assert model_l6.config.num_hidden_layers == 6

        # Test L12 variant
        config_l12 = DEFAULT_CONFIG_L12
        model_l12 = MiniLM(config_l12)
        assert isinstance(model_l12, MiniLM)
        assert model_l12.config.num_hidden_layers == 12

    def test_model_has_correct_structure(self, model):
        """Test model has correct component structure."""
        # Should have embeddings
        assert hasattr(model, "embeddings")

        # Should have encoder layers
        assert hasattr(model, "encoder")


# ============================================================================
# Model Size Tests
# ============================================================================


@pytest.mark.unit
class TestModelSize:
    """Test model parameter count."""

    def test_model_is_small(self):
        """Test that MiniLM is a small model."""
        # L6 variant
        config_l6 = DEFAULT_CONFIG_L6
        assert config_l6.hidden_size == 384
        assert config_l6.num_hidden_layers == 6
        # MiniLM-L6 is ~22MB

        # L12 variant
        config_l12 = DEFAULT_CONFIG_L12
        assert config_l12.hidden_size == 384
        assert config_l12.num_hidden_layers == 12
        # MiniLM-L12 is ~120MB


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_model
class TestMiniLMIntegration:
    """Integration tests requiring model weights."""

    def test_model_can_be_instantiated_with_default_config(self):
        """Test that model can be created with default config."""
        config = DEFAULT_CONFIG_L6
        model = MiniLM(config)

        assert model is not None
        assert isinstance(model, MiniLM)

    def test_both_variants_can_be_instantiated(self):
        """Test that both L6 and L12 variants can be created."""
        # L6 variant
        config_l6 = DEFAULT_CONFIG_L6
        model_l6 = MiniLM(config_l6)
        assert model_l6 is not None

        # L12 variant
        config_l12 = DEFAULT_CONFIG_L12
        model_l12 = MiniLM(config_l12)
        assert model_l12 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
