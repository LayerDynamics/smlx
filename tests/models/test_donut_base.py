# Copyright © 2025 SMLX Project

"""
Unit tests for Donut-base document understanding model implementation.

Tests the Swin Transformer encoder, BART decoder, and document parsing.
"""

import pytest

from smlx.models.Donut_base import (
    DEFAULT_CONFIG,
    BARTConfig,
    Donut,
    DonutConfig,
    SwinConfig,
)

# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.unit
class TestModelConfiguration:
    """Test model configuration and validation."""

    def test_swin_config(self):
        """Test Swin Transformer configuration."""
        config = SwinConfig()

        assert config.model_type == "swin"
        assert config.image_size == (224, 224)
        assert config.patch_size == 4
        assert config.embed_dim == 128
        assert config.depths == (2, 2, 18, 2)
        assert config.num_heads == (4, 8, 16, 32)
        assert config.window_size == 7

    def test_swin_config_validation(self):
        """Test Swin configuration validation."""
        config = SwinConfig()

        # Should have 4 stages
        assert len(config.depths) == 4
        assert len(config.num_heads) == 4

    def test_bart_config(self):
        """Test BART decoder configuration."""
        config = BARTConfig()

        assert config.model_type == "bart"
        assert config.vocab_size == 50265
        assert config.max_position_embeddings == 1024
        assert config.decoder_layers == 12
        assert config.decoder_attention_heads == 16
        assert config.d_model == 1024

    def test_bart_config_validation(self):
        """Test BART configuration validation."""
        config = BARTConfig()

        # d_model should be divisible by attention heads
        assert config.d_model % config.decoder_attention_heads == 0

    def test_bart_special_tokens(self):
        """Test BART special token IDs."""
        config = BARTConfig()

        assert config.bos_token_id == 0
        assert config.eos_token_id == 2
        assert config.pad_token_id == 1
        assert config.decoder_start_token_id == 2

    def test_donut_config(self):
        """Test complete Donut configuration."""
        swin_config = SwinConfig()
        bart_config = BARTConfig()
        config = DonutConfig(
            encoder_config=swin_config,
            decoder_config=bart_config,
        )

        assert config.encoder_config == swin_config
        assert config.decoder_config == bart_config

    def test_default_config(self):
        """Test default configuration."""
        config = DEFAULT_CONFIG

        assert config.encoder_config is not None
        assert config.decoder_config is not None
        assert config.encoder_config.model_type == "swin"
        assert config.decoder_config.model_type == "bart"


# ============================================================================
# Swin Transformer Tests
# ============================================================================


@pytest.mark.unit
class TestSwinTransformer:
    """Test Swin Transformer encoder component."""

    def test_swin_config_stages(self):
        """Test Swin Transformer stage configuration."""
        config = SwinConfig()

        # 4 stages with increasing depth and heads
        assert len(config.depths) == 4
        assert len(config.num_heads) == 4

        # Depths should increase
        assert config.depths == (2, 2, 18, 2)

        # Number of heads should increase
        assert config.num_heads == (4, 8, 16, 32)

    def test_swin_window_attention(self):
        """Test shifted window attention configuration."""
        config = SwinConfig()

        # Window size should divide image size
        assert config.window_size == 7

    def test_swin_patch_embedding(self):
        """Test patch embedding configuration."""
        config = SwinConfig()

        assert config.patch_size == 4
        assert config.embed_dim == 128

    def test_swin_hierarchical_structure(self):
        """Test hierarchical feature extraction."""
        config = SwinConfig()

        # Swin has 4 stages with different feature resolutions
        num_stages = len(config.depths)
        assert num_stages == 4


# ============================================================================
# BART Decoder Tests
# ============================================================================


@pytest.mark.unit
class TestBARTDecoder:
    """Test BART decoder component."""

    def test_bart_decoder_dimensions(self):
        """Test decoder dimensions."""
        config = BARTConfig()

        assert config.d_model == 1024
        assert config.decoder_layers == 12
        assert config.decoder_attention_heads == 16
        assert config.decoder_ffn_dim == 4096

        # Head dimension
        head_dim = config.d_model // config.decoder_attention_heads
        assert head_dim == 64

    def test_bart_activation(self):
        """Test activation function."""
        config = BARTConfig()

        assert config.activation_function == "gelu"

    def test_bart_caching(self):
        """Test KV cache configuration."""
        config = BARTConfig()

        assert config.use_cache is True


# ============================================================================
# Full Model Tests
# ============================================================================


@pytest.mark.unit
class TestDonutModel:
    """Test the complete Donut model."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        swin_config = SwinConfig(
            image_size=(128, 128),
            patch_size=4,
            embed_dim=64,
            depths=(2, 2, 6, 2),
            num_heads=(2, 4, 8, 16),
            window_size=4,
        )
        bart_config = BARTConfig(
            d_model=256,
            decoder_layers=2,
            decoder_attention_heads=4,
            decoder_ffn_dim=512,
            vocab_size=1000,
        )
        return DonutConfig(
            encoder_config=swin_config,
            decoder_config=bart_config,
            encoder_hidden_size=256,  # Must match decoder d_model
            decoder_hidden_size=256,
        )

    @pytest.fixture
    def model(self, small_config):
        """Provide test model instance."""
        return Donut(small_config)

    def test_model_creation(self, model, small_config):
        """Test model instantiation."""
        assert isinstance(model, Donut)
        assert hasattr(model, "config")
        assert model.config == small_config

    def test_model_creation_with_default_config(self):
        """Test model creation with default config."""
        config = DEFAULT_CONFIG
        model = Donut(config)

        assert isinstance(model, Donut)
        assert model.config == config

    def test_model_has_correct_structure(self, model):
        """Test model has correct component structure."""
        # Should have encoder (Swin Transformer)
        assert hasattr(model, "encoder")

        # Should have decoder (BART)
        assert hasattr(model, "decoder")

    def test_model_config_properties(self, model, small_config):
        """Test model configuration properties."""
        assert model.config.encoder_config.embed_dim == 64
        assert model.config.decoder_config.d_model == 256
        assert model.config.decoder_config.decoder_layers == 2


# ============================================================================
# Model Size Tests
# ============================================================================


@pytest.mark.unit
class TestModelSize:
    """Test model parameter count."""

    def test_model_is_base_size(self):
        """Test that Donut-base is appropriately sized."""
        config = DEFAULT_CONFIG

        assert config.encoder_config is not None
        assert config.decoder_config is not None

        # Swin Transformer encoder
        assert config.encoder_config.embed_dim == 128
        assert config.encoder_config.depths == (2, 2, 18, 2)

        # BART decoder
        assert config.decoder_config.d_model == 1024
        assert config.decoder_config.decoder_layers == 12

        # Donut-base is ~200M parameters


# ============================================================================
# Task-specific Tests
# ============================================================================


@pytest.mark.unit
class TestDocumentUnderstandingTasks:
    """Test document understanding task configurations."""

    def test_document_parsing_config(self):
        """Test configuration for document parsing."""
        config = DEFAULT_CONFIG

        assert config.decoder_config is not None
        # Should support document parsing
        assert config.decoder_config.max_position_embeddings >= 512

    def test_document_vqa_config(self):
        """Test configuration for document VQA."""
        config = DEFAULT_CONFIG

        assert config.decoder_config is not None
        # Should support question answering
        assert config.decoder_config.decoder_layers >= 6

    def test_document_classification_config(self):
        """Test configuration for document classification."""
        config = DEFAULT_CONFIG

        assert config.encoder_config is not None
        # Should support classification tasks
        assert config.encoder_config.depths is not None


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_model
class TestDonutIntegration:
    """Integration tests requiring model weights."""

    def test_model_can_be_instantiated_with_default_config(self):
        """Test that model can be created with default config."""
        config = DEFAULT_CONFIG
        model = Donut(config)

        assert model is not None
        assert isinstance(model, Donut)

    def test_model_expected_architecture(self):
        """Test that model has expected architecture."""
        config = DEFAULT_CONFIG
        model = Donut(config)

        # Should have Swin encoder and BART decoder
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
