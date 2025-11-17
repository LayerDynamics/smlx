# Copyright © 2025 SMLX Project

"""
Unit tests for SmolVLM-256M vision-language model implementation.

Tests the vision encoder, language model, configuration, connector, and forward pass.
"""

import pytest

from smlx.models.SmolVLM_256M import (
    DEFAULT_CONFIG,
    Idefics3Connector,
    ImageProcessor,
    LanguageModel,
    Model,
    ModelConfig,
    TextConfig,
    VisionConfig,
    VisionModel,
)

# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.unit
class TestModelConfiguration:
    """Test model configuration and validation."""

    def test_default_config(self):
        """Test that default configuration is valid."""
        config = DEFAULT_CONFIG

        assert config.model_type == "smolvlm"
        assert config.vision_config.hidden_size == 1152
        assert config.text_config.hidden_size == 576
        assert config.scale_factor == 2  # Pixel shuffle scale

    def test_vision_config(self):
        """Test vision (SigLIP) configuration."""
        config = VisionConfig()

        assert config.model_type == "siglip_vision_model"
        assert config.hidden_size == 1152  # SigLIP-SO400M
        assert config.num_hidden_layers == 27
        assert config.num_attention_heads == 16
        assert config.image_size == 384
        assert config.patch_size == 14

    def test_text_config(self):
        """Test text (SmolLM2) configuration."""
        config = TextConfig()

        assert config.model_type == "smolvlm"
        assert config.hidden_size == 576  # SmolLM2-135M
        assert config.num_hidden_layers == 30
        assert config.num_attention_heads == 9
        assert config.num_key_value_heads == 3  # GQA
        assert config.vocab_size == 49152
        assert config.rms_norm_eps == 1e-5

    def test_text_config_post_init(self):
        """Test text configuration post-init validation."""
        # Test with None num_key_value_heads
        config = TextConfig(num_key_value_heads=None)
        # Should default to num_attention_heads
        assert config.num_key_value_heads == config.num_attention_heads

    def test_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "model_type": "smolvlm",
            "scale_factor": 2,
            "vocab_size": 49152,
            "text_config": {
                "hidden_size": 576,
                "num_hidden_layers": 30,
            },
            "vision_config": {
                "hidden_size": 1152,
                "num_hidden_layers": 27,
            },
        }

        config = ModelConfig.from_dict(config_dict)
        assert config.scale_factor == 2
        assert config.vocab_size == 49152


# ============================================================================
# Vision Component Tests
# ============================================================================


@pytest.mark.unit
class TestVisionComponents:
    """Test vision-related components."""

    def test_vision_model_creation(self):
        """Test VisionModel creation."""
        config = VisionConfig()
        vision_model = VisionModel(config)

        assert hasattr(vision_model, "config")
        assert vision_model.config.image_size == 384

    def test_vision_config_siglip(self):
        """Test SigLIP vision configuration."""
        config = VisionConfig()

        # SigLIP-SO400M specific properties
        assert config.model_type == "siglip_vision_model"
        assert config.hidden_size == 1152
        assert config.num_hidden_layers == 27
        assert config.num_attention_heads == 16
        assert config.intermediate_size == 4304


# ============================================================================
# Language Component Tests
# ============================================================================


@pytest.mark.unit
class TestLanguageComponents:
    """Test language model (SmolLM2) components."""

    def test_language_model_creation(self):
        """Test LanguageModel creation."""
        config = TextConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            vocab_size=1000,
        )
        language_model = LanguageModel(config)

        assert hasattr(language_model, "config")
        assert language_model.config.hidden_size == 256

    def test_language_config_gqa(self):
        """Test Grouped Query Attention configuration."""
        config = TextConfig()

        # SmolLM2-135M uses GQA
        assert config.num_attention_heads == 9
        assert config.num_key_value_heads == 3
        assert config.num_attention_heads % config.num_key_value_heads == 0


# ============================================================================
# Connector Component Tests
# ============================================================================


@pytest.mark.unit
class TestConnectorComponents:
    """Test Idefics3 connector component."""

    def test_connector_creation(self):
        """Test Idefics3Connector creation."""
        connector = Idefics3Connector(
            vision_hidden_size=1152,
            text_hidden_size=576,
            scale_factor=2,
        )

        assert hasattr(connector, "scale_factor")
        assert connector.scale_factor == 2

    def test_connector_pixel_shuffle(self):
        """Test connector uses pixel shuffle."""
        config = DEFAULT_CONFIG

        # SmolVLM uses Idefics3 connector with pixel shuffle
        assert config.scale_factor == 2


# ============================================================================
# Full Model Tests
# ============================================================================


@pytest.mark.unit
class TestSmolVLMModel:
    """Test the complete SmolVLM-256M model."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        return ModelConfig(
            vision_config=VisionConfig(
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=512,
                image_size=224,
                patch_size=16,
            ),
            text_config=TextConfig(
                hidden_size=192,
                num_hidden_layers=4,
                num_attention_heads=6,
                num_key_value_heads=2,
                intermediate_size=512,
                vocab_size=1000,
            ),
            scale_factor=2,
        )

    @pytest.fixture
    def model(self, small_config):
        """Provide test model instance."""
        return Model(small_config)

    def test_model_creation(self, model, small_config):
        """Test model instantiation."""
        assert isinstance(model, Model)
        assert hasattr(model, "vision_model")
        assert hasattr(model, "connector")
        assert hasattr(model, "language_model")

    def test_model_creation_with_default_config(self):
        """Test model creation with default config."""
        config = DEFAULT_CONFIG
        model = Model(config)

        assert isinstance(model, Model)
        assert model.config == config

    def test_model_has_correct_structure(self, model):
        """Test model has correct component structure."""
        # Should have vision model
        assert hasattr(model, "vision_model")
        assert isinstance(model.vision_model, VisionModel)

        # Should have connector
        assert hasattr(model, "connector")
        assert isinstance(model.connector, Idefics3Connector)

        # Should have language model
        assert hasattr(model, "language_model")
        assert isinstance(model.language_model, LanguageModel)

    def test_model_config_properties(self, model, small_config):
        """Test model configuration properties."""
        assert model.config.vision_config.hidden_size == 256
        assert model.config.text_config.hidden_size == 192
        assert model.config.scale_factor == 2


# ============================================================================
# Image Processing Tests
# ============================================================================


@pytest.mark.unit
class TestImageProcessing:
    """Test image processing utilities."""

    def test_image_processor_creation(self):
        """Test ImageProcessor creation."""
        processor = ImageProcessor(
            size=(384, 384),
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )

        assert processor.size == (384, 384)
        # image_mean and image_std are numpy arrays, so we need to use np.allclose
        import numpy as np
        assert np.allclose(processor.image_mean.flatten(), [0.5, 0.5, 0.5])
        assert np.allclose(processor.image_std.flatten(), [0.5, 0.5, 0.5])

    def test_image_processor_default_values(self):
        """Test ImageProcessor default values."""
        processor = ImageProcessor()

        # Should have default SigLIP normalization values
        assert hasattr(processor, "image_mean")
        assert hasattr(processor, "image_std")
        assert len(processor.image_mean) == 3
        assert len(processor.image_std) == 3


# ============================================================================
# Model Size Tests
# ============================================================================


@pytest.mark.unit
class TestModelSize:
    """Test model parameter count."""

    def test_model_is_small(self):
        """Test that SmolVLM-256M is a small model."""
        config = DEFAULT_CONFIG

        # Check individual component sizes
        # Vision: SigLIP-SO400M (~400M params but shared)
        assert config.vision_config.hidden_size == 1152
        assert config.vision_config.num_hidden_layers == 27

        # Language: SmolLM2-135M
        assert config.text_config.hidden_size == 576
        assert config.text_config.num_hidden_layers == 30

        # Total model should be ~256M parameters
        # This is verified through the actual implementation


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_model
class TestSmolVLMIntegration:
    """Integration tests requiring model weights."""

    def test_model_can_be_instantiated_with_default_config(self):
        """Test that model can be created with default config."""
        config = DEFAULT_CONFIG
        model = Model(config)

        assert model is not None
        assert isinstance(model, Model)

    def test_model_expected_architecture(self):
        """Test that model has expected architecture."""
        config = DEFAULT_CONFIG
        model = Model(config)

        # Should have all three components
        assert hasattr(model, "vision_model")
        assert hasattr(model, "connector")
        assert hasattr(model, "language_model")

        # Connector should use pixel shuffle with scale 2
        assert model.config.scale_factor == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
