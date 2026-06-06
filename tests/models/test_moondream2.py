# Copyright © 2025 SMLX Project

"""
Unit tests for Moondream2 vision-language model implementation.

Tests the vision encoder, language model, configuration, and forward pass.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.models.Moondream2 import (
    Moondream2,
    VisionProjector,
    VisionEncoder,
    PhiModel,
    PhiAttention,
    PhiMLP,
    DetectionHead,
    CoordinateDecoder,
    ModelConfig,
    VisionConfig,
    TextConfig,
    RegionConfig,
    DEFAULT_CONFIG_2B,
    DEFAULT_CONFIG_05B,
)


# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.unit
class TestModelConfiguration:
    """Test model configuration and validation."""

    def test_default_config_2b(self):
        """Test that default 2B configuration is valid."""
        config = DEFAULT_CONFIG_2B

        assert config.model_type == "moondream"
        assert config.variant == "2b"
        assert config.vision_config.hidden_size == 1152
        assert config.text_config.hidden_size == 2048
        assert config.text_config.num_hidden_layers == 24

    def test_default_config_05b(self):
        """Test that default 0.5B configuration is valid."""
        config = DEFAULT_CONFIG_05B

        assert config.model_type == "moondream"
        assert config.variant == "0.5b"
        assert config.vision_config.hidden_size == 768
        assert config.text_config.hidden_size == 1024
        assert config.text_config.num_hidden_layers == 16

    def test_vision_config(self):
        """Test vision configuration."""
        config = VisionConfig()

        assert config.model_type == "moondream_vision"
        assert config.image_size == 378
        assert config.patch_size == 14
        assert config.use_tiling is True
        assert config.max_crops == 4

    def test_text_config(self):
        """Test text (Phi) configuration."""
        config = TextConfig()

        assert config.model_type == "phi"
        assert config.vocab_size == 51200
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 24
        assert config.hidden_act == "gelu_new"
        assert config.partial_rotary_factor == 0.5

    def test_region_config(self):
        """Test region/detection configuration."""
        config = RegionConfig()

        assert config.use_fourier_features is True
        assert config.fourier_feature_dim == 256
        assert config.max_detections == 100
        assert config.confidence_threshold == 0.5

    def test_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "model_type": "moondream",
            "variant": "2b",
            "vision_config": {
                "hidden_size": 1152,
                "num_hidden_layers": 27,
            },
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
            },
            "region_config": {
                "max_detections": 100,
            },
        }

        config = ModelConfig.from_dict(config_dict)
        assert config.vision_config.hidden_size == 1152
        assert config.text_config.hidden_size == 2048
        assert config.region_config.max_detections == 100


# ============================================================================
# Vision Component Tests
# ============================================================================


@pytest.mark.unit
class TestVisionComponents:
    """Test vision-related components."""

    def test_vision_projector_creation(self):
        """Test VisionProjector module creation."""
        projector = VisionProjector(vision_hidden_size=1152, text_hidden_size=2048)

        assert hasattr(projector, "fc1")
        assert hasattr(projector, "fc2")
        assert hasattr(projector, "activation")
        assert isinstance(projector.fc1, nn.Linear)
        assert isinstance(projector.fc2, nn.Linear)

    def test_vision_projector_forward(self):
        """Test VisionProjector forward pass."""
        projector = VisionProjector(vision_hidden_size=1152, text_hidden_size=2048)

        batch_size = 2
        num_patches = 50
        # VisionProjector expects concatenated global + local features (2 * 1152 = 2304)
        vision_features = mx.random.normal((batch_size, num_patches, 2304))

        output = projector(vision_features)

        assert output.shape == (batch_size, num_patches, 2048)
        assert output.dtype == vision_features.dtype

    def test_vision_encoder_creation(self):
        """Test VisionEncoder module creation."""
        config = VisionConfig()
        encoder = VisionEncoder(config)

        assert hasattr(encoder, "embeddings")
        assert encoder.config.image_size == 378
        assert encoder.config.patch_size == 14


# ============================================================================
# Language Model Component Tests
# ============================================================================


@pytest.mark.unit
class TestLanguageComponents:
    """Test language model (Phi) components."""

    @pytest.fixture
    def text_config(self):
        """Provide test text configuration."""
        return TextConfig(
            hidden_size=512,  # Smaller for testing
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
        )

    def test_phi_attention_creation(self, text_config):
        """Test PhiAttention module creation."""
        attention = PhiAttention(text_config)

        assert attention.num_heads == text_config.num_attention_heads
        assert hasattr(attention, "q_proj")
        assert hasattr(attention, "k_proj")
        assert hasattr(attention, "v_proj")
        assert hasattr(attention, "o_proj")

    def test_phi_mlp_creation(self, text_config):
        """Test PhiMLP module creation."""
        mlp = PhiMLP(text_config)

        assert hasattr(mlp, "fc1")
        assert hasattr(mlp, "fc2")

    def test_phi_mlp_forward(self, text_config):
        """Test PhiMLP forward pass."""
        mlp = PhiMLP(text_config)

        batch_size = 2
        seq_len = 10
        x = mx.random.normal((batch_size, seq_len, text_config.hidden_size))

        output = mlp(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype


# ============================================================================
# Region Module Tests
# ============================================================================


@pytest.mark.unit
class TestRegionModules:
    """Test region/detection modules."""

    def test_detection_head_creation(self):
        """Test DetectionHead module creation."""
        head = DetectionHead(hidden_size=2048, max_detections=100)

        assert hasattr(head, "hidden_size")

    def test_coordinate_decoder_creation(self):
        """Test CoordinateDecoder module creation."""
        decoder = CoordinateDecoder(hidden_size=2048)

        assert hasattr(decoder, "hidden_size")


# ============================================================================
# Full Model Tests
# ============================================================================


@pytest.mark.unit
class TestMoondream2Model:
    """Test the complete Moondream2 model."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        return ModelConfig(
            variant="test",
            vision_config=VisionConfig(
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=512,
                image_size=224,
                patch_size=16,
            ),
            text_config=TextConfig(
                hidden_size=512,
                num_hidden_layers=4,
                num_attention_heads=8,
                intermediate_size=2048,
                vocab_size=1000,
            ),
            region_config=RegionConfig(),
        )

    @pytest.fixture
    def model(self, small_config):
        """Provide test model instance."""
        return Moondream2(small_config)

    def test_model_creation(self, model, small_config):
        """Test model instantiation."""
        assert isinstance(model, Moondream2)
        assert isinstance(model.vision_encoder, VisionEncoder)
        assert isinstance(model.vision_projection, VisionProjector)
        assert isinstance(model.language_model, PhiModel)
        assert hasattr(model, "lm_head")
        assert hasattr(model, "detection_head")
        assert hasattr(model, "point_decoder")

    def test_model_submodules(self, model, small_config):
        """Test model has correct submodules."""
        # Vision encoder should match config
        assert model.vision_encoder.config.hidden_size == small_config.vision_config.hidden_size

        # Language model should match config
        assert model.language_model.config.hidden_size == small_config.text_config.hidden_size

        # LM head should map to vocab size
        assert hasattr(model.lm_head, "weight")

    def test_encode_image_shape(self, model, small_config):
        """Test image encoding output shape."""
        batch_size = 2
        image_size = small_config.vision_config.image_size
        num_channels = small_config.vision_config.num_channels

        # Create dummy image
        pixel_values = mx.random.normal((batch_size, num_channels, image_size, image_size))

        # Encode image (without tiling to keep test simple)
        vision_features = model.encode_image(pixel_values, use_tiling=False)

        # Check shape
        assert len(vision_features.shape) == 3  # [B, num_patches, hidden_size]
        assert vision_features.shape[0] == batch_size
        assert vision_features.shape[2] == small_config.text_config.hidden_size

    def test_model_with_config_variants(self):
        """Test model creation with different config variants."""
        # Test with 2B config (without full initialization)
        config_2b = DEFAULT_CONFIG_2B
        model_2b = Moondream2(config_2b)
        assert model_2b.config.variant == "2b"

        # Test with 0.5B config
        config_05b = DEFAULT_CONFIG_05B
        model_05b = Moondream2(config_05b)
        assert model_05b.config.variant == "0.5b"


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_model
class TestMoondream2Integration:
    """Integration tests requiring model weights."""

    def test_model_can_be_instantiated_with_default_config(self):
        """Test that model can be created with default configs."""
        config = DEFAULT_CONFIG_2B
        model = Moondream2(config)

        assert model is not None
        assert isinstance(model, Moondream2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
