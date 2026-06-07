# Copyright © 2025 SMLX Project

"""
Unit tests for TinyLLaVA vision-language model implementation.

Tests the vision encoder, language model, configuration, and forward pass.
"""

import mlx.core as mx
import pytest

from smlx.models.TinyLLaVA import (
    DEFAULT_CONFIG_1_5B,
    DEFAULT_CONFIG_2_0B,
    DEFAULT_CONFIG_3_1B,
    ConnectorConfig,
    ImageProcessor,
    ModelConfig,
    TextConfig,
    TinyLLaVA,
    VisionConfig,
)

# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.unit
class TestModelConfiguration:
    """Test model configuration and validation."""

    def test_default_config_1_5b(self):
        """Test that default 1.5B configuration is valid."""
        config = DEFAULT_CONFIG_1_5B

        assert config.model_type == "tinyllava"
        assert config.vision_config.hidden_size == 1152
        assert config.text_config.hidden_size == 2048
        assert config.text_config.num_hidden_layers == 22

    def test_default_config_2_0b(self):
        """Test that default 2.0B configuration is valid."""
        config = DEFAULT_CONFIG_2_0B

        assert config.model_type == "tinyllava"
        # StableLM-2 config
        assert config.text_config.model_type in ["llama", "stablelm"]

    def test_default_config_3_1b(self):
        """Test that default 3.1B configuration is valid."""
        config = DEFAULT_CONFIG_3_1B

        assert config.model_type == "tinyllava"
        # Phi-2 config
        assert config.text_config.model_type in ["llama", "phi"]

    def test_vision_config(self):
        """Test vision configuration."""
        config = VisionConfig()

        assert config.model_type == "siglip_vision_model"
        assert config.hidden_size == 1152
        assert config.num_hidden_layers == 27
        assert config.image_size == 384
        assert config.patch_size == 14
        assert config.hidden_act == "gelu_pytorch_tanh"

    def test_text_config(self):
        """Test text (TinyLlama) configuration."""
        config = TextConfig()

        assert config.model_type == "llama"
        assert config.vocab_size == 32000
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 22
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 4  # GQA
        assert config.hidden_act == "silu"

    def test_connector_config(self):
        """Test connector/projector configuration."""
        config = ConnectorConfig()

        assert config.projector_type == "mlp2x_gelu"
        assert config.projector_hidden_act == "gelu"
        assert config.use_resampler is False

    def test_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "model_type": "tinyllava",
            "vision_config": {
                "hidden_size": 1152,
                "num_hidden_layers": 27,
            },
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 22,
            },
            "connector_config": {
                "projector_type": "mlp2x_gelu",
            },
        }

        config = ModelConfig.from_dict(config_dict)
        assert config.vision_config.hidden_size == 1152
        assert config.text_config.hidden_size == 2048


# ============================================================================
# Vision Component Tests
# ============================================================================


@pytest.mark.unit
class TestVisionComponents:
    """Test vision-related components."""

    def test_vision_config_siglip(self):
        """Test SigLIP vision configuration."""
        config = VisionConfig()

        # SigLIP specific properties
        assert config.model_type == "siglip_vision_model"
        assert config.hidden_size == 1152
        assert config.num_hidden_layers == 27
        assert config.num_attention_heads == 16
        assert config.intermediate_size == 4304

    def test_image_processor_creation(self):
        """Test ImageProcessor creation."""
        processor = ImageProcessor(image_size=384)

        assert processor.image_size == 384
        assert hasattr(processor, "image_mean")
        assert hasattr(processor, "image_std")


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
                num_key_value_heads=2,
                intermediate_size=1024,
                vocab_size=1000,
            ),
            connector_config=ConnectorConfig(),
        )

    def test_connector_shapes(self, small_config):
        """Test connector/projector dimensions."""
        vision_dim = small_config.vision_config.hidden_size
        text_dim = small_config.text_config.hidden_size

        # Vision features should be projected to text dimension
        assert vision_dim == 256
        assert text_dim == 512


# ============================================================================
# Full Model Tests
# ============================================================================


@pytest.mark.unit
class TestTinyLLaVAModel:
    """Test the complete TinyLLaVA model."""

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
                hidden_size=512,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                intermediate_size=1024,
                vocab_size=1000,
            ),
            connector_config=ConnectorConfig(),
        )

    @pytest.fixture
    def model(self, small_config):
        """Provide test model instance."""
        return TinyLLaVA(small_config)

    def test_model_creation(self, model, small_config):
        """Test model instantiation."""
        assert isinstance(model, TinyLLaVA)
        assert hasattr(model, "vision_tower")
        assert hasattr(model, "multi_modal_projector")
        assert hasattr(model, "language_model")

    def test_model_config_variants(self):
        """Test model creation with different config variants."""
        # Test with 1.5B config (without full initialization)
        config_1_5b = DEFAULT_CONFIG_1_5B
        model_1_5b = TinyLLaVA(config_1_5b)
        assert model_1_5b.config.text_config.hidden_size == 2048

        # Test with 2.0B config
        config_2_0b = DEFAULT_CONFIG_2_0B
        model_2_0b = TinyLLaVA(config_2_0b)
        assert model_2_0b is not None

        # Test with 3.1B config
        config_3_1b = DEFAULT_CONFIG_3_1B
        model_3_1b = TinyLLaVA(config_3_1b)
        assert model_3_1b is not None

    def test_model_has_correct_structure(self, model):
        """Test model has correct component structure."""
        # Should have vision tower
        assert hasattr(model, "vision_tower")

        # Should have multimodal projector
        assert hasattr(model, "multi_modal_projector")

        # Should have language model
        assert hasattr(model, "language_model")

        # Should have language model head
        assert hasattr(model, "lm_head") or hasattr(model.language_model, "lm_head")


# ============================================================================
# Connector/Projector Tests
# ============================================================================


@pytest.mark.unit
class TestConnectorModule:
    """Test connector/projector module."""

    def test_mlp2x_projector_config(self):
        """Test MLP2x projector configuration."""
        config = ConnectorConfig(projector_type="mlp2x_gelu")

        assert config.projector_type == "mlp2x_gelu"
        assert config.projector_hidden_act == "gelu"

    def test_resampler_config(self):
        """Test resampler configuration."""
        config = ConnectorConfig(
            use_resampler=True,
            num_query_tokens=128,
            resampler_n_layers=3,
        )

        assert config.use_resampler is True
        assert config.num_query_tokens == 128
        assert config.resampler_n_layers == 3

    def test_resampler_attention_config(self):
        """Test resampler attention configuration parameters."""
        config = ConnectorConfig(
            use_resampler=True,
            num_query_tokens=64,
            resampler_n_layers=2,
            resampler_hidden_size=512,
            resampler_n_heads=8,
            resampler_head_dim=64,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
        )

        assert config.use_resampler is True
        assert config.num_query_tokens == 64
        assert config.resampler_n_layers == 2
        assert config.resampler_hidden_size == 512
        assert config.resampler_n_heads == 8
        assert config.resampler_head_dim == 64
        assert config.num_key_value_heads == 2
        assert config.rms_norm_eps == 1e-5


# ============================================================================
# Perceiver Resampler Component Tests
# ============================================================================


@pytest.mark.unit
class TestPerceiverResamplerComponents:
    """Test individual Perceiver Resampler components."""

    def test_perceiver_cross_attention_creation(self):
        """Test PerceiverCrossAttention module creation."""
        from smlx.models.TinyLLaVA.connector import PerceiverCrossAttention

        attn = PerceiverCrossAttention(
            hidden_size=768,
            num_heads=16,
            num_kv_heads=4,
            head_dim=96,
        )

        assert attn.num_heads == 16
        assert attn.num_kv_heads == 4
        assert attn.head_dim == 96
        assert attn.scale == 96**-0.5

    def test_perceiver_cross_attention_forward(self):
        """Test PerceiverCrossAttention forward pass."""
        from smlx.models.TinyLLaVA.connector import PerceiverCrossAttention

        attn = PerceiverCrossAttention(
            hidden_size=768,
            num_heads=16,
            num_kv_heads=4,
            head_dim=96,
        )

        # Create test inputs
        B, num_queries, num_patches = 2, 128, 196
        latents = mx.random.normal((B, num_queries, 768))
        context = mx.random.normal((B, num_patches, 768))

        # Forward pass
        output = attn(latents, context)

        # Check output shape
        assert output.shape == (B, num_queries, 768)

    def test_gated_mlp_creation(self):
        """Test GatedMLP module creation."""
        from smlx.models.TinyLLaVA.connector import GatedMLP

        mlp = GatedMLP(dim=768, hidden_dim=3072, output_size=768)

        assert hasattr(mlp, "gate_proj")
        assert hasattr(mlp, "up_proj")
        assert hasattr(mlp, "down_proj")

    def test_gated_mlp_forward(self):
        """Test GatedMLP forward pass."""
        from smlx.models.TinyLLaVA.connector import GatedMLP

        mlp = GatedMLP(dim=768, hidden_dim=3072, output_size=768)

        # Create test input
        B, seq_len = 2, 128
        x = mx.random.normal((B, seq_len, 768))

        # Forward pass
        output = mlp(x)

        # Check output shape
        assert output.shape == (B, seq_len, 768)

    def test_perceiver_layer_creation(self):
        """Test PerceiverLayer module creation."""
        from smlx.models.TinyLLaVA.connector import PerceiverLayer

        layer = PerceiverLayer(
            hidden_size=768,
            num_heads=16,
            num_kv_heads=4,
            head_dim=96,
            rms_norm_eps=1e-6,
        )

        assert hasattr(layer, "input_latents_norm")
        assert hasattr(layer, "input_context_norm")
        assert hasattr(layer, "cross_attn")
        assert hasattr(layer, "post_attention_layernorm")
        assert hasattr(layer, "mlp")

    def test_perceiver_layer_forward(self):
        """Test PerceiverLayer forward pass."""
        from smlx.models.TinyLLaVA.connector import PerceiverLayer

        layer = PerceiverLayer(
            hidden_size=768,
            num_heads=16,
            num_kv_heads=4,
            head_dim=96,
            rms_norm_eps=1e-6,
        )

        # Create test inputs
        B, num_queries, num_patches = 2, 128, 196
        latents = mx.random.normal((B, num_queries, 768))
        context = mx.random.normal((B, num_patches, 768))

        # Forward pass
        output = layer(latents, context)

        # Check output shape
        assert output.shape == (B, num_queries, 768)

    def test_resampler_projector_creation(self):
        """Test ResamplerProjector module creation."""
        from smlx.models.TinyLLaVA.connector import ResamplerProjector

        projector = ResamplerProjector(
            vision_hidden_size=1152,
            text_hidden_size=2048,
            num_query_tokens=128,
            num_layers=3,
            resampler_hidden_size=768,
            num_heads=16,
            num_kv_heads=4,
            head_dim=96,
            rms_norm_eps=1e-6,
        )

        assert projector.num_query_tokens == 128
        assert projector.resampler_hidden_size == 768
        assert len(projector.layers) == 3
        assert hasattr(projector, "latents")
        assert hasattr(projector, "vision_projection")
        assert hasattr(projector, "norm")
        assert hasattr(projector, "output_projection")

    def test_resampler_projector_forward(self):
        """Test ResamplerProjector forward pass."""
        from smlx.models.TinyLLaVA.connector import ResamplerProjector

        projector = ResamplerProjector(
            vision_hidden_size=1152,
            text_hidden_size=2048,
            num_query_tokens=128,
            num_layers=3,
            resampler_hidden_size=768,
            num_heads=16,
            num_kv_heads=4,
            head_dim=96,
            rms_norm_eps=1e-6,
        )

        # Create test input (vision features)
        B, num_patches = 2, 196
        vision_features = mx.random.normal((B, num_patches, 1152))

        # Forward pass
        output = projector(vision_features)

        # Check output shape
        assert output.shape == (B, 128, 2048)

    def test_resampler_latents_initialization(self):
        """Test that resampler latents are initialized to ones (not zeros)."""
        from smlx.models.TinyLLaVA.connector import ResamplerProjector

        projector = ResamplerProjector(
            vision_hidden_size=1152,
            text_hidden_size=2048,
            num_query_tokens=64,
            num_layers=2,
            resampler_hidden_size=512,
        )

        # Check latents shape
        assert projector.latents.shape == (64, 512)

        # Check latents are ones (not zeros)
        # Use mx.allclose to handle floating point comparisons
        assert mx.allclose(projector.latents, mx.ones_like(projector.latents)).item()

    def test_build_projector_with_resampler(self):
        """Test build_projector function with resampler config."""
        from smlx.models.TinyLLaVA.connector import ResamplerProjector, build_projector

        config = ConnectorConfig(
            projector_type="resampler",
            use_resampler=True,
            num_query_tokens=128,
            resampler_n_layers=3,
            resampler_hidden_size=768,
            resampler_n_heads=16,
            num_key_value_heads=4,
            resampler_head_dim=96,
            rms_norm_eps=1e-6,
        )

        projector = build_projector(
            config=config,
            vision_hidden_size=1152,
            text_hidden_size=2048,
        )

        # Should create ResamplerProjector
        assert isinstance(projector, ResamplerProjector)
        assert projector.num_query_tokens == 128

    def test_build_projector_default_mlp(self):
        """Test build_projector function defaults to MLP."""
        from smlx.models.TinyLLaVA.connector import MLPProjector, build_projector

        config = ConnectorConfig(projector_type="mlp2x_gelu")

        projector = build_projector(
            config=config,
            vision_hidden_size=1152,
            text_hidden_size=2048,
        )

        # Should create MLPProjector
        assert isinstance(projector, MLPProjector)


# ============================================================================
# Image Processing Tests
# ============================================================================


@pytest.mark.unit
class TestImageProcessing:
    """Test image processing utilities."""

    def test_image_processor_initialization(self):
        """Test ImageProcessor initialization."""
        processor = ImageProcessor(
            image_size=384,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )

        assert processor.image_size == 384
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
        # image_mean and image_std are reshaped to (1, 1, 3) for broadcasting
        assert processor.image_mean.shape == (1, 1, 3)
        assert processor.image_std.shape == (1, 1, 3)


# ============================================================================
# Image-token merge (single & multiple images)
# ============================================================================


@pytest.mark.unit
class TestImageTokenMerge:
    """Test prepare_inputs_for_generation injects image features at <image> tokens.

    Exercised on a REAL (small, randomly-initialised) TinyLLaVA model — no test
    doubles. We pass image_features directly so the vision tower isn't run, but
    the text path (the real ``language_model.embed_tokens``) and the model's real
    ``config.image_token_index`` drive the merge. Image patches use distinctive
    sentinel values (100.0 / 200.0) that cannot collide with the model's real
    (small, ~N(0, σ²)) text embeddings, so we can assert exactly where each
    image's patches land and that text positions keep their real embeddings.
    """

    HIDDEN = 32

    @pytest.fixture(scope="class")
    def model(self):
        """A real small TinyLLaVA (tiny vision + text stacks, real embed table)."""
        cfg = ModelConfig(
            vision_config=VisionConfig(
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=256,
                image_size=224,
                patch_size=16,
            ),
            text_config=TextConfig(
                hidden_size=self.HIDDEN,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=64,
                vocab_size=1000,
            ),
            connector_config=ConnectorConfig(),
        )
        return TinyLLaVA(cfg)

    def _img_token(self, model):
        return model.config.image_token_index

    def test_single_image_injection(self, model):
        img = self._img_token(model)
        feats = mx.arange(1 * 4 * self.HIDDEN).reshape(1, 4, self.HIDDEN).astype(mx.float32)
        ids = mx.array([[1, img, 2]])  # 2 text tokens + 1 image (4 patches)
        combined, _ = model.prepare_inputs_for_generation(ids, image_features=feats)
        assert combined.shape == (1, 2 + 4, self.HIDDEN)

    def test_multiple_image_injection(self, model):
        img = self._img_token(model)
        feats = mx.arange(2 * 3 * self.HIDDEN).reshape(2, 3, self.HIDDEN).astype(mx.float32)
        # text + <image> + text + <image> + text -> 3 text + 2*3 patches
        ids = mx.array([[1, img, 2, img, 3]])
        combined, _ = model.prepare_inputs_for_generation(ids, image_features=feats)
        assert combined.shape == (1, 3 + 2 * 3, self.HIDDEN)

    def test_image_count_mismatch_raises(self, model):
        img = self._img_token(model)
        feats = mx.arange(1 * 3 * self.HIDDEN).reshape(1, 3, self.HIDDEN).astype(mx.float32)
        ids = mx.array([[img, 1, img]])  # 2 image tokens, 1 image
        with pytest.raises(ValueError, match="does not match"):
            model.prepare_inputs_for_generation(ids, image_features=feats)

    def test_batched_injection_places_features_in_correct_row(self, model):
        """Batch>1: each row's image lands in that row, not flattened across rows."""
        import numpy as np

        img = self._img_token(model)
        patches = 2
        # Two rows, each with one <image> token at a different column.
        ids = mx.array(
            [
                [1, img, 2, 3],  # image at col 1
                [4, 5, img, 6],  # image at col 2
            ]
        )
        # Reference text embeddings straight from the real embedding table.
        ref = np.array(model.language_model.embed_tokens(ids))

        # Row 0 -> image 0 (all 100.0), row 1 -> image 1 (all 200.0).
        feats = mx.concatenate(
            [
                mx.full((1, patches, self.HIDDEN), 100.0),
                mx.full((1, patches, self.HIDDEN), 200.0),
            ],
            axis=0,
        )
        combined, _ = model.prepare_inputs_for_generation(ids, image_features=feats)
        # Each row: 3 text tokens + `patches` image tokens.
        assert combined.shape == (2, 3 + patches, self.HIDDEN)

        out = np.array(combined)
        # Row 0: text(col 0), image patches (100.0), text(cols 2, 3 of the original).
        assert np.allclose(out[0, 0, :], ref[0, 0, :])
        assert np.allclose(out[0, 1 : 1 + patches, :], 100.0)
        assert np.allclose(out[0, 1 + patches, :], ref[0, 2, :])
        assert np.allclose(out[0, 2 + patches, :], ref[0, 3, :])
        # Row 1: text(cols 0, 1), image patches (200.0), text(col 3 of the original).
        assert np.allclose(out[1, 0, :], ref[1, 0, :])
        assert np.allclose(out[1, 1, :], ref[1, 1, :])
        assert np.allclose(out[1, 2 : 2 + patches, :], 200.0)
        assert np.allclose(out[1, 2 + patches, :], ref[1, 3, :])
        # The image features must NOT bleed into the wrong row.
        assert not np.any(np.isclose(out[0], 200.0))
        assert not np.any(np.isclose(out[1], 100.0))

    def test_ragged_batch_raises(self, model):
        """Rows with differing <image>-token counts cannot pack into one tensor."""
        img = self._img_token(model)
        # Row 0 has two <image> tokens, row 1 has one -> ragged expanded lengths.
        ids = mx.array(
            [
                [img, 1, img],
                [2, img, 3],
            ]
        )
        feats = mx.arange(3 * 2 * self.HIDDEN).reshape(3, 2, self.HIDDEN).astype(mx.float32)
        with pytest.raises(ValueError, match="Ragged image batch"):
            model.prepare_inputs_for_generation(ids, image_features=feats)


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_model
class TestTinyLLaVAIntegration:
    """Integration tests requiring model weights."""

    def test_model_can_be_instantiated_with_default_configs(self):
        """Test that model can be created with default configs."""
        # Test 1.5B variant
        config = DEFAULT_CONFIG_1_5B
        model = TinyLLaVA(config)
        assert model is not None
        assert isinstance(model, TinyLLaVA)

    def test_all_variants_can_be_instantiated(self):
        """Test that all model variants can be created."""
        configs = [DEFAULT_CONFIG_1_5B, DEFAULT_CONFIG_2_0B, DEFAULT_CONFIG_3_1B]

        for config in configs:
            model = TinyLLaVA(config)
            assert model is not None
            assert isinstance(model, TinyLLaVA)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
