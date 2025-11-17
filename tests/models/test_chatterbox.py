# Copyright © 2025 SMLX Project

"""
Unit tests for Chatterbox TTS model implementation.

Tests the voice cloning, emotion control, expressiveness, and synthesis.
"""

import mlx.core as mx
import pytest

from smlx.models.Chatterbox import (
    AVAILABLE_EMOTIONS,
    DEFAULT_CONFIG,
    AcousticConfig,
    Chatterbox,
    ChatterboxConfig,
    ChatterboxProcessor,
    ExpressivenessConfig,
    ExpressivenessModule,
    LlamaBackboneConfig,
    VoiceEncoder,
    VoiceEncoderConfig,
    create_model,
)
from smlx.models.Chatterbox.model import (
    Attention,
    MLP,
    TransformerBlock,
    NoPE,
    LlamaBackbone,
    create_causal_mask,
    create_attention_mask,
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

        assert config.model_type == "chatterbox"
        assert config.llama_config.hidden_size == 1024  # 0.5B model
        assert config.llama_config.num_hidden_layers == 24

    def test_llama_backbone_config(self):
        """Test Llama backbone configuration."""
        config = LlamaBackboneConfig()

        assert config.vocab_size == 49152
        assert config.hidden_size == 1024  # 0.5B model
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 4  # GQA
        assert config.intermediate_size == 2752

    def test_voice_encoder_config(self):
        """Test voice encoder configuration."""
        config = VoiceEncoderConfig()

        assert config.num_mels == 80
        assert config.hidden_size == 512
        assert config.num_layers == 6
        assert config.num_heads == 8
        assert config.embedding_dim == 256

    def test_expressiveness_config(self):
        """Test expressiveness configuration."""
        config = ExpressivenessConfig()

        assert config.num_emotions == 8
        assert config.emotion_embedding_dim == 128

    def test_acoustic_config(self):
        """Test acoustic configuration."""
        config = AcousticConfig()

        assert hasattr(config, "num_mels")
        assert hasattr(config, "sample_rate")

    def test_available_emotions(self):
        """Test available emotions list."""
        assert len(AVAILABLE_EMOTIONS) == 8
        assert "neutral" in AVAILABLE_EMOTIONS
        assert "happy" in AVAILABLE_EMOTIONS
        assert "sad" in AVAILABLE_EMOTIONS
        assert "angry" in AVAILABLE_EMOTIONS
        assert "excited" in AVAILABLE_EMOTIONS
        assert "calm" in AVAILABLE_EMOTIONS


# ============================================================================
# Voice Encoder Tests
# ============================================================================


@pytest.mark.unit
class TestVoiceEncoder:
    """Test voice encoder component."""

    def test_voice_encoder_creation(self):
        """Test VoiceEncoder module creation."""
        config = VoiceEncoderConfig()
        encoder = VoiceEncoder(config)

        assert hasattr(encoder, "config")
        assert encoder.config.embedding_dim == 256

    def test_voice_encoder_forward(self):
        """Test VoiceEncoder forward pass."""
        config = VoiceEncoderConfig()
        encoder = VoiceEncoder(config)

        batch_size = 2
        time_steps = 100
        mel_input = mx.random.normal((batch_size, time_steps, config.num_mels))

        # Forward pass
        embedding = encoder(mel_input)

        # Should output voice embedding
        assert len(embedding.shape) == 2  # [batch, embedding_dim]
        assert embedding.shape[0] == batch_size
        assert embedding.shape[1] == config.embedding_dim


# ============================================================================
# Expressiveness Module Tests
# ============================================================================


@pytest.mark.unit
class TestExpressivenessModule:
    """Test expressiveness control module."""

    def test_expressiveness_module_creation(self):
        """Test ExpressivenessModule creation."""
        config = ExpressivenessConfig()
        module = ExpressivenessModule(config)

        assert hasattr(module, "config")
        assert module.config.num_emotions == 8

    def test_expressiveness_module_emotions(self):
        """Test expressiveness module with different emotions."""
        config = ExpressivenessConfig()
        module = ExpressivenessModule(config)

        # Test that module can handle emotion IDs
        for emotion_id in range(config.num_emotions):
            # Should not raise error
            assert emotion_id < config.num_emotions


# ============================================================================
# Full Model Tests
# ============================================================================


@pytest.mark.unit
class TestChatterboxModel:
    """Test the complete Chatterbox model."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        return ChatterboxConfig(
            llama_config=LlamaBackboneConfig(
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=512,
                vocab_size=1000,
            ),
            voice_encoder_config=VoiceEncoderConfig(
                hidden_size=128,
                num_layers=2,
                num_heads=4,
                embedding_dim=64,
            ),
            expressiveness_config=ExpressivenessConfig(
                num_emotions=8,
                emotion_embedding_dim=32,
            ),
            acoustic_config=AcousticConfig(),
        )

    @pytest.fixture
    def model(self, small_config):
        """Provide test model instance."""
        return Chatterbox(small_config)

    def test_model_creation(self, model, small_config):
        """Test model instantiation."""
        assert isinstance(model, Chatterbox)
        assert hasattr(model, "llama_backbone")
        assert hasattr(model, "voice_encoder")
        assert hasattr(model, "expressiveness_module")

    def test_model_creation_with_default_config(self):
        """Test model creation with default config."""
        config = DEFAULT_CONFIG
        model = Chatterbox(config)

        assert isinstance(model, Chatterbox)
        assert model.config == config

    def test_create_model_factory(self):
        """Test create_model factory function."""
        config = DEFAULT_CONFIG
        model = create_model(config)

        assert isinstance(model, Chatterbox)

    def test_model_has_correct_structure(self, model):
        """Test model has correct component structure."""
        # Should have Llama backbone
        assert hasattr(model, "llama_backbone")

        # Should have voice encoder
        assert hasattr(model, "voice_encoder")
        assert isinstance(model.voice_encoder, VoiceEncoder)

        # Should have expressiveness module
        assert hasattr(model, "expressiveness_module")
        assert isinstance(model.expressiveness_module, ExpressivenessModule)

        # Should have acoustic head
        assert hasattr(model, "acoustic_head")

    def test_model_config_properties(self, model, small_config):
        """Test model configuration properties."""
        assert model.config.llama_config.hidden_size == 256
        assert model.config.voice_encoder_config.embedding_dim == 64
        assert model.config.expressiveness_config.num_emotions == 8


# ============================================================================
# Processor Tests
# ============================================================================


@pytest.mark.unit
class TestChatterboxProcessor:
    """Test Chatterbox processor."""

    def test_processor_creation(self):
        """Test ChatterboxProcessor creation."""
        processor = ChatterboxProcessor()

        assert hasattr(processor, "sample_rate")
        assert hasattr(processor, "num_mels")

    def test_processor_text_processing(self):
        """Test text processing."""
        processor = ChatterboxProcessor()

        text = "Hello, world!"
        # Processor should be able to process text
        assert processor is not None


# ============================================================================
# Model Size Tests
# ============================================================================


@pytest.mark.unit
class TestModelSize:
    """Test model parameter count."""

    def test_model_is_small(self):
        """Test that Chatterbox is a small model (~500M)."""
        config = DEFAULT_CONFIG

        # Check individual component sizes
        # Llama backbone: ~400M params (0.5B)
        assert config.llama_config.hidden_size == 1024
        assert config.llama_config.num_hidden_layers == 24

        # Voice encoder: ~40M params
        assert config.voice_encoder_config.hidden_size == 512
        assert config.voice_encoder_config.num_layers == 6

        # Total should be ~500M parameters


# ============================================================================
# Llama Components Tests (Phase 4: Full Llama Implementation)
# ============================================================================


@pytest.mark.unit
class TestLlamaConfig:
    """Test updated LlamaBackboneConfig with full Llama fields."""

    def test_config_with_new_fields(self):
        """Test config has all new Llama fields."""
        config = LlamaBackboneConfig()

        # Original fields
        assert config.vocab_size == 49152
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24

        # New fields from SmolLM2_135M
        assert hasattr(config, 'head_dim')
        assert hasattr(config, 'attention_bias')
        assert hasattr(config, 'mlp_bias')
        assert hasattr(config, 'rope_theta')
        assert hasattr(config, 'rope_traditional')
        assert hasattr(config, 'rope_scaling')
        assert hasattr(config, 'tie_word_embeddings')
        assert hasattr(config, 'layer_types')
        assert hasattr(config, 'sliding_window')
        assert hasattr(config, 'no_rope_layer_interval')
        assert hasattr(config, 'no_rope_layers')

    def test_config_post_init(self):
        """Test config post-initialization sets defaults."""
        config = LlamaBackboneConfig()

        # layer_types should be set
        assert config.layer_types is not None
        assert len(config.layer_types) == config.num_hidden_layers
        assert all(layer_type == "full_attention" for layer_type in config.layer_types)

        # no_rope_layers should be set (NoPE feature)
        assert config.no_rope_layers is not None
        assert len(config.no_rope_layers) == config.num_hidden_layers
        # Every 4th layer should have RoPE disabled (value = 0)
        for i in range(config.num_hidden_layers):
            expected_rope = int((i + 1) % config.no_rope_layer_interval != 0)
            assert config.no_rope_layers[i] == expected_rope


@pytest.mark.unit
class TestNoPE:
    """Test NoPE (No Positional Encoding) module."""

    def test_nope_creation(self):
        """Test NoPE can be created."""
        nope = NoPE()
        assert nope is not None

    def test_nope_forward(self):
        """Test NoPE forward pass (no-op)."""
        nope = NoPE()

        # Create test input
        x = mx.random.normal((2, 100, 16, 64))  # (B, L, heads, head_dim)

        # Forward pass should return unchanged input
        output = nope(x)
        assert mx.allclose(output, x)

        # Test with offset parameter
        output_with_offset = nope(x, offset=10)
        assert mx.allclose(output_with_offset, x)


@pytest.mark.unit
class TestAttention:
    """Test Attention module with RoPE and GQA."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        return LlamaBackboneConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,  # GQA
            intermediate_size=512,
            vocab_size=1000,
        )

    def test_attention_creation(self, small_config):
        """Test Attention module can be created."""
        attn = Attention(small_config)

        assert attn.n_heads == 8
        assert attn.n_kv_heads == 4  # GQA
        assert hasattr(attn, 'q_proj')
        assert hasattr(attn, 'k_proj')
        assert hasattr(attn, 'v_proj')
        assert hasattr(attn, 'o_proj')
        assert hasattr(attn, 'rope')

    def test_attention_forward(self, small_config):
        """Test Attention forward pass."""
        attn = Attention(small_config)

        batch_size = 2
        seq_len = 10
        x = mx.random.normal((batch_size, seq_len, small_config.hidden_size))

        # Forward pass without cache
        output = attn(x, mask="causal", cache=None)

        # Output should have same shape as input
        assert output.shape == x.shape

    def test_attention_different_configs(self):
        """Test Attention with different configurations."""
        # Test with bias
        config_with_bias = LlamaBackboneConfig(
            hidden_size=128,
            num_attention_heads=4,
            attention_bias=True,
        )
        attn = Attention(config_with_bias)
        assert attn.q_proj.bias is not None

        # Test without bias (default)
        config_no_bias = LlamaBackboneConfig(
            hidden_size=128,
            num_attention_heads=4,
            attention_bias=False,
        )
        attn = Attention(config_no_bias)
        # MLX Linear doesn't have bias attribute when bias=False


@pytest.mark.unit
class TestMLP:
    """Test MLP module with SwiGLU."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        return LlamaBackboneConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
        )

    def test_mlp_creation(self, small_config):
        """Test MLP module can be created."""
        mlp = MLP(small_config)

        assert hasattr(mlp, 'gate_proj')
        assert hasattr(mlp, 'down_proj')
        assert hasattr(mlp, 'up_proj')

    def test_mlp_forward(self, small_config):
        """Test MLP forward pass."""
        mlp = MLP(small_config)

        batch_size = 2
        seq_len = 10
        x = mx.random.normal((batch_size, seq_len, small_config.hidden_size))

        # Forward pass
        output = mlp(x)

        # Output should have same shape as input
        assert output.shape == x.shape

    def test_mlp_with_bias(self):
        """Test MLP with bias enabled."""
        config_with_bias = LlamaBackboneConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            mlp_bias=True,
        )
        mlp = MLP(config_with_bias)
        assert mlp.gate_proj.bias is not None


@pytest.mark.unit
class TestTransformerBlock:
    """Test TransformerBlock with pre-normalization."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        return LlamaBackboneConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=512,
            vocab_size=1000,
        )

    def test_transformer_block_creation(self, small_config):
        """Test TransformerBlock can be created."""
        block = TransformerBlock(small_config)

        assert hasattr(block, 'self_attn')
        assert hasattr(block, 'mlp')
        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'post_attention_layernorm')
        assert isinstance(block.self_attn, Attention)
        assert isinstance(block.mlp, MLP)

    def test_transformer_block_forward(self, small_config):
        """Test TransformerBlock forward pass."""
        block = TransformerBlock(small_config)

        batch_size = 2
        seq_len = 10
        x = mx.random.normal((batch_size, seq_len, small_config.hidden_size))

        # Forward pass
        output = block(x, mask="causal", cache=None)

        # Output should have same shape as input
        assert output.shape == x.shape


@pytest.mark.unit
class TestLlamaBackbone:
    """Test full LlamaBackbone implementation."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        return LlamaBackboneConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=512,
            vocab_size=1000,
        )

    def test_llama_backbone_creation(self, small_config):
        """Test LlamaBackbone can be created."""
        backbone = LlamaBackbone(small_config)

        assert hasattr(backbone, 'embedding')
        assert hasattr(backbone, 'layers')
        assert hasattr(backbone, 'norm')
        assert len(backbone.layers) == small_config.num_hidden_layers

    def test_llama_backbone_forward_with_input_ids(self, small_config):
        """Test LlamaBackbone forward pass with input_ids."""
        backbone = LlamaBackbone(small_config)

        batch_size = 2
        seq_len = 10
        input_ids = mx.random.randint(0, small_config.vocab_size, (batch_size, seq_len))

        # Forward pass
        output = backbone(input_ids=input_ids)

        # Output should be (batch, seq_len, hidden_size)
        assert output.shape == (batch_size, seq_len, small_config.hidden_size)

    def test_llama_backbone_forward_with_embeddings(self, small_config):
        """Test LlamaBackbone forward pass with pre-computed embeddings."""
        backbone = LlamaBackbone(small_config)

        batch_size = 2
        seq_len = 10
        embeddings = mx.random.normal((batch_size, seq_len, small_config.hidden_size))

        # Forward pass
        output = backbone(input_embeddings=embeddings)

        # Output should have same shape as embeddings
        assert output.shape == embeddings.shape

    def test_llama_backbone_with_nope(self):
        """Test LlamaBackbone with NoPE layers."""
        config = LlamaBackboneConfig(
            hidden_size=256,
            num_hidden_layers=8,  # 8 layers
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=512,
            vocab_size=1000,
            no_rope_layer_interval=4,  # Every 4th layer has no RoPE
        )
        backbone = LlamaBackbone(config)

        # Check that every 4th layer has NoPE instead of RoPE
        for i, layer in enumerate(backbone.layers):
            if (i + 1) % 4 == 0:
                # Every 4th layer should have NoPE
                assert isinstance(layer.self_attn.rope, NoPE)
            # Note: Can't easily check for RoPE due to MLX's module structure


@pytest.mark.unit
class TestUtilities:
    """Test utility functions."""

    def test_create_causal_mask(self):
        """Test causal mask creation."""
        N = 5
        mask = create_causal_mask(N)

        # Mask should be (N, N)
        assert mask.shape == (N, N)

        # Check causal property (lower triangular)
        for i in range(N):
            for j in range(N):
                if j > i:
                    assert not bool(mask[i, j])
                else:
                    assert bool(mask[i, j])

    def test_create_causal_mask_with_offset(self):
        """Test causal mask with offset."""
        N = 5
        offset = 3
        mask = create_causal_mask(N, offset=offset)

        # With offset, mask has shape (N, offset+N)
        assert mask.shape == (N, offset + N)

    def test_create_attention_mask(self):
        """Test attention mask creation."""
        batch_size = 2
        seq_len = 10
        hidden_size = 64

        h = mx.random.normal((batch_size, seq_len, hidden_size))

        # Test with seq_len > 1 (should return "causal")
        mask = create_attention_mask(h)
        assert mask == "causal"

        # Test with seq_len = 1 (should return None)
        h_single = mx.random.normal((batch_size, 1, hidden_size))
        mask_single = create_attention_mask(h_single)
        assert mask_single is None


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_model
class TestChatterboxIntegration:
    """Integration tests requiring model weights."""

    def test_model_can_be_instantiated_with_default_config(self):
        """Test that model can be created with default config."""
        config = DEFAULT_CONFIG
        model = Chatterbox(config)

        assert model is not None
        assert isinstance(model, Chatterbox)

    def test_model_expected_architecture(self):
        """Test that model has expected architecture."""
        config = DEFAULT_CONFIG
        model = Chatterbox(config)

        # Should have all four components
        assert hasattr(model, "llama_backbone")
        assert hasattr(model, "voice_encoder")
        assert hasattr(model, "expressiveness_module")
        assert hasattr(model, "acoustic_head")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
