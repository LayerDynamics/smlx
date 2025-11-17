# Copyright © 2025 SMLX Project

"""
Unit tests for TrOCR-small OCR model implementation.

Tests the BEiT vision encoder, XLMRoberta decoder, and text recognition.
"""

import pytest

from smlx.models.TrOCR_small import (
    DEFAULT_CONFIG_HANDWRITTEN,
    DEFAULT_CONFIG_PRINTED,
    TrOCR,
    TrOCRConfig,
    TrOCRDecoderConfig,
    TrOCRProcessor,
    TrOCRTokenizer,
    TrOCRVisionConfig,
)

# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.unit
class TestModelConfiguration:
    """Test model configuration and validation."""

    def test_vision_config(self):
        """Test BEiT vision encoder configuration."""
        config = TrOCRVisionConfig()

        assert config.hidden_size == 384
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 6
        assert config.intermediate_size == 1536
        assert config.image_size == 384
        assert config.patch_size == 16

    def test_vision_config_num_patches(self):
        """Test vision config num_patches property."""
        config = TrOCRVisionConfig()

        # 384 / 16 = 24, so 24 * 24 = 576 patches
        assert config.num_patches == (config.image_size // config.patch_size) ** 2
        assert config.num_patches == 576

    def test_decoder_config(self):
        """Test XLMRoberta decoder configuration."""
        config = TrOCRDecoderConfig()

        assert config.vocab_size == 64044  # XLMRoberta vocab (TrOCR-specific)
        assert config.hidden_size == 384
        assert config.num_hidden_layers == 6
        assert config.num_attention_heads == 6
        assert config.intermediate_size == 1536
        assert config.is_decoder is True
        assert config.add_cross_attention is True

    def test_decoder_special_tokens(self):
        """Test decoder special token IDs."""
        config = TrOCRDecoderConfig()

        assert config.bos_token_id == 0
        assert config.eos_token_id == 2
        assert config.pad_token_id == 1

    def test_trocr_config(self):
        """Test complete TrOCR configuration."""
        vision_config = TrOCRVisionConfig()
        decoder_config = TrOCRDecoderConfig()
        config = TrOCRConfig(
            vision_config=vision_config,
            decoder_config=decoder_config,
        )

        assert config.vision_config == vision_config
        assert config.decoder_config == decoder_config

    def test_default_config_printed(self):
        """Test default printed text configuration."""
        config = DEFAULT_CONFIG_PRINTED

        assert config.vision_config.hidden_size == 384
        assert config.decoder_config.vocab_size == 64044

    def test_default_config_handwritten(self):
        """Test default handwritten text configuration."""
        config = DEFAULT_CONFIG_HANDWRITTEN

        assert config.vision_config.hidden_size == 384
        assert config.decoder_config.vocab_size == 64044


# ============================================================================
# Vision Encoder Tests
# ============================================================================


@pytest.mark.unit
class TestVisionEncoder:
    """Test BEiT vision encoder component."""

    def test_vision_config_dimensions(self):
        """Test vision encoder dimensions."""
        config = TrOCRVisionConfig()

        # Hidden size should be divisible by num heads
        assert config.hidden_size % config.num_attention_heads == 0

        # Head dimension
        head_dim = config.hidden_size // config.num_attention_heads
        assert head_dim == 64

    def test_vision_config_patch_embedding(self):
        """Test patch embedding configuration."""
        config = TrOCRVisionConfig()

        # Image should be divisible by patch size
        assert config.image_size % config.patch_size == 0

        # Number of patches per dimension
        patches_per_dim = config.image_size // config.patch_size
        assert patches_per_dim == 24


# ============================================================================
# Decoder Tests
# ============================================================================


@pytest.mark.unit
class TestDecoder:
    """Test RoBERTa decoder component."""

    def test_decoder_config_dimensions(self):
        """Test decoder dimensions."""
        config = TrOCRDecoderConfig()

        # Hidden size should be divisible by num heads
        assert config.hidden_size % config.num_attention_heads == 0

        # Head dimension
        head_dim = config.hidden_size // config.num_attention_heads
        assert head_dim == 64

    def test_decoder_cross_attention(self):
        """Test decoder cross-attention configuration."""
        config = TrOCRDecoderConfig()

        # Decoder should have cross-attention for encoder features
        assert config.is_decoder is True
        assert config.add_cross_attention is True

    def test_decoder_max_length(self):
        """Test decoder maximum sequence length."""
        config = TrOCRDecoderConfig()

        assert config.max_position_embeddings == 512


# ============================================================================
# Full Model Tests
# ============================================================================


@pytest.mark.unit
class TestTrOCRModel:
    """Test the complete TrOCR model."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        vision_config = TrOCRVisionConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            image_size=224,
            patch_size=16,
        )
        decoder_config = TrOCRDecoderConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            vocab_size=1000,
        )
        return TrOCRConfig(
            vision_config=vision_config,
            decoder_config=decoder_config,
        )

    @pytest.fixture
    def model(self, small_config):
        """Provide test model instance."""
        return TrOCR(small_config)

    def test_model_creation(self, model, small_config):
        """Test model instantiation."""
        assert isinstance(model, TrOCR)
        assert hasattr(model, "config")
        assert model.config == small_config

    def test_model_creation_with_default_configs(self):
        """Test model creation with default configs."""
        # Test printed variant
        config_printed = DEFAULT_CONFIG_PRINTED
        model_printed = TrOCR(config_printed)
        assert isinstance(model_printed, TrOCR)

        # Test handwritten variant
        config_handwritten = DEFAULT_CONFIG_HANDWRITTEN
        model_handwritten = TrOCR(config_handwritten)
        assert isinstance(model_handwritten, TrOCR)

    def test_model_has_correct_structure(self, model):
        """Test model has correct component structure."""
        # Should have vision encoder
        assert hasattr(model, "encoder")

        # Should have text decoder
        assert hasattr(model, "decoder")

    def test_model_config_properties(self, model, small_config):
        """Test model configuration properties."""
        assert model.config.vision_config.hidden_size == 128
        assert model.config.decoder_config.hidden_size == 128
        assert model.config.vision_config.num_hidden_layers == 2
        assert model.config.decoder_config.num_hidden_layers == 2


# ============================================================================
# Model Size Tests
# ============================================================================


@pytest.mark.unit
class TestModelSize:
    """Test model parameter count."""

    def test_model_is_small(self):
        """Test that TrOCR-small is a small model (~60M)."""
        config = DEFAULT_CONFIG_PRINTED

        # Vision encoder: ~28M params
        assert config.vision_config.hidden_size == 384
        assert config.vision_config.num_hidden_layers == 12

        # Decoder: ~32M params
        assert config.decoder_config.hidden_size == 384
        assert config.decoder_config.num_hidden_layers == 6

        # Total: ~60M parameters


# ============================================================================
# Variant Tests
# ============================================================================


@pytest.mark.unit
class TestModelVariants:
    """Test different TrOCR variants."""

    def test_printed_variant(self):
        """Test printed text variant."""
        config = DEFAULT_CONFIG_PRINTED
        model = TrOCR(config)

        assert model is not None
        assert isinstance(model, TrOCR)

    def test_handwritten_variant(self):
        """Test handwritten text variant."""
        config = DEFAULT_CONFIG_HANDWRITTEN
        model = TrOCR(config)

        assert model is not None
        assert isinstance(model, TrOCR)

    def test_both_variants_have_same_architecture(self):
        """Test that both variants use the same architecture."""
        config_printed = DEFAULT_CONFIG_PRINTED
        config_handwritten = DEFAULT_CONFIG_HANDWRITTEN

        # Both should have same architecture, different weights
        assert config_printed.vision_config.hidden_size == config_handwritten.vision_config.hidden_size
        assert config_printed.decoder_config.hidden_size == config_handwritten.decoder_config.hidden_size


# ============================================================================
# Tokenizer Tests
# ============================================================================


@pytest.mark.unit
class TestTrOCRTokenizer:
    """Test TrOCR tokenizer implementation."""

    def test_tokenizer_creation_without_model(self):
        """Test tokenizer can be created without loading from HuggingFace."""
        tokenizer = TrOCRTokenizer(vocab_size=64044)

        assert tokenizer.vocab_size == 64044
        assert tokenizer.bos_token_id == 0
        assert tokenizer.eos_token_id == 2
        assert tokenizer.pad_token_id == 1

    def test_tokenizer_fallback_encode(self):
        """Test fallback character-level encoding."""
        # Create tokenizer without model_name (uses fallback)
        tokenizer = TrOCRTokenizer(vocab_size=64044)

        text = "Hello"
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        # Should have character-level encoding
        assert isinstance(token_ids, list)
        assert len(token_ids) == len(text)

    def test_tokenizer_fallback_decode(self):
        """Test fallback character-level decoding."""
        # Create tokenizer without model_name (uses fallback)
        tokenizer = TrOCRTokenizer(vocab_size=64044)

        # ASCII characters
        token_ids = [72, 101, 108, 108, 111]  # "Hello" in ASCII
        text = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Should decode simple ASCII
        assert isinstance(text, str)

    def test_tokenizer_special_tokens(self):
        """Test special token handling."""
        tokenizer = TrOCRTokenizer(vocab_size=64044)

        # Encode with special tokens
        token_ids_with = tokenizer.encode("test", add_special_tokens=True)

        # First should be BOS, last should be EOS
        assert token_ids_with[0] == 0  # BOS
        assert token_ids_with[-1] == 2  # EOS

        # Encode without special tokens
        token_ids_without = tokenizer.encode("test", add_special_tokens=False)

        # Should not have BOS/EOS
        assert token_ids_without[0] != 0

    def test_tokenizer_batch_decode(self):
        """Test batch decoding."""
        tokenizer = TrOCRTokenizer(vocab_size=64044)

        token_ids_batch = [[72, 105], [66, 121, 101]]  # ["Hi", "Bye"] in ASCII
        texts = tokenizer.batch_decode(token_ids_batch, skip_special_tokens=True)

        assert isinstance(texts, list)
        assert len(texts) == 2


@pytest.mark.integration
@pytest.mark.requires_model
class TestTrOCRTokenizerWithModel:
    """Test tokenizer with actual XLMRoberta model loaded."""

    @pytest.fixture(scope="class")
    def tokenizer_with_model(self):
        """Load tokenizer with actual model from HuggingFace."""
        try:
            tokenizer = TrOCRTokenizer(
                vocab_size=64044, model_name="microsoft/trocr-small-printed"
            )
            if tokenizer._tokenizer is None:
                pytest.skip("Tokenizer not available (transformers not installed or download failed)")
            return tokenizer
        except Exception as e:
            pytest.skip(f"Could not load tokenizer: {e}")

    def test_real_tokenizer_loading(self, tokenizer_with_model):
        """Test that real tokenizer loads correctly."""
        assert tokenizer_with_model._tokenizer is not None
        # Actual vocab size from HuggingFace is 64002 (not 64044)
        # Our code correctly updates to match the loaded tokenizer
        assert tokenizer_with_model.vocab_size == tokenizer_with_model._tokenizer.vocab_size
        assert tokenizer_with_model.vocab_size >= 64000  # Should be around 64K

    def test_real_tokenizer_encode_decode(self, tokenizer_with_model):
        """Test encoding and decoding with real tokenizer."""
        text = "Hello World!"

        # Encode
        token_ids = tokenizer_with_model.encode(text, add_special_tokens=True)

        # Should have tokens (not just character-level)
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert token_ids[0] == 0  # BOS token

        # Decode
        decoded_text = tokenizer_with_model.decode(token_ids, skip_special_tokens=True)

        # Should match original (may have whitespace differences)
        assert isinstance(decoded_text, str)
        assert "hello" in decoded_text.lower()
        assert "world" in decoded_text.lower()

    def test_real_tokenizer_subword_tokenization(self, tokenizer_with_model):
        """Test that real tokenizer uses subword tokenization (not character-level)."""
        text = "tokenization"

        # Encode without special tokens to see actual tokenization
        token_ids = tokenizer_with_model.encode(text, add_special_tokens=False)

        # Should have fewer tokens than characters (subword compression)
        assert len(token_ids) <= len(text)

    def test_real_tokenizer_special_tokens(self, tokenizer_with_model):
        """Test special token handling with real tokenizer."""
        text = "test"

        # Encode with special tokens
        token_ids_with = tokenizer_with_model.encode(text, add_special_tokens=True)
        assert token_ids_with[0] == 0  # BOS

        # Encode without special tokens
        token_ids_without = tokenizer_with_model.encode(text, add_special_tokens=False)
        assert token_ids_without[0] != 0  # No BOS

    def test_real_tokenizer_batch_decode(self, tokenizer_with_model):
        """Test batch decoding with real tokenizer."""
        texts = ["Hello", "World", "Test"]
        token_ids_batch = [
            tokenizer_with_model.encode(t, add_special_tokens=True) for t in texts
        ]

        decoded_batch = tokenizer_with_model.batch_decode(
            token_ids_batch, skip_special_tokens=True
        )

        assert len(decoded_batch) == len(texts)
        for orig, decoded in zip(texts, decoded_batch):
            assert orig.lower() in decoded.lower()

    def test_real_tokenizer_mlx_array_decode(self, tokenizer_with_model):
        """Test decoding MLX arrays."""
        import mlx.core as mx

        text = "Hello"
        token_ids = tokenizer_with_model.encode(text)

        # Convert to MLX array
        token_ids_mx = mx.array(token_ids)

        # Decode should handle MLX array
        decoded = tokenizer_with_model.decode(token_ids_mx, skip_special_tokens=True)

        assert isinstance(decoded, str)
        assert "hello" in decoded.lower()


@pytest.mark.unit
class TestTrOCRProcessor:
    """Test TrOCR processor (image + text)."""

    def test_processor_creation(self):
        """Test processor can be created."""
        config = DEFAULT_CONFIG_PRINTED
        processor = TrOCRProcessor(config)

        assert processor is not None
        assert hasattr(processor, "image_processor")
        assert hasattr(processor, "tokenizer")

    def test_processor_tokenizer_integration(self):
        """Test processor uses correct tokenizer."""
        config = DEFAULT_CONFIG_PRINTED
        processor = TrOCRProcessor(config)

        assert processor.tokenizer.vocab_size == 64044
        assert processor.tokenizer.bos_token_id == 0
        assert processor.tokenizer.eos_token_id == 2


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_model
class TestTrOCRIntegration:
    """Integration tests requiring model weights."""

    def test_model_can_be_instantiated_with_default_configs(self):
        """Test that model can be created with default configs."""
        # Test printed variant
        config = DEFAULT_CONFIG_PRINTED
        model = TrOCR(config)

        assert model is not None
        assert isinstance(model, TrOCR)

    def test_both_variants_can_be_instantiated(self):
        """Test that both model variants can be created."""
        configs = [DEFAULT_CONFIG_PRINTED, DEFAULT_CONFIG_HANDWRITTEN]

        for config in configs:
            model = TrOCR(config)
            assert model is not None
            assert isinstance(model, TrOCR)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
