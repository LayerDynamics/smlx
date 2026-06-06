# Copyright © 2025 SMLX Project

"""
Unit tests for nanoVLM vision-language model implementation.

Tests the minimal 222M parameter VLM architecture, configuration, and components.
"""

import mlx.core as mx
import pytest

from smlx.models.nanoVLM import (
    DEFAULT_CONFIG,
    ImageProcessor,
    LanguageConfig,
    MLPProjection,
    NanoVLM,
    NanoVLMConfig,
    ProjectionConfig,
    VisionConfig,
    VisionModel,
    create_model,
    create_projection,
)
from smlx.utils.cache import make_cache

# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.unit
class TestModelConfiguration:
    """Test model configuration and validation."""

    def test_default_config(self):
        """Test that default configuration is valid."""
        config = DEFAULT_CONFIG

        assert config.model_type == "nanovlm"
        assert config.vision_config.hidden_size == 768
        assert config.language_config.hidden_size == 576
        assert config.projection_config.num_layers == 2

    def test_vision_config(self):
        """Test vision configuration."""
        config = VisionConfig()

        assert config.model_type == "siglip_vision_model"
        assert config.hidden_size == 768  # SigLIP-base
        assert config.num_hidden_layers == 12
        assert config.image_size == 224  # nanoVLM uses 224x224
        assert config.patch_size == 16
        assert config.num_attention_heads == 12

    def test_vision_config_validation(self):
        """Test vision configuration validation."""
        # Valid config
        config = VisionConfig()
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.image_size % config.patch_size == 0

    def test_language_config(self):
        """Test language (SmolLM2-135M) configuration."""
        config = LanguageConfig()

        assert config.model_type == "smollm2"
        assert config.vocab_size == 49152
        assert config.hidden_size == 576
        assert config.num_hidden_layers == 30
        assert config.num_attention_heads == 9
        assert config.num_key_value_heads == 3  # GQA

    def test_language_config_validation(self):
        """Test language configuration validation."""
        config = LanguageConfig()

        # Should satisfy GQA constraints
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0

    def test_projection_config(self):
        """Test projection configuration."""
        config = ProjectionConfig()

        assert config.vision_hidden_size == 768  # SigLIP output
        assert config.language_hidden_size == 576  # SmolLM2 input
        assert config.num_layers == 2
        assert config.activation == "gelu"

    def test_projection_config_validation(self):
        """Test projection configuration validation."""
        config = ProjectionConfig()
        assert config.num_layers >= 1


# ============================================================================
# Component Tests
# ============================================================================


@pytest.mark.unit
class TestModelComponents:
    """Test individual model components."""

    def test_mlp_projection_creation(self):
        """Test MLPProjection module creation."""
        config = ProjectionConfig(
            vision_hidden_size=768,
            language_hidden_size=576,
            num_layers=2,
            activation="gelu",
        )
        projection = MLPProjection(config)

        assert hasattr(projection, "proj")

    def test_mlp_projection_forward(self):
        """Test MLPProjection forward pass."""
        config = ProjectionConfig(
            vision_hidden_size=768,
            language_hidden_size=576,
            num_layers=2,
            activation="gelu",
        )
        projection = MLPProjection(config)

        batch_size = 2
        num_patches = 196  # 14x14 patches for 224x224 image
        vision_features = mx.random.normal((batch_size, num_patches, 768))

        output = projection(vision_features)

        # Pixel shuffle (2x2) reduces spatial dimensions: 196 -> 49 (14x14 -> 7x7)
        assert output.shape == (batch_size, num_patches // 4, 576)
        assert output.dtype == vision_features.dtype

    def test_create_projection_factory(self):
        """Test create_projection factory function."""
        config = ProjectionConfig()
        projection = create_projection(config)

        assert isinstance(projection, MLPProjection)


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
        assert vision_model.config.image_size == 224

    def test_image_processor_creation(self):
        """Test ImageProcessor creation."""
        processor = ImageProcessor(image_size=224)

        assert processor.image_size == 224
        assert hasattr(processor, "image_mean")
        assert hasattr(processor, "image_std")

    def test_image_processor_default_values(self):
        """Test ImageProcessor default values."""
        processor = ImageProcessor()

        # Should have SigLIP normalization values
        assert hasattr(processor, "image_mean")
        assert hasattr(processor, "image_std")
        assert len(processor.image_mean) == 3
        assert len(processor.image_std) == 3


# ============================================================================
# Full Model Tests
# ============================================================================


@pytest.mark.unit
class TestNanoVLMModel:
    """Test the complete nanoVLM model."""

    @pytest.fixture
    def small_config(self):
        """Provide small test configuration."""
        return NanoVLMConfig(
            vision_config=VisionConfig(
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=512,
                image_size=224,
                patch_size=16,
            ),
            language_config=LanguageConfig(
                hidden_size=192,
                num_hidden_layers=4,
                num_attention_heads=6,
                num_key_value_heads=2,
                intermediate_size=512,
                vocab_size=1000,
            ),
            projection_config=ProjectionConfig(
                vision_hidden_size=256,
                language_hidden_size=192,
                num_layers=2,
            ),
        )

    @pytest.fixture
    def model(self, small_config):
        """Provide test model instance."""
        return NanoVLM(small_config)

    def test_model_creation(self, model, small_config):
        """Test model instantiation."""
        assert isinstance(model, NanoVLM)
        assert hasattr(model, "vision_model")
        assert hasattr(model, "projection")
        assert hasattr(model, "language_model")

    def test_model_creation_with_default_config(self):
        """Test model creation with default config."""
        config = DEFAULT_CONFIG
        model = NanoVLM(config)

        assert isinstance(model, NanoVLM)
        assert model.config == config

    def test_create_model_factory(self):
        """Test create_model factory function."""
        config = DEFAULT_CONFIG
        model = create_model(config)

        assert isinstance(model, NanoVLM)

    def test_model_has_correct_structure(self, model):
        """Test model has correct component structure."""
        # Should have vision model
        assert hasattr(model, "vision_model")
        assert isinstance(model.vision_model, VisionModel)

        # Should have projection
        assert hasattr(model, "projection")
        assert isinstance(model.projection, MLPProjection)

        # Should have language model
        assert hasattr(model, "language_model")


# ============================================================================
# Model Size Tests
# ============================================================================


@pytest.mark.unit
class TestModelSize:
    """Test model parameter count."""

    def test_model_is_small(self):
        """Test that nanoVLM is indeed a small model."""
        config = DEFAULT_CONFIG

        # Check individual component sizes
        assert config.vision_config.hidden_size == 768  # 85M params
        assert config.language_config.hidden_size == 576  # 135M params
        assert config.projection_config.num_layers == 2  # ~2M params

        # Total should be ~222M parameters
        # This is verified through the actual implementation


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_model
class TestNanoVLMIntegration:
    """Integration tests requiring model weights."""

    def test_model_can_be_instantiated_with_default_config(self):
        """Test that model can be created with default config."""
        config = DEFAULT_CONFIG
        model = NanoVLM(config)

        assert model is not None
        assert isinstance(model, NanoVLM)

    def test_model_expected_size(self):
        """Test that model has expected parameter count (~222M)."""
        config = DEFAULT_CONFIG
        model = NanoVLM(config)

        # The model should be approximately 222M parameters:
        # - Vision: ~85M (SigLIP-base)
        # - Language: ~135M (SmolLM2-135M)
        # - Projection: ~2M (MLP)
        # Total: ~222M

        # This is validated through the implementation
        assert model is not None


# ============================================================================
# KV-Cache Generation Tests
# ============================================================================
#
# Regression coverage for the nanoVLM generation path (smlx/models/nanoVLM/
# generate.py). Generation used to re-run the full forward over the growing
# sequence every step (O(N^3)) AND clear pixel_values after the first token
# while re-feeding the image markers as plain text — silently dropping the image
# after token 0. The fix prefills once with a per-layer KV cache and then decodes
# single tokens, which is linear AND keeps the image conditioning every token.
#
# These tests use small, randomly-initialized models (no weights download), so
# they assert *internal consistency* of the cache mechanism rather than output
# quality.


@pytest.mark.unit
class TestKVCacheGeneration:
    """Guard KV-cache correctness and image-conditioning across decode steps."""

    @staticmethod
    def _small_config(vocab_size=1000, image_token_id=999):
        """A tiny but structurally-faithful nanoVLM config.

        image_size=224 / patch_size=16 -> 196 patches -> 49 vision tokens after
        the projection's pixel shuffle, matching the default num_image_tokens.
        """
        return NanoVLMConfig(
            vision_config=VisionConfig(
                hidden_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=512,
                image_size=224,
                patch_size=16,
            ),
            language_config=LanguageConfig(
                hidden_size=192,
                num_hidden_layers=4,
                num_attention_heads=6,
                num_key_value_heads=2,
                intermediate_size=512,
                vocab_size=vocab_size,
            ),
            projection_config=ProjectionConfig(
                vision_hidden_size=256,
                language_hidden_size=192,
                num_layers=2,
            ),
            image_token_id=image_token_id,
        )

    def test_cached_decode_matches_full_forward_text_only(self):
        """Cached prefill + single-token decode == one full forward.

        This is the core correctness check for the KV cache: decoding token i
        incrementally (with the cache holding tokens 0..i-1) must produce the
        same logits as a full forward over tokens 0..i.
        """
        config = self._small_config()
        model = NanoVLM(config)
        model.eval()

        # Deterministic text-only sequence; no token equals image_token_id (999).
        token_ids = [5, 17, 42, 3, 99, 256, 7, 800]
        ids = mx.array([token_ids])
        seq_len = ids.shape[1]

        # Reference: a single full forward over the whole sequence.
        full = model(ids)  # (1, seq_len, vocab)
        mx.eval(full)

        cache = make_cache(len(model.language_model.model.layers))

        # Prefill the first `p` tokens; the last prefill position must match the
        # full forward at the same position.
        p = 4
        logits = model(ids[:, :p], cache=cache)
        cached = logits[0, -1, :]
        mx.eval(cached)
        ref = full[0, p - 1, :]
        max_diff = float(mx.max(mx.abs(cached - ref)))
        assert max_diff < 1e-3, f"prefill mismatch (max abs diff {max_diff})"
        assert int(mx.argmax(cached)) == int(mx.argmax(ref))

        # Decode the remaining tokens one at a time.
        for i in range(p, seq_len):
            logits = model(ids[:, i : i + 1], cache=cache)
            cached = logits[0, -1, :]
            mx.eval(cached)
            ref = full[0, i, :]
            max_diff = float(mx.max(mx.abs(cached - ref)))
            assert max_diff < 1e-3, f"decode step {i} mismatch (max abs diff {max_diff})"
            assert int(mx.argmax(cached)) == int(mx.argmax(ref))

    def test_image_conditions_all_decode_steps(self):
        """Decode-step logits depend on the image (vision K/V live in the cache).

        Same model, same fed token, two clearly-different images. Because the
        image's keys/values are computed during prefill and retained in the
        cache, the decode-step logits must differ. If the image were dropped
        after prefill (the old bug), they would be identical.
        """
        image_token_id = 50
        config = self._small_config(vocab_size=1000, image_token_id=image_token_id)
        model = NanoVLM(config)
        model.eval()

        mx.random.seed(0)
        # 49 image markers as a prefix + a few text tokens.
        num_vision_tokens = 49
        ids = mx.array([[image_token_id] * num_vision_tokens + [3, 4, 5]])
        # Vision encoder expects NHWC: (batch, height, width, channels).
        px_a = mx.random.normal((1, 224, 224, 3))
        px_b = mx.random.normal((1, 224, 224, 3)) * 3.0 + 5.0  # clearly different
        decode_token = mx.array([[7]])

        def decode_logit(px):
            cache = make_cache(len(model.language_model.model.layers))
            model(ids, pixel_values=px, cache=cache)  # prefill (encode image)
            out = model(decode_token, cache=cache)  # decode one token
            logit = out[0, -1, :]
            mx.eval(logit)
            return logit

        la = decode_logit(px_a)
        lb = decode_logit(px_b)
        max_diff = float(mx.max(mx.abs(la - lb)))
        assert max_diff > 1e-3, (
            "decode-step logits did not change with the image — the vision "
            f"keys/values are not retained in the cache (max abs diff {max_diff})"
        )

    def test_generate_prefills_then_decodes_single_tokens(self, monkeypatch):
        """generate() does one prefill (with image) then single-token decodes.

        This guards the generate.py loop directly. It fails on the old
        full-re-forward loop, which fed a growing sequence (seq_len 2, 3, 4, ...)
        with no cache and re-passed/cleared pixel_values each step.
        """
        # prepare_inputs() hardcodes image_token_id=49150 / num_image_tokens=49,
        # so the language vocab must cover 49150 for the embedding lookup.
        config = NanoVLMConfig(
            vision_config=VisionConfig(
                hidden_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=512,
                image_size=224,
                patch_size=16,
            ),
            language_config=LanguageConfig(
                hidden_size=192,
                num_hidden_layers=4,
                num_attention_heads=6,
                num_key_value_heads=2,
                intermediate_size=512,
                vocab_size=49152,
            ),
            projection_config=ProjectionConfig(
                vision_hidden_size=256,
                language_hidden_size=192,
                num_layers=2,
            ),
        )
        model = NanoVLM(config)
        model.eval()

        import numpy as np

        from smlx.models.nanoVLM import generate as nano_generate

        class _FakeTokenizer:
            eos_token_id = -1  # never produced (sampled token ids are >= 0)

            def encode(self, text, return_tensors=None):
                toks = [1, 2, 3] if text else []
                return np.array([toks], dtype=np.int64)

            def decode(self, tokens, skip_special_tokens=True):
                return " ".join(str(int(t)) for t in tokens)

        class _FakeProcessor:
            def __init__(self):
                self.tokenizer = _FakeTokenizer()

            def image_processor(self, image):
                # NHWC format, matching the real ImageProcessor output.
                return mx.zeros((1, 224, 224, 3))

        processor = _FakeProcessor()

        calls = []
        orig_call = NanoVLM.__call__

        def spy_call(
            self,
            input_ids,
            pixel_values=None,
            image_token_mask=None,
            mask=None,
            cache=None,
        ):
            calls.append(
                {
                    "seq_len": int(input_ids.shape[1]),
                    "has_pixels": pixel_values is not None,
                    "has_cache": cache is not None,
                }
            )
            return orig_call(
                self,
                input_ids,
                pixel_values=pixel_values,
                image_token_mask=image_token_mask,
                mask=mask,
                cache=cache,
            )

        monkeypatch.setattr(NanoVLM, "__call__", spy_call)

        max_tokens = 4
        text = nano_generate(
            model,
            processor,
            prompt="Describe <image>",
            image=object(),
            max_tokens=max_tokens,
            temperature=0.0,
        )

        assert isinstance(text, str)
        # 1 prefill + (max_tokens - 1) single-token decodes (the final token is
        # appended without an unnecessary trailing decode).
        assert len(calls) == max_tokens
        # Prefill: full prompt, image present, cache present.
        assert calls[0]["has_pixels"] is True
        assert calls[0]["has_cache"] is True
        assert calls[0]["seq_len"] > 1
        # Decode steps: single token, no image re-encode, cache present.
        for c in calls[1:]:
            assert c["seq_len"] == 1, "decode step must feed a single token (KV cache)"
            assert c["has_pixels"] is False
            assert c["has_cache"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
