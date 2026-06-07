#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Unit tests for smolGenCad model components.

Tests encoder, decoder, model head, and complete model.
"""

import pytest

import mlx.core as mx
import mlx.nn as nn

from smlx.models.smolGenCad.config import (
    DecoderConfig,
    EncoderConfig,
    SmolGenCadConfig,
)
from smlx.models.smolGenCad.decoder import CADDecoder
from smlx.models.smolGenCad.encoder import TextEncoder
from smlx.models.smolGenCad.model import CADHead, SmolGenCad


@pytest.fixture
def encoder_config():
    """Create encoder configuration."""
    return EncoderConfig()


@pytest.fixture
def decoder_config():
    """Create decoder configuration."""
    return DecoderConfig()


@pytest.fixture
def model_config():
    """Create full model configuration."""
    return SmolGenCadConfig()


@pytest.fixture
def text_encoder(encoder_config):
    """Create text encoder instance."""
    return TextEncoder(encoder_config)


@pytest.fixture
def cad_decoder(decoder_config):
    """Create CAD decoder instance."""
    return CADDecoder(decoder_config)


@pytest.fixture
def cad_head(model_config):
    """Create CAD head instance."""
    return CADHead(model_config)


@pytest.fixture
def model(model_config):
    """Create full model instance."""
    return SmolGenCad(model_config)


@pytest.mark.unit
class TestCADHead:
    """Test CAD vocabulary head."""

    def test_initialization(self, model_config):
        """Test CAD head initialization."""
        head = CADHead(model_config)
        assert head.vocab_size == 1104
        assert isinstance(head.lm_head, nn.Linear)

    def test_forward_pass_shape(self, cad_head):
        """Test forward pass produces correct shape."""
        batch_size, seq_len, hidden_size = 2, 10, 256
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_size))

        logits = cad_head(hidden_states)

        assert logits.shape == (batch_size, seq_len, 1104)

    def test_forward_pass_dtype(self, cad_head):
        """Test forward pass maintains dtype."""
        hidden_states = mx.random.normal((1, 5, 256), dtype=mx.float32)
        logits = cad_head(hidden_states)

        assert logits.dtype == mx.float32

    def test_vocab_size_matches_config(self, model_config):
        """Test vocab size matches configuration."""
        head = CADHead(model_config)
        assert head.vocab_size == 1104


@pytest.mark.unit
class TestTextEncoder:
    """Test text encoder."""

    def test_initialization(self, encoder_config):
        """Test encoder initialization."""
        encoder = TextEncoder(encoder_config)
        assert encoder.config == encoder_config
        assert hasattr(encoder, "model")

    def test_hidden_size_property(self, text_encoder):
        """Test hidden size property."""
        assert text_encoder.hidden_size == 576  # SmolLM2-135M

    def test_forward_pass_shape(self, text_encoder):
        """Test forward pass produces correct shape."""
        batch_size, seq_len = 2, 20
        input_ids = mx.random.randint(0, 49152, (batch_size, seq_len))

        embeddings = text_encoder(input_ids)

        assert embeddings.shape == (batch_size, seq_len, 576)

    def test_forward_pass_with_different_lengths(self, text_encoder):
        """Test encoder handles different sequence lengths."""
        # Short sequence
        short_ids = mx.random.randint(0, 49152, (1, 10))
        short_output = text_encoder(short_ids)
        assert short_output.shape == (1, 10, 576)

        # Long sequence
        long_ids = mx.random.randint(0, 49152, (1, 100))
        long_output = text_encoder(long_ids)
        assert long_output.shape == (1, 100, 576)

    def test_batch_processing(self, text_encoder):
        """Test encoder handles batches correctly."""
        batch_sizes = [1, 2, 4, 8]
        seq_len = 32

        for batch_size in batch_sizes:
            input_ids = mx.random.randint(0, 49152, (batch_size, seq_len))
            embeddings = text_encoder(input_ids)
            assert embeddings.shape == (batch_size, seq_len, 576)


@pytest.mark.unit
class TestCADDecoder:
    """Test CAD decoder."""

    def test_initialization(self, decoder_config):
        """Test decoder initialization."""
        decoder = CADDecoder(decoder_config)
        assert decoder.config == decoder_config
        assert hasattr(decoder, "embed_tokens")
        assert hasattr(decoder, "layers")
        assert hasattr(decoder, "norm")

    def test_num_layers(self, cad_decoder, decoder_config):
        """Test decoder has correct number of layers."""
        assert len(cad_decoder.layers) == decoder_config.num_hidden_layers

    def test_forward_pass_shape(self, cad_decoder):
        """Test decoder forward pass produces correct shape."""
        batch_size, seq_len = 2, 15
        cad_input_ids = mx.random.randint(0, 1104, (batch_size, seq_len))

        # Encoder hidden states (from text encoder)
        encoder_hidden_states = mx.random.normal((batch_size, 20, 576))

        decoder_output = cad_decoder(cad_input_ids, encoder_hidden_states=encoder_hidden_states)

        assert decoder_output.shape == (batch_size, seq_len, 256)

    def test_decoder_without_encoder(self, cad_decoder):
        """Test decoder can run without encoder states (for testing)."""
        batch_size, seq_len = 1, 10
        cad_input_ids = mx.random.randint(0, 1104, (batch_size, seq_len))

        # Should work without encoder states (cross-attention will be skipped)
        decoder_output = cad_decoder(cad_input_ids, encoder_hidden_states=None)

        assert decoder_output.shape == (batch_size, seq_len, 256)

    def test_causal_masking(self, cad_decoder):
        """Test decoder uses causal masking internally."""
        batch_size, seq_len = 1, 10
        cad_input_ids = mx.random.randint(0, 1104, (batch_size, seq_len))
        encoder_hidden_states = mx.random.normal((batch_size, 20, 576))

        # Decoder creates causal mask internally
        decoder_output = cad_decoder(cad_input_ids, encoder_hidden_states=encoder_hidden_states)

        assert decoder_output.shape == (batch_size, seq_len, 256)

    def test_batch_processing(self, cad_decoder):
        """Test decoder handles batches correctly."""
        batch_sizes = [1, 2, 4]
        cad_seq_len = 15
        text_seq_len = 20

        for batch_size in batch_sizes:
            cad_input_ids = mx.random.randint(0, 1104, (batch_size, cad_seq_len))
            encoder_hidden_states = mx.random.normal((batch_size, text_seq_len, 576))

            decoder_output = cad_decoder(cad_input_ids, encoder_hidden_states=encoder_hidden_states)

            assert decoder_output.shape == (batch_size, cad_seq_len, 256)


@pytest.mark.unit
class TestSmolGenCadModel:
    """Test complete SmolGenCad model."""

    def test_initialization(self, model_config):
        """Test model initialization."""
        model = SmolGenCad(model_config)
        assert model.config == model_config
        assert isinstance(model.encoder, TextEncoder)
        assert isinstance(model.decoder, CADDecoder)
        assert isinstance(model.cad_head, CADHead)

    def test_encode_method(self, model):
        """Test encode method."""
        batch_size, text_len = 2, 25
        input_ids = mx.random.randint(0, 49152, (batch_size, text_len))

        encoder_outputs = model.encode(input_ids)

        assert encoder_outputs.shape == (batch_size, text_len, 576)

    def test_decode_method(self, model):
        """Test decode method."""
        batch_size, cad_len, text_len = 2, 15, 25
        cad_input_ids = mx.random.randint(0, 1104, (batch_size, cad_len))
        encoder_hidden_states = mx.random.normal((batch_size, text_len, 576))

        decoder_outputs = model.decode(cad_input_ids, encoder_hidden_states)

        assert decoder_outputs.shape == (batch_size, cad_len, 256)

    def test_forward_pass(self, model):
        """Test complete forward pass."""
        batch_size = 2
        text_len, cad_len = 25, 15

        input_ids = mx.random.randint(0, 49152, (batch_size, text_len))
        cad_input_ids = mx.random.randint(0, 1104, (batch_size, cad_len))

        logits = model(input_ids, cad_input_ids)

        assert logits.shape == (batch_size, cad_len, 1104)

    def test_forward_pass_dtype(self, model):
        """Test forward pass maintains dtype."""
        input_ids = mx.random.randint(0, 49152, (1, 10))
        cad_input_ids = mx.random.randint(0, 1104, (1, 5))

        logits = model(input_ids, cad_input_ids)

        assert logits.dtype == mx.float32

    def test_generate_step(self, model):
        """Test single generation step."""
        batch_size = 1
        text_len, cad_len = 20, 5

        cad_input_ids = mx.random.randint(0, 1104, (batch_size, cad_len))
        encoder_hidden_states = mx.random.normal((batch_size, text_len, 576))

        next_token, cache = model.generate_step(
            cad_input_ids,
            encoder_hidden_states,
            temperature=1.0,
        )

        # Next token should be a scalar or 1D array
        assert next_token.size == 1

    def test_generate_step_with_temperature(self, model):
        """Test generation with different temperatures."""
        batch_size = 1
        text_len, cad_len = 20, 5

        cad_input_ids = mx.random.randint(0, 1104, (batch_size, cad_len))
        encoder_hidden_states = mx.random.normal((batch_size, text_len, 576))

        # Test different temperatures
        for temp in [0.5, 1.0, 1.5]:
            next_token, _ = model.generate_step(
                cad_input_ids,
                encoder_hidden_states,
                temperature=temp,
            )
            assert next_token.size == 1

    def test_generate_step_greedy(self, model):
        """Test greedy generation (temperature=0)."""
        batch_size = 1
        text_len, cad_len = 20, 5

        cad_input_ids = mx.random.randint(0, 1104, (batch_size, cad_len))
        encoder_hidden_states = mx.random.normal((batch_size, text_len, 576))

        next_token, _ = model.generate_step(
            cad_input_ids,
            encoder_hidden_states,
            temperature=0.0,  # Greedy
        )

        assert next_token.size == 1

    def test_generate_step_top_k(self, model):
        """Test top-k sampling."""
        batch_size = 1
        text_len, cad_len = 20, 5

        cad_input_ids = mx.random.randint(0, 1104, (batch_size, cad_len))
        encoder_hidden_states = mx.random.normal((batch_size, text_len, 576))

        next_token, _ = model.generate_step(
            cad_input_ids,
            encoder_hidden_states,
            top_k=50,
        )

        assert next_token.size == 1

    def test_generate_step_top_p(self, model):
        """Test nucleus (top-p) sampling."""
        batch_size = 1
        text_len, cad_len = 20, 5

        cad_input_ids = mx.random.randint(0, 1104, (batch_size, cad_len))
        encoder_hidden_states = mx.random.normal((batch_size, text_len, 576))

        next_token, _ = model.generate_step(
            cad_input_ids,
            encoder_hidden_states,
            top_p=0.9,
        )

        assert next_token.size == 1

    def test_num_params_property(self, model):
        """num_params must count the whole (nested) parameter tree.

        Regression: parameters() is a nested dict+list tree, so iterating its
        top-level .values() saw sub-trees (not arrays) and undercounted to 0.
        The model is ~147M params; assert it matches a tree_flatten recount and
        is in the right order of magnitude, not merely ``>= 0``.
        """
        from mlx.utils import tree_flatten

        num_params = model.num_params
        expected = sum(arr.size for _, arr in tree_flatten(model.parameters()))
        assert isinstance(num_params, int)
        assert num_params == expected
        assert num_params > 100_000_000, f"expected ~147M params, got {num_params}"

    def test_num_params_millions_property(self, model):
        """num_params_millions reports the real size (regression for the 0.0M bug)."""
        num_params_millions = model.num_params_millions
        assert isinstance(num_params_millions, float)
        assert num_params_millions == pytest.approx(model.num_params / 1_000_000)
        assert num_params_millions > 100.0

    def test_sanitize_weights(self, model):
        """Test weight sanitization."""
        weights = {
            "encoder.model.layers.0.self_attn.rotary_emb.inv_freq": mx.array([1.0]),
            "encoder.model.layers.0.self_attn.q_proj.weight": mx.random.normal((256, 256)),
            "decoder.layers.0.self_attn.q_proj.weight": mx.random.normal((256, 256)),
        }

        sanitized = model.sanitize(weights)

        # Should remove rotary_emb.inv_freq
        assert "encoder.model.layers.0.self_attn.rotary_emb.inv_freq" not in sanitized
        # Should keep other weights
        assert "encoder.model.layers.0.self_attn.q_proj.weight" in sanitized
        assert "decoder.layers.0.self_attn.q_proj.weight" in sanitized


@pytest.mark.unit
class TestModelIntegration:
    """Test integration between model components."""

    def test_encoder_decoder_dimension_compatibility(self, model):
        """Test encoder output dimensions match decoder cross-attention."""
        batch_size = 1
        text_len = 20

        # Encode text
        input_ids = mx.random.randint(0, 49152, (batch_size, text_len))
        encoder_outputs = model.encode(input_ids)

        # Encoder outputs should be [batch, text_len, 576]
        assert encoder_outputs.shape[2] == 576

        # Decoder expects encoder_hidden_size = 576
        assert model.config.decoder.encoder_hidden_size == 576

    def test_decoder_head_dimension_compatibility(self, model):
        """Test decoder output dimensions match head input."""
        batch_size, cad_len = 1, 10
        cad_input_ids = mx.random.randint(0, 1104, (batch_size, cad_len))
        encoder_hidden_states = mx.random.normal((batch_size, 20, 576))

        # Decode
        decoder_outputs = model.decode(cad_input_ids, encoder_hidden_states)

        # Decoder outputs should be [batch, cad_len, 256]
        assert decoder_outputs.shape[2] == 256

        # Head expects hidden_size = 256
        assert model.config.decoder.hidden_size == 256

    def test_end_to_end_forward_pass(self, model):
        """Test complete end-to-end forward pass."""
        batch_size = 2
        text_len, cad_len = 30, 20

        # Random text and CAD inputs
        input_ids = mx.random.randint(0, 49152, (batch_size, text_len))
        cad_input_ids = mx.random.randint(0, 1104, (batch_size, cad_len))

        # Forward pass
        logits = model(input_ids, cad_input_ids)

        # Check output shape
        assert logits.shape == (batch_size, cad_len, 1104)

        # Check logits are finite
        assert mx.all(mx.isfinite(logits)).item()

    def test_generation_loop_simulation(self, model):
        """Test simulated autoregressive generation loop."""
        batch_size = 1
        text_len = 25
        max_new_tokens = 10

        # Encode text
        input_ids = mx.random.randint(0, 49152, (batch_size, text_len))
        encoder_outputs = model.encode(input_ids)

        # Start with BOS token
        cad_ids = mx.array([[1]], dtype=mx.int32)  # BOS token

        # Generate tokens
        for _ in range(max_new_tokens):
            next_token, _ = model.generate_step(
                cad_ids,
                encoder_outputs,
                temperature=1.0,
            )

            # Append to sequence
            cad_ids = mx.concatenate([cad_ids, mx.array([[next_token]], dtype=mx.int32)], axis=1)

        # Should have BOS + max_new_tokens
        assert cad_ids.shape == (batch_size, 1 + max_new_tokens)

    def test_batch_consistency(self, model):
        """Test batch processing gives consistent shapes."""
        batch_sizes = [1, 2, 4]
        text_len, cad_len = 20, 15

        for batch_size in batch_sizes:
            input_ids = mx.random.randint(0, 49152, (batch_size, text_len))
            cad_input_ids = mx.random.randint(0, 1104, (batch_size, cad_len))

            logits = model(input_ids, cad_input_ids)

            assert logits.shape == (batch_size, cad_len, 1104)


@pytest.mark.unit
class TestModelConfiguration:
    """Test model configuration handling."""

    def test_custom_encoder_config(self):
        """Test model with custom encoder configuration."""
        config = SmolGenCadConfig(encoder=EncoderConfig(hidden_size=576))  # Explicit
        model = SmolGenCad(config)

        assert model.encoder.hidden_size == 576

    def test_custom_decoder_config(self):
        """Test model with custom decoder configuration."""
        config = SmolGenCadConfig(decoder=DecoderConfig(num_hidden_layers=6, hidden_size=256))
        model = SmolGenCad(config)

        assert len(model.decoder.layers) == 6
        assert model.config.decoder.hidden_size == 256

    def test_config_dimension_matching(self):
        """Test configuration ensures dimension matching."""
        config = SmolGenCadConfig()

        # Encoder hidden size should match decoder's encoder_hidden_size
        assert config.encoder.hidden_size == config.decoder.encoder_hidden_size

    def test_model_with_dict_config(self):
        """Test creating model from dict config."""
        config_dict = {
            "model_type": "smolGenCad",
            "encoder": {"hidden_size": 576},
            "decoder": {"hidden_size": 256},
        }

        config = SmolGenCadConfig.from_dict(config_dict)
        model = SmolGenCad(config)

        assert model.encoder.hidden_size == 576
        assert model.decoder.config.hidden_size == 256
