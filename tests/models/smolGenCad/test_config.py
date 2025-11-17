#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Unit tests for smolGenCad configuration.

Tests configuration classes and validation.
"""

import pytest

from smlx.models.smolGenCad.config import (
    CADVocabularyConfig,
    DecoderConfig,
    EncoderConfig,
    SmolGenCadConfig,
)


class TestCADVocabularyConfig:
    """Test CAD vocabulary configuration."""

    def test_default_config(self):
        """Test default vocabulary configuration."""
        config = CADVocabularyConfig()
        assert config.num_commands == 50
        assert config.max_sequence_length == 272
        assert config.max_parameters_per_command == 16

    def test_custom_config(self):
        """Test custom vocabulary configuration."""
        config = CADVocabularyConfig(
            num_commands=100, max_sequence_length=500
        )
        assert config.num_commands == 100
        assert config.max_sequence_length == 500

    def test_coordinate_ranges(self):
        """Test coordinate range settings."""
        config = CADVocabularyConfig()
        assert config.min_coordinate < config.max_coordinate
        assert config.min_distance > 0

    def test_validation_positive_values(self):
        """Test validation requires positive values."""
        with pytest.raises(AssertionError):
            CADVocabularyConfig(num_commands=0)

        with pytest.raises(AssertionError):
            CADVocabularyConfig(max_sequence_length=-1)


class TestEncoderConfig:
    """Test encoder configuration."""

    def test_default_smollm2_config(self):
        """Test default encoder uses SmolLM2-135M settings."""
        config = EncoderConfig()
        assert config.model_type == "smollm2"
        assert config.hidden_size == 576
        assert config.num_hidden_layers == 30
        assert config.num_attention_heads == 9
        assert config.num_key_value_heads == 3

    def test_encoder_vocab_size(self):
        """Test encoder vocabulary size."""
        config = EncoderConfig()
        assert config.vocab_size == 49152  # SmolLM2 vocab

    def test_encoder_architecture_params(self):
        """Test encoder architecture parameters."""
        config = EncoderConfig()
        assert config.intermediate_size == 1536
        assert config.max_position_embeddings == 2048
        assert config.rms_norm_eps == 1e-5


class TestDecoderConfig:
    """Test decoder configuration."""

    def test_default_decoder_config(self):
        """Test default decoder configuration."""
        config = DecoderConfig()
        assert config.model_type == "cad_decoder"
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 8
        assert config.num_attention_heads == 8

    def test_decoder_has_cross_attention(self):
        """Test decoder enables cross-attention."""
        config = DecoderConfig()
        assert config.cross_attention is True

    def test_decoder_encoder_hidden_size(self):
        """Test decoder knows encoder hidden size."""
        config = DecoderConfig()
        assert config.encoder_hidden_size == 576  # Match encoder

    def test_decoder_mlp_size(self):
        """Test decoder MLP intermediate size."""
        config = DecoderConfig()
        assert config.intermediate_size == 1024  # 4x hidden_size

    def test_decoder_dropout(self):
        """Test decoder has dropout."""
        config = DecoderConfig()
        assert config.dropout == 0.1


@pytest.mark.unit
class TestSmolGenCadConfig:
    """Test complete model configuration."""

    def test_default_config(self):
        """Test default model configuration."""
        config = SmolGenCadConfig()
        assert config.model_type == "smolGenCad"
        assert isinstance(config.encoder, EncoderConfig)
        assert isinstance(config.decoder, DecoderConfig)
        assert isinstance(config.vocabulary, CADVocabularyConfig)

    def test_config_has_generation_params(self):
        """Test configuration has generation parameters."""
        config = SmolGenCadConfig()
        assert config.max_new_tokens == 272
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.top_k == 50

    def test_config_has_special_tokens(self):
        """Test configuration defines special tokens."""
        config = SmolGenCadConfig()
        assert config.bos_token_id == 1
        assert config.eos_token_id == 2
        assert config.pad_token_id == 0

    def test_config_validates_dimensions(self):
        """Test configuration validates encoder-decoder dimensions."""
        # Should work with matching dimensions
        config = SmolGenCadConfig()
        assert (
            config.decoder.encoder_hidden_size == config.encoder.hidden_size
        )

    def test_config_dimension_mismatch_raises_error(self):
        """Test dimension mismatch raises error."""
        with pytest.raises(AssertionError):
            SmolGenCadConfig(
                encoder=EncoderConfig(hidden_size=576),
                decoder=DecoderConfig(encoder_hidden_size=999),  # Mismatch
            )

    def test_total_parameters_estimation(self):
        """Test total parameter estimation."""
        config = SmolGenCadConfig()
        total_params = config.total_parameters()
        # Should be around 158M
        assert 100 <= total_params <= 200  # In millions

    def test_custom_nested_config(self):
        """Test creating config with custom sub-configs."""
        config = SmolGenCadConfig(
            decoder=DecoderConfig(num_hidden_layers=6),
            vocabulary=CADVocabularyConfig(max_sequence_length=100),
        )
        assert config.decoder.num_hidden_layers == 6
        assert config.vocabulary.max_sequence_length == 100

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "model_type": "smolGenCad",
            "max_new_tokens": 200,
            "temperature": 0.9,
        }
        config = SmolGenCadConfig.from_dict(config_dict)
        assert config.max_new_tokens == 200
        assert config.temperature == 0.9

    def test_config_from_dict_with_nested(self):
        """Test creating config from dict with nested configs."""
        config_dict = {
            "model_type": "smolGenCad",
            "decoder": {"num_hidden_layers": 6, "hidden_size": 128},
        }
        config = SmolGenCadConfig.from_dict(config_dict)
        assert config.decoder.num_hidden_layers == 6
        assert config.decoder.hidden_size == 128

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = SmolGenCadConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "model_type" in config_dict
        assert "encoder" in config_dict
        assert "decoder" in config_dict
        assert "vocabulary" in config_dict

    def test_config_post_init_instantiates_sub_configs(self):
        """Test __post_init__ instantiates dict sub-configs."""
        config = SmolGenCadConfig(
            encoder={"hidden_size": 576},  # Dict instead of object
            decoder={"hidden_size": 256},
        )
        # Should be converted to proper config objects
        assert isinstance(config.encoder, EncoderConfig)
        assert isinstance(config.decoder, DecoderConfig)
        assert config.encoder.hidden_size == 576

    def test_generation_params_customization(self):
        """Test customizing generation parameters."""
        config = SmolGenCadConfig(
            temperature=0.5, top_p=0.9, top_k=100
        )
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.top_k == 100


@pytest.mark.unit
class TestConfigSerialization:
    """Test configuration serialization."""

    def test_roundtrip_serialization(self):
        """Test config can be serialized and deserialized."""
        original = SmolGenCadConfig(
            max_new_tokens=200, temperature=0.9
        )

        # To dict and back
        config_dict = original.to_dict()
        restored = SmolGenCadConfig.from_dict(config_dict)

        assert restored.max_new_tokens == original.max_new_tokens
        assert restored.temperature == original.temperature
        assert restored.decoder.num_hidden_layers == original.decoder.num_hidden_layers
