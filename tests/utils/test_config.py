"""Tests for smlx.utils.config module."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from smlx.utils.config import (
    LANGUAGE_MODEL_DEFAULTS,
    BaseModelArgs,
    estimate_parameters,
    load_config,
    merge_configs,
    validate_config,
)


@dataclass
class SampleModelArgs(BaseModelArgs):
    """Sample model configuration for testing."""

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    vocab_size: int = 50257


class TestBaseModelArgs:
    """Test BaseModelArgs class."""

    def test_from_dict_basic(self):
        """Test creating config from dictionary."""
        config_dict = {
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "vocab_size": 30000,
        }

        args = SampleModelArgs.from_dict(config_dict)

        assert args.hidden_size == 512
        assert args.num_hidden_layers == 6
        assert args.num_attention_heads == 8
        assert args.vocab_size == 30000

    def test_from_dict_filters_unknown(self):
        """Test that from_dict filters unknown parameters."""
        config_dict = {
            "hidden_size": 512,
            "unknown_param": "should_be_ignored",
            "another_unknown": 123,
        }

        args = SampleModelArgs.from_dict(config_dict)

        # Should use default for unspecified params
        assert args.hidden_size == 512
        assert args.num_hidden_layers == 12  # Default

    def test_to_dict(self):
        """Test converting config to dictionary."""
        args = SampleModelArgs(hidden_size=512, num_hidden_layers=6)

        config_dict = args.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["hidden_size"] == 512
        assert config_dict["num_hidden_layers"] == 6

    def test_save_load(self):
        """Test saving and loading config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"

            args = SampleModelArgs(hidden_size=512, num_hidden_layers=6)
            args.save(filepath)

            loaded = SampleModelArgs.load(filepath)

            assert loaded.hidden_size == 512
            assert loaded.num_hidden_layers == 6

    def test_save_creates_dir(self):
        """Test that save creates parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "config.json"

            args = SampleModelArgs()
            args.save(filepath)

            assert filepath.exists()
            assert filepath.parent.exists()


class TestLoadConfig:
    """Test load_config function."""

    def test_load_config_from_path(self):
        """Test loading config from file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"

            config_dict = {"hidden_size": 512, "num_hidden_layers": 6}
            with open(filepath, "w") as f:
                json.dump(config_dict, f)

            args = load_config(config_path=filepath, config_class=SampleModelArgs)

            assert args.hidden_size == 512
            assert args.num_hidden_layers == 6

    def test_load_config_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {"hidden_size": 512, "num_hidden_layers": 6}

        args = load_config(config_dict=config_dict, config_class=SampleModelArgs)

        assert args.hidden_size == 512
        assert args.num_hidden_layers == 6

    def test_load_config_with_defaults(self):
        """Test loading config with default values."""
        default_config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        }

        override_config = {"hidden_size": 512}

        args = load_config(
            config_dict=override_config,
            config_class=SampleModelArgs,
            default_config=default_config,
        )

        # Override should take precedence
        assert args.hidden_size == 512
        # Others should come from defaults
        assert args.num_hidden_layers == 12

    def test_load_config_no_source(self):
        """Test that load_config raises error with no source."""
        with pytest.raises(ValueError, match="Either config_path or config_dict"):
            load_config(config_class=SampleModelArgs)


class TestMergeConfigs:
    """Test merge_configs function."""

    def test_merge_configs_basic(self):
        """Test merging two configs."""
        base = {"hidden_size": 768, "num_hidden_layers": 12}
        override = {"hidden_size": 512}

        merged = merge_configs(base, override)

        assert merged["hidden_size"] == 512  # Overridden
        assert merged["num_hidden_layers"] == 12  # From base

    def test_merge_configs_add_new_keys(self):
        """Test that merge adds new keys from override."""
        base = {"hidden_size": 768}
        override = {"num_hidden_layers": 6}

        merged = merge_configs(base, override)

        assert "hidden_size" in merged
        assert "num_hidden_layers" in merged

    def test_merge_configs_preserves_original(self):
        """Test that merge doesn't modify original dicts."""
        base = {"hidden_size": 768}
        override = {"hidden_size": 512}

        merged = merge_configs(base, override)

        # Merged should have override value
        assert merged["hidden_size"] == 512
        # Originals should be unchanged
        assert base["hidden_size"] == 768
        assert override["hidden_size"] == 512


class TestValidateConfig:
    """Test validate_config function."""

    def test_validate_config_valid(self):
        """Test validating a valid config."""
        args = SampleModelArgs(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            vocab_size=50257,
        )

        # Should not raise
        validate_config(args)

    def test_validate_config_negative_hidden_size(self):
        """Test validation catches negative hidden_size."""

        @dataclass
        class BadConfig(BaseModelArgs):
            hidden_size: int = -1

        args = BadConfig()

        with pytest.raises(ValueError, match="hidden_size must be positive"):
            validate_config(args)

    def test_validate_config_negative_layers(self):
        """Test validation catches negative num_hidden_layers."""

        @dataclass
        class BadConfig(BaseModelArgs):
            num_hidden_layers: int = -1

        args = BadConfig()

        with pytest.raises(ValueError, match="num_hidden_layers must be positive"):
            validate_config(args)

    def test_validate_config_negative_vocab_size(self):
        """Test validation catches negative vocab_size."""

        @dataclass
        class BadConfig(BaseModelArgs):
            vocab_size: int = -1

        args = BadConfig()

        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_config(args)

    def test_validate_config_strict_false(self):
        """Test validation with strict=False only warns."""

        @dataclass
        class BadConfig(BaseModelArgs):
            hidden_size: int = -1

        args = BadConfig()

        # Should not raise, only warn
        validate_config(args, strict=False)


class TestEstimateParameters:
    """Test estimate_parameters function."""

    def test_estimate_parameters_basic(self):
        """Test basic parameter estimation."""

        @dataclass
        class SmallModel(BaseModelArgs):
            hidden_size: int = 256
            num_hidden_layers: int = 6
            num_attention_heads: int = 8
            vocab_size: int = 10000
            intermediate_size: int = 1024

        args = SmallModel()
        params = estimate_parameters(args)

        assert params > 0
        # Should be reasonable for a small model
        assert params < 100_000_000  # Less than 100M

    def test_estimate_parameters_with_head_dim(self):
        """Test parameter estimation with explicit head_dim."""

        @dataclass
        class ModelWithHeadDim(BaseModelArgs):
            hidden_size: int = 768
            num_hidden_layers: int = 12
            num_attention_heads: int = 12
            head_dim: int = 64
            vocab_size: int = 50000
            intermediate_size: int = 3072

        args = ModelWithHeadDim()
        params = estimate_parameters(args)

        assert params > 0

    def test_estimate_parameters_no_transformer_attrs(self):
        """Test that estimation returns 0 for non-transformer configs."""

        @dataclass
        class NonTransformerConfig(BaseModelArgs):
            some_param: int = 100

        args = NonTransformerConfig()
        params = estimate_parameters(args)

        # Can't estimate without transformer attributes
        assert params == 0

    def test_estimate_parameters_realistic(self):
        """Test parameter estimation for realistic model sizes."""

        @dataclass
        class SmolLMConfig(BaseModelArgs):
            hidden_size: int = 576
            num_hidden_layers: int = 30
            num_attention_heads: int = 9
            num_key_value_heads: int = 3
            vocab_size: int = 49152
            intermediate_size: int = 1536
            tie_word_embeddings: bool = False

        args = SmolLMConfig()
        params = estimate_parameters(args)

        # Should be roughly 135M parameters
        assert 100_000_000 < params < 200_000_000


class TestLanguageModelDefaults:
    """Test default configurations."""

    def test_language_model_defaults_exist(self):
        """Test that language model defaults are defined."""
        assert isinstance(LANGUAGE_MODEL_DEFAULTS, dict)
        assert "hidden_size" in LANGUAGE_MODEL_DEFAULTS
        assert "num_hidden_layers" in LANGUAGE_MODEL_DEFAULTS

    def test_language_model_defaults_valid(self):
        """Test that defaults have reasonable values."""
        assert LANGUAGE_MODEL_DEFAULTS["hidden_size"] > 0
        assert LANGUAGE_MODEL_DEFAULTS["num_hidden_layers"] > 0
        assert LANGUAGE_MODEL_DEFAULTS["vocab_size"] > 0


class TestConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_from_dict_empty(self):
        """Test from_dict with empty dictionary."""
        args = SampleModelArgs.from_dict({})

        # Should use all defaults
        assert args.hidden_size == 768
        assert args.num_hidden_layers == 12

    def test_to_dict_all_fields(self):
        """Test that to_dict includes all fields."""
        args = SampleModelArgs()
        config_dict = args.to_dict()

        assert "hidden_size" in config_dict
        assert "num_hidden_layers" in config_dict
        assert "num_attention_heads" in config_dict
        assert "vocab_size" in config_dict

    def test_load_config_missing_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config(
                config_path="/nonexistent/config.json",
                config_class=SampleModelArgs,
            )

    def test_merge_configs_empty_dicts(self):
        """Test merging empty dictionaries."""
        merged = merge_configs({}, {})
        assert merged == {}

    def test_validate_config_minimal(self):
        """Test validation with minimal config."""

        @dataclass
        class MinimalConfig(BaseModelArgs):
            value: int = 1

        args = MinimalConfig()

        # Should not raise (no validation rules apply)
        validate_config(args)


class TestConfigIntegration:
    """Test configuration workflow integration."""

    def test_save_load_roundtrip(self):
        """Test that save and load preserve all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"

            original = SampleModelArgs(
                hidden_size=512,
                num_hidden_layers=8,
                num_attention_heads=8,
                vocab_size=32000,
            )

            original.save(filepath)
            loaded = SampleModelArgs.load(filepath)

            assert loaded.hidden_size == original.hidden_size
            assert loaded.num_hidden_layers == original.num_hidden_layers
            assert loaded.num_attention_heads == original.num_attention_heads
            assert loaded.vocab_size == original.vocab_size

    def test_load_hf_style_config(self):
        """Test loading HuggingFace-style config with extra fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"

            # HF configs often have extra metadata
            hf_config = {
                "model_type": "gpt2",
                "hidden_size": 512,
                "num_hidden_layers": 6,
                "_name_or_path": "gpt2-small",
                "architectures": ["GPT2Model"],
                "transformers_version": "4.0.0",
            }

            with open(filepath, "w") as f:
                json.dump(hf_config, f)

            # Should load successfully, ignoring unknown fields
            args = load_config(config_path=filepath, config_class=SampleModelArgs)

            assert args.hidden_size == 512
            assert args.num_hidden_layers == 6
