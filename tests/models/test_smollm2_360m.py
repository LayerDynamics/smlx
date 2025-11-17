# Copyright © 2025 SMLX Project

"""
Unit tests for SmolLM2-360M model implementation.

Tests the core model architecture, configuration, and forward pass.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.models.SmolLM2_360M import (
    MLP,
    Attention,
    KVCache,
    LlamaModel,
    Model,
    ModelArgs,
    NoPE,
    TransformerBlock,
    get_default_config,
    validate_config,
)

# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.unit
class TestModelConfiguration:
    """Test model configuration and validation."""

    def test_default_config(self):
        """Test that default configuration is valid."""
        config = get_default_config()

        assert config.model_type == "smollm"
        assert config.hidden_size == 960
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 15
        assert config.num_key_value_heads == 5
        assert config.vocab_size == 49152
        assert config.no_rope_layer_interval == 4

    def test_config_validation(self):
        """Test configuration validation."""
        config = get_default_config()
        # Should not raise
        validate_config(config)

    def test_config_post_init(self):
        """Test that post_init sets defaults correctly."""
        config = get_default_config()

        # Check no_rope_layers was auto-generated
        assert config.no_rope_layers is not None, "no_rope_layers should be set by post_init"
        assert len(config.no_rope_layers) == config.num_hidden_layers

        # Every 4th layer should have no RoPE (value=0)
        for i in range(config.num_hidden_layers):
            expected = int((i + 1) % config.no_rope_layer_interval != 0)
            assert config.no_rope_layers[i] == expected, \
                f"Layer {i}: expected {expected}, got {config.no_rope_layers[i]}"

    def test_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "model_type": "smollm",
            "hidden_size": 960,
            "num_hidden_layers": 32,
            "intermediate_size": 2560,
            "num_attention_heads": 15,
            "num_key_value_heads": 5,
            "vocab_size": 49152,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 8192,
        }

        config = ModelArgs.from_dict(config_dict)
        assert config.hidden_size == 960
        assert config.num_hidden_layers == 32


# ============================================================================
# Model Component Tests
# ============================================================================


@pytest.mark.unit
class TestModelComponents:
    """Test individual model components."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return get_default_config()

    def test_attention_creation(self, config):
        """Test Attention layer creation."""
        attention = Attention(config)
        assert isinstance(attention, nn.Module)

        # Check shapes
        assert attention.n_heads == config.num_attention_heads
        assert attention.n_kv_heads == config.num_key_value_heads

    def test_mlp_creation(self, config):
        """Test MLP layer creation."""
        mlp = MLP(config)
        assert isinstance(mlp, nn.Module)

    def test_transformer_block_creation(self, config):
        """Test TransformerBlock creation."""
        block = TransformerBlock(config)
        assert isinstance(block, nn.Module)

    def test_nope_creation(self, config):
        """Test NoPE (No Positional Encoding) module."""
        nope = NoPE()
        assert isinstance(nope, nn.Module)

    def test_attention_forward(self, config):
        """Test Attention forward pass."""
        attention = Attention(config)

        # Create input
        batch_size = 2
        seq_len = 10
        x = mx.random.normal((batch_size, seq_len, config.hidden_size))

        # Forward pass
        output = attention(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_mlp_forward(self, config):
        """Test MLP forward pass."""
        mlp = MLP(config)

        # Create input
        batch_size = 2
        seq_len = 10
        x = mx.random.normal((batch_size, seq_len, config.hidden_size))

        # Forward pass
        output = mlp(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_transformer_block_forward(self, config):
        """Test TransformerBlock forward pass."""
        block = TransformerBlock(config)

        # Create input
        batch_size = 2
        seq_len = 10
        x = mx.random.normal((batch_size, seq_len, config.hidden_size))

        # Forward pass
        output = block(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)


# ============================================================================
# Full Model Tests
# ============================================================================


@pytest.mark.unit
class TestFullModel:
    """Test the complete model."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return get_default_config()

    def test_model_creation(self, config):
        """Test model creation."""
        model = Model(config)
        assert isinstance(model, nn.Module)

    def test_model_forward(self, config):
        """Test model forward pass."""
        model = Model(config)

        # Create input token IDs
        batch_size = 2
        seq_len = 10
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = model(input_ids)

        # Check output shape
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape

    def test_model_with_cache(self, config):
        """Test model with KV cache."""
        model = Model(config)

        # Create cache
        cache = [KVCache() for _ in range(config.num_hidden_layers)]

        # First forward pass
        batch_size = 1
        seq_len = 5
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
        logits1 = model(input_ids, cache=cache)

        # Check cache was populated
        assert cache[0].offset == seq_len

        # Second forward pass (single token)
        next_token = mx.random.randint(0, config.vocab_size, (batch_size, 1))
        logits2 = model(next_token, cache=cache)

        # Check cache was updated
        assert cache[0].offset == seq_len + 1

        # Check output shapes
        assert logits1.shape == (batch_size, seq_len, config.vocab_size)
        assert logits2.shape == (batch_size, 1, config.vocab_size)


# ============================================================================
# KV Cache Tests
# ============================================================================


@pytest.mark.unit
class TestKVCache:
    """Test KV cache implementations."""

    def test_kv_cache_creation(self):
        """Test KVCache creation."""
        cache = KVCache()
        assert cache.offset == 0
        assert cache.keys is None
        assert cache.values is None

    def test_kv_cache_update(self):
        """Test KVCache update."""
        cache = KVCache()

        # Create key and value tensors
        batch_size = 2
        num_heads = 4
        seq_len = 10
        head_dim = 64

        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # Update cache
        updated_keys, updated_values = cache.update_and_fetch(keys, values)

        # Check cache state
        assert cache.offset == seq_len
        assert updated_keys.shape == keys.shape
        assert updated_values.shape == values.shape


# ============================================================================
# Parameter Count Tests
# ============================================================================


@pytest.mark.unit
class TestParameterCount:
    """Test model parameter counts."""

    def test_parameter_count(self):
        """Test that model has approximately 360M parameters."""
        config = get_default_config()
        model = Model(config)

        # Count parameters - need to handle nested dicts
        def count_params(params_dict):
            total = 0
            for k, v in params_dict.items():
                if isinstance(v, dict):
                    total += count_params(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            total += count_params(item)
                elif hasattr(v, 'size'):
                    total += v.size
            return total

        num_params = count_params(model.parameters())

        # Should be approximately 360M parameters
        # Allow 10% margin for minor variations
        expected = 360_000_000
        margin = expected * 0.10
        assert expected - margin <= num_params <= expected + margin, \
            f"Expected ~360M params, got {num_params:,}"

        print(f"\nModel has {num_params:,} parameters")
