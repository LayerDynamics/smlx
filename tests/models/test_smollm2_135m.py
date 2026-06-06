# Copyright © 2025 SMLX Project

"""
Unit tests for SmolLM2-135M model implementation.

Tests the core model architecture, configuration, and forward pass.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from smlx.models.SmolLM2_135M import (
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
        assert config.hidden_size == 576
        assert config.num_hidden_layers == 30
        assert config.num_attention_heads == 9
        assert config.num_key_value_heads == 3
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

        # Check no_rope_layers was set by post_init
        assert config.no_rope_layers is not None, "no_rope_layers should be set by post_init"
        assert len(config.no_rope_layers) == config.num_hidden_layers

        # Default behavior: all layers use RoPE (value=1, NoPE disabled)
        # This is correct for standard SmolLM2-135M checkpoints
        for i in range(config.num_hidden_layers):
            assert config.no_rope_layers[i] == 1, \
                f"Layer {i}: expected RoPE enabled (1), got {config.no_rope_layers[i]}"

    def test_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "model_type": "smollm",
            "hidden_size": 576,
            "num_hidden_layers": 30,
            "intermediate_size": 1536,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "vocab_size": 49152,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 8192,
        }

        config = ModelArgs.from_dict(config_dict)
        assert config.hidden_size == 576
        assert config.num_hidden_layers == 30


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
        """Test Attention module creation."""
        attn = Attention(config)

        assert attn.n_heads == config.num_attention_heads
        assert attn.n_kv_heads == config.num_key_value_heads
        assert hasattr(attn, "rope")

    def test_attention_forward(self, config):
        """Test Attention forward pass."""
        attn = Attention(config)

        batch_size = 2
        seq_len = 10
        x = mx.random.normal((batch_size, seq_len, config.hidden_size))

        output = attn(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_mlp_creation(self, config):
        """Test MLP module creation."""
        mlp = MLP(config)

        assert hasattr(mlp, "gate_proj")
        assert hasattr(mlp, "up_proj")
        assert hasattr(mlp, "down_proj")

    def test_mlp_forward(self, config):
        """Test MLP forward pass."""
        mlp = MLP(config)

        batch_size = 2
        seq_len = 10
        x = mx.random.normal((batch_size, seq_len, config.hidden_size))

        output = mlp(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_transformer_block(self, config):
        """Test TransformerBlock creation and forward pass."""
        block = TransformerBlock(config)

        batch_size = 2
        seq_len = 10
        x = mx.random.normal((batch_size, seq_len, config.hidden_size))

        output = block(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_nope_module(self):
        """Test NoPE (No Positional Encoding) module."""
        nope = NoPE()

        x = mx.random.normal((2, 10, 64))
        output = nope(x, offset=5)

        # NoPE should return input unchanged
        assert mx.allclose(output, x)


# ============================================================================
# Full Model Tests
# ============================================================================


@pytest.mark.unit
class TestSmolLM2Model:
    """Test the complete SmolLM2-135M model."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return get_default_config()

    @pytest.fixture
    def model(self, config):
        """Provide test model instance."""
        return Model(config)

    def test_model_creation(self, model, config):
        """Test model instantiation."""
        assert isinstance(model, Model)
        assert isinstance(model.model, LlamaModel)
        assert len(model.layers) == config.num_hidden_layers

    def test_model_forward_shape(self, model, config):
        """Test model forward pass output shape."""
        batch_size = 2
        seq_len = 10

        # Create input token IDs
        inputs = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = model(inputs)

        # Check output shape
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape

    def test_model_forward_dtype(self, model, config):
        """Test model forward pass dtype."""
        inputs = mx.random.randint(0, config.vocab_size, (1, 5))
        logits = model(inputs)

        # Should return float32 by default
        assert logits.dtype in [mx.float32, mx.float16, mx.bfloat16]

    def test_model_with_cache(self, model, config):
        """Test model forward pass with KV cache."""
        seq_len = 10
        inputs = mx.random.randint(0, config.vocab_size, (1, seq_len))

        # Create cache
        cache = [KVCache() for _ in model.layers]

        # Forward pass with cache
        logits = model(inputs, cache=cache)

        assert logits.shape == (1, seq_len, config.vocab_size)

        # Check that cache was updated
        for c in cache:
            assert c.offset > 0
            assert c.keys is not None
            assert c.values is not None

    def test_nope_layers_applied(self, model, config):
        """Test that NoPE is applied to correct layers."""
        # Check that every 4th layer has NoPE instead of RoPE
        for idx, use_rope in enumerate(config.no_rope_layers):
            rope_module = model.model.layers[idx].self_attn.rope

            if not use_rope:
                # Should be NoPE (no-op)
                assert isinstance(rope_module, NoPE), \
                    f"Layer {idx} should have NoPE but has {type(rope_module)}"
            else:
                # Should be regular RoPE
                assert isinstance(rope_module, nn.RoPE), \
                    f"Layer {idx} should have RoPE but has {type(rope_module)}"

    def test_model_sanitize(self, model):
        """Test weight sanitization."""
        # Create mock weights with rotary_emb.inv_freq
        weights = {
            "model.embed_tokens.weight": mx.zeros((100, 64)),
            "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((32,)),
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64)),
        }

        sanitized = model.sanitize(weights)

        # Should remove rotary_emb.inv_freq
        assert "model.layers.0.self_attn.rotary_emb.inv_freq" not in sanitized
        # Should keep other weights
        assert "model.embed_tokens.weight" in sanitized
        assert "model.layers.0.self_attn.q_proj.weight" in sanitized

    def test_model_tied_embeddings(self, config):
        """Test model with tied embeddings."""
        config.tie_word_embeddings = True
        model = Model(config)

        # Should not have lm_head when embeddings are tied
        assert not hasattr(model, "lm_head")

        # Forward pass should still work
        inputs = mx.random.randint(0, config.vocab_size, (1, 5))
        logits = model(inputs)
        assert logits.shape == (1, 5, config.vocab_size)

    def test_model_untied_embeddings(self, config):
        """Test model with untied embeddings."""
        config.tie_word_embeddings = False
        model = Model(config)

        # Should have lm_head when embeddings are untied
        assert hasattr(model, "lm_head")
        assert isinstance(model.lm_head, nn.Linear)

        # Forward pass should work
        inputs = mx.random.randint(0, config.vocab_size, (1, 5))
        logits = model(inputs)
        assert logits.shape == (1, 5, config.vocab_size)


# ============================================================================
# Cache Tests
# ============================================================================


@pytest.mark.unit
class TestKVCache:
    """Test KV cache implementation."""

    def test_cache_initialization(self):
        """Test cache starts empty."""
        cache = KVCache()

        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0

    def test_cache_update(self):
        """Test cache update and fetch."""
        cache = KVCache()

        # Create test keys and values
        batch_size, n_heads, seq_len, head_dim = 2, 4, 5, 64
        keys = mx.random.normal((batch_size, n_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, n_heads, seq_len, head_dim))

        # Update cache
        all_keys, all_values = cache.update_and_fetch(keys, values)

        # Check that cache was updated
        assert cache.offset == seq_len
        assert all_keys.shape == keys.shape
        assert all_values.shape == values.shape
        assert mx.allclose(all_keys, keys)
        assert mx.allclose(all_values, values)

    def test_cache_incremental_update(self):
        """Test incrementally adding to cache."""
        cache = KVCache()

        batch_size, n_heads, head_dim = 1, 4, 64

        # First update
        keys1 = mx.random.normal((batch_size, n_heads, 5, head_dim))
        values1 = mx.random.normal((batch_size, n_heads, 5, head_dim))
        cache.update_and_fetch(keys1, values1)

        assert cache.offset == 5

        # Second update
        keys2 = mx.random.normal((batch_size, n_heads, 3, head_dim))
        values2 = mx.random.normal((batch_size, n_heads, 3, head_dim))
        all_keys, all_values = cache.update_and_fetch(keys2, values2)

        # Cache should now have 8 tokens
        assert cache.offset == 8
        assert all_keys.shape[2] == 8
        assert all_values.shape[2] == 8


# ============================================================================
# Parameter Count Test
# ============================================================================


@pytest.mark.unit
def test_parameter_count():
    """Test that model has approximately 135M parameters."""
    config = get_default_config()
    model = Model(config)

    # Count parameters recursively
    def count_params(params):
        """Recursively count parameters in nested dict/list."""
        total = 0
        if isinstance(params, dict):
            for value in params.values():
                total += count_params(value)
        elif isinstance(params, list):
            for item in params:
                total += count_params(item)
        elif isinstance(params, mx.array):
            total += params.size
        return total

    total_params = count_params(model.parameters())

    # Should be approximately 135M (allow 10% tolerance)
    expected_params = 135_000_000
    tolerance = 0.1 * expected_params

    assert abs(total_params - expected_params) < tolerance, \
        f"Expected ~{expected_params/1e6:.1f}M params, got {total_params/1e6:.1f}M"

    print(f"Total parameters: {total_params/1e6:.2f}M")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
