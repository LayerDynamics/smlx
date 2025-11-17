"""
Pytest configuration and fixtures for quantization tests.

Provides test models and utilities for integration testing.
"""

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest


def _can_download_models():
    """Check if we should attempt to download models for testing."""
    # Allow opt-in via environment variable
    return os.getenv("SMLX_DOWNLOAD_TEST_MODELS", "0") == "1"


def _check_huggingface_available():
    """Check if HuggingFace transformers is available."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def tiny_llama_model():
    """
    Load a tiny LLaMA-style model for integration testing.

    Uses TinyLlama-1.1B or creates a minimal synthetic model if not available.
    """
    if not _can_download_models() or not _check_huggingface_available():
        pytest.skip("Test models not available. Set SMLX_DOWNLOAD_TEST_MODELS=1 to enable")

    # Try to load TinyLlama (smallest real LLaMA model)
    try:
        from transformers import AutoTokenizer

        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # For now, create a synthetic model structure
        # Full model loading would require mlx-lm integration
        pytest.skip("Model loading requires mlx-lm integration")

    except Exception as e:
        pytest.skip(f"Could not load test model: {e}")


@pytest.fixture
def synthetic_transformer_model():
    """
    Create a minimal synthetic transformer model for testing.

    This is a tiny model that mimics LLaMA structure but is fast to initialize.
    """
    class SyntheticAttention(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.q_proj = nn.Linear(dims, dims)
            self.k_proj = nn.Linear(dims, dims)
            self.v_proj = nn.Linear(dims, dims)
            self.o_proj = nn.Linear(dims, dims)

        def __call__(self, x, mask=None):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            # Simplified attention
            scores = (q @ k.transpose(0, 2, 1)) / (q.shape[-1] ** 0.5)
            if mask is not None:
                scores = scores + mask
            attn = mx.softmax(scores, axis=-1)
            return self.o_proj(attn @ v)

    class SyntheticMLP(nn.Module):
        def __init__(self, dims, hidden_dims):
            super().__init__()
            self.gate_proj = nn.Linear(dims, hidden_dims)
            self.up_proj = nn.Linear(dims, hidden_dims)
            self.down_proj = nn.Linear(hidden_dims, dims)

        def __call__(self, x):
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class SyntheticTransformerBlock(nn.Module):
        def __init__(self, dims, hidden_dims):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(dims)
            self.self_attn = SyntheticAttention(dims)
            self.post_attention_layernorm = nn.LayerNorm(dims)
            self.mlp = SyntheticMLP(dims, hidden_dims)

        def __call__(self, x, mask=None):
            h = x + self.self_attn(self.input_layernorm(x), mask)
            return h + self.mlp(self.post_attention_layernorm(h))

    class SyntheticModel(nn.Module):
        def __init__(self, vocab_size=1000, dims=256, num_layers=4):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(vocab_size, dims)
            self.model.layers = [
                SyntheticTransformerBlock(dims, hidden_dims=512)
                for _ in range(num_layers)
            ]
            self.model.lm_head = nn.Linear(dims, vocab_size, bias=False)

        def __call__(self, input_ids):
            x = self.model.embed_tokens(input_ids)
            for layer in self.model.layers:
                x = layer(x)
            return self.model.lm_head(x)

        def create_attention_mask(self, input_ids):
            # Create causal mask
            batch_size, seq_len = input_ids.shape
            mask = mx.triu(
                mx.full((seq_len, seq_len), -1e9),
                k=1
            )
            return mask

    return SyntheticModel()


@pytest.fixture
def synthetic_calibration_data():
    """Generate synthetic calibration data for testing."""
    # Random token IDs
    num_samples = 32
    sequence_length = 128
    vocab_size = 1000

    return mx.random.randint(0, vocab_size, (num_samples, sequence_length))


@pytest.fixture
def small_calibration_data():
    """Generate small calibration dataset for quick tests."""
    num_samples = 8
    sequence_length = 64
    vocab_size = 1000

    return mx.random.randint(0, vocab_size, (num_samples, sequence_length))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_download: mark test as requiring model downloads"
    )
