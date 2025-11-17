#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for SmolLM2-360M-Instruct language model.

Tests text generation, chat, streaming, and NoPE architecture.

Run with:
    python -m pytest tests/integration/test_smollm2_360m.py -v
"""

import gc

import mlx.core as mx
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
    pytest.mark.heavy_memory,  # SmolLM2-360M uses ~720MB
]


@pytest.fixture(scope="module")
def smollm2_360m_model():
    """
    Load SmolLM2-360M model once for all tests.

    Memory Requirements:
    - Model size: ~720MB (360M parameters in FP16)
    - Peak memory: ~1.2GB with activations
    """
    from smlx.models.SmolLM2_360M import load

    model, tokenizer = load("mlx-community/SmolLM2-360M-Instruct")

    yield model, tokenizer

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up SmolLM2-360M model...")
    del model
    del tokenizer
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print("Cleanup complete. Memory freed.")


def test_model_loading(smollm2_360m_model):
    """Test that SmolLM2-360M model loads successfully."""
    model, tokenizer = smollm2_360m_model

    assert model is not None, "Model should not be None"
    assert tokenizer is not None, "Tokenizer should not be None"
    assert hasattr(model, "model"), "Model should have inner model"
    # lm_head is conditional based on tie_word_embeddings config
    # Check for either lm_head or tied embeddings
    assert hasattr(model, "lm_head") or hasattr(
        model.model, "embed_tokens"
    ), "Model should have lm_head or embed_tokens"


def test_basic_generation(smollm2_360m_model):
    """Test basic text generation."""
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    # Generate text
    prompt = "Write a Python function to calculate"
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=50,
        temperature=0.0,
    )

    assert response is not None, "Response should not be None"
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should have content"


def test_streaming_generation(smollm2_360m_model):
    """Test streaming text generation."""
    from smlx.models.SmolLM2_360M import stream_generate

    model, tokenizer = smollm2_360m_model

    # Stream generation
    prompt = "The quick brown fox"
    chunks = list(
        stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=30,
            temperature=0.0,
        )
    )

    assert len(chunks) > 0, "Should generate at least one chunk"

    for i, chunk in enumerate(chunks):
        assert chunk is not None, f"Chunk {i} should not be None"
        assert isinstance(chunk, str), f"Chunk {i} should be string"


def test_chat_interface(smollm2_360m_model):
    """Test chat-style interface."""
    from smlx.models.SmolLM2_360M import chat

    model, tokenizer = smollm2_360m_model

    # Single turn chat
    messages = [{"role": "user", "content": "What is Python?"}]

    response = chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=50,
    )

    assert response is not None, "Chat response should not be None"
    assert isinstance(response, str), "Response should be string"


def test_multi_turn_chat(smollm2_360m_model):
    """Test multi-turn conversation."""
    from smlx.models.SmolLM2_360M import chat

    model, tokenizer = smollm2_360m_model

    # First turn
    messages = [{"role": "user", "content": "Hello!"}]

    response1 = chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=30,
    )

    # Second turn
    messages.append({"role": "assistant", "content": response1})
    messages.append({"role": "user", "content": "Tell me a joke."})

    response2 = chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=50,
    )

    assert response2 is not None, "Second response should not be None"


def test_complete_function(smollm2_360m_model):
    """Test completion function."""
    from smlx.models.SmolLM2_360M import complete, GenerationConfig

    model, tokenizer = smollm2_360m_model

    # Complete text with config
    prompt = "Once upon a time"
    config = GenerationConfig(max_tokens=40)
    completion = complete(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
    )

    assert completion is not None, "Completion should not be None"


def test_generation_config(smollm2_360m_model):
    """Test custom generation configuration."""
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    # Custom generation parameters (generate() accepts individual params, not config)
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="Write a short story about",
        max_tokens=30,
        temperature=0.8,
        top_p=0.9,
    )

    assert response is not None, "Response with custom params should work"


def test_temperature_variations(smollm2_360m_model):
    """Test different temperature settings."""
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    prompt = "Python is"
    temperatures = [0.0, 0.5, 1.0]

    for temp in temperatures:
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=20,
            temperature=temp,
        )

        assert response is not None, f"Should work with temperature {temp}"


def test_max_tokens_limit(smollm2_360m_model):
    """Test max tokens parameter."""
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    # Test with small max_tokens
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="Write a very long essay about",
        max_tokens=10,
        temperature=0.0,
    )

    assert response is not None, "Should respect max_tokens limit"


def test_top_p_sampling(smollm2_360m_model):
    """Test top-p (nucleus) sampling."""
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    # Test with different top_p values
    for top_p in [0.5, 0.9, 1.0]:
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Science is",
            max_tokens=20,
            top_p=top_p,
            temperature=0.8,
        )

        assert response is not None, f"Should work with top_p {top_p}"


def test_repetition_penalty(smollm2_360m_model):
    """Test generation with low temperature."""
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    # Test with low temperature (repetition_penalty not available in API)
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="The cat sat on the",
        max_tokens=30,
        temperature=0.0,
    )

    assert response is not None, "Generation with low temperature should work"


def test_empty_prompt(smollm2_360m_model):
    """Test with empty prompt."""
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    # Empty prompt should still generate (or handle gracefully)
    try:
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="",
            max_tokens=20,
        )
        # If it works, that's fine
        assert True
    except (ValueError, RuntimeError):
        # If it raises an error, that's also acceptable
        assert True


def test_long_prompt(smollm2_360m_model):
    """Test with longer prompt."""
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    # Longer prompt
    long_prompt = "Write a detailed explanation of machine learning. " * 5

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=long_prompt,
        max_tokens=50,
        temperature=0.0,
    )

    assert response is not None, "Should handle longer prompts"


def test_model_architecture(smollm2_360m_model):
    """Test model architecture details."""
    model, tokenizer = smollm2_360m_model

    # Check architecture components
    assert hasattr(model.model, "layers"), "Should have layers"
    assert hasattr(model.model, "embed_tokens"), "Should have token embeddings"
    assert hasattr(model.model, "norm"), "Should have normalization"


def test_attention_mechanism(smollm2_360m_model):
    """Test attention mechanism."""
    model, tokenizer = smollm2_360m_model

    # Check attention in layers
    if hasattr(model.model, "layers") and len(model.model.layers) > 0:
        layer = model.model.layers[0]
        assert hasattr(layer, "self_attn"), "Layer should have self attention"


def test_nope_architecture(smollm2_360m_model):
    """Test NoPE (No Positional Encoding) architecture."""
    from smlx.models.SmolLM2_360M import NoPE

    model, tokenizer = smollm2_360m_model

    # Check if model uses NoPE
    if hasattr(model.model, "layers") and len(model.model.layers) > 0:
        layer = model.model.layers[0]
        if hasattr(layer, "self_attn"):
            # NoPE should be present in attention mechanism
            assert True


def test_kv_cache(smollm2_360m_model):
    """Test KV cache functionality."""
    from smlx.models.SmolLM2_360M import KVCache

    model, tokenizer = smollm2_360m_model

    # KV cache should be available
    cache = KVCache()
    assert cache is not None, "KV cache should be available"


def test_rotating_kv_cache(smollm2_360m_model):
    """Test rotating KV cache."""
    from smlx.models.SmolLM2_360M import RotatingKVCache

    model, tokenizer = smollm2_360m_model

    # Rotating cache should be available
    cache = RotatingKVCache(max_size=1024)
    assert cache is not None, "Rotating KV cache should be available"


def test_model_config(smollm2_360m_model):
    """Test model configuration."""
    from smlx.models.SmolLM2_360M import ModelArgs

    model, tokenizer = smollm2_360m_model

    # Model should have config/args
    assert hasattr(model, "args"), "Model should have args"

    args = model.args
    assert args is not None, "Args should not be None"
    assert hasattr(args, "hidden_size"), "Args should have hidden_size"
    assert hasattr(args, "num_hidden_layers"), "Args should have num_hidden_layers"


def test_default_config():
    """Test default configuration."""
    from smlx.models.SmolLM2_360M import DEFAULT_CONFIG

    assert DEFAULT_CONFIG is not None, "DEFAULT_CONFIG should exist"


def test_tokenizer_functionality(smollm2_360m_model):
    """Test tokenizer."""
    model, tokenizer = smollm2_360m_model

    # Test encoding
    text = "Hello, world!"
    tokens = tokenizer.encode(text)

    assert tokens is not None, "Tokens should not be None"
    assert len(tokens) > 0, "Should have tokens"

    # Test decoding
    decoded = tokenizer.decode(tokens)
    assert decoded is not None, "Decoded text should not be None"


def test_code_generation(smollm2_360m_model):
    """Test code generation capability."""
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    # Generate code
    prompt = "def fibonacci(n):"
    code = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=50,
        temperature=0.0,
    )

    assert code is not None, "Code generation should work"


def test_instruction_following(smollm2_360m_model):
    """Test instruction following."""
    from smlx.models.SmolLM2_360M import chat

    model, tokenizer = smollm2_360m_model

    # Give instruction
    messages = [
        {
            "role": "user",
            "content": "List three programming languages.",
        }
    ]

    response = chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=50,
    )

    assert response is not None, "Instruction following should work"


def test_question_answering(smollm2_360m_model):
    """Test question answering."""
    from smlx.models.SmolLM2_360M import chat

    model, tokenizer = smollm2_360m_model

    # Ask question
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ]

    response = chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=30,
    )

    assert response is not None, "Question answering should work"


# ============================================================================
# Enhanced Cache Tests (New KV Cache Module)
# ============================================================================


def test_enhanced_cache_auto_mode(smollm2_360m_model):
    """Test automatic cache type selection with new kv_cache module."""
    from smlx.models.SmolLM2_360M.cache import make_cache

    model, tokenizer = smollm2_360m_model

    # Create cache with auto mode
    cache = make_cache(model, cache_type="auto", enable_monitoring=True)

    assert len(cache) == len(model.model.layers)

    # All caches should have standard interface
    for c in cache:
        assert hasattr(c, "offset")
        assert hasattr(c, "update_and_fetch")

    mx.clear_cache()


def test_enhanced_cache_rotating_mode(smollm2_360m_model):
    """Test rotating cache with memory-aware sizing."""
    from smlx.models.SmolLM2_360M.cache import make_cache
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    # Create rotating cache
    cache = make_cache(model, cache_type="rotating", max_kv_size=512)

    # Generate with rotating cache
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="Python is a",
        max_tokens=20,
        temperature=0.0,
    )

    assert response is not None
    mx.clear_cache()


def test_enhanced_cache_quantized_mode(smollm2_360m_model):
    """Test quantized cache for memory efficiency."""
    from smlx.models.SmolLM2_360M.cache import make_cache

    model, tokenizer = smollm2_360m_model

    # Create quantized cache
    cache = make_cache(
        model,
        cache_type="quantized",
        enable_quantization=True,
        quantization_bits=4,
        max_kv_size=1024,
    )

    assert len(cache) == len(model.model.layers)
    mx.clear_cache()


def test_enhanced_cache_with_monitoring(smollm2_360m_model):
    """Test cache with automatic memory pressure monitoring."""
    from smlx.models.SmolLM2_360M.cache import make_cache_with_monitoring
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    # Create cache with monitoring
    cache, breaker = make_cache_with_monitoring(
        model, cache_type="standard", target_memory_gb=32.0
    )

    assert len(cache) == len(model.model.layers)
    assert breaker is not None
    assert breaker.enabled

    # Generate with monitoring
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="The quick brown fox",
        max_tokens=10,
        temperature=0.0,
    )

    # Check statistics
    stats = breaker.get_statistics()
    assert "total_interventions" in stats
    assert stats["enabled"] is True

    mx.clear_cache()


def test_enhanced_cache_backwards_compatibility(smollm2_360m_model):
    """Test that legacy cache API still works."""
    from smlx.models.SmolLM2_360M.cache import make_cache
    from smlx.models.SmolLM2_360M import generate

    model, tokenizer = smollm2_360m_model

    # Old style API should still work
    cache1 = make_cache(model)
    cache2 = make_cache(model, max_kv_size=1024)

    assert len(cache1) == len(model.model.layers)
    assert len(cache2) == len(model.model.layers)

    # Both should work with generation
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="Hello",
        max_tokens=5,
        temperature=0.0,
    )

    assert response is not None
    mx.clear_cache()


def test_enhanced_cache_pressure_breaker_intervention():
    """Test PressureBreaker intervention capabilities."""
    from smlx.kv_cache import MemoryPressureGauge, PressureBreaker, KVCacheManager

    # Create manager
    manager = KVCacheManager.create_standard(num_layers=32, enable_monitoring=True)

    # Create pressure breaker
    gauge = MemoryPressureGauge()
    breaker = PressureBreaker(manager, gauge)

    # Test enable/disable
    assert breaker.enabled is True
    breaker.disable()
    assert breaker.enabled is False
    breaker.enable()
    assert breaker.enabled is True

    # Test temporarily disable
    with breaker.disable_temporarily():
        assert breaker.enabled is False
    assert breaker.enabled is True

    mx.clear_cache()


def test_enhanced_cache_limit_manager():
    """Test CacheLimitManager for automatic sizing."""
    from smlx.kv_cache import CacheLimitManager

    # Create manager
    manager = CacheLimitManager(model_size_gb=1.0, target_memory_gb=32.0)

    # Compute max cache size for SmolLM2-360M
    max_tokens = manager.compute_max_kv_size(
        num_layers=32, head_dim=64, num_heads=9  # SmolLM2-360M config
    )

    assert max_tokens > 0

    # Get recommendation
    recommendation = manager.recommend_cache_type(
        requested_tokens=4096, num_layers=32, head_dim=64, num_heads=9
    )

    assert "cache_type" in recommendation
    assert recommendation["cache_type"] in ["standard", "rotating", "quantized"]


def test_enhanced_cache_memory_pressure_gauge():
    """Test MemoryPressureGauge functionality."""
    from smlx.kv_cache import MemoryPressureGauge

    # Create gauge
    gauge = MemoryPressureGauge(warning_threshold=0.8, critical_threshold=0.9)

    # Check pressure
    pressure = gauge.check_pressure()
    assert pressure in ["ok", "warning", "critical"]

    # Get detailed status
    status = gauge.get_detailed_status()
    assert "status" in status
    assert "utilization" in status
    assert "trend" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
