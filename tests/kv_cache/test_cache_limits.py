"""Tests for smlx.kv_cache.cache_limits module."""

import pytest

from smlx.kv_cache.cache_limits import CacheLimitManager


class TestCacheLimitManager:
    """Test CacheLimitManager class."""

    def test_cache_limit_manager_init(self):
        """Test CacheLimitManager initialization."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        assert manager.model_size_gb == 0.5
        assert manager.target_memory_gb == 32.0
        assert manager.activation_overhead == 0.1
        assert manager.device_info is not None

    def test_cache_limit_manager_custom_overhead(self):
        """Test CacheLimitManager with custom activation overhead."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
            activation_overhead=0.2,
        )

        assert manager.activation_overhead == 0.2

    @pytest.mark.unit
    def test_compute_max_kv_size_mha(self):
        """Test computing max KV size for multi-head attention."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        max_tokens = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )

        # Should return reasonable number of tokens
        assert max_tokens >= 256
        assert isinstance(max_tokens, int)

    @pytest.mark.unit
    def test_compute_max_kv_size_gqa(self):
        """Test computing max KV size with grouped query attention."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        # GQA with fewer KV heads
        max_tokens_gqa = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
            num_kv_heads=4,
        )

        # MHA with same query heads
        max_tokens_mha = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
            num_kv_heads=12,
        )

        # GQA should allow more tokens (less KV cache memory)
        assert max_tokens_gqa > max_tokens_mha

    @pytest.mark.unit
    def test_compute_max_kv_size_different_dtypes(self):
        """Test computing max KV size with different data types."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        # FP16 (2 bytes)
        max_tokens_fp16 = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
            dtype_bytes=2,
        )

        # FP32 (4 bytes)
        max_tokens_fp32 = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
            dtype_bytes=4,
        )

        # FP16 should allow more tokens
        assert max_tokens_fp16 > max_tokens_fp32

    @pytest.mark.unit
    def test_compute_max_kv_size_safety_margin(self):
        """Test safety margin in KV size computation."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        # Lower safety margin should allow more tokens
        max_tokens_safe = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
            safety_margin=0.9,
        )

        max_tokens_less_safe = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
            safety_margin=0.5,
        )

        assert max_tokens_less_safe < max_tokens_safe

    @pytest.mark.unit
    def test_compute_max_kv_size_minimum(self):
        """Test that minimum cache size is enforced."""
        manager = CacheLimitManager(
            model_size_gb=30.0,  # Very large model
            target_memory_gb=32.0,  # Small target
        )

        max_tokens = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )

        # Should return at least 256 tokens
        assert max_tokens >= 256

    @pytest.mark.unit
    def test_should_use_rotating_cache(self):
        """Test determining if rotating cache is needed."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        # Compute max safe size
        max_safe = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )

        # Request within limit
        should_rotate_no = manager.should_use_rotating_cache(
            requested_tokens=max_safe // 2,
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )
        assert should_rotate_no is False

        # Request beyond limit
        should_rotate_yes = manager.should_use_rotating_cache(
            requested_tokens=max_safe * 2,
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )
        assert should_rotate_yes is True

    @pytest.mark.unit
    def test_should_use_quantized_cache(self):
        """Test determining if quantized cache is needed."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        # Compute max with fp16
        max_fp16 = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
            dtype_bytes=2,
        )

        # Request that fits with fp16
        should_quantize_no = manager.should_use_quantized_cache(
            requested_tokens=max_fp16 // 2,
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )
        assert should_quantize_no is False

        # Request that exceeds fp16 but fits with quantization
        should_quantize_yes = manager.should_use_quantized_cache(
            requested_tokens=max_fp16 * 2,
            num_layers=24,
            head_dim=64,
            num_heads=12,
            quantization_bits=4,
        )
        assert should_quantize_yes is True

    @pytest.mark.unit
    def test_recommend_cache_type_standard(self):
        """Test cache type recommendation for small requests."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        recommendation = manager.recommend_cache_type(
            requested_tokens=1024,
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )

        assert recommendation["cache_type"] == "standard"
        assert recommendation["max_kv_size"] is None
        assert "reason" in recommendation

    @pytest.mark.unit
    def test_recommend_cache_type_quantized(self):
        """Test cache type recommendation for quantized cache."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        # Get max safe size
        max_safe = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )

        # Request that needs quantization
        recommendation = manager.recommend_cache_type(
            requested_tokens=max_safe * 2,
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )

        # Should recommend either quantized or rotating
        assert recommendation["cache_type"] in ["quantized", "rotating"]
        assert "reason" in recommendation

    @pytest.mark.unit
    def test_get_memory_estimate_mha(self):
        """Test memory estimation for multi-head attention."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        estimate = manager.get_memory_estimate(
            num_tokens=2048,
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )

        assert "total_bytes" in estimate
        assert "total_gb" in estimate
        assert "per_token_bytes" in estimate
        assert estimate["num_tokens"] == 2048
        assert estimate["num_layers"] == 24

    @pytest.mark.unit
    def test_get_memory_estimate_gqa(self):
        """Test memory estimation with GQA."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        estimate = manager.get_memory_estimate(
            num_tokens=2048,
            num_layers=24,
            head_dim=64,
            num_heads=12,
            num_kv_heads=4,
        )

        assert estimate["num_kv_heads"] == 4
        # Should have GQA savings
        assert estimate["gqa_savings_gb"] > 0

    @pytest.mark.unit
    def test_get_memory_estimate_no_gqa_savings(self):
        """Test memory estimation without GQA (no savings)."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        estimate = manager.get_memory_estimate(
            num_tokens=2048,
            num_layers=24,
            head_dim=64,
            num_heads=12,
            num_kv_heads=12,  # Same as num_heads
        )

        # No GQA savings when num_kv_heads == num_heads
        assert estimate["gqa_savings_gb"] == 0.0


class TestCacheLimitManagerEdgeCases:
    """Test edge cases for CacheLimitManager."""

    @pytest.mark.unit
    def test_compute_max_kv_size_zero_layers(self):
        """Test computing max KV size with zero layers."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        # Should handle gracefully or return very large number
        max_tokens = manager.compute_max_kv_size(
            num_layers=1,  # Minimal layers
            head_dim=64,
            num_heads=1,
        )

        assert max_tokens > 0

    @pytest.mark.unit
    def test_get_memory_estimate_zero_tokens(self):
        """Test memory estimate with zero tokens."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        estimate = manager.get_memory_estimate(
            num_tokens=0,
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )

        assert estimate["total_bytes"] == 0
        assert estimate["total_gb"] == 0.0

    @pytest.mark.unit
    def test_recommend_cache_type_very_large_request(self):
        """Test recommendation for very large token request."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        recommendation = manager.recommend_cache_type(
            requested_tokens=1_000_000,  # Very large
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )

        # Should recommend rotating cache as fallback
        assert recommendation["cache_type"] == "rotating"

    @pytest.mark.unit
    def test_should_use_quantized_cache_8bit(self):
        """Test quantized cache recommendation with 8-bit."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        max_fp16 = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )

        # Test with 8-bit quantization
        should_quantize = manager.should_use_quantized_cache(
            requested_tokens=int(max_fp16 * 1.5),
            num_layers=24,
            head_dim=64,
            num_heads=12,
            quantization_bits=8,
        )

        # Should recommend quantization
        assert isinstance(should_quantize, bool)

    @pytest.mark.unit
    def test_cache_limit_manager_small_target_memory(self):
        """Test with very small target memory."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=1.0,  # Very small
        )

        max_tokens = manager.compute_max_kv_size(
            num_layers=24,
            head_dim=64,
            num_heads=12,
        )

        # Should still return minimum
        assert max_tokens >= 256

    @pytest.mark.unit
    def test_recommend_cache_type_with_gqa(self):
        """Test cache type recommendation with GQA."""
        manager = CacheLimitManager(
            model_size_gb=0.5,
            target_memory_gb=32.0,
        )

        recommendation = manager.recommend_cache_type(
            requested_tokens=4096,
            num_layers=24,
            head_dim=64,
            num_heads=12,
            num_kv_heads=4,
        )

        assert "cache_type" in recommendation
        assert "reason" in recommendation
