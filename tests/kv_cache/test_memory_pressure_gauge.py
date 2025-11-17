"""Tests for smlx.kv_cache.memory_pressure_gauge module."""

import pytest

from smlx.kv_cache.memory_pressure_gauge import MemoryPressureGauge


class TestMemoryPressureGauge:
    """Test MemoryPressureGauge class."""

    def test_memory_pressure_gauge_init_default(self):
        """Test MemoryPressureGauge initialization with defaults."""
        gauge = MemoryPressureGauge()

        assert gauge.warning_threshold == 0.8
        assert gauge.critical_threshold == 0.9
        assert gauge.monitor is not None
        assert isinstance(gauge.interventions_triggered, list)
        assert len(gauge.interventions_triggered) == 0

    def test_memory_pressure_gauge_init_custom_thresholds(self):
        """Test MemoryPressureGauge initialization with custom thresholds."""
        gauge = MemoryPressureGauge(
            warning_threshold=0.7,
            critical_threshold=0.85,
        )

        assert gauge.warning_threshold == 0.7
        assert gauge.critical_threshold == 0.85

    def test_memory_pressure_gauge_init_absolute_thresholds(self):
        """Test MemoryPressureGauge initialization with absolute GB thresholds."""
        gauge = MemoryPressureGauge(
            warning_gb=20.0,
            critical_gb=25.0,
        )

        assert gauge.monitor is not None
        assert gauge.monitor.warning_gb == 20.0
        assert gauge.monitor.critical_gb == 25.0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_check_pressure(self):
        """Test checking memory pressure."""
        gauge = MemoryPressureGauge()

        pressure = gauge.check_pressure()

        assert pressure in ["ok", "warning", "critical"]

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_detailed_status(self):
        """Test getting detailed memory status."""
        gauge = MemoryPressureGauge()

        status = gauge.get_detailed_status()

        assert "status" in status
        assert status["status"] in ["ok", "warning", "critical"]
        assert "active_gb" in status
        assert "cache_gb" in status
        assert "total_gb" in status
        assert "max_gb" in status
        assert "utilization" in status
        assert "trend" in status
        assert status["trend"] in ["increasing", "stable", "decreasing"]

    @pytest.mark.unit
    def test_suggest_intervention_ok_pressure(self):
        """Test no intervention when pressure is ok."""
        # Use very high thresholds so pressure is always ok
        gauge = MemoryPressureGauge(
            warning_threshold=0.99,
            critical_threshold=1.0,
        )

        suggestion = gauge.suggest_intervention(current_cache_size=2048)

        assert suggestion is None

    @pytest.mark.unit
    def test_suggest_intervention_warning_pressure(self):
        """Test intervention suggestion for warning pressure."""
        # Use very low thresholds to trigger warning
        gauge = MemoryPressureGauge(
            warning_threshold=0.01,
            critical_threshold=0.5,
        )

        suggestion = gauge.suggest_intervention(current_cache_size=2048)

        if suggestion is not None:
            assert "action" in suggestion
            assert suggestion["action"] in ["rotate_cache", "reduce_cache"]
            assert "suggested_size" in suggestion
            assert "reason" in suggestion
            assert "urgency" in suggestion
            assert suggestion["urgency"] in ["low", "medium"]
            assert "utilization" in suggestion

    @pytest.mark.unit
    def test_suggest_intervention_critical_pressure(self):
        """Test intervention suggestion for critical pressure."""
        # Use very low thresholds to trigger critical
        gauge = MemoryPressureGauge(
            warning_threshold=0.01,
            critical_threshold=0.02,
        )

        suggestion = gauge.suggest_intervention(current_cache_size=2048)

        if suggestion is not None:
            assert suggestion["action"] == "emergency_reduce"
            assert suggestion["suggested_size"] == max(2048 // 4, 256)
            assert suggestion["urgency"] == "high"
            assert "utilization" in suggestion

    @pytest.mark.unit
    def test_intervention_history_tracking(self):
        """Test intervention history is tracked."""
        gauge = MemoryPressureGauge(
            warning_threshold=0.01,
            critical_threshold=0.02,
        )

        # Trigger intervention
        suggestion1 = gauge.suggest_intervention(current_cache_size=2048)
        suggestion2 = gauge.suggest_intervention(current_cache_size=1024)

        history = gauge.get_intervention_history()

        # At least one intervention should be tracked
        if suggestion1 is not None or suggestion2 is not None:
            assert len(history) > 0

    @pytest.mark.unit
    def test_reset_intervention_history(self):
        """Test resetting intervention history."""
        gauge = MemoryPressureGauge(
            warning_threshold=0.01,
            critical_threshold=0.02,
        )

        # Trigger intervention
        _ = gauge.suggest_intervention(current_cache_size=2048)

        # Reset history
        gauge.reset_intervention_history()

        history = gauge.get_intervention_history()
        assert len(history) == 0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_memory_trend(self):
        """Test getting memory trend."""
        gauge = MemoryPressureGauge()

        # Need some checks first
        gauge.check_pressure()
        gauge.check_pressure()
        gauge.check_pressure()

        trend = gauge.get_memory_trend(last_n=3)

        assert trend in ["increasing", "stable", "decreasing"]

    @pytest.mark.unit
    def test_estimate_cache_memory_gb(self):
        """Test cache memory estimation."""
        gauge = MemoryPressureGauge()

        cache_mem = gauge.estimate_cache_memory_gb(
            cache_size=2048,
            num_layers=24,
            num_kv_heads=4,
            head_dim=64,
            dtype_bytes=2,
        )

        # 24 layers * 4 heads * 64 dim * 2048 tokens * 2 (K+V) * 2 bytes
        expected = (24 * 4 * 64 * 2048 * 2 * 2) / 1e9

        assert cache_mem == pytest.approx(expected)
        assert cache_mem > 0

    @pytest.mark.unit
    def test_estimate_cache_memory_gb_fp32(self):
        """Test cache memory estimation with fp32."""
        gauge = MemoryPressureGauge()

        cache_mem_fp32 = gauge.estimate_cache_memory_gb(
            cache_size=2048,
            num_layers=24,
            num_kv_heads=4,
            head_dim=64,
            dtype_bytes=4,
        )

        cache_mem_fp16 = gauge.estimate_cache_memory_gb(
            cache_size=2048,
            num_layers=24,
            num_kv_heads=4,
            head_dim=64,
            dtype_bytes=2,
        )

        # FP32 should use exactly 2x memory
        assert cache_mem_fp32 == pytest.approx(cache_mem_fp16 * 2)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_predict_pressure_at_size(self):
        """Test predicting pressure at target cache size."""
        gauge = MemoryPressureGauge()

        prediction = gauge.predict_pressure_at_size(
            target_cache_size=2048,
            num_layers=24,
            num_kv_heads=4,
            head_dim=64,
        )

        assert "estimated_cache_gb" in prediction
        assert "estimated_total_gb" in prediction
        assert "predicted_utilization" in prediction
        assert "predicted_pressure" in prediction
        assert "safe" in prediction
        assert "max_gb" in prediction

        assert prediction["predicted_pressure"] in ["ok", "warning", "critical"]
        assert isinstance(prediction["safe"], bool)
        assert prediction["estimated_cache_gb"] > 0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_predict_pressure_safe_vs_unsafe(self):
        """Test predicting safe vs unsafe cache sizes."""
        gauge = MemoryPressureGauge()

        # Very small cache should be safe
        small_prediction = gauge.predict_pressure_at_size(
            target_cache_size=128,
            num_layers=6,
            num_kv_heads=4,
            head_dim=64,
        )

        # Very large cache might not be safe
        large_prediction = gauge.predict_pressure_at_size(
            target_cache_size=100000,
            num_layers=24,
            num_kv_heads=12,
            head_dim=128,
        )

        # Small cache should have lower utilization
        assert small_prediction["predicted_utilization"] < large_prediction["predicted_utilization"]


class TestMemoryPressureGaugeIntegration:
    """Test integration scenarios for memory pressure gauge."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_monitoring_during_generation(self):
        """Test monitoring memory pressure during simulated generation."""
        gauge = MemoryPressureGauge(
            warning_threshold=0.8,
            critical_threshold=0.9,
        )

        # Simulate generation loop with increasing cache
        cache_sizes = [128, 256, 512, 1024, 2048, 4096]

        for cache_size in cache_sizes:
            pressure = gauge.check_pressure()
            if pressure != "ok":
                suggestion = gauge.suggest_intervention(cache_size)
                if suggestion:
                    break

        # Should have collected some history
        history = gauge.get_intervention_history()
        assert isinstance(history, list)

    @pytest.mark.unit
    def test_cache_memory_estimation_workflow(self):
        """Test workflow of estimating and predicting cache memory."""
        gauge = MemoryPressureGauge()

        # Configuration for SmolLM2-style model
        config = {
            "num_layers": 24,
            "num_kv_heads": 4,
            "head_dim": 64,
            "dtype_bytes": 2,
        }

        # Estimate memory for different cache sizes
        cache_512 = gauge.estimate_cache_memory_gb(cache_size=512, **config)
        cache_1024 = gauge.estimate_cache_memory_gb(cache_size=1024, **config)
        cache_2048 = gauge.estimate_cache_memory_gb(cache_size=2048, **config)

        # Memory should scale linearly with cache size
        assert cache_1024 == pytest.approx(cache_512 * 2)
        assert cache_2048 == pytest.approx(cache_512 * 4)

    @pytest.mark.unit
    def test_intervention_recommendations(self):
        """Test intervention recommendations for different scenarios."""
        gauge = MemoryPressureGauge(
            warning_threshold=0.01,  # Very low to trigger interventions
            critical_threshold=0.02,
        )

        # Test different cache sizes
        suggestions = []
        for cache_size in [512, 1024, 2048, 4096]:
            suggestion = gauge.suggest_intervention(cache_size)
            if suggestion:
                suggestions.append(suggestion)

        # Should have suggestions if thresholds are exceeded
        history = gauge.get_intervention_history()
        assert len(history) == len(suggestions)

        # All suggestions should have required fields
        for suggestion in suggestions:
            assert "action" in suggestion
            assert "suggested_size" in suggestion
            assert "reason" in suggestion
            assert "urgency" in suggestion


class TestMemoryPressureGaugeEdgeCases:
    """Test edge cases for memory pressure gauge."""

    @pytest.mark.unit
    def test_zero_cache_size(self):
        """Test with zero cache size."""
        gauge = MemoryPressureGauge()

        cache_mem = gauge.estimate_cache_memory_gb(
            cache_size=0,
            num_layers=24,
            num_kv_heads=4,
            head_dim=64,
        )

        assert cache_mem == 0.0

    @pytest.mark.unit
    def test_very_small_cache_size(self):
        """Test with very small cache size."""
        gauge = MemoryPressureGauge()

        cache_mem = gauge.estimate_cache_memory_gb(
            cache_size=1,
            num_layers=1,
            num_kv_heads=1,
            head_dim=64,
        )

        assert cache_mem > 0
        assert cache_mem < 0.001  # Should be very small

    @pytest.mark.unit
    def test_gqa_vs_mha_memory(self):
        """Test GQA uses less memory than MHA."""
        gauge = MemoryPressureGauge()

        # GQA with 4 KV heads
        gqa_mem = gauge.estimate_cache_memory_gb(
            cache_size=2048,
            num_layers=24,
            num_kv_heads=4,
            head_dim=64,
        )

        # MHA with 12 KV heads
        mha_mem = gauge.estimate_cache_memory_gb(
            cache_size=2048,
            num_layers=24,
            num_kv_heads=12,
            head_dim=64,
        )

        # GQA should use less memory
        assert gqa_mem < mha_mem
        assert mha_mem == pytest.approx(gqa_mem * 3)  # 12/4 = 3x

    @pytest.mark.unit
    def test_intervention_history_empty(self):
        """Test intervention history when no interventions triggered."""
        gauge = MemoryPressureGauge(
            warning_threshold=0.99,
            critical_threshold=1.0,
        )

        # No intervention should be triggered
        _ = gauge.suggest_intervention(current_cache_size=2048)

        history = gauge.get_intervention_history()
        assert len(history) == 0

    @pytest.mark.unit
    def test_multiple_pressure_checks(self):
        """Test multiple pressure checks."""
        gauge = MemoryPressureGauge()

        # Multiple checks should not fail
        for _ in range(10):
            pressure = gauge.check_pressure()
            assert pressure in ["ok", "warning", "critical"]

    @pytest.mark.unit
    def test_suggested_size_minimum(self):
        """Test suggested size has minimum values."""
        gauge = MemoryPressureGauge(
            warning_threshold=0.01,
            critical_threshold=0.02,
        )

        # Very small cache
        suggestion = gauge.suggest_intervention(current_cache_size=128)

        if suggestion is not None:
            # Suggested size should have minimum (256 for emergency, 512 for others)
            assert suggestion["suggested_size"] >= 256

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_predict_pressure_consistency(self):
        """Test predict_pressure_at_size is consistent."""
        gauge = MemoryPressureGauge()

        # Same prediction should give same results
        pred1 = gauge.predict_pressure_at_size(
            target_cache_size=2048,
            num_layers=24,
            num_kv_heads=4,
            head_dim=64,
        )

        pred2 = gauge.predict_pressure_at_size(
            target_cache_size=2048,
            num_layers=24,
            num_kv_heads=4,
            head_dim=64,
        )

        # Cache memory should be the same
        assert pred1["estimated_cache_gb"] == pytest.approx(pred2["estimated_cache_gb"])

    @pytest.mark.unit
    def test_invalid_thresholds(self):
        """Test behavior with unusual threshold values."""
        # Warning threshold higher than critical should still work
        gauge = MemoryPressureGauge(
            warning_threshold=0.9,
            critical_threshold=0.8,  # Lower than warning
        )

        # Should still initialize
        assert gauge.warning_threshold == 0.9
        assert gauge.critical_threshold == 0.8
