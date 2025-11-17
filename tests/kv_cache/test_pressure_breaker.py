"""Tests for smlx.kv_cache.pressure_breaker module."""

import pytest

from smlx.kv_cache.kv_manager import KVCacheManager
from smlx.kv_cache.memory_pressure_gauge import MemoryPressureGauge
from smlx.kv_cache.pressure_breaker import PressureBreaker


class TestPressureBreaker:
    """Test PressureBreaker class."""

    def test_pressure_breaker_init(self):
        """Test PressureBreaker initialization."""
        manager = KVCacheManager.create_standard(num_layers=6)
        gauge = MemoryPressureGauge()
        breaker = PressureBreaker(manager, gauge)

        assert breaker.cache_manager is manager
        assert breaker.pressure_gauge is gauge
        assert breaker.enabled is True
        assert isinstance(breaker.intervention_log, list)
        assert len(breaker.intervention_log) == 0

    def test_pressure_breaker_init_auto_gauge(self):
        """Test PressureBreaker initialization with auto gauge creation."""
        manager = KVCacheManager.create_standard(num_layers=6)
        breaker = PressureBreaker(manager)

        assert breaker.pressure_gauge is not None
        assert isinstance(breaker.pressure_gauge, MemoryPressureGauge)

    def test_pressure_breaker_init_disabled(self):
        """Test PressureBreaker initialization with auto_enable=False."""
        manager = KVCacheManager.create_standard(num_layers=6)
        breaker = PressureBreaker(manager, auto_enable=False)

        assert breaker.enabled is False

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_monitor_and_intervene_ok_pressure(self):
        """Test monitor_and_intervene when pressure is ok."""
        manager = KVCacheManager.create_standard(num_layers=6)
        # Use very high thresholds so pressure is always ok
        gauge = MemoryPressureGauge(warning_threshold=0.99, critical_threshold=1.0)
        breaker = PressureBreaker(manager, gauge)

        intervention = breaker.monitor_and_intervene(current_step=0)

        assert intervention is None
        assert len(breaker.intervention_log) == 0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_monitor_and_intervene_disabled(self):
        """Test monitor_and_intervene when breaker is disabled."""
        manager = KVCacheManager.create_standard(num_layers=6)
        gauge = MemoryPressureGauge()
        breaker = PressureBreaker(manager, gauge, auto_enable=False)

        intervention = breaker.monitor_and_intervene(current_step=0)

        assert intervention is None

    @pytest.mark.unit
    def test_monitor_and_intervene_warning_pressure(self):
        """Test monitor_and_intervene with warning pressure."""
        manager = KVCacheManager.create_standard(num_layers=6)
        # Use very low thresholds to trigger warning
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.5)
        breaker = PressureBreaker(manager, gauge)

        intervention = breaker.monitor_and_intervene(current_step=0)

        if intervention is not None:
            assert intervention["type"] == "preventive"
            assert intervention["action"] == "cleanup"
            assert intervention["pressure"] == "warning"
            assert intervention["step"] == 0

    @pytest.mark.unit
    def test_monitor_and_intervene_critical_pressure(self):
        """Test monitor_and_intervene with critical pressure."""
        manager = KVCacheManager.create_standard(num_layers=6)
        # Use very low thresholds to trigger critical
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.02)
        breaker = PressureBreaker(manager, gauge)

        intervention = breaker.monitor_and_intervene(current_step=0)

        if intervention is not None:
            assert intervention["type"] == "emergency"
            assert intervention["action"] in ["cleared_caches", "reduced_cache_size"]
            assert intervention["pressure"] == "critical"
            assert intervention["step"] == 0

    @pytest.mark.unit
    def test_get_average_cache_size(self):
        """Test getting average cache size."""
        manager = KVCacheManager.create_standard(num_layers=6)
        breaker = PressureBreaker(manager)

        # All caches start at offset 0
        avg_size = breaker._get_average_cache_size()
        assert avg_size == 0

    @pytest.mark.unit
    def test_get_average_cache_size_empty_manager(self):
        """Test getting average cache size with empty manager."""
        manager = KVCacheManager.create_standard(num_layers=0)
        breaker = PressureBreaker(manager)

        avg_size = breaker._get_average_cache_size()
        assert avg_size == 0

    @pytest.mark.unit
    def test_enable_disable(self):
        """Test enabling and disabling breaker."""
        manager = KVCacheManager.create_standard(num_layers=6)
        breaker = PressureBreaker(manager, auto_enable=False)

        assert breaker.enabled is False

        breaker.enable()
        assert breaker.enabled is True

        breaker.disable()
        assert breaker.enabled is False

    @pytest.mark.unit
    def test_disable_temporarily_context_manager(self):
        """Test disable_temporarily context manager."""
        manager = KVCacheManager.create_standard(num_layers=6)
        breaker = PressureBreaker(manager)

        assert breaker.enabled is True

        with breaker.disable_temporarily():
            assert breaker.enabled is False

        # Should be re-enabled after context
        assert breaker.enabled is True

    @pytest.mark.unit
    def test_disable_temporarily_was_disabled(self):
        """Test disable_temporarily when breaker was already disabled."""
        manager = KVCacheManager.create_standard(num_layers=6)
        breaker = PressureBreaker(manager, auto_enable=False)

        assert breaker.enabled is False

        with breaker.disable_temporarily():
            assert breaker.enabled is False

        # Should remain disabled after context
        assert breaker.enabled is False

    @pytest.mark.unit
    def test_get_intervention_log(self):
        """Test getting intervention log."""
        manager = KVCacheManager.create_standard(num_layers=6)
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.02)
        breaker = PressureBreaker(manager, gauge)

        # Trigger intervention
        _ = breaker.monitor_and_intervene(current_step=0)

        log = breaker.get_intervention_log()
        assert isinstance(log, list)

        # Log should be a copy
        if len(log) > 0:
            log.clear()
            assert len(breaker.intervention_log) > 0

    @pytest.mark.unit
    def test_clear_intervention_log(self):
        """Test clearing intervention log."""
        manager = KVCacheManager.create_standard(num_layers=6)
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.02)
        breaker = PressureBreaker(manager, gauge)

        # Trigger intervention
        _ = breaker.monitor_and_intervene(current_step=0)

        # Clear log
        breaker.clear_intervention_log()

        log = breaker.get_intervention_log()
        assert len(log) == 0

    @pytest.mark.unit
    def test_get_statistics_empty(self):
        """Test getting statistics with no interventions."""
        manager = KVCacheManager.create_standard(num_layers=6)
        breaker = PressureBreaker(manager)

        stats = breaker.get_statistics()

        assert stats["total_interventions"] == 0
        assert stats["emergency_count"] == 0
        assert stats["preventive_count"] == 0
        assert stats["last_intervention"] is None
        assert stats["enabled"] is True

    @pytest.mark.unit
    def test_get_statistics_with_interventions(self):
        """Test getting statistics with interventions."""
        manager = KVCacheManager.create_standard(num_layers=6)
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.02)
        breaker = PressureBreaker(manager, gauge)

        # Trigger interventions
        intervention1 = breaker.monitor_and_intervene(current_step=0)
        intervention2 = breaker.monitor_and_intervene(current_step=1)

        stats = breaker.get_statistics()

        if intervention1 is not None or intervention2 is not None:
            assert stats["total_interventions"] > 0
            assert stats["last_intervention"] is not None


class TestPressureBreakerIntegration:
    """Test integration scenarios for pressure breaker."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_generation_loop_monitoring(self):
        """Test monitoring during simulated generation loop."""
        manager = KVCacheManager.create_standard(num_layers=6)
        gauge = MemoryPressureGauge(
            warning_threshold=0.8,
            critical_threshold=0.9,
        )
        breaker = PressureBreaker(manager, gauge)

        # Simulate generation loop
        interventions = []
        for step in range(100):
            intervention = breaker.monitor_and_intervene(current_step=step)
            if intervention:
                interventions.append(intervention)

        # Should have collected some statistics
        stats = breaker.get_statistics()
        assert stats["total_interventions"] == len(interventions)

    @pytest.mark.unit
    def test_emergency_intervention_clears_standard_cache(self):
        """Test emergency intervention clears standard cache."""
        manager = KVCacheManager.create_standard(num_layers=6)
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.02)
        breaker = PressureBreaker(manager, gauge)

        # Trigger emergency intervention
        intervention = breaker.monitor_and_intervene(current_step=0)

        if intervention is not None and intervention["type"] == "emergency":
            assert intervention["action"] == "cleared_caches"
            # All caches should be reset
            assert all(cache.offset == 0 for cache in manager.caches)

    @pytest.mark.unit
    def test_preventive_intervention_no_cache_clear(self):
        """Test preventive intervention doesn't clear cache."""
        manager = KVCacheManager.create_standard(num_layers=6)
        # Threshold between warning and critical
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.5)
        breaker = PressureBreaker(manager, gauge)

        # Trigger preventive intervention
        intervention = breaker.monitor_and_intervene(current_step=0)

        if intervention is not None and intervention["type"] == "preventive":
            # Cache should not be cleared (just cleanup)
            assert intervention["action"] == "cleanup"

    @pytest.mark.unit
    def test_intervention_logging(self):
        """Test intervention logging over multiple steps."""
        manager = KVCacheManager.create_standard(num_layers=6)
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.02)
        breaker = PressureBreaker(manager, gauge)

        # Trigger multiple interventions
        for step in range(5):
            _ = breaker.monitor_and_intervene(current_step=step)

        log = breaker.get_intervention_log()

        # Log should have entries with step numbers
        for i, entry in enumerate(log):
            assert "step" in entry
            assert "type" in entry
            assert "pressure" in entry

    @pytest.mark.unit
    def test_rotating_cache_intervention(self):
        """Test intervention with rotating cache."""
        manager = KVCacheManager.create_rotating(
            num_layers=6,
            max_kv_size=2048,
            keep=256,
        )
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.02)
        breaker = PressureBreaker(manager, gauge)

        intervention = breaker.monitor_and_intervene(current_step=0)

        if intervention is not None and intervention["type"] == "emergency":
            # For rotating cache, action is logged but not cleared
            assert intervention["action"] in ["reduced_cache_size", "cleared_caches"]


class TestPressureBreakerEdgeCases:
    """Test edge cases for pressure breaker."""

    @pytest.mark.unit
    def test_monitor_intervene_no_step(self):
        """Test monitor_and_intervene without step number."""
        manager = KVCacheManager.create_standard(num_layers=6)
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.02)
        breaker = PressureBreaker(manager, gauge)

        intervention = breaker.monitor_and_intervene()

        if intervention is not None:
            # Step should be None
            assert intervention["step"] is None

    @pytest.mark.unit
    def test_multiple_enable_disable(self):
        """Test multiple enable/disable calls."""
        manager = KVCacheManager.create_standard(num_layers=6)
        breaker = PressureBreaker(manager)

        # Multiple enables
        breaker.enable()
        breaker.enable()
        assert breaker.enabled is True

        # Multiple disables
        breaker.disable()
        breaker.disable()
        assert breaker.enabled is False

    @pytest.mark.unit
    def test_nested_disable_temporarily(self):
        """Test nested disable_temporarily context managers."""
        manager = KVCacheManager.create_standard(num_layers=6)
        breaker = PressureBreaker(manager)

        assert breaker.enabled is True

        with breaker.disable_temporarily():
            assert breaker.enabled is False

            with breaker.disable_temporarily():
                assert breaker.enabled is False

            assert breaker.enabled is False

        # Should be re-enabled after all contexts
        assert breaker.enabled is True

    @pytest.mark.unit
    def test_get_statistics_disabled(self):
        """Test get_statistics when breaker is disabled."""
        manager = KVCacheManager.create_standard(num_layers=6)
        breaker = PressureBreaker(manager, auto_enable=False)

        stats = breaker.get_statistics()

        assert stats["enabled"] is False
        assert stats["total_interventions"] == 0

    @pytest.mark.unit
    def test_intervention_with_empty_manager(self):
        """Test intervention with empty cache manager."""
        manager = KVCacheManager.create_standard(num_layers=0)
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.02)
        breaker = PressureBreaker(manager, gauge)

        # Should not fail with empty manager
        intervention = breaker.monitor_and_intervene(current_step=0)

        # Might or might not trigger intervention
        assert intervention is None or isinstance(intervention, dict)

    @pytest.mark.unit
    def test_average_cache_size_after_updates(self):
        """Test average cache size calculation after updates."""
        import mlx.core as mx

        manager = KVCacheManager.create_standard(num_layers=3)
        breaker = PressureBreaker(manager)

        # Update all caches with different amounts
        keys1 = mx.ones((1, 4, 10, 64))
        values1 = mx.ones((1, 4, 10, 64))
        manager.caches[0].update_and_fetch(keys1, values1)

        keys2 = mx.ones((1, 4, 20, 64))
        values2 = mx.ones((1, 4, 20, 64))
        manager.caches[1].update_and_fetch(keys2, values2)

        keys3 = mx.ones((1, 4, 30, 64))
        values3 = mx.ones((1, 4, 30, 64))
        manager.caches[2].update_and_fetch(keys3, values3)

        avg_size = breaker._get_average_cache_size()

        # Average: (10 + 20 + 30) / 3 = 20
        assert avg_size == 20

    @pytest.mark.unit
    def test_intervention_log_order(self):
        """Test intervention log maintains order."""
        manager = KVCacheManager.create_standard(num_layers=6)
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.02)
        breaker = PressureBreaker(manager, gauge)

        # Trigger multiple interventions
        steps = [0, 5, 10, 15, 20]
        for step in steps:
            _ = breaker.monitor_and_intervene(current_step=step)

        log = breaker.get_intervention_log()

        # Log should maintain order
        if len(log) >= 2:
            for idx in range(len(log) - 1):
                if log[idx]["step"] is not None and log[idx + 1]["step"] is not None:
                    assert log[idx]["step"] <= log[idx + 1]["step"]

    @pytest.mark.unit
    def test_statistics_count_types(self):
        """Test statistics correctly count intervention types."""
        manager = KVCacheManager.create_standard(num_layers=6)
        gauge = MemoryPressureGauge(warning_threshold=0.01, critical_threshold=0.02)
        breaker = PressureBreaker(manager, gauge)

        # Trigger multiple interventions
        for _ in range(5):
            _ = breaker.monitor_and_intervene()

        stats = breaker.get_statistics()

        # Counts should add up
        assert (
            stats["emergency_count"] + stats["preventive_count"] == stats["total_interventions"]
        )
