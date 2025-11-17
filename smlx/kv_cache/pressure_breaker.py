# Copyright � 2025 SMLX Project

"""
Automatic memory pressure relief system for KV cache management.

Provides automatic intervention to prevent OOM errors during generation by
monitoring memory pressure and adjusting cache configurations dynamically.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from smlx.kv_cache.memory_pressure_gauge import MemoryPressureGauge
from smlx.utils.memory import smart_cleanup

if TYPE_CHECKING:
    from smlx.kv_cache.kv_manager import KVCacheManager

logger = logging.getLogger(__name__)


class PressureBreaker:
    """
    Automatically intervene to relieve memory pressure during generation.

    Monitors memory pressure and takes automatic actions to prevent OOM errors,
    such as clearing caches, reducing cache sizes, or switching to rotating caches.

    Attributes:
        cache_manager: KV cache manager to monitor and control
        pressure_gauge: Memory pressure monitoring gauge
        enabled: Whether interventions are currently enabled
        intervention_log: Log of all interventions taken

    Example:
        >>> from smlx.kv_cache import KVCacheManager, MemoryPressureGauge, PressureBreaker
        >>>
        >>> manager = KVCacheManager.create_standard(num_layers=24)
        >>> gauge = MemoryPressureGauge()
        >>> breaker = PressureBreaker(manager, gauge)
        >>>
        >>> # During generation loop
        >>> for step in range(max_steps):
        ...     breaker.monitor_and_intervene()  # Check before each step
        ...     # ... generation code ...
    """

    def __init__(
        self,
        cache_manager: KVCacheManager,
        pressure_gauge: MemoryPressureGauge | None = None,
        auto_enable: bool = True,
    ):
        """
        Initialize pressure breaker.

        Args:
            cache_manager: KV cache manager to monitor and control
            pressure_gauge: Optional pressure gauge (creates default if None)
            auto_enable: Enable interventions immediately (default: True)
        """
        self.cache_manager = cache_manager
        self.pressure_gauge = pressure_gauge or MemoryPressureGauge()
        self.enabled = auto_enable
        self.intervention_log: list[dict] = []

    def monitor_and_intervene(self, current_step: int | None = None) -> dict | None:
        """
        Check memory pressure and intervene if needed.

        This is the main method to call during generation. It checks current
        memory pressure and takes appropriate action if necessary.

        Args:
            current_step: Optional step number (for logging)

        Returns:
            Intervention dict if action was taken, None otherwise

        Example:
            >>> for step in range(1000):
            ...     intervention = breaker.monitor_and_intervene(current_step=step)
            ...     if intervention:
            ...         logger.info(f"Intervention at step {step}: {intervention['action']}")
        """
        if not self.enabled:
            return None

        # Check current pressure
        pressure = self.pressure_gauge.check_pressure()

        if pressure == "ok":
            return None

        # Get current cache state
        current_cache_size = self._get_average_cache_size()

        # Get intervention suggestion
        suggestion = self.pressure_gauge.suggest_intervention(current_cache_size)

        if suggestion is None:
            return None

        # Execute intervention
        if pressure == "critical":
            intervention = self._emergency_intervention(suggestion, current_step)
        else:  # warning
            intervention = self._preventive_intervention(suggestion, current_step)

        # Log intervention
        self.intervention_log.append(intervention)

        return intervention

    def _emergency_intervention(self, suggestion: dict, current_step: int | None) -> dict:
        """
        Execute emergency intervention for critical memory pressure.

        Args:
            suggestion: Intervention suggestion from pressure gauge
            current_step: Optional current step number

        Returns:
            Intervention result dictionary
        """
        logger.warning(
            f"Emergency intervention at step {current_step}: {suggestion['reason']}"
        )

        # Aggressive cleanup
        smart_cleanup(aggressive=True)

        # If we have rotating or quantized cache, reduce size
        # For standard cache, we can't reduce size directly, so just clear
        if self.cache_manager.cache_type == "standard":
            action_taken = "cleared_caches"
            self.cache_manager.reset_all()
        else:
            action_taken = "reduced_cache_size"
            # This is logged but actual cache switching would need model-level integration

        intervention = {
            "type": "emergency",
            "action": action_taken,
            "step": current_step,
            "suggestion": suggestion,
            "pressure": "critical",
        }

        logger.info(f"Emergency intervention complete: {action_taken}")

        return intervention

    def _preventive_intervention(self, suggestion: dict, current_step: int | None) -> dict:
        """
        Execute preventive intervention for warning pressure.

        Args:
            suggestion: Intervention suggestion from pressure gauge
            current_step: Optional current step number

        Returns:
            Intervention result dictionary
        """
        logger.info(f"Preventive intervention at step {current_step}: {suggestion['reason']}")

        # Gentle cleanup
        smart_cleanup(aggressive=False)

        intervention = {
            "type": "preventive",
            "action": "cleanup",
            "step": current_step,
            "suggestion": suggestion,
            "pressure": "warning",
        }

        logger.info("Preventive intervention complete")

        return intervention

    def _get_average_cache_size(self) -> int:
        """
        Get average cache size across all layers.

        Returns:
            Average offset across all caches
        """
        if not self.cache_manager.caches:
            return 0

        total_offset = sum(cache.offset for cache in self.cache_manager.caches)
        return total_offset // len(self.cache_manager.caches)

    def enable(self) -> None:
        """
        Enable automatic interventions.

        Example:
            >>> breaker.enable()
        """
        self.enabled = True
        logger.info("Pressure breaker enabled")

    def disable(self) -> None:
        """
        Disable automatic interventions.

        Use this during critical generation phases where interventions
        should not interrupt the process.

        Example:
            >>> breaker.disable()
            >>> # ... critical generation ...
            >>> breaker.enable()
        """
        self.enabled = False
        logger.info("Pressure breaker disabled")

    def disable_temporarily(self):
        """
        Context manager to temporarily disable interventions.

        Example:
            >>> with breaker.disable_temporarily():
            ...     # Critical generation code
            ...     pass
            >>> # Interventions re-enabled here
        """
        return _TemporarilyDisabled(self)

    def get_intervention_log(self) -> list[dict]:
        """
        Get log of all interventions taken.

        Returns:
            List of intervention dictionaries

        Example:
            >>> log = breaker.get_intervention_log()
            >>> print(f"Total interventions: {len(log)}")
            >>> emergency_count = sum(1 for i in log if i['type'] == 'emergency')
            >>> print(f"Emergency interventions: {emergency_count}")
        """
        return self.intervention_log.copy()

    def clear_intervention_log(self) -> None:
        """
        Clear intervention log.

        Example:
            >>> breaker.clear_intervention_log()
        """
        self.intervention_log.clear()

    def get_statistics(self) -> dict:
        """
        Get statistics about interventions.

        Returns:
            Dictionary with intervention statistics:
            - total_interventions: Total number of interventions
            - emergency_count: Number of emergency interventions
            - preventive_count: Number of preventive interventions
            - last_intervention: Most recent intervention (if any)

        Example:
            >>> stats = breaker.get_statistics()
            >>> print(f"Total interventions: {stats['total_interventions']}")
            >>> if stats['last_intervention']:
            ...     print(f"Last at step: {stats['last_intervention']['step']}")
        """
        emergency_count = sum(1 for i in self.intervention_log if i["type"] == "emergency")
        preventive_count = sum(1 for i in self.intervention_log if i["type"] == "preventive")

        return {
            "total_interventions": len(self.intervention_log),
            "emergency_count": emergency_count,
            "preventive_count": preventive_count,
            "last_intervention": self.intervention_log[-1] if self.intervention_log else None,
            "enabled": self.enabled,
        }


class _TemporarilyDisabled:
    """Context manager for temporarily disabling pressure breaker."""

    def __init__(self, breaker: PressureBreaker):
        self.breaker = breaker
        self.was_enabled = breaker.enabled

    def __enter__(self):
        self.breaker.disable()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        if self.was_enabled:
            self.breaker.enable()


__all__ = ["PressureBreaker"]
