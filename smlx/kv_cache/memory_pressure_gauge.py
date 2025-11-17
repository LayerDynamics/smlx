# Copyright � 2025 SMLX Project

"""
Memory pressure monitoring for KV cache management.

Provides real-time memory pressure detection and intervention suggestions
specifically for KV cache operations during generation.
"""

from __future__ import annotations

from smlx.utils.memory import MemoryMonitor, get_active_memory_gb, get_cache_memory_gb


class MemoryPressureGauge:
    """
    Monitor memory pressure and suggest KV cache interventions.

    Extends basic memory monitoring with cache-specific pressure detection
    and intervention recommendations. Tracks memory usage trends and provides
    actionable suggestions for cache management.

    Attributes:
        warning_threshold: Utilization that triggers warnings (0.0-1.0)
        critical_threshold: Utilization that triggers critical alerts (0.0-1.0)
        monitor: Underlying MemoryMonitor
        interventions_triggered: List of interventions that have been triggered

    Example:
        >>> gauge = MemoryPressureGauge(
        ...     warning_threshold=0.8,
        ...     critical_threshold=0.9
        ... )
        >>>
        >>> # Check pressure during generation
        >>> pressure = gauge.check_pressure()
        >>> if pressure == 'critical':
        ...     suggestion = gauge.suggest_intervention(current_cache_size=2048)
        ...     print(suggestion)
    """

    def __init__(
        self,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.9,
        warning_gb: float | None = None,
        critical_gb: float | None = None,
    ):
        """
        Initialize memory pressure gauge.

        Args:
            warning_threshold: Warning threshold as fraction of max memory (default: 0.8)
            critical_threshold: Critical threshold as fraction of max memory (default: 0.9)
            warning_gb: Optional absolute warning threshold in GB (overrides threshold)
            critical_gb: Optional absolute critical threshold in GB (overrides threshold)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # Calculate absolute thresholds if not provided
        if warning_gb is None or critical_gb is None:
            from smlx.utils.memory import get_device_info

            device_info = get_device_info()
            max_gb = device_info.get("max_recommended_working_set_size_gb", 36.0)

            if warning_gb is None:
                warning_gb = max_gb * warning_threshold
            if critical_gb is None:
                critical_gb = max_gb * critical_threshold

        # At this point, both should be float
        assert warning_gb is not None and critical_gb is not None
        self.monitor = MemoryMonitor(warning_gb=warning_gb, critical_gb=critical_gb)
        self.interventions_triggered: list[dict] = []

    def check_pressure(self) -> str:
        """
        Check current memory pressure level.

        Returns:
            Pressure level: 'ok', 'warning', or 'critical'

        Example:
            >>> pressure = gauge.check_pressure()
            >>> if pressure != 'ok':
            ...     print(f"Memory pressure: {pressure}")
        """
        status = self.monitor.check()
        return status["status"]

    def get_detailed_status(self) -> dict:
        """
        Get detailed memory status with cache-specific information.

        Returns:
            Dictionary with detailed status:
            - status: 'ok', 'warning', or 'critical'
            - active_gb: Current active memory
            - cache_gb: Current cache memory
            - total_gb: Total memory in use
            - max_gb: Maximum available memory
            - utilization: Memory utilization (0-1)
            - recommendations: List of suggested actions
            - trend: Memory trend ('increasing', 'stable', 'decreasing')

        Example:
            >>> status = gauge.get_detailed_status()
            >>> print(f"Utilization: {status['utilization']:.1%}")
            >>> print(f"Trend: {status['trend']}")
        """
        status = self.monitor.check()
        trend = self.monitor.get_trend()
        status["trend"] = trend
        return status

    def suggest_intervention(
        self,
        current_cache_size: int,
    ) -> dict | None:
        """
        Suggest cache intervention based on current pressure.

        Args:
            current_cache_size: Current cache size in tokens

        Returns:
            Dictionary with intervention suggestion or None if no action needed:
            - action: Type of intervention ('reduce_cache', 'rotate_cache',
              'quantize_cache', 'clear_cache')
            - suggested_size: Recommended cache size (if applicable)
            - reason: Explanation of intervention
            - urgency: 'low', 'medium', or 'high'

        Example:
            >>> suggestion = gauge.suggest_intervention(current_cache_size=4096)
            >>> if suggestion:
            ...     print(f"Action: {suggestion['action']}")
            ...     print(f"Reason: {suggestion['reason']}")
        """
        pressure = self.check_pressure()

        if pressure == "ok":
            return None

        status = self.get_detailed_status()
        utilization = status["utilization"]

        # Critical pressure - aggressive interventions
        if pressure == "critical":
            # Record intervention
            intervention = {
                "action": "emergency_reduce",
                "suggested_size": max(current_cache_size // 4, 256),
                "reason": f"Critical memory pressure ({utilization:.1%} utilization). "
                "Emergency cache reduction required.",
                "urgency": "high",
                "utilization": utilization,
                "timestamp": status.get("timestamp"),
            }
            self.interventions_triggered.append(intervention)
            return intervention

        # Warning pressure - preventive interventions
        if pressure == "warning":
            trend = status.get("trend", "stable")

            if trend == "increasing":
                # Memory growing - switch to rotating cache
                intervention = {
                    "action": "rotate_cache",
                    "suggested_size": max(current_cache_size * 2 // 3, 512),
                    "reason": f"Warning memory pressure ({utilization:.1%}) with increasing trend. "
                    "Switch to rotating cache.",
                    "urgency": "medium",
                    "utilization": utilization,
                    "timestamp": status.get("timestamp"),
                }
            else:
                # Memory stable but high - reduce cache size
                intervention = {
                    "action": "reduce_cache",
                    "suggested_size": max(current_cache_size // 2, 512),
                    "reason": f"Warning memory pressure ({utilization:.1%}). "
                    "Reduce cache size preventively.",
                    "urgency": "low",
                    "utilization": utilization,
                    "timestamp": status.get("timestamp"),
                }

            self.interventions_triggered.append(intervention)
            return intervention

        return None

    def get_intervention_history(self) -> list[dict]:
        """
        Get history of triggered interventions.

        Returns:
            List of intervention dictionaries

        Example:
            >>> history = gauge.get_intervention_history()
            >>> print(f"Total interventions: {len(history)}")
        """
        return self.interventions_triggered.copy()

    def reset_intervention_history(self) -> None:
        """
        Clear intervention history.

        Example:
            >>> gauge.reset_intervention_history()
        """
        self.interventions_triggered.clear()

    def get_memory_trend(self, last_n: int = 10) -> str:
        """
        Get memory usage trend.

        Args:
            last_n: Number of recent checks to analyze (default: 10)

        Returns:
            Trend string: 'increasing', 'stable', or 'decreasing'

        Example:
            >>> trend = gauge.get_memory_trend()
            >>> if trend == 'increasing':
            ...     print("Memory leak or growing cache detected")
        """
        return self.monitor.get_trend(last_n=last_n)

    def estimate_cache_memory_gb(
        self,
        cache_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_bytes: int = 2,
    ) -> float:
        """
        Estimate memory usage for a given cache configuration.

        Args:
            cache_size: Number of tokens in cache
            num_layers: Number of transformer layers
            num_kv_heads: Number of key/value heads
            head_dim: Dimension of each attention head
            dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)

        Returns:
            Estimated memory in GB

        Example:
            >>> cache_mem = gauge.estimate_cache_memory_gb(
            ...     cache_size=2048,
            ...     num_layers=24,
            ...     num_kv_heads=4,
            ...     head_dim=64
            ... )
            >>> print(f"Cache memory: {cache_mem:.2f} GB")
        """
        # K and V: num_layers * num_kv_heads * head_dim * cache_size * 2 (K+V) * dtype_bytes
        bytes_total = num_layers * num_kv_heads * head_dim * cache_size * 2 * dtype_bytes
        return bytes_total / 1e9

    def predict_pressure_at_size(
        self,
        target_cache_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_bytes: int = 2,
    ) -> dict:
        """
        Predict memory pressure at a target cache size.

        Args:
            target_cache_size: Target number of tokens in cache
            num_layers: Number of transformer layers
            num_kv_heads: Number of key/value heads
            head_dim: Dimension of each attention head
            dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)

        Returns:
            Dictionary with prediction:
            - estimated_cache_gb: Estimated cache memory
            - estimated_total_gb: Estimated total memory
            - predicted_pressure: Predicted pressure level
            - safe: Whether target size is safe

        Example:
            >>> prediction = gauge.predict_pressure_at_size(
            ...     target_cache_size=4096,
            ...     num_layers=24,
            ...     num_kv_heads=4,
            ...     head_dim=64
            ... )
            >>> if not prediction['safe']:
            ...     print("Target size will cause memory pressure!")
        """
        # Estimate cache memory at target size
        estimated_cache_gb = self.estimate_cache_memory_gb(
            cache_size=target_cache_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
        )

        # Get current memory status
        current_active_gb = get_active_memory_gb()
        current_cache_gb = get_cache_memory_gb()

        # Predict total memory
        # Assume cache grows by the difference
        cache_delta = estimated_cache_gb - current_cache_gb
        estimated_total_gb = current_active_gb + current_cache_gb + max(cache_delta, 0)

        # Get device max
        from smlx.utils.memory import get_device_info

        device_info = get_device_info()
        max_gb = device_info.get("max_recommended_working_set_size_gb", 36.0)

        # Calculate predicted utilization
        predicted_utilization = estimated_total_gb / max_gb

        # Determine predicted pressure
        if predicted_utilization >= self.critical_threshold:
            predicted_pressure = "critical"
            safe = False
        elif predicted_utilization >= self.warning_threshold:
            predicted_pressure = "warning"
            safe = False
        else:
            predicted_pressure = "ok"
            safe = True

        return {
            "estimated_cache_gb": estimated_cache_gb,
            "estimated_total_gb": estimated_total_gb,
            "predicted_utilization": predicted_utilization,
            "predicted_pressure": predicted_pressure,
            "safe": safe,
            "max_gb": max_gb,
        }


__all__ = ["MemoryPressureGauge"]
