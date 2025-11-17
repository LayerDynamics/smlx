#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Graceful Degradation for memory-constrained inference.

Automatically adjusts generation parameters based on memory pressure
to allow inference to continue even when system is under stress.

Features:
- Real-time memory monitoring
- Automatic parameter adjustment (max_tokens, temperature, batch_size)
- Progressive degradation levels (normal → reduced → minimal)
- KV cache management under pressure
- Integration with MemoryMonitor

Example:
    >>> from smlx.utils.degradation import GracefulDegradation
    >>> from smlx.models.SmolLM2_135M import load, generate
    >>>
    >>> model, tokenizer = load()
    >>> degradation = GracefulDegradation()
    >>>
    >>> # Parameters auto-adjust based on memory
    >>> params = degradation.adjust_params(
    ...     max_tokens=500,
    ...     temperature=0.7,
    ...     batch_size=4
    ... )
    >>> result = generate(model, tokenizer, prompt="Hello", **params)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from smlx.config.memory import get_default_config
from smlx.utils.memory import MemoryMonitor, smart_cleanup


logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """
    Degradation levels for memory pressure.

    NORMAL: No degradation, full quality parameters
    REDUCED: Moderate degradation, some quality reduction
    MINIMAL: Heavy degradation, prioritize stability over quality
    CRITICAL: Emergency mode, bare minimum to function
    """

    NORMAL = "normal"
    REDUCED = "reduced"
    MINIMAL = "minimal"
    CRITICAL = "critical"


@dataclass
class DegradationParams:
    """
    Parameters for a specific degradation level.

    Attributes:
        level: Degradation level
        max_tokens_multiplier: Multiplier for max_tokens (0-1)
        temperature_override: Override temperature (None = no override)
        force_greedy: Force greedy sampling (temperature=0)
        disable_kv_cache: Disable KV cache
        max_kv_size: Maximum KV cache size (None = no limit)
        batch_size_divisor: Divide batch_size by this amount
        use_rotating_cache: Use rotating KV cache
        description: Human-readable description
    """

    level: DegradationLevel
    max_tokens_multiplier: float
    temperature_override: Optional[float] = None
    force_greedy: bool = False
    disable_kv_cache: bool = False
    max_kv_size: Optional[int] = None
    batch_size_divisor: int = 1
    use_rotating_cache: bool = False
    description: str = ""


# Predefined degradation levels
DEGRADATION_PRESETS = {
    DegradationLevel.NORMAL: DegradationParams(
        level=DegradationLevel.NORMAL,
        max_tokens_multiplier=1.0,
        temperature_override=None,
        force_greedy=False,
        disable_kv_cache=False,
        max_kv_size=None,
        batch_size_divisor=1,
        use_rotating_cache=False,
        description="Full quality, no degradation",
    ),
    DegradationLevel.REDUCED: DegradationParams(
        level=DegradationLevel.REDUCED,
        max_tokens_multiplier=0.75,
        temperature_override=None,
        force_greedy=False,
        disable_kv_cache=False,
        max_kv_size=2048,
        batch_size_divisor=2,
        use_rotating_cache=True,
        description="Moderate quality reduction: 75% tokens, rotating cache",
    ),
    DegradationLevel.MINIMAL: DegradationParams(
        level=DegradationLevel.MINIMAL,
        max_tokens_multiplier=0.5,
        temperature_override=0.3,
        force_greedy=False,
        disable_kv_cache=False,
        max_kv_size=1024,
        batch_size_divisor=4,
        use_rotating_cache=True,
        description="Heavy reduction: 50% tokens, greedy-ish sampling, small cache",
    ),
    DegradationLevel.CRITICAL: DegradationParams(
        level=DegradationLevel.CRITICAL,
        max_tokens_multiplier=0.25,
        temperature_override=None,
        force_greedy=True,
        disable_kv_cache=False,
        max_kv_size=512,
        batch_size_divisor=8,
        use_rotating_cache=True,
        description="Emergency mode: 25% tokens, greedy only, minimal cache",
    ),
}


class GracefulDegradation:
    """
    Graceful degradation manager for memory-constrained inference.

    Monitors memory usage and automatically adjusts generation parameters
    to maintain stability under memory pressure. Uses progressive degradation
    levels to balance quality and reliability.

    Args:
        monitor: MemoryMonitor instance (creates new if None)
        auto_cleanup: Automatically cleanup memory when degrading (default: True)
        min_level: Minimum allowed degradation level (default: CRITICAL)
        verbose: Log degradation decisions (default: True)

    Example:
        >>> degradation = GracefulDegradation(verbose=True)
        >>>
        >>> # Check current level
        >>> level = degradation.get_current_level()
        >>> print(f"Degradation level: {level.value}")
        >>>
        >>> # Adjust parameters
        >>> params = degradation.adjust_params(
        ...     max_tokens=500,
        ...     temperature=0.7,
        ...     batch_size=4
        ... )
    """

    def __init__(
        self,
        monitor: Optional[MemoryMonitor] = None,
        auto_cleanup: bool = True,
        min_level: DegradationLevel = DegradationLevel.CRITICAL,
        verbose: bool = True,
    ):
        self.monitor = monitor or MemoryMonitor()
        self.auto_cleanup = auto_cleanup
        self.min_level = min_level
        self.verbose = verbose
        self._current_level = DegradationLevel.NORMAL
        self._last_cleanup_level: Optional[DegradationLevel] = None

    def get_current_level(self) -> DegradationLevel:
        """
        Determine current degradation level based on memory status.

        Returns:
            Current degradation level

        Example:
            >>> degradation = GracefulDegradation()
            >>> level = degradation.get_current_level()
            >>> if level == DegradationLevel.CRITICAL:
            ...     print("Memory is critically low!")
        """
        status = self.monitor.check()
        utilization = status['utilization']

        # Determine level based on utilization
        if utilization >= 0.95:
            level = DegradationLevel.CRITICAL
        elif utilization >= 0.85:
            level = DegradationLevel.MINIMAL
        elif utilization >= 0.75:
            level = DegradationLevel.REDUCED
        else:
            level = DegradationLevel.NORMAL

        # Update current level
        if level != self._current_level:
            if self.verbose:
                logger.info(
                    f"Degradation level changed: {self._current_level.value} → "
                    f"{level.value} (memory: {utilization:.1%})"
                )
            self._current_level = level

            # Trigger cleanup if degrading and auto_cleanup enabled
            if (
                self.auto_cleanup
                and level != DegradationLevel.NORMAL
                and level != self._last_cleanup_level
            ):
                aggressive = level in (
                    DegradationLevel.MINIMAL,
                    DegradationLevel.CRITICAL,
                )
                freed_gb = smart_cleanup(aggressive=aggressive)
                if self.verbose:
                    logger.info(
                        f"Auto-cleanup freed {freed_gb:.2f}GB "
                        f"(aggressive={aggressive})"
                    )
                self._last_cleanup_level = level

        return level

    def get_degradation_params(
        self, level: Optional[DegradationLevel] = None
    ) -> DegradationParams:
        """
        Get degradation parameters for a specific level.

        Args:
            level: Degradation level (uses current if None)

        Returns:
            DegradationParams for the level

        Example:
            >>> degradation = GracefulDegradation()
            >>> params = degradation.get_degradation_params(DegradationLevel.MINIMAL)
            >>> print(params.description)
        """
        if level is None:
            level = self.get_current_level()
        return DEGRADATION_PRESETS[level]

    def adjust_params(
        self, max_tokens: int = 256, temperature: float = 0.7, **kwargs
    ) -> Dict[str, Any]:
        """
        Adjust generation parameters based on current memory pressure.

        Args:
            max_tokens: Requested max_tokens
            temperature: Requested temperature
            **kwargs: Additional parameters (batch_size, max_kv_size, etc.)

        Returns:
            Adjusted parameters dictionary

        Example:
            >>> degradation = GracefulDegradation()
            >>> params = degradation.adjust_params(
            ...     max_tokens=500,
            ...     temperature=0.7,
            ...     batch_size=4
            ... )
            >>> print(f"Adjusted max_tokens: {params['max_tokens']}")
        """
        level = self.get_current_level()
        deg_params = self.get_degradation_params(level)

        adjusted = kwargs.copy()

        # Adjust max_tokens
        original_max_tokens = max_tokens
        adjusted['max_tokens'] = max(
            32, int(max_tokens * deg_params.max_tokens_multiplier)
        )

        # Adjust temperature
        if deg_params.force_greedy:
            adjusted['temperature'] = 0.0
        elif deg_params.temperature_override is not None:
            adjusted['temperature'] = min(temperature, deg_params.temperature_override)
        else:
            adjusted['temperature'] = temperature

        # Adjust batch_size if present
        if 'batch_size' in kwargs:
            original_batch = kwargs['batch_size']
            adjusted['batch_size'] = max(
                1, original_batch // deg_params.batch_size_divisor
            )
        else:
            adjusted['batch_size'] = 1

        # Adjust KV cache settings
        if deg_params.disable_kv_cache:
            adjusted['cache'] = None
        else:
            adjusted['use_rotating_cache'] = deg_params.use_rotating_cache
            if deg_params.max_kv_size is not None:
                adjusted['max_kv_size'] = deg_params.max_kv_size

        # Log adjustments if verbose
        if self.verbose and level != DegradationLevel.NORMAL:
            changes = []
            if adjusted['max_tokens'] != original_max_tokens:
                changes.append(
                    f"max_tokens: {original_max_tokens} → {adjusted['max_tokens']}"
                )
            if adjusted['temperature'] != temperature:
                changes.append(
                    f"temperature: {temperature} → {adjusted['temperature']}"
                )
            if 'batch_size' in kwargs and adjusted['batch_size'] != kwargs[
                'batch_size'
            ]:
                changes.append(
                    f"batch_size: {kwargs['batch_size']} → {adjusted['batch_size']}"
                )

            if changes:
                logger.info(
                    f"Graceful degradation ({level.value}): {', '.join(changes)}"
                )

        return adjusted

    def reset(self) -> None:
        """
        Reset degradation state.

        Clears monitoring history and resets to NORMAL level.

        Example:
            >>> degradation = GracefulDegradation()
            >>> # ... perform inference ...
            >>> degradation.reset()  # Start fresh
        """
        self.monitor.reset()
        self._current_level = DegradationLevel.NORMAL
        self._last_cleanup_level = None
        if self.verbose:
            logger.info("Degradation state reset")


def with_graceful_degradation(
    max_tokens: int = 256,
    temperature: float = 0.7,
    monitor: Optional[MemoryMonitor] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to get degraded parameters.

    Args:
        max_tokens: Requested max_tokens
        temperature: Requested temperature
        monitor: Optional MemoryMonitor instance
        **kwargs: Additional parameters

    Returns:
        Adjusted parameters based on current memory

    Example:
        >>> from smlx.utils.degradation import with_graceful_degradation
        >>>
        >>> params = with_graceful_degradation(
        ...     max_tokens=500,
        ...     temperature=0.8,
        ...     batch_size=4
        ... )
        >>> # Use params in generation...
    """
    degradation = GracefulDegradation(monitor=monitor, verbose=True)
    return degradation.adjust_params(
        max_tokens=max_tokens, temperature=temperature, **kwargs
    )


__all__ = [
    'DegradationLevel',
    'DegradationParams',
    'DEGRADATION_PRESETS',
    'GracefulDegradation',
    'with_graceful_degradation',
]
