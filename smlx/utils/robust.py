#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Robust Inference Wrapper for production-ready model execution.

Provides automatic error handling, retry logic, and graceful degradation
for model inference to prevent crashes in production environments.

Features:
- Automatic retry with exponential backoff
- OOM (Out of Memory) detection and recovery
- Parameter adjustment on failure (max_tokens, batch_size)
- Partial result returns on failure
- Comprehensive error logging
- Integration with MemoryWatchdog

Example:
    >>> from smlx.utils.robust import robust_generate
    >>> from smlx.models.SmolLM2_135M import load
    >>>
    >>> model, tokenizer = load()
    >>> result = robust_generate(
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     prompt="Hello world",
    ...     max_tokens=500,
    ...     max_retries=3
    ... )
    >>> if result.success:
    ...     print(result.text)
    ... else:
    ...     print(f"Error: {result.error_message}")
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from smlx.utils.memory import MemoryMonitor, smart_cleanup

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """
    Result of robust inference attempt.

    Attributes:
        success: Whether inference succeeded
        text: Generated text (if successful)
        tokens: Generated tokens (if successful)
        partial_text: Partial text if inference failed mid-generation
        error_message: Error message if failed
        error_type: Type of error that occurred
        attempts: Number of attempts made
        final_params: Final parameters used
        metadata: Additional metadata (timing, memory, etc.)
    """

    success: bool
    text: Optional[str] = None
    tokens: Optional[list[int]] = None
    partial_text: Optional[str] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    attempts: int = 1
    final_params: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RobustInferenceWrapper:
    """
    Wrapper for robust model inference with automatic error recovery.

    Handles common failure scenarios including OOM errors, Metal GPU errors,
    and computation graph issues. Automatically adjusts parameters and retries
    with graceful degradation.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_backoff: Initial backoff delay in seconds (default: 1.0)
        backoff_multiplier: Backoff multiplier for exponential backoff (default: 2.0)
        enable_watchdog: Enable memory watchdog during inference (default: True)
        auto_cleanup: Automatically cleanup memory on errors (default: True)
        return_partial: Return partial results on failure (default: True)

    Example:
        >>> wrapper = RobustInferenceWrapper(max_retries=3)
        >>> result = wrapper.generate(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     prompt="Hello",
        ...     max_tokens=500
        ... )
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        backoff_multiplier: float = 2.0,
        enable_watchdog: bool = True,
        auto_cleanup: bool = True,
        return_partial: bool = True,
    ):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_multiplier = backoff_multiplier
        self.enable_watchdog = enable_watchdog
        self.auto_cleanup = auto_cleanup
        self.return_partial = return_partial
        self.memory_monitor = MemoryMonitor()

    def _is_oom_error(self, error: Exception) -> bool:
        """
        Check if error is an Out of Memory error.

        Args:
            error: Exception to check

        Returns:
            True if OOM error
        """
        error_str = str(error).lower()
        oom_indicators = [
            'out of memory',
            'oom',
            'memory allocation failed',
            'metal',
            'cannot allocate',
            'resource exhausted',
        ]
        return any(indicator in error_str for indicator in oom_indicators)

    def _is_metal_error(self, error: Exception) -> bool:
        """
        Check if error is a Metal GPU error.

        Args:
            error: Exception to check

        Returns:
            True if Metal error
        """
        error_str = str(error).lower()
        metal_indicators = ['metal', 'gpu', 'device']
        return any(indicator in error_str for indicator in metal_indicators)

    def _adjust_parameters(
        self, params: dict[str, Any], attempt: int
    ) -> dict[str, Any]:
        """
        Adjust generation parameters for retry attempt.

        Progressively reduces resource-intensive parameters to avoid OOM.

        Args:
            params: Original parameters
            attempt: Current attempt number (1-indexed)

        Returns:
            Adjusted parameters
        """
        adjusted = params.copy()

        # Reduce max_tokens progressively
        if 'max_tokens' in adjusted:
            original_max_tokens = adjusted['max_tokens']
            reduction_factor = 0.5 ** attempt  # 50%, 25%, 12.5%, etc.
            adjusted['max_tokens'] = max(
                32, int(original_max_tokens * reduction_factor)
            )
            logger.info(
                f"Reducing max_tokens: {original_max_tokens} → "
                f"{adjusted['max_tokens']}"
            )

        # Reduce batch_size if present
        if 'batch_size' in adjusted:
            adjusted['batch_size'] = max(1, adjusted['batch_size'] // 2)
            logger.info(f"Reducing batch_size to {adjusted['batch_size']}")

        # Enable rotating cache if not already
        if 'use_rotating_cache' in adjusted and not adjusted['use_rotating_cache']:
            adjusted['use_rotating_cache'] = True
            logger.info("Enabling rotating KV cache")

        # Reduce KV cache size
        if 'max_kv_size' in adjusted:
            adjusted['max_kv_size'] = max(512, adjusted['max_kv_size'] // 2)
            logger.info(f"Reducing max_kv_size to {adjusted['max_kv_size']}")

        return adjusted

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> InferenceResult:
        """
        Robust text generation with automatic error recovery.

        Args:
            model: Model instance
            tokenizer: Tokenizer instance
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            InferenceResult with generation outcome

        Example:
            >>> wrapper = RobustInferenceWrapper()
            >>> result = wrapper.generate(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     prompt="Hello world",
            ...     max_tokens=500
            ... )
            >>> if result.success:
            ...     print(result.text)
        """
        params = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            **kwargs,
        }

        start_time = time.time()
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Inference attempt {attempt}/{self.max_retries}")

                # Adjust parameters on retry
                if attempt > 1:
                    params = self._adjust_parameters(params, attempt - 1)

                # Cleanup before attempt
                if self.auto_cleanup and attempt > 1:
                    freed_gb = smart_cleanup(aggressive=attempt > 2)
                    logger.info(f"Cleaned up {freed_gb:.2f}GB before retry")

                # Monitor memory
                mem_status = self.memory_monitor.check()
                if mem_status['status'] == 'critical':
                    logger.warning(
                        f"Memory critical before inference: "
                        f"{mem_status['total_gb']:.2f}GB"
                    )
                    smart_cleanup(aggressive=True)

                # Attempt generation
                from mlx_lm import generate as lm_generate
                from mlx_lm.sample_utils import make_sampler

                sampler = make_sampler(
                    temp=params.get("temperature", 0.0),
                    top_p=params.get("top_p", 1.0),
                )
                text = lm_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=params.get("max_tokens", 256),
                    sampler=sampler,
                )

                # Success!
                tokens = tokenizer.encode(text)
                elapsed_time = time.time() - start_time

                return InferenceResult(
                    success=True,
                    text=text,
                    tokens=tokens,
                    attempts=attempt,
                    final_params=params,
                    metadata={
                        'elapsed_time': elapsed_time,
                        'tokens_per_second': len(tokens) / elapsed_time
                        if elapsed_time > 0
                        else 0,
                    },
                )

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                logger.error(
                    f"Attempt {attempt} failed with {error_type}: {str(e)}"
                )

                # Check error type
                is_oom = self._is_oom_error(e)
                is_metal = self._is_metal_error(e)

                if is_oom or is_metal:
                    logger.warning("Memory/Metal error detected")
                    # Aggressive cleanup
                    freed_gb = smart_cleanup(aggressive=True)
                    logger.info(f"Freed {freed_gb:.2f}GB")

                    # More aggressive parameter reduction for OOM
                    if is_oom and attempt < self.max_retries:
                        params['max_tokens'] = max(32, params['max_tokens'] // 2)

                # Exponential backoff before retry
                if attempt < self.max_retries:
                    backoff_time = self.initial_backoff * (
                        self.backoff_multiplier ** (attempt - 1)
                    )
                    logger.info(f"Backing off {backoff_time:.1f}s before retry")
                    time.sleep(backoff_time)

        # All retries failed
        elapsed_time = time.time() - start_time

        return InferenceResult(
            success=False,
            partial_text=None,  # Could implement partial decoding
            error_message=str(last_error),
            error_type=type(last_error).__name__,
            attempts=self.max_retries,
            final_params=params,
            metadata={'elapsed_time': elapsed_time},
        )


def robust_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_retries: int = 3,
    **kwargs,
) -> InferenceResult:
    """
    Convenience function for robust text generation.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_retries: Maximum retry attempts
        **kwargs: Additional generation parameters

    Returns:
        InferenceResult with generation outcome

    Example:
        >>> from smlx.utils.robust import robust_generate
        >>> from smlx.models.SmolLM2_135M import load
        >>>
        >>> model, tokenizer = load()
        >>> result = robust_generate(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     prompt="Once upon a time",
        ...     max_tokens=200
        ... )
        >>> if result.success:
        ...     print(result.text)
        ... else:
        ...     print(f"Failed after {result.attempts} attempts")
    """
    wrapper = RobustInferenceWrapper(max_retries=max_retries)
    return wrapper.generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        **kwargs,
    )


__all__ = [
    'InferenceResult',
    'RobustInferenceWrapper',
    'robust_generate',
]
