"""
Curriculum learning for SMLX Gym.

This module provides curriculum learning utilities that gradually increase
task difficulty during training. This can improve learning efficiency and
final performance, especially for complex tasks.

Reference: SMLX_Gym.md, Section 4.4 (Advanced Features)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import gymnasium as gym


class CurriculumStage(Enum):
    """Curriculum learning stages."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class CurriculumMetrics:
    """
    Metrics for curriculum progression.

    Attributes:
        current_stage: Current curriculum stage
        success_rate: Success rate in current stage
        average_return: Average return in current stage
        episodes_in_stage: Episodes completed in current stage
        total_stage_changes: Total number of stage transitions
    """

    current_stage: CurriculumStage = CurriculumStage.EASY
    success_rate: float = 0.0
    average_return: float = 0.0
    episodes_in_stage: int = 0
    total_stage_changes: int = 0


class CurriculumScheduler(ABC):
    """
    Abstract base class for curriculum schedulers.

    Curriculum schedulers determine when to progress to the next difficulty level
    based on agent performance metrics.

    Example:
        ```python
        from smlx.gym.curriculum import CurriculumScheduler, CurriculumStage

        class MyScheduler(CurriculumScheduler):
            def should_progress(self, metrics):
                return metrics.success_rate > 0.8

            def get_stage_config(self, stage):
                return {"difficulty": stage.value}

        scheduler = MyScheduler()
        ```
    """

    def __init__(self):
        """Initialize curriculum scheduler."""
        self.metrics = CurriculumMetrics()

    @abstractmethod
    def should_progress(self, metrics: CurriculumMetrics) -> bool:
        """
        Determine if agent should progress to next stage.

        Args:
            metrics: Current curriculum metrics

        Returns:
            True if agent should progress, False otherwise
        """
        pass

    @abstractmethod
    def should_regress(self, metrics: CurriculumMetrics) -> bool:
        """
        Determine if agent should regress to previous stage.

        Args:
            metrics: Current curriculum metrics

        Returns:
            True if agent should regress, False otherwise
        """
        pass

    @abstractmethod
    def get_stage_config(self, stage: CurriculumStage) -> dict[str, Any]:
        """
        Get environment configuration for a given stage.

        Args:
            stage: Curriculum stage

        Returns:
            Dictionary of environment configuration parameters
        """
        pass

    def get_next_stage(self, current_stage: CurriculumStage) -> Optional[CurriculumStage]:
        """
        Get next curriculum stage.

        Args:
            current_stage: Current stage

        Returns:
            Next stage, or None if at final stage
        """
        stages = list(CurriculumStage)
        current_idx = stages.index(current_stage)

        if current_idx < len(stages) - 1:
            return stages[current_idx + 1]
        return None

    def get_previous_stage(
        self, current_stage: CurriculumStage
    ) -> Optional[CurriculumStage]:
        """
        Get previous curriculum stage.

        Args:
            current_stage: Current stage

        Returns:
            Previous stage, or None if at first stage
        """
        stages = list(CurriculumStage)
        current_idx = stages.index(current_stage)

        if current_idx > 0:
            return stages[current_idx - 1]
        return None


class ThresholdScheduler(CurriculumScheduler):
    """
    Threshold-based curriculum scheduler.

    Progresses to next stage when success rate exceeds threshold,
    regresses when success rate falls below minimum threshold.

    Example:
        ```python
        from smlx.gym.curriculum import ThresholdScheduler, CurriculumStage

        scheduler = ThresholdScheduler(
            success_threshold=0.8,
            min_success_threshold=0.3,
            min_episodes_per_stage=100,
            stage_configs={
                CurriculumStage.EASY: {"max_steps": 100},
                CurriculumStage.MEDIUM: {"max_steps": 200},
                CurriculumStage.HARD: {"max_steps": 500},
            }
        )

        # Check if should progress
        if scheduler.should_progress(metrics):
            next_stage = scheduler.get_next_stage()
        ```
    """

    def __init__(
        self,
        success_threshold: float = 0.8,
        min_success_threshold: float = 0.3,
        min_episodes_per_stage: int = 100,
        stage_configs: Optional[dict[CurriculumStage, dict[str, Any]]] = None,
    ):
        """
        Initialize threshold scheduler.

        Args:
            success_threshold: Success rate threshold for progression
            min_success_threshold: Minimum success rate (regress if below)
            min_episodes_per_stage: Minimum episodes before allowing progression
            stage_configs: Configuration for each curriculum stage
        """
        super().__init__()

        self.success_threshold = success_threshold
        self.min_success_threshold = min_success_threshold
        self.min_episodes_per_stage = min_episodes_per_stage

        # Default stage configurations
        if stage_configs is None:
            self.stage_configs = {
                CurriculumStage.EASY: {"difficulty": 0.25},
                CurriculumStage.MEDIUM: {"difficulty": 0.5},
                CurriculumStage.HARD: {"difficulty": 0.75},
                CurriculumStage.EXPERT: {"difficulty": 1.0},
            }
        else:
            self.stage_configs = stage_configs

    def should_progress(self, metrics: CurriculumMetrics) -> bool:
        """Check if agent should progress to next stage."""
        # Must complete minimum episodes in current stage
        if metrics.episodes_in_stage < self.min_episodes_per_stage:
            return False

        # Must achieve success threshold
        return metrics.success_rate >= self.success_threshold

    def should_regress(self, metrics: CurriculumMetrics) -> bool:
        """Check if agent should regress to previous stage."""
        # Must complete minimum episodes in current stage
        if metrics.episodes_in_stage < self.min_episodes_per_stage:
            return False

        # Regress if performance too low
        return metrics.success_rate < self.min_success_threshold

    def get_stage_config(self, stage: CurriculumStage) -> dict[str, Any]:
        """Get configuration for a given stage."""
        return self.stage_configs.get(stage, {})


class CurriculumWrapper(gym.Wrapper):
    """
    Curriculum learning wrapper for Gymnasium environments.

    Wraps an environment to support curriculum learning, automatically
    adjusting difficulty based on agent performance.

    The wrapper tracks agent performance and progresses/regresses through
    curriculum stages based on a provided scheduler.

    Example:
        ```python
        import gymnasium as gym
        from smlx.gym.curriculum import CurriculumWrapper, ThresholdScheduler

        # Create environment
        env = gym.make("CartPole-v1")

        # Create curriculum scheduler
        scheduler = ThresholdScheduler(success_threshold=0.8)

        # Wrap environment
        curriculum_env = CurriculumWrapper(env, scheduler)

        # Train with curriculum learning
        obs, info = curriculum_env.reset()
        for _ in range(1000):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = curriculum_env.step(action)

            if terminated or truncated:
                print(f"Current stage: {curriculum_env.current_stage}")
                obs, info = curriculum_env.reset()
        ```
    """

    def __init__(
        self,
        env: gym.Env,
        scheduler: CurriculumScheduler,
        apply_config_fn: Optional[Callable[[gym.Env, dict], None]] = None,
    ):
        """
        Initialize curriculum wrapper.

        Args:
            env: Gymnasium environment to wrap
            scheduler: Curriculum scheduler
            apply_config_fn: Optional function to apply stage config to environment
                           Function signature: apply_config_fn(env, config) -> None
        """
        super().__init__(env)

        self.scheduler = scheduler
        self.apply_config_fn = apply_config_fn

        # Current curriculum stage
        self.current_stage = CurriculumStage.EASY

        # Episode tracking
        self.episode_returns: list[float] = []
        self.episode_successes: list[bool] = []
        self.episodes_in_current_stage = 0

        # Apply initial stage configuration
        self._apply_stage_config()

    def reset(self, **kwargs):
        """Reset environment and check for stage transitions."""
        # Update curriculum metrics
        self._update_metrics()

        # Check for stage transitions
        self._check_stage_transition()

        # Reset environment
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track episode statistics
        if terminated or truncated:
            episode_return = info.get("episode", {}).get("r", 0.0)
            success = info.get("success", False)

            self.episode_returns.append(episode_return)
            self.episode_successes.append(success)
            self.episodes_in_current_stage += 1

        return obs, reward, terminated, truncated, info

    def _update_metrics(self):
        """Update curriculum metrics based on recent performance."""
        if len(self.episode_returns) == 0:
            return

        # Compute metrics over recent episodes (last 100)
        recent_window = 100
        recent_returns = self.episode_returns[-recent_window:]
        recent_successes = self.episode_successes[-recent_window:]

        self.scheduler.metrics.current_stage = self.current_stage
        self.scheduler.metrics.success_rate = (
            sum(recent_successes) / len(recent_successes) if recent_successes else 0.0
        )
        self.scheduler.metrics.average_return = (
            sum(recent_returns) / len(recent_returns) if recent_returns else 0.0
        )
        self.scheduler.metrics.episodes_in_stage = self.episodes_in_current_stage

    def _check_stage_transition(self):
        """Check if curriculum stage should change."""
        # Check for progression
        if self.scheduler.should_progress(self.scheduler.metrics):
            next_stage = self.scheduler.get_next_stage(self.current_stage)
            if next_stage is not None:
                print(
                    f"Curriculum: Progressing from {self.current_stage.value} "
                    f"to {next_stage.value}"
                )
                self.current_stage = next_stage
                self.episodes_in_current_stage = 0
                self.scheduler.metrics.total_stage_changes += 1
                self._apply_stage_config()
                return

        # Check for regression
        if self.scheduler.should_regress(self.scheduler.metrics):
            prev_stage = self.scheduler.get_previous_stage(self.current_stage)
            if prev_stage is not None:
                print(
                    f"Curriculum: Regressing from {self.current_stage.value} "
                    f"to {prev_stage.value}"
                )
                self.current_stage = prev_stage
                self.episodes_in_current_stage = 0
                self.scheduler.metrics.total_stage_changes += 1
                self._apply_stage_config()

    def _apply_stage_config(self):
        """Apply stage configuration to environment."""
        stage_config = self.scheduler.get_stage_config(self.current_stage)

        if self.apply_config_fn is not None:
            self.apply_config_fn(self.env, stage_config)
        else:
            # Try to apply config directly to environment
            for key, value in stage_config.items():
                if hasattr(self.env, key):
                    setattr(self.env, key, value)


def create_curriculum_env(
    env_id: str,
    scheduler: CurriculumScheduler,
    apply_config_fn: Optional[Callable[[gym.Env, dict], None]] = None,
    **env_kwargs,
) -> CurriculumWrapper:
    """
    Factory function for creating curriculum learning environments.

    Args:
        env_id: Gymnasium environment ID
        scheduler: Curriculum scheduler
        apply_config_fn: Optional function to apply stage config
        **env_kwargs: Additional arguments for gym.make()

    Returns:
        Curriculum-wrapped environment

    Example:
        ```python
        from smlx.gym.curriculum import create_curriculum_env, ThresholdScheduler

        # Create scheduler
        scheduler = ThresholdScheduler(success_threshold=0.8)

        # Create curriculum environment
        env = create_curriculum_env("CartPole-v1", scheduler)

        # Train with curriculum learning
        obs, info = env.reset()
        ```
    """
    env = gym.make(env_id, **env_kwargs)
    return CurriculumWrapper(env, scheduler, apply_config_fn)
