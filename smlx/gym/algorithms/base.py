"""
Base RL agent class for SMLX Gym.

This module provides the abstract base class for all RL algorithms in SMLX Gym.
All algorithms use MLX for Metal GPU acceleration throughout training.

Reference: SMLX_Gym.md, Section 4.3 (Algorithms)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import mlx.core as mx


@dataclass
class AgentConfig:
    """
    Configuration for RL agents.

    Attributes:
        gamma: Discount factor
        learning_rate: Learning rate for optimizer
        max_episodes: Maximum training episodes
        max_steps_per_episode: Maximum steps per episode
        log_interval: Episodes between logging
        save_interval: Episodes between checkpoints
        eval_interval: Episodes between evaluations
        eval_episodes: Number of evaluation episodes
    """

    gamma: float = 0.99
    learning_rate: float = 1e-3
    max_episodes: int = 1000
    max_steps_per_episode: Optional[int] = None
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    eval_episodes: int = 10
    seed: Optional[int] = None
    device: str = "mps"  # MLX uses Metal Performance Shaders
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """
    Training metrics for RL agents.

    Attributes:
        episode: Current episode number
        episode_return: Total return for episode
        episode_length: Length of episode
        average_return: Moving average of returns
        average_length: Moving average of episode lengths
        loss: Current loss value
        success_rate: Success rate (if applicable)
        training_time: Total training time in seconds
        metadata: Additional metrics
    """

    episode: int = 0
    episode_return: float = 0.0
    episode_length: int = 0
    average_return: float = 0.0
    average_length: float = 0.0
    loss: float = 0.0
    success_rate: float = 0.0
    training_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RLAgent(ABC):
    """
    Abstract base class for RL agents.

    All RL algorithms in SMLX Gym should inherit from this class and implement
    the abstract methods. The base class provides common functionality for:
    - Episode execution
    - Training loops
    - Metrics tracking
    - Model saving/loading

    All agents use MLX arrays for Metal GPU acceleration throughout the
    training pipeline.

    Example:
        ```python
        import gymnasium as gym
        from smlx.gym.algorithms.base import RLAgent

        class MyAgent(RLAgent):
            def select_action(self, observation):
                # Implement action selection
                pass

            def train_step(self, batch):
                # Implement training step
                pass

        # Create and train agent
        env = gym.make("CartPole-v1")
        agent = MyAgent(env)
        response = agent.run(num_episodes=1000)
        ```
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[AgentConfig] = None,
        **kwargs,
    ):
        """
        Initialize RL agent.

        Args:
            env: Gymnasium environment
            config: Agent configuration (defaults to AgentConfig())
            **kwargs: Additional arguments (added to config.metadata)
        """
        self.env = env
        self.config = config if config is not None else AgentConfig()

        # Add kwargs to metadata
        if kwargs:
            self.config.metadata.update(kwargs)

        # Set random seed if provided
        if self.config.seed is not None:
            mx.random.seed(self.config.seed)

        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.metrics_history: list[TrainingMetrics] = []
        self.current_metrics = TrainingMetrics()

        # True while _evaluate() is running. Subclasses that separate exploration
        # from exploitation (e.g. epsilon-greedy) can check this to act greedily.
        self.eval_mode = False

        # Episode statistics (for moving average)
        self._episode_returns: list[float] = []
        self._episode_lengths: list[int] = []
        self._window_size = 100  # Window for moving average

    @abstractmethod
    def select_action(self, observation: mx.array) -> Any:
        """
        Select action given current observation.

        Args:
            observation: Current state observation (MLX array)

        Returns:
            Selected action (type depends on action space)
        """
        pass

    @abstractmethod
    def train_step(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """
        Perform one training step using batch of experiences.

        Args:
            batch: Batch of experiences containing MLX arrays

        Returns:
            Dictionary of training metrics (e.g., {'loss': 0.5})
        """
        pass

    def run(
        self,
        num_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
        verbose: bool = True,
    ) -> TrainingMetrics:
        """
        Run training loop.

        Args:
            num_episodes: Number of episodes to train (overrides config)
            max_steps: Maximum steps per episode (overrides config)
            verbose: Whether to print training progress

        Returns:
            Final training metrics

        Example:
            ```python
            agent = MyAgent(env)
            metrics = agent.run(num_episodes=1000, verbose=True)
            print(f"Final return: {metrics.average_return}")
            ```
        """
        import time

        start_time = time.time()

        # Use config values if not overridden
        num_episodes = num_episodes or self.config.max_episodes
        max_steps = max_steps or self.config.max_steps_per_episode

        for episode in range(num_episodes):
            # Run episode
            episode_return, episode_length, success = self._run_episode(max_steps)

            # Update statistics
            self._episode_returns.append(episode_return)
            self._episode_lengths.append(episode_length)

            # Keep only recent episodes for moving average
            if len(self._episode_returns) > self._window_size:
                self._episode_returns = self._episode_returns[-self._window_size :]
                self._episode_lengths = self._episode_lengths[-self._window_size :]

            # Update metrics
            self.episode_count += 1
            self.current_metrics.episode = self.episode_count
            self.current_metrics.episode_return = episode_return
            self.current_metrics.episode_length = episode_length
            self.current_metrics.average_return = float(
                sum(self._episode_returns) / len(self._episode_returns)
            )
            self.current_metrics.average_length = float(
                sum(self._episode_lengths) / len(self._episode_lengths)
            )
            self.current_metrics.training_time = time.time() - start_time

            # Log progress
            if verbose and (episode + 1) % self.config.log_interval == 0:
                self._log_progress()

            # Save checkpoint
            if (episode + 1) % self.config.save_interval == 0:
                self.save(f"checkpoint_episode_{episode + 1}.safetensors")

            # Evaluate
            if (episode + 1) % self.config.eval_interval == 0:
                self._evaluate()

        return self.current_metrics

    def _run_episode(self, max_steps: Optional[int] = None) -> tuple[float, int, bool]:
        """
        Run single episode.

        Args:
            max_steps: Maximum steps per episode

        Returns:
            tuple of (episode_return, episode_length, success)
        """
        observation, info = self.env.reset()

        # Convert observation to MLX array
        if not isinstance(observation, mx.array):
            observation = mx.array(observation)

        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            # Select action
            action = self.select_action(observation)

            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Convert next observation to MLX array
            if not isinstance(next_obs, mx.array):
                next_obs = mx.array(next_obs)

            episode_return += float(reward)
            episode_length += 1
            observation = next_obs
            self.total_steps += 1

            if max_steps and episode_length >= max_steps:
                break

        success = info.get("success", False) if "success" in info else False
        return episode_return, episode_length, success

    def _log_progress(self):
        """Log training progress to console."""
        print(
            f"Episode {self.current_metrics.episode}: "
            f"Return={self.current_metrics.average_return:.2f}, "
            f"Length={self.current_metrics.average_length:.1f}, "
            f"Loss={self.current_metrics.loss:.4f}"
        )

    def _evaluate(self) -> dict[str, float]:
        """Run greedy evaluation episodes and record the results.

        Runs ``config.eval_episodes`` episodes with the current policy, performing
        no training updates and without disturbing the training counters or the
        moving-average windows. ``self.eval_mode`` is True for the duration so
        subclasses that explore during training can act greedily here.

        Returns:
            Dict with ``eval_average_return``, ``eval_average_length``,
            ``eval_success_rate`` and ``eval_episodes``.
        """
        num_eval = max(1, self.config.eval_episodes)
        max_steps = self.config.max_steps_per_episode

        returns: list[float] = []
        lengths: list[int] = []
        successes = 0

        self.eval_mode = True
        try:
            for _ in range(num_eval):
                observation, info = self.env.reset()
                if not isinstance(observation, mx.array):
                    observation = mx.array(observation)

                episode_return = 0.0
                episode_length = 0
                done = False
                while not done:
                    action = self.select_action(observation)
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated

                    if not isinstance(next_obs, mx.array):
                        next_obs = mx.array(next_obs)

                    episode_return += float(reward)
                    episode_length += 1
                    observation = next_obs

                    if max_steps and episode_length >= max_steps:
                        break

                returns.append(episode_return)
                lengths.append(episode_length)
                if isinstance(info, dict) and info.get("success", False):
                    successes += 1
        finally:
            self.eval_mode = False

        eval_metrics = {
            "eval_average_return": float(sum(returns) / len(returns)),
            "eval_average_length": float(sum(lengths) / len(lengths)),
            "eval_success_rate": float(successes / num_eval),
            "eval_episodes": float(num_eval),
        }

        # Record on the current metrics so downstream consumers can read them.
        self.current_metrics.metadata["eval"] = eval_metrics
        self.current_metrics.success_rate = eval_metrics["eval_success_rate"]

        print(
            f"  [eval] episodes={num_eval} "
            f"avg_return={eval_metrics['eval_average_return']:.2f} "
            f"avg_length={eval_metrics['eval_average_length']:.1f} "
            f"success_rate={eval_metrics['eval_success_rate']:.1%}"
        )

        return eval_metrics

    def save(self, path: str):
        """
        Save agent state to disk.

        Subclasses should override this to save model parameters.

        Args:
            path: Path to save agent state
        """
        pass

    def load(self, path: str):
        """
        Load agent state from disk.

        Subclasses should override this to load model parameters.

        Args:
            path: Path to load agent state from
        """
        pass

    def reset(self):
        """Reset agent to initial state."""
        self.episode_count = 0
        self.total_steps = 0
        self.metrics_history = []
        self.current_metrics = TrainingMetrics()
        self._episode_returns = []
        self._episode_lengths = []


class RandomAgent(RLAgent):
    """
    Random agent that samples actions uniformly.

    Useful for baseline comparisons and testing environments.

    Example:
        ```python
        import gymnasium as gym
        from smlx.gym.algorithms.base import RandomAgent

        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        # Run random agent
        metrics = agent.run(num_episodes=100)
        print(f"Random agent return: {metrics.average_return}")
        ```
    """

    def select_action(self, observation: mx.array) -> Any:
        """Sample random action from action space."""
        return self.env.action_space.sample()

    def train_step(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """Random agent doesn't train."""
        return {"loss": 0.0}
