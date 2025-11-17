"""
Environment wrappers for SMLX Gym.

This module provides modular environment modifications following the Decorator
pattern. Wrappers can be chained together to create flexible environment pipelines.

Reference: Gym_Claude.md, lines 14-16, 162-164; Comprehensive Guide, lines 159-180
"""

from collections import deque
from typing import Any, Optional

import gymnasium as gym
import mlx.core as mx
import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper


class MLXObservationWrapper(ObservationWrapper):
    """
    Ensure observations are MLX arrays.

    This wrapper converts all observations to MLX arrays for Metal GPU
    acceleration throughout the training pipeline.

    Example:
        ```python
        env = gym.make("CartPole-v1")
        env = MLXObservationWrapper(env)

        obs, info = env.reset()
        assert isinstance(obs, mx.array)
        ```
    """

    def observation(self, obs: Any) -> mx.array:
        """Convert observation to MLX array"""
        if isinstance(obs, mx.array):
            return obs
        elif isinstance(obs, np.ndarray):
            return mx.array(obs)
        elif isinstance(obs, (list, tuple)):
            return mx.array(obs)
        else:
            return mx.array(obs)


class NormalizeObservation(ObservationWrapper):
    """
    Normalize observations to zero mean and unit variance.

    Uses running statistics to normalize observations online during training.
    This can improve learning stability and speed convergence.

    Reference: Comprehensive Guide, lines 159-180

    Args:
        env: Environment to wrap
        epsilon: Small constant for numerical stability

    Example:
        ```python
        env = gym.make("CartPole-v1")
        env = NormalizeObservation(env)

        # Observations are automatically normalized
        obs, info = env.reset()
        ```
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.running_mean: Optional[mx.array] = None
        self.running_var: Optional[mx.array] = None
        self.count = 0

    def observation(self, obs: Any) -> mx.array:
        """Normalize observation using running statistics"""
        obs = mx.array(obs) if not isinstance(obs, mx.array) else obs

        if self.running_mean is None or self.running_var is None:
            self.running_mean = mx.zeros_like(obs)
            self.running_var = mx.ones_like(obs)

        self.count += 1

        # Update running statistics
        # After the check above, these are guaranteed to be mx.array
        assert self.running_mean is not None and self.running_var is not None
        delta = obs - self.running_mean
        self.running_mean = self.running_mean + delta / float(self.count)
        self.running_var = self.running_var + delta * (obs - self.running_mean)

        # Normalize
        # Type narrowing: both are guaranteed to be mx.array after the check above
        assert self.running_mean is not None and self.running_var is not None
        std = mx.sqrt(self.running_var / float(self.count) + self.epsilon)
        return (obs - self.running_mean) / std


class NormalizeReward(RewardWrapper):
    """
    Normalize rewards using running statistics.

    Uses return-based normalization to scale rewards appropriately.
    This can improve learning when reward scales vary significantly.

    Reference: Comprehensive Guide, lines 166-180

    Args:
        env: Environment to wrap
        epsilon: Small constant for numerical stability
        gamma: Discount factor for return calculation

    Example:
        ```python
        env = gym.make("CartPole-v1")
        env = NormalizeReward(env, gamma=0.99)

        # Rewards are automatically normalized
        obs, reward, terminated, truncated, info = env.step(action)
        ```
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8, gamma: float = 0.99):
        super().__init__(env)
        self.epsilon = epsilon
        self.gamma = gamma
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0
        self.return_val = 0.0

    def reward(self, reward: float) -> float:
        """Normalize reward using return-based statistics"""
        self.return_val = reward + self.gamma * self.return_val
        self.count += 1

        # Update running statistics
        delta = self.return_val - self.running_mean
        self.running_mean += delta / self.count
        self.running_var += delta * (self.return_val - self.running_mean)

        # Normalize
        std = np.sqrt(self.running_var / self.count + self.epsilon)
        return reward / max(std, self.epsilon)

    def reset(self, **kwargs):
        """Reset return accumulator"""
        self.return_val = 0.0
        return super().reset(**kwargs)


class ClipReward(RewardWrapper):
    """
    Clip rewards to a specified range.

    Useful for stabilizing training when reward magnitudes vary significantly.

    Args:
        env: Environment to wrap
        min_reward: Minimum reward value
        max_reward: Maximum reward value

    Example:
        ```python
        env = gym.make("CartPole-v1")
        env = ClipReward(env, min_reward=-1.0, max_reward=1.0)

        # Rewards are clipped to [-1, 1]
        obs, reward, terminated, truncated, info = env.step(action)
        assert -1.0 <= reward <= 1.0
        ```
    """

    def __init__(self, env: gym.Env, min_reward: float = -1.0, max_reward: float = 1.0):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, reward: float) -> float:
        """Clip reward to range"""
        return np.clip(reward, self.min_reward, self.max_reward)


class FrameStack(ObservationWrapper):
    """
    Stack the last N observations for temporal context.

    Useful for tasks where temporal information is important (e.g., velocity
    from multiple position observations).

    Reference: Gym_Claude.md, lines 162-164

    Args:
        env: Environment to wrap
        num_stack: Number of frames to stack

    Example:
        ```python
        env = gym.make("CartPole-v1")
        env = FrameStack(env, num_stack=4)

        # Observations now contain last 4 frames stacked
        obs, info = env.reset()
        assert obs.shape[0] == 4 * original_obs_shape[0]
        ```
    """

    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        # Modify observation space
        if isinstance(env.observation_space, gym.spaces.Box):
            low = np.repeat(env.observation_space.low, num_stack, axis=0)
            high = np.repeat(env.observation_space.high, num_stack, axis=0)
            # Use np.float32 or np.int32 directly instead of space.dtype
            dtype = np.float32 if np.issubdtype(env.observation_space.dtype, np.floating) else np.int32
            self.observation_space = gym.spaces.Box(
                low=low, high=high, dtype=dtype
            )
        else:
            # For non-Box spaces, keep original
            self.observation_space = env.observation_space

    def reset(self, **kwargs):
        """Reset and fill frame buffer"""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def observation(self, obs: Any) -> mx.array:
        """Stack frames"""
        self.frames.append(obs)
        return self._get_obs()

    def _get_obs(self) -> mx.array:
        """Get stacked observation"""
        # Convert all frames to MLX arrays
        mlx_frames = [
            mx.array(f) if not isinstance(f, mx.array) else f for f in self.frames
        ]
        return mx.concatenate(mlx_frames, axis=0)


class RecordEpisodeStatistics(Wrapper):
    """
    Record episode statistics for logging and evaluation.

    Tracks cumulative reward and episode length, storing them in the info
    dictionary when episodes end.

    Reference: Gym_Claude.md, lines 14-16; Comprehensive Guide, lines 194-226

    Args:
        env: Environment to wrap

    Example:
        ```python
        env = gym.make("CartPole-v1")
        env = RecordEpisodeStatistics(env)

        # Episode statistics available in info dict
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode return: {info['episode']['r']}")
            print(f"Episode length: {info['episode']['l']}")
        ```
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.current_return = 0.0
        self.current_length = 0

    def reset(self, **kwargs):
        """Reset episode statistics"""
        obs, info = self.env.reset(**kwargs)
        self.current_return = 0.0
        self.current_length = 0
        return obs, info

    def step(self, action):
        """Record episode statistics"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_return += float(reward)
        self.current_length += 1

        if terminated or truncated:
            info["episode"] = {
                "r": self.current_return,
                "l": self.current_length,
                "t": self.current_length,  # timesteps
            }
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)

        return obs, reward, terminated, truncated, info

    def get_episode_statistics(self) -> dict[str, float]:
        """
        Get summary statistics of recent episodes.

        Returns:
            Dictionary with mean/std of returns and lengths
        """
        return {
            "mean_return": (
                float(np.mean(self.episode_returns)) if self.episode_returns else 0.0
            ),
            "mean_length": (
                float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
            ),
            "std_return": float(np.std(self.episode_returns)) if self.episode_returns else 0.0,
            "num_episodes": len(self.episode_returns),
        }


class TimeLimit(Wrapper):
    """
    Limit episode length to maximum number of steps.

    Sets truncated=True when max steps reached, allowing value function
    to bootstrap correctly.

    Args:
        env: Environment to wrap
        max_episode_steps: Maximum steps per episode

    Example:
        ```python
        env = gym.make("CartPole-v1")
        env = TimeLimit(env, max_episode_steps=500)

        # Episode truncates after 500 steps
        ```
    """

    def __init__(self, env: gym.Env, max_episode_steps: int):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

    def reset(self, **kwargs):
        """Reset step counter"""
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Enforce time limit"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        if self.current_step >= self.max_episode_steps:
            truncated = True

        return obs, reward, terminated, truncated, info


class FlattenObservation(ObservationWrapper):
    """
    Flatten observation space to 1D vector.

    Useful for converting Dict or complex observations to flat vectors
    for neural network input.

    Example:
        ```python
        env = gym.make("MyDictObsEnv-v0")
        env = FlattenObservation(env)

        # Observations are now flat 1D vectors
        obs, info = env.reset()
        assert len(obs.shape) == 1
        ```
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        from gymnasium.spaces.utils import flatten_space
        self.observation_space = flatten_space(env.observation_space)

    def observation(self, obs: Any) -> mx.array:
        """Flatten observation"""
        from gymnasium.spaces.utils import flatten as gym_flatten
        # Convert to numpy for gym's flatten
        if isinstance(obs, mx.array):
            obs = np.array(obs)
        elif isinstance(obs, dict):
            obs = {k: np.array(v) if isinstance(v, mx.array) else v for k, v in obs.items()}

        flat_obs = gym_flatten(self.env.observation_space, obs)
        # flat_obs should be array-like, handle edge cases
        if isinstance(flat_obs, dict):
            raise TypeError("flatten returned dict which is unexpected")
        return mx.array(flat_obs)


class RescaleAction(ActionWrapper):
    """
    Rescale continuous actions from [-1, 1] to environment's action space.

    Useful for normalizing action spaces for neural network outputs.

    Args:
        env: Environment to wrap

    Example:
        ```python
        env = gym.make("Pendulum-v1")  # Action space is [-2, 2]
        env = RescaleAction(env)  # Now expects actions in [-1, 1]

        # Agent outputs action in [-1, 1]
        action = 0.5
        obs, reward, terminated, truncated, info = env.step(action)
        ```
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Check action space is Box
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("RescaleAction only works with Box action spaces")

        # Create new action space in [-1, 1]
        # Use np.float32 instead of the space's dtype to avoid type issues
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

        self.low = env.action_space.low
        self.high = env.action_space.high

    def action(self, action: Any) -> np.ndarray:
        """Rescale action from [-1, 1] to original bounds"""
        # Ensure action is in [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        # Scale to [low, high]
        return self.low + (action + 1.0) * 0.5 * (self.high - self.low)


class EpisodeLogger(Wrapper):
    """
    Log detailed episode information to console.

    Useful for debugging and monitoring training progress.

    Args:
        env: Environment to wrap
        log_every: Log every N episodes

    Example:
        ```python
        env = gym.make("CartPole-v1")
        env = EpisodeLogger(env, log_every=10)

        # Logs episode statistics every 10 episodes
        ```
    """

    def __init__(self, env: gym.Env, log_every: int = 1):
        super().__init__(env)
        self.log_every = log_every
        self.episode_count = 0
        self.episode_return = 0.0
        self.episode_length = 0

    def reset(self, **kwargs):
        """Reset episode tracking"""
        obs, info = self.env.reset(**kwargs)
        self.episode_return = 0.0
        self.episode_length = 0
        return obs, info

    def step(self, action):
        """Track and log episode"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += float(reward)
        self.episode_length += 1

        if terminated or truncated:
            self.episode_count += 1
            if self.episode_count % self.log_every == 0:
                print(
                    f"Episode {self.episode_count}: "
                    f"Return={self.episode_return:.2f}, "
                    f"Length={self.episode_length}, "
                    f"Terminated={terminated}"
                )

        return obs, reward, terminated, truncated, info


def make_env_with_wrappers(
    env_id: str,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    clip_reward: bool = False,
    frame_stack: Optional[int] = None,
    time_limit: Optional[int] = None,
    record_stats: bool = True,
    **env_kwargs,
) -> gym.Env:
    """
    Create environment with standard wrapper pipeline.

    Convenience function for creating environments with common wrappers.

    Args:
        env_id: Gymnasium environment ID
        normalize_obs: Apply observation normalization
        normalize_reward: Apply reward normalization
        clip_reward: Clip rewards to [-1, 1]
        frame_stack: Number of frames to stack (None to disable)
        time_limit: Maximum episode steps (None for no limit)
        record_stats: Record episode statistics
        **env_kwargs: Additional arguments for gym.make()

    Returns:
        Wrapped environment

    Example:
        ```python
        env = make_env_with_wrappers(
            "CartPole-v1",
            normalize_obs=True,
            normalize_reward=True,
            time_limit=500,
            record_stats=True
        )
        ```
    """
    env = gym.make(env_id, **env_kwargs)

    if time_limit is not None:
        env = TimeLimit(env, max_episode_steps=time_limit)

    if normalize_obs:
        env = NormalizeObservation(env)

    if frame_stack is not None:
        env = FrameStack(env, num_stack=frame_stack)

    if normalize_reward:
        env = NormalizeReward(env)

    if clip_reward:
        env = ClipReward(env)

    if record_stats:
        env = RecordEpisodeStatistics(env)

    # Ensure MLX observations
    env = MLXObservationWrapper(env)

    return env
