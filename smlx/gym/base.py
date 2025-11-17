"""
MLX-optimized Gymnasium environment base class.

This module provides the base class for all SMLX gym environments,
optimized for Apple Silicon via MLX arrays and Metal GPU acceleration.
"""

from typing import Any, Optional

import gymnasium as gym
import mlx.core as mx
from gymnasium import spaces


class MLXEnv(gym.Env):
    """
    Base environment class optimized for MLX arrays and Apple Silicon.

    All SMLX gym environments should inherit from this class to ensure
    compatibility with Metal GPU acceleration and the MLX ecosystem.

    This class adapts the Gymnasium interface to use MLX arrays instead
    of NumPy arrays, enabling Metal GPU acceleration throughout the
    reinforcement learning training pipeline.

    Attributes:
        observation_space: Gymnasium space defining valid observations
        action_space: Gymnasium space defining valid actions
        metadata: Environment metadata (render modes, FPS, etc.)
        render_mode: Current rendering mode

    Example:
        ```python
        class MyEnv(MLXEnv):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
                )
                self.action_space = spaces.Discrete(2)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                observation = self._mlx_random_uniform((4,), -1.0, 1.0)
                return observation, {}

            def step(self, action):
                # Environment dynamics
                observation = self._mlx_random_uniform((4,), -1.0, 1.0)
                reward = 0.0
                terminated = False
                truncated = False
                info = {}
                return observation, reward, terminated, truncated, info
        ```
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the MLX environment.

        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        self.render_mode = render_mode

        # Subclasses must define these
        self.observation_space: Optional[spaces.Space] = None
        self.action_space: Optional[spaces.Space] = None

        # MLX-specific: use Metal-optimized RNG
        self._mlx_rng_key = mx.random.key(0)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to an initial state.

        This method must be called before the first step. It initializes
        the environment and returns the initial observation.

        Args:
            seed: Random seed for reproducibility. If provided, sets the
                  RNG state for both NumPy (via super().reset()) and MLX.
            options: Additional options for reset (environment-specific)

        Returns:
            observation: Initial observation (MLX array or dict containing MLX arrays)
            info: Auxiliary information dictionary

        Note:
            Subclasses should override this method and call super().reset(seed=seed)
            to ensure proper seeding of both NumPy and MLX RNGs.
        """
        super().reset(seed=seed)

        # Seed MLX RNG for Metal-accelerated randomness
        if seed is not None:
            self._mlx_rng_key = mx.random.key(seed)

        # Base implementation - subclasses should override
        # Return empty observation and info if not overridden
        return mx.array([]), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Execute one timestep within the environment.

        Takes an action and returns the next observation, reward,
        and episode termination flags.

        Args:
            action: Action from action_space to execute

        Returns:
            observation: Observation (MLX array or dict containing MLX arrays)
            reward: Scalar reward value
            terminated: Whether episode ended naturally (goal reached or
                       failure state). When True, value function should
                       not bootstrap from next state.
            truncated: Whether episode was cut off (time limit, boundary).
                      When True, value function should bootstrap from
                      next state.
            info: Auxiliary information dictionary (for debugging/logging)

        Note:
            The distinction between terminated and truncated is critical
            for correct value function learning. See Gymnasium docs:
            https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/

            Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement step()")

    def render(self):
        """
        Render the environment state.

        The rendering behavior depends on self.render_mode:
        - 'human': Display to screen or console
        - 'rgb_array': Return numpy array for video recording
        - None: No rendering

        Returns:
            If render_mode is 'rgb_array', returns numpy array of shape
            (height, width, 3) with RGB values in [0, 255].
            Otherwise, returns None.
        """
        if self.render_mode == "human":
            # Human-readable visualization (override in subclass)
            pass
        elif self.render_mode == "rgb_array":
            # Return numpy array for recording (override in subclass)
            pass

    def close(self):
        """
        Cleanup resources when environment is no longer needed.

        Override this method if your environment needs to cleanup
        resources like:
        - Rendering windows
        - File handles
        - Network connections
        - External processes

        This method should be idempotent (safe to call multiple times).
        """
        pass

    # MLX-specific helper methods

    def _mlx_random_uniform(
        self, shape: tuple[int, ...], low: float = 0.0, high: float = 1.0
    ) -> mx.array:
        """
        Generate uniform random MLX array using Metal GPU.

        This is a convenience method for generating random observations
        or initializing environment state using MLX's GPU-accelerated
        random number generation.

        Args:
            shape: Shape of the output array
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)

        Returns:
            MLX array with random values uniformly distributed in [low, high)

        Example:
            ```python
            # Generate random 4D observation
            obs = self._mlx_random_uniform((4,), -1.0, 1.0)
            ```
        """
        # Generate uniform random values using MLX RNG
        uniform_vals = mx.random.uniform(shape=shape, key=self._mlx_rng_key)

        # Scale to [low, high)
        return low + (high - low) * uniform_vals

    def _mlx_random_normal(
        self, shape: tuple[int, ...], mean: float = 0.0, std: float = 1.0
    ) -> mx.array:
        """
        Generate normal random MLX array using Metal GPU.

        This is a convenience method for generating random observations
        with Gaussian distribution using MLX's GPU-accelerated random
        number generation.

        Args:
            shape: Shape of the output array
            mean: Mean of the normal distribution
            std: Standard deviation of the normal distribution

        Returns:
            MLX array with random values from normal distribution

        Example:
            ```python
            # Generate random observation with noise
            noise = self._mlx_random_normal((4,), mean=0.0, std=0.1)
            obs = base_obs + noise
            ```
        """
        # Generate standard normal using MLX RNG
        normal_vals = mx.random.normal(shape=shape, key=self._mlx_rng_key)

        # Scale to desired mean and std
        return mean + std * normal_vals

    def _mlx_random_integers(
        self, shape: tuple[int, ...], low: int = 0, high: int = 10
    ) -> mx.array:
        """
        Generate random integer MLX array using Metal GPU.

        Args:
            shape: Shape of the output array
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)

        Returns:
            MLX array with random integers in [low, high)

        Example:
            ```python
            # Generate random discrete observation
            discrete_obs = self._mlx_random_integers((1,), 0, 5)
            ```
        """
        return mx.random.randint(low, high, shape=shape, key=self._mlx_rng_key)
