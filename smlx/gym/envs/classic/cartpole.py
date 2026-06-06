"""
CartPole environment with MLX support.

Classic control task where the goal is to balance a pole on a cart by moving
the cart left or right. This is an MLX-native implementation optimized for
Metal GPU acceleration.

Reference: SMLX_Gym.md, Section 4.1 (Environment Implementations)
"""

from typing import Any, Optional

import mlx.core as mx
import numpy as np
from gymnasium import spaces

from smlx.gym.base import MLXEnv


class CartPoleEnv(MLXEnv):
    """
    CartPole balancing environment with MLX.

    The agent must balance a pole on a cart by applying forces to move the
    cart left or right. The episode ends if the pole falls too far or the
    cart moves too far from the center.

    Observation Space:
        Box(4) with:
        - cart position: [-4.8, 4.8]
        - cart velocity: [-inf, inf]
        - pole angle: [-0.418 rad, 0.418 rad]
        - pole angular velocity: [-inf, inf]

    Action Space:
        Discrete(2):
        - 0: Push cart to the left
        - 1: Push cart to the right

    Rewards:
        +1 for every step the pole remains upright

    Episode Termination:
        - Pole angle > ±12 degrees
        - Cart position > ±2.4
        - Episode length > 500 steps

    Example:
        ```python
        from smlx.gym.envs.classic import CartPoleEnv
        from smlx.gym.algorithms import DQNAgent

        # Create environment
        env = CartPoleEnv()

        # Train agent
        agent = DQNAgent(env)
        agent.run(num_episodes=1000)
        ```
    """

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize CartPole environment.

        Args:
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__(render_mode=render_mode)

        # Physics parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # Half-pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # Time step

        # Thresholds
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        # Episode limit
        self.max_episode_steps = 500

        # Define observation and action spaces
        high = mx.array([self.x_threshold * 2, np.inf, self.theta_threshold_radians * 2, np.inf])
        self.observation_space = spaces.Box(
            low=-np.array(high), high=np.array(high), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        # State (as MLX array)
        self.state: Optional[mx.array] = None
        self.steps_beyond_terminated: Optional[int] = None
        self.current_step = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[mx.array, dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation: Initial state (MLX array)
            info: Additional information
        """
        super().reset(seed=seed)

        # Initialize state randomly using MLX
        self.state = mx.random.uniform(-0.05, 0.05, (4,))
        self.steps_beyond_terminated = None
        self.current_step = 0

        return self.state, {}

    def step(self, action: int) -> tuple[mx.array, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0 or 1)

        Returns:
            observation: Next state (MLX array)
            reward: Reward for this step
            terminated: Whether episode ended (pole fell or cart out of bounds)
            truncated: Whether episode was truncated (max steps reached)
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        # Extract state components (using MLX operations)
        x, x_dot, theta, theta_dot = self.state[0], self.state[1], self.state[2], self.state[3]

        # Apply force based on action
        force = self.force_mag if action == 1 else -self.force_mag

        # Physics equations (using MLX operations for Metal GPU acceleration)
        costheta = mx.cos(theta)
        sintheta = mx.sin(theta)

        # Compute acceleration using MLX operations
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Update state using Euler's method (MLX operations)
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        # Create new state (MLX array)
        self.state = mx.stack([x, x_dot, theta, theta_dot])

        # Check termination conditions
        terminated = bool(
            float(x) < -self.x_threshold
            or float(x) > self.x_threshold
            or float(theta) < -self.theta_threshold_radians
            or float(theta) > self.theta_threshold_radians
        )

        # Increment step counter
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        # Compute reward
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            self.steps_beyond_terminated += 1
            reward = 0.0

        return self.state, reward, terminated, truncated, {}

    def render(self):
        """Render the current state."""
        if self.render_mode == "human" and self.state is not None:
            x, _, theta, _ = self.state
            print(
                f"Cart position: {float(x):.2f}, "
                f"Pole angle: {float(theta) * 180 / np.pi:.2f}°"
            )

    def close(self):
        """Clean up resources."""
        pass
