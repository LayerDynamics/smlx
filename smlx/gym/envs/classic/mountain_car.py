"""
MountainCar environment with MLX support.

Classic control task where the goal is to drive an underpowered car up a steep
mountain by building momentum. This is an MLX-native implementation optimized for
Metal GPU acceleration.

Reference: SMLX_Gym.md, Section 4.1 (Environment Implementations)
"""

from typing import Any, Optional

import mlx.core as mx
import numpy as np
from gymnasium import spaces

from smlx.gym.base import MLXEnv


class MountainCarEnv(MLXEnv):
    """
    MountainCar environment with MLX.

    The agent must drive an underpowered car up a steep mountain. The car
    cannot climb the mountain directly, so it must build momentum by driving
    back and forth.

    Observation Space:
        Box(2) with:
        - position: [-1.2, 0.6]
        - velocity: [-0.07, 0.07]

    Action Space:
        Discrete(3):
        - 0: Accelerate to the left
        - 1: Don't accelerate
        - 2: Accelerate to the right

    Rewards:
        -1 for every step until the goal is reached

    Episode Termination:
        - Car position >= 0.5 (goal reached)
        - Episode length > 200 steps

    Example:
        ```python
        from smlx.gym.envs.classic import MountainCarEnv
        from smlx.gym.algorithms import DQNAgent

        # Create environment
        env = MountainCarEnv()

        # Train agent
        agent = DQNAgent(env)
        agent.run(num_episodes=1000)
        ```
    """

    def __init__(self, render_mode: Optional[str] = None, goal_velocity: float = 0.0):
        """
        Initialize MountainCar environment.

        Args:
            render_mode: Rendering mode ('human' or None)
            goal_velocity: Minimum velocity required at goal position
        """
        super().__init__(render_mode=render_mode)

        # Physics parameters
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        self.force = 0.001
        self.gravity = 0.0025

        # Episode limit
        self.max_episode_steps = 200

        # Define observation and action spaces
        low = mx.array([self.min_position, -self.max_speed])
        high = mx.array([self.max_position, self.max_speed])
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # State (as MLX array)
        self.state: Optional[mx.array] = None
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

        # Initialize position randomly between -0.6 and -0.4
        position = mx.random.uniform(-0.6, -0.4, (1,))[0]
        velocity = mx.array(0.0)

        self.state = mx.stack([position, velocity])
        self.current_step = 0

        return self.state, {}

    def step(self, action: int) -> tuple[mx.array, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0, 1, or 2)

        Returns:
            observation: Next state (MLX array)
            reward: Reward for this step
            terminated: Whether episode ended (goal reached)
            truncated: Whether episode was truncated (max steps reached)
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        # Extract state components
        position, velocity = self.state[0], self.state[1]

        # Update velocity based on action and gravity
        # Action: 0 = left, 1 = no push, 2 = right
        force = (action - 1) * self.force

        # Compute new velocity using MLX operations
        # velocity += force + gravity * cos(3 * position)
        velocity = velocity + force + mx.cos(3 * position) * (-self.gravity)
        velocity = mx.clip(velocity, -self.max_speed, self.max_speed)

        # Update position using MLX operations
        position = position + velocity
        position = mx.clip(position, self.min_position, self.max_position)

        # Reset velocity if at left boundary
        if float(position) == self.min_position and float(velocity) < 0:
            velocity = mx.array(0.0)

        # Create new state
        self.state = mx.stack([position, velocity])

        # Check if goal reached
        terminated = bool(
            float(position) >= self.goal_position and float(velocity) >= self.goal_velocity
        )

        # Increment step counter
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        # Reward is -1 for every step except when goal is reached
        reward = 0.0 if terminated else -1.0

        return self.state, reward, terminated, truncated, {}

    def render(self):
        """Render the current state."""
        if self.render_mode == "human" and self.state is not None:
            position, velocity = self.state
            print(
                f"Position: {float(position):.3f}, "
                f"Velocity: {float(velocity):.3f}, "
                f"Step: {self.current_step}/{self.max_episode_steps}"
            )

    def close(self):
        """Clean up resources."""
        pass


class MountainCarContinuousEnv(MLXEnv):
    """
    Continuous MountainCar environment with MLX.

    Similar to MountainCar but with continuous action space for more precise control.

    Observation Space:
        Box(2) with:
        - position: [-1.2, 0.6]
        - velocity: [-0.07, 0.07]

    Action Space:
        Box(1) with:
        - force: [-1.0, 1.0]

    Rewards:
        Reward is based on how quickly the goal is reached and fuel efficiency

    Example:
        ```python
        from smlx.gym.envs.classic import MountainCarContinuousEnv
        from smlx.gym.algorithms import PPOAgent

        # Create environment
        env = MountainCarContinuousEnv()

        # Train agent (PPO works well with continuous actions)
        agent = PPOAgent(env)
        agent.run(num_episodes=1000)
        ```
    """

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize continuous MountainCar environment.

        Args:
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__(render_mode=render_mode)

        # Physics parameters (same as discrete version)
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.power = 0.0015

        # Episode limit
        self.max_episode_steps = 999

        # Define observation and action spaces
        low = mx.array([self.min_position, -self.max_speed])
        high = mx.array([self.max_position, self.max_speed])
        self.observation_space = spaces.Box(
            low=np.array(low), high=np.array(high), dtype=np.float32
        )

        # Continuous action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # State (as MLX array)
        self.state: Optional[mx.array] = None
        self.current_step = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[mx.array, dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Initialize position randomly between -0.6 and -0.4
        position = mx.random.uniform(-0.6, -0.4, (1,))[0]
        velocity = mx.array(0.0)

        self.state = mx.stack([position, velocity])
        self.current_step = 0

        return self.state, {}

    def step(self, action: float) -> tuple[mx.array, float, bool, bool, dict[str, Any]]:
        """
        Execute one step with continuous action.

        Args:
            action: Continuous force value in [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        # Convert action to MLX array if needed
        action_array: mx.array
        if not isinstance(action, mx.array):
            if isinstance(action, (list, tuple)):
                action_array = mx.array(action[0])
            else:
                action_array = mx.array(float(action))
        else:
            action_array = action

        # Clip action to valid range
        action_array = mx.clip(action_array, -1.0, 1.0)

        # Extract state
        position, velocity = self.state[0], self.state[1]

        # Update velocity with continuous force
        force = action_array * self.power
        velocity = velocity + force - 0.0025 * mx.cos(3 * position)
        velocity = mx.clip(velocity, -self.max_speed, self.max_speed)

        # Update position
        position = position + velocity
        position = mx.clip(position, self.min_position, self.max_position)

        # Reset velocity if at left boundary
        if float(position) == self.min_position and float(velocity) < 0:
            velocity = mx.array(0.0)

        # Update state
        self.state = mx.stack([position, velocity])

        # Check termination
        terminated = bool(float(position) >= self.goal_position)

        # Increment step counter
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        # Reward: penalize for taking too long and using too much power
        reward = 100.0 if terminated else -(float(action_array) ** 2) * 0.1

        return self.state, reward, terminated, truncated, {}

    def render(self):
        """Render the current state."""
        if self.render_mode == "human" and self.state is not None:
            position, velocity = self.state
            print(
                f"Position: {float(position):.3f}, "
                f"Velocity: {float(velocity):.3f}, "
                f"Step: {self.current_step}/{self.max_episode_steps}"
            )

    def close(self):
        """Clean up resources."""
        pass
