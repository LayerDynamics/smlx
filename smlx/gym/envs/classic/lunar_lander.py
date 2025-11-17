"""
LunarLander environment with MLX support.

Control a lunar lander spacecraft to land safely on the moon's surface.
This is a simplified MLX-native implementation optimized for Metal GPU acceleration.

Reference: SMLX_Gym.md, Section 4.1 (Environment Implementations)
"""

from typing import Any, Optional

import mlx.core as mx
import numpy as np
from gymnasium import spaces

from smlx.gym.base import MLXEnv


class LunarLanderEnv(MLXEnv):
    """
    LunarLander environment with MLX.

    The agent must land a spacecraft on the moon's surface between two flags.
    The lander has two side engines and a main engine for control.

    Observation Space:
        Box(8) with:
        - x position (0 is center)
        - y position (0 is ground level)
        - x velocity
        - y velocity
        - angle (0 is upright)
        - angular velocity
        - left leg contact (0 or 1)
        - right leg contact (0 or 1)

    Action Space:
        Discrete(4):
        - 0: Do nothing
        - 1: Fire left orientation engine
        - 2: Fire main engine
        - 3: Fire right orientation engine

    Rewards:
        - Moving from top of screen to landing pad: +100 to +140 points
        - Landing in safe zone: +100 points
        - Crash: -100 points
        - Leg ground contact: +10 points each
        - Firing main engine: -0.3 points per frame
        - Episode completion: +100 points

    Episode Termination:
        - Lander crashes
        - Lander lands successfully
        - Episode length > 1000 steps

    Example:
        ```python
        from smlx.gym.envs.classic import LunarLanderEnv
        from smlx.gym.algorithms import DQNAgent

        # Create environment
        env = LunarLanderEnv()

        # Train agent
        agent = DQNAgent(env)
        agent.run(num_episodes=2000)
        ```
    """

    def __init__(self, render_mode: Optional[str] = None, continuous: bool = False):
        """
        Initialize LunarLander environment.

        Args:
            render_mode: Rendering mode ('human' or None)
            continuous: Whether to use continuous action space
        """
        super().__init__(render_mode=render_mode)

        # Physics parameters
        self.gravity = -10.0
        self.enable_wind = False
        self.wind_power = 15.0
        self.turbulence_power = 1.5

        # Lander parameters
        self.main_engine_power = 13.0
        self.side_engine_power = 0.6

        # Episode limit
        self.max_episode_steps = 1000

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        if continuous:
            # Continuous: [main_engine, left/right]
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            # Discrete: do nothing, left, main, right
            self.action_space = spaces.Discrete(4)

        self.continuous = continuous

        # State (as MLX array)
        self.state: Optional[mx.array] = None
        self.current_step = 0
        self.prev_shaping: Optional[float] = None

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

        # Initialize lander at top center with small random perturbations
        x = mx.random.uniform(-0.5, 0.5, (1,))[0]
        y = mx.array(1.5)  # Start at top
        vx = mx.random.uniform(-0.5, 0.5, (1,))[0]
        vy = mx.array(0.0)
        angle = mx.random.uniform(-0.5, 0.5, (1,))[0]
        angular_velocity = mx.random.uniform(-0.5, 0.5, (1,))[0]
        left_leg_contact = mx.array(0.0)
        right_leg_contact = mx.array(0.0)

        self.state = mx.stack(
            [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
        )

        self.current_step = 0
        self.prev_shaping = None

        return self.state, {}

    def step(
        self, action: Any
    ) -> tuple[mx.array, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (int for discrete, array for continuous)

        Returns:
            observation: Next state (MLX array)
            reward: Reward for this step
            terminated: Whether episode ended (crashed or landed)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        # Extract state components
        x, y, vx, vy, angle, angular_vel, leg_l, leg_r = (
            self.state[0],
            self.state[1],
            self.state[2],
            self.state[3],
            self.state[4],
            self.state[5],
            self.state[6],
            self.state[7],
        )

        # Process action
        if self.continuous:
            # Continuous action: [main_engine, left/right]
            if isinstance(action, (list, tuple, np.ndarray)):
                action = mx.array(action)
            main_power = float(mx.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
            side_power = float(action[1])
        else:
            # Discrete action
            main_power = 1.0 if action == 2 else 0.0
            side_power = -1.0 if action == 1 else (1.0 if action == 3 else 0.0)

        # Apply physics
        dt = 1.0 / 50.0  # 50 FPS

        # Main engine (upward thrust)
        vy = vy + main_power * self.main_engine_power * dt

        # Side engines (rotation)
        angular_vel = angular_vel + side_power * self.side_engine_power * dt

        # Gravity
        vy = vy + self.gravity * dt

        # Update velocities and positions
        x = x + vx * dt
        y = y + vy * dt
        angle = angle + angular_vel * dt

        # Simple ground collision detection
        on_ground = float(y) <= 0.0

        if on_ground:
            # Landing logic
            y = mx.array(0.0)
            vy = mx.array(0.0)

            # Check if upright enough for landing
            if abs(float(angle)) < 0.3 and abs(float(vx)) < 1.0:
                leg_l = mx.array(1.0)
                leg_r = mx.array(1.0)
            else:
                # Crashed
                leg_l = mx.array(0.0)
                leg_r = mx.array(0.0)

        # Create new state
        self.state = mx.stack([x, y, vx, vy, angle, angular_vel, leg_l, leg_r])

        # Compute reward using shaping
        shaping = (
            -100 * mx.sqrt(x * x + y * y)  # Distance from center
            - 100 * mx.sqrt(vx * vx + vy * vy)  # Velocity magnitude
            - 100 * mx.abs(angle)  # Angle from upright
            + 10 * leg_l  # Left leg contact
            + 10 * leg_r  # Right leg contact
        )

        if self.prev_shaping is not None:
            reward = float(shaping - self.prev_shaping)
        else:
            reward = 0.0
        self.prev_shaping = float(shaping)

        # Penalty for using main engine
        reward -= main_power * 0.3

        # Check termination
        terminated = False
        if on_ground:
            if float(leg_l) > 0 and float(leg_r) > 0:
                # Successful landing
                reward += 100
                terminated = True
            else:
                # Crashed
                reward -= 100
                terminated = True

        # Out of bounds
        if abs(float(x)) > 2.0 or float(y) > 2.0:
            reward -= 100
            terminated = True

        # Increment step counter
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        return self.state, reward, terminated, truncated, {}

    def render(self):
        """Render the current state."""
        if self.render_mode == "human" and self.state is not None:
            x, y, vx, vy, angle, _, leg_l, leg_r = self.state
            print(
                f"Pos: ({float(x):.2f}, {float(y):.2f}), "
                f"Vel: ({float(vx):.2f}, {float(vy):.2f}), "
                f"Angle: {float(angle) * 180 / np.pi:.1f}°, "
                f"Legs: {int(leg_l)}/{int(leg_r)}"
            )

    def close(self):
        """Clean up resources."""
        pass
