"""
Classic control environments with MLX support.

MLX-native implementations of classic RL control tasks:
- CartPole: Balance a pole on a cart
- MountainCar: Drive an underpowered car up a mountain
- LunarLander: Land a spacecraft on the moon

All environments use MLX arrays for Metal GPU acceleration.

Example:
    ```python
    from smlx.gym.envs.classic import CartPoleEnv, MountainCarEnv

    # Create environments
    cartpole = CartPoleEnv()
    mountain_car = MountainCarEnv()

    # Train agents
    ```
"""

from smlx.gym.envs.classic.cartpole import CartPoleEnv
from smlx.gym.envs.classic.lunar_lander import LunarLanderEnv
from smlx.gym.envs.classic.mountain_car import MountainCarContinuousEnv, MountainCarEnv

__all__ = [
    "CartPoleEnv",
    "MountainCarEnv",
    "MountainCarContinuousEnv",
    "LunarLanderEnv",
]
