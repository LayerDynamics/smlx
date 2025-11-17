"""
RL algorithms for SMLX Gym.

This module provides implementations of popular RL algorithms optimized
for Apple Silicon using MLX:

- DQN: Deep Q-Network (value-based, off-policy)
- PPO: Proximal Policy Optimization (policy-based, on-policy)
- A3C: Asynchronous Advantage Actor-Critic (policy-based, on-policy)

All algorithms use MLX for Metal GPU acceleration throughout training.

Example:
    ```python
    import gymnasium as gym
    from smlx.gym.algorithms import DQNAgent, PPOAgent

    env = gym.make("CartPole-v1")

    # DQN for discrete action spaces
    dqn_agent = DQNAgent(env, hidden_dim=128)
    dqn_agent.run(num_episodes=1000)

    # PPO for more complex tasks
    ppo_agent = PPOAgent(env, hidden_dim=64)
    ppo_agent.run(num_episodes=1000)
    ```
"""

# Base classes
from smlx.gym.algorithms.base import AgentConfig, RandomAgent, RLAgent, TrainingMetrics

# DQN
from smlx.gym.algorithms.dqn import DoubleDQNAgent, DQNAgent, QNetwork

# PPO
from smlx.gym.algorithms.ppo import ActorCriticNetwork, PPOAgent

# A3C
from smlx.gym.algorithms.a3c import A3CAgent, A3CNetwork

__all__ = [
    # Base
    "RLAgent",
    "RandomAgent",
    "AgentConfig",
    "TrainingMetrics",
    # DQN
    "DQNAgent",
    "DoubleDQNAgent",
    "QNetwork",
    # PPO
    "PPOAgent",
    "ActorCriticNetwork",
    # A3C
    "A3CAgent",
    "A3CNetwork",
]
