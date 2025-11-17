"""
SMLX Gym - Gymnasium/RL Environment Integrations for SMLX.

This package provides Gymnasium environment integrations and RL algorithms
optimized for Apple Silicon using MLX.

Main Components:
- base: Base environment classes (MLXEnv)
- spaces: MLX-compatible space definitions
- wrappers: Environment wrappers for preprocessing
- replay: Experience replay buffers
- algorithms: RL algorithms (DQN, PPO, A3C)
- curriculum: Curriculum learning utilities
- multi_agent: Multi-agent RL support
- envs: Custom environments (text, vision, audio)
- utils: Recording and visualization utilities

Example:
    ```python
    import gymnasium as gym
    from smlx.gym import make_env_with_wrappers
    from smlx.gym.algorithms import DQNAgent

    # Create environment with wrappers
    env = make_env_with_wrappers("CartPole-v1", normalize_obs=True)

    # Create and train agent
    agent = DQNAgent(env, hidden_dim=128)
    metrics = agent.run(num_episodes=1000)
    ```
"""

# Base environment and spaces
from smlx.gym.base import MLXEnv

# Wrappers
from smlx.gym.wrappers import (
    ClipReward,
    EpisodeLogger,
    FlattenObservation,
    FrameStack,
    MLXObservationWrapper,
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
    RescaleAction,
    TimeLimit,
    make_env_with_wrappers,
)

# Replay buffers
from smlx.gym.replay import (
    EpisodeBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    Transition,
    compute_gae,
    compute_returns,
)

# Curriculum learning
from smlx.gym.curriculum import (
    CurriculumScheduler,
    CurriculumStage,
    CurriculumWrapper,
    ThresholdScheduler,
    create_curriculum_env,
)

# Multi-agent
from smlx.gym.multi_agent import (
    AgentRole,
    MultiAgentConfig,
    MultiAgentEnv,
    ParallelEnvWrapper,
    TeamRewardWrapper,
    create_parallel_envs,
)

__all__ = [
    # Base
    "MLXEnv",
    # Wrappers
    "MLXObservationWrapper",
    "NormalizeObservation",
    "NormalizeReward",
    "ClipReward",
    "FrameStack",
    "RecordEpisodeStatistics",
    "TimeLimit",
    "FlattenObservation",
    "RescaleAction",
    "EpisodeLogger",
    "make_env_with_wrappers",
    # Replay
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "EpisodeBuffer",
    "Transition",
    "compute_returns",
    "compute_gae",
    # Curriculum
    "CurriculumScheduler",
    "CurriculumStage",
    "CurriculumWrapper",
    "ThresholdScheduler",
    "create_curriculum_env",
    # Multi-agent
    "MultiAgentEnv",
    "MultiAgentConfig",
    "AgentRole",
    "ParallelEnvWrapper",
    "TeamRewardWrapper",
    "create_parallel_envs",
]
