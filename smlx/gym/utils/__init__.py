"""
Utility modules for SMLX Gym.

This module provides utilities for:
- Recording episodes and trajectories
- Visualizing training progress and agent behavior

Example:
    ```python
    from smlx.gym.utils import EpisodeRecorder, plot_training_metrics

    # Record episodes
    env = EpisodeRecorder(env, save_dir="recordings")

    # Visualize training
    plot_training_metrics(episodes, returns, losses)
    ```
"""

# Recording utilities
from smlx.gym.utils.recording import (
    EpisodeRecorder,
    EpisodeRecording,
    TrajectoryLogger,
    load_recordings,
    replay_episode,
)

# Visualization utilities
from smlx.gym.utils.visualization import (
    create_training_dashboard,
    plot_episode_trajectory,
    plot_policy_distribution,
    plot_reward_distribution,
    plot_training_metrics,
    plot_value_function,
)

__all__ = [
    # Recording
    "EpisodeRecorder",
    "EpisodeRecording",
    "TrajectoryLogger",
    "replay_episode",
    "load_recordings",
    # Visualization
    "plot_training_metrics",
    "plot_episode_trajectory",
    "plot_value_function",
    "plot_policy_distribution",
    "plot_reward_distribution",
    "create_training_dashboard",
]
