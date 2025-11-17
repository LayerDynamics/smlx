"""
Episode recording utilities for SMLX Gym.

This module provides utilities for recording and saving episodes, including
video recording, trajectory logging, and replay functionality.

Reference: SMLX_Gym.md, Section 4.4 (Advanced Features)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mlx.core as mx
import numpy as np


@dataclass
class EpisodeRecording:
    """
    Recording of a single episode.

    Attributes:
        observations: List of observations (MLX arrays or numpy arrays)
        actions: List of actions taken
        rewards: List of rewards received
        infos: List of info dicts
        metadata: Episode metadata (return, length, etc.)
    """

    observations: list[Any]
    actions: list[Any]
    rewards: list[float]
    infos: list[dict[str, Any]]
    metadata: dict[str, Any]

    def save(self, path: str):
        """
        Save episode recording to disk.

        Args:
            path: Path to save recording
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert MLX arrays to numpy for saving
        observations_np = []
        for obs in self.observations:
            if isinstance(obs, mx.array):
                observations_np.append(np.array(obs))
            elif isinstance(obs, np.ndarray):
                observations_np.append(obs)
            else:
                observations_np.append(obs)

        actions_np = []
        for action in self.actions:
            if isinstance(action, mx.array):
                actions_np.append(np.array(action))
            elif isinstance(action, np.ndarray):
                actions_np.append(action)
            else:
                actions_np.append(action)

        # Save as npz file
        np.savez(
            save_path,
            observations=np.array(observations_np, dtype=object),
            actions=np.array(actions_np, dtype=object),
            rewards=np.array(self.rewards),
            infos=np.array(self.infos, dtype=object),
            metadata=np.array([self.metadata], dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "EpisodeRecording":
        """
        Load episode recording from disk.

        Args:
            path: Path to load recording from

        Returns:
            Loaded episode recording
        """
        data = np.load(path, allow_pickle=True)

        return cls(
            observations=list(data["observations"]),
            actions=list(data["actions"]),
            rewards=list(data["rewards"]),
            infos=list(data["infos"]),
            metadata=data["metadata"][0],
        )


class EpisodeRecorder(gym.Wrapper):
    """
    Wrapper for recording episodes.

    Records full episodes including observations, actions, rewards, and
    additional info. Can save recordings to disk for later replay or analysis.

    Example:
        ```python
        import gymnasium as gym
        from smlx.gym.utils.recording import EpisodeRecorder

        # Wrap environment with recorder
        env = gym.make("CartPole-v1")
        env = EpisodeRecorder(env, save_dir="recordings")

        # Episodes are automatically recorded
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                # Recording is automatically saved
                print(f"Episode saved: {env.last_recording_path}")
                break
        ```
    """

    def __init__(
        self,
        env: gym.Env,
        save_dir: Optional[str] = None,
        save_every: int = 1,
        auto_save: bool = True,
    ):
        """
        Initialize episode recorder.

        Args:
            env: Environment to wrap
            save_dir: Directory to save recordings (None for no auto-save)
            save_every: Save every N episodes
            auto_save: Whether to automatically save recordings
        """
        super().__init__(env)

        self.save_dir = Path(save_dir) if save_dir else None
        self.save_every = save_every
        self.auto_save = auto_save

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Current episode recording
        self.current_observations: list[Any] = []
        self.current_actions: list[Any] = []
        self.current_rewards: list[float] = []
        self.current_infos: list[dict[str, Any]] = []

        # Episode tracking
        self.episode_count = 0
        self.recordings: list[EpisodeRecording] = []
        self.last_recording_path: Optional[str] = None

    def reset(self, **kwargs):
        """Reset environment and start new recording."""
        # Save previous episode if it exists
        # (buffers are now cleared in _finalize_episode, so this check still works)
        if len(self.current_observations) > 0:
            self._finalize_episode()

        obs, info = self.env.reset(**kwargs)

        # Record initial observation
        self.current_observations.append(obs)
        self.current_infos.append(info)

        return obs, info

    def step(self, action):
        """Step environment and record transition."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Record transition
        self.current_actions.append(action)
        self.current_rewards.append(float(reward))
        self.current_observations.append(obs)
        self.current_infos.append(info)

        # Finalize episode if done
        if terminated or truncated:
            self._finalize_episode()

        return obs, reward, terminated, truncated, info

    def _finalize_episode(self):
        """Finalize and save current episode recording."""
        if len(self.current_actions) == 0:
            return

        # Compute episode statistics
        episode_return = sum(self.current_rewards)
        episode_length = len(self.current_actions)

        # Create episode recording
        metadata = {
            "episode": self.episode_count,
            "return": episode_return,
            "length": episode_length,
        }

        recording = EpisodeRecording(
            observations=self.current_observations.copy(),
            actions=self.current_actions.copy(),
            rewards=self.current_rewards.copy(),
            infos=self.current_infos.copy(),
            metadata=metadata,
        )

        self.recordings.append(recording)
        self.episode_count += 1

        # Auto-save if enabled
        if self.auto_save and self.save_dir:
            if self.episode_count % self.save_every == 0:
                self.last_recording_path = str(
                    self.save_dir / f"episode_{self.episode_count}.npz"
                )
                recording.save(self.last_recording_path)

        # Clear buffers after finalizing to prevent double-finalization
        self.current_observations = []
        self.current_actions = []
        self.current_rewards = []
        self.current_infos = []

    def get_recordings(self, n: Optional[int] = None) -> list[EpisodeRecording]:
        """
        Get recent episode recordings.

        Args:
            n: Number of recordings to return (None for all)

        Returns:
            List of episode recordings
        """
        if n is None:
            return self.recordings
        else:
            return self.recordings[-n:]

    def clear_recordings(self):
        """Clear all episode recordings from memory."""
        self.recordings = []


class TrajectoryLogger:
    """
    Logger for agent trajectories.

    Logs trajectories to JSON files for analysis and visualization.

    Example:
        ```python
        from smlx.gym.utils.recording import TrajectoryLogger

        logger = TrajectoryLogger("logs/trajectories.jsonl")

        # Log trajectory
        logger.log({
            "episode": 1,
            "return": 195.0,
            "length": 195,
            "observations": [...],
            "actions": [...]
        })

        logger.close()
        ```
    """

    def __init__(self, log_path: str):
        """
        Initialize trajectory logger.

        Args:
            log_path: Path to log file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(self.log_path, "a")

    def log(self, trajectory: dict[str, Any]):
        """
        Log a trajectory.

        Args:
            trajectory: Dictionary containing trajectory data
        """
        # Convert MLX arrays and numpy types to native Python types
        trajectory_serializable = {}
        for key, value in trajectory.items():
            if isinstance(value, mx.array):
                trajectory_serializable[key] = np.array(value).tolist()
            elif isinstance(value, np.ndarray):
                trajectory_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                # Handle numpy scalar types
                trajectory_serializable[key] = value.item()
            elif isinstance(value, list):
                # Convert any arrays in lists
                converted = []
                for item in value:
                    if isinstance(item, (mx.array, np.ndarray)):
                        converted.append(np.array(item).tolist())
                    elif isinstance(item, (np.integer, np.floating)):
                        converted.append(item.item())
                    else:
                        converted.append(item)
                trajectory_serializable[key] = converted
            else:
                trajectory_serializable[key] = value

        # Write as JSON line
        self.file.write(json.dumps(trajectory_serializable) + "\n")
        self.file.flush()

    def close(self):
        """Close log file."""
        self.file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def replay_episode(
    env: gym.Env, recording: EpisodeRecording, render: bool = True, **reset_kwargs
) -> tuple[float, int]:
    """
    Replay a recorded episode in an environment.

    Useful for visualizing recorded episodes or verifying determinism.

    Args:
        env: Environment to replay in
        recording: Episode recording to replay
        render: Whether to render during replay
        **reset_kwargs: Additional keyword arguments for env.reset() (e.g., seed)

    Returns:
        tuple of (total_return, episode_length)

    Example:
        ```python
        import gymnasium as gym
        from smlx.gym.utils.recording import EpisodeRecording, replay_episode

        # Load recording
        recording = EpisodeRecording.load("episode_100.npz")

        # Replay in environment with deterministic seed
        env = gym.make("CartPole-v1", render_mode="human")
        total_return, length = replay_episode(env, recording, render=True, seed=42)
        print(f"Replayed episode: return={total_return}, length={length}")
        ```
    """
    env.reset(**reset_kwargs)

    total_return = 0.0

    for action in recording.actions:
        if render:
            env.render()

        _, reward, terminated, truncated, _ = env.step(action)
        total_return += float(reward)

        if terminated or truncated:
            break

    return total_return, len(recording.actions)


def load_recordings(directory: str) -> list[EpisodeRecording]:
    """
    Load all episode recordings from a directory.

    Args:
        directory: Directory containing recording files

    Returns:
        List of episode recordings

    Example:
        ```python
        from smlx.gym.utils.recording import load_recordings

        # Load all recordings
        recordings = load_recordings("recordings")
        print(f"Loaded {len(recordings)} episodes")

        # Analyze recordings
        returns = [r.metadata["return"] for r in recordings]
        print(f"Average return: {sum(returns) / len(returns)}")
        ```
    """
    recordings = []
    directory_path = Path(directory)

    if not directory_path.exists():
        return recordings

    for file_path in sorted(directory_path.glob("*.npz")):
        try:
            recording = EpisodeRecording.load(str(file_path))
            recordings.append(recording)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    return recordings
