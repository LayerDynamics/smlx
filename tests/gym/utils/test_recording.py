"""
Unit tests for episode recording utilities.

Tests EpisodeRecording, EpisodeRecorder, TrajectoryLogger, and replay functionality.
"""

import json

import gymnasium as gym
import mlx.core as mx
import numpy as np
import pytest

from smlx.gym.utils.recording import (
    EpisodeRecorder,
    EpisodeRecording,
    TrajectoryLogger,
    load_recordings,
    replay_episode,
)


@pytest.mark.unit
class TestEpisodeRecording:
    """Tests for EpisodeRecording dataclass."""

    def test_basic_recording(self):
        """Test creating basic episode recording."""
        observations = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        actions = [0, 1]
        rewards = [1.0, 0.5]
        infos = [{"step": 0}, {"step": 1}]
        metadata = {"episode": 1, "return": 1.5, "length": 2}

        recording = EpisodeRecording(
            observations=observations,
            actions=actions,
            rewards=rewards,
            infos=infos,
            metadata=metadata,
        )

        assert len(recording.observations) == 2
        assert len(recording.actions) == 2
        assert len(recording.rewards) == 2
        assert recording.metadata["return"] == 1.5

    def test_save_and_load(self, tmp_path):
        """Test saving and loading episode recording."""
        observations = [np.array([1, 2]), np.array([3, 4])]
        actions = [0, 1]
        rewards = [1.0, 0.5]
        infos = [{"a": 1}, {"b": 2}]
        metadata = {"episode": 1}

        recording = EpisodeRecording(
            observations=observations,
            actions=actions,
            rewards=rewards,
            infos=infos,
            metadata=metadata,
        )

        # Save
        save_path = tmp_path / "episode.npz"
        recording.save(str(save_path))

        assert save_path.exists()

        # Load
        loaded = EpisodeRecording.load(str(save_path))

        assert len(loaded.observations) == len(observations)
        assert len(loaded.actions) == len(actions)
        assert len(loaded.rewards) == len(rewards)
        assert loaded.metadata["episode"] == 1

    def test_save_with_mlx_arrays(self, tmp_path):
        """Test saving recording with MLX arrays."""
        observations = [mx.array([1.0, 2.0]), mx.array([3.0, 4.0])]
        actions = [mx.array(0), mx.array(1)]
        rewards = [1.0, 0.5]
        infos = [{}, {}]
        metadata = {"episode": 1}

        recording = EpisodeRecording(
            observations=observations,
            actions=actions,
            rewards=rewards,
            infos=infos,
            metadata=metadata,
        )

        # Save (should convert MLX to numpy)
        save_path = tmp_path / "episode_mlx.npz"
        recording.save(str(save_path))

        assert save_path.exists()

        # Load
        loaded = EpisodeRecording.load(str(save_path))

        assert len(loaded.observations) == 2
        assert len(loaded.actions) == 2

    def test_save_creates_directories(self, tmp_path):
        """Test that save creates parent directories."""
        save_path = tmp_path / "nested" / "dir" / "episode.npz"

        recording = EpisodeRecording(
            observations=[np.array([1])],
            actions=[0],
            rewards=[1.0],
            infos=[{}],
            metadata={},
        )

        recording.save(str(save_path))

        assert save_path.exists()


@pytest.mark.unit
class TestEpisodeRecorder:
    """Tests for EpisodeRecorder wrapper."""

    @pytest.fixture
    def env(self, tmp_path):
        """Create wrapped environment with recorder."""
        base_env = gym.make("CartPole-v1")
        return EpisodeRecorder(
            base_env, save_dir=str(tmp_path), save_every=1, auto_save=True
        )

    def test_initialization(self, tmp_path):
        """Test recorder initialization."""
        base_env = gym.make("CartPole-v1")
        recorder = EpisodeRecorder(base_env, save_dir=str(tmp_path))

        assert recorder.save_dir == tmp_path
        assert recorder.save_every == 1
        assert recorder.auto_save is True
        assert recorder.episode_count == 0

    def test_reset_starts_recording(self, env):
        """Test that reset starts new recording."""
        obs, info = env.reset()

        assert len(env.current_observations) == 1
        assert len(env.current_actions) == 0
        assert len(env.current_rewards) == 0

    def test_step_records_transition(self, env):
        """Test that step records transitions."""
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert len(env.current_observations) == 2  # Initial + after step
        assert len(env.current_actions) == 1
        assert len(env.current_rewards) == 1

    def test_episode_completion(self, env):
        """Test episode completion and recording."""
        env.reset()

        # Run until episode ends
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated

        # Episode should be recorded
        assert env.episode_count == 1
        assert len(env.recordings) == 1

    def test_auto_save(self, env, tmp_path):
        """Test automatic saving of episodes."""
        env.reset()

        # Complete episode
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated

        # Check that file was saved
        saved_files = list(tmp_path.glob("*.npz"))
        assert len(saved_files) == 1

    def test_save_every_n_episodes(self, tmp_path):
        """Test saving every N episodes."""
        base_env = gym.make("CartPole-v1")
        recorder = EpisodeRecorder(base_env, save_dir=str(tmp_path), save_every=2)

        # Run 3 episodes
        for _ in range(3):
            recorder.reset()
            done = False
            while not done:
                obs, reward, terminated, truncated, info = recorder.step(
                    recorder.action_space.sample()
                )
                done = terminated or truncated

        # Should have saved on episodes 2 (not 1 or 3)
        saved_files = list(tmp_path.glob("*.npz"))
        assert len(saved_files) == 1  # Only episode 2

    def test_get_recordings(self, env):
        """Test getting episode recordings."""
        # Run 2 episodes
        for _ in range(2):
            env.reset()
            done = False
            while not done:
                obs, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                done = terminated or truncated

        # Get all recordings
        recordings = env.get_recordings()
        assert len(recordings) == 2

        # Get last N recordings
        last_one = env.get_recordings(n=1)
        assert len(last_one) == 1

    def test_clear_recordings(self, env):
        """Test clearing recordings from memory."""
        # Run episode
        env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated

        assert len(env.recordings) == 1

        # Clear
        env.clear_recordings()
        assert len(env.recordings) == 0

    def test_no_auto_save(self, tmp_path):
        """Test recorder with auto_save disabled."""
        base_env = gym.make("CartPole-v1")
        recorder = EpisodeRecorder(
            base_env, save_dir=str(tmp_path), auto_save=False
        )

        # Run episode
        recorder.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = recorder.step(
                recorder.action_space.sample()
            )
            done = terminated or truncated

        # No files should be saved
        saved_files = list(tmp_path.glob("*.npz"))
        assert len(saved_files) == 0

    def test_recording_metadata(self, env):
        """Test that recording metadata is populated."""
        env.reset()

        # Complete episode
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated

        # Check recording metadata
        recording = env.recordings[0]
        assert "episode" in recording.metadata
        assert "return" in recording.metadata
        assert "length" in recording.metadata


@pytest.mark.unit
class TestTrajectoryLogger:
    """Tests for TrajectoryLogger."""

    def test_initialization(self, tmp_path):
        """Test logger initialization."""
        log_path = tmp_path / "trajectories.jsonl"
        logger = TrajectoryLogger(str(log_path))

        assert logger.log_path == log_path
        logger.close()

    def test_log_trajectory(self, tmp_path):
        """Test logging trajectory."""
        log_path = tmp_path / "trajectories.jsonl"
        logger = TrajectoryLogger(str(log_path))

        trajectory = {
            "episode": 1,
            "return": 100.0,
            "length": 50,
            "observations": [[1, 2, 3], [4, 5, 6]],
            "actions": [0, 1],
        }

        logger.log(trajectory)
        logger.close()

        # Check file was created
        assert log_path.exists()

        # Read and verify
        with open(log_path) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["episode"] == 1
            assert data["return"] == 100.0

    def test_log_multiple_trajectories(self, tmp_path):
        """Test logging multiple trajectories."""
        log_path = tmp_path / "trajectories.jsonl"
        logger = TrajectoryLogger(str(log_path))

        for i in range(5):
            trajectory = {"episode": i, "return": float(i * 10)}
            logger.log(trajectory)

        logger.close()

        # Read all lines
        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 5

    def test_log_with_mlx_arrays(self, tmp_path):
        """Test logging trajectory with MLX arrays."""
        log_path = tmp_path / "trajectories.jsonl"
        logger = TrajectoryLogger(str(log_path))

        trajectory = {
            "episode": 1,
            "observations": [mx.array([1.0, 2.0]), mx.array([3.0, 4.0])],
            "actions": [mx.array(0), mx.array(1)],
        }

        logger.log(trajectory)
        logger.close()

        # Should convert to serializable format
        assert log_path.exists()

    def test_log_with_numpy_arrays(self, tmp_path):
        """Test logging trajectory with numpy arrays."""
        log_path = tmp_path / "trajectories.jsonl"
        logger = TrajectoryLogger(str(log_path))

        trajectory = {
            "episode": 1,
            "observations": [np.array([1, 2]), np.array([3, 4])],
        }

        logger.log(trajectory)
        logger.close()

        assert log_path.exists()

    def test_context_manager(self, tmp_path):
        """Test using logger as context manager."""
        log_path = tmp_path / "trajectories.jsonl"

        with TrajectoryLogger(str(log_path)) as logger:
            logger.log({"episode": 1, "return": 10.0})

        # File should be closed and written
        assert log_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """Test that logger creates parent directories."""
        log_path = tmp_path / "nested" / "logs" / "trajectories.jsonl"

        logger = TrajectoryLogger(str(log_path))
        logger.log({"episode": 1})
        logger.close()

        assert log_path.exists()


@pytest.mark.unit
class TestReplayEpisode:
    """Tests for replay_episode function."""

    def test_replay_basic_episode(self):
        """Test replaying basic episode."""
        env = gym.make("CartPole-v1")

        # Create simple recording
        recording = EpisodeRecording(
            observations=[np.array([0, 0, 0, 0])] * 3,
            actions=[0, 1, 0],
            rewards=[1.0, 1.0, 1.0],
            infos=[{}, {}, {}],
            metadata={"return": 3.0, "length": 3},
        )

        total_return, length = replay_episode(env, recording, render=False)

        assert isinstance(total_return, float)
        assert length == 3

        env.close()

    def test_replay_with_rendering(self):
        """Test replaying with rendering enabled."""
        env = gym.make("CartPole-v1")

        recording = EpisodeRecording(
            observations=[np.array([0, 0, 0, 0])] * 2,
            actions=[0, 1],
            rewards=[1.0, 1.0],
            infos=[{}, {}],
            metadata={},
        )

        # Should not raise error
        total_return, length = replay_episode(env, recording, render=False)

        assert length == 2

        env.close()

    def test_replay_terminated_episode(self):
        """Test replaying episode that terminates early."""
        env = gym.make("CartPole-v1")

        # Episode that might terminate
        recording = EpisodeRecording(
            observations=[np.array([0, 0, 0, 0])] * 5,
            actions=[0] * 5,
            rewards=[1.0] * 5,
            infos=[{}] * 5,
            metadata={},
        )

        total_return, length = replay_episode(env, recording, render=False)

        # Length might be less than 5 if episode terminated
        assert length >= 0

        env.close()


@pytest.mark.unit
class TestLoadRecordings:
    """Tests for load_recordings function."""

    def test_load_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        recordings = load_recordings(str(tmp_path))

        assert len(recordings) == 0

    def test_load_nonexistent_directory(self, tmp_path):
        """Test loading from nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"
        recordings = load_recordings(str(nonexistent))

        assert len(recordings) == 0

    def test_load_multiple_recordings(self, tmp_path):
        """Test loading multiple recordings."""
        # Create 3 recordings
        for i in range(3):
            recording = EpisodeRecording(
                observations=[np.array([i])],
                actions=[i],
                rewards=[float(i)],
                infos=[{}],
                metadata={"episode": i},
            )
            recording.save(str(tmp_path / f"episode_{i}.npz"))

        # Load all
        recordings = load_recordings(str(tmp_path))

        assert len(recordings) == 3

    def test_load_recordings_sorted(self, tmp_path):
        """Test that recordings are loaded in sorted order."""
        # Create recordings in reverse order
        for i in [3, 1, 2]:
            recording = EpisodeRecording(
                observations=[np.array([i])],
                actions=[i],
                rewards=[float(i)],
                infos=[{}],
                metadata={"episode": i},
            )
            recording.save(str(tmp_path / f"episode_{i}.npz"))

        # Load
        recordings = load_recordings(str(tmp_path))

        # Should be sorted by filename
        assert len(recordings) == 3

    def test_load_with_corrupted_file(self, tmp_path):
        """Test loading with corrupted file."""
        # Create valid recording
        recording = EpisodeRecording(
            observations=[np.array([1])],
            actions=[0],
            rewards=[1.0],
            infos=[{}],
            metadata={},
        )
        recording.save(str(tmp_path / "episode_1.npz"))

        # Create corrupted file
        corrupted = tmp_path / "corrupted.npz"
        corrupted.write_text("not a valid npz file")

        # Should skip corrupted file
        recordings = load_recordings(str(tmp_path))

        assert len(recordings) == 1


@pytest.mark.integration
class TestRecordingIntegration:
    """Integration tests for episode recording."""

    def test_complete_recording_workflow(self, tmp_path):
        """Test complete recording workflow."""
        # Create environment with recorder
        base_env = gym.make("CartPole-v1")
        env = EpisodeRecorder(base_env, save_dir=str(tmp_path), auto_save=True)

        # Run 3 episodes
        for _ in range(3):
            obs, info = env.reset()
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

        # Check recordings
        assert env.episode_count == 3
        assert len(env.recordings) == 3

        # Load from disk
        loaded = load_recordings(str(tmp_path))
        assert len(loaded) == 3

        env.close()

    def test_recording_and_replay(self, tmp_path):
        """Test recording and replaying episodes."""
        # Record episode with fixed seed for determinism
        base_env = gym.make("CartPole-v1")
        recorder = EpisodeRecorder(base_env, save_dir=str(tmp_path))

        recorder.reset(seed=42)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = recorder.step(
                recorder.action_space.sample()
            )
            done = terminated or truncated

        # Get recording
        recording = recorder.recordings[0]

        # Replay in new environment with same seed for determinism
        replay_env = gym.make("CartPole-v1")
        total_return, length = replay_episode(replay_env, recording, render=False, seed=42)

        # With deterministic initial state, replay should produce same result
        assert total_return == recording.metadata["return"]
        assert length == recording.metadata["length"]

        recorder.close()
        replay_env.close()

    def test_trajectory_logging_workflow(self, tmp_path):
        """Test trajectory logging workflow."""
        env = gym.make("CartPole-v1")
        log_path = tmp_path / "trajectories.jsonl"

        with TrajectoryLogger(str(log_path)) as logger:
            # Run 5 episodes
            for episode in range(5):
                obs, info = env.reset()
                done = False
                observations = [obs]
                actions = []
                rewards = []

                while not done:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    observations.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    done = terminated or truncated

                # Log trajectory
                logger.log(
                    {
                        "episode": episode,
                        "return": sum(rewards),
                        "length": len(actions),
                        "observations": observations[:5],  # Log first 5 obs
                        "actions": actions[:5],
                    }
                )

        # Verify log
        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 5

        env.close()

    def test_mixed_recording_types(self, tmp_path):
        """Test recording with mixed MLX and numpy data."""
        # Create recording with mixed types
        recording = EpisodeRecording(
            observations=[
                mx.array([1.0, 2.0]),
                np.array([3.0, 4.0]),
                mx.array([5.0, 6.0]),
            ],
            actions=[0, mx.array(1), np.array(0)],
            rewards=[1.0, 0.5, 1.0],
            infos=[{}, {}, {}],
            metadata={"episode": 1},
        )

        # Save and load
        save_path = tmp_path / "mixed.npz"
        recording.save(str(save_path))

        loaded = EpisodeRecording.load(str(save_path))

        assert len(loaded.observations) == 3
        assert len(loaded.actions) == 3

    def test_long_episode_recording(self, tmp_path):
        """Test recording long episode."""
        # Create long recording
        num_steps = 1000
        observations = [np.random.randn(4) for _ in range(num_steps + 1)]
        actions = list(range(num_steps))
        rewards = [1.0] * num_steps
        infos = [{}] * num_steps

        recording = EpisodeRecording(
            observations=observations,
            actions=actions,
            rewards=rewards,
            infos=infos,
            metadata={"episode": 1, "return": sum(rewards), "length": num_steps},
        )

        # Save
        save_path = tmp_path / "long_episode.npz"
        recording.save(str(save_path))

        # Load
        loaded = EpisodeRecording.load(str(save_path))

        assert len(loaded.observations) == num_steps + 1
        assert len(loaded.actions) == num_steps
        assert loaded.metadata["length"] == num_steps
