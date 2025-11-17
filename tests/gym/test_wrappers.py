"""
Unit tests for environment wrappers.

Tests various wrappers that modify environment behavior.
"""

import gymnasium as gym
import mlx.core as mx
import numpy as np
import pytest

from smlx.gym.wrappers import (
    ClipReward,
    EpisodeLogger,
    FrameStack,
    MLXObservationWrapper,
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
    TimeLimit,
    make_env_with_wrappers,
)


@pytest.mark.unit
class TestMLXObservationWrapper:
    """Tests for MLXObservationWrapper."""

    @pytest.fixture
    def env(self):
        """Create wrapped environment."""
        base_env = gym.make("CartPole-v1")
        return MLXObservationWrapper(base_env)

    def test_observation_is_mlx_array(self, env):
        """Test that observations are converted to MLX arrays."""
        observation, _ = env.reset()
        assert isinstance(observation, mx.array)

    def test_step_observation_is_mlx_array(self, env):
        """Test that step observations are MLX arrays."""
        env.reset()
        observation, _, _, _, _ = env.step(0)
        assert isinstance(observation, mx.array)

    def test_observation_shape_preserved(self, env):
        """Test that observation shape is preserved."""
        base_env = env.env
        base_obs, _ = base_env.reset()

        env.reset()
        wrapped_obs, _ = env.reset()

        assert wrapped_obs.shape == base_obs.shape

    def test_multiple_types(self):
        """Test conversion from various types."""
        env = gym.make("CartPole-v1")
        wrapper = MLXObservationWrapper(env)

        # NumPy array
        obs_np = np.array([1.0, 2.0, 3.0, 4.0])
        result = wrapper.observation(obs_np)
        assert isinstance(result, mx.array)

        # List
        obs_list = [1.0, 2.0, 3.0, 4.0]
        result = wrapper.observation(obs_list)
        assert isinstance(result, mx.array)

        # Already MLX array
        obs_mlx = mx.array([1.0, 2.0, 3.0, 4.0])
        result = wrapper.observation(obs_mlx)
        assert isinstance(result, mx.array)


@pytest.mark.unit
class TestNormalizeObservation:
    """Tests for NormalizeObservation wrapper."""

    @pytest.fixture
    def env(self):
        """Create wrapped environment."""
        base_env = gym.make("CartPole-v1")
        return NormalizeObservation(base_env)

    def test_normalization_updates(self, env):
        """Test that running statistics update."""
        observation, _ = env.reset()
        # MLX arrays don't have .copy(), use mx.array() constructor instead
        initial_mean = mx.array(env.running_mean) if env.running_mean is not None else None

        for _ in range(10):
            observation, _, _, _, _ = env.step(env.action_space.sample())

        # Mean should have changed
        if initial_mean is not None:
            assert not np.allclose(np.array(env.running_mean), np.array(initial_mean))

    def test_observation_normalized(self, env):
        """Test that observations have reasonable scale."""
        # Run several steps to stabilize normalization
        env.reset()
        for _ in range(20):
            observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()

        # Observations should be roughly normalized
        # (mean close to 0, but may not be exact due to running stats)
        obs_values = np.array(observation)
        assert np.abs(np.mean(obs_values)) < 5.0  # Rough check


@pytest.mark.unit
class TestNormalizeReward:
    """Tests for NormalizeReward wrapper."""

    @pytest.fixture
    def env(self):
        """Create wrapped environment."""
        base_env = gym.make("CartPole-v1")
        return NormalizeReward(base_env, gamma=0.99)

    def test_reward_normalization(self, env):
        """Test that rewards are normalized."""
        env.reset()

        rewards = []
        for _ in range(10):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            rewards.append(reward)
            if terminated or truncated:
                break

        # Rewards should be normalized (smaller magnitude)
        assert all(abs(r) < 10.0 for r in rewards)

    def test_return_resets(self, env):
        """Test that return resets between episodes."""
        env.reset()
        initial_return = env.return_val

        # Run episode
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break

        # Reset
        env.reset()
        assert env.return_val == 0.0


@pytest.mark.unit
class TestClipReward:
    """Tests for ClipReward wrapper."""

    @pytest.fixture
    def env(self):
        """Create wrapped environment with clipping."""
        base_env = gym.make("CartPole-v1")
        return ClipReward(base_env, min_reward=-0.5, max_reward=0.5)

    def test_rewards_clipped(self, env):
        """Test that rewards are clipped to range."""
        env.reset()

        for _ in range(10):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert -0.5 <= reward <= 0.5
            if terminated or truncated:
                break


@pytest.mark.unit
class TestFrameStack:
    """Tests for FrameStack wrapper."""

    @pytest.fixture
    def env(self):
        """Create wrapped environment with frame stacking."""
        base_env = gym.make("CartPole-v1")
        return FrameStack(base_env, num_stack=4)

    def test_observation_shape_increased(self, env):
        """Test that observation shape is multiplied by num_stack."""
        base_env = env.env
        base_shape = base_env.observation_space.shape[0]

        wrapped_shape = env.observation_space.shape[0]
        assert wrapped_shape == base_shape * 4

    def test_frames_stacked(self, env):
        """Test that frames are actually stacked."""
        observation, _ = env.reset()
        assert observation.shape[0] == 4 * 4  # 4 observations * 4 stack

    def test_frames_update(self, env):
        """Test that frame buffer updates with new observations."""
        obs1, _ = env.reset()
        obs2, _, _, _, _ = env.step(0)

        # Observations should be different
        assert not mx.array_equal(obs1, obs2)


@pytest.mark.unit
class TestRecordEpisodeStatistics:
    """Tests for RecordEpisodeStatistics wrapper."""

    @pytest.fixture
    def env(self):
        """Create wrapped environment."""
        base_env = gym.make("CartPole-v1")
        return RecordEpisodeStatistics(base_env)

    def test_episode_info_added(self, env):
        """Test that episode info is added to info dict."""
        observation, _ = env.reset()
        done = False

        while not done:
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            done = terminated or truncated

        assert "episode" in info
        assert "r" in info["episode"]  # return
        assert "l" in info["episode"]  # length

    def test_statistics_tracked(self, env):
        """Test that statistics are tracked."""
        # Run a few episodes
        for _ in range(3):
            env.reset()
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                done = terminated or truncated

        stats = env.get_episode_statistics()
        assert stats["num_episodes"] == 3
        assert "mean_return" in stats
        assert "mean_length" in stats

    def test_return_accumulation(self, env):
        """Test that returns are accumulated correctly."""
        env.reset()
        expected_return = 0.0
        done = False

        while not done:
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            expected_return += reward
            done = terminated or truncated

        assert abs(info["episode"]["r"] - expected_return) < 1e-6


@pytest.mark.unit
class TestTimeLimit:
    """Tests for TimeLimit wrapper."""

    @pytest.fixture
    def env(self):
        """Create wrapped environment with time limit."""
        base_env = gym.make("CartPole-v1")
        return TimeLimit(base_env, max_episode_steps=50)

    def test_episode_truncates(self, env):
        """Test that episode truncates at max steps."""
        env.reset()

        for step in range(100):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                if truncated:
                    assert step + 1 <= 50
                break

    def test_step_counter_resets(self, env):
        """Test that step counter resets."""
        env.reset()
        assert env.current_step == 0

        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break

        env.reset()
        assert env.current_step == 0


@pytest.mark.unit
class TestMakeEnvWithWrappers:
    """Tests for make_env_with_wrappers factory function."""

    def test_create_basic_env(self):
        """Test creating environment with basic wrappers."""
        env = make_env_with_wrappers(
            "CartPole-v1",
            normalize_obs=False,
            normalize_reward=False,
            record_stats=False,
        )

        # Should at least have MLXObservationWrapper
        observation, _ = env.reset()
        assert isinstance(observation, mx.array)
        env.close()

    def test_create_with_normalization(self):
        """Test creating environment with normalization."""
        env = make_env_with_wrappers(
            "CartPole-v1", normalize_obs=True, normalize_reward=True
        )

        # Run a few steps
        env.reset()
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break

        env.close()

    def test_create_with_time_limit(self):
        """Test creating environment with time limit."""
        env = make_env_with_wrappers("CartPole-v1", time_limit=20)

        env.reset()
        for _ in range(30):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break

        env.close()

    def test_create_with_frame_stack(self):
        """Test creating environment with frame stacking."""
        env = make_env_with_wrappers("CartPole-v1", frame_stack=4)

        observation, _ = env.reset()
        base_dim = 4  # CartPole has 4D observation
        assert observation.shape[0] == base_dim * 4

        env.close()

    def test_create_with_all_wrappers(self):
        """Test creating environment with all wrappers."""
        env = make_env_with_wrappers(
            "CartPole-v1",
            normalize_obs=True,
            normalize_reward=True,
            clip_reward=True,
            frame_stack=2,
            time_limit=100,
            record_stats=True,
        )

        # Run episode
        observation, _ = env.reset()
        assert isinstance(observation, mx.array)

        for _ in range(10):
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                break

        env.close()


@pytest.mark.unit
class TestEpisodeLogger:
    """Tests for EpisodeLogger wrapper."""

    @pytest.fixture
    def env(self):
        """Create wrapped environment with logger."""
        base_env = gym.make("CartPole-v1")
        return EpisodeLogger(base_env, log_every=1)

    def test_episode_counter_increments(self, env):
        """Test that episode counter increments."""
        initial_count = env.episode_count

        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated

        assert env.episode_count == initial_count + 1

    def test_return_tracked(self, env):
        """Test that episode return is tracked."""
        env.reset()
        expected_return = 0.0
        done = False

        while not done:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            expected_return += reward
            done = terminated or truncated

        assert abs(env.episode_return - expected_return) < 1e-6


@pytest.mark.integration
class TestWrapperChaining:
    """Integration tests for chaining multiple wrappers."""

    def test_multiple_wrappers(self):
        """Test chaining multiple wrappers together."""
        env = gym.make("CartPole-v1")
        env = RecordEpisodeStatistics(env)
        env = NormalizeObservation(env)
        env = NormalizeReward(env)
        env = MLXObservationWrapper(env)

        # Run episode
        observation, _ = env.reset()
        assert isinstance(observation, mx.array)

        done = False
        while not done:
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            done = terminated or truncated

        # Should have episode statistics
        assert "episode" in info

        env.close()

    def test_wrapper_order_matters(self):
        """Test that wrapper order affects behavior."""
        # MLX wrapper before normalization
        env1 = gym.make("CartPole-v1")
        env1 = MLXObservationWrapper(env1)
        env1 = NormalizeObservation(env1)

        # MLX wrapper after normalization
        env2 = gym.make("CartPole-v1")
        env2 = NormalizeObservation(env2)
        env2 = MLXObservationWrapper(env2)

        # Both should work but may have slightly different behavior
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        assert isinstance(obs1, mx.array)
        assert isinstance(obs2, mx.array)

        env1.close()
        env2.close()
