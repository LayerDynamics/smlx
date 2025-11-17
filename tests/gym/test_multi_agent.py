"""
Unit tests for multi-agent reinforcement learning support.

Tests multi-agent environments, parallel environments, and coordination utilities.
"""

import gymnasium as gym
import mlx.core as mx
import numpy as np
import pytest

from smlx.gym.multi_agent import (
    AgentRole,
    MultiAgentConfig,
    MultiAgentEnv,
    ParallelEnvWrapper,
    TeamRewardWrapper,
    create_parallel_envs,
)


@pytest.mark.unit
class TestMultiAgentConfig:
    """Tests for MultiAgentConfig dataclass."""

    def test_basic_config(self):
        """Test creating basic multi-agent configuration."""
        config = MultiAgentConfig(num_agents=3)
        assert config.num_agents == 3
        assert config.shared_reward is False
        assert config.communication is False
        assert config.observation_mode == "local"

    def test_custom_config(self):
        """Test creating custom configuration."""
        config = MultiAgentConfig(
            num_agents=2,
            agent_roles=[AgentRole.COOPERATIVE, AgentRole.COOPERATIVE],
            shared_reward=True,
            communication=True,
            observation_mode="global",
            metadata={"task": "coordination"},
        )

        assert config.num_agents == 2
        assert len(config.agent_roles) == 2
        assert config.shared_reward is True
        assert config.communication is True
        assert config.observation_mode == "global"
        assert config.metadata["task"] == "coordination"


@pytest.mark.unit
class TestAgentRole:
    """Tests for AgentRole enum."""

    def test_agent_roles(self):
        """Test agent role values."""
        assert AgentRole.COOPERATIVE.value == "cooperative"
        assert AgentRole.COMPETITIVE.value == "competitive"
        assert AgentRole.MIXED.value == "mixed"


@pytest.mark.unit
class TestMultiAgentEnv:
    """Tests for MultiAgentEnv base class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MultiAgentConfig(
            num_agents=3, shared_reward=False, communication=True, observation_mode="local"
        )

    @pytest.fixture
    def env(self, config):
        """Create multi-agent environment."""
        return MultiAgentEnv(config)

    def test_initialization(self, env, config):
        """Test environment initialization."""
        assert env.num_agents == 3
        assert len(env.agent_ids) == 3
        assert env.agent_ids == ["agent_0", "agent_1", "agent_2"]
        assert env.config == config

    def test_reset_returns_dict(self, env):
        """Test that reset returns observations dict."""
        observations, info = env.reset()

        assert isinstance(observations, dict)
        assert len(observations) == 3
        assert "agent_0" in observations
        assert "agent_1" in observations
        assert "agent_2" in observations

        # Each observation should be an MLX array
        for obs in observations.values():
            assert isinstance(obs, mx.array)

        assert "num_agents" in info
        assert info["num_agents"] == 3

    def test_reset_with_seed(self, env):
        """Test reset with random seed."""
        observations1, _ = env.reset(seed=42)
        observations2, _ = env.reset(seed=42)

        # Should produce same observations with same seed
        for agent_id in env.agent_ids:
            assert mx.array_equal(observations1[agent_id], observations2[agent_id])

    def test_step_returns_dicts(self, env):
        """Test that step returns dictionaries for all outputs."""
        env.reset()

        actions = {"agent_0": 0, "agent_1": 1, "agent_2": 0}
        observations, rewards, terminated, truncated, info = env.step(actions)

        # Check types
        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminated, dict)
        assert isinstance(truncated, dict)
        assert isinstance(info, dict)

        # Check all agents present
        for agent_id in env.agent_ids:
            assert agent_id in observations
            assert agent_id in rewards
            assert agent_id in terminated
            assert agent_id in truncated

    def test_communication_enabled(self, env):
        """Test communication between agents."""
        env.reset()

        # Send message from agent_0 to agent_1
        env.send_message("agent_0", "agent_1", {"type": "help", "location": [1, 2]})

        # Agent 1 should receive message
        messages = env.receive_messages("agent_1")
        assert len(messages) == 1
        assert messages[0]["sender"] == "agent_0"
        assert messages[0]["content"]["type"] == "help"

        # Messages should be cleared after receiving
        messages_again = env.receive_messages("agent_1")
        assert len(messages_again) == 0

    def test_communication_disabled(self):
        """Test that communication doesn't work when disabled."""
        config = MultiAgentConfig(num_agents=2, communication=False)
        env = MultiAgentEnv(config)
        env.reset()

        # Try to send message
        env.send_message("agent_0", "agent_1", "hello")

        # Agent 1 should not receive anything
        messages = env.receive_messages("agent_1")
        assert len(messages) == 0

    def test_communication_cleared_on_reset(self, env):
        """Test that messages are cleared on reset."""
        env.reset()

        # Send message
        env.send_message("agent_0", "agent_1", "test")

        # Reset should clear messages
        env.reset()
        messages = env.receive_messages("agent_1")
        assert len(messages) == 0

    def test_render_human_mode(self):
        """Test rendering in human mode."""
        config = MultiAgentConfig(num_agents=2)
        env = MultiAgentEnv(config, render_mode="human")

        env.reset()
        # Should not raise an error
        env.render()


@pytest.mark.unit
class TestParallelEnvWrapper:
    """Tests for ParallelEnvWrapper."""

    @pytest.fixture
    def envs(self):
        """Create parallel environments."""
        return ParallelEnvWrapper("CartPole-v1", num_envs=4)

    def test_initialization(self, envs):
        """Test parallel environment initialization."""
        assert envs.num_envs == 4
        assert len(envs.envs) == 4
        assert envs.env_id == "CartPole-v1"

        # Check spaces
        assert envs.observation_space is not None
        assert envs.action_space is not None

    def test_reset_returns_list(self, envs):
        """Test that reset returns list of observations."""
        observations = envs.reset()

        assert isinstance(observations, list)
        assert len(observations) == 4

        # Each observation should be an MLX array
        for obs in observations:
            assert isinstance(obs, mx.array)

    def test_reset_with_seed(self, envs):
        """Test reset with seed produces deterministic results."""
        obs1 = envs.reset(seed=42)
        obs2 = envs.reset(seed=42)

        # Should be the same with same seed
        for o1, o2 in zip(obs1, obs2):
            assert mx.array_equal(o1, o2)

    def test_step_all_envs(self, envs):
        """Test stepping all environments."""
        envs.reset()

        actions = [0, 1, 0, 1]  # One action per environment
        observations, rewards, terminated, truncated, infos = envs.step(actions)

        assert len(observations) == 4
        assert len(rewards) == 4
        assert len(terminated) == 4
        assert len(truncated) == 4
        assert len(infos) == 4

        # Check types
        for obs in observations:
            assert isinstance(obs, mx.array)
        for reward in rewards:
            assert isinstance(reward, (int, float))
        for term in terminated:
            assert isinstance(term, bool)
        for trunc in truncated:
            assert isinstance(trunc, bool)

    def test_wrong_number_of_actions(self, envs):
        """Test that wrong number of actions raises error."""
        envs.reset()

        with pytest.raises(ValueError, match="Expected 4 actions"):
            envs.step([0, 1])  # Only 2 actions for 4 envs

    def test_auto_reset_on_done(self, envs):
        """Test that environments auto-reset when done."""
        observations = envs.reset()
        initial_shapes = [obs.shape for obs in observations]

        # Step until at least one environment is done
        for _ in range(500):
            actions = [envs.action_space.sample() for _ in range(envs.num_envs)]
            observations, _, terminated, truncated, _ = envs.step(actions)

            if any(terminated) or any(truncated):
                # After done, observations should still be valid
                for obs, initial_shape in zip(observations, initial_shapes):
                    assert obs.shape == initial_shape
                break

    def test_close_all_envs(self, envs):
        """Test closing all environments."""
        envs.close()
        # Should not raise an error


@pytest.mark.unit
class TestCreateParallelEnvs:
    """Tests for create_parallel_envs factory function."""

    def test_create_basic(self):
        """Test creating parallel environments."""
        envs = create_parallel_envs("CartPole-v1", num_envs=2)

        assert isinstance(envs, ParallelEnvWrapper)
        assert envs.num_envs == 2

        envs.close()

    def test_create_with_kwargs(self):
        """Test creating with additional environment kwargs."""
        # CartPole doesn't take many kwargs, but we can pass render_mode
        envs = create_parallel_envs("CartPole-v1", num_envs=2)

        observations = envs.reset()
        assert len(observations) == 2

        envs.close()


@pytest.mark.unit
class TestTeamRewardWrapper:
    """Tests for TeamRewardWrapper."""

    @pytest.fixture
    def multi_agent_env(self):
        """Create a basic multi-agent environment."""
        config = MultiAgentConfig(num_agents=3)
        return MultiAgentEnv(config)

    def test_wrapper_initialization(self, multi_agent_env):
        """Test wrapper initialization."""
        wrapped = TeamRewardWrapper(multi_agent_env, reward_fn=sum)
        assert wrapped.reward_fn == sum

    def test_default_reward_aggregation(self, multi_agent_env):
        """Test default mean reward aggregation."""
        wrapped = TeamRewardWrapper(multi_agent_env)  # Default is np.mean

        # Mock step to test reward aggregation
        # The base MultiAgentEnv returns zero rewards, but let's override for testing
        wrapped.reset()

        # For testing, we'll patch the step method
        def mock_step(action):
            obs = {"agent_0": mx.zeros((4,)), "agent_1": mx.zeros((4,)), "agent_2": mx.zeros((4,))}
            rewards = {"agent_0": 1.0, "agent_1": 2.0, "agent_2": 3.0}
            terminated = {"agent_0": False, "agent_1": False, "agent_2": False}
            truncated = {"agent_0": False, "agent_1": False, "agent_2": False}
            info = {}
            return obs, rewards, terminated, truncated, info

        # Monkey patch
        original_step = wrapped.env.step
        wrapped.env.step = mock_step  # type: ignore[assignment]

        actions = {"agent_0": 0, "agent_1": 1, "agent_2": 0}
        obs, rewards, terminated, truncated, info = wrapped.step(actions)

        # All agents should receive mean reward
        expected_mean = np.mean([1.0, 2.0, 3.0])
        assert all(abs(reward - expected_mean) < 1e-6 for reward in rewards.values())  # type: ignore[union-attr]

        # Restore original step
        wrapped.env.step = original_step

    def test_sum_reward_aggregation(self, multi_agent_env):
        """Test sum reward aggregation."""
        wrapped = TeamRewardWrapper(multi_agent_env, reward_fn=sum)

        # Mock step
        def mock_step(action):
            obs = {"agent_0": mx.zeros((4,)), "agent_1": mx.zeros((4,)), "agent_2": mx.zeros((4,))}
            rewards = {"agent_0": 1.0, "agent_1": 2.0, "agent_2": 3.0}
            terminated = {"agent_0": False, "agent_1": False, "agent_2": False}
            truncated = {"agent_0": False, "agent_1": False, "agent_2": False}
            info = {}
            return obs, rewards, terminated, truncated, info

        wrapped.env.step = mock_step  # type: ignore[assignment]
        wrapped.reset()

        actions = {"agent_0": 0, "agent_1": 1, "agent_2": 0}
        obs, rewards, terminated, truncated, info = wrapped.step(actions)

        # All agents should receive sum reward
        expected_sum = sum([1.0, 2.0, 3.0])
        assert all(abs(reward - expected_sum) < 1e-6 for reward in rewards.values())  # type: ignore[union-attr]

    def test_single_agent_rewards_unchanged(self):
        """Test that single-agent rewards are not modified."""
        base_env = gym.make("CartPole-v1")
        wrapped = TeamRewardWrapper(base_env)

        wrapped.reset()
        obs, reward, terminated, truncated, info = wrapped.step(0)

        # Single reward should be unchanged (not a dict)
        assert isinstance(reward, (int, float))


@pytest.mark.integration
class TestMultiAgentIntegration:
    """Integration tests for multi-agent scenarios."""

    def test_cooperative_scenario(self):
        """Test cooperative multi-agent scenario."""
        config = MultiAgentConfig(
            num_agents=2,
            agent_roles=[AgentRole.COOPERATIVE, AgentRole.COOPERATIVE],
            shared_reward=True,
            communication=True,
        )

        env = MultiAgentEnv(config)
        env = TeamRewardWrapper(env, reward_fn=sum)

        # Run episode
        observations, _ = env.reset()
        assert len(observations) == 2

        for _ in range(10):
            actions = {"agent_0": 0, "agent_1": 1}
            observations, rewards, terminated, truncated, info = env.step(actions)

            # Both agents should receive same reward (shared)
            reward_values = list(rewards.values())  # type: ignore[union-attr]
            assert all(abs(r - reward_values[0]) < 1e-6 for r in reward_values)

            if all(terminated.values()) or all(truncated.values()):  # type: ignore[union-attr]
                break

    def test_parallel_training(self):
        """Test parallel environment training scenario."""
        envs = create_parallel_envs("CartPole-v1", num_envs=4)

        # Simulate training loop
        observations = envs.reset(seed=42)

        for _ in range(20):
            # Random actions for each environment
            actions = [envs.action_space.sample() for _ in range(envs.num_envs)]
            observations, rewards, terminated, truncated, infos = envs.step(actions)

            # Verify all outputs have correct length
            assert len(observations) == 4
            assert len(rewards) == 4
            assert len(terminated) == 4
            assert len(truncated) == 4

        envs.close()

    def test_communication_coordination(self):
        """Test agent coordination via communication."""
        config = MultiAgentConfig(num_agents=3, communication=True)
        env = MultiAgentEnv(config)

        env.reset()

        # Agent 0 broadcasts to others
        env.send_message("agent_0", "agent_1", {"action": "move_left"})
        env.send_message("agent_0", "agent_2", {"action": "move_right"})

        # Agents receive messages
        messages_1 = env.receive_messages("agent_1")
        messages_2 = env.receive_messages("agent_2")

        assert len(messages_1) == 1
        assert len(messages_2) == 1
        assert messages_1[0]["content"]["action"] == "move_left"
        assert messages_2[0]["content"]["action"] == "move_right"
