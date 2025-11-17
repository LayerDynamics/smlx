"""
Unit tests for base RL agent classes.

Tests the base agent class and configuration for all RL algorithms.
"""

import gymnasium as gym
import mlx.core as mx
import pytest

from smlx.gym.algorithms.base import (
    AgentConfig,
    RandomAgent,
    RLAgent,
    TrainingMetrics,
)


@pytest.mark.unit
class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_config_creation(self):
        """Test creating agent configuration."""
        config = AgentConfig(
            gamma=0.95,
            learning_rate=1e-4,
            max_episodes=500,
            log_interval=5,
        )

        assert config.gamma == 0.95
        assert config.learning_rate == 1e-4
        assert config.max_episodes == 500
        assert config.log_interval == 5

    def test_config_defaults(self):
        """Test default configuration values."""
        config = AgentConfig()

        assert config.gamma == 0.99
        assert config.learning_rate == 1e-3
        assert config.max_episodes == 1000
        assert config.log_interval == 10
        assert config.save_interval == 100
        assert config.eval_interval == 50
        assert config.eval_episodes == 10


@pytest.mark.unit
class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating training metrics."""
        metrics = TrainingMetrics(
            episode=100,
            episode_return=150.5,
            episode_length=200,
            average_return=145.0,
            loss=0.25,
        )

        assert metrics.episode == 100
        assert metrics.episode_return == 150.5
        assert metrics.episode_length == 200
        assert metrics.average_return == 145.0
        assert metrics.loss == 0.25

    def test_metrics_defaults(self):
        """Test default metrics values."""
        metrics = TrainingMetrics()

        assert metrics.episode == 0
        assert metrics.episode_return == 0.0
        assert metrics.episode_length == 0
        assert metrics.average_return == 0.0
        assert metrics.loss == 0.0


@pytest.mark.unit
class TestRLAgent:
    """Tests for RLAgent base class."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("CartPole-v1")

    @pytest.fixture
    def agent(self, env):
        """Create random agent for testing."""
        return RandomAgent(env)

    def test_agent_initialization(self, agent, env):
        """Test agent initialization."""
        assert agent.env == env
        assert isinstance(agent.config, AgentConfig)
        assert agent.episode_count == 0
        assert agent.total_steps == 0
        assert len(agent.metrics_history) == 0

    def test_agent_with_custom_config(self, env):
        """Test agent initialization with custom config."""
        config = AgentConfig(gamma=0.95, learning_rate=1e-4)
        agent = RandomAgent(env, config=config)

        assert agent.config.gamma == 0.95
        assert agent.config.learning_rate == 1e-4

    def test_agent_with_seed(self, env):
        """Test agent initialization with random seed."""
        agent1 = RandomAgent(env, config=AgentConfig(seed=42))
        agent2 = RandomAgent(env, config=AgentConfig(seed=42))

        # Both agents should work with the seed
        assert agent1.config.seed == 42
        assert agent2.config.seed == 42

    def test_run_episode(self, agent):
        """Test running a single episode."""
        episode_return, episode_length, success = agent._run_episode(max_steps=100)

        assert isinstance(episode_return, float)
        assert isinstance(episode_length, int)
        assert isinstance(success, bool)
        assert episode_return >= 0.0
        assert episode_length > 0

    def test_run_multiple_episodes(self, agent):
        """Test running multiple episodes."""
        metrics = agent.run(num_episodes=5, max_steps=100, verbose=False)

        assert agent.episode_count == 5
        assert metrics.episode == 5
        assert metrics.average_return >= 0.0
        assert metrics.average_length > 0.0
        assert metrics.training_time > 0.0

    def test_episode_statistics_tracking(self, agent):
        """Test that episode statistics are tracked correctly."""
        agent.run(num_episodes=10, max_steps=50, verbose=False)

        assert len(agent._episode_returns) > 0
        assert len(agent._episode_lengths) > 0
        assert agent.episode_count == 10

    def test_moving_average_window(self, agent):
        """Test that moving average window is maintained."""
        # Run more episodes than window size
        agent.run(num_episodes=150, max_steps=50, verbose=False)

        # Should maintain only last 100 episodes
        assert len(agent._episode_returns) == 100
        assert len(agent._episode_lengths) == 100

    def test_reset_agent(self, agent):
        """Test agent reset functionality."""
        # Run some episodes
        agent.run(num_episodes=5, max_steps=50, verbose=False)

        # Reset agent
        agent.reset()

        assert agent.episode_count == 0
        assert agent.total_steps == 0
        assert len(agent.metrics_history) == 0
        assert len(agent._episode_returns) == 0

    def test_max_steps_limit(self, agent):
        """Test that max_steps limits episode length."""
        episode_return, episode_length, success = agent._run_episode(max_steps=20)

        assert episode_length <= 20

    def test_abstract_methods_raise_not_implemented(self, env):
        """Test that RLAgent is abstract and cannot be instantiated directly."""
        # RLAgent is an abstract class and cannot be instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            RLAgent(env)

        # Test that a minimal concrete implementation works
        class MinimalAgent(RLAgent):
            def select_action(self, state):
                return env.action_space.sample()

            def train_step(self, batch):
                return {"loss": 0.0}

        # This should work fine
        agent = MinimalAgent(env)
        state = mx.array([1.0, 2.0, 3.0, 4.0])
        action = agent.select_action(state)
        assert action in [0, 1]  # CartPole has 2 actions

        metrics = agent.train_step({})
        assert "loss" in metrics


@pytest.mark.unit
class TestRandomAgent:
    """Tests for RandomAgent."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("CartPole-v1")

    @pytest.fixture
    def agent(self, env):
        """Create random agent."""
        return RandomAgent(env)

    def test_agent_creation(self, agent):
        """Test creating random agent."""
        assert isinstance(agent, RandomAgent)
        assert isinstance(agent, RLAgent)

    def test_select_action_in_action_space(self, agent, env):
        """Test that selected actions are valid."""
        observation = mx.array([1.0, 2.0, 3.0, 4.0])

        for _ in range(20):
            action = agent.select_action(observation)
            assert env.action_space.contains(action)

    def test_train_step_returns_zero_loss(self, agent):
        """Test that random agent has zero loss."""
        batch = {
            "states": mx.random.normal((10, 4)),
            "actions": mx.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            "rewards": mx.random.normal((10,)),
        }

        metrics = agent.train_step(batch)

        assert "loss" in metrics
        assert metrics["loss"] == 0.0

    def test_run_produces_valid_metrics(self, agent):
        """Test running random agent produces valid metrics."""
        metrics = agent.run(num_episodes=10, max_steps=50, verbose=False)

        assert metrics.episode == 10
        assert metrics.average_return >= 0.0
        assert metrics.average_length > 0.0
        assert metrics.training_time >= 0.0

    def test_different_environments(self):
        """Test random agent works with different environments."""
        env_ids = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]

        for env_id in env_ids:
            env = gym.make(env_id)
            agent = RandomAgent(env)

            metrics = agent.run(num_episodes=2, max_steps=50, verbose=False)

            assert metrics.episode == 2
            env.close()


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for RL agents."""

    def test_agent_training_loop(self):
        """Test complete agent training loop."""
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        # Train agent
        metrics = agent.run(num_episodes=20, max_steps=100, verbose=False)

        # Check metrics are reasonable
        assert metrics.episode == 20
        assert metrics.average_return > 0.0
        assert agent.total_steps > 0

        env.close()

    def test_agent_save_and_load(self):
        """Test agent save and load (default implementation does nothing)."""
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        # Default implementation doesn't raise error
        agent.save("test_checkpoint.safetensors")
        agent.load("test_checkpoint.safetensors")

    def test_multiple_agents_same_env_type(self):
        """Test creating multiple agents for same environment."""
        env1 = gym.make("CartPole-v1")
        env2 = gym.make("CartPole-v1")

        agent1 = RandomAgent(env1)
        agent2 = RandomAgent(env2)

        # Both agents should work independently
        metrics1 = agent1.run(num_episodes=5, max_steps=50, verbose=False)
        metrics2 = agent2.run(num_episodes=5, max_steps=50, verbose=False)

        assert metrics1.episode == 5
        assert metrics2.episode == 5

        env1.close()
        env2.close()

    def test_agent_metrics_history_tracking(self):
        """Test that metrics history is properly tracked."""
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        # Run with logging
        config = AgentConfig(log_interval=2)
        agent.config = config
        agent.run(num_episodes=10, max_steps=50, verbose=False)

        # Metrics should be tracked
        assert agent.episode_count == 10
        assert agent.current_metrics.episode == 10

        env.close()
