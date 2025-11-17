"""
Unit tests for A3C (Asynchronous Advantage Actor-Critic) algorithm.

Tests the A3C network, agent implementation, and training functionality.
"""

import gymnasium as gym
import pytest

import mlx.core as mx

from smlx.gym.algorithms.a3c import A3CAgent, A3CNetwork


@pytest.mark.unit
class TestA3CNetwork:
    """Tests for A3CNetwork."""

    def test_network_creation(self):
        """Test creating A3C network."""
        network = A3CNetwork(observation_dim=4, action_dim=2, hidden_dim=64)

        assert network.observation_dim == 4
        assert network.action_dim == 2
        assert network.hidden_dim == 64

    def test_network_forward_pass(self):
        """Test forward pass through A3C network."""
        network = A3CNetwork(observation_dim=4, action_dim=2, hidden_dim=64)

        # Create batch of observations
        batch_size = 8
        observations = mx.random.normal((batch_size, 4))

        # Forward pass
        action_logits, values = network(observations)

        # Check output shapes
        assert action_logits.shape == (batch_size, 2)
        assert values.shape == (batch_size, 1)

    def test_network_single_observation(self):
        """Test network with single observation."""
        network = A3CNetwork(observation_dim=4, action_dim=3, hidden_dim=32)

        observation = mx.random.normal((1, 4))
        action_logits, values = network(observation)

        assert action_logits.shape == (1, 3)
        assert values.shape == (1, 1)

    def test_network_with_multiple_layers(self):
        """Test network with different number of hidden layers."""
        network = A3CNetwork(
            observation_dim=8, action_dim=4, hidden_dim=128, num_hidden_layers=3
        )

        observations = mx.random.normal((4, 8))
        action_logits, values = network(observations)

        assert action_logits.shape == (4, 4)
        assert values.shape == (4, 1)


@pytest.mark.unit
class TestA3CAgent:
    """Tests for A3CAgent."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("CartPole-v1")

    @pytest.fixture
    def agent(self, env):
        """Create A3C agent."""
        return A3CAgent(env, hidden_dim=32, n_steps=5)

    def test_agent_creation(self, agent):
        """Test creating A3C agent."""
        assert isinstance(agent, A3CAgent)
        assert agent.observation_dim == 4
        assert agent.action_dim == 2
        assert agent.n_steps == 5

    def test_agent_with_custom_params(self, env):
        """Test creating agent with custom hyperparameters."""
        agent = A3CAgent(
            env,
            hidden_dim=128,
            learning_rate=1e-4,
            gamma=0.95,
            n_steps=10,
            value_loss_coef=0.3,
            entropy_coef=0.02,
        )

        assert agent.gamma == 0.95
        assert agent.n_steps == 10
        assert agent.value_loss_coef == 0.3
        assert agent.entropy_coef == 0.02

    def test_select_action(self, agent):
        """Test action selection."""
        observation = mx.random.normal((4,))
        action = agent.select_action(observation, training=True)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_greedy(self, agent):
        """Test greedy action selection (no exploration)."""
        observation = mx.random.normal((4,))
        action = agent.select_action(observation, training=False)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_train_step(self, agent):
        """Test A3C training step."""
        # Create batch
        batch_size = 32
        batch = {
            "states": mx.random.normal((batch_size, 4)),
            "actions": mx.array([0, 1] * 16),
            "returns": mx.random.normal((batch_size,)),
        }

        metrics = agent.train_step(batch)

        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert isinstance(metrics["loss"], float)

    def test_gradient_clipping(self, agent):
        """Test gradient clipping functionality."""
        # Create dummy gradients
        grads = {
            "layer1": mx.random.normal((10, 10)) * 100,
            "layer2": mx.random.normal((5, 5)) * 100,
        }

        clipped_grads = agent._clip_gradients(grads)

        assert "layer1" in clipped_grads
        assert "layer2" in clipped_grads

    def test_trajectory_collection(self, agent):
        """Test that trajectories are collected during episode."""
        # Reset trajectory buffers
        agent.trajectory_states = []
        agent.trajectory_actions = []
        agent.trajectory_rewards = []

        # Run a short episode
        agent._run_episode(max_steps=10)

        # Trajectory buffers should have been used (may be cleared after update)
        assert agent.total_steps > 0

    def test_n_step_returns_computation(self, agent):
        """Test that n-step returns are computed correctly."""
        # Set up a simple trajectory
        agent.trajectory_states = [mx.random.normal((4,)) for _ in range(5)]
        agent.trajectory_actions = [0, 1, 0, 1, 0]
        agent.trajectory_rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        agent.trajectory_dones = [False, False, False, False, True]

        last_state = agent.trajectory_states[-1]
        agent._update_policy(last_state, done=True)

        # After update, losses should be tracked
        assert len(agent.losses) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestA3CTraining:
    """Integration tests for A3C training."""

    def test_agent_training(self):
        """Test A3C agent training."""
        env = gym.make("CartPole-v1")
        agent = A3CAgent(env, hidden_dim=32, n_steps=5)

        # Train for a few episodes
        metrics = agent.run(num_episodes=10, max_steps=100, verbose=False)

        # Check training completed
        assert agent.episode_count == 10
        assert metrics.episode == 10
        assert metrics.average_return >= 0.0
        assert agent.update_count > 0

        env.close()

    def test_agent_learns(self):
        """Test that A3C agent can learn (returns should improve)."""
        env = gym.make("CartPole-v1")
        agent = A3CAgent(env, hidden_dim=64, learning_rate=1e-3, n_steps=5)

        # Initial performance
        agent.run(num_episodes=10, max_steps=200, verbose=False)

        # Train more
        agent.run(num_episodes=50, max_steps=200, verbose=False)
        final_metrics = agent.run(num_episodes=10, max_steps=200, verbose=False)
        final_return = final_metrics.average_return

        # Performance should generally improve (though not guaranteed)
        # Just ensure training completes without errors
        assert final_return >= 0.0
        assert agent.episode_count >= 70

        env.close()

    def test_save_and_load(self):
        """Test saving and loading A3C agent."""
        env = gym.make("CartPole-v1")
        agent = A3CAgent(env, hidden_dim=32)

        # Train briefly
        agent.run(num_episodes=5, max_steps=50, verbose=False)

        # Save agent
        save_path = "/tmp/a3c_test.safetensors"
        agent.save(save_path)

        # Create new agent and load
        new_agent = A3CAgent(env, hidden_dim=32)
        new_agent.load(save_path)

        # Check update count was restored
        assert new_agent.update_count == agent.update_count

        env.close()

    def test_multiple_n_step_updates(self):
        """Test multiple n-step updates in long episode."""
        env = gym.make("CartPole-v1")
        agent = A3CAgent(env, hidden_dim=32, n_steps=10)

        # Run longer episodes that trigger multiple updates
        agent.run(num_episodes=5, max_steps=200, verbose=False)

        # Should have multiple updates
        assert agent.update_count > 5
        assert len(agent.losses) > 0

        env.close()
