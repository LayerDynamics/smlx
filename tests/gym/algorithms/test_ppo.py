"""
Unit tests for PPO (Proximal Policy Optimization) algorithm.

Tests the PPO network, agent implementation, GAE computation, and training.
"""

import gymnasium as gym
import pytest

import mlx.core as mx

from smlx.gym.algorithms.ppo import ActorCriticNetwork, PPOAgent


@pytest.mark.unit
class TestActorCriticNetwork:
    """Tests for ActorCriticNetwork."""

    def test_network_creation(self):
        """Test creating actor-critic network."""
        network = ActorCriticNetwork(observation_dim=4, action_dim=2, hidden_dim=64)

        assert network.observation_dim == 4
        assert network.action_dim == 2
        assert network.hidden_dim == 64

    def test_network_forward_pass(self):
        """Test forward pass through actor-critic network."""
        network = ActorCriticNetwork(observation_dim=4, action_dim=2, hidden_dim=64)

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
        network = ActorCriticNetwork(observation_dim=8, action_dim=4, hidden_dim=32)

        observation = mx.random.normal((1, 8))
        action_logits, values = network(observation)

        assert action_logits.shape == (1, 4)
        assert values.shape == (1, 1)

    def test_network_with_multiple_layers(self):
        """Test network with different number of hidden layers."""
        network = ActorCriticNetwork(
            observation_dim=10, action_dim=5, hidden_dim=128, num_hidden_layers=3
        )

        observations = mx.random.normal((4, 10))
        action_logits, values = network(observations)

        assert action_logits.shape == (4, 5)
        assert values.shape == (4, 1)


@pytest.mark.unit
class TestPPOAgent:
    """Tests for PPOAgent."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("CartPole-v1")

    @pytest.fixture
    def agent(self, env):
        """Create PPO agent."""
        return PPOAgent(env, hidden_dim=32, n_steps=128)

    def test_agent_creation(self, agent):
        """Test creating PPO agent."""
        assert isinstance(agent, PPOAgent)
        assert agent.observation_dim == 4
        assert agent.action_dim == 2
        assert agent.n_steps == 128

    def test_agent_with_custom_params(self, env):
        """Test creating agent with custom hyperparameters."""
        agent = PPOAgent(
            env,
            hidden_dim=128,
            learning_rate=1e-4,
            gamma=0.95,
            gae_lambda=0.9,
            clip_epsilon=0.1,
            value_loss_coef=0.3,
            entropy_coef=0.02,
            n_steps=1024,
            batch_size=32,
            n_epochs=5,
        )

        assert agent.gamma == 0.95
        assert agent.gae_lambda == 0.9
        assert agent.clip_epsilon == 0.1
        assert agent.value_loss_coef == 0.3
        assert agent.entropy_coef == 0.02
        assert agent.n_steps == 1024
        assert agent.batch_size == 32
        assert agent.n_epochs == 5

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

    def test_select_action_stores_log_probs(self, agent):
        """Test that action selection stores log probabilities."""
        observation = mx.random.normal((4,))
        initial_len = len(agent.trajectory_log_probs)

        agent.select_action(observation, training=True)

        # Log prob should be stored
        assert len(agent.trajectory_log_probs) == initial_len + 1

    def test_train_step(self, agent):
        """Test PPO training step."""
        # Create batch
        batch_size = 64
        batch = {
            "states": mx.random.normal((batch_size, 4)),
            "actions": mx.array([0, 1] * 32),
            "old_log_probs": mx.random.normal((batch_size,)),
            "advantages": mx.random.normal((batch_size,)),
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

    def test_update_policy(self, agent):
        """Test policy update with collected trajectories."""
        # Set up a simple trajectory
        agent.trajectory_states = [mx.random.normal((4,)) for _ in range(20)]
        agent.trajectory_actions = [0, 1] * 10
        agent.trajectory_rewards = [1.0] * 20
        agent.trajectory_dones = [False] * 19 + [True]
        agent.trajectory_log_probs = [-0.5] * 20
        agent.trajectory_values = [0.5] * 20

        agent._update_policy()

        # After update, losses should be tracked
        assert len(agent.losses) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestPPOTraining:
    """Integration tests for PPO training."""

    def test_agent_training(self):
        """Test PPO agent training."""
        env = gym.make("CartPole-v1")
        agent = PPOAgent(env, hidden_dim=32, n_steps=128, batch_size=32, n_epochs=3)

        # Train for a few episodes
        metrics = agent.run(num_episodes=10, max_steps=200, verbose=False)

        # Check training completed
        assert agent.episode_count == 10
        assert metrics.episode == 10
        assert metrics.average_return >= 0.0
        assert agent.update_count > 0

        env.close()

    def test_agent_learns(self):
        """Test that PPO agent can learn."""
        env = gym.make("CartPole-v1")
        agent = PPOAgent(
            env,
            hidden_dim=64,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=5,
        )

        # Train
        agent.run(num_episodes=50, max_steps=200, verbose=False)

        # Agent should have trained
        assert agent.update_count > 0
        assert len(agent.losses) > 0

        env.close()

    def test_save_and_load(self):
        """Test saving and loading PPO agent."""
        env = gym.make("CartPole-v1")
        agent = PPOAgent(env, hidden_dim=32)

        # Train briefly
        agent.run(num_episodes=5, max_steps=100, verbose=False)

        # Save agent
        save_path = "/tmp/ppo_test.safetensors"
        agent.save(save_path)

        # Create new agent and load
        new_agent = PPOAgent(env, hidden_dim=32)
        new_agent.load(save_path)

        # Check update count was restored
        assert new_agent.update_count == agent.update_count

        env.close()

    def test_multiple_epochs(self):
        """Test that multiple epochs of updates occur."""
        env = gym.make("CartPole-v1")
        agent = PPOAgent(env, hidden_dim=32, n_steps=64, batch_size=16, n_epochs=10)

        # Run episodes
        agent.run(num_episodes=5, max_steps=100, verbose=False)

        # Should have multiple updates due to epochs
        assert agent.update_count > 0
        assert len(agent.losses) > 0

        env.close()

    def test_gae_computation(self):
        """Test that GAE (Generalized Advantage Estimation) is computed."""
        env = gym.make("CartPole-v1")
        agent = PPOAgent(
            env, hidden_dim=32, gamma=0.99, gae_lambda=0.95, n_steps=100
        )

        # Run episode with trajectory collection
        agent.run(num_episodes=3, max_steps=150, verbose=False)

        # GAE should have been computed during updates
        assert agent.update_count > 0

        env.close()
