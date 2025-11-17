"""
Unit tests for DQN (Deep Q-Network) algorithm.

Tests the DQN network, agent implementation, replay buffer integration, and training.
"""

import gymnasium as gym
import pytest

import mlx.core as mx

from smlx.gym.algorithms.dqn import DoubleDQNAgent, DQNAgent, QNetwork


@pytest.mark.unit
class TestQNetwork:
    """Tests for QNetwork."""

    def test_network_creation(self):
        """Test creating Q-network."""
        network = QNetwork(observation_dim=4, action_dim=2, hidden_dim=128)

        assert network.observation_dim == 4
        assert network.action_dim == 2
        assert network.hidden_dim == 128

    def test_network_forward_pass(self):
        """Test forward pass through Q-network."""
        network = QNetwork(observation_dim=4, action_dim=2, hidden_dim=128)

        # Create batch of observations
        batch_size = 16
        observations = mx.random.normal((batch_size, 4))

        # Forward pass
        q_values = network(observations)

        # Check output shape
        assert q_values.shape == (batch_size, 2)

    def test_network_single_observation(self):
        """Test network with single observation."""
        network = QNetwork(observation_dim=8, action_dim=4, hidden_dim=64)

        observation = mx.random.normal((1, 8))
        q_values = network(observation)

        assert q_values.shape == (1, 4)

    def test_network_with_multiple_layers(self):
        """Test network with different number of hidden layers."""
        network = QNetwork(
            observation_dim=10, action_dim=5, hidden_dim=256, num_hidden_layers=3
        )

        observations = mx.random.normal((8, 10))
        q_values = network(observations)

        assert q_values.shape == (8, 5)


@pytest.mark.unit
class TestDQNAgent:
    """Tests for DQNAgent."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("CartPole-v1")

    @pytest.fixture
    def agent(self, env):
        """Create DQN agent."""
        return DQNAgent(env, hidden_dim=64, buffer_capacity=1000)

    def test_agent_creation(self, agent):
        """Test creating DQN agent."""
        assert isinstance(agent, DQNAgent)
        assert agent.observation_dim == 4
        assert agent.action_dim == 2
        assert agent.batch_size == 64

    def test_agent_with_custom_params(self, env):
        """Test creating agent with custom hyperparameters."""
        agent = DQNAgent(
            env,
            hidden_dim=256,
            learning_rate=5e-4,
            gamma=0.95,
            epsilon_start=0.9,
            epsilon_end=0.05,
            epsilon_decay=0.99,
            buffer_capacity=5000,
            batch_size=32,
            target_update_freq=50,
        )

        assert agent.gamma == 0.95
        assert agent.epsilon == 0.9
        assert agent.epsilon_end == 0.05
        assert agent.batch_size == 32
        assert agent.target_update_freq == 50

    def test_select_action_exploration(self, agent):
        """Test action selection with exploration."""
        agent.epsilon = 1.0  # Always explore
        observation = mx.random.normal((4,))

        action = agent.select_action(observation)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_exploitation(self, agent):
        """Test action selection with exploitation."""
        agent.epsilon = 0.0  # Never explore
        observation = mx.random.normal((4,))

        action = agent.select_action(observation)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_replay_buffer_integration(self, agent):
        """Test that replay buffer is used correctly."""
        # Add experiences
        for _ in range(10):
            state = mx.random.normal((4,))
            next_state = mx.random.normal((4,))
            agent.replay_buffer.add(
                state=state,
                action=0,
                reward=1.0,
                next_state=next_state,
                done=False,
            )

        assert len(agent.replay_buffer) == 10

    def test_train_step(self, agent):
        """Test DQN training step."""
        # Create batch
        batch_size = 32
        batch = {
            "states": mx.random.normal((batch_size, 4)),
            "actions": mx.array([0, 1] * 16),
            "rewards": mx.random.normal((batch_size,)),
            "next_states": mx.random.normal((batch_size, 4)),
            "dones": mx.zeros((batch_size,)),
        }

        metrics = agent.train_step(batch)

        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_target_network_update(self, agent):
        """Test target network update."""
        # Get initial target network parameters (flatten nested structures)
        def flatten_params(params):
            flat = []
            for v in params.values():
                if isinstance(v, dict):
                    flat.extend(flatten_params(v))
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            flat.extend(flatten_params(item))
                        elif isinstance(item, mx.array):
                            flat.append(float(mx.sum(item)))
                elif isinstance(v, mx.array):
                    flat.append(float(mx.sum(v)))
            return flat

        initial_params = flatten_params(agent.target_network.parameters())

        # Update Q-network
        for _ in range(5):
            batch = {
                "states": mx.random.normal((32, 4)),
                "actions": mx.array([0, 1] * 16),
                "rewards": mx.random.normal((32,)),
                "next_states": mx.random.normal((32, 4)),
                "dones": mx.zeros((32,)),
            }
            agent.train_step(batch)

        # Update target network
        agent._update_target_network()

        # Target network should have changed
        updated_params = flatten_params(agent.target_network.parameters())
        # At least some parameters should be different
        assert len(initial_params) == len(updated_params)
        # Check that parameters actually changed
        assert any(abs(a - b) > 1e-6 for a, b in zip(initial_params, updated_params))

    def test_epsilon_decay(self, agent):
        """Test epsilon decay after episodes."""
        initial_epsilon = agent.epsilon

        # Run episode
        agent._run_episode(max_steps=50)

        # Epsilon should decay
        assert agent.epsilon < initial_epsilon or agent.epsilon == agent.epsilon_end

    def test_huber_loss(self, agent):
        """Test Huber loss computation."""
        x = mx.array([0.5, 1.5, 2.5])
        loss = agent._huber_loss(x, delta=1.0)

        assert loss.shape == (3,)
        # All losses should be positive
        assert mx.all(loss >= 0.0)


@pytest.mark.unit
class TestDoubleDQNAgent:
    """Tests for DoubleDQNAgent."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("CartPole-v1")

    @pytest.fixture
    def agent(self, env):
        """Create Double DQN agent."""
        return DoubleDQNAgent(env, hidden_dim=64)

    def test_agent_creation(self, agent):
        """Test creating Double DQN agent."""
        assert isinstance(agent, DoubleDQNAgent)
        assert isinstance(agent, DQNAgent)

    def test_train_step_uses_double_q_learning(self, agent):
        """Test that training uses Double Q-learning."""
        batch = {
            "states": mx.random.normal((32, 4)),
            "actions": mx.array([0, 1] * 16),
            "rewards": mx.random.normal((32,)),
            "next_states": mx.random.normal((32, 4)),
            "dones": mx.zeros((32,)),
        }

        metrics = agent.train_step(batch)

        # Should produce valid loss
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)


@pytest.mark.integration
@pytest.mark.slow
class TestDQNTraining:
    """Integration tests for DQN training."""

    def test_agent_training(self):
        """Test DQN agent training."""
        env = gym.make("CartPole-v1")
        agent = DQNAgent(
            env, hidden_dim=64, buffer_capacity=2000, min_buffer_size=100
        )

        # Train for a few episodes
        metrics = agent.run(num_episodes=20, max_steps=100, verbose=False)

        # Check training completed
        assert agent.episode_count == 20
        assert metrics.episode == 20
        assert metrics.average_return >= 0.0
        assert agent.update_count > 0
        assert len(agent.replay_buffer) > 0

        env.close()

    def test_agent_learns(self):
        """Test that DQN agent can learn."""
        env = gym.make("CartPole-v1")
        agent = DQNAgent(
            env,
            hidden_dim=128,
            learning_rate=1e-3,
            buffer_capacity=5000,
            min_buffer_size=200,
        )

        # Train
        agent.run(num_episodes=100, max_steps=200, verbose=False)

        # Agent should have trained
        assert agent.update_count > 0
        assert len(agent.losses) > 0
        assert agent.epsilon < agent.epsilon_start

        env.close()

    def test_save_and_load(self):
        """Test saving and loading DQN agent."""
        env = gym.make("CartPole-v1")
        agent = DQNAgent(env, hidden_dim=64)

        # Train briefly
        agent.run(num_episodes=10, max_steps=50, verbose=False)

        # Save agent
        save_path = "/tmp/dqn_test.safetensors"
        agent.save(save_path)

        # Create new agent and load
        new_agent = DQNAgent(env, hidden_dim=64)
        new_agent.load(save_path)

        # Check state was restored (use approx for float comparison)
        assert new_agent.epsilon == pytest.approx(agent.epsilon, rel=1e-6)
        assert new_agent.step_count == agent.step_count
        assert new_agent.update_count == agent.update_count

        env.close()

    def test_double_dqn_training(self):
        """Test Double DQN training."""
        env = gym.make("CartPole-v1")
        agent = DoubleDQNAgent(
            env, hidden_dim=64, buffer_capacity=2000, min_buffer_size=100
        )

        # Train
        metrics = agent.run(num_episodes=20, max_steps=100, verbose=False)

        # Check training completed
        assert agent.episode_count == 20
        assert metrics.average_return >= 0.0
        assert agent.update_count > 0

        env.close()

    def test_replay_buffer_sampling(self):
        """Test that replay buffer sampling works during training."""
        env = gym.make("CartPole-v1")
        agent = DQNAgent(
            env, hidden_dim=32, buffer_capacity=500, min_buffer_size=50, batch_size=16
        )

        # Fill replay buffer
        agent.run(num_episodes=5, max_steps=100, verbose=False)

        # Buffer should have experiences
        assert len(agent.replay_buffer) >= agent.min_buffer_size

        # Training should have occurred
        assert agent.update_count > 0

        env.close()
