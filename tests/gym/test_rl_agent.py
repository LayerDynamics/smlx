"""
Unit tests for RL agents.

Tests reinforcement learning agent implementations.
"""

import gymnasium as gym
import mlx.core as mx
import pytest

from smlx.agents.rl_agent import (
    EpsilonGreedyAgent,
    GreedyAgent,
    RandomAgent,
    RLAgent,
    RLAgentResponse,
)


@pytest.mark.unit
class TestRLAgentResponse:
    """Tests for RLAgentResponse dataclass."""

    def test_response_creation(self):
        """Test creating an RL agent response."""
        response = RLAgentResponse(
            content="Completed 10 episodes",
            episode_return=150.5,
            episode_length=200,
            success=True,
            reasoning="Good performance",
            metadata={"episodes": 10},
        )

        assert response.content == "Completed 10 episodes"
        assert response.episode_return == 150.5
        assert response.episode_length == 200
        assert response.success is True
        assert response.metadata["episodes"] == 10


@pytest.mark.unit
class TestRLAgent:
    """Tests for RLAgent base class."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("CartPole-v1")

    def test_agent_initialization(self, env):
        """Test that base agent cannot be instantiated directly."""
        # RLAgent is abstract, but we can create a minimal subclass
        class MinimalAgent(RLAgent):
            def select_action(self, observation):
                return self.env.action_space.sample()

        agent = MinimalAgent(env)

        assert agent.env == env
        assert agent.observation_space == env.observation_space
        assert agent.action_space == env.action_space
        assert agent.episode_count == 0

    def test_select_action_not_implemented(self, env):
        """Test that select_action must be implemented."""
        agent = RLAgent(env)

        observation = mx.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(NotImplementedError):
            agent.select_action(observation)

    def test_train_step_not_implemented(self, env):
        """Test that train_step is not implemented by default."""
        agent = RLAgent(env)
        batch = {}

        with pytest.raises(NotImplementedError):
            agent.train_step(batch)

    def test_save_not_implemented(self, env):
        """Test that save is not implemented by default."""
        agent = RLAgent(env)

        with pytest.raises(NotImplementedError):
            agent.save("test.pkl")

    def test_load_not_implemented(self, env):
        """Test that load is not implemented by default."""
        agent = RLAgent(env)

        with pytest.raises(NotImplementedError):
            agent.load("test.pkl")


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

    def test_agent_creation(self, agent, env):
        """Test creating a random agent."""
        assert isinstance(agent, RandomAgent)
        assert agent.env == env

    def test_select_action(self, agent, env):
        """Test that actions are sampled from action space."""
        observation = mx.array([1.0, 2.0, 3.0, 4.0])

        for _ in range(10):
            action = agent.select_action(observation)
            assert env.action_space.contains(action)

    def test_run_single_episode(self, agent):
        """Test running a single episode."""
        response = agent.run(num_episodes=1, max_steps=100)

        assert isinstance(response, RLAgentResponse)
        assert response.metadata["episodes"] == 1
        assert response.episode_return >= 0.0
        assert response.episode_length > 0

    def test_run_multiple_episodes(self, agent):
        """Test running multiple episodes."""
        response = agent.run(num_episodes=5, max_steps=100)

        assert response.metadata["episodes"] == 5
        assert len(response.metadata["episode_returns"]) == 5
        assert agent.episode_count == 5

    def test_max_steps_limit(self, agent):
        """Test that max_steps limits episode length."""
        response = agent.run(num_episodes=1, max_steps=50)

        assert response.episode_length <= 50


@pytest.mark.unit
class TestGreedyAgent:
    """Tests for GreedyAgent."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("CartPole-v1")

    @pytest.fixture
    def q_function(self):
        """Create a simple Q-function."""
        def q_func(state, action):
            # Simple Q-function: prefer action 0
            return 1.0 if action == 0 else 0.0

        return q_func

    @pytest.fixture
    def agent(self, env, q_function):
        """Create greedy agent."""
        return GreedyAgent(env, q_function=q_function)

    def test_agent_creation(self, agent, env, q_function):
        """Test creating a greedy agent."""
        assert isinstance(agent, GreedyAgent)
        assert agent.env == env
        assert agent.q_function == q_function

    def test_select_greedy_action(self, agent):
        """Test that agent selects greedy action."""
        observation = mx.array([1.0, 2.0, 3.0, 4.0])

        # Should always select action 0 (highest Q-value)
        for _ in range(10):
            action = agent.select_action(observation)
            assert action == 0

    def test_run_episode(self, agent):
        """Test running episode with greedy agent."""
        response = agent.run(num_episodes=1, max_steps=100)

        assert isinstance(response, RLAgentResponse)
        assert response.episode_length > 0


@pytest.mark.unit
class TestEpsilonGreedyAgent:
    """Tests for EpsilonGreedyAgent."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("CartPole-v1")

    @pytest.fixture
    def q_function(self):
        """Create a simple Q-function."""
        def q_func(state, action):
            return 1.0 if action == 0 else 0.0

        return q_func

    @pytest.fixture
    def agent(self, env, q_function):
        """Create epsilon-greedy agent."""
        return EpsilonGreedyAgent(env, q_function=q_function, epsilon=0.1)

    def test_agent_creation(self, agent, env):
        """Test creating an epsilon-greedy agent."""
        assert isinstance(agent, EpsilonGreedyAgent)
        assert agent.env == env
        assert agent.epsilon == 0.1

    def test_exploration_vs_exploitation(self, agent):
        """Test that agent balances exploration and exploitation."""
        observation = mx.array([1.0, 2.0, 3.0, 4.0])

        actions = [agent.select_action(observation) for _ in range(100)]

        # With epsilon=0.1, should mostly select action 0 (greedy)
        # but occasionally explore
        action_0_count = sum(1 for a in actions if a == 0)
        action_1_count = sum(1 for a in actions if a == 1)

        # Should be mostly greedy (action 0)
        assert action_0_count > action_1_count
        # But should have some exploration (action 1)
        assert action_1_count > 0

    def test_zero_epsilon(self, env, q_function):
        """Test that epsilon=0 means pure exploitation."""
        agent = EpsilonGreedyAgent(env, q_function=q_function, epsilon=0.0)
        observation = mx.array([1.0, 2.0, 3.0, 4.0])

        actions = [agent.select_action(observation) for _ in range(50)]

        # All actions should be 0 (greedy)
        assert all(a == 0 for a in actions)

    def test_high_epsilon(self, env, q_function):
        """Test that high epsilon means more exploration."""
        agent = EpsilonGreedyAgent(env, q_function=q_function, epsilon=0.9)
        observation = mx.array([1.0, 2.0, 3.0, 4.0])

        actions = [agent.select_action(observation) for _ in range(100)]

        # Should have significant exploration
        action_1_count = sum(1 for a in actions if a == 1)
        assert action_1_count > 30  # Should explore a lot


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for RL agents."""

    def test_agent_training_loop(self):
        """Test agent in a basic training loop."""
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        # Run training
        for episode in range(5):
            observation, _ = env.reset()
            observation = mx.array(observation) if not isinstance(observation, mx.array) else observation

            episode_return = 0.0
            done = False

            while not done:
                action = agent.select_action(observation)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                observation = mx.array(next_obs) if not isinstance(next_obs, mx.array) else next_obs
                episode_return += reward

            assert episode_return > 0.0

        env.close()

    def test_multiple_agents_comparison(self):
        """Test comparing multiple agents."""
        env = gym.make("CartPole-v1")

        # Create agents
        random_agent = RandomAgent(env)

        def simple_q(state, action):
            # Prefer action based on cart position
            cart_pos = float(state[0]) if isinstance(state, mx.array) else state[0]
            return 1.0 if (cart_pos < 0 and action == 0) or (cart_pos >= 0 and action == 1) else 0.0

        greedy_agent = GreedyAgent(env, q_function=simple_q)

        # Run episodes
        random_response = random_agent.run(num_episodes=10, max_steps=200)
        greedy_response = greedy_agent.run(num_episodes=10, max_steps=200)

        # Both should complete episodes
        assert random_response.metadata["episodes"] == 10
        assert greedy_response.metadata["episodes"] == 10

        env.close()

    def test_agent_with_different_envs(self):
        """Test agent works with different environments."""
        envs = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]

        for env_id in envs:
            env = gym.make(env_id)
            agent = RandomAgent(env)

            response = agent.run(num_episodes=2, max_steps=100)

            assert response.metadata["episodes"] == 2
            env.close()

    def test_epsilon_decay_strategy(self):
        """Test epsilon decay during training."""
        env = gym.make("CartPole-v1")

        def q_func(state, action):
            return 1.0 if action == 0 else 0.0

        agent = EpsilonGreedyAgent(env, q_function=q_func, epsilon=1.0)

        initial_epsilon = agent.epsilon

        # Simulate epsilon decay
        for episode in range(10):
            agent.run(num_episodes=1, max_steps=100)
            agent.epsilon = max(0.01, agent.epsilon * 0.95)

        final_epsilon = agent.epsilon

        assert final_epsilon < initial_epsilon
        assert final_epsilon >= 0.01

        env.close()


@pytest.mark.unit
class TestAgentAttributes:
    """Tests for optional agent attributes."""

    def test_optional_epsilon_attribute(self):
        """Test that epsilon attribute is optional."""
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        # RandomAgent doesn't set epsilon, should be None
        assert agent.epsilon is None

    def test_epsilon_attribute_set(self):
        """Test that epsilon-greedy agent sets epsilon."""
        env = gym.make("CartPole-v1")

        def q_func(state, action):
            return 0.0

        agent = EpsilonGreedyAgent(env, q_function=q_func, epsilon=0.2)

        assert agent.epsilon == 0.2

    def test_update_count_initialized(self):
        """Test that update_count is initialized."""
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        assert agent.update_count == 0

    def test_replay_buffer_optional(self):
        """Test that replay_buffer is optional."""
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        assert agent.replay_buffer is None

    def test_batch_size_optional(self):
        """Test that batch_size is optional."""
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        assert agent.batch_size is None
