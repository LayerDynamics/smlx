"""
Unit tests for experience replay buffers.

Tests replay buffer implementations for off-policy RL.
"""

import mlx.core as mx
import numpy as np
import pytest

from smlx.gym.replay import (
    EpisodeBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    Transition,
    compute_gae,
    compute_returns,
)


@pytest.mark.unit
class TestTransition:
    """Tests for Transition dataclass."""

    def test_transition_creation(self):
        """Test creating a transition."""
        state = mx.array([1.0, 2.0, 3.0, 4.0])
        next_state = mx.array([2.0, 3.0, 4.0, 5.0])
        action = 0
        reward = 1.0
        done = False

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info={"test": "data"},
        )

        assert mx.array_equal(transition.state, state)
        assert transition.action == action
        assert transition.reward == reward
        assert mx.array_equal(transition.next_state, next_state)
        assert transition.done == done
        assert transition.info == {"test": "data"}


@pytest.mark.unit
class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    @pytest.fixture
    def buffer(self):
        """Create a test buffer."""
        return ReplayBuffer(capacity=100)

    def test_buffer_creation(self, buffer):
        """Test buffer initialization."""
        assert buffer.capacity == 100
        assert len(buffer) == 0
        assert buffer.position == 0

    def test_add_transition(self, buffer):
        """Test adding a transition."""
        state = mx.array([1.0, 2.0, 3.0, 4.0])
        next_state = mx.array([2.0, 3.0, 4.0, 5.0])

        buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)

        assert len(buffer) == 1

    def test_add_multiple_transitions(self, buffer):
        """Test adding multiple transitions."""
        for i in range(10):
            state = mx.array([float(i)] * 4)
            next_state = mx.array([float(i + 1)] * 4)
            buffer.add(state, action=i % 2, reward=float(i), next_state=next_state, done=False)

        assert len(buffer) == 10

    def test_buffer_overflow(self):
        """Test that buffer wraps around when full."""
        buffer = ReplayBuffer(capacity=5)

        for i in range(10):
            state = mx.array([float(i)] * 4)
            next_state = mx.array([float(i + 1)] * 4)
            buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)

        assert len(buffer) == 5  # Should not exceed capacity

    def test_sample_batch(self, buffer):
        """Test sampling a batch."""
        # Add some transitions
        for i in range(50):
            state = mx.array([float(i)] * 4)
            next_state = mx.array([float(i + 1)] * 4)
            buffer.add(state, action=i % 2, reward=float(i), next_state=next_state, done=False)

        # Sample batch
        batch = buffer.sample(batch_size=32)

        assert "states" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert "next_states" in batch
        assert "dones" in batch

        assert batch["states"].shape[0] == 32
        assert len(batch["actions"]) == 32
        assert batch["rewards"].shape[0] == 32

    def test_sample_larger_than_buffer(self, buffer):
        """Test that sampling larger than buffer size raises error."""
        # Add only 10 transitions
        for i in range(10):
            state = mx.array([float(i)] * 4)
            next_state = mx.array([float(i + 1)] * 4)
            buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)

        with pytest.raises(ValueError, match="Cannot sample"):
            buffer.sample(batch_size=20)

    def test_clear_buffer(self, buffer):
        """Test clearing the buffer."""
        # Add transitions
        for i in range(10):
            state = mx.array([float(i)] * 4)
            next_state = mx.array([float(i + 1)] * 4)
            buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.position == 0

    def test_batch_types(self, buffer):
        """Test that batch contains correct types."""
        # Add transitions
        for i in range(50):
            state = mx.array([float(i)] * 4)
            next_state = mx.array([float(i + 1)] * 4)
            buffer.add(state, action=i % 2, reward=float(i), next_state=next_state, done=i % 10 == 0)

        batch = buffer.sample(batch_size=32)

        assert isinstance(batch["states"], mx.array)
        assert isinstance(batch["actions"], mx.array)  # Actions are stacked into MLX array
        assert isinstance(batch["rewards"], mx.array)
        assert isinstance(batch["next_states"], mx.array)
        assert isinstance(batch["dones"], mx.array)


@pytest.mark.unit
class TestPrioritizedReplayBuffer:
    """Tests for PrioritizedReplayBuffer."""

    @pytest.fixture
    def buffer(self):
        """Create a test prioritized buffer."""
        return PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

    def test_buffer_creation(self, buffer):
        """Test prioritized buffer initialization."""
        assert buffer.capacity == 100
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4
        assert len(buffer) == 0

    def test_add_with_priority(self, buffer):
        """Test adding transitions with priorities."""
        state = mx.array([1.0, 2.0, 3.0, 4.0])
        next_state = mx.array([2.0, 3.0, 4.0, 5.0])

        buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False, priority=0.5)

        assert len(buffer) == 1
        assert buffer.priorities[0] == 0.5

    def test_default_max_priority(self, buffer):
        """Test that new transitions get max priority by default."""
        # Add first transition with explicit priority
        state = mx.array([1.0, 2.0, 3.0, 4.0])
        next_state = mx.array([2.0, 3.0, 4.0, 5.0])
        buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False, priority=0.8)

        # Add second without priority (should use max)
        buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)

        assert buffer.priorities[1] == 0.8  # Should be max priority

    def test_sample_with_importance_weights(self, buffer):
        """Test sampling returns importance weights."""
        # Add transitions with different priorities
        for i in range(50):
            state = mx.array([float(i)] * 4)
            next_state = mx.array([float(i + 1)] * 4)
            priority = 0.1 if i < 25 else 0.9  # Two groups with different priorities
            buffer.add(
                state, action=i % 2, reward=float(i), next_state=next_state, done=False, priority=priority
            )

        batch = buffer.sample(batch_size=32)

        assert "weights" in batch
        assert batch["weights"].shape[0] == 32
        assert "indices" in batch

    def test_update_priorities(self, buffer):
        """Test updating priorities."""
        # Add transitions
        for i in range(10):
            state = mx.array([float(i)] * 4)
            next_state = mx.array([float(i + 1)] * 4)
            buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False, priority=0.5)

        # Sample and update
        batch = buffer.sample(batch_size=5)
        indices = batch["indices"]
        new_priorities = [0.9] * 5

        buffer.update_priorities(indices, new_priorities)

        # Check priorities updated
        for idx in indices:
            assert buffer.priorities[idx] == 0.9

    def test_beta_annealing(self, buffer):
        """Test beta annealing for importance sampling."""
        initial_beta = buffer.beta

        buffer.anneal_beta(1.0)

        assert buffer.beta == 1.0
        assert buffer.beta > initial_beta


@pytest.mark.unit
class TestEpisodeBuffer:
    """Tests for EpisodeBuffer."""

    @pytest.fixture
    def buffer(self):
        """Create a test episode buffer."""
        return EpisodeBuffer(max_episodes=10)

    def test_buffer_creation(self, buffer):
        """Test episode buffer initialization."""
        assert buffer.max_episodes == 10
        assert len(buffer.episodes) == 0

    def test_add_transition_to_current_episode(self, buffer):
        """Test adding transitions to current episode."""
        buffer.start_episode()

        state = mx.array([1.0, 2.0, 3.0, 4.0])
        next_state = mx.array([2.0, 3.0, 4.0, 5.0])

        buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)

        assert len(buffer.current_episode) == 1

    def test_end_episode(self, buffer):
        """Test ending an episode."""
        buffer.start_episode()

        # Add transitions
        for i in range(5):
            state = mx.array([float(i)] * 4)
            next_state = mx.array([float(i + 1)] * 4)
            buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)

        buffer.end_episode()

        assert len(buffer.episodes) == 1
        assert len(buffer.current_episode) == 0

    def test_multiple_episodes(self, buffer):
        """Test storing multiple episodes."""
        for episode in range(3):
            buffer.start_episode()
            for i in range(10):
                state = mx.array([float(i)] * 4)
                next_state = mx.array([float(i + 1)] * 4)
                buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)
            buffer.end_episode()

        assert len(buffer.episodes) == 3

    def test_max_episodes_limit(self):
        """Test that buffer respects max episodes limit."""
        buffer = EpisodeBuffer(max_episodes=3)

        # Add 5 episodes
        for episode in range(5):
            buffer.start_episode()
            for i in range(5):
                state = mx.array([float(i)] * 4)
                next_state = mx.array([float(i + 1)] * 4)
                buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)
            buffer.end_episode()

        assert len(buffer.episodes) == 3  # Should only keep 3

    def test_get_all_transitions(self, buffer):
        """Test getting all transitions from all episodes."""
        # Add episodes
        for episode in range(2):
            buffer.start_episode()
            for i in range(5):
                state = mx.array([float(i)] * 4)
                next_state = mx.array([float(i + 1)] * 4)
                buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)
            buffer.end_episode()

        transitions = buffer.get_all_transitions()

        assert len(transitions) == 10  # 2 episodes * 5 transitions


@pytest.mark.unit
class TestComputeReturns:
    """Tests for compute_returns function."""

    def test_basic_returns(self):
        """Test computing basic discounted returns."""
        rewards = [1.0, 1.0, 1.0, 1.0]
        gamma = 0.9

        returns = compute_returns(rewards, gamma)

        # Expected: [1 + 0.9 + 0.81 + 0.729, 1 + 0.9 + 0.81, 1 + 0.9, 1]
        assert len(returns) == 4
        assert returns[0] > returns[1] > returns[2] > returns[3]
        assert abs(returns[3] - 1.0) < 1e-6

    def test_returns_with_terminal(self):
        """Test returns computation stops at terminal states."""
        rewards = [1.0, 1.0, 1.0, 1.0]
        dones = [False, False, True, False]
        gamma = 0.9

        returns = compute_returns(rewards, gamma, dones)

        # Return at index 2 should not include reward at index 3
        assert len(returns) == 4

    def test_zero_gamma(self):
        """Test returns with gamma=0 (no discounting)."""
        rewards = [1.0, 2.0, 3.0]
        gamma = 0.0

        returns = compute_returns(rewards, gamma)

        # With gamma=0, returns should equal immediate rewards
        assert abs(returns[0] - 1.0) < 1e-6
        assert abs(returns[1] - 2.0) < 1e-6
        assert abs(returns[2] - 3.0) < 1e-6


@pytest.mark.unit
class TestComputeGAE:
    """Tests for compute_gae function."""

    def test_basic_gae(self):
        """Test computing basic GAE."""
        rewards = [1.0, 1.0, 1.0, 1.0]
        # values should include bootstrap value (length T+1)
        values = [0.5, 0.6, 0.7, 0.8, 0.0]
        gamma = 0.99
        lambda_gae = 0.95

        advantages, returns = compute_gae(rewards, values, None, gamma, lambda_gae)

        assert len(advantages) == 4
        assert isinstance(advantages, mx.array)
        assert len(returns) == 4
        assert isinstance(returns, mx.array)

    def test_gae_with_terminal(self):
        """Test GAE computation with terminal states."""
        rewards = [1.0, 1.0, 1.0, 1.0]
        # values should include bootstrap value (length T+1)
        values = [0.5, 0.6, 0.7, 0.8, 0.0]
        dones = [False, False, True, False]
        gamma = 0.99
        lambda_gae = 0.95

        advantages, returns = compute_gae(rewards, values, dones, gamma, lambda_gae)

        # Advantage at terminal state should not bootstrap
        assert len(advantages) == 4
        assert len(returns) == 4

    def test_zero_lambda(self):
        """Test GAE with lambda=0 (TD error)."""
        rewards = [1.0, 1.0, 1.0]
        # values should include bootstrap value (length T+1)
        values = [0.5, 0.5, 0.5, 0.0]
        gamma = 0.99
        lambda_gae = 0.0

        advantages, returns = compute_gae(rewards, values, None, gamma, lambda_gae)

        # With lambda=0, should be TD error: r + gamma * V(s') - V(s)
        expected = 1.0 + 0.99 * 0.5 - 0.5
        assert abs(float(advantages[0]) - expected) < 1e-6


@pytest.mark.integration
class TestReplayBufferIntegration:
    """Integration tests for replay buffers."""

    def test_buffer_in_training_loop(self):
        """Test using buffer in a simulated training loop."""
        buffer = ReplayBuffer(capacity=1000)

        # Simulate collecting experience
        for episode in range(10):
            for step in range(20):
                state = mx.array(np.random.randn(4))
                action = np.random.randint(0, 2)
                reward = np.random.randn()
                next_state = mx.array(np.random.randn(4))
                done = step == 19

                buffer.add(state, action, reward, next_state, done)

        # Training: sample batches
        for _ in range(5):
            if len(buffer) >= 64:
                batch = buffer.sample(64)
                assert batch["states"].shape[0] == 64

    def test_prioritized_buffer_in_training(self):
        """Test using prioritized buffer in training."""
        buffer = PrioritizedReplayBuffer(capacity=1000)

        # Collect experience
        for i in range(100):
            state = mx.array(np.random.randn(4))
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = mx.array(np.random.randn(4))
            done = i % 20 == 19
            priority = abs(reward)  # Use reward magnitude as priority

            buffer.add(state, action, reward, next_state, done, priority=priority)

        # Sample and update priorities
        batch = buffer.sample(32)
        indices = batch["indices"]
        new_priorities = np.random.rand(32).tolist()
        buffer.update_priorities(indices, new_priorities)

        assert len(buffer) == 100
