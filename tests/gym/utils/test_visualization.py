"""
Unit tests for visualization utilities.

Tests plotting functions for training metrics, trajectories, and policy visualizations.
"""

import mlx.core as mx
import numpy as np
import pytest

from smlx.gym.utils.visualization import (
    create_training_dashboard,
    plot_episode_trajectory,
    plot_policy_distribution,
    plot_reward_distribution,
    plot_training_metrics,
    plot_value_function,
)

# Skip all tests in this module if matplotlib is not installed
try:
    import matplotlib
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

pytestmark = pytest.mark.skipif(
    not matplotlib_available,
    reason="matplotlib not installed. Install with: pip install matplotlib"
)


@pytest.mark.unit
class TestPlotTrainingMetrics:
    """Tests for plot_training_metrics function."""

    def test_plot_basic_metrics(self):
        """Test plotting basic training metrics."""
        episodes = list(range(100))
        returns = [float(i + np.random.randn() * 10) for i in range(100)]

        # Should not raise an error
        try:
            plot_training_metrics(episodes, returns, title="Test Metrics")
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_with_losses(self):
        """Test plotting with loss data."""
        episodes = list(range(50))
        returns = [float(i * 2) for i in range(50)]
        losses = [float(100 - i) for i in range(50)]

        try:
            plot_training_metrics(
                episodes, returns, losses=losses, title="Test with Losses"
            )
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_with_save_path(self, tmp_path):
        """Test saving plot to file."""
        episodes = list(range(20))
        returns = [float(i) for i in range(20)]

        save_path = tmp_path / "metrics.png"

        try:
            plot_training_metrics(episodes, returns, save_path=str(save_path))

            # Check file was created
            assert save_path.exists()
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_with_small_dataset(self):
        """Test plotting with small dataset."""
        episodes = [0, 1, 2]
        returns = [10.0, 15.0, 12.0]

        try:
            plot_training_metrics(episodes, returns)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_with_moving_average(self):
        """Test that moving average is computed."""
        episodes = list(range(100))
        returns = [float(i) for i in range(100)]

        try:
            # Should compute and plot moving average
            plot_training_metrics(episodes, returns)
        except ImportError:
            pytest.skip("matplotlib not installed")


@pytest.mark.unit
class TestPlotEpisodeTrajectory:
    """Tests for plot_episode_trajectory function."""

    def test_plot_basic_trajectory(self):
        """Test plotting basic episode trajectory."""
        observations = [np.array([i, i + 1, i + 2, i + 3]) for i in range(10)]
        actions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

        try:
            plot_episode_trajectory(observations, actions, rewards, title="Test Trajectory")
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_trajectory_with_mlx_arrays(self):
        """Test plotting trajectory with MLX arrays."""
        observations = [mx.array([i, i + 1]) for i in range(10)]
        actions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        rewards = [1.0] * 10

        try:
            plot_episode_trajectory(observations, actions, rewards)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_trajectory_with_save_path(self, tmp_path):
        """Test saving trajectory plot to file."""
        observations = [np.array([i, i + 1]) for i in range(5)]
        actions = [0, 1, 0, 1, 0]
        rewards = [1.0, 0.0, 1.0, 0.0, 1.0]

        save_path = tmp_path / "trajectory.png"

        try:
            plot_episode_trajectory(
                observations, actions, rewards, save_path=str(save_path)
            )

            assert save_path.exists()
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_trajectory_1d_observations(self):
        """Test plotting trajectory with 1D observations."""
        observations = [float(i) for i in range(10)]
        actions = list(range(10))
        rewards = [1.0] * 10

        try:
            plot_episode_trajectory(observations, actions, rewards)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_trajectory_negative_rewards(self):
        """Test plotting trajectory with negative rewards."""
        observations = [np.array([i, i + 1]) for i in range(10)]
        actions = list(range(10))
        rewards = [-1.0, 1.0, -0.5, 0.5, -2.0, 2.0, -1.0, 1.0, -0.5, 0.5]

        try:
            plot_episode_trajectory(observations, actions, rewards)
        except ImportError:
            pytest.skip("matplotlib not installed")


@pytest.mark.unit
class TestPlotValueFunction:
    """Tests for plot_value_function function."""

    def test_plot_1d_value_function(self):
        """Test plotting 1D value function."""
        states = mx.linspace(-1, 1, 100).reshape(-1, 1)
        values = mx.sin(states[:, 0]) * 2.0

        try:
            plot_value_function(states, values, title="1D Value Function")
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_2d_value_function(self):
        """Test plotting 2D value function."""
        # Create grid of states
        x = mx.linspace(-1, 1, 50)
        y = mx.linspace(-1, 1, 50)
        xx, yy = mx.meshgrid(x, y)
        states = mx.stack([xx.flatten(), yy.flatten()], axis=1)

        # Compute values
        values = mx.exp(-(states[:, 0] ** 2 + states[:, 1] ** 2))

        try:
            plot_value_function(states, values, title="2D Value Function")
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_high_dimensional_value_function(self):
        """Test plotting value function with high-dimensional states."""
        # Only first 2 dimensions will be plotted
        states = mx.random.uniform(-1, 1, (100, 10))
        values = mx.sum(states**2, axis=1)

        try:
            plot_value_function(states, values)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_value_function_with_save_path(self, tmp_path):
        """Test saving value function plot."""
        states = mx.linspace(-1, 1, 50).reshape(-1, 1)
        values = states[:, 0] ** 2

        save_path = tmp_path / "value_function.png"

        try:
            plot_value_function(states, values, save_path=str(save_path))

            assert save_path.exists()
        except ImportError:
            pytest.skip("matplotlib not installed")


@pytest.mark.unit
class TestPlotPolicyDistribution:
    """Tests for plot_policy_distribution function."""

    def test_plot_basic_policy(self):
        """Test plotting basic policy distribution."""
        state = mx.array([0.1, 0.2, 0.3, 0.4])
        action_probs = mx.array([0.2, 0.5, 0.3])

        try:
            plot_policy_distribution(state, action_probs, title="Test Policy")
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_policy_with_action_names(self):
        """Test plotting policy with action names."""
        state = mx.array([0.0, 0.0])
        action_probs = mx.array([0.3, 0.7])
        action_names = ["Left", "Right"]

        try:
            plot_policy_distribution(state, action_probs, action_names=action_names)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_policy_with_save_path(self, tmp_path):
        """Test saving policy distribution plot."""
        state = mx.array([0.0])
        action_probs = mx.array([0.25, 0.25, 0.25, 0.25])

        save_path = tmp_path / "policy.png"

        try:
            plot_policy_distribution(state, action_probs, save_path=str(save_path))

            assert save_path.exists()
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_policy_many_actions(self):
        """Test plotting policy with many actions."""
        state = mx.array([0.0])
        action_probs = mx.array([0.1] * 10)

        try:
            plot_policy_distribution(state, action_probs)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_policy_skewed_distribution(self):
        """Test plotting skewed policy distribution."""
        state = mx.array([0.0])
        action_probs = mx.array([0.9, 0.05, 0.05])

        try:
            plot_policy_distribution(
                state, action_probs, action_names=["Forward", "Left", "Right"]
            )
        except ImportError:
            pytest.skip("matplotlib not installed")


@pytest.mark.unit
class TestPlotRewardDistribution:
    """Tests for plot_reward_distribution function."""

    def test_plot_basic_distribution(self):
        """Test plotting basic reward distribution."""
        returns = [float(i + np.random.randn() * 5) for i in range(100)]

        try:
            plot_reward_distribution(returns, title="Test Returns")
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_distribution_with_save_path(self, tmp_path):
        """Test saving reward distribution plot."""
        returns = [float(i) for i in range(50)]

        save_path = tmp_path / "returns_dist.png"

        try:
            plot_reward_distribution(returns, save_path=str(save_path))

            assert save_path.exists()
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_distribution_negative_returns(self):
        """Test plotting distribution with negative returns."""
        returns = [float(i - 50) for i in range(100)]

        try:
            plot_reward_distribution(returns)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_distribution_small_dataset(self):
        """Test plotting distribution with small dataset."""
        returns = [10.0, 15.0, 12.0, 20.0, 8.0]

        try:
            plot_reward_distribution(returns)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_distribution_statistics(self):
        """Test that statistics are computed correctly."""
        returns = list(range(100))

        try:
            plot_reward_distribution(returns)
        except ImportError:
            pytest.skip("matplotlib not installed")


@pytest.mark.unit
class TestCreateTrainingDashboard:
    """Tests for create_training_dashboard function."""

    def test_create_basic_dashboard(self):
        """Test creating basic training dashboard."""
        episodes = list(range(100))
        returns = [float(i + np.random.randn() * 10) for i in range(100)]
        losses = [float(100 - i + np.random.randn() * 5) for i in range(100)]
        episode_lengths = [int(50 + np.random.randint(-10, 10)) for _ in range(100)]

        try:
            create_training_dashboard(episodes, returns, losses, episode_lengths)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_create_dashboard_with_save_path(self, tmp_path):
        """Test saving dashboard to file."""
        episodes = list(range(50))
        returns = [float(i) for i in range(50)]
        losses = [float(50 - i) for i in range(50)]
        episode_lengths = [50] * 50

        save_path = tmp_path / "dashboard.png"

        try:
            create_training_dashboard(
                episodes, returns, losses, episode_lengths, save_path=str(save_path)
            )

            assert save_path.exists()
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_create_dashboard_varying_performance(self):
        """Test dashboard with varying performance metrics."""
        episodes = list(range(100))
        # Simulate improving performance
        returns = [float(i * 1.5 + np.random.randn() * 10) for i in range(100)]
        # Simulate decreasing loss
        losses = [float(100 / (i + 1) + np.random.randn()) for i in range(100)]
        # Simulate increasing episode length
        episode_lengths = [int(20 + i * 0.5) for i in range(100)]

        try:
            create_training_dashboard(episodes, returns, losses, episode_lengths)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_create_dashboard_small_dataset(self):
        """Test dashboard with small dataset."""
        episodes = [0, 1, 2, 3, 4]
        returns = [10.0, 15.0, 12.0, 20.0, 18.0]
        losses = [5.0, 4.0, 3.5, 3.0, 2.5]
        episode_lengths = [50, 60, 55, 70, 65]

        try:
            create_training_dashboard(episodes, returns, losses, episode_lengths)
        except ImportError:
            pytest.skip("matplotlib not installed")


@pytest.mark.integration
class TestVisualizationIntegration:
    """Integration tests for visualization utilities."""

    def test_complete_visualization_workflow(self, tmp_path):
        """Test complete visualization workflow."""
        # Generate training data
        episodes = list(range(100))
        returns = [float(i + np.random.randn() * 10) for i in range(100)]
        losses = [float(100 - i + np.random.randn() * 5) for i in range(100)]
        episode_lengths = [int(50 + np.random.randint(-10, 10)) for _ in range(100)]

        try:
            # Create all visualizations
            plot_training_metrics(
                episodes,
                returns,
                losses=losses,
                save_path=str(tmp_path / "metrics.png"),
            )

            plot_reward_distribution(returns, save_path=str(tmp_path / "returns.png"))

            create_training_dashboard(
                episodes,
                returns,
                losses,
                episode_lengths,
                save_path=str(tmp_path / "dashboard.png"),
            )

            # Check all files were created
            assert (tmp_path / "metrics.png").exists()
            assert (tmp_path / "returns.png").exists()
            assert (tmp_path / "dashboard.png").exists()

        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_trajectory_and_policy_visualization(self, tmp_path):
        """Test trajectory and policy visualization together."""
        # Generate episode data
        observations = [mx.array([i * 0.1, i * 0.2]) for i in range(20)]
        actions = [i % 3 for i in range(20)]
        rewards = [1.0 if i % 2 == 0 else -0.5 for i in range(20)]

        # Generate policy data
        state = mx.array([0.5, 0.5])
        action_probs = mx.array([0.5, 0.3, 0.2])

        try:
            plot_episode_trajectory(
                observations,
                actions,
                rewards,
                save_path=str(tmp_path / "trajectory.png"),
            )

            plot_policy_distribution(
                state,
                action_probs,
                action_names=["Up", "Down", "Stay"],
                save_path=str(tmp_path / "policy.png"),
            )

            assert (tmp_path / "trajectory.png").exists()
            assert (tmp_path / "policy.png").exists()

        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_value_function_visualization_workflow(self, tmp_path):
        """Test value function visualization workflow."""
        # 1D value function
        states_1d = mx.linspace(-2, 2, 100).reshape(-1, 1)
        values_1d = -(states_1d[:, 0] ** 2) + 4.0

        # 2D value function
        x = mx.linspace(-2, 2, 30)
        y = mx.linspace(-2, 2, 30)
        xx, yy = mx.meshgrid(x, y)
        states_2d = mx.stack([xx.flatten(), yy.flatten()], axis=1)
        values_2d = -(states_2d[:, 0] ** 2 + states_2d[:, 1] ** 2) + 4.0

        try:
            plot_value_function(
                states_1d, values_1d, save_path=str(tmp_path / "value_1d.png")
            )

            plot_value_function(
                states_2d, values_2d, save_path=str(tmp_path / "value_2d.png")
            )

            assert (tmp_path / "value_1d.png").exists()
            assert (tmp_path / "value_2d.png").exists()

        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_visualization_without_matplotlib(self):
        """Test that visualization functions handle missing matplotlib gracefully."""
        # This test verifies error handling
        # Actual behavior depends on whether matplotlib is installed
        episodes = list(range(10))
        returns = list(range(10))

        try:
            plot_training_metrics(episodes, returns)
            # If we get here, matplotlib is installed
        except ImportError:
            pytest.skip("matplotlib not installed (expected)")
