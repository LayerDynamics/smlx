"""
Visualization utilities for SMLX Gym.

This module provides utilities for visualizing training progress, episode
recordings, and agent behavior. Includes plotting functions for metrics,
trajectories, and policy visualizations.

Reference: SMLX_Gym.md, Section 4.4 (Advanced Features)
"""

from typing import Any, Optional

import mlx.core as mx
import numpy as np

try:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError:
    Axes = Any  # type: ignore
    Figure = Any  # type: ignore


def plot_training_metrics(
    episodes: list[int],
    returns: list[float],
    losses: Optional[list[float]] = None,
    title: str = "Training Progress",
    save_path: Optional[str] = None,
):
    """
    Plot training metrics over episodes.

    Args:
        episodes: List of episode numbers
        returns: List of episode returns
        losses: Optional list of losses
        title: Plot title
        save_path: Optional path to save figure

    Example:
        ```python
        from smlx.gym.utils.visualization import plot_training_metrics

        episodes = list(range(100))
        returns = [agent.episode_returns[i] for i in episodes]
        losses = [agent.losses[i] for i in episodes]

        plot_training_metrics(episodes, returns, losses, save_path="training.png")
        ```
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot returns
    color = "tab:blue"
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return", color=color)
    ax1.plot(episodes, returns, color=color, alpha=0.6, label="Return")

    # Plot moving average
    window = min(10, len(returns) // 10)
    if window > 1:
        returns_ma = np.convolve(
            returns, np.ones(window) / window, mode="valid"
        )
        episodes_ma = episodes[window - 1 :]
        ax1.plot(episodes_ma, returns_ma, color=color, linewidth=2, label=f"MA({window})")

    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot losses if provided
    if losses is not None:
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Loss", color=color)
        ax2.plot(episodes[: len(losses)], losses, color=color, alpha=0.6, label="Loss")
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.legend(loc="upper right")

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_episode_trajectory(
    observations: list[Any],
    actions: list[Any],
    rewards: list[float],
    title: str = "Episode Trajectory",
    save_path: Optional[str] = None,
):
    """
    Plot trajectory from a single episode.

    Args:
        observations: List of observations
        actions: List of actions
        rewards: List of rewards
        title: Plot title
        save_path: Optional path to save figure

    Example:
        ```python
        from smlx.gym.utils.visualization import plot_episode_trajectory
        from smlx.gym.utils.recording import EpisodeRecording

        recording = EpisodeRecording.load("episode_100.npz")
        plot_episode_trajectory(
            recording.observations,
            recording.actions,
            recording.rewards,
            save_path="trajectory.png"
        )
        ```
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    # Convert observations to numpy if needed
    obs_array = []
    for obs in observations:
        if isinstance(obs, mx.array):
            obs_array.append(np.array(obs))
        elif isinstance(obs, np.ndarray):
            obs_array.append(obs)
        else:
            obs_array.append(np.array(obs))

    obs_array = np.array(obs_array)

    # Create subplots
    n_subplots = 2 + (1 if obs_array.ndim > 1 else 0)
    fig, axes_raw = plt.subplots(n_subplots, 1, figsize=(10, 3 * n_subplots))

    # Ensure axes is always a list for consistent indexing
    if n_subplots == 1:
        axes: list[Axes] = [axes_raw]  # type: ignore
    else:
        axes = list(axes_raw.flat)  # type: ignore

    # Plot observations
    if obs_array.ndim > 1:
        for i in range(min(4, obs_array.shape[1])):  # Plot first 4 dimensions
            axes[0].plot(obs_array[:, i], label=f"obs[{i}]", alpha=0.7)
        axes[0].set_ylabel("Observation")
        axes[0].set_xlabel("Step")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Observations")
        ax_idx = 1
    else:
        ax_idx = 0

    # Plot actions
    axes[ax_idx].plot(actions, marker="o", markersize=3, alpha=0.7)
    axes[ax_idx].set_ylabel("Action")
    axes[ax_idx].set_xlabel("Step")
    axes[ax_idx].grid(True, alpha=0.3)
    axes[ax_idx].set_title("Actions")

    # Plot rewards
    axes[ax_idx + 1].plot(rewards, color="green", alpha=0.7)
    axes[ax_idx + 1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[ax_idx + 1].set_ylabel("Reward")
    axes[ax_idx + 1].set_xlabel("Step")
    axes[ax_idx + 1].grid(True, alpha=0.3)
    axes[ax_idx + 1].set_title(f"Rewards (Total: {sum(rewards):.2f})")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_value_function(
    states: mx.array,
    values: mx.array,
    title: str = "Value Function",
    save_path: Optional[str] = None,
):
    """
    Plot value function over state space.

    Args:
        states: State points (MLX array) [N, state_dim]
        values: Value estimates (MLX array) [N]
        title: Plot title
        save_path: Optional path to save figure

    Example:
        ```python
        import mlx.core as mx
        from smlx.gym.utils.visualization import plot_value_function

        # Sample states
        states = mx.random.uniform(-1, 1, (1000, 2))

        # Get value estimates
        values = agent.critic(states).squeeze()

        plot_value_function(states, values, save_path="value_function.png")
        ```
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    # Convert to numpy
    states_np = np.array(states)
    values_np = np.array(values)

    if states_np.shape[1] == 1:
        # 1D state space - line plot
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(states_np[:, 0])
        plt.plot(states_np[sorted_idx, 0], values_np[sorted_idx])
        plt.xlabel("State")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)

    elif states_np.shape[1] == 2:
        # 2D state space - scatter plot with color
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            states_np[:, 0], states_np[:, 1], c=values_np, cmap="viridis", alpha=0.6
        )
        plt.colorbar(scatter, label="Value")
        plt.xlabel("State Dim 0")
        plt.ylabel("State Dim 1")
        plt.grid(True, alpha=0.3)

    else:
        # High-dimensional - plot first 2 dimensions
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            states_np[:, 0], states_np[:, 1], c=values_np, cmap="viridis", alpha=0.6
        )
        plt.colorbar(scatter, label="Value")
        plt.xlabel("State Dim 0")
        plt.ylabel("State Dim 1")
        plt.title(title + " (first 2 dims)")
        plt.grid(True, alpha=0.3)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_policy_distribution(
    state: mx.array,
    action_probs: mx.array,
    action_names: Optional[list[str]] = None,
    title: str = "Policy Distribution",
    save_path: Optional[str] = None,
):
    """
    Plot policy distribution for a given state.

    Args:
        state: State (MLX array)
        action_probs: Action probabilities (MLX array)
        action_names: Optional action names
        title: Plot title
        save_path: Optional path to save figure

    Example:
        ```python
        import mlx.core as mx
        from smlx.gym.utils.visualization import plot_policy_distribution

        # Get policy for a state
        state = mx.array([0.1, 0.2, 0.3, 0.4])
        action_logits, _ = agent.actor_critic(state.reshape(1, -1))
        action_probs = mx.softmax(action_logits[0])

        plot_policy_distribution(
            state,
            action_probs,
            action_names=["Left", "Right"],
            save_path="policy.png"
        )
        ```
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    # Convert to numpy
    action_probs_np = np.array(action_probs)

    # Create action labels
    if action_names is None:
        action_names = [f"Action {i}" for i in range(len(action_probs_np))]

    plt.figure(figsize=(10, 6))
    plt.bar(action_names, action_probs_np, alpha=0.7, color="steelblue")
    plt.ylabel("Probability")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_reward_distribution(
    returns: list[float],
    title: str = "Return Distribution",
    save_path: Optional[str] = None,
):
    """
    Plot distribution of episode returns.

    Args:
        returns: List of episode returns
        title: Plot title
        save_path: Optional path to save figure

    Example:
        ```python
        from smlx.gym.utils.visualization import plot_reward_distribution

        returns = agent.episode_returns
        plot_reward_distribution(returns, save_path="returns_dist.png")
        ```
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    plt.figure(figsize=(10, 6))

    # Histogram
    plt.hist(returns, bins=30, alpha=0.7, color="steelblue", edgecolor="black")

    # Statistics
    mean_return = float(np.mean(returns))
    median_return = float(np.median(returns))
    std_return = float(np.std(returns))

    plt.axvline(mean_return, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_return:.2f}")
    plt.axvline(median_return, color="green", linestyle="--", linewidth=2, label=f"Median: {median_return:.2f}")

    plt.xlabel("Episode Return")
    plt.ylabel("Frequency")
    plt.title(f"{title}\nStd: {std_return:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def create_training_dashboard(
    episodes: list[int],
    returns: list[float],
    losses: list[float],
    episode_lengths: list[int],
    save_path: Optional[str] = None,
):
    """
    Create comprehensive training dashboard with multiple plots.

    Args:
        episodes: List of episode numbers
        returns: List of episode returns
        losses: List of losses
        episode_lengths: List of episode lengths
        save_path: Optional path to save figure

    Example:
        ```python
        from smlx.gym.utils.visualization import create_training_dashboard

        create_training_dashboard(
            episodes=list(range(len(agent.episode_returns))),
            returns=agent.episode_returns,
            losses=agent.losses,
            episode_lengths=agent.episode_lengths,
            save_path="dashboard.png"
        )
        ```
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Returns over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(episodes, returns, alpha=0.6, label="Return")
    window = min(10, len(returns) // 10)
    if window > 1:
        returns_ma = np.convolve(returns, np.ones(window) / window, mode="valid")
        episodes_ma = episodes[window - 1 :]
        ax1.plot(episodes_ma, returns_ma, linewidth=2, label=f"MA({window})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.set_title("Episode Returns")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Loss over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(episodes[: len(losses)], losses, color="red", alpha=0.6)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss")
    ax2.grid(True, alpha=0.3)

    # 3. Episode lengths over time
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(episodes, episode_lengths, color="green", alpha=0.6)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Length")
    ax3.set_title("Episode Lengths")
    ax3.grid(True, alpha=0.3)

    # 4. Return distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(returns, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    mean_return = float(np.mean(returns))
    ax4.axvline(mean_return, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_return:.2f}")
    ax4.set_xlabel("Return")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Return Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Summary statistics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    stats_text = f"""
    Summary Statistics:

    Episodes: {len(episodes)}

    Returns:
      Mean: {np.mean(returns):.2f}
      Median: {np.median(returns):.2f}
      Std: {np.std(returns):.2f}
      Min: {np.min(returns):.2f}
      Max: {np.max(returns):.2f}

    Episode Lengths:
      Mean: {np.mean(episode_lengths):.1f}
      Median: {np.median(episode_lengths):.1f}

    Final Loss: {losses[-1]:.4f}
    """
    ax5.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment="center", family="monospace")

    plt.suptitle("Training Dashboard", fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved dashboard to {save_path}")
    else:
        plt.show()

    plt.close()
