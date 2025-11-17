"""
RL agent evaluation framework.

Provides comprehensive evaluation utilities for trained RL agents including
deterministic evaluation, comparison against baselines, and detailed metrics reporting.

Reference: SMLX_Gym.md, Section 6 (Evaluation)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mlx.core as mx
import numpy as np

from smlx.agents.rl_agent import RandomAgent, RLAgent
from smlx.utils.memory import clear_cache, reset_peak_memory


@dataclass
class EvalConfig:
    """Configuration for RL agent evaluation."""

    num_episodes: int = 100
    """Number of evaluation episodes"""

    max_steps_per_episode: int = 1000
    """Maximum steps per episode"""

    seed: int = 42
    """Random seed for reproducibility"""

    deterministic: bool = True
    """Whether to use deterministic policy (no exploration)"""

    render: bool = False
    """Whether to render episodes"""

    save_trajectories: bool = False
    """Whether to save episode trajectories"""

    video_freq: Optional[int] = None
    """Record video every N episodes (None to disable)"""

    checkpoint_path: Optional[str] = None
    """Path to agent checkpoint to load"""


@dataclass
class EvalResults:
    """Results from RL agent evaluation."""

    # Episode metrics
    mean_return: float
    """Mean episode return"""

    std_return: float
    """Standard deviation of returns"""

    min_return: float
    """Minimum episode return"""

    max_return: float
    """Maximum episode return"""

    median_return: float
    """Median episode return"""

    mean_length: float
    """Mean episode length"""

    std_length: float
    """Standard deviation of episode length"""

    success_rate: float
    """Success rate (fraction of successful episodes)"""

    # Performance metrics
    total_time: float
    """Total evaluation time (seconds)"""

    steps_per_second: float
    """Environment steps per second"""

    # Memory metrics
    peak_memory_mb: float
    """Peak memory usage (MB)"""

    # Raw data
    episode_returns: list[float] = field(default_factory=list)
    """List of all episode returns"""

    episode_lengths: list[int] = field(default_factory=list)
    """List of all episode lengths"""

    episode_successes: list[bool] = field(default_factory=list)
    """List of episode success flags"""

    trajectories: list[dict[str, Any]] = field(default_factory=list)
    """Episode trajectories (if save_trajectories=True)"""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional evaluation metadata"""

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "median_return": self.median_return,
            "mean_length": self.mean_length,
            "std_length": self.std_length,
            "success_rate": self.success_rate,
            "total_time": self.total_time,
            "steps_per_second": self.steps_per_second,
            "peak_memory_mb": self.peak_memory_mb,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "episode_successes": self.episode_successes,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """Pretty print evaluation results."""
        lines = [
            "=" * 60,
            "RL Evaluation Results",
            "=" * 60,
            "",
            "Episode Metrics:",
            f"  Mean Return:        {self.mean_return:.2f} � {self.std_return:.2f}",
            f"  Median Return:      {self.median_return:.2f}",
            f"  Min/Max Return:     {self.min_return:.2f} / {self.max_return:.2f}",
            f"  Mean Length:        {self.mean_length:.1f} � {self.std_length:.1f}",
            f"  Success Rate:       {self.success_rate * 100:.1f}%",
            "",
            "Performance:",
            f"  Total Time:         {self.total_time:.2f}s",
            f"  Steps/Second:       {self.steps_per_second:.1f}",
            f"  Peak Memory:        {self.peak_memory_mb:.1f} MB",
            "",
            f"Episodes Evaluated:   {len(self.episode_returns)}",
            "=" * 60,
        ]
        return "\n".join(lines)


def evaluate_agent(
    agent: RLAgent,
    env: Optional[gym.Env] = None,
    config: Optional[EvalConfig] = None,
) -> EvalResults:
    """
    Evaluate a trained RL agent.

    Runs agent in deterministic mode (no exploration) for comprehensive
    evaluation across multiple episodes.

    Args:
        agent: Trained RL agent to evaluate
        env: Environment (uses agent.env if not provided)
        config: Evaluation configuration

    Returns:
        EvalResults with comprehensive metrics

    Example:
        ```python
        import gymnasium as gym
        from smlx.agents.rl_agent import RandomAgent
        from smlx.evals.rl_eval import evaluate_agent, EvalConfig

        # Create and train agent
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        # Evaluate
        config = EvalConfig(num_episodes=100, deterministic=True)
        results = evaluate_agent(agent, config=config)

        print(results)
        print(f"Mean return: {results.mean_return:.2f}")
        ```
    """
    if config is None:
        config = EvalConfig()

    if env is None:
        env = agent.env

    # Load checkpoint if provided
    if config.checkpoint_path is not None:
        agent.load(config.checkpoint_path)

    # Reset memory tracking
    reset_peak_memory()
    clear_cache()

    # Set seed
    mx.random.seed(config.seed)
    env.reset(seed=config.seed)

    # Disable exploration for deterministic evaluation
    original_epsilon = None
    if config.deterministic and hasattr(agent, "epsilon"):
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0

    # Tracking metrics
    episode_returns = []
    episode_lengths = []
    episode_successes = []
    trajectories = []

    total_steps = 0
    start_time = time.time()

    # Run evaluation episodes
    for _ in range(config.num_episodes):
        observation, info = env.reset()

        if not isinstance(observation, mx.array):
            observation = mx.array(observation)

        episode_return = 0.0
        episode_length = 0
        done = False

        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "infos": [],
        } if config.save_trajectories else None

        while not done and episode_length < config.max_steps_per_episode:
            # Select action (deterministic if epsilon=0)
            action = agent.select_action(observation)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if not isinstance(next_obs, mx.array):
                next_obs = mx.array(next_obs)

            # Track trajectory
            if config.save_trajectories:
                assert trajectory is not None  # Type guard for Pylance
                trajectory["observations"].append(observation)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(float(reward))
                trajectory["infos"].append(info)

            # Render if requested
            if config.render:
                env.render()

            episode_return += float(reward)
            episode_length += 1
            total_steps += 1
            observation = next_obs

        # Record episode metrics
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        success = info.get("success", False) if "success" in info else False
        episode_successes.append(success)

        if config.save_trajectories:
            trajectories.append(trajectory)

    total_time = time.time() - start_time

    # Restore original epsilon
    if original_epsilon is not None:
        agent.epsilon = original_epsilon

    # Get peak memory
    peak_memory_mb = mx.metal.get_peak_memory() / (1024**2) if mx.metal.is_available() else 0.0

    # Compute statistics
    mean_return = float(np.mean(episode_returns))
    std_return = float(np.std(episode_returns))
    min_return = float(np.min(episode_returns))
    max_return = float(np.max(episode_returns))
    median_return = float(np.median(episode_returns))

    mean_length = float(np.mean(episode_lengths))
    std_length = float(np.std(episode_lengths))

    success_rate = float(np.mean(episode_successes))
    steps_per_second = total_steps / total_time if total_time > 0 else 0.0

    # Build metadata
    metadata = {
        "env_name": env.spec.id if hasattr(env, "spec") and env.spec else "Unknown",
        "agent_type": type(agent).__name__,
        "num_episodes": config.num_episodes,
        "deterministic": config.deterministic,
        "seed": config.seed,
    }

    return EvalResults(
        mean_return=mean_return,
        std_return=std_return,
        min_return=min_return,
        max_return=max_return,
        median_return=median_return,
        mean_length=mean_length,
        std_length=std_length,
        success_rate=success_rate,
        total_time=total_time,
        steps_per_second=steps_per_second,
        peak_memory_mb=peak_memory_mb,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_successes=episode_successes,
        trajectories=trajectories,
        metadata=metadata,
    )


def evaluate_multiple_agents(
    agents: dict[str, RLAgent],
    env: gym.Env,
    config: Optional[EvalConfig] = None,
) -> dict[str, EvalResults]:
    """
    Evaluate multiple agents on the same environment.

    Args:
        agents: Dictionary mapping agent names to agent instances
        env: Environment to evaluate on
        config: Evaluation configuration

    Returns:
        Dictionary mapping agent names to their evaluation results

    Example:
        ```python
        import gymnasium as gym
        from smlx.agents.rl_agent import RandomAgent
        from smlx.evals.rl_eval import evaluate_multiple_agents

        env = gym.make("CartPole-v1")

        agents = {
            "random": RandomAgent(env),
            "trained": RandomAgent(env),  # Would be trained agent
        }

        results = evaluate_multiple_agents(agents, env)

        for name, result in results.items():
            print(f"{name}: {result.mean_return:.2f}")
        ```
    """
    results = {}

    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        result = evaluate_agent(agent, env, config)
        results[name] = result

        # Print summary
        print(f"  Mean Return: {result.mean_return:.2f} � {result.std_return:.2f}")
        print(f"  Success Rate: {result.success_rate * 100:.1f}%")

    return results


def compare_to_baseline(
    agent: RLAgent,
    env: gym.Env,
    baseline: str = "random",
    config: Optional[EvalConfig] = None,
) -> dict[str, Any]:
    """
    Compare agent performance to a baseline.

    Args:
        agent: Agent to evaluate
        env: Environment
        baseline: Baseline type ("random" or custom agent)
        config: Evaluation configuration

    Returns:
        Dictionary with comparison results

    Example:
        ```python
        import gymnasium as gym
        from smlx.agents.rl_agent import RandomAgent
        from smlx.evals.rl_eval import compare_to_baseline

        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        comparison = compare_to_baseline(agent, env, baseline="random")

        print(f"Improvement: {comparison['improvement_percent']:.1f}%")
        ```
    """
    # Evaluate agent
    agent_results = evaluate_agent(agent, env, config)

    # Create baseline agent
    if baseline == "random":
        baseline_agent = RandomAgent(env)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    # Evaluate baseline
    baseline_results = evaluate_agent(baseline_agent, env, config)

    # Compute comparison metrics
    improvement = agent_results.mean_return - baseline_results.mean_return
    improvement_percent = (improvement / abs(baseline_results.mean_return)) * 100 if baseline_results.mean_return != 0 else 0.0

    return {
        "agent_results": agent_results,
        "baseline_results": baseline_results,
        "improvement": improvement,
        "improvement_percent": improvement_percent,
        "agent_better": agent_results.mean_return > baseline_results.mean_return,
    }


def generate_eval_report(
    results: dict[str, EvalResults],
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a formatted evaluation report for multiple agents.

    Args:
        results: Dictionary mapping agent names to evaluation results
        save_path: Optional path to save report as text file

    Returns:
        Formatted report string

    Example:
        ```python
        from smlx.evals.rl_eval import generate_eval_report

        # After evaluating multiple agents
        report = generate_eval_report(results, save_path="eval_report.txt")
        print(report)
        ```
    """
    lines = [
        "=" * 80,
        "RL Agent Evaluation Report",
        "=" * 80,
        "",
    ]

    # Sort agents by mean return
    sorted_agents = sorted(
        results.items(),
        key=lambda x: x[1].mean_return,
        reverse=True,
    )

    # Summary table
    lines.append("Summary (sorted by mean return):")
    lines.append("-" * 80)
    lines.append(
        f"{'Agent':<20} {'Mean Return':<15} {'Success Rate':<15} {'Mean Length':<15}"
    )
    lines.append("-" * 80)

    for name, result in sorted_agents:
        lines.append(
            f"{name:<20} "
            f"{result.mean_return:>7.2f} � {result.std_return:<5.2f} "
            f"{result.success_rate * 100:>6.1f}%        "
            f"{result.mean_length:>7.1f}"
        )

    lines.append("-" * 80)
    lines.append("")

    # Detailed results for each agent
    lines.append("Detailed Results:")
    lines.append("=" * 80)

    for name, result in sorted_agents:
        lines.append("")
        lines.append(f"Agent: {name}")
        lines.append("-" * 80)
        lines.append(str(result))

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)

    # Save if path provided
    if save_path is not None:
        Path(save_path).write_text(report)
        lines.append(f"\nReport saved to: {save_path}")

    return report


def save_eval_results(
    results: EvalResults,
    save_path: str,
    format: str = "json",
) -> None:
    """
    Save evaluation results to disk.

    Args:
        results: Evaluation results to save
        save_path: Path to save results
        format: Format to save in ("json" or "pickle")

    Example:
        ```python
        from smlx.evals.rl_eval import save_eval_results

        # After evaluation
        save_eval_results(results, "eval_results.json", format="json")
        ```
    """
    import json
    import pickle

    if format == "json":
        with open(save_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
    elif format == "pickle":
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_eval_results(load_path: str, format: str = "json") -> EvalResults:
    """
    Load evaluation results from disk.

    Args:
        load_path: Path to load results from
        format: Format to load from ("json" or "pickle")

    Returns:
        Loaded evaluation results

    Example:
        ```python
        from smlx.evals.rl_eval import load_eval_results

        results = load_eval_results("eval_results.json", format="json")
        print(results)
        ```
    """
    import json
    import pickle

    if format == "json":
        with open(load_path) as f:
            data = json.load(f)
        return EvalResults(**data)
    elif format == "pickle":
        with open(load_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")
