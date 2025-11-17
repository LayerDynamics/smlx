"""
Benchmark suite for reinforcement learning agents.

Provides functions for benchmarking RL agent performance across various
environments and metrics including episode return, success rate, and training speed.

Reference: SMLX_Gym.md, Section 5 (Benchmarks)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import gymnasium as gym
import mlx.core as mx

from smlx.agents.rl_agent import RLAgent
from smlx.utils.memory import clear_cache, reset_peak_memory


@dataclass
class RLBenchmarkConfig:
    """Configuration for RL benchmarks."""

    num_episodes: int = 100
    """Number of episodes to run"""

    max_steps_per_episode: int = 1000
    """Maximum steps per episode"""

    num_eval_episodes: int = 10
    """Number of episodes for final evaluation"""

    warmup_episodes: int = 5
    """Number of warmup episodes (not counted in benchmark)"""

    measure_training_time: bool = True
    """Whether to measure training time separately"""

    measure_inference_time: bool = True
    """Whether to measure inference time separately"""

    seed: int = 0
    """Random seed for reproducibility"""

    record_video: bool = False
    """Whether to record video of episodes"""

    video_freq: int = 100
    """Record video every N episodes"""


@dataclass
class RLBenchmarkStats:
    """Statistics from RL benchmark."""

    # Episode metrics
    mean_episode_return: float
    """Average episode return"""

    std_episode_return: float
    """Standard deviation of episode return"""

    mean_episode_length: float
    """Average episode length"""

    success_rate: float
    """Success rate (if applicable)"""

    # Performance metrics
    total_time: float
    """Total benchmark time (seconds)"""

    training_time: float
    """Time spent training (seconds)"""

    inference_time: float
    """Time spent on inference (seconds)"""

    steps_per_second: float
    """Environment steps per second"""

    episodes_per_second: float
    """Episodes per second"""

    # Memory metrics
    peak_memory_mb: float
    """Peak memory usage (MB)"""

    # Training metrics
    final_epsilon: float = 0.0
    """Final exploration rate (if applicable)"""

    total_updates: int = 0
    """Total number of parameter updates"""

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional benchmark metadata"""

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "mean_episode_return": self.mean_episode_return,
            "std_episode_return": self.std_episode_return,
            "mean_episode_length": self.mean_episode_length,
            "success_rate": self.success_rate,
            "total_time": self.total_time,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "steps_per_second": self.steps_per_second,
            "episodes_per_second": self.episodes_per_second,
            "peak_memory_mb": self.peak_memory_mb,
            "final_epsilon": self.final_epsilon,
            "total_updates": self.total_updates,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """Pretty print statistics."""
        lines = [
            "=" * 60,
            "RL Benchmark Results",
            "=" * 60,
            "",
            "Episode Metrics:",
            f"  Mean Return:        {self.mean_episode_return:.2f} � {self.std_episode_return:.2f}",
            f"  Mean Length:        {self.mean_episode_length:.1f}",
            f"  Success Rate:       {self.success_rate * 100:.1f}%",
            "",
            "Performance Metrics:",
            f"  Total Time:         {self.total_time:.2f}s",
            f"  Training Time:      {self.training_time:.2f}s",
            f"  Inference Time:     {self.inference_time:.2f}s",
            f"  Steps/Second:       {self.steps_per_second:.1f}",
            f"  Episodes/Second:    {self.episodes_per_second:.2f}",
            "",
            "Memory:",
            f"  Peak Memory:        {self.peak_memory_mb:.1f} MB",
            "",
            "Training:",
            f"  Final Epsilon:      {self.final_epsilon:.4f}",
            f"  Total Updates:      {self.total_updates}",
            "=" * 60,
        ]
        return "\n".join(lines)


def benchmark_rl_agent(
    agent: RLAgent,
    env: Optional[gym.Env] = None,
    config: Optional[RLBenchmarkConfig] = None,
) -> RLBenchmarkStats:
    """
    Benchmark an RL agent.

    Measures agent performance across multiple episodes, tracking returns,
    success rates, training speed, and memory usage.

    Args:
        agent: RL agent to benchmark
        env: Environment (uses agent.env if not provided)
        config: Benchmark configuration

    Returns:
        RLBenchmarkStats with performance metrics

    Example:
        ```python
        import gymnasium as gym
        from smlx.agents.rl_agent import RandomAgent
        from smlx.bench.suites.rl import benchmark_rl_agent, RLBenchmarkConfig

        # Create environment and agent
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)

        # Run benchmark
        config = RLBenchmarkConfig(num_episodes=100)
        stats = benchmark_rl_agent(agent, config=config)

        print(stats)
        ```
    """
    if config is None:
        config = RLBenchmarkConfig()

    if env is None:
        env = agent.env

    # Reset memory tracking
    reset_peak_memory()
    clear_cache()

    # Set seed
    mx.random.seed(config.seed)
    env.reset(seed=config.seed)

    # Warmup episodes
    if config.warmup_episodes > 0:
        for _ in range(config.warmup_episodes):
            _run_single_episode(agent, env, config.max_steps_per_episode)

    # Clear cache after warmup
    clear_cache()

    # Tracking metrics
    episode_returns = []
    episode_lengths = []
    episode_successes = []

    total_start_time = time.time()
    total_training_time = 0.0
    total_inference_time = 0.0
    total_steps = 0

    # Benchmark episodes
    for _ in range(config.num_episodes):
        # Run episode with timing
        (
            episode_return,
            episode_length,
            success,
            training_time,
            inference_time,
        ) = _run_episode_with_timing(
            agent,
            env,
            config.max_steps_per_episode,
            measure_training=config.measure_training_time,
            measure_inference=config.measure_inference_time,
        )

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_successes.append(1.0 if success else 0.0)

        total_training_time += training_time
        total_inference_time += inference_time
        total_steps += episode_length

    total_time = time.time() - total_start_time

    # Get peak memory
    peak_memory_mb = mx.metal.get_peak_memory() / (1024**2) if mx.metal.is_available() else 0.0

    # Compute statistics
    import numpy as np

    mean_return = float(np.mean(episode_returns))
    std_return = float(np.std(episode_returns))
    mean_length = float(np.mean(episode_lengths))
    success_rate = float(np.mean(episode_successes))

    steps_per_second = total_steps / total_time if total_time > 0 else 0.0
    episodes_per_second = config.num_episodes / total_time if total_time > 0 else 0.0

    # Get agent-specific metrics
    final_epsilon = 0.0
    total_updates = 0

    if agent.epsilon is not None:
        final_epsilon = agent.epsilon

    total_updates = agent.update_count

    # Build metadata
    metadata = {
        "env_name": env.spec.id if hasattr(env, "spec") and env.spec else "Unknown",
        "agent_type": type(agent).__name__,
        "num_episodes": config.num_episodes,
        "max_steps_per_episode": config.max_steps_per_episode,
        "seed": config.seed,
    }

    return RLBenchmarkStats(
        mean_episode_return=mean_return,
        std_episode_return=std_return,
        mean_episode_length=mean_length,
        success_rate=success_rate,
        total_time=total_time,
        training_time=total_training_time,
        inference_time=total_inference_time,
        steps_per_second=steps_per_second,
        episodes_per_second=episodes_per_second,
        peak_memory_mb=peak_memory_mb,
        final_epsilon=final_epsilon,
        total_updates=total_updates,
        metadata=metadata,
    )


def benchmark_environment(
    env_id: str,
    agent_fn: Callable,
    config: Optional[RLBenchmarkConfig] = None,
    num_seeds: int = 3,
) -> dict[str, Any]:
    """
    Benchmark an agent on an environment across multiple seeds.

    Args:
        env_id: Gymnasium environment ID
        agent_fn: Function that creates agent given environment
        config: Benchmark configuration
        num_seeds: Number of random seeds to test

    Returns:
        Dictionary with aggregated statistics

    Example:
        ```python
        from smlx.bench.suites.rl import benchmark_environment
        from smlx.agents.rl_agent import RandomAgent

        def create_agent(env):
            return RandomAgent(env)

        results = benchmark_environment(
            env_id="CartPole-v1",
            agent_fn=create_agent,
            num_seeds=5
        )

        print(f"Mean return: {results['mean_return']:.2f}")
        ```
    """
    if config is None:
        config = RLBenchmarkConfig()

    all_returns = []
    all_lengths = []
    all_success_rates = []
    all_stats = []

    for seed in range(num_seeds):
        # Create environment
        env = gym.make(env_id)

        # Create agent
        agent = agent_fn(env)

        # Update config seed
        config.seed = seed

        # Run benchmark
        stats = benchmark_rl_agent(agent, env, config)

        all_returns.append(stats.mean_episode_return)
        all_lengths.append(stats.mean_episode_length)
        all_success_rates.append(stats.success_rate)
        all_stats.append(stats)

        # Cleanup
        env.close()

    # Aggregate results
    import numpy as np

    results = {
        "env_id": env_id,
        "num_seeds": num_seeds,
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "mean_length": float(np.mean(all_lengths)),
        "mean_success_rate": float(np.mean(all_success_rates)),
        "all_stats": all_stats,
    }

    return results


def compare_agents(
    agents: dict[str, RLAgent],
    env: gym.Env,
    config: Optional[RLBenchmarkConfig] = None,
) -> dict[str, RLBenchmarkStats]:
    """
    Compare multiple agents on the same environment.

    Args:
        agents: Dictionary mapping agent names to agent instances
        env: Environment to test on
        config: Benchmark configuration

    Returns:
        Dictionary mapping agent names to their benchmark stats

    Example:
        ```python
        import gymnasium as gym
        from smlx.agents.rl_agent import RandomAgent, GreedyAgent
        from smlx.bench.suites.rl import compare_agents

        env = gym.make("CartPole-v1")

        agents = {
            "random": RandomAgent(env),
            "greedy": GreedyAgent(env, q_function=lambda s, a: 0.0),
        }

        results = compare_agents(agents, env)

        for name, stats in results.items():
            print(f"{name}: {stats.mean_episode_return:.2f}")
        ```
    """
    results = {}

    for name, agent in agents.items():
        print(f"\nBenchmarking {name}...")
        stats = benchmark_rl_agent(agent, env, config)
        results[name] = stats

        # Print summary
        print(f"  Mean Return: {stats.mean_episode_return:.2f}")
        print(f"  Success Rate: {stats.success_rate * 100:.1f}%")

    return results


def _run_single_episode(
    agent: RLAgent, env: gym.Env, max_steps: int
) -> tuple[float, int, bool]:
    """Run a single episode without timing details."""
    observation, info = env.reset()

    if not isinstance(observation, mx.array):
        observation = mx.array(observation)

    episode_return = 0.0
    episode_length = 0
    done = False

    while not done and episode_length < max_steps:
        action = agent.select_action(observation)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not isinstance(next_obs, mx.array):
            next_obs = mx.array(next_obs)

        episode_return += float(reward)
        episode_length += 1
        observation = next_obs

    success = info.get("success", False) if "success" in info else False
    return episode_return, episode_length, success


def _run_episode_with_timing(
    agent: RLAgent,
    env: gym.Env,
    max_steps: int,
    measure_training: bool = True,
    measure_inference: bool = True,
) -> tuple[float, int, bool, float, float]:
    """Run episode and measure training/inference time separately."""
    observation, info = env.reset()

    if not isinstance(observation, mx.array):
        observation = mx.array(observation)

    episode_return = 0.0
    episode_length = 0
    done = False

    training_time = 0.0
    inference_time = 0.0

    while not done and episode_length < max_steps:
        # Measure inference time
        if measure_inference:
            inf_start = time.time()

        action = agent.select_action(observation)

        if measure_inference:
            inference_time += time.time() - inf_start

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not isinstance(next_obs, mx.array):
            next_obs = mx.array(next_obs)

        # Measure training time (if agent has train_step method)
        if measure_training and agent.replay_buffer is not None:
            # Store transition
            if hasattr(agent.replay_buffer, "add"):
                agent.replay_buffer.add(observation, action, float(reward), next_obs, done)

            # Train if enough experiences
            if (
                len(agent.replay_buffer) >= agent.min_buffer_size
                and hasattr(agent, "train_step")
                and agent.batch_size is not None
            ):
                train_start = time.time()

                batch = agent.replay_buffer.sample(agent.batch_size)
                agent.train_step(batch)

                training_time += time.time() - train_start

        episode_return += float(reward)
        episode_length += 1
        observation = next_obs

    success = info.get("success", False) if "success" in info else False
    return episode_return, episode_length, success, training_time, inference_time
