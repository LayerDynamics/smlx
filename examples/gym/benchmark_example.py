"""
RL benchmarking example.

Demonstrates:
- Benchmarking RL agents
- Comparing multiple agents
- Using benchmark configuration
- Generating performance reports
"""

import gymnasium as gym

from smlx.agents.rl_agent import RandomAgent
from smlx.bench.suites.rl import (
    RLBenchmarkConfig,
    benchmark_rl_agent,
    compare_agents,
)


def basic_benchmark_example():
    """Basic agent benchmarking."""
    print("=" * 60)
    print("Basic Agent Benchmarking")
    print("=" * 60)

    # Create environment and agent
    env = gym.make("CartPole-v1")
    agent = RandomAgent(env)

    # Configure benchmark
    config = RLBenchmarkConfig(
        num_episodes=50,
        max_steps_per_episode=500,
        warmup_episodes=5,
        measure_training_time=True,
        measure_inference_time=True,
        seed=42,
    )

    print(f"Environment: {env.spec.id}")
    print(f"Agent: {type(agent).__name__}")
    print(f"Benchmark episodes: {config.num_episodes}")
    print(f"Warmup episodes: {config.warmup_episodes}")
    print()

    # Run benchmark
    print("Running benchmark...")
    stats = benchmark_rl_agent(agent, env, config)

    # Print results
    print()
    print(stats)

    env.close()
    print()


def compare_agents_example():
    """Compare multiple agents."""
    print("=" * 60)
    print("Comparing Multiple Agents")
    print("=" * 60)

    # Create environment
    env = gym.make("CartPole-v1")

    # Create different agents
    # In practice, you would have different agent types or trained vs untrained
    agents = {
        "Random-1": RandomAgent(env),
        "Random-2": RandomAgent(env),
    }

    print(f"Environment: {env.spec.id}")
    print(f"Agents to compare: {len(agents)}")
    print(f"Agent types: {list(agents.keys())}")
    print()

    # Configure benchmark
    config = RLBenchmarkConfig(
        num_episodes=30,
        max_steps_per_episode=500,
        warmup_episodes=3,
        seed=42,
    )

    # Compare agents
    results = compare_agents(agents, env, config)

    # Print comparison
    print()
    print("=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print()

    for name, stats in results.items():
        print(f"{name}:")
        print(f"  Mean Return: {stats.mean_episode_return:.2f} ± {stats.std_episode_return:.2f}")
        print(f"  Success Rate: {stats.success_rate * 100:.1f}%")
        print(f"  Steps/Second: {stats.steps_per_second:.1f}")
        print(f"  Peak Memory: {stats.peak_memory_mb:.1f} MB")
        print()

    env.close()


def detailed_metrics_example():
    """Benchmark with detailed metrics analysis."""
    print("=" * 60)
    print("Detailed Metrics Analysis")
    print("=" * 60)

    # Create environment and agent
    env = gym.make("CartPole-v1")
    agent = RandomAgent(env)

    # Configure comprehensive benchmark
    config = RLBenchmarkConfig(
        num_episodes=100,
        max_steps_per_episode=500,
        num_eval_episodes=20,
        warmup_episodes=10,
        measure_training_time=True,
        measure_inference_time=True,
        seed=42,
    )

    print(f"Running comprehensive benchmark with {config.num_episodes} episodes...")
    print()

    # Run benchmark
    stats = benchmark_rl_agent(agent, env, config)

    # Analyze metrics
    print("=" * 60)
    print("Performance Analysis")
    print("=" * 60)
    print()

    # Episode metrics
    print("Episode Metrics:")
    print(f"  Mean Return: {stats.mean_episode_return:.2f}")
    print(f"  Std Return: {stats.std_episode_return:.2f}")
    print(f"  Mean Length: {stats.mean_episode_length:.1f}")
    print(f"  Success Rate: {stats.success_rate * 100:.1f}%")
    print()

    # Performance metrics
    print("Performance Metrics:")
    print(f"  Total Time: {stats.total_time:.2f}s")
    print(f"  Training Time: {stats.training_time:.2f}s ({stats.training_time/stats.total_time*100:.1f}%)")
    print(f"  Inference Time: {stats.inference_time:.2f}s ({stats.inference_time/stats.total_time*100:.1f}%)")
    print(f"  Steps/Second: {stats.steps_per_second:.1f}")
    print(f"  Episodes/Second: {stats.episodes_per_second:.2f}")
    print()

    # Memory metrics
    print("Memory Metrics:")
    print(f"  Peak Memory: {stats.peak_memory_mb:.1f} MB")
    print()

    # Metadata
    print("Metadata:")
    for key, value in stats.metadata.items():
        print(f"  {key}: {value}")
    print()

    # Convert to dictionary and save
    stats_dict = stats.to_dict()
    print(f"Total metrics collected: {len(stats_dict)} keys")

    env.close()
    print()


def main():
    """Run all benchmarking examples."""
    print("\n" + "=" * 60)
    print("RL Benchmarking Examples")
    print("=" * 60)
    print()

    basic_benchmark_example()
    compare_agents_example()
    detailed_metrics_example()

    print("=" * 60)
    print("Benchmarking Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
