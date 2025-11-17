"""
Basic CartPole example with random agent.

Demonstrates:
- Creating a gym environment
- Using RandomAgent
- Running episodes
- Recording statistics with wrappers
"""

import gymnasium as gym

from smlx.agents.rl_agent import RandomAgent
from smlx.gym.wrappers import RecordEpisodeStatistics


def main():
    """Run basic CartPole example with random agent."""
    # Create environment with statistics recording
    env = gym.make("CartPole-v1")
    env = RecordEpisodeStatistics(env)

    # Create random agent
    agent = RandomAgent(env)

    print("=" * 60)
    print("Basic CartPole Example - Random Agent")
    print("=" * 60)
    print(f"Environment: {env.spec.id}")
    print(f"Agent: {type(agent).__name__}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("=" * 60)
    print()

    # Run episodes
    num_episodes = 10
    print(f"Running {num_episodes} episodes...")
    print()

    response = agent.run(num_episodes=num_episodes, max_steps=500)

    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Episodes completed: {response.metadata['episodes']}")
    print(f"Average return: {response.episode_return:.2f}")
    print(f"Average length: {response.episode_length:.1f}")
    print(f"Success rate: {response.metadata['success_rate'] * 100:.1f}%")
    print(f"Total steps: {response.metadata['total_steps']}")
    print()

    # Get statistics from wrapper
    if hasattr(env, "get_episode_statistics"):
        stats = env.get_episode_statistics()
        print("Episode Statistics (from wrapper):")
        print(f"  Mean return: {stats['mean_return']:.2f}")
        print(f"  Std return: {stats['std_return']:.2f}")
        print(f"  Mean length: {stats['mean_length']:.1f}")
        print(f"  Episodes: {stats['num_episodes']}")

    print("=" * 60)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
