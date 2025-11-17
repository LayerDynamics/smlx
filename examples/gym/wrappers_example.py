"""
Environment wrappers example.

Demonstrates:
- Using environment wrappers
- Chaining multiple wrappers
- Observation and reward normalization
- Episode statistics recording
- Creating custom wrapper pipelines
"""

import gymnasium as gym
import numpy as np

from smlx.agents.rl_agent import RandomAgent
from smlx.gym.wrappers import (
    ClipReward,
    EpisodeLogger,
    FrameStack,
    MLXObservationWrapper,
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
    TimeLimit,
    make_env_with_wrappers,
)


def basic_wrapper_example():
    """Basic wrapper usage."""
    print("=" * 60)
    print("Basic Wrapper Usage")
    print("=" * 60)

    # Create base environment
    env = gym.make("CartPole-v1")
    print(f"Base environment: {env.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print()

    # Add statistics recording wrapper
    env = RecordEpisodeStatistics(env)
    print("Added RecordEpisodeStatistics wrapper")

    # Add MLX observation wrapper
    env = MLXObservationWrapper(env)
    print("Added MLXObservationWrapper")
    print()

    # Run episode
    agent = RandomAgent(env)
    observation, _ = env.reset()
    print(f"Observation type after reset: {type(observation)}")
    print(f"Observation shape: {observation.shape}")
    print()

    # Run a few steps
    for step in range(5):
        action = agent.select_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1}: reward={reward:.2f}, done={terminated or truncated}")

        if terminated or truncated:
            if "episode" in info:
                print(f"Episode finished! Return: {info['episode']['r']:.2f}, Length: {info['episode']['l']}")
            break

    env.close()
    print()


def normalization_example():
    """Observation and reward normalization."""
    print("=" * 60)
    print("Observation and Reward Normalization")
    print("=" * 60)

    # Create environment with normalization
    env = gym.make("CartPole-v1")
    env = NormalizeObservation(env)
    env = NormalizeReward(env, gamma=0.99)
    env = MLXObservationWrapper(env)

    print(f"Environment: {env.spec.id}")
    print("Wrappers: NormalizeObservation, NormalizeReward, MLXObservationWrapper")
    print()

    # Run episode to show normalization
    agent = RandomAgent(env)
    observation, _ = env.reset()

    print("Normalized observations (first 5 steps):")
    for step in range(5):
        action = agent.select_action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

        print(f"Step {step + 1}:")
        print(f"  Observation: {np.array(observation)}")
        print(f"  Reward: {reward:.4f}")

        if terminated or truncated:
            break

    env.close()
    print()


def frame_stack_example():
    """Frame stacking for temporal context."""
    print("=" * 60)
    print("Frame Stacking")
    print("=" * 60)

    # Create environment with frame stacking
    env = gym.make("CartPole-v1")
    base_obs_shape = env.observation_space.shape
    print(f"Base observation shape: {base_obs_shape}")

    env = FrameStack(env, num_stack=4)
    stacked_obs_shape = env.observation_space.shape
    print(f"Stacked observation shape: {stacked_obs_shape}")
    print(f"Stack multiplier: {stacked_obs_shape[0] // base_obs_shape[0]}x")
    print()

    # Run episode
    agent = RandomAgent(env)
    observation, _ = env.reset()

    print(f"Observation after reset contains {observation.shape[0]} stacked frames")
    print(f"Each frame has {observation.shape[0] // 4} dimensions")
    print()

    # Take a few steps
    for step in range(3):
        action = agent.select_action(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        print(f"Step {step + 1}: observation shape = {observation.shape}")

        if terminated or truncated:
            break

    env.close()
    print()


def reward_clipping_example():
    """Reward clipping for stability."""
    print("=" * 60)
    print("Reward Clipping")
    print("=" * 60)

    # Create environment with reward clipping
    env = gym.make("CartPole-v1")
    env = ClipReward(env, min_reward=-1.0, max_reward=1.0)

    print(f"Environment: {env.spec.id}")
    print("Rewards clipped to [-1.0, 1.0]")
    print()

    # Run episode
    agent = RandomAgent(env)
    observation, _ = env.reset()

    print("Rewards (first 10 steps):")
    for step in range(10):
        action = agent.select_action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        print(f"  Step {step + 1}: reward = {reward:.2f}")

        if terminated or truncated:
            break

    env.close()
    print()


def comprehensive_wrapper_pipeline():
    """Comprehensive wrapper pipeline using factory function."""
    print("=" * 60)
    print("Comprehensive Wrapper Pipeline")
    print("=" * 60)

    # Create environment with multiple wrappers using factory
    env = make_env_with_wrappers(
        env_id="CartPole-v1",
        normalize_obs=True,
        normalize_reward=True,
        clip_reward=False,
        frame_stack=None,  # None to disable
        time_limit=500,
        record_stats=True,
    )

    print("Environment created with wrapper pipeline:")
    print("  - TimeLimit(500)")
    print("  - NormalizeObservation")
    print("  - NormalizeReward")
    print("  - RecordEpisodeStatistics")
    print("  - MLXObservationWrapper")
    print()

    # Run multiple episodes
    agent = RandomAgent(env)
    num_episodes = 5

    print(f"Running {num_episodes} episodes...")
    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

            if done and "episode" in info:
                print(
                    f"Episode {episode + 1}: "
                    f"Return={info['episode']['r']:.2f}, "
                    f"Length={info['episode']['l']}"
                )

    # Get episode statistics
    if hasattr(env, "get_episode_statistics"):
        stats = env.get_episode_statistics()
        print()
        print("Episode Statistics:")
        print(f"  Mean return: {stats['mean_return']:.2f}")
        print(f"  Std return: {stats['std_return']:.2f}")
        print(f"  Mean length: {stats['mean_length']:.1f}")
        print(f"  Episodes recorded: {stats['num_episodes']}")

    env.close()
    print()


def episode_logger_example():
    """Episode logging for debugging."""
    print("=" * 60)
    print("Episode Logging")
    print("=" * 60)

    # Create environment with episode logger
    env = gym.make("CartPole-v1")
    env = EpisodeLogger(env, log_every=1)  # Log every episode

    print(f"Environment: {env.spec.id}")
    print("Logger will print episode info after each episode")
    print()

    # Run episodes (logger will print automatically)
    agent = RandomAgent(env)
    for episode in range(3):
        observation, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()
    print()


def main():
    """Run all wrapper examples."""
    print("\n" + "=" * 60)
    print("Environment Wrappers Examples")
    print("=" * 60)
    print()

    basic_wrapper_example()
    normalization_example()
    frame_stack_example()
    reward_clipping_example()
    comprehensive_wrapper_pipeline()
    episode_logger_example()

    print("=" * 60)
    print("Wrapper Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
