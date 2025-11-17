"""
DQN training example on CartPole.

Demonstrates:
- Training a DQN agent
- Using experience replay
- Epsilon-greedy exploration with decay
- Evaluating trained agent
"""

import gymnasium as gym

from smlx.gym.algorithms.dqn import DQNAgent
from smlx.gym.wrappers import RecordEpisodeStatistics


def main():
    """Train DQN agent on CartPole."""
    # Create environment
    env = gym.make("CartPole-v1")
    env = RecordEpisodeStatistics(env)

    # Get environment dimensions
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("=" * 60)
    print("DQN Training on CartPole")
    print("=" * 60)
    print(f"Environment: {env.spec.id}")
    print(f"Observation dim: {observation_dim}")
    print(f"Action dim: {action_dim}")
    print("=" * 60)
    print()

    # Create DQN agent
    agent = DQNAgent(
        env=env,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=128,
        num_hidden_layers=2,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        target_update_freq=10,
        min_buffer_size=1000,
    )

    # Training loop
    num_episodes = 200
    print_every = 20

    print(f"Training for {num_episodes} episodes...")
    print(f"Epsilon decay: {agent.epsilon} -> {agent.epsilon_min}")
    print()

    for episode in range(1, num_episodes + 1):
        # Run episode
        observation, _ = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(observation)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.add(observation, action, float(reward), next_obs, done)

            # Train if enough experiences
            if len(agent.replay_buffer) >= agent.min_buffer_size:
                batch = agent.replay_buffer.sample(agent.batch_size)
                agent.train_step(batch)

            episode_return += float(reward)
            episode_length += 1
            observation = next_obs

        # Update target network
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Print progress
        if episode % print_every == 0:
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Return: {episode_return:.1f} | "
                f"Length: {episode_length} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.replay_buffer)}"
            )

    print()
    print("Training complete!")
    print()

    # Evaluate trained agent
    print("=" * 60)
    print("Evaluating Trained Agent (epsilon=0)")
    print("=" * 60)

    # Set epsilon to 0 for greedy evaluation
    agent.epsilon = 0.0

    eval_episodes = 20
    eval_returns = []

    for episode in range(eval_episodes):
        observation, _ = env.reset()
        episode_return = 0.0
        done = False

        while not done:
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += float(reward)

        eval_returns.append(episode_return)

    # Print evaluation results
    import numpy as np

    mean_return = np.mean(eval_returns)
    std_return = np.std(eval_returns)
    min_return = np.min(eval_returns)
    max_return = np.max(eval_returns)

    print(f"Evaluation over {eval_episodes} episodes:")
    print(f"  Mean return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"  Min return: {min_return:.2f}")
    print(f"  Max return: {max_return:.2f}")
    print("=" * 60)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
