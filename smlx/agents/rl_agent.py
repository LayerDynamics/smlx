"""
Reinforcement Learning Agent implementation.

Extends SMLX BaseAgent for RL environments, providing episode-based
interaction and training functionality.

Reference: Gym_OpenAI.md, lines 125-206
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import mlx.core as mx

from smlx.agents.base import AgentResponse, BaseAgent


@dataclass
class RLAgentResponse(AgentResponse):
    """
    Extended response for RL agents.

    Adds RL-specific metrics like episode return, length, and success rate.
    """

    episode_return: float = 0.0
    episode_length: int = 0
    success: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    # Training metrics
    episode: int = 0
    average_return: float = 0.0
    average_length: float = 0.0
    training_time: float = 0.0


class RLAgent(BaseAgent):
    """
    Base class for reinforcement learning agents.

    Integrates with SMLX agent system while providing RL-specific functionality
    for interacting with Gymnasium environments.

    This class handles:
    - Episode-based interaction with environments
    - Action selection (to be implemented by subclasses)
    - Training (to be implemented by subclasses)
    - Episode statistics tracking

    Attributes:
        env: Gymnasium environment
        observation_space: Environment's observation space
        action_space: Environment's action space
        episode_count: Number of episodes completed
        epsilon: Exploration rate (optional, for exploration strategies)
        update_count: Number of parameter updates (optional, for learning agents)
        replay_buffer: Experience replay buffer (optional, for off-policy agents)
        batch_size: Training batch size (optional, for learning agents)
        min_buffer_size: Minimum buffer size before training (optional)

    Example:
        ```python
        class MyRLAgent(RLAgent):
            def select_action(self, observation):
                # Random policy
                return self.env.action_space.sample()

        env = gym.make("CartPole-v1")
        agent = MyRLAgent(env)
        response = agent.run(num_episodes=10)
        print(f"Average return: {response.episode_return}")
        ```
    """

    def __init__(self, env: gym.Env, **kwargs):
        """
        Initialize RL agent.

        Args:
            env: Gymnasium environment
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(**kwargs)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.episode_count = 0

        # Optional attributes for learning agents
        # Subclasses can override these as needed
        self.epsilon: Optional[float] = None
        self.update_count: int = 0
        self.replay_buffer: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.min_buffer_size: int = 1000

    def run(
        self, num_episodes: int = 1, max_steps: Optional[int] = None, **kwargs
    ) -> RLAgentResponse:
        """
        Run agent in environment for specified episodes.

        This is the main entry point for executing the agent. It runs
        multiple episodes and returns aggregated statistics.

        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode (None for no limit)
            **kwargs: Additional arguments (passed to subclass methods)

        Returns:
            RLAgentResponse with episode statistics

        Example:
            ```python
            agent = MyRLAgent(env)

            # Run for 100 episodes
            response = agent.run(num_episodes=100)
            print(f"Success rate: {response.metadata['success_rate']}")

            # Run with step limit
            response = agent.run(num_episodes=10, max_steps=500)
            ```
        """
        import time

        start_time = time.time()
        total_return = 0.0
        total_length = 0
        success_count = 0
        episode_returns = []

        for _ in range(num_episodes):
            episode_return, episode_length, success = self._run_episode(max_steps)
            total_return += episode_return
            total_length += episode_length
            success_count += int(success)
            episode_returns.append(episode_return)
            self.episode_count += 1

        training_time = time.time() - start_time
        avg_return = total_return / num_episodes
        avg_length = total_length / num_episodes

        return RLAgentResponse(
            content=f"Completed {num_episodes} episodes",
            episode_return=avg_return,
            episode_length=int(avg_length),
            success=success_count > 0,
            reasoning=f"Average return: {avg_return:.2f}",
            episode=self.episode_count,
            average_return=avg_return,
            average_length=avg_length,
            training_time=training_time,
            metadata={
                "episodes": num_episodes,
                "success_rate": success_count / num_episodes,
                "total_steps": total_length,
                "episode_returns": episode_returns,
            },
        )

    def _run_episode(self, max_steps: Optional[int] = None) -> tuple[float, int, bool]:
        """
        Run single episode and return (return, length, success).

        This method executes one complete episode from reset to terminal state.
        Subclasses can override this for custom episode logic (e.g., training).

        Args:
            max_steps: Maximum steps (None for no limit)

        Returns:
            tuple of (episode_return, episode_length, success)

        Example:
            ```python
            class TrainingAgent(RLAgent):
                def _run_episode(self, max_steps=None):
                    # Custom episode logic with training
                    episode_return, episode_length, success = super()._run_episode(max_steps)
                    # Perform training updates
                    self.train()
                    return episode_return, episode_length, success
            ```
        """
        observation, info = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            action = self.select_action(observation)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            episode_return += float(reward)
            episode_length += 1
            observation = next_obs
            done = terminated or truncated

            if max_steps and episode_length >= max_steps:
                break

        success = info.get("success", False) if "success" in info else False
        return episode_return, episode_length, success

    def select_action(self, observation: mx.array) -> Any:
        """
        Select action given observation.

        This method must be implemented by subclasses to define the
        agent's policy (how it selects actions).

        Args:
            observation: Current observation from environment

        Returns:
            Action to take (type depends on action_space)

        Raises:
            NotImplementedError: If subclass doesn't implement this method

        Example:
            ```python
            class RandomAgent(RLAgent):
                def select_action(self, observation):
                    return self.env.action_space.sample()

            class NNAgent(RLAgent):
                def select_action(self, observation):
                    q_values = self.q_network(observation)
                    return int(mx.argmax(q_values))
            ```
        """
        raise NotImplementedError("Subclasses must implement select_action()")

    def train_step(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """
        Perform one training step.

        This method should be implemented by subclasses that learn from
        experience. It typically updates the agent's parameters using a
        batch of transitions.

        Args:
            batch: Dictionary containing batch data
                   (e.g., states, actions, rewards, next_states, dones)

        Returns:
            Dictionary of training metrics (e.g., loss, gradients)

        Raises:
            NotImplementedError: If subclass doesn't implement this method

        Example:
            ```python
            class DQNAgent(RLAgent):
                def train_step(self, batch):
                    states = batch["states"]
                    actions = batch["actions"]
                    rewards = batch["rewards"]

                    # Compute loss
                    loss = self.compute_dqn_loss(states, actions, rewards)

                    # Update network
                    self.optimizer.update(self.q_network, grads)

                    return {"loss": float(loss)}
            ```
        """
        raise NotImplementedError("Subclasses should implement train_step()")

    def save(self, path: str):
        """
        Save agent state to disk.

        Subclasses should override this to save model parameters,
        optimizer state, and any other necessary state.

        Args:
            path: Path to save agent state

        Example:
            ```python
            agent.save("checkpoints/agent_episode_1000.pkl")
            ```
        """
        raise NotImplementedError("Subclasses should implement save()")

    def load(self, path: str):
        """
        Load agent state from disk.

        Subclasses should override this to load model parameters,
        optimizer state, and any other necessary state.

        Args:
            path: Path to load agent state from

        Example:
            ```python
            agent = MyAgent(env)
            agent.load("checkpoints/agent_episode_1000.pkl")
            ```
        """
        raise NotImplementedError("Subclasses should implement load()")


class RandomAgent(RLAgent):
    """
    Random agent that samples actions uniformly.

    Useful for:
    - Baseline comparisons
    - Testing environment implementations
    - Collecting initial experience

    Example:
        ```python
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env)
        response = agent.run(num_episodes=10)
        print(f"Random policy return: {response.episode_return}")
        ```
    """

    def select_action(self, observation: mx.array) -> Any:
        """Select random action"""
        return self.env.action_space.sample()


class GreedyAgent(RLAgent):
    """
    Greedy agent that always selects best action according to Q-function.

    Used for:
    - Evaluation (no exploration)
    - Deterministic policy execution

    Attributes:
        q_function: Function mapping (state, action) -> Q-value

    Example:
        ```python
        def my_q_function(state, action):
            # Return Q-value for state-action pair
            return 0.0

        agent = GreedyAgent(env, q_function=my_q_function)
        response = agent.run(num_episodes=10)
        ```
    """

    def __init__(self, env: gym.Env, q_function: Any, **kwargs):
        """
        Initialize greedy agent.

        Args:
            env: Gymnasium environment
            q_function: Function or network for Q-values
            **kwargs: Additional arguments passed to RLAgent
        """
        super().__init__(env, **kwargs)
        self.q_function = q_function

    def select_action(self, observation: mx.array) -> Any:
        """Select greedy action (argmax Q-value)"""
        if isinstance(self.action_space, gym.spaces.Discrete):
            # For discrete actions, evaluate all and take argmax
            q_values = []
            for action in range(self.action_space.n):
                q_value = self.q_function(observation, action)
                q_values.append(q_value)
            return int(mx.argmax(mx.array(q_values)))
        else:
            # For continuous actions, need different strategy
            raise NotImplementedError(
                "Greedy action selection for continuous actions not implemented"
            )


class EpsilonGreedyAgent(RLAgent):
    """
    Epsilon-greedy agent that balances exploration and exploitation.

    Selects random action with probability epsilon, otherwise selects
    greedy action according to Q-function.

    Attributes:
        q_function: Function mapping (state, action) -> Q-value
        epsilon: Exploration probability

    Example:
        ```python
        def my_q_function(state, action):
            return 0.0

        agent = EpsilonGreedyAgent(
            env,
            q_function=my_q_function,
            epsilon=0.1  # 10% exploration
        )
        response = agent.run(num_episodes=100)
        ```
    """

    def __init__(self, env: gym.Env, q_function: Any, epsilon: float = 0.1, **kwargs):
        """
        Initialize epsilon-greedy agent.

        Args:
            env: Gymnasium environment
            q_function: Function or network for Q-values
            epsilon: Exploration probability [0, 1]
            **kwargs: Additional arguments passed to RLAgent
        """
        super().__init__(env, **kwargs)
        self.q_function = q_function
        self.epsilon = epsilon

    def select_action(self, observation: mx.array) -> Any:
        """Select epsilon-greedy action"""
        import random

        # self.epsilon is guaranteed to be set in __init__
        assert self.epsilon is not None, "epsilon should be set in __init__"
        if random.random() < self.epsilon:
            # Explore: random action
            return self.env.action_space.sample()
        else:
            # Exploit: greedy action
            if isinstance(self.action_space, gym.spaces.Discrete):
                q_values = []
                for action in range(self.action_space.n):
                    q_value = self.q_function(observation, action)
                    q_values.append(q_value)
                return int(mx.argmax(mx.array(q_values)))
            else:
                raise NotImplementedError(
                    "Greedy action selection for continuous actions not implemented"
                )
