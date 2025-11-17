"""
Deep Q-Network (DQN) algorithm implementation.

This module provides a complete DQN implementation using MLX for Metal GPU
acceleration. Includes Q-network architecture, experience replay, target
networks, and epsilon-greedy exploration.

Reference: SMLX_Gym.md, Section 4.3 (Algorithms)
Paper: Mnih et al. (2015) - Human-level control through deep reinforcement learning
"""

from typing import Optional

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from smlx.agents.rl_agent import RLAgent
from smlx.gym.replay import ReplayBuffer


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """
    Flatten a nested dictionary by joining keys with dots.

    Args:
        d: Dictionary to flatten
        prefix: Prefix to prepend to keys

    Returns:
        Flattened dictionary
    """
    result = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            # Skip empty dicts (e.g., from ReLU layers with no parameters)
            if len(value) > 0:
                result.update(flatten_dict(value, new_key))
        elif isinstance(value, list):
            # Handle lists (e.g., layers in network parameters)
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    # Skip empty dicts (e.g., from ReLU layers with no parameters)
                    if len(item) > 0:
                        result.update(flatten_dict(item, f"{new_key}[{i}]"))
                else:
                    result[f"{new_key}[{i}]"] = item
        else:
            result[new_key] = value
    return result


def unflatten_dict(d: dict) -> dict:
    """
    Unflatten a dictionary by splitting keys on dots and handling array indices.

    Args:
        d: Flattened dictionary

    Returns:
        Nested dictionary
    """
    import re

    result = {}
    for key, value in d.items():
        parts = key.split(".")
        current = result

        for i, part in enumerate(parts[:-1]):
            # Check if part has array index notation like "layers[0]"
            match = re.match(r"^(.+)\[(\d+)\]$", part)
            if match:
                name, idx = match.groups()
                idx = int(idx)
                if name not in current:
                    current[name] = []
                # Extend list if needed
                while len(current[name]) <= idx:
                    current[name].append({})
                current = current[name][idx]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]

        # Handle last part
        last_part = parts[-1]
        match = re.match(r"^(.+)\[(\d+)\]$", last_part)
        if match:
            name, idx = match.groups()
            idx = int(idx)
            if name not in current:
                current[name] = []
            while len(current[name]) <= idx:
                current[name].append({})
            current[name][idx] = value
        else:
            current[last_part] = value

    return result


class QNetwork(nn.Module):
    """
    Q-Network for DQN.

    Simple feedforward network that estimates Q-values for each action
    given a state observation.

    Architecture:
        state -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(num_actions)

    Example:
        ```python
        import mlx.core as mx
        from smlx.gym.algorithms.dqn import QNetwork

        # Create Q-network
        q_net = QNetwork(
            observation_dim=4,
            action_dim=2,
            hidden_dim=128
        )

        # Forward pass
        state = mx.random.normal((1, 4))
        q_values = q_net(state)
        print(q_values.shape)  # (1, 2)
        ```
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
    ):
        """
        Initialize Q-network.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers (default: 2)
        """
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Build network layers
        layers = []

        # Input layer
        layers.append(nn.Linear(observation_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through Q-network.

        Args:
            x: State observations [batch_size, observation_dim]

        Returns:
            Q-values for each action [batch_size, action_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x


class DQNAgent(RLAgent):
    """
    Deep Q-Network (DQN) agent.

    Implements the DQN algorithm with:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Huber loss for stability

    The agent learns to approximate Q-values using a neural network and
    uses these to select actions that maximize expected return.

    Reference:
        Mnih et al. (2015) - Human-level control through deep reinforcement learning
        https://www.nature.com/articles/nature14236

    Example:
        ```python
        import gymnasium as gym
        from smlx.gym.algorithms.dqn import DQNAgent

        # Create environment
        env = gym.make("CartPole-v1")

        # Create DQN agent
        agent = DQNAgent(
            env=env,
            hidden_dim=128,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_capacity=10000,
            batch_size=64,
            target_update_freq=100
        )

        # Train agent
        response = agent.run(num_episodes=1000)
        print(f"Average return: {response.episode_return}")

        # Evaluate (with no exploration)
        agent.epsilon = 0.0
        eval_response = agent.run(num_episodes=10)
        ```
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        train_freq: int = 1,
        min_buffer_size: int = 1000,
        **kwargs,
    ):
        """
        Initialize DQN agent.

        Args:
            env: Gymnasium environment
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay factor per episode
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            train_freq: Steps between training updates
            min_buffer_size: Minimum buffer size before training
            **kwargs: Additional arguments passed to RLAgent
        """
        super().__init__(env, **kwargs)

        # Validate discrete action space
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("DQN only supports Discrete action spaces")

        # Get dimensions
        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_dim = int(mx.prod(mx.array(env.observation_space.shape)))
        else:
            raise ValueError("DQN only supports Box observation spaces")

        self.action_dim = int(env.action_space.n)

        # Hyperparameters
        self.gamma: float = gamma
        self.epsilon: float = epsilon_start
        self.epsilon_start: float = epsilon_start
        self.epsilon_end: float = epsilon_end
        self.epsilon_decay: float = epsilon_decay
        self.batch_size: int = batch_size
        self.target_update_freq: int = target_update_freq
        self.train_freq: int = train_freq
        self.min_buffer_size: int = min_buffer_size

        # Create Q-networks
        self.q_network = QNetwork(
            self.observation_dim,
            int(self.action_dim),
            hidden_dim,
            num_hidden_layers,
        )

        self.target_network = QNetwork(
            self.observation_dim,
            int(self.action_dim),
            hidden_dim,
            num_hidden_layers,
        )

        # Copy weights to target network
        self._update_target_network()

        # Create optimizer
        self.optimizer = optim.Adam(learning_rate=learning_rate)

        # Create replay buffer
        self.replay_buffer: ReplayBuffer = ReplayBuffer(capacity=buffer_capacity)

        # Training state
        self.step_count: int = 0
        self.update_count: int = 0
        self.losses: list[float] = []

    def select_action(self, observation: mx.array) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            observation: Current observation

        Returns:
            Selected action (int)
        """
        # Epsilon-greedy exploration
        if float(mx.random.uniform()) < self.epsilon:
            # Random action
            return int(self.env.action_space.sample())

        # Greedy action based on Q-values
        # Flatten observation if needed
        if len(observation.shape) > 1:
            obs = mx.reshape(observation, (-1,))
        else:
            obs = observation

        # Add batch dimension
        obs_batch = mx.expand_dims(obs, 0)

        # Get Q-values
        q_values = self.q_network(obs_batch)

        # Select action with highest Q-value
        action = int(mx.argmax(q_values[0]))

        return action

    def train_step(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """
        Perform one training step using batch from replay buffer.

        Args:
            batch: Batch from replay buffer containing:
                   states, actions, rewards, next_states, dones

        Returns:
            Dictionary with training metrics (loss)
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Flatten observations if needed
        if len(states.shape) > 2:
            batch_size = states.shape[0]
            states = mx.reshape(states, (batch_size, -1))
            next_states = mx.reshape(next_states, (batch_size, -1))

        def loss_fn(model):
            # Current Q-values
            q_values = model(states)
            # Gather Q-values for taken actions
            q_values_for_actions = mx.take_along_axis(
                q_values, mx.expand_dims(actions.astype(mx.int32), 1), axis=1
            ).squeeze(1)

            # Target Q-values (no gradient through target network)
            next_q_values = mx.stop_gradient(self.target_network(next_states))
            max_next_q_values = mx.max(next_q_values, axis=1)

            # Compute targets: r + gamma * max_a' Q(s', a') * (1 - done)
            targets = rewards + self.gamma * max_next_q_values * (1.0 - dones)

            # Huber loss for stability
            loss = mx.mean(self._huber_loss(q_values_for_actions - targets))

            return loss

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(self.q_network)

        # Update Q-network
        self.optimizer.update(self.q_network, grads)

        # Evaluate parameters to complete the update
        mx.eval(self.q_network.parameters())

        self.update_count += 1

        return {"loss": float(loss)}

    def _huber_loss(self, x: mx.array, delta: float = 1.0) -> mx.array:
        """
        Huber loss for stable training.

        Args:
            x: Input tensor
            delta: Threshold for switching between L1 and L2 loss

        Returns:
            Huber loss
        """
        abs_x = mx.abs(x)
        quadratic = mx.minimum(abs_x, delta)
        linear = abs_x - quadratic
        return 0.5 * quadratic**2 + delta * linear

    def _update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.update(self.q_network.parameters())

    def _run_episode(self, max_steps: Optional[int] = None) -> tuple[float, int, bool]:
        """
        Run single episode with training.

        Overrides base class to add DQN-specific training logic.

        Args:
            max_steps: Maximum steps per episode

        Returns:
            tuple of (episode_return, episode_length, success)
        """
        observation, info = self.env.reset()

        # Convert observation to MLX array
        if not isinstance(observation, mx.array):
            observation = mx.array(observation)

        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            # Select action
            action = self.select_action(observation)

            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Convert next observation to MLX array
            if not isinstance(next_obs, mx.array):
                next_obs = mx.array(next_obs)

            # Store transition in replay buffer
            self.replay_buffer.add(
                state=observation,
                action=action,
                reward=float(reward),
                next_state=next_obs,
                done=done,
            )

            # Train if enough experiences
            if (
                len(self.replay_buffer) >= self.min_buffer_size
                and self.step_count % self.train_freq == 0
            ):
                batch = self.replay_buffer.sample(self.batch_size)
                # Type note: batch values should all be mx.array, not list
                metrics = self.train_step(batch)  # type: ignore[arg-type]
                self.losses.append(metrics["loss"])

            # Update target network periodically
            if self.step_count % self.target_update_freq == 0:
                self._update_target_network()

            episode_return += float(reward)
            episode_length += 1
            observation = next_obs
            self.step_count += 1

            if max_steps and episode_length >= max_steps:
                break

        # Decay epsilon after episode
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        success = info.get("success", False) if "success" in info else False
        return episode_return, episode_length, success

    def save(self, path: str):
        """
        Save agent state to disk.

        Args:
            path: Path to save agent state
        """
        # Flatten nested dictionaries and ensure all values are MLX arrays
        state = {}

        # Save Q-network parameters
        for key, value in flatten_dict(self.q_network.parameters(), "q_network").items():
            state[key] = value

        # Save target network parameters
        for key, value in flatten_dict(self.target_network.parameters(), "target_network").items():
            state[key] = value

        # Save optimizer state (flatten and convert scalars to arrays)
        for key, value in flatten_dict(self.optimizer.state, "optimizer").items():
            if isinstance(value, mx.array):
                state[key] = value
            elif isinstance(value, (int, float)):
                state[key] = mx.array(value)
            # Skip other types that can't be serialized

        # Save metadata as arrays
        state["epsilon"] = mx.array(self.epsilon)
        state["step_count"] = mx.array(self.step_count)
        state["update_count"] = mx.array(self.update_count)

        mx.save_safetensors(path, state)

    def load(self, path: str):
        """
        Load agent state from disk.

        Args:
            path: Path to load agent state from
        """
        loaded_state = mx.load(path)
        if not isinstance(loaded_state, dict):
            raise ValueError(f"Expected to load a state dict, but got {type(loaded_state)}")
        state: dict[str, mx.array] = loaded_state

        # Extract Q-network parameters
        q_params = {}
        target_params = {}
        optimizer_state = {}

        for key, value in state.items():
            if key.startswith("q_network."):
                q_params[key[len("q_network.") :]] = value
            elif key.startswith("target_network."):
                target_params[key[len("target_network.") :]] = value
            elif key.startswith("optimizer."):
                optimizer_state[key[len("optimizer.") :]] = value

        # Unflatten and update networks
        self.q_network.update(unflatten_dict(q_params))
        self.target_network.update(unflatten_dict(target_params))
        self.optimizer.state = unflatten_dict(optimizer_state)  # type: ignore[assignment]

        # Load metadata
        self.epsilon = float(state["epsilon"])
        self.step_count = int(state["step_count"])
        self.update_count = int(state["update_count"])


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN agent.

    Extends DQN with Double Q-learning to reduce overestimation bias.
    Uses online network to select actions and target network to evaluate them.

    Reference:
        van Hasselt et al. (2016) - Deep Reinforcement Learning with Double Q-learning
        https://arxiv.org/abs/1509.06461

    Example:
        ```python
        import gymnasium as gym
        from smlx.gym.algorithms.dqn import DoubleDQNAgent

        env = gym.make("CartPole-v1")
        agent = DoubleDQNAgent(env=env, hidden_dim=128)

        response = agent.run(num_episodes=1000)
        ```
    """

    def train_step(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """
        Perform one training step with Double DQN target computation.

        Args:
            batch: Batch from replay buffer

        Returns:
            Dictionary with training metrics
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Flatten observations if needed
        if len(states.shape) > 2:
            batch_size = states.shape[0]
            states = mx.reshape(states, (batch_size, -1))
            next_states = mx.reshape(next_states, (batch_size, -1))

        def loss_fn(model):
            # Current Q-values
            q_values = model(states)
            q_values_for_actions = mx.take_along_axis(
                q_values, mx.expand_dims(actions.astype(mx.int32), 1), axis=1
            ).squeeze(1)

            # Double DQN target computation
            # Use online network to select actions
            next_q_values_online = mx.stop_gradient(model(next_states))
            next_actions = mx.argmax(next_q_values_online, axis=1)

            # Use target network to evaluate actions
            next_q_values_target = mx.stop_gradient(self.target_network(next_states))
            max_next_q_values = mx.take_along_axis(
                next_q_values_target,
                mx.expand_dims(next_actions.astype(mx.int32), 1),
                axis=1,
            ).squeeze(1)

            targets = rewards + self.gamma * max_next_q_values * (1.0 - dones)

            # Huber loss
            loss = mx.mean(self._huber_loss(q_values_for_actions - targets))

            return loss

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(self.q_network)

        # Update Q-network
        self.optimizer.update(self.q_network, grads)
        mx.eval(self.q_network.parameters())

        self.update_count += 1

        return {"loss": float(loss)}
