"""
Proximal Policy Optimization (PPO) algorithm implementation.

This module provides a complete PPO implementation using MLX for Metal GPU
acceleration. Includes actor-critic networks, GAE, clipped policy objective,
and value function training.

Reference: SMLX_Gym.md, Section 4.3 (Algorithms)
Paper: Schulman et al. (2017) - Proximal Policy Optimization Algorithms
"""

from typing import Optional

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map, tree_reduce

from smlx.gym.algorithms.base import RLAgent
from smlx.gym.replay import EpisodeBuffer, compute_gae


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


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO.

    Combined network with shared features and separate policy/value heads.
    This architecture allows for efficient computation and shared representations.

    Architecture:
        state -> Shared(hidden) -> Actor head -> action logits
                                -> Critic head -> value estimate

    Example:
        ```python
        import mlx.core as mx
        from smlx.gym.algorithms.ppo import ActorCriticNetwork

        # Create actor-critic network
        ac_net = ActorCriticNetwork(
            observation_dim=4,
            action_dim=2,
            hidden_dim=64
        )

        # Forward pass
        state = mx.random.normal((1, 4))
        action_logits, value = ac_net(state)
        ```
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
    ):
        """
        Initialize actor-critic network.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Number of actions (for discrete) or action dimension (for continuous)
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers (default: 2)
        """
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Shared feature layers
        shared_layers = []
        shared_layers.append(nn.Linear(observation_dim, hidden_dim))
        shared_layers.append(nn.Tanh())

        for _ in range(num_hidden_layers - 1):
            shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
            shared_layers.append(nn.Tanh())

        self.shared_layers = shared_layers

        # Actor head (policy)
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        Forward pass through actor-critic network.

        Args:
            x: State observations [batch_size, observation_dim]

        Returns:
            action_logits: Logits for each action [batch_size, action_dim]
            values: Value estimates [batch_size, 1]
        """
        # Shared features
        for layer in self.shared_layers:
            x = layer(x)

        # Actor output (action logits)
        action_logits = self.actor_head(x)

        # Critic output (value estimate)
        values = self.critic_head(x)

        return action_logits, values


class PPOAgent(RLAgent):
    """
    Proximal Policy Optimization (PPO) agent.

    Implements the PPO algorithm with:
    - Actor-critic architecture
    - Generalized Advantage Estimation (GAE)
    - Clipped policy objective
    - Value function loss
    - Entropy bonus for exploration

    PPO is an on-policy algorithm that maintains a balance between exploration
    and exploitation while ensuring stable training through policy clipping.

    Reference:
        Schulman et al. (2017) - Proximal Policy Optimization Algorithms
        https://arxiv.org/abs/1707.06347

    Example:
        ```python
        import gymnasium as gym
        from smlx.gym.algorithms.ppo import PPOAgent

        # Create environment
        env = gym.make("CartPole-v1")

        # Create PPO agent
        agent = PPOAgent(
            env=env,
            hidden_dim=64,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            n_steps=2048,
            batch_size=64,
            n_epochs=10
        )

        # Train agent
        response = agent.run(num_episodes=1000)
        print(f"Average return: {response.average_return}")
        ```
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        max_grad_norm: float = 0.5,
        **kwargs,
    ):
        """
        Initialize PPO agent.

        Args:
            env: Gymnasium environment
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            n_steps: Steps to collect before update
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            max_grad_norm: Maximum gradient norm for clipping
            **kwargs: Additional arguments passed to RLAgent
        """
        super().__init__(env, **kwargs)

        # Validate discrete action space
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("PPO currently only supports Discrete action spaces")

        # Get dimensions
        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_dim = int(mx.prod(mx.array(env.observation_space.shape)))
        else:
            raise ValueError("PPO only supports Box observation spaces")

        self.action_dim = int(env.action_space.n)

        # Hyperparameters
        self.gamma: float = gamma
        self.gae_lambda: float = gae_lambda
        self.clip_epsilon: float = clip_epsilon
        self.value_loss_coef: float = value_loss_coef
        self.entropy_coef: float = entropy_coef
        self.n_steps: int = n_steps
        self.batch_size: int = batch_size
        self.n_epochs: int = n_epochs
        self.max_grad_norm: float = max_grad_norm

        # Create actor-critic network
        self.ac_network = ActorCriticNetwork(
            self.observation_dim, self.action_dim, hidden_dim, num_hidden_layers
        )

        # Create optimizer
        self.optimizer = optim.Adam(learning_rate=learning_rate)

        # Episode buffer for collecting trajectories
        self.episode_buffer = EpisodeBuffer(capacity=1000)

        # Training state
        self.trajectory_states: list[mx.array] = []
        self.trajectory_actions: list[int] = []
        self.trajectory_rewards: list[float] = []
        self.trajectory_dones: list[bool] = []
        self.trajectory_log_probs: list[float] = []
        self.trajectory_values: list[float] = []

        self.update_count: int = 0
        self.losses: list[float] = []

    def select_action(self, observation: mx.array, training: bool = True) -> int:
        """
        Select action using current policy.

        Args:
            observation: Current observation
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action (int)
        """
        # Flatten observation if needed
        if len(observation.shape) > 1:
            obs = mx.reshape(observation, (-1,))
        else:
            obs = observation

        # Add batch dimension
        obs_batch = mx.expand_dims(obs, 0)

        # Get action logits and value
        action_logits, value = self.ac_network(obs_batch)

        # Compute action probabilities
        action_probs = mx.softmax(action_logits[0])

        if training:
            # Sample action from policy
            action_probs_np = np.array(action_probs)
            action = int(np.random.choice(self.action_dim, p=action_probs_np))

            # Compute log probability
            log_prob = float(mx.log(action_probs[action]))

            # Store for training
            self.trajectory_log_probs.append(log_prob)
            self.trajectory_values.append(float(value[0, 0]))
        else:
            # Greedy action (for evaluation)
            action = int(mx.argmax(action_probs))

        return action

    def train_step(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """
        Perform one PPO training step using collected trajectories.

        Args:
            batch: Batch from trajectories containing:
                   states, actions, old_log_probs, advantages, returns

        Returns:
            Dictionary with training metrics
        """
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Flatten observations if needed
        if len(states.shape) > 2:
            batch_size = states.shape[0]
            states = mx.reshape(states, (batch_size, -1))

        def loss_fn(model):
            # Forward pass
            action_logits, values = model(states)

            # Compute current log probabilities
            action_probs = mx.softmax(action_logits, axis=-1)
            action_indices = actions.astype(mx.int32)
            current_log_probs = mx.log(
                mx.take_along_axis(
                    action_probs, mx.expand_dims(action_indices, 1), axis=1
                ).squeeze(1)
            )

            # PPO clipped policy loss
            ratio = mx.exp(current_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = mx.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -mx.mean(mx.minimum(surr1, surr2))

            # Value function loss
            value_pred = values.squeeze(1)
            value_loss = mx.mean((value_pred - returns) ** 2)

            # Entropy bonus (for exploration)
            entropy = -mx.mean(mx.sum(action_probs * mx.log(action_probs + 1e-8), axis=-1))

            # Total loss
            total_loss = (
                policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            )

            return total_loss, (policy_loss, value_loss, entropy)

        # Compute loss and gradients
        # MLX's value_and_grad doesn't support has_aux, so we need to handle auxiliary outputs differently
        def loss_and_aux_fn(model):
            total_loss, aux = loss_fn(model)
            return total_loss

        loss_val, grads = mx.value_and_grad(loss_and_aux_fn)(self.ac_network)
        # Get auxiliary values separately
        _, (policy_loss, value_loss, entropy) = loss_fn(self.ac_network)
        loss = loss_val

        # Gradient clipping
        grads = self._clip_gradients(grads)

        # Update network
        self.optimizer.update(self.ac_network, grads)

        # Evaluate parameters
        mx.eval(self.ac_network.parameters())

        self.update_count += 1

        return {
            "loss": float(loss),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy),
        }

    def _clip_gradients(self, grads: dict) -> dict:
        """
        Clip gradients by global norm.

        Args:
            grads: Dictionary of gradients

        Returns:
            Clipped gradients
        """
        # Compute global norm using tree_reduce
        norm_squared = tree_reduce(lambda acc, g: acc + mx.sum(g * g), grads, 0.0)
        # Ensure norm_squared is an MLX array before passing to sqrt
        # tree_reduce with initial value 0.0 will never return None
        global_norm = mx.sqrt(mx.array(norm_squared) if norm_squared is not None else mx.array(0.0))

        # Clip if necessary
        normalizer = mx.minimum(self.max_grad_norm / (global_norm + 1e-6), 1.0)
        clipped_grads = tree_map(lambda g: g * normalizer, grads)

        return clipped_grads

    def _run_episode(self, max_steps: Optional[int] = None) -> tuple[float, int, bool]:
        """
        Run single episode and collect trajectory.

        Overrides base class to collect full trajectories for PPO updates.

        Args:
            max_steps: Maximum steps per episode

        Returns:
            tuple of (episode_return, episode_length, success)
        """
        observation, info = self.env.reset()

        # Convert observation to MLX array
        if not isinstance(observation, mx.array):
            observation = mx.array(observation)

        # Reset trajectory buffers
        self.trajectory_states = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
        self.trajectory_dones = []
        self.trajectory_log_probs = []
        self.trajectory_values = []

        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            # Store state
            self.trajectory_states.append(observation)

            # Select action
            action = self.select_action(observation, training=True)
            self.trajectory_actions.append(action)

            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Convert next observation to MLX array
            if not isinstance(next_obs, mx.array):
                next_obs = mx.array(next_obs)

            # Store transition
            self.trajectory_rewards.append(float(reward))
            self.trajectory_dones.append(done)

            episode_return += float(reward)
            episode_length += 1
            observation = next_obs
            self.total_steps += 1

            # Train if we've collected enough steps
            if len(self.trajectory_states) >= self.n_steps:
                self._update_policy()
                # Reset trajectory buffers
                self.trajectory_states = []
                self.trajectory_actions = []
                self.trajectory_rewards = []
                self.trajectory_dones = []
                self.trajectory_log_probs = []
                self.trajectory_values = []

            if max_steps and episode_length >= max_steps:
                break

        # Update policy with remaining trajectory if episode ends
        if len(self.trajectory_states) > 0:
            self._update_policy()

        success = info.get("success", False) if "success" in info else False
        return episode_return, episode_length, success

    def _update_policy(self):
        """Update policy using collected trajectories."""
        if len(self.trajectory_states) == 0:
            return

        # Convert lists to MLX arrays
        states = mx.stack(self.trajectory_states)
        actions = mx.array(self.trajectory_actions)
        rewards = mx.array(self.trajectory_rewards)
        dones = mx.array([float(d) for d in self.trajectory_dones])
        old_log_probs = mx.array(self.trajectory_log_probs)

        # Get bootstrap value for last state
        last_state = self.trajectory_states[-1]
        if len(last_state.shape) > 1:
            last_state = mx.reshape(last_state, (-1,))
        _, last_value = self.ac_network(mx.expand_dims(last_state, 0))
        last_value = float(last_value[0, 0])

        # Compute advantages using GAE
        values = mx.array(self.trajectory_values + [last_value])
        advantages, returns = compute_gae(rewards, values, dones, self.gamma, self.gae_lambda)

        # Normalize advantages
        advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)

        # Multiple epochs of updates
        for _ in range(self.n_epochs):
            # Sample mini-batches
            n_samples = len(states)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end].tolist()  # Convert to list for MLX indexing

                # Create mini-batch
                mini_batch = {
                    "states": states[batch_indices],
                    "actions": actions[batch_indices],
                    "old_log_probs": old_log_probs[batch_indices],
                    "advantages": advantages[batch_indices],
                    "returns": returns[batch_indices],
                }

                # Training step
                metrics = self.train_step(mini_batch)
                self.losses.append(metrics["loss"])
                self.current_metrics.loss = metrics["loss"]

    def save(self, path: str):
        """
        Save agent state to disk.

        Args:
            path: Path to save agent state
        """
        # Flatten nested dictionaries and ensure all values are MLX arrays
        state = {}

        # Save actor-critic network parameters
        for key, value in flatten_dict(self.ac_network.parameters(), "ac_network").items():
            state[key] = value

        # Save optimizer state (flatten and convert scalars to arrays)
        for key, value in flatten_dict(self.optimizer.state, "optimizer").items():
            if isinstance(value, mx.array):
                state[key] = value
            elif isinstance(value, (int, float)):
                state[key] = mx.array(value)
            # Skip other types that can't be serialized

        # Save metadata as arrays
        state["update_count"] = mx.array(self.update_count)

        mx.save_safetensors(path, state)

    def load(self, path: str):
        """
        Load agent state from disk.

        Args:
            path: Path to load agent state from
        """
        flat_state = mx.load(path)
        if not isinstance(flat_state, dict):
            raise ValueError(f"Expected to load a state dict, but got {type(flat_state)}")

        # Unflatten the state dictionary
        state = unflatten_dict(flat_state)

        # Load network parameters
        if "ac_network" in state:
            self.ac_network.update(state["ac_network"])  # type: ignore[arg-type]

        # Load optimizer state
        if "optimizer" in state:
            self.optimizer.state = state["optimizer"]  # type: ignore[assignment]

        # Load metadata
        if "update_count" in flat_state:
            self.update_count = int(flat_state["update_count"])
