"""
Asynchronous Advantage Actor-Critic (A3C) algorithm implementation.

This module provides a simplified A3C implementation using MLX for Metal GPU
acceleration. Note: This is a single-threaded version suitable for educational
purposes and small-scale experiments.

Reference: SMLX_Gym.md, Section 4.3 (Algorithms)
Paper: Mnih et al. (2016) - Asynchronous Methods for Deep Reinforcement Learning
"""

from typing import Optional

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from smlx.gym.algorithms.base import RLAgent


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


class A3CNetwork(nn.Module):
    """
    Actor-Critic network for A3C.

    Similar to PPO's actor-critic but designed for A3C's asynchronous updates.

    Architecture:
        state -> Shared(hidden) -> Actor head -> action logits
                                -> Critic head -> value estimate

    Example:
        ```python
        import mlx.core as mx
        from smlx.gym.algorithms.a3c import A3CNetwork

        # Create A3C network
        a3c_net = A3CNetwork(
            observation_dim=4,
            action_dim=2,
            hidden_dim=64
        )

        # Forward pass
        state = mx.random.normal((1, 4))
        action_logits, value = a3c_net(state)
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
        Initialize A3C network.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers
        """
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Shared feature layers
        shared_layers = []
        shared_layers.append(nn.Linear(observation_dim, hidden_dim))
        shared_layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
            shared_layers.append(nn.ReLU())

        self.shared_layers = shared_layers

        # Actor head (policy)
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        Forward pass through A3C network.

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


class A3CAgent(RLAgent):
    """
    Asynchronous Advantage Actor-Critic (A3C) agent.

    Simplified single-threaded implementation of A3C suitable for small-scale
    experiments and education. For production use, consider PPO which is
    more sample-efficient and easier to tune.

    Implements:
    - Actor-critic architecture
    - N-step returns
    - Entropy bonus for exploration
    - Value function loss

    Reference:
        Mnih et al. (2016) - Asynchronous Methods for Deep RL
        https://arxiv.org/abs/1602.01783

    Example:
        ```python
        import gymnasium as gym
        from smlx.gym.algorithms.a3c import A3CAgent

        # Create environment
        env = gym.make("CartPole-v1")

        # Create A3C agent
        agent = A3CAgent(
            env=env,
            hidden_dim=64,
            learning_rate=1e-3,
            gamma=0.99,
            n_steps=5,
            entropy_coef=0.01
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
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        n_steps: int = 5,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        **kwargs,
    ):
        """
        Initialize A3C agent.

        Args:
            env: Gymnasium environment
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            n_steps: Number of steps for n-step returns
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            **kwargs: Additional arguments passed to RLAgent
        """
        super().__init__(env, **kwargs)

        # Validate discrete action space
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("A3C currently only supports Discrete action spaces")

        # Get dimensions
        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_dim = int(mx.prod(mx.array(env.observation_space.shape)))
        else:
            raise ValueError("A3C only supports Box observation spaces")

        self.action_dim = int(env.action_space.n)

        # Hyperparameters
        self.gamma: float = gamma
        self.n_steps: int = n_steps
        self.value_loss_coef: float = value_loss_coef
        self.entropy_coef: float = entropy_coef
        self.max_grad_norm: float = max_grad_norm

        # Create A3C network
        self.network = A3CNetwork(
            self.observation_dim, self.action_dim, hidden_dim, num_hidden_layers
        )

        # Create optimizer
        self.optimizer = optim.Adam(learning_rate=learning_rate)

        # Training state
        self.trajectory_states: list[mx.array] = []
        self.trajectory_actions: list[int] = []
        self.trajectory_rewards: list[float] = []
        self.trajectory_dones: list[bool] = []

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
        action_logits, _ = self.network(obs_batch)

        # Compute action probabilities
        action_probs = mx.softmax(action_logits[0])

        if training:
            # Sample action from policy
            action_probs_np = np.array(action_probs)
            action = int(np.random.choice(self.action_dim, p=action_probs_np))
        else:
            # Greedy action (for evaluation)
            action = int(mx.argmax(action_probs))

        return action

    def train_step(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """
        Perform one A3C training step using collected trajectory.

        Args:
            batch: Batch from trajectory containing:
                   states, actions, returns

        Returns:
            Dictionary with training metrics
        """
        states = batch["states"]
        actions = batch["actions"]
        returns = batch["returns"]

        # Flatten observations if needed
        if len(states.shape) > 2:
            batch_size = states.shape[0]
            states = mx.reshape(states, (batch_size, -1))

        def loss_fn(model):
            # Forward pass
            action_logits, values = model(states)

            # Compute action log probabilities
            action_probs = mx.softmax(action_logits, axis=-1)
            action_indices = actions.astype(mx.int32)
            log_probs = mx.log(
                mx.take_along_axis(action_probs, mx.expand_dims(action_indices, 1), axis=1).squeeze(
                    1
                )
            )

            # Compute advantages
            value_pred = values.squeeze(1)
            advantages = returns - value_pred

            # Policy loss (negative log likelihood weighted by advantages)
            policy_loss = -mx.mean(log_probs * mx.stop_gradient(advantages))

            # Value function loss
            value_loss = mx.mean((value_pred - returns) ** 2)

            # Entropy bonus (for exploration)
            entropy = -mx.mean(mx.sum(action_probs * mx.log(action_probs + 1e-8), axis=-1))

            # Total loss
            total_loss = (
                policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            )

            return total_loss

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(self.network)

        # Compute auxiliary metrics separately for logging
        action_logits, values = self.network(states)
        action_probs = mx.softmax(action_logits, axis=-1)
        action_indices = actions.astype(mx.int32)
        log_probs = mx.log(
            mx.take_along_axis(action_probs, mx.expand_dims(action_indices, 1), axis=1).squeeze(1)
        )
        value_pred = values.squeeze(1)
        advantages = returns - value_pred
        policy_loss = -mx.mean(log_probs * mx.stop_gradient(advantages))
        value_loss = mx.mean((value_pred - returns) ** 2)
        entropy = -mx.mean(mx.sum(action_probs * mx.log(action_probs + 1e-8), axis=-1))

        # Gradient clipping
        grads = self._clip_gradients(grads)

        # Update network
        self.optimizer.update(self.network, grads)

        # Evaluate parameters
        mx.eval(self.network.parameters())

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
        # Flatten gradients to compute global norm
        def flatten_grads(d):
            """Recursively flatten nested gradient dictionary."""
            flat = []
            for v in d.values():
                if isinstance(v, dict):
                    flat.extend(flatten_grads(v))
                elif isinstance(v, mx.array):
                    flat.append(v)
            return flat

        # Compute global norm
        flat_grads = flatten_grads(grads)
        if not flat_grads:
            return grads

        # Compute squared norm of each gradient and sum them
        squared_norms = [mx.sum(g * g) for g in flat_grads]
        # Stack into a single array and sum
        total_squared_norm = mx.sum(mx.stack(squared_norms))
        global_norm = mx.sqrt(total_squared_norm)

        # Clip if necessary
        if float(global_norm) > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (float(global_norm) + 1e-6)

            # Apply clipping to all gradients
            def clip_dict(d):
                """Recursively clip gradients in nested dictionary."""
                clipped = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        clipped[k] = clip_dict(v)
                    elif isinstance(v, mx.array):
                        clipped[k] = v * clip_coef
                    else:
                        clipped[k] = v
                return clipped

            grads = clip_dict(grads)

        return grads

    def _run_episode(self, max_steps: Optional[int] = None) -> tuple[float, int, bool]:
        """
        Run single episode and collect trajectory.

        Overrides base class to collect n-step trajectories for A3C updates.

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

            # Train every n steps
            if len(self.trajectory_states) >= self.n_steps:
                self._update_policy(next_obs, done)
                # Reset trajectory buffers
                self.trajectory_states = []
                self.trajectory_actions = []
                self.trajectory_rewards = []
                self.trajectory_dones = []

            if max_steps and episode_length >= max_steps:
                break

        # Update policy with remaining trajectory if episode ends
        if len(self.trajectory_states) > 0:
            self._update_policy(observation, done)

        success = info.get("success", False) if "success" in info else False
        return episode_return, episode_length, success

    def _update_policy(self, last_state: mx.array, done: bool):
        """
        Update policy using collected n-step trajectory.

        Args:
            last_state: Last state in trajectory
            done: Whether episode is done
        """
        if len(self.trajectory_states) == 0:
            return

        # Convert lists to MLX arrays
        states = mx.stack(self.trajectory_states)
        actions = mx.array(self.trajectory_actions)
        rewards = mx.array(self.trajectory_rewards)

        # Compute n-step returns
        if done:
            # Bootstrap value is 0 if episode ended
            bootstrap_value = 0.0
        else:
            # Get bootstrap value from last state
            if len(last_state.shape) > 1:
                last_state = mx.reshape(last_state, (-1,))
            _, last_value = self.network(mx.expand_dims(last_state, 0))
            bootstrap_value = float(last_value[0, 0])

        # Helper function to safely extract float from list_or_scalar
        def to_scalar_float(val) -> float:
            """Recursively extract scalar float from potentially nested structure."""
            while isinstance(val, list):
                val = val[0]
            return float(val)

        # Compute n-step returns backwards
        returns = []
        R = bootstrap_value
        rewards_list_raw = rewards.tolist()
        # Ensure rewards_list is a list (handle scalar case)
        if not isinstance(rewards_list_raw, list):
            rewards_list_raw = [rewards_list_raw]

        # Convert to list of floats (handle potential nested lists)
        rewards_list: list[float] = [to_scalar_float(r) for r in rewards_list_raw]

        # Compute returns
        for r in reversed(rewards_list):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_mx = mx.array(returns)

        # Prepare batch
        batch = {"states": states, "actions": actions, "returns": returns_mx}

        # Training step
        metrics = self.train_step(batch)
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

        # Save network parameters
        for key, value in flatten_dict(self.network.parameters(), "network").items():
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
            path: Path to load agent state from (should be .safetensors format)
        """
        loaded_state = mx.load(path)
        if not isinstance(loaded_state, dict):
            raise ValueError(f"Expected to load a state dict, but got {type(loaded_state)}")
        state: dict[str, mx.array] = loaded_state

        # Extract network parameters
        network_params = {}
        optimizer_state = {}

        for key, value in state.items():
            if key.startswith("network."):
                network_params[key[len("network.") :]] = value
            elif key.startswith("optimizer."):
                optimizer_state[key[len("optimizer.") :]] = value

        # Unflatten and update network
        self.network.update(unflatten_dict(network_params))
        self.optimizer.state = unflatten_dict(optimizer_state)  # type: ignore[assignment]

        # Load metadata
        self.update_count = int(state["update_count"])
