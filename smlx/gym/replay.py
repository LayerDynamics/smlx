"""
Experience replay buffers for reinforcement learning.

This module provides replay buffer implementations for storing and sampling
experience tuples in off-policy RL algorithms. All buffers use MLX arrays
for Metal GPU acceleration.

Reference: SMLX_Gym.md, Section 4.2 (Training Infrastructure)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import gymnasium as gym
import mlx.core as mx
import numpy as np


@dataclass
class Transition:
    """
    Represents a single experience transition.

    Attributes:
        state: Current state observation
        action: Action taken
        reward: Reward received
        next_state: Next state observation
        done: Whether episode terminated
        info: Additional information
    """

    state: mx.array
    action: Any
    reward: float
    next_state: mx.array
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL algorithms.

    Stores transitions and samples random batches for training. Uses MLX
    arrays for Metal GPU acceleration throughout the sampling pipeline.

    The buffer implements a ring buffer that overwrites old experiences
    when full. This is memory-efficient and suitable for continuous learning.

    Features:
    - Uniform random sampling
    - Efficient MLX array batching
    - Optional per-step metadata
    - Memory-efficient ring buffer

    Example:
        ```python
        import mlx.core as mx
        from smlx.gym.replay import ReplayBuffer

        # Create buffer
        buffer = ReplayBuffer(capacity=10000)

        # Store experiences
        state = mx.random.normal((4,))
        action = 0
        reward = 1.0
        next_state = mx.random.normal((4,))
        done = False

        buffer.add(state, action, reward, next_state, done)

        # Sample batch
        batch = buffer.sample(batch_size=32)
        states = batch['states']  # MLX array [32, 4]
        actions = batch['actions']  # MLX array [32]
        rewards = batch['rewards']  # MLX array [32]
        ```

    Integration with RL Training:
        ```python
        # Training loop
        for episode in range(num_episodes):
            state, info = env.reset()

            while not done:
                # Agent selects action
                action = agent.select_action(state)

                # Environment step
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store in replay buffer
                buffer.add(state, action, reward, next_state, done)

                # Train if enough experiences
                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    loss = agent.train_step(batch)

                state = next_state
        ```
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def add(
        self,
        state: mx.array,
        action: Any,
        reward: float,
        next_state: mx.array,
        done: bool,
        info: Optional[dict[str, Any]] = None,
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state (MLX array)
            action: Action taken (int, float, or MLX array)
            reward: Reward received
            next_state: Next state (MLX array)
            done: Whether episode terminated
            info: Optional additional information
        """
        # Ensure states are MLX arrays
        if not isinstance(state, mx.array):
            state = mx.array(state)
        if not isinstance(next_state, mx.array):
            next_state = mx.array(next_state)

        # Create transition
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info if info is not None else {},
        )

        # Add to buffer (automatically overwrites old transitions when full)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict[str, Union[mx.array, list]]:
        """
        Sample a random batch of transitions.

        Returns a dictionary of MLX arrays for efficient GPU training.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing:
            - 'states': MLX array of states [batch_size, ...]
            - 'actions': MLX array of actions [batch_size, ...]
            - 'rewards': MLX array of rewards [batch_size]
            - 'next_states': MLX array of next states [batch_size, ...]
            - 'dones': MLX array of done flags [batch_size]

        Raises:
            ValueError: If batch_size > buffer size
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with "
                f"{len(self.buffer)} transitions"
            )

        # Sample random indices
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        # Gather transitions
        transitions = [self.buffer[idx] for idx in indices]

        # Stack into batched MLX arrays
        states = mx.stack([t.state for t in transitions])
        next_states = mx.stack([t.next_state for t in transitions])
        rewards = mx.array([t.reward for t in transitions])
        dones = mx.array([float(t.done) for t in transitions])

        # Handle actions (may be int, float, or array)
        actions = []
        for t in transitions:
            if isinstance(t.action, mx.array):
                actions.append(t.action)
            else:
                actions.append(mx.array(t.action))

        # Try to stack actions (works if all same shape)
        try:
            actions = mx.stack(actions)
        except Exception:
            # If actions have different shapes, return as list
            # This can happen with complex action spaces
            pass

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def clear(self):
        """Clear all transitions from the buffer."""
        self.buffer.clear()
        self.position = 0

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) >= self.capacity


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) buffer.

    Extends ReplayBuffer with priority-based sampling where transitions
    with higher TD-error are sampled more frequently. This can improve
    learning efficiency by focusing on more informative experiences.

    Uses proportional prioritization: P(i) = p_i^alpha / sum(p_j^alpha)

    Features:
    - Priority-based sampling
    - Importance sampling weights for bias correction
    - Efficient priority updates
    - Annealing beta parameter

    Reference:
        Schaul et al. (2015) - Prioritized Experience Replay
        https://arxiv.org/abs/1511.05952

    Example:
        ```python
        import mlx.core as mx
        from smlx.gym.replay import PrioritizedReplayBuffer

        # Create prioritized buffer
        buffer = PrioritizedReplayBuffer(
            capacity=10000,
            alpha=0.6,  # Prioritization exponent
            beta=0.4,   # Importance sampling weight
        )

        # Store experience with priority
        buffer.add(state, action, reward, next_state, done)

        # Sample with priorities
        batch = buffer.sample(batch_size=32)
        states = batch['states']
        weights = batch['weights']  # Importance sampling weights

        # Train and get TD-errors
        td_errors = agent.train_step(batch)

        # Update priorities based on TD-errors
        buffer.update_priorities(batch['indices'], td_errors)
        ```
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling weight (0 = no correction, 1 = full correction)
            beta_increment: Amount to increment beta per sample (annealing)
            epsilon: Small constant to prevent zero priorities
        """
        super().__init__(capacity=capacity)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Priority arrays using numpy (for efficient indexing)
        # Will convert to MLX arrays when sampling
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(
        self,
        state: mx.array,
        action: Any,
        reward: float,
        next_state: mx.array,
        done: bool,
        info: Optional[dict[str, Any]] = None,
        priority: Optional[float] = None,
    ):
        """
        Add transition with specified or maximum priority.

        New experiences get max priority by default to ensure they're sampled.
        Optionally, a specific priority can be provided.

        Args:
            state: Current state (MLX array)
            action: Action taken
            reward: Reward received
            next_state: Next state (MLX array)
            done: Whether episode terminated
            info: Optional additional information
            priority: Optional priority value (defaults to max_priority)
        """
        # Add to buffer
        super().add(state, action, reward, next_state, done, info)

        # Set priority (use provided priority or default to maximum)
        idx = (len(self.buffer) - 1) % self.capacity
        if priority is not None:
            self.priorities[idx] = priority
            # For the first transition, replace the initial max_priority
            # For subsequent transitions, only update if priority is higher
            if len(self.buffer) == 1:
                self.max_priority = priority
            else:
                self.max_priority = max(self.max_priority, priority)
        else:
            self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int) -> dict[str, Any]:
        """
        Sample batch with priority-based sampling.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing:
            - All standard replay buffer fields
            - 'indices': Sampled indices (for priority updates)
            - 'weights': Importance sampling weights (MLX array)

        Raises:
            ValueError: If batch_size > buffer size
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with "
                f"{len(self.buffer)} transitions"
            )

        # Get priorities for current buffer contents
        n = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha

        # Normalize to probabilities
        probs = probs / probs.sum()

        # Sample indices according to priorities
        indices = np.random.choice(n, batch_size, replace=False, p=probs)

        # Compute importance sampling weights
        # w_i = (1 / (N * P(i)))^beta
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize by max for stability

        # Convert weights to MLX array
        weights_mx = mx.array(weights)

        # Get standard batch
        transitions = [self.buffer[idx] for idx in indices]

        # Stack into batched MLX arrays
        states = mx.stack([t.state for t in transitions])
        next_states = mx.stack([t.next_state for t in transitions])
        rewards = mx.array([t.reward for t in transitions])
        dones = mx.array([float(t.done) for t in transitions])

        # Handle actions
        actions = []
        for t in transitions:
            if isinstance(t.action, mx.array):
                actions.append(t.action)
            else:
                actions.append(mx.array(t.action))

        try:
            actions = mx.stack(actions)
        except Exception:
            pass  # Keep as list if shapes differ

        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "indices": indices,  # For priority updates
            "weights": weights_mx,  # Importance sampling weights
        }

    def update_priorities(self, indices: np.ndarray, priorities: Union[mx.array, np.ndarray]):
        """
        Update priorities for transitions.

        Args:
            indices: Indices of transitions to update
            priorities: Priority values or TD-errors (MLX array or array-like)
                       If values are very small (< epsilon), epsilon will be added
                       to prevent zero priorities
        """
        # Convert priorities to numpy for priority storage
        priorities_np: np.ndarray
        if isinstance(priorities, mx.array):
            priorities_np = np.array(priorities)
        else:
            priorities_np = np.asarray(priorities)

        # Take absolute values and add epsilon only for very small values
        priorities_np = np.abs(priorities_np)
        # Only add epsilon to prevent exactly zero priorities
        priorities_np = np.maximum(priorities_np, self.epsilon)

        self.priorities[indices] = priorities_np

        # Update max priority
        self.max_priority = max(self.max_priority, priorities_np.max())

    def anneal_beta(self, target_beta: float):
        """
        Anneal beta parameter for importance sampling.

        Args:
            target_beta: Target beta value (will be clipped to [0, 1])
        """
        self.beta = min(1.0, max(0.0, target_beta))


class EpisodeBuffer:
    """
    Buffer for storing complete episodes.

    Useful for on-policy algorithms that require full trajectories
    (e.g., PPO, A3C) or for computing returns over complete episodes.

    Example:
        ```python
        from smlx.gym.replay import EpisodeBuffer

        buffer = EpisodeBuffer(max_episodes=100)

        # Start new episode
        buffer.start_episode()

        state, info = env.reset()
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Add transition to current episode
            buffer.add(state, action, reward, next_state, done)

            state = next_state

        # End episode
        buffer.end_episode()

        # Get all transitions from all episodes
        transitions = buffer.get_all_transitions()
        ```
    """

    def __init__(self, max_episodes: int = 1000, capacity: Optional[int] = None):
        """
        Initialize episode buffer.

        Args:
            max_episodes: Maximum number of episodes to store
            capacity: Alias for max_episodes (deprecated, for backwards compatibility)
        """
        # Support both max_episodes and capacity parameters
        if capacity is not None:
            max_episodes = capacity

        if max_episodes <= 0:
            raise ValueError(f"max_episodes must be positive, got {max_episodes}")

        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
        self.current_episode: list[dict[str, Any]] = []

    def start_episode(self):
        """Start a new episode (initialize current_episode)."""
        self.current_episode = []

    def add(
        self,
        state: mx.array,
        action: Any,
        reward: float,
        next_state: mx.array,
        done: bool,
        info: Optional[dict[str, Any]] = None,
    ):
        """
        Add a transition to the current episode.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            info: Additional information
        """
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "info": info or {},
        }
        self.current_episode.append(transition)

    def end_episode(self):
        """End the current episode and add it to the buffer."""
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []

    def add_episode(self, episode: list[dict[str, Any]]):
        """
        Add a complete episode to the buffer.

        Args:
            episode: List of dictionaries containing step information
                    Each dict should have: state, action, reward, etc.
        """
        self.episodes.append(episode)

    def get_episodes(self, n: Optional[int] = None) -> list[list[dict[str, Any]]]:
        """
        Get recent episodes from buffer.

        Args:
            n: Number of episodes to return (None for all)

        Returns:
            List of episodes
        """
        if n is None:
            return list(self.episodes)
        else:
            return list(self.episodes)[-n:]

    def get_all_transitions(self) -> list[dict[str, Any]]:
        """
        Get all transitions from all episodes.

        Returns:
            Flattened list of all transitions from all episodes
        """
        transitions = []
        for episode in self.episodes:
            transitions.extend(episode)
        return transitions

    def clear(self):
        """Clear all episodes from the buffer."""
        self.episodes.clear()
        self.current_episode = []

    def __len__(self) -> int:
        """Return number of episodes in buffer."""
        return len(self.episodes)


def compute_returns(
    rewards: Union[mx.array, list, np.ndarray],
    gamma: float = 0.99,
    dones: Optional[Union[mx.array, list, np.ndarray]] = None,
    normalize: bool = False,
) -> mx.array:
    """
    Compute discounted returns for a batch of episodes.

    Uses MLX operations for Metal GPU acceleration.

    Args:
        rewards: Array of rewards [T] or [B, T] (can be list, numpy array, or MLX array)
        gamma: Discount factor
        dones: Array of done flags [T] or [B, T] (optional, defaults to all False)
        normalize: Whether to normalize returns (zero mean, unit std)

    Returns:
        MLX array of discounted returns with same shape as rewards

    Example:
        ```python
        import mlx.core as mx
        from smlx.gym.replay import compute_returns

        # Episode rewards
        rewards = [1.0, 1.0, 1.0, 0.0, 1.0]
        dones = [False, False, True, False, True]

        # Compute returns with discount
        returns = compute_returns(rewards, gamma=0.99, dones=dones)
        print(returns)  # [2.97, 1.99, 1.0, 0.99, 1.0]
        ```
    """
    # Convert inputs to MLX arrays
    if not isinstance(rewards, mx.array):
        rewards = mx.array(rewards)

    if dones is None:
        # No terminal states
        dones = mx.zeros(rewards.shape)
    elif not isinstance(dones, mx.array):
        # Convert list/numpy to MLX array and convert bools to floats
        dones_list = dones
        if isinstance(dones_list, (list, tuple)):
            dones_list = [float(d) for d in dones_list]
        dones = mx.array(dones_list)

    # Handle both 1D and 2D arrays
    returns = mx.zeros_like(rewards)

    if len(rewards.shape) == 1:
        # Single episode
        T = len(rewards)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = rewards[t] + gamma * G * (1.0 - dones[t])
            returns[t] = G
    else:
        # Batch of episodes
        B, T = rewards.shape
        for b in range(B):
            G = 0.0
            for t in range(T - 1, -1, -1):
                G = rewards[b, t] + gamma * G * (1.0 - dones[b, t])
                returns[b, t] = G

    # Normalize if requested
    if normalize:
        mean = mx.mean(returns)
        std = mx.std(returns)
        if std > 1e-8:
            returns = (returns - mean) / (std + 1e-8)

    return returns


def compute_gae(
    rewards: Union[mx.array, list, np.ndarray],
    values: Union[mx.array, list, np.ndarray],
    dones: Optional[Union[mx.array, list, np.ndarray]] = None,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> tuple[mx.array, mx.array]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Uses MLX operations for Metal GPU acceleration.

    Reference:
        Schulman et al. (2016) - High-Dimensional Continuous Control Using GAE
        https://arxiv.org/abs/1506.02438

    Args:
        rewards: Array of rewards [T]
        values: Array of value estimates [T+1], where values[T] is the bootstrap value
        dones: Array of done flags [T] (optional, defaults to all False)
        gamma: Discount factor
        lambda_: GAE lambda parameter (bias-variance tradeoff)

    Returns:
        Tuple of (advantages, returns) as MLX arrays with shape [T]

    Example:
        ```python
        from smlx.gym.replay import compute_gae

        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.6, 0.7, 0.0]  # Last value is bootstrap
        dones = [False, False, True]

        advantages, returns = compute_gae(rewards, values, dones)
        ```
    """
    # Convert inputs to MLX arrays
    if not isinstance(rewards, mx.array):
        rewards = mx.array(rewards)
    if not isinstance(values, mx.array):
        values = mx.array(values)

    T = len(rewards)

    # Extract current and next values
    current_values = values[:T]
    next_values = values[1:]  # Bootstrap value is last element

    if dones is None:
        # No terminal states
        dones = mx.zeros(T)
    elif not isinstance(dones, mx.array):
        # Convert list/numpy to MLX array and convert bools to floats
        dones_list = dones
        if isinstance(dones_list, (list, tuple)):
            dones_list = [float(d) for d in dones_list]
        dones = mx.array(dones_list)

    advantages = []
    gae = 0.0

    for t in range(T - 1, -1, -1):
        # TD error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        delta = float(rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - current_values[t])

        # GAE: A_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        gae = delta + gamma * lambda_ * (1.0 - float(dones[t])) * gae
        advantages.insert(0, gae)

    # Convert advantages to MLX array
    advantages = mx.array(advantages)

    # Compute returns: R_t = A_t + V(s_t)
    returns = advantages + current_values

    return advantages, returns


def validate_transition(
    state: mx.array,
    action: Any,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> bool:
    """
    Validate that a transition is compatible with environment spaces.

    Uses gymnasium space definitions to check if state and action are valid.

    Args:
        state: State observation (MLX array)
        action: Action taken
        observation_space: Gymnasium observation space
        action_space: Gymnasium action space

    Returns:
        True if transition is valid, False otherwise

    Example:
        ```python
        import gymnasium as gym
        import mlx.core as mx
        from smlx.gym.replay import validate_transition

        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)

        state = mx.array([0.1, 0.2, 0.3, 0.4])
        action = 0

        is_valid = validate_transition(state, action, obs_space, action_space)
        ```
    """
    # Convert MLX array to numpy for gymnasium compatibility
    state_np = np.array(state)

    # Check observation space
    if not observation_space.contains(state_np):
        return False

    # Check action space
    if isinstance(action, mx.array):
        action_check = np.array(action)
    else:
        action_check = action

    if not action_space.contains(action_check):
        return False

    return True
