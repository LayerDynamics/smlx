"""
Multi-agent reinforcement learning support for SMLX Gym.

This module provides utilities for multi-agent RL including multi-agent
environments, coordination, and training infrastructure.

Reference: SMLX_Gym.md, Section 4.4 (Advanced Features)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import gymnasium as gym
import mlx.core as mx
import numpy as np

from smlx.gym.base import MLXEnv


class AgentRole(Enum):
    """Agent roles in multi-agent scenarios."""

    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    MIXED = "mixed"


@dataclass
class MultiAgentConfig:
    """
    Configuration for multi-agent environments.

    Attributes:
        num_agents: Number of agents
        agent_roles: Role for each agent
        shared_reward: Whether agents share rewards
        communication: Whether agents can communicate
        observation_mode: Observation mode ('local', 'global', 'partial')
    """

    num_agents: int
    agent_roles: list[AgentRole] = field(default_factory=list)
    shared_reward: bool = False
    communication: bool = False
    observation_mode: str = "local"
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiAgentEnv(MLXEnv):
    """
    Multi-agent environment base class.

    Provides infrastructure for multi-agent RL including:
    - Multiple agent observations and actions
    - Flexible reward structures (individual, shared, team-based)
    - Communication channels
    - Coordination mechanisms

    All observations and actions use MLX arrays for Metal GPU acceleration.

    Example:
        ```python
        from smlx.gym.multi_agent import MultiAgentEnv, MultiAgentConfig

        # Create multi-agent configuration
        config = MultiAgentConfig(
            num_agents=3,
            shared_reward=False,
            observation_mode='local'
        )

        # Create multi-agent environment
        env = MyMultiAgentEnv(config)

        # Reset returns dict of observations
        obs_dict, info = env.reset()
        # obs_dict = {"agent_0": obs0, "agent_1": obs1, "agent_2": obs2}

        # Step with dict of actions
        actions_dict = {
            "agent_0": action0,
            "agent_1": action1,
            "agent_2": action2
        }
        obs_dict, rewards_dict, terminated_dict, truncated_dict, info = env.step(actions_dict)
        ```
    """

    def __init__(self, config: MultiAgentConfig, render_mode: Optional[str] = None):
        """
        Initialize multi-agent environment.

        Args:
            config: Multi-agent configuration
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__(render_mode=render_mode)

        self.config = config
        self.num_agents = config.num_agents

        # Agent identifiers
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        # Communication channels (if enabled)
        self.messages: dict[str, Any] = {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, mx.array], dict[str, Any]]:
        """
        Reset environment for all agents.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observations: Dict mapping agent_id to observation (MLX array)
            info: Additional information
        """
        super().reset(seed=seed)

        # Clear communication channels
        self.messages = {}

        # Subclasses should override to provide actual observations
        observations = {agent_id: mx.zeros((4,)) for agent_id in self.agent_ids}

        info = {"num_agents": self.num_agents}

        return observations, info

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[str, mx.array],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """
        Execute one step for all agents.

        Args:
            actions: Dict mapping agent_id to action

        Returns:
            observations: Dict mapping agent_id to next observation
            rewards: Dict mapping agent_id to reward
            terminated: Dict mapping agent_id to terminated flag
            truncated: Dict mapping agent_id to truncated flag
            info: Additional information
        """
        # Subclasses should override to provide actual environment dynamics
        observations = {agent_id: mx.zeros((4,)) for agent_id in self.agent_ids}
        rewards = dict.fromkeys(self.agent_ids, 0.0)
        terminated = dict.fromkeys(self.agent_ids, False)
        truncated = dict.fromkeys(self.agent_ids, False)
        info = {}

        return observations, rewards, terminated, truncated, info

    def send_message(self, sender_id: str, receiver_id: str, message: Any):
        """
        Send message from one agent to another.

        Args:
            sender_id: Sender agent ID
            receiver_id: Receiver agent ID
            message: Message content
        """
        if not self.config.communication:
            return

        if receiver_id not in self.messages:
            self.messages[receiver_id] = []

        self.messages[receiver_id].append({"sender": sender_id, "content": message})

    def receive_messages(self, agent_id: str) -> list[dict[str, Any]]:
        """
        Receive messages for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of messages
        """
        messages = self.messages.get(agent_id, [])
        self.messages[agent_id] = []  # Clear messages after receiving
        return messages

    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            print("\n" + "=" * 60)
            print(f"Multi-Agent Environment ({self.num_agents} agents)")
            print("=" * 60 + "\n")


class ParallelEnvWrapper:
    """
    Wrapper for running multiple single-agent environments in parallel.

    This is useful for distributed training where each agent trains in
    its own environment instance.

    Example:
        ```python
        import gymnasium as gym
        from smlx.gym.multi_agent import ParallelEnvWrapper

        # Create parallel environments
        envs = ParallelEnvWrapper("CartPole-v1", num_envs=4)

        # Reset all environments
        observations = envs.reset()
        # observations is list of 4 observations

        # Step all environments with different actions
        actions = [0, 1, 0, 1]
        observations, rewards, terminated, truncated, infos = envs.step(actions)
        ```
    """

    def __init__(self, env_id: str, num_envs: int, **env_kwargs):
        """
        Initialize parallel environment wrapper.

        Args:
            env_id: Gymnasium environment ID
            num_envs: Number of parallel environments
            **env_kwargs: Additional arguments for gym.make()
        """
        self.env_id = env_id
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs

        # Create parallel environments
        self.envs = [gym.make(env_id, **env_kwargs) for _ in range(num_envs)]

        # Use first environment for space definitions
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, seed: Optional[int] = None) -> list[mx.array]:
        """
        Reset all environments.

        Args:
            seed: Random seed (will be incremented for each environment)

        Returns:
            List of observations (MLX arrays)
        """
        observations = []
        for i, env in enumerate(self.envs):
            env_seed = (seed + i) if seed is not None else None
            obs, _ = env.reset(seed=env_seed)

            # Convert to MLX array
            if not isinstance(obs, mx.array):
                obs = mx.array(obs)

            observations.append(obs)

        return observations

    def step(
        self, actions: list[Any]
    ) -> tuple[list[mx.array], list[float], list[bool], list[bool], list[dict]]:
        """
        Step all environments.

        Args:
            actions: List of actions (one per environment)

        Returns:
            observations: List of observations (MLX arrays)
            rewards: List of rewards
            terminated: List of terminated flags
            truncated: List of truncated flags
            infos: List of info dicts
        """
        if len(actions) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, got {len(actions)}"
            )

        observations = []
        rewards = []
        terminated_list = []
        truncated_list = []
        infos = []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)

            # Convert to MLX array
            if not isinstance(obs, mx.array):
                obs = mx.array(obs)

            observations.append(obs)
            rewards.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            infos.append(info)

            # Auto-reset if episode ended
            if terminated or truncated:
                obs, info = env.reset()
                if not isinstance(obs, mx.array):
                    obs = mx.array(obs)
                observations[-1] = obs

        return observations, rewards, terminated_list, truncated_list, infos

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def create_parallel_envs(
    env_id: str, num_envs: int, **env_kwargs
) -> ParallelEnvWrapper:
    """
    Factory function for creating parallel environments.

    Args:
        env_id: Gymnasium environment ID
        num_envs: Number of parallel environments
        **env_kwargs: Additional arguments for gym.make()

    Returns:
        ParallelEnvWrapper instance

    Example:
        ```python
        from smlx.gym.multi_agent import create_parallel_envs

        # Create 8 parallel CartPole environments
        envs = create_parallel_envs("CartPole-v1", num_envs=8)

        # Train with parallel data collection
        obs = envs.reset()
        for _ in range(1000):
            actions = [agent.select_action(o) for o in obs]
            obs, rewards, terminated, truncated, infos = envs.step(actions)
        ```
    """
    return ParallelEnvWrapper(env_id, num_envs, **env_kwargs)


class TeamRewardWrapper(gym.Wrapper):
    """
    Wrapper for team-based rewards in multi-agent environments.

    Converts individual rewards to team rewards, useful for cooperative
    multi-agent scenarios.

    Example:
        ```python
        from smlx.gym.multi_agent import TeamRewardWrapper

        # Wrap multi-agent environment
        env = TeamRewardWrapper(multi_agent_env, reward_fn=sum)

        # All agents receive sum of individual rewards
        obs, rewards, terminated, truncated, info = env.step(actions)
        ```
    """

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Callable = np.mean,
    ):
        """
        Initialize team reward wrapper.

        Args:
            env: Environment to wrap (should be multi-agent)
            reward_fn: Function to aggregate rewards (default: mean)
        """
        super().__init__(env)
        self.reward_fn = reward_fn

    def step(self, action):
        """Step environment and apply team reward."""
        obs, rewards, terminated, truncated, info = self.env.step(action)

        # If rewards is a dict (multi-agent), aggregate
        if isinstance(rewards, dict):
            team_reward = self.reward_fn(list(rewards.values()))
            # Broadcast team reward to all agents
            rewards = dict.fromkeys(rewards.keys(), team_reward)

        return obs, rewards, terminated, truncated, info
