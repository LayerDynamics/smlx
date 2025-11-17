# Copyright � 2025 SMLX Project

"""
Gym environment endpoints for reinforcement learning.

Provides REST API for creating, managing, and interacting with gym environments.
"""

import uuid
from typing import Any, Optional

import gymnasium as gym
import mlx.core as mx
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# Global environment registry
# Maps environment_id -> (env, metadata)
_environments: dict[str, tuple[gym.Env, dict[str, Any]]] = {}


# ============================================================================
# Schemas
# ============================================================================


class CreateEnvRequest(BaseModel):
    """Request to create a new environment."""

    env_id: str = Field(..., description="Gymnasium environment ID (e.g., 'CartPole-v1')")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    wrappers: Optional[list[str]] = Field(
        default=None, description="List of wrapper names to apply"
    )
    wrapper_kwargs: Optional[dict[str, Any]] = Field(
        default=None, description="Keyword arguments for wrappers"
    )


class CreateEnvResponse(BaseModel):
    """Response from creating environment."""

    environment_id: str = Field(..., description="Unique environment instance ID")
    env_id: str = Field(..., description="Gymnasium environment ID")
    observation_space: dict[str, Any] = Field(..., description="Observation space info")
    action_space: dict[str, Any] = Field(..., description="Action space info")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Environment metadata")


class ResetRequest(BaseModel):
    """Request to reset environment."""

    seed: Optional[int] = Field(default=None, description="Random seed")
    options: Optional[dict[str, Any]] = Field(default=None, description="Reset options")


class ResetResponse(BaseModel):
    """Response from resetting environment."""

    observation: list[float] = Field(..., description="Initial observation")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional info")


class StepRequest(BaseModel):
    """Request to take environment step."""

    action: Any = Field(..., description="Action to take")


class StepResponse(BaseModel):
    """Response from environment step."""

    observation: list[float] = Field(..., description="New observation")
    reward: float = Field(..., description="Reward received")
    terminated: bool = Field(..., description="Whether episode terminated")
    truncated: bool = Field(..., description="Whether episode truncated")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional info")


class EnvInfo(BaseModel):
    """Information about an environment instance."""

    environment_id: str = Field(..., description="Unique environment instance ID")
    env_id: str = Field(..., description="Gymnasium environment ID")
    observation_space: dict[str, Any] = Field(..., description="Observation space info")
    action_space: dict[str, Any] = Field(..., description="Action space info")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Environment metadata")


class ListEnvsResponse(BaseModel):
    """Response from listing environments."""

    environments: list[EnvInfo] = Field(..., description="List of active environments")
    count: int = Field(..., description="Number of active environments")


# ============================================================================
# Helper Functions
# ============================================================================


def space_to_dict(space: gym.spaces.Space) -> dict[str, Any]:
    """Convert gym space to dictionary representation."""
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "low": space.low.tolist() if hasattr(space.low, "tolist") else float(space.low),
            "high": space.high.tolist()
            if hasattr(space.high, "tolist")
            else float(space.high),
            "dtype": str(space.dtype),
        }
    elif isinstance(space, gym.spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
        }
    elif isinstance(space, gym.spaces.MultiBinary):
        # space.n can be int or tuple of ints
        n_value = space.n
        if isinstance(n_value, int):
            n_converted = n_value
        else:
            # It's a tuple, keep as list for JSON
            n_converted = list(n_value)
        return {
            "type": "MultiBinary",
            "n": n_converted,
        }
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": space.nvec.tolist(),
        }
    elif isinstance(space, gym.spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {key: space_to_dict(subspace) for key, subspace in space.spaces.items()},
        }
    else:
        return {"type": type(space).__name__}


def observation_to_list(obs: Any) -> list[float]:
    """Convert observation to list format for JSON response."""
    if isinstance(obs, mx.array):
        obs = np.array(obs)

    if isinstance(obs, np.ndarray):
        return obs.flatten().tolist()
    elif isinstance(obs, (list, tuple)):
        return list(obs)
    elif isinstance(obs, (int, float)):
        return [float(obs)]
    elif isinstance(obs, dict):
        # For dict observations, flatten all values
        result = []
        for value in obs.values():
            if isinstance(value, (mx.array, np.ndarray)):
                result.extend(np.array(value).flatten().tolist())
            elif isinstance(value, (list, tuple)):
                result.extend(value)
            else:
                result.append(float(value))
        return result
    else:
        raise ValueError(f"Unsupported observation type: {type(obs)}")


# ============================================================================
# Routes
# ============================================================================


@router.post("/gym/envs", response_model=CreateEnvResponse)
async def create_environment(request: CreateEnvRequest):
    """
    Create a new gym environment instance.

    Creates and initializes a gym environment, returning a unique ID for
    subsequent interactions.

    Example:
        ```bash
        curl -X POST http://localhost:8000/v1/gym/envs \\
          -H "Content-Type: application/json" \\
          -d '{"env_id": "CartPole-v1", "seed": 42}'
        ```
    """
    try:
        # Create environment
        env = gym.make(request.env_id)

        # Apply wrappers if specified
        if request.wrappers:
            from smlx.gym.wrappers import (
                ClipReward,
                EpisodeLogger,
                FrameStack,
                MLXObservationWrapper,
                NormalizeObservation,
                NormalizeReward,
                RecordEpisodeStatistics,
            )

            wrapper_map = {
                "normalize_obs": NormalizeObservation,
                "normalize_reward": NormalizeReward,
                "clip_reward": ClipReward,
                "frame_stack": FrameStack,
                "record_stats": RecordEpisodeStatistics,
                "episode_logger": EpisodeLogger,
                "mlx_obs": MLXObservationWrapper,
            }

            for wrapper_name in request.wrappers:
                if wrapper_name in wrapper_map:
                    wrapper_cls = wrapper_map[wrapper_name]
                    kwargs = {}
                    if request.wrapper_kwargs and wrapper_name in request.wrapper_kwargs:
                        kwargs = request.wrapper_kwargs[wrapper_name]
                    env = wrapper_cls(env, **kwargs)

        # Reset with seed if provided
        if request.seed is not None:
            env.reset(seed=request.seed)

        # Generate unique environment ID
        environment_id = str(uuid.uuid4())

        # Store environment
        metadata = {
            "env_id": request.env_id,
            "seed": request.seed,
            "wrappers": request.wrappers or [],
        }
        _environments[environment_id] = (env, metadata)

        # Get space info
        obs_space_dict = space_to_dict(env.observation_space)
        action_space_dict = space_to_dict(env.action_space)

        return CreateEnvResponse(
            environment_id=environment_id,
            env_id=request.env_id,
            observation_space=obs_space_dict,
            action_space=action_space_dict,
            metadata=metadata,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/gym/envs/{environment_id}/reset", response_model=ResetResponse)
async def reset_environment(environment_id: str, request: ResetRequest = ResetRequest()):
    """
    Reset an environment to initial state.

    Example:
        ```bash
        curl -X POST http://localhost:8000/v1/gym/envs/{env_id}/reset \\
          -H "Content-Type: application/json" \\
          -d '{"seed": 42}'
        ```
    """
    if environment_id not in _environments:
        raise HTTPException(status_code=404, detail="Environment not found")

    try:
        env, _ = _environments[environment_id]

        # Reset environment
        reset_kwargs = {}
        if request.seed is not None:
            reset_kwargs["seed"] = request.seed
        if request.options is not None:
            reset_kwargs["options"] = request.options

        observation, info = env.reset(**reset_kwargs)

        # Convert observation to list
        obs_list = observation_to_list(observation)

        return ResetResponse(observation=obs_list, info=info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/gym/envs/{environment_id}/step", response_model=StepResponse)
async def step_environment(environment_id: str, request: StepRequest):
    """
    Take a step in the environment.

    Example:
        ```bash
        curl -X POST http://localhost:8000/v1/gym/envs/{env_id}/step \\
          -H "Content-Type: application/json" \\
          -d '{"action": 0}'
        ```
    """
    if environment_id not in _environments:
        raise HTTPException(status_code=404, detail="Environment not found")

    try:
        env, _ = _environments[environment_id]

        # Take step
        observation, reward, terminated, truncated, info = env.step(request.action)

        # Convert observation to list
        obs_list = observation_to_list(observation)

        return StepResponse(
            observation=obs_list,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/gym/envs/{environment_id}", response_model=EnvInfo)
async def get_environment_info(environment_id: str):
    """
    Get information about an environment instance.

    Example:
        ```bash
        curl http://localhost:8000/v1/gym/envs/{env_id}
        ```
    """
    if environment_id not in _environments:
        raise HTTPException(status_code=404, detail="Environment not found")

    try:
        env, metadata = _environments[environment_id]

        obs_space_dict = space_to_dict(env.observation_space)
        action_space_dict = space_to_dict(env.action_space)

        return EnvInfo(
            environment_id=environment_id,
            env_id=metadata["env_id"],
            observation_space=obs_space_dict,
            action_space=action_space_dict,
            metadata=metadata,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/gym/envs", response_model=ListEnvsResponse)
async def list_environments():
    """
    List all active environment instances.

    Example:
        ```bash
        curl http://localhost:8000/v1/gym/envs
        ```
    """
    try:
        env_infos = []

        for environment_id, (env, metadata) in _environments.items():
            obs_space_dict = space_to_dict(env.observation_space)
            action_space_dict = space_to_dict(env.action_space)

            env_info = EnvInfo(
                environment_id=environment_id,
                env_id=metadata["env_id"],
                observation_space=obs_space_dict,
                action_space=action_space_dict,
                metadata=metadata,
            )
            env_infos.append(env_info)

        return ListEnvsResponse(environments=env_infos, count=len(env_infos))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/gym/envs/{environment_id}")
async def close_environment(environment_id: str):
    """
    Close and remove an environment instance.

    Example:
        ```bash
        curl -X DELETE http://localhost:8000/v1/gym/envs/{env_id}
        ```
    """
    if environment_id not in _environments:
        raise HTTPException(status_code=404, detail="Environment not found")

    try:
        env, _ = _environments[environment_id]
        env.close()
        del _environments[environment_id]

        return {"message": "Environment closed successfully", "environment_id": environment_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/gym/envs")
async def close_all_environments():
    """
    Close all active environment instances.

    Example:
        ```bash
        curl -X DELETE http://localhost:8000/v1/gym/envs
        ```
    """
    try:
        count = len(_environments)

        for env, _ in _environments.values():
            env.close()

        _environments.clear()

        return {"message": f"Closed {count} environment(s)", "count": count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
