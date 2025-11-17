"""
MLX-aware space adapters for Gymnasium environments.

This module provides utilities for converting between Gymnasium spaces
and MLX arrays, enabling Metal GPU-accelerated sampling and validation.
"""

from typing import Any, Optional, Union, cast

import mlx.core as mx
import numpy as np
from gymnasium import spaces


class MLXSpace:
    """
    Adapter for converting Gym spaces to MLX-compatible representations.

    This class provides static methods for:
    - Sampling from Gymnasium spaces using MLX's Metal-accelerated RNG
    - Converting observations between NumPy and MLX formats
    - Validating observations against space definitions

    Example:
        ```python
        space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Sample using MLX
        key = mx.random.key(42)
        sample = MLXSpace.sample(space, key=key)

        # Convert to MLX
        obs = MLXSpace.to_mlx(np.array([0.1, 0.2, 0.3, 0.4]))

        # Convert back to NumPy for Gym compatibility
        np_obs = MLXSpace.to_numpy(obs)
        ```
    """

    @staticmethod
    def sample(
        space: spaces.Space, key: Optional[mx.array] = None
    ) -> Union[mx.array, dict, tuple]:
        """
        Sample from a Gymnasium space, returning MLX array.

        Uses MLX's Metal-accelerated random number generation when a key
        is provided. Falls back to Gymnasium's default sampling otherwise.

        Args:
            space: Gymnasium space to sample from
            key: Optional MLX random key for Metal-accelerated sampling

        Returns:
            Sampled value as MLX array (or dict/tuple for composite spaces)

        Raises:
            NotImplementedError: If space type is not supported

        Example:
            ```python
            # Discrete space
            space = spaces.Discrete(5)
            action = MLXSpace.sample(space)

            # Box space with custom key
            space = spaces.Box(0, 1, shape=(4,))
            key = mx.random.key(42)
            obs = MLXSpace.sample(space, key=key)
            ```
        """
        if isinstance(space, spaces.Discrete):
            # Sample discrete action as MLX array
            if key is None:
                return mx.array(space.sample())
            else:
                return mx.random.randint(0, int(space.n), (), key=key)

        elif isinstance(space, spaces.Box):
            # Sample continuous values as MLX array
            if key is None:
                return mx.array(space.sample())
            else:
                # Handle infinite bounds
                low = space.low
                high = space.high

                # Convert to finite bounds for sampling
                if np.any(np.isinf(low)):
                    low = np.where(np.isinf(low), -1e6, low)
                if np.any(np.isinf(high)):
                    high = np.where(np.isinf(high), 1e6, high)

                # Sample using uniform distribution
                uniform_samples = mx.random.uniform(shape=space.shape, key=key)

                # Scale to [low, high]
                low_mx = mx.array(low)
                high_mx = mx.array(high)
                return low_mx + (high_mx - low_mx) * uniform_samples

        elif isinstance(space, spaces.Dict):
            # Recursively sample dictionary spaces
            return {k: MLXSpace.sample(v, key) for k, v in space.spaces.items()}

        elif isinstance(space, spaces.Tuple):
            # Recursively sample tuple spaces
            return tuple(MLXSpace.sample(s, key) for s in space.spaces)

        elif isinstance(space, spaces.MultiDiscrete):
            # Sample multiple discrete values
            if key is None:
                return mx.array(space.sample())
            else:
                # Sample each discrete dimension independently
                samples = []
                for nvec in space.nvec:
                    sample = mx.random.randint(0, int(nvec), (), key=key)
                    samples.append(sample)
                return mx.array(samples)

        elif isinstance(space, spaces.MultiBinary):
            # Sample binary values
            if key is None:
                return mx.array(space.sample())
            else:
                return mx.random.randint(0, 2, shape=space.shape, key=key)

        else:
            raise NotImplementedError(
                f"Space type {type(space)} not supported. "
                f"Supported types: Discrete, Box, Dict, Tuple, MultiDiscrete, MultiBinary"
            )

    @staticmethod
    def to_mlx(observation: Any) -> Union[mx.array, dict, tuple]:
        """
        Convert Gymnasium observation to MLX array.

        Handles various observation types including NumPy arrays,
        Python lists, scalars, and composite observations (dict/tuple).

        Args:
            observation: Observation from Gymnasium environment

        Returns:
            Observation as MLX array (or dict/tuple for composite observations)

        Example:
            ```python
            # NumPy array
            np_obs = np.array([1, 2, 3, 4])
            mlx_obs = MLXSpace.to_mlx(np_obs)

            # Dict observation
            dict_obs = {"agent": np.array([0, 1]), "target": np.array([5, 5])}
            mlx_dict = MLXSpace.to_mlx(dict_obs)
            ```
        """
        if isinstance(observation, mx.array):
            # Already MLX array
            return observation
        elif isinstance(observation, np.ndarray):
            # Convert NumPy array to MLX
            return mx.array(observation)
        elif isinstance(observation, tuple):
            # Check if it's a composite observation or simple array
            if observation and isinstance(observation[0], (dict, tuple)):
                # Composite observation
                return tuple(MLXSpace.to_mlx(o) for o in observation)
            else:
                # Simple array
                return mx.array(observation)
        elif isinstance(observation, list):
            # Check if it's a composite observation or simple array
            if observation and isinstance(observation[0], (dict, tuple)):
                # Composite observation - return as tuple for type safety
                return tuple(MLXSpace.to_mlx(o) for o in observation)
            else:
                # Simple array
                return mx.array(observation)
        elif isinstance(observation, dict):
            # Convert dictionary recursively
            return {k: MLXSpace.to_mlx(v) for k, v in observation.items()}
        elif isinstance(observation, (int, float, bool)):
            # Convert scalar to MLX array
            return mx.array(observation)
        else:
            # Try to convert directly
            try:
                return mx.array(observation)
            except Exception as e:
                raise TypeError(
                    f"Cannot convert observation of type {type(observation)} to MLX array. "
                    f"Error: {e}"
                ) from e

    @staticmethod
    def to_numpy(
        mlx_array: Union[mx.array, dict, tuple]
    ) -> Union[np.ndarray, dict, tuple]:
        """
        Convert MLX array to NumPy for Gymnasium compatibility.

        Some Gymnasium wrappers and tools expect NumPy arrays. This method
        converts MLX arrays back to NumPy format.

        Args:
            mlx_array: MLX array or composite observation

        Returns:
            NumPy array (or dict/tuple for composite observations)

        Example:
            ```python
            mlx_obs = mx.array([1.0, 2.0, 3.0])
            np_obs = MLXSpace.to_numpy(mlx_obs)
            assert isinstance(np_obs, np.ndarray)
            ```
        """
        if isinstance(mlx_array, mx.array):
            # Convert MLX array to NumPy
            return np.array(mlx_array)
        elif isinstance(mlx_array, dict):
            # Convert dictionary recursively
            return {k: MLXSpace.to_numpy(v) for k, v in mlx_array.items()}
        elif isinstance(mlx_array, tuple):
            # Convert tuple recursively
            return tuple(MLXSpace.to_numpy(v) for v in mlx_array)
        elif isinstance(mlx_array, np.ndarray):
            # Already NumPy
            return mlx_array
        else:
            # Return as-is (scalar or other type)
            return mlx_array

    @staticmethod
    def contains(space: spaces.Space, x: Union[mx.array, Any]) -> bool:
        """
        Check if observation is valid for the given space.

        Converts MLX arrays to NumPy for Gymnasium's contains() check.

        Args:
            space: Gymnasium space
            x: Observation to check

        Returns:
            True if observation is valid for space

        Example:
            ```python
            space = spaces.Box(low=0, high=1, shape=(4,))
            obs = mx.array([0.1, 0.2, 0.3, 0.4])
            assert MLXSpace.contains(space, obs)

            invalid_obs = mx.array([0.1, 0.2, 0.3, 1.5])
            assert not MLXSpace.contains(space, invalid_obs)
            ```
        """
        # Convert to NumPy for Gymnasium's contains check
        if isinstance(x, mx.array):
            x = np.array(x)
        elif isinstance(x, dict):
            x = {k: np.array(v) if isinstance(v, mx.array) else v for k, v in x.items()}

        return space.contains(x)

    @staticmethod
    def get_shape(space: spaces.Space) -> Union[tuple[int, ...], dict, tuple]:
        """
        Get the shape of observations from a space.

        Args:
            space: Gymnasium space

        Returns:
            Shape tuple (or dict/tuple for composite spaces)

        Example:
            ```python
            space = spaces.Box(low=0, high=1, shape=(4, 4, 3))
            shape = MLXSpace.get_shape(space)
            assert shape == (4, 4, 3)
            ```
        """
        if isinstance(space, (spaces.Box, spaces.MultiBinary)):
            return space.shape
        elif isinstance(space, spaces.Discrete):
            return ()
        elif isinstance(space, spaces.MultiDiscrete):
            return (len(space.nvec),)
        elif isinstance(space, spaces.Dict):
            # Return dict of shapes
            return {k: MLXSpace.get_shape(v) for k, v in space.spaces.items()}
        elif isinstance(space, spaces.Tuple):
            # Return tuple of shapes
            return tuple(MLXSpace.get_shape(s) for s in space.spaces)
        else:
            raise NotImplementedError(f"Cannot get shape for space type {type(space)}")

    @staticmethod
    def get_dtype(space: spaces.Space) -> Any:
        """
        Get the dtype of observations from a space.

        Args:
            space: Gymnasium space

        Returns:
            dtype (NumPy dtype for compatibility)

        Example:
            ```python
            space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
            dtype = MLXSpace.get_dtype(space)
            assert dtype == np.float32
            ```
        """
        if isinstance(space, (spaces.Box, spaces.MultiBinary)):
            return space.dtype
        elif isinstance(space, spaces.Discrete):
            return np.int64
        elif isinstance(space, spaces.MultiDiscrete):
            return np.int64
        elif isinstance(space, spaces.Dict):
            # Return dict of dtypes
            return {k: MLXSpace.get_dtype(v) for k, v in space.spaces.items()}
        elif isinstance(space, spaces.Tuple):
            # Return tuple of dtypes
            return tuple(MLXSpace.get_dtype(s) for s in space.spaces)
        else:
            raise NotImplementedError(f"Cannot get dtype for space type {type(space)}")


def flatten_space(space: spaces.Space) -> spaces.Space:
    """
    Flatten a space into a single Box space.

    Useful for converting complex observation spaces into flat vectors
    for neural network input.

    Args:
        space: Gymnasium space to flatten

    Returns:
        Flattened Box space

    Example:
        ```python
        dict_space = spaces.Dict({
            "position": spaces.Box(0, 1, shape=(2,)),
            "velocity": spaces.Box(-1, 1, shape=(2,))
        })
        flat_space = flatten_space(dict_space)
        assert flat_space.shape == (4,)
        ```
    """
    from gymnasium.spaces.utils import flatten_space as gym_flatten_space
    result = gym_flatten_space(space)
    # flatten_space may return different space types, not just Box
    # We cast to avoid type checker warnings but the actual type depends on input
    return cast(spaces.Space, result)  # type: ignore[return-value]


def flatten_observation(space: spaces.Space, obs: Any) -> mx.array:
    """
    Flatten an observation according to its space.

    Args:
        space: Original observation space
        obs: Observation to flatten

    Returns:
        Flattened observation as MLX array

    Example:
        ```python
        dict_space = spaces.Dict({
            "position": spaces.Box(0, 1, shape=(2,)),
            "velocity": spaces.Box(-1, 1, shape=(2,))
        })
        obs = {"position": mx.array([0.5, 0.5]), "velocity": mx.array([0.1, -0.1])}
        flat_obs = flatten_observation(dict_space, obs)
        assert flat_obs.shape == (4,)
        ```
    """
    from gymnasium.spaces.utils import flatten as gym_flatten
    # Convert to NumPy for Gymnasium's flatten
    np_obs = MLXSpace.to_numpy(obs)
    flat_np = gym_flatten(space, np_obs)

    # flat_np should be array-like, handle edge cases
    if isinstance(flat_np, dict):
        raise TypeError("flatten returned dict which is unexpected")

    # Convert back to MLX
    return mx.array(flat_np)
