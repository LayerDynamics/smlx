# Copyright � 2025 SMLX Project

"""
Tests for gym environment endpoints.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from smlx.server.routes import gym as gym_routes
from smlx.server.routes.gym import router


@pytest.fixture
def app():
    """Create test app with gym router."""
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    # Clear any existing environments before each test
    gym_routes._environments.clear()
    client = TestClient(app)
    yield client
    # Clean up after test
    gym_routes._environments.clear()


@pytest.fixture
def mock_env():
    """Create mock gym environment."""
    import gymnasium as gym

    env = Mock(spec=gym.Env)

    # Mock observation space (Box)
    obs_space = Mock(spec=gym.spaces.Box)
    obs_space.shape = (4,)
    obs_space.low = np.array([-1.0, -1.0, -1.0, -1.0])
    obs_space.high = np.array([1.0, 1.0, 1.0, 1.0])
    obs_space.dtype = np.float32
    env.observation_space = obs_space

    # Mock action space (Discrete)
    action_space = Mock(spec=gym.spaces.Discrete)
    action_space.n = 2
    env.action_space = action_space

    # Mock reset
    env.reset = Mock(return_value=(np.array([0.1, 0.2, 0.3, 0.4]), {}))

    # Mock step
    env.step = Mock(
        return_value=(
            np.array([0.2, 0.3, 0.4, 0.5]),  # observation
            1.0,  # reward
            False,  # terminated
            False,  # truncated
            {},  # info
        )
    )

    # Mock close
    env.close = Mock()

    return env


@pytest.mark.unit
class TestHelperFunctions:
    """Tests for helper functions."""

    def test_space_to_dict_box(self):
        """Test converting Box space to dict."""
        import gymnasium as gym

        space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        result = gym_routes.space_to_dict(space)

        assert result["type"] == "Box"
        assert result["shape"] == [4]
        assert result["low"] == [-1.0, -1.0, -1.0, -1.0]
        assert result["high"] == [1.0, 1.0, 1.0, 1.0]
        assert "float32" in result["dtype"]

    def test_space_to_dict_discrete(self):
        """Test converting Discrete space to dict."""
        import gymnasium as gym

        space = gym.spaces.Discrete(5)
        result = gym_routes.space_to_dict(space)

        assert result["type"] == "Discrete"
        assert result["n"] == 5

    def test_space_to_dict_multibinary(self):
        """Test converting MultiBinary space to dict."""
        import gymnasium as gym

        space = gym.spaces.MultiBinary(8)
        result = gym_routes.space_to_dict(space)

        assert result["type"] == "MultiBinary"
        assert result["n"] == 8

    def test_space_to_dict_multidiscrete(self):
        """Test converting MultiDiscrete space to dict."""
        import gymnasium as gym

        space = gym.spaces.MultiDiscrete([3, 4, 5])
        result = gym_routes.space_to_dict(space)

        assert result["type"] == "MultiDiscrete"
        assert result["nvec"] == [3, 4, 5]

    def test_space_to_dict_dict_space(self):
        """Test converting Dict space to dict."""
        import gymnasium as gym

        space = gym.spaces.Dict(
            {"obs": gym.spaces.Box(low=0, high=1, shape=(2,)), "action": gym.spaces.Discrete(3)}
        )
        result = gym_routes.space_to_dict(space)

        assert result["type"] == "Dict"
        assert "obs" in result["spaces"]
        assert "action" in result["spaces"]
        assert result["spaces"]["obs"]["type"] == "Box"
        assert result["spaces"]["action"]["type"] == "Discrete"

    def test_observation_to_list_numpy(self):
        """Test converting numpy array observation to list."""
        obs = np.array([1.0, 2.0, 3.0])
        result = gym_routes.observation_to_list(obs)

        assert result == [1.0, 2.0, 3.0]

    def test_observation_to_list_multidimensional(self):
        """Test converting multidimensional numpy array to list."""
        obs = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = gym_routes.observation_to_list(obs)

        assert result == [1.0, 2.0, 3.0, 4.0]

    def test_observation_to_list_scalar(self):
        """Test converting scalar observation to list."""
        obs = 5.0
        result = gym_routes.observation_to_list(obs)

        assert result == [5.0]

    def test_observation_to_list_list(self):
        """Test converting list observation."""
        obs = [1.0, 2.0, 3.0]
        result = gym_routes.observation_to_list(obs)

        assert result == [1.0, 2.0, 3.0]

    def test_observation_to_list_dict(self):
        """Test converting dict observation to list."""
        obs = {"sensor1": np.array([1.0, 2.0]), "sensor2": np.array([3.0, 4.0])}
        result = gym_routes.observation_to_list(obs)

        # Should flatten all values
        assert len(result) == 4
        assert 1.0 in result
        assert 4.0 in result


@pytest.mark.integration
class TestCreateEnvironment:
    """Tests for create environment endpoint."""

    @patch("smlx.server.routes.gym.gym.make")
    def test_create_simple_environment(self, mock_make, client, mock_env):
        """Test creating a simple CartPole environment."""
        mock_make.return_value = mock_env

        response = client.post(
            "/v1/gym/envs", json={"env_id": "CartPole-v1", "seed": 42, "wrappers": None}
        )

        assert response.status_code == 200
        data = response.json()

        assert "environment_id" in data
        assert data["env_id"] == "CartPole-v1"
        assert data["observation_space"]["type"] == "Box"
        assert data["action_space"]["type"] == "Discrete"
        assert data["action_space"]["n"] == 2
        assert data["metadata"]["seed"] == 42

        # Verify environment was stored
        assert len(gym_routes._environments) == 1

    @patch("smlx.server.routes.gym.gym.make")
    @patch("smlx.gym.wrappers.NormalizeObservation")
    def test_create_environment_with_wrappers(self, mock_wrapper, mock_make, client, mock_env):
        """Test creating environment with wrappers."""
        mock_make.return_value = mock_env
        mock_wrapper.return_value = mock_env

        response = client.post(
            "/v1/gym/envs",
            json={
                "env_id": "CartPole-v1",
                "wrappers": ["normalize_obs"],
                "wrapper_kwargs": {"normalize_obs": {"epsilon": 1e-8}},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "environment_id" in data
        assert "normalize_obs" in data["metadata"]["wrappers"]

    @patch("smlx.server.routes.gym.gym.make")
    def test_create_environment_without_seed(self, mock_make, client, mock_env):
        """Test creating environment without seed."""
        mock_make.return_value = mock_env

        response = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1"})

        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["seed"] is None

    @patch("smlx.server.routes.gym.gym.make")
    def test_create_environment_invalid_env_id(self, mock_make, client):
        """Test creating environment with invalid ID."""
        mock_make.side_effect = Exception("No registered env with id: InvalidEnv")

        response = client.post("/v1/gym/envs", json={"env_id": "InvalidEnv-v999"})

        assert response.status_code == 500
        assert "No registered env" in response.json()["detail"]

    def test_create_environment_missing_required_field(self, client):
        """Test creating environment without required env_id."""
        response = client.post("/v1/gym/envs", json={})

        assert response.status_code == 422  # Validation error


@pytest.mark.integration
class TestResetEnvironment:
    """Tests for reset environment endpoint."""

    @patch("smlx.server.routes.gym.gym.make")
    def test_reset_environment_simple(self, mock_make, client, mock_env):
        """Test resetting an environment."""
        mock_make.return_value = mock_env

        # First create an environment
        create_response = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1"})
        env_id = create_response.json()["environment_id"]

        # Reset it
        reset_response = client.post(f"/v1/gym/envs/{env_id}/reset", json={})

        assert reset_response.status_code == 200
        data = reset_response.json()
        assert "observation" in data
        assert isinstance(data["observation"], list)
        assert len(data["observation"]) == 4
        assert "info" in data

    @patch("smlx.server.routes.gym.gym.make")
    def test_reset_environment_with_seed(self, mock_make, client, mock_env):
        """Test resetting environment with seed."""
        mock_make.return_value = mock_env

        # Create environment
        create_response = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1"})
        env_id = create_response.json()["environment_id"]

        # Reset with seed
        reset_response = client.post(f"/v1/gym/envs/{env_id}/reset", json={"seed": 123})

        assert reset_response.status_code == 200
        # Verify reset was called with seed
        mock_env.reset.assert_called_with(seed=123)

    @patch("smlx.server.routes.gym.gym.make")
    def test_reset_environment_with_options(self, mock_make, client, mock_env):
        """Test resetting environment with options."""
        mock_make.return_value = mock_env

        # Create environment
        create_response = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1"})
        env_id = create_response.json()["environment_id"]

        # Reset with options
        reset_response = client.post(
            f"/v1/gym/envs/{env_id}/reset", json={"seed": 42, "options": {"difficulty": "hard"}}
        )

        assert reset_response.status_code == 200
        # Verify reset was called with both seed and options
        mock_env.reset.assert_called_with(seed=42, options={"difficulty": "hard"})

    def test_reset_nonexistent_environment(self, client):
        """Test resetting a non-existent environment."""
        response = client.post("/v1/gym/envs/nonexistent-id/reset", json={})

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.integration
class TestStepEnvironment:
    """Tests for step environment endpoint."""

    @patch("smlx.server.routes.gym.gym.make")
    def test_step_environment(self, mock_make, client, mock_env):
        """Test taking a step in environment."""
        mock_make.return_value = mock_env

        # Create environment
        create_response = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1"})
        env_id = create_response.json()["environment_id"]

        # Take step
        step_response = client.post(f"/v1/gym/envs/{env_id}/step", json={"action": 1})

        assert step_response.status_code == 200
        data = step_response.json()
        assert "observation" in data
        assert "reward" in data
        assert "terminated" in data
        assert "truncated" in data
        assert "info" in data
        assert isinstance(data["observation"], list)
        assert isinstance(data["reward"], float)
        assert isinstance(data["terminated"], bool)
        assert isinstance(data["truncated"], bool)

        # Verify step was called with correct action
        mock_env.step.assert_called_once_with(1)

    @patch("smlx.server.routes.gym.gym.make")
    def test_step_environment_episode_end(self, mock_make, client, mock_env):
        """Test step that ends episode."""
        # Configure mock to return terminated=True
        mock_env.step = Mock(
            return_value=(
                np.array([0.2, 0.3, 0.4, 0.5]),
                10.0,  # Final reward
                True,  # terminated
                False,
                {},
            )
        )
        mock_make.return_value = mock_env

        # Create environment
        create_response = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1"})
        env_id = create_response.json()["environment_id"]

        # Take step
        step_response = client.post(f"/v1/gym/envs/{env_id}/step", json={"action": 0})

        assert step_response.status_code == 200
        data = step_response.json()
        assert data["terminated"] is True
        assert data["reward"] == 10.0

    def test_step_nonexistent_environment(self, client):
        """Test stepping in non-existent environment."""
        response = client.post("/v1/gym/envs/nonexistent-id/step", json={"action": 0})

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_step_missing_action(self, client):
        """Test step without required action."""
        response = client.post("/v1/gym/envs/some-id/step", json={})

        assert response.status_code == 422  # Validation error


@pytest.mark.integration
class TestGetEnvironmentInfo:
    """Tests for get environment info endpoint."""

    @patch("smlx.server.routes.gym.gym.make")
    def test_get_environment_info(self, mock_make, client, mock_env):
        """Test getting environment information."""
        mock_make.return_value = mock_env

        # Create environment
        create_response = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1", "seed": 42})
        env_id = create_response.json()["environment_id"]

        # Get info
        info_response = client.get(f"/v1/gym/envs/{env_id}")

        assert info_response.status_code == 200
        data = info_response.json()
        assert data["environment_id"] == env_id
        assert data["env_id"] == "CartPole-v1"
        assert data["observation_space"]["type"] == "Box"
        assert data["action_space"]["type"] == "Discrete"
        assert data["metadata"]["seed"] == 42

    def test_get_nonexistent_environment_info(self, client):
        """Test getting info for non-existent environment."""
        response = client.get("/v1/gym/envs/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.integration
class TestListEnvironments:
    """Tests for list environments endpoint."""

    def test_list_empty_environments(self, client):
        """Test listing environments when none exist."""
        response = client.get("/v1/gym/envs")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["environments"] == []

    @patch("smlx.server.routes.gym.gym.make")
    def test_list_multiple_environments(self, mock_make, client, mock_env):
        """Test listing multiple environments."""
        mock_make.return_value = mock_env

        # Create multiple environments
        env_ids = []
        for i in range(3):
            response = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1", "seed": i})
            env_ids.append(response.json()["environment_id"])

        # List all
        list_response = client.get("/v1/gym/envs")

        assert list_response.status_code == 200
        data = list_response.json()
        assert data["count"] == 3
        assert len(data["environments"]) == 3

        # Verify all environment IDs are present
        returned_ids = [env["environment_id"] for env in data["environments"]]
        for env_id in env_ids:
            assert env_id in returned_ids


@pytest.mark.integration
class TestCloseEnvironment:
    """Tests for close environment endpoint."""

    @patch("smlx.server.routes.gym.gym.make")
    def test_close_environment(self, mock_make, client, mock_env):
        """Test closing a single environment."""
        mock_make.return_value = mock_env

        # Create environment
        create_response = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1"})
        env_id = create_response.json()["environment_id"]

        # Close it
        close_response = client.delete(f"/v1/gym/envs/{env_id}")

        assert close_response.status_code == 200
        data = close_response.json()
        assert data["message"] == "Environment closed successfully"
        assert data["environment_id"] == env_id

        # Verify environment was closed
        mock_env.close.assert_called_once()

        # Verify environment was removed from registry
        assert len(gym_routes._environments) == 0

    def test_close_nonexistent_environment(self, client):
        """Test closing a non-existent environment."""
        response = client.delete("/v1/gym/envs/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.integration
class TestCloseAllEnvironments:
    """Tests for close all environments endpoint."""

    def test_close_all_empty(self, client):
        """Test closing all when no environments exist."""
        response = client.delete("/v1/gym/envs")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert "Closed 0" in data["message"]

    @patch("smlx.server.routes.gym.gym.make")
    def test_close_all_multiple_environments(self, mock_make, client, mock_env):
        """Test closing all environments."""
        mock_make.return_value = mock_env

        # Create multiple environments
        for i in range(3):
            client.post("/v1/gym/envs", json={"env_id": "CartPole-v1", "seed": i})

        # Close all
        close_response = client.delete("/v1/gym/envs")

        assert close_response.status_code == 200
        data = close_response.json()
        assert data["count"] == 3
        assert "Closed 3" in data["message"]

        # Verify all environments were closed (called 3 times during creation + 3 times during close)
        # Actually, close is only called when explicitly closing, not during creation
        assert mock_env.close.call_count == 3

        # Verify registry is empty
        assert len(gym_routes._environments) == 0


@pytest.mark.integration
class TestEnvironmentWorkflow:
    """Tests for complete environment workflow."""

    @patch("smlx.server.routes.gym.gym.make")
    def test_full_episode_workflow(self, mock_make, client, mock_env):
        """Test complete episode: create -> reset -> step -> close."""
        # Configure mock for episode that ends after 3 steps
        step_count = [0]

        def step_side_effect(action):
            step_count[0] += 1
            terminated = step_count[0] >= 3
            return (
                np.array([0.1 * step_count[0]] * 4),
                1.0,
                terminated,
                False,
                {"step": step_count[0]},
            )

        mock_env.step = Mock(side_effect=step_side_effect)
        mock_make.return_value = mock_env

        # 1. Create environment
        create_resp = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1", "seed": 42})
        assert create_resp.status_code == 200
        env_id = create_resp.json()["environment_id"]

        # 2. Reset environment
        reset_resp = client.post(f"/v1/gym/envs/{env_id}/reset", json={"seed": 42})
        assert reset_resp.status_code == 200

        # 3. Take multiple steps
        for i in range(3):
            step_resp = client.post(f"/v1/gym/envs/{env_id}/step", json={"action": 1})
            assert step_resp.status_code == 200
            data = step_resp.json()

            if i < 2:
                assert data["terminated"] is False
            else:
                assert data["terminated"] is True
                assert data["info"]["step"] == 3

        # 4. Get environment info
        info_resp = client.get(f"/v1/gym/envs/{env_id}")
        assert info_resp.status_code == 200

        # 5. Close environment
        close_resp = client.delete(f"/v1/gym/envs/{env_id}")
        assert close_resp.status_code == 200

        # Verify we can't access it anymore
        info_resp = client.get(f"/v1/gym/envs/{env_id}")
        assert info_resp.status_code == 404

    @patch("smlx.server.routes.gym.gym.make")
    def test_multiple_environments_independent(self, mock_make, client, mock_env):
        """Test that multiple environments are independent."""
        mock_make.return_value = mock_env

        # Create two environments
        env1_resp = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1", "seed": 1})
        env1_id = env1_resp.json()["environment_id"]

        env2_resp = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1", "seed": 2})
        env2_id = env2_resp.json()["environment_id"]

        # Verify they have different IDs
        assert env1_id != env2_id

        # Step in first environment
        step1_resp = client.post(f"/v1/gym/envs/{env1_id}/step", json={"action": 0})
        assert step1_resp.status_code == 200

        # Step in second environment
        step2_resp = client.post(f"/v1/gym/envs/{env2_id}/step", json={"action": 1})
        assert step2_resp.status_code == 200

        # Close first environment
        client.delete(f"/v1/gym/envs/{env1_id}")

        # Second environment should still work
        step2_resp = client.post(f"/v1/gym/envs/{env2_id}/step", json={"action": 0})
        assert step2_resp.status_code == 200

        # First should be gone
        info1_resp = client.get(f"/v1/gym/envs/{env1_id}")
        assert info1_resp.status_code == 404


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_invalid_json_payload(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/v1/gym/envs",
            content=b"invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    @patch("smlx.server.routes.gym.gym.make")
    def test_environment_error_during_step(self, mock_make, client, mock_env):
        """Test handling of environment errors during step."""
        mock_env.step = Mock(side_effect=Exception("Environment simulation error"))
        mock_make.return_value = mock_env

        # Create environment
        create_resp = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1"})
        env_id = create_resp.json()["environment_id"]

        # Try to step - should handle error gracefully
        step_resp = client.post(f"/v1/gym/envs/{env_id}/step", json={"action": 0})
        assert step_resp.status_code == 500
        assert "Environment simulation error" in step_resp.json()["detail"]

    @patch("smlx.server.routes.gym.gym.make")
    def test_environment_error_during_reset(self, mock_make, client, mock_env):
        """Test handling of environment errors during reset."""
        mock_env.reset = Mock(side_effect=Exception("Reset failed"))
        mock_make.return_value = mock_env

        # Create environment
        create_resp = client.post("/v1/gym/envs", json={"env_id": "CartPole-v1"})
        env_id = create_resp.json()["environment_id"]

        # Try to reset - should handle error gracefully
        reset_resp = client.post(f"/v1/gym/envs/{env_id}/reset", json={})
        assert reset_resp.status_code == 500
        assert "Reset failed" in reset_resp.json()["detail"]
