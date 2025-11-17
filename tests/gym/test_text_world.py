"""
Unit tests for TextWorldEnv.

Tests text-based environments for language tasks.
"""

import pytest
from gymnasium import spaces

from smlx.gym.envs.text_world import (
    InstructionFollowingEnv,
    QAEnv,
    Task,
    TextWorldEnv,
    create_text_env,
)


@pytest.mark.unit
class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            prompt="What is 2+2?",
            target="4",
            reward=1.0,
            metadata={"difficulty": "easy"},
        )

        assert task.prompt == "What is 2+2?"
        assert task.target == "4"
        assert task.reward == 1.0
        assert task.metadata == {"difficulty": "easy"}

    def test_task_defaults(self):
        """Test task default values."""
        task = Task(prompt="Test", target="Answer")

        assert task.reward == 1.0
        assert task.metadata == {}


@pytest.mark.unit
class TestTextWorldEnv:
    """Tests for TextWorldEnv."""

    @pytest.fixture
    def sample_tasks(self):
        """Sample tasks for testing."""
        return [
            Task(prompt="What is the capital of France?", target="Paris", reward=1.0),
            Task(prompt="What is 2+2?", target="4", reward=1.0),
            Task(prompt="Name a color.", target="red", reward=1.0),
        ]

    @pytest.fixture
    def env(self, sample_tasks):
        """Create a test environment."""
        return TextWorldEnv(
            tasks=sample_tasks,
            max_episode_length=1,
            reward_on_success=1.0,
            reward_on_failure=0.0,
        )

    def test_env_creation(self, env, sample_tasks):
        """Test environment creation."""
        assert len(env.tasks) == len(sample_tasks)
        assert env.max_episode_length == 1
        assert env.reward_on_success == 1.0
        assert env.reward_on_failure == 0.0

    def test_observation_space(self, env):
        """Test observation space is correctly defined."""
        assert isinstance(env.observation_space, spaces.Dict)
        assert "prompt" in env.observation_space.spaces
        assert "target" in env.observation_space.spaces
        assert isinstance(env.observation_space["prompt"], spaces.Text)
        assert isinstance(env.observation_space["target"], spaces.Text)

    def test_action_space(self, env):
        """Test action space is correctly defined."""
        assert isinstance(env.action_space, spaces.Text)

    def test_reset(self, env):
        """Test environment reset."""
        observation, info = env.reset()

        assert "prompt" in observation
        assert "target" in observation
        assert isinstance(observation["prompt"], str)
        assert isinstance(observation["target"], str)
        assert isinstance(info, dict)

    def test_reset_with_seed(self, env):
        """Test reset with seed for reproducibility."""
        observation1, _ = env.reset(seed=42)
        env.reset()  # Reset without seed
        observation2, _ = env.reset(seed=42)

        assert observation1["prompt"] == observation2["prompt"]
        assert observation1["target"] == observation2["target"]

    def test_correct_answer(self, env):
        """Test correct answer gives positive reward."""
        observation, _ = env.reset()
        target = observation["target"]

        observation, reward, terminated, truncated, info = env.step(target)

        assert reward == env.reward_on_success
        assert terminated is True
        assert info["success"] is True

    def test_incorrect_answer(self, env):
        """Test incorrect answer gives zero or negative reward."""
        observation, _ = env.reset()

        observation, reward, terminated, truncated, info = env.step("wrong answer")

        assert reward == env.reward_on_failure
        assert terminated is True
        assert info["success"] is False

    def test_partial_matching(self):
        """Test partial answer matching."""
        tasks = [Task(prompt="Name a fruit", target="apple", reward=1.0)]
        env = TextWorldEnv(
            tasks=tasks, partial_reward=True, similarity_threshold=0.5
        )

        env.reset()
        observation, reward, terminated, truncated, info = env.step("apples")

        assert reward > 0.0  # Should get partial credit
        assert "similarity" in info

    def test_episode_length_limit(self):
        """Test episode length limit."""
        tasks = [Task(prompt="Test", target="answer", reward=1.0)]
        env = TextWorldEnv(tasks=tasks, max_episode_length=3)

        env.reset()

        # Take multiple steps
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step("wrong")
            if truncated:
                break

        assert env.current_step == 3


@pytest.mark.unit
class TestQAEnv:
    """Tests for QAEnv."""

    def test_qa_env_creation(self):
        """Test QA environment creation."""
        qa_pairs = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital of France?", "answer": "Paris"},
        ]

        env = QAEnv(qa_pairs=qa_pairs)

        assert len(env.tasks) == 2
        assert env.tasks[0].prompt == "What is 2+2?"
        assert env.tasks[0].target == "4"

    def test_qa_env_correct_answer(self):
        """Test QA with correct answer."""
        qa_pairs = [{"question": "What is 2+2?", "answer": "4"}]
        env = QAEnv(qa_pairs=qa_pairs)

        observation, _ = env.reset()
        observation, reward, terminated, truncated, info = env.step("4")

        assert reward > 0.0
        assert info["success"] is True

    def test_qa_env_case_insensitive(self):
        """Test QA matching is case insensitive."""
        qa_pairs = [{"question": "What is the capital?", "answer": "Paris"}]
        env = QAEnv(qa_pairs=qa_pairs)

        observation, _ = env.reset()
        observation, reward, terminated, truncated, info = env.step("paris")

        assert info["success"] is True


@pytest.mark.unit
class TestInstructionFollowingEnv:
    """Tests for InstructionFollowingEnv."""

    def test_instruction_env_creation(self):
        """Test instruction following environment creation."""
        instructions = [
            {
                "instruction": "Write a greeting.",
                "expected_output": "Hello",
                "reward": 1.0,
            },
            {
                "instruction": "Name three colors.",
                "expected_output": "red, blue, green",
                "reward": 2.0,
            },
        ]

        env = InstructionFollowingEnv(instructions=instructions)

        assert len(env.tasks) == 2
        assert env.tasks[0].prompt == "Write a greeting."
        assert env.tasks[0].target == "Hello"
        assert env.tasks[0].reward == 1.0

    def test_instruction_following(self):
        """Test instruction following."""
        instructions = [
            {"instruction": "Say hello", "expected_output": "hello", "reward": 1.0}
        ]
        env = InstructionFollowingEnv(instructions=instructions)

        observation, _ = env.reset()
        observation, reward, terminated, truncated, info = env.step("hello world")

        assert reward > 0.0  # Should get partial credit


@pytest.mark.unit
class TestCreateTextEnv:
    """Tests for create_text_env factory function."""

    def test_create_qa_env(self):
        """Test creating QA environment."""
        tasks = [Task(prompt="Q1", target="A1"), Task(prompt="Q2", target="A2")]
        env = create_text_env("qa", tasks=tasks)

        assert isinstance(env, QAEnv)
        assert len(env.tasks) == 2

    def test_create_instruction_env(self):
        """Test creating instruction environment."""
        tasks = [Task(prompt="Inst1", target="Out1"), Task(prompt="Inst2", target="Out2")]
        env = create_text_env("instruction", tasks=tasks)

        assert isinstance(env, InstructionFollowingEnv)
        assert len(env.tasks) == 2

    def test_create_generic_env(self):
        """Test creating generic text environment."""
        tasks = [Task(prompt="Test", target="Result")]
        env = create_text_env("generic", tasks=tasks)

        assert isinstance(env, TextWorldEnv)

    def test_invalid_env_type(self):
        """Test creating environment with invalid type."""
        tasks = [Task(prompt="Test", target="Result")]

        with pytest.raises(ValueError, match="Unknown environment type"):
            create_text_env("invalid_type", tasks=tasks)

    def test_empty_tasks_list(self):
        """Test creating environment with empty tasks."""
        with pytest.raises(ValueError, match="tasks list cannot be empty"):
            create_text_env("qa", tasks=[])


@pytest.mark.unit
class TestTextWorldEnvIntegration:
    """Integration tests for TextWorldEnv."""

    def test_full_episode(self):
        """Test running a complete episode."""
        tasks = [
            Task(prompt="Question 1", target="Answer 1", reward=1.0),
            Task(prompt="Question 2", target="Answer 2", reward=1.0),
        ]
        env = TextWorldEnv(tasks=tasks, max_episode_length=5)

        observation, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # Simulate agent giving correct answer
            action = observation["target"]
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        assert total_reward > 0.0
        env.close()

    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        tasks = [Task(prompt="Test", target="Answer", reward=1.0)]
        env = TextWorldEnv(tasks=tasks)

        for _ in range(3):
            observation, _ = env.reset()
            action = observation["target"]
            observation, reward, terminated, truncated, info = env.step(action)
            assert terminated is True

        env.close()
