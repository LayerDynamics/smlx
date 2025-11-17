"""
Unit tests for vision-based RL environments.

Tests VisionTaskEnv, ImageClassificationEnv, and VisualQAEnv.
"""

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from smlx.gym.envs.vision_task import (
    ImageClassificationEnv,
    VisionTask,
    VisionTaskEnv,
    VisualQAEnv,
    create_vision_env,
)


@pytest.mark.unit
class TestVisionTask:
    """Tests for VisionTask dataclass."""

    def test_basic_task(self):
        """Test creating basic vision task."""
        # Create a simple PIL image
        image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        task = VisionTask(image=image, target=0)

        assert isinstance(task.image, Image.Image)
        assert task.target == 0
        assert task.prompt == ""
        assert task.metadata == {}

    def test_task_with_prompt(self):
        """Test vision task with prompt."""
        image = Image.new("RGB", (100, 100))
        task = VisionTask(
            image=image,
            target="red",
            prompt="What color is this?",
            metadata={"category": "color"},
        )

        assert task.prompt == "What color is this?"
        assert task.metadata["category"] == "color"

    def test_task_with_path(self):
        """Test vision task with image path."""
        task = VisionTask(image="test.jpg", target=1)
        assert task.image == "test.jpg"


@pytest.mark.unit
class TestVisionTaskEnv:
    """Tests for VisionTaskEnv base class."""

    @pytest.fixture
    def tasks(self):
        """Create test vision tasks."""
        images = [
            Image.new("RGB", (224, 224), color=(255, 0, 0)),  # Red
            Image.new("RGB", (224, 224), color=(0, 255, 0)),  # Green
            Image.new("RGB", (224, 224), color=(0, 0, 255)),  # Blue
        ]
        return [
            VisionTask(image=images[0], target=0, prompt="Identify this image"),
            VisionTask(image=images[1], target=1, prompt="Identify this image"),
            VisionTask(image=images[2], target=2, prompt="Identify this image"),
        ]

    @pytest.fixture
    def env(self, tasks):
        """Create vision task environment."""
        return VisionTaskEnv(
            tasks=tasks,
            num_classes=3,
            image_size=(224, 224),
            normalize=True,
            max_episode_steps=10,
            shuffle_tasks=False,
        )

    def test_initialization(self, env, tasks):
        """Test environment initialization."""
        assert env.tasks == tasks
        assert env.num_classes == 3
        assert env.image_size == (224, 224)
        assert env.normalize is True
        assert env.max_episode_steps == 10

    def test_empty_tasks_raises_error(self):
        """Test that empty task list raises error."""
        with pytest.raises(ValueError, match="at least one vision task"):
            VisionTaskEnv(tasks=[])

    def test_observation_space(self, env):
        """Test observation space definition."""
        obs_space = env.observation_space

        assert "image" in obs_space.spaces
        assert "prompt" in obs_space.spaces
        assert "image_shape" in obs_space.spaces

        # Image should be [C, H, W]
        image_space = obs_space.spaces["image"]
        assert image_space.shape == (3, 224, 224)

    def test_action_space(self, env):
        """Test action space definition."""
        assert env.action_space.n == 3

    def test_reset(self, env):
        """Test environment reset."""
        observation, info = env.reset()

        # Check observation structure
        assert isinstance(observation, dict)
        assert "image" in observation
        assert "prompt" in observation
        assert "image_shape" in observation

        # Check MLX arrays
        assert isinstance(observation["image"], mx.array)
        assert isinstance(observation["image_shape"], mx.array)

        # Check image shape [C, H, W]
        assert observation["image"].shape == (3, 224, 224)

        # Check info
        assert "task_index" in info
        assert "task_target" in info

    def test_reset_with_seed(self, env):
        """Test reset with seed for reproducibility."""
        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)

        # With same seed and no shuffle, should get same task
        assert info1["task_index"] == info2["task_index"]

    def test_reset_with_task_index(self, env):
        """Test reset with specific task index."""
        observation, info = env.reset(options={"task_index": 1})

        assert info["task_index"] == 1
        assert env.current_task == env.tasks[1]

    def test_step_with_correct_action(self, env):
        """Test step with correct action."""
        env.reset()

        # Get target from current task
        target = env.current_task.target

        # Step with correct action
        observation, reward, terminated, truncated, info = env.step(target)

        assert reward == 1.0
        assert terminated is True
        assert info["correct"] is True

    def test_step_with_incorrect_action(self, env):
        """Test step with incorrect action."""
        env.reset()

        # Get target and pick wrong action
        target = env.current_task.target
        wrong_action = (target + 1) % env.num_classes

        # Step with wrong action
        observation, reward, terminated, truncated, info = env.step(wrong_action)

        assert reward == 0.0
        assert terminated is False
        assert info["correct"] is False

    def test_step_max_episodes(self, env):
        """Test truncation at max episode steps."""
        env.reset()

        for step in range(env.max_episode_steps + 1):
            observation, reward, terminated, truncated, info = env.step(0)

            if truncated:
                assert step + 1 >= env.max_episode_steps
                break

    def test_image_is_mlx_array(self, env):
        """Test that image observations are MLX arrays."""
        observation, info = env.reset()

        image = observation["image"]
        assert isinstance(image, mx.array)
        assert image.shape == (3, 224, 224)

    def test_image_preprocessing(self, env):
        """Test image preprocessing pipeline."""
        observation, info = env.reset()

        image = observation["image"]

        # Image should be preprocessed and normalized
        assert isinstance(image, mx.array)
        # After normalization, values should be roughly in [-3, 3] range
        image_np = np.array(image)
        assert image_np.min() >= -10.0
        assert image_np.max() <= 10.0

    def test_episode_statistics(self, env):
        """Test episode statistics tracking."""
        env.reset()

        # Step through episode
        for _ in range(3):
            target = env.current_task.target
            observation, reward, terminated, truncated, info = env.step(target)

            if terminated or truncated:
                break

        # Check statistics
        assert env.episode_total > 0
        assert "accuracy" in info

    def test_render_human_mode(self):
        """Test rendering in human mode."""
        image = Image.new("RGB", (224, 224))
        tasks = [VisionTask(image=image, target=0)]
        env = VisionTaskEnv(tasks=tasks, render_mode="human")

        env.reset()
        # Should not raise an error
        env.render()
        env.close()

    def test_render_rgb_array_mode(self):
        """Test rendering in rgb_array mode."""
        image = Image.new("RGB", (224, 224), color=(255, 0, 0))
        tasks = [VisionTask(image=image, target=0)]
        env = VisionTaskEnv(tasks=tasks, render_mode="rgb_array")

        env.reset()
        rgb_array = env.render()

        assert rgb_array is not None
        assert isinstance(rgb_array, np.ndarray)

    def test_custom_reward_function(self):
        """Test custom reward function."""

        def custom_reward(prediction, target):
            return 0.5 if prediction == target else 0.0

        image = Image.new("RGB", (224, 224))
        tasks = [VisionTask(image=image, target=0)]
        env = VisionTaskEnv(tasks=tasks, reward_fn=custom_reward)

        env.reset()
        observation, reward, terminated, truncated, info = env.step(0)

        assert reward == 0.5

    def test_normalization_parameters(self):
        """Test custom normalization parameters."""
        image = Image.new("RGB", (224, 224))
        tasks = [VisionTask(image=image, target=0)]

        custom_mean = [0.5, 0.5, 0.5]
        custom_std = [0.5, 0.5, 0.5]

        env = VisionTaskEnv(
            tasks=tasks, normalize=True, mean=custom_mean, std=custom_std
        )

        assert env.mean == custom_mean
        assert env.std == custom_std


@pytest.mark.unit
class TestImageClassificationEnv:
    """Tests for ImageClassificationEnv."""

    @pytest.fixture
    def tasks(self):
        """Create image classification tasks."""
        return [
            VisionTask(image=Image.new("RGB", (224, 224)), target=0),
            VisionTask(image=Image.new("RGB", (224, 224)), target=1),
            VisionTask(image=Image.new("RGB", (224, 224)), target=2),
        ]

    @pytest.fixture
    def env(self, tasks):
        """Create image classification environment."""
        return ImageClassificationEnv(
            tasks=tasks, num_classes=3, class_names=["cat", "dog", "bird"]
        )

    def test_initialization(self, env):
        """Test environment initialization."""
        assert env.num_classes == 3
        assert env.class_names == ["cat", "dog", "bird"]
        assert env.top_k == 1

    def test_class_names_validation(self):
        """Test that class names must match num_classes."""
        tasks = [VisionTask(image=Image.new("RGB", (224, 224)), target=0)]

        with pytest.raises(ValueError, match="Number of class names"):
            ImageClassificationEnv(tasks=tasks, num_classes=3, class_names=["a", "b"])

    def test_correct_classification(self, env):
        """Test correct classification."""
        env.reset()
        target_class = env.current_task.target

        observation, reward, terminated, truncated, info = env.step(target_class)

        assert reward == 1.0
        assert terminated is True

    def test_incorrect_classification(self, env):
        """Test incorrect classification."""
        env.reset()
        target_class = env.current_task.target
        wrong_class = (target_class + 1) % env.num_classes

        observation, reward, terminated, truncated, info = env.step(wrong_class)

        assert reward == 0.0
        assert terminated is False

    def test_top_k_predictions(self):
        """Test top-k prediction support."""
        tasks = [VisionTask(image=Image.new("RGB", (224, 224)), target=1)]
        env = ImageClassificationEnv(tasks=tasks, num_classes=5, top_k=3)

        env.reset()

        # Provide top-3 predictions as a list
        top_k_preds = [0, 1, 2]  # Target is 1, so this should match

        observation, reward, terminated, truncated, info = env.step(top_k_preds)

        # Should get reward since target is in top-k
        assert reward == 1.0

    def test_top_k_miss(self):
        """Test top-k predictions that miss target."""
        tasks = [VisionTask(image=Image.new("RGB", (224, 224)), target=4)]
        env = ImageClassificationEnv(tasks=tasks, num_classes=5, top_k=3)

        env.reset()

        # Provide top-3 predictions that don't include target
        top_k_preds = [0, 1, 2]  # Target is 4

        observation, reward, terminated, truncated, info = env.step(top_k_preds)

        # Should not get reward
        assert reward == 0.0

    def test_classification_with_mlx_array_prediction(self):
        """Test classification with MLX array prediction."""
        tasks = [VisionTask(image=Image.new("RGB", (224, 224)), target=1)]
        env = ImageClassificationEnv(tasks=tasks, num_classes=3)

        env.reset()

        # Provide prediction as MLX array
        prediction = mx.array([0, 1, 2])

        observation, reward, terminated, truncated, info = env.step(prediction)

        # Should handle MLX array
        assert isinstance(reward, float)


@pytest.mark.unit
class TestVisualQAEnv:
    """Tests for VisualQAEnv."""

    @pytest.fixture
    def tasks(self):
        """Create visual QA tasks."""
        return [
            VisionTask(
                image=Image.new("RGB", (224, 224), color=(255, 0, 0)),
                prompt="What color is this?",
                target="red",
            ),
            VisionTask(
                image=Image.new("RGB", (224, 224)),
                prompt="How many objects?",
                target="3",
            ),
        ]

    @pytest.fixture
    def env(self, tasks):
        """Create visual QA environment."""
        return VisualQAEnv(tasks=tasks, partial_credit=False, case_sensitive=False)

    def test_initialization(self, env):
        """Test environment initialization."""
        assert env.partial_credit is False
        assert env.case_sensitive is False

    def test_exact_match_reward(self, env):
        """Test exact match reward."""
        env.reset()
        target = env.current_task.target

        observation, reward, terminated, truncated, info = env.step(target)

        assert reward == 1.0
        assert terminated is True

    def test_case_insensitive_matching(self, env):
        """Test case insensitive matching."""
        env.reset()
        target = env.current_task.target

        # Mixed case should still match
        observation, reward, terminated, truncated, info = env.step(target.upper())

        assert reward == 1.0

    def test_case_sensitive_mode(self):
        """Test case sensitive matching."""
        tasks = [
            VisionTask(
                image=Image.new("RGB", (224, 224)), prompt="Test?", target="Red"
            )
        ]
        env = VisualQAEnv(tasks=tasks, case_sensitive=True)

        env.reset()

        # Wrong case should not match
        observation, reward, terminated, truncated, info = env.step("red")

        assert reward == 0.0

    def test_partial_credit(self):
        """Test partial credit reward."""
        tasks = [
            VisionTask(
                image=Image.new("RGB", (224, 224)),
                prompt="Describe this",
                target="red square",
            )
        ]
        env = VisualQAEnv(tasks=tasks, partial_credit=True)

        env.reset()

        # Partial match: only "red"
        observation, reward, terminated, truncated, info = env.step("red")

        # Should get partial credit (substring match)
        assert 0.0 < reward <= 1.0

    def test_partial_credit_word_overlap(self):
        """Test partial credit based on word overlap."""
        tasks = [
            VisionTask(
                image=Image.new("RGB", (224, 224)),
                prompt="What is this?",
                target="a red car",
            )
        ]
        env = VisualQAEnv(tasks=tasks, partial_credit=True)

        env.reset()

        # Some word overlap
        observation, reward, terminated, truncated, info = env.step("red car")

        # Should get some credit
        assert reward > 0.0

    def test_step_with_string_action(self):
        """Test that VQA accepts string actions."""
        tasks = [
            VisionTask(
                image=Image.new("RGB", (224, 224)), prompt="Test?", target="answer"
            )
        ]
        env = VisualQAEnv(tasks=tasks)

        env.reset()

        # Should accept string action
        observation, reward, terminated, truncated, info = env.step("answer")

        assert "answer" in info
        assert info["answer"] == "answer"

    def test_info_contains_answer(self, env):
        """Test that info dict contains answer."""
        env.reset()

        observation, reward, terminated, truncated, info = env.step("some answer")

        assert "answer" in info
        assert "target" in info
        assert info["answer"] == "some answer"


@pytest.mark.unit
class TestCreateVisionEnv:
    """Tests for create_vision_env factory function."""

    def test_create_basic_vision_env(self):
        """Test creating basic vision environment."""
        tasks = [VisionTask(image=Image.new("RGB", (224, 224)), target=0)]
        env = create_vision_env("vision", tasks)

        assert isinstance(env, VisionTaskEnv)
        env.close()

    def test_create_classification_env(self):
        """Test creating classification environment."""
        tasks = [VisionTask(image=Image.new("RGB", (224, 224)), target=0)]
        env = create_vision_env("classification", tasks, num_classes=10)

        assert isinstance(env, ImageClassificationEnv)
        assert env.num_classes == 10
        env.close()

    def test_create_vqa_env(self):
        """Test creating VQA environment."""
        tasks = [
            VisionTask(
                image=Image.new("RGB", (224, 224)), prompt="What?", target="answer"
            )
        ]
        env = create_vision_env("vqa", tasks, partial_credit=True)

        assert isinstance(env, VisualQAEnv)
        assert env.partial_credit is True
        env.close()

    def test_invalid_task_type(self):
        """Test that invalid task type raises error."""
        tasks = [VisionTask(image=Image.new("RGB", (224, 224)), target=0)]

        with pytest.raises(ValueError, match="Unknown task_type"):
            create_vision_env("invalid_type", tasks)


@pytest.mark.integration
class TestVisionTaskIntegration:
    """Integration tests for vision task environments."""

    def test_complete_classification_episode(self):
        """Test complete classification episode."""
        tasks = [
            VisionTask(image=Image.new("RGB", (224, 224)), target=i % 3)
            for i in range(10)
        ]

        env = ImageClassificationEnv(
            tasks=tasks, num_classes=3, max_episode_steps=15
        )

        observation, info = env.reset()

        for _ in range(15):
            # Random classification
            action = np.random.randint(0, 3)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Should have tracked accuracy
        assert "accuracy" in info

    def test_complete_vqa_episode(self):
        """Test complete VQA episode."""
        tasks = [
            VisionTask(
                image=Image.new("RGB", (224, 224)),
                prompt=f"Question {i}?",
                target=f"answer_{i}",
            )
            for i in range(5)
        ]

        env = VisualQAEnv(tasks=tasks, max_episode_steps=10)

        observation, info = env.reset()

        for step in range(10):
            # Random answer
            answer = f"answer_{step % 5}"
            observation, reward, terminated, truncated, info = env.step(answer)

            if terminated or truncated:
                break

        assert "final_accuracy" in info or "accuracy" in info

    def test_image_preprocessing_pipeline(self):
        """Test complete image preprocessing pipeline."""
        # Create image with known colors
        image = Image.new("RGB", (100, 100), color=(128, 64, 32))
        tasks = [VisionTask(image=image, target=0)]

        env = VisionTaskEnv(
            tasks=tasks, image_size=(224, 224), normalize=True
        )

        observation, info = env.reset()

        # Image should be resized and normalized
        processed_image = observation["image"]
        assert processed_image.shape == (3, 224, 224)
        assert isinstance(processed_image, mx.array)

    def test_multi_task_switching(self):
        """Test switching between multiple tasks."""
        tasks = [
            VisionTask(image=Image.new("RGB", (224, 224)), target=i)
            for i in range(5)
        ]

        env = VisionTaskEnv(
            tasks=tasks, num_classes=5, max_episode_steps=10, shuffle_tasks=False
        )

        # Track which tasks we see
        seen_targets = set()

        for _ in range(10):
            observation, info = env.reset()
            seen_targets.add(info["task_target"])

            # Take wrong action to move to next task
            observation, reward, terminated, truncated, info = env.step(
                (info["task_target"] + 1) % 5
            )

        # Should have seen multiple tasks
        assert len(seen_targets) > 1

    def test_image_shape_tracking(self):
        """Test image shape tracking in observations."""
        # Different size images
        tasks = [
            VisionTask(image=Image.new("RGB", (100, 200)), target=0),
            VisionTask(image=Image.new("RGB", (300, 150)), target=1),
        ]

        env = VisionTaskEnv(tasks=tasks, image_size=(224, 224))

        # Check first task
        observation, info = env.reset(options={"task_index": 0})
        image_shape = observation["image_shape"]
        assert int(image_shape[0]) == 100  # width
        assert int(image_shape[1]) == 200  # height

        # Check second task
        observation, info = env.reset(options={"task_index": 1})
        image_shape = observation["image_shape"]
        assert int(image_shape[0]) == 300
        assert int(image_shape[1]) == 150

    def test_episode_statistics_accumulation(self):
        """Test that episode statistics accumulate correctly."""
        tasks = [VisionTask(image=Image.new("RGB", (224, 224)), target=0) for _ in range(5)]

        env = VisionTaskEnv(tasks=tasks, num_classes=1, max_episode_steps=10)

        observation, info = env.reset()

        # Make several predictions
        correct_count = 0
        for i in range(5):
            if i < 2:
                # Correct answers
                observation, reward, terminated, truncated, info = env.step(0)
                correct_count += 1
            else:
                # Wrong answers (would be wrong if target is 0)
                observation, reward, terminated, truncated, info = env.step(0)
                if reward > 0:
                    correct_count += 1

            if terminated or truncated:
                break

        # Check final accuracy
        assert info["episode_correct"] >= 0
        assert info["episode_total"] >= correct_count

    def test_different_image_sizes(self):
        """Test handling images of different sizes."""
        tasks = [
            VisionTask(image=Image.new("RGB", (50, 50)), target=0),
            VisionTask(image=Image.new("RGB", (500, 500)), target=1),
            VisionTask(image=Image.new("RGB", (100, 300)), target=2),
        ]

        env = VisionTaskEnv(tasks=tasks, num_classes=3, image_size=(224, 224))

        for task_idx in range(3):
            observation, info = env.reset(options={"task_index": task_idx})

            # All should be resized to same size
            assert observation["image"].shape == (3, 224, 224)

    def test_vlm_integration_scenario(self):
        """Test scenario simulating VLM integration."""
        tasks = [
            VisionTask(
                image=Image.new("RGB", (224, 224), color=(255, 0, 0)),
                prompt="What color is this image?",
                target="red",
            ),
            VisionTask(
                image=Image.new("RGB", (224, 224), color=(0, 255, 0)),
                prompt="What color is this image?",
                target="green",
            ),
        ]

        env = VisualQAEnv(tasks=tasks, partial_credit=True)

        observation, info = env.reset()

        # Get image and prompt
        image = observation["image"]
        prompt = observation["prompt"]

        assert isinstance(image, mx.array)
        assert isinstance(prompt, str)
        assert "color" in prompt.lower()

        # Simulate VLM prediction
        vlm_answer = "red"
        observation, reward, terminated, truncated, info = env.step(vlm_answer)

        # Should get reward if correct
        assert reward >= 0.0
        assert "answer" in info
