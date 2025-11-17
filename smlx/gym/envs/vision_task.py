"""
Vision-based environment for visual RL tasks.

This module provides environments for vision-based reinforcement learning tasks
such as image classification, visual question answering, object detection, and
other computer vision tasks. All observations are MLX arrays for Metal GPU acceleration.

Reference: SMLX_Gym.md, Section 4.1 (Environment Implementations)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import mlx.core as mx
import numpy as np
from PIL import Image

from smlx.gym.base import MLXEnv
from smlx.utils.vision import load_image, preprocess_image

# Suppress unused import warning - field is used in dataclass
_ = field


@dataclass
class VisionTask:
    """
    Represents a vision-based task.

    Attributes:
        image: Image source (path, URL, or PIL Image)
        target: Target label or answer
        prompt: Optional text prompt for VQA tasks
        metadata: Additional task metadata
    """

    image: Union[str, Path, Image.Image]
    target: Any
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class VisionTaskEnv(MLXEnv):
    """
    Vision-based environment for image understanding tasks.

    This environment provides a flexible interface for vision-based RL tasks
    including image classification, visual question answering, and object
    detection. All observations are returned as MLX arrays for Metal GPU
    acceleration throughout the training pipeline.

    The environment can be configured with:
    - Custom vision task datasets
    - Flexible reward functions
    - Image preprocessing pipelines
    - Episode length limits

    Observation Space:
        Dict space containing:
        - 'image': MLX array of shape [C, H, W] with preprocessed image
        - 'prompt': Text prompt (for VQA tasks)
        - 'image_shape': Original image dimensions

    Action Space:
        Discrete space for classification tasks, or custom for other tasks

    Rewards:
        - Default: Exact match with target label
        - Custom: User-provided reward function

    Example:
        ```python
        from smlx.gym.envs.vision_task import VisionTask, VisionTaskEnv

        # Create vision tasks
        tasks = [
            VisionTask(image="cat.jpg", target=0, prompt="What animal is this?"),
            VisionTask(image="dog.jpg", target=1, prompt="What animal is this?"),
        ]

        env = VisionTaskEnv(
            tasks=tasks,
            num_classes=2,
            image_size=(224, 224),
            max_episode_steps=10
        )

        # Use with vision model
        obs, info = env.reset()
        image_array = obs['image']  # MLX array [C, H, W]
        print(f"Image shape: {image_array.shape}")

        # Agent predicts class
        predicted_class = 0
        next_obs, reward, terminated, truncated, info = env.step(predicted_class)
        ```

    Integration with Vision Models:
        ```python
        from smlx.models.SmolVLM_256M import load, generate

        # Load vision-language model
        model, processor = load("HuggingFaceTB/SmolVLM-256M-Instruct")

        env = VisionTaskEnv(tasks=vision_tasks)
        obs, info = env.reset()

        # Process image and prompt
        image = obs['image']
        prompt = obs['prompt']

        # Generate response
        response = generate(model, processor, prompt, image)

        # Extract predicted class from response
        predicted_class = parse_response(response)

        # Take action
        next_obs, reward, terminated, truncated, info = env.step(predicted_class)
        ```
    """

    def __init__(
        self,
        tasks: list[VisionTask],
        num_classes: int = 1000,
        image_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
        max_episode_steps: int = 100,
        reward_fn: Optional[Callable[[Any, Any], float]] = None,
        shuffle_tasks: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize vision-based environment.

        Args:
            tasks: List of VisionTask objects defining the environment tasks
            num_classes: Number of classes for classification tasks
            image_size: Target image size (width, height) for preprocessing
            normalize: Whether to normalize images with mean/std
            mean: Mean values for normalization [R, G, B] (default: ImageNet)
            std: Std values for normalization [R, G, B] (default: ImageNet)
            max_episode_steps: Maximum steps per episode
            reward_fn: Custom reward function (prediction, target) -> reward
            shuffle_tasks: Whether to shuffle tasks between episodes
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__(render_mode=render_mode)

        if not tasks:
            raise ValueError("Must provide at least one vision task")

        self.tasks = tasks
        self.num_classes = num_classes
        self.image_size = image_size
        self.normalize = normalize
        self.max_episode_steps = max_episode_steps
        self.shuffle_tasks = shuffle_tasks

        # Set normalization parameters (ImageNet defaults)
        if normalize:
            self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
            self.std = std if std is not None else [0.229, 0.224, 0.225]
        else:
            self.mean = None
            self.std = None

        # Set reward function
        if reward_fn is None:
            self.reward_fn = self._default_reward
        else:
            self.reward_fn = reward_fn

        # Define observation space
        # Image observations are MLX arrays of shape [C, H, W]
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=-10.0,  # Allow negative values after normalization
                    high=10.0,
                    shape=(3, image_size[1], image_size[0]),  # [C, H, W]
                    dtype=np.float32,
                ),
                "prompt": gym.spaces.Text(max_length=500),
                "image_shape": gym.spaces.Box(
                    low=0, high=10000, shape=(2,), dtype=np.int32
                ),
            }
        )

        # Action space: Discrete for classification
        self.action_space = gym.spaces.Discrete(num_classes)

        # Episode state
        self.current_task: Optional[VisionTask] = None
        self.current_image: Optional[mx.array] = None
        self.original_image: Optional[Image.Image] = None
        self.current_step = 0
        self.task_index = 0

        # Statistics
        self.episode_correct = 0
        self.episode_total = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options. Supported keys:
                    - 'task_index': Specific task index to start with

        Returns:
            observation: Dict containing image (MLX array), prompt, and metadata
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Shuffle tasks if enabled
        if self.shuffle_tasks and seed is not None:
            import random
            random.seed(seed)
            random.shuffle(self.tasks)

        # Select task
        if options and "task_index" in options:
            self.task_index = options["task_index"] % len(self.tasks)
        elif not self.shuffle_tasks:
            # When not shuffling, cycle through tasks sequentially
            # Don't reset to 0 on subsequent resets
            pass  # Keep current task_index
        else:
            self.task_index = 0

        self.current_task = self.tasks[self.task_index]
        self.current_step = 0
        self.episode_correct = 0
        self.episode_total = 0

        # Load and preprocess image using MLX
        self._load_current_image()

        # Build observation
        observation = self._get_observation()

        info = {
            "task_index": self.task_index,
            "task_target": self.current_task.target,
            "task_prompt": self.current_task.prompt,
            "task_metadata": self.current_task.metadata,
            "image_source": str(self.current_task.image),
        }

        return observation, info

    def step(
        self, action: Union[int, list, tuple, np.ndarray, mx.array]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Predicted class or action from the agent.
                   Can be int (single prediction) or list/array (top-k predictions)

        Returns:
            observation: Next observation dict with MLX arrays
            reward: Reward for this step
            terminated: Whether episode ended naturally (task completed)
            truncated: Whether episode was truncated (max steps reached)
            info: Additional information
        """
        if self.current_task is None:
            raise RuntimeError("Must call reset() before step()")

        # Calculate reward using MLX operations if needed
        target = self.current_task.target
        reward = self.reward_fn(action, target)

        # Update statistics
        self.episode_total += 1
        if reward >= 1.0:
            self.episode_correct += 1

        # Update step counter
        self.current_step += 1

        # Check termination conditions
        terminated = bool(reward >= 1.0)  # Task completed successfully
        truncated = self.current_step >= self.max_episode_steps

        # Move to next task if not done
        if not (terminated or truncated):
            self.task_index = (self.task_index + 1) % len(self.tasks)
            self.current_task = self.tasks[self.task_index]
            self._load_current_image()

        # Build next observation
        observation = self._get_observation()

        # Build info dict
        info = {
            "prediction": action,
            "target": target,
            "reward": reward,
            "correct": reward >= 1.0,
            "episode_correct": self.episode_correct,
            "episode_total": self.episode_total,
            "accuracy": (
                self.episode_correct / self.episode_total if self.episode_total > 0 else 0.0
            ),
        }

        if terminated or truncated:
            info["final_accuracy"] = (
                self.episode_correct / self.episode_total if self.episode_total > 0 else 0.0
            )

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the current environment state.

        In 'human' mode, displays the current image and task information.
        In 'rgb_array' mode, returns the current image as numpy array.
        """
        if self.render_mode == "human":
            print("\n" + "=" * 60)
            print(f"Step: {self.current_step}/{self.max_episode_steps}")
            print(f"Task {self.task_index + 1}/{len(self.tasks)}")
            print("=" * 60)

            if self.current_task:
                print(f"\nPrompt: {self.current_task.prompt}")
                print(f"Target: {self.current_task.target}")
                print(f"Image: {self.current_task.image}")

            print(f"\nAccuracy: {self.episode_correct}/{self.episode_total}")
            print("=" * 60 + "\n")

            # Note: We don't call image.show() as it spawns external processes
            # Users can access env.original_image directly if they need to display it

        elif self.render_mode == "rgb_array":
            if self.original_image:
                return np.array(self.original_image)
            return None

    def _load_current_image(self):
        """
        Load and preprocess the current task's image using MLX.

        Converts the image to an MLX array for Metal GPU acceleration.
        """
        if self.current_task is None:
            return

        # Load image using utility function
        self.original_image = load_image(self.current_task.image)

        # Preprocess image to MLX array using utility function
        # This returns shape [C, H, W]
        # Use "exact" mode to ensure consistent dimensions for gym environment
        self.current_image = preprocess_image(
            self.original_image,
            target_size=self.image_size,
            resize_mode="exact",
            mean=self.mean,
            std=self.std,
            rescale_factor=1.0 / 255.0,
        )

        # Ensure it's an MLX array for Metal GPU operations
        if not isinstance(self.current_image, mx.array):
            self.current_image = mx.array(self.current_image)

    def _get_observation(self) -> dict[str, Any]:
        """
        Build observation dictionary with MLX arrays.

        Returns:
            Dict containing image (MLX array), prompt, and shape info
        """
        if self.current_task is None or self.current_image is None:
            # Return empty observation
            return {
                "image": mx.zeros((3, self.image_size[1], self.image_size[0])),
                "prompt": "",
                "image_shape": mx.array([0, 0]),
            }

        # Get original image dimensions
        if self.original_image:
            width, height = self.original_image.size
            image_shape = mx.array([width, height])
        else:
            image_shape = mx.array([self.image_size[0], self.image_size[1]])

        return {
            "image": self.current_image,  # MLX array [C, H, W]
            "prompt": self.current_task.prompt,
            "image_shape": image_shape,  # MLX array [W, H]
        }

    def _default_reward(
        self, prediction: Union[int, list, tuple, np.ndarray, mx.array], target: Any
    ) -> float:
        """
        Default reward function: exact match.

        Uses MLX operations for comparison when applicable.

        Args:
            prediction: Agent's prediction (int or array for top-k)
            target: Target label

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Convert to MLX arrays for GPU operations if needed
        if isinstance(prediction, (int, float)) and isinstance(target, (int, float)):
            pred_mx = mx.array(prediction)
            target_mx = mx.array(target)
            match = mx.array_equal(pred_mx, target_mx)
            return 1.0 if match else 0.0
        else:
            return 1.0 if prediction == target else 0.0

    def close(self):
        """Clean up environment resources."""
        self.original_image = None
        self.current_image = None


class ImageClassificationEnv(VisionTaskEnv):
    """
    Image classification environment.

    Specialized environment for multi-class image classification tasks.
    Provides utilities for loading image datasets and computing
    classification metrics.

    Example:
        ```python
        from smlx.gym.envs.vision_task import VisionTask, ImageClassificationEnv

        # Create classification tasks
        tasks = [
            VisionTask(image="images/cat_001.jpg", target=0),  # cat
            VisionTask(image="images/dog_001.jpg", target=1),  # dog
            VisionTask(image="images/cat_002.jpg", target=0),  # cat
        ]

        env = ImageClassificationEnv(
            tasks=tasks,
            num_classes=2,
            class_names=["cat", "dog"]
        )

        # Use with vision model
        obs, info = env.reset()
        image = obs['image']  # MLX array for model input

        # Model prediction (using MLX operations)
        import mlx.core as mx
        logits = model(mx.expand_dims(image, 0))  # Add batch dimension
        predicted_class = int(mx.argmax(logits, axis=-1)[0])

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(predicted_class)
        ```
    """

    def __init__(
        self,
        tasks: list[VisionTask],
        num_classes: int,
        class_names: Optional[list[str]] = None,
        top_k: int = 1,
        **kwargs,
    ):
        """
        Initialize image classification environment.

        Args:
            tasks: List of vision tasks with image and target class
            num_classes: Number of classes
            class_names: Optional list of class names for logging
            top_k: Consider top-k predictions for reward (default: 1)
            **kwargs: Additional arguments passed to VisionTaskEnv
        """
        super().__init__(tasks=tasks, num_classes=num_classes, **kwargs)

        self.class_names = class_names
        self.top_k = top_k

        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(
                f"Number of class names ({len(class_names)}) must match "
                f"num_classes ({num_classes})"
            )

    def _default_reward(
        self, prediction: Union[int, list, tuple, np.ndarray, mx.array], target: int
    ) -> float:
        """
        Classification reward with top-k support.

        Args:
            prediction: Predicted class (int) or array of top-k predictions
            target: Target class

        Returns:
            1.0 if target in top-k predictions, 0.0 otherwise
        """
        # Handle single prediction
        if isinstance(prediction, (int, np.integer)):
            pred_mx = mx.array(prediction)
            target_mx = mx.array(target)
            return 1.0 if mx.array_equal(pred_mx, target_mx) else 0.0

        # Handle top-k predictions (array)
        if isinstance(prediction, (list, tuple, np.ndarray, mx.array)):
            pred_mx = mx.array(prediction) if not isinstance(prediction, mx.array) else prediction
            target_mx = mx.array(target)
            # Check if target is in top-k predictions using MLX operations
            matches = pred_mx == target_mx
            # Handle both scalar and array results from comparison
            if isinstance(matches, bool) or matches.ndim == 0:
                return 1.0 if bool(matches) else 0.0
            else:
                return 1.0 if bool(mx.any(matches)) else 0.0

        return 0.0


class VisualQAEnv(VisionTaskEnv):
    """
    Visual Question Answering environment.

    Environment for VQA tasks where the agent must answer questions about images.
    Integrates with vision-language models like SmolVLM.

    Example:
        ```python
        from smlx.gym.envs.vision_task import VisionTask, VisualQAEnv

        # Create VQA tasks
        tasks = [
            VisionTask(
                image="beach.jpg",
                prompt="What is the weather like?",
                target="sunny"
            ),
            VisionTask(
                image="kitchen.jpg",
                prompt="How many chairs are there?",
                target="4"
            ),
        ]

        env = VisualQAEnv(tasks=tasks, partial_credit=True)

        # Use with VLM
        obs, info = env.reset()
        image = obs['image']  # MLX array
        question = obs['prompt']

        # Generate answer using VLM
        answer = vlm.generate(image, question)

        # Environment validates answer
        next_obs, reward, terminated, truncated, info = env.step(answer)
        ```
    """

    def __init__(
        self,
        tasks: list[VisionTask],
        partial_credit: bool = False,
        case_sensitive: bool = False,
        **kwargs,
    ):
        """
        Initialize visual QA environment.

        Args:
            tasks: List of VQA tasks with image, prompt, and target answer
            partial_credit: Whether to give partial credit for similar answers
            case_sensitive: Whether answer matching is case-sensitive
            **kwargs: Additional arguments passed to VisionTaskEnv
        """
        # Set custom reward function
        if partial_credit:
            reward_fn = self._partial_credit_reward
        else:
            reward_fn = self._exact_match_reward

        # Don't pass reward_fn to parent since we override step() completely
        super().__init__(tasks=tasks, num_classes=1, **kwargs)

        # Store our custom reward function
        if reward_fn:
            self.reward_fn = reward_fn

        self.case_sensitive = case_sensitive
        self.partial_credit = partial_credit

    def step(
        self, action: str
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Step with text answer.

        Args:
            action: Text answer from VLM

        Returns:
            Standard gym step outputs
        """
        if self.current_task is None:
            raise RuntimeError("Must call reset() before step()")

        # For VQA, action is a string answer, not an integer
        # Calculate reward directly here instead of calling parent
        target = self.current_task.target
        reward = self.reward_fn(action, target)

        # Update statistics
        self.episode_total += 1
        if reward >= 1.0:
            self.episode_correct += 1

        # Update step counter
        self.current_step += 1

        # Check termination conditions
        terminated = bool(reward >= 1.0)
        truncated = self.current_step >= self.max_episode_steps

        # Move to next task if not done
        if not (terminated or truncated):
            self.task_index = (self.task_index + 1) % len(self.tasks)
            self.current_task = self.tasks[self.task_index]
            self._load_current_image()

        # Build next observation
        observation = self._get_observation()

        # Build info dict
        info = {
            "answer": action,
            "target": target,
            "reward": reward,
            "correct": reward >= 1.0,
            "episode_correct": self.episode_correct,
            "episode_total": self.episode_total,
            "accuracy": (
                self.episode_correct / self.episode_total if self.episode_total > 0 else 0.0
            ),
        }

        if terminated or truncated:
            info["final_accuracy"] = (
                self.episode_correct / self.episode_total if self.episode_total > 0 else 0.0
            )

        return observation, reward, terminated, truncated, info

    def _exact_match_reward(self, answer: str, target: str) -> float:
        """
        Exact match reward for VQA.

        Args:
            answer: Generated answer
            target: Target answer

        Returns:
            1.0 for exact match, 0.0 otherwise
        """
        if not self.case_sensitive:
            answer = answer.lower().strip()
            target = target.lower().strip()
        else:
            answer = answer.strip()
            target = target.strip()

        return 1.0 if answer == target else 0.0

    def _partial_credit_reward(self, answer: str, target: str) -> float:
        """
        Partial credit reward based on string similarity.

        Args:
            answer: Generated answer
            target: Target answer

        Returns:
            Reward in [0, 1] based on similarity
        """
        if not self.case_sensitive:
            answer = answer.lower().strip()
            target = target.lower().strip()
        else:
            answer = answer.strip()
            target = target.strip()

        # Exact match
        if answer == target:
            return 1.0

        # Substring match
        if target in answer or answer in target:
            return 0.7

        # Word overlap
        answer_words = set(answer.split())
        target_words = set(target.split())
        if answer_words and target_words:
            overlap = len(answer_words & target_words)
            total = len(answer_words | target_words)
            if total > 0:
                return 0.5 * (overlap / total)

        return 0.0


def create_vision_env(
    task_type: str,
    tasks: list[VisionTask],
    **kwargs,
) -> VisionTaskEnv:
    """
    Factory function for creating vision environments.

    All returned environments use MLX arrays for Metal GPU acceleration.

    Args:
        task_type: Type of environment ('vision', 'classification', 'vqa')
        tasks: List of vision tasks
        **kwargs: Additional arguments passed to environment constructor

    Returns:
        Vision environment instance with MLX support

    Raises:
        ValueError: If task_type is not recognized

    Example:
        ```python
        from smlx.gym.envs.vision_task import VisionTask, create_vision_env

        tasks = [VisionTask(image="img.jpg", target=0)]

        # Create classification environment
        env = create_vision_env('classification', tasks, num_classes=10)

        # Create VQA environment
        vqa_tasks = [VisionTask(image="img.jpg", prompt="What?", target="answer")]
        env = create_vision_env('vqa', vqa_tasks, partial_credit=True)
        ```
    """
    if task_type == "vision":
        return VisionTaskEnv(tasks=tasks, **kwargs)
    elif task_type == "classification":
        return ImageClassificationEnv(tasks=tasks, **kwargs)
    elif task_type == "vqa":
        return VisualQAEnv(tasks=tasks, **kwargs)
    else:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Supported types: 'vision', 'classification', 'vqa'"
        )
