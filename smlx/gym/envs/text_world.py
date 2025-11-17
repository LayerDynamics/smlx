"""
Text-based environment for language grounding tasks.

This module provides a flexible text-based environment that can be used for
various NLP tasks such as question answering, instruction following, dialogue,
and other language-based reinforcement learning scenarios.

Reference: SMLX_Gym.md, Section 4.1 (Environment Implementations)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import gymnasium as gym
import mlx.core as mx
from gymnasium import spaces

from smlx.gym.base import MLXEnv


@dataclass
class Task:
    """
    Represents a text-based task.

    Attributes:
        prompt: The task prompt or question
        target: The target response or answer
        reward: Reward value for completing this task (default: 1.0)
        context: Optional context information
        metadata: Additional task metadata
    """

    prompt: str
    target: str
    reward: float = 1.0
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class TextWorldEnv(MLXEnv):
    """
    Text-based environment for language tasks.

    This environment provides a flexible interface for text-based RL tasks
    including question answering, instruction following, and dialogue. It
    uses text strings for both observations and actions, making it ideal
    for language model agents.

    The environment can be configured with:
    - Custom task datasets
    - Flexible reward functions
    - Episode length limits
    - Text preprocessing

    Observation Space:
        Dict space containing:
        - 'prompt': Text prompt (string)
        - 'context': Optional context (string)
        - 'history': Conversation history (string)

    Action Space:
        Text space (represented as integers for compatibility)
        Agents should generate text responses that are converted to actions.

    Rewards:
        - Default: Exact match with target response
        - Custom: User-provided reward function

    Example:
        ```python
        # Create environment with custom tasks
        tasks = [
            Task(prompt="What is 2+2?", target="4"),
            Task(prompt="What color is the sky?", target="blue"),
        ]

        env = TextWorldEnv(
            tasks=tasks,
            max_episode_steps=10,
            reward_fn=lambda response, target: 1.0 if response == target else 0.0
        )

        # Use with an RL agent
        obs, info = env.reset()
        print(obs['prompt'])  # "What is 2+2?"

        # Agent generates response
        response = "4"
        next_obs, reward, terminated, truncated, info = env.step(response)
        print(reward)  # 1.0
        ```

    Integration with Language Models:
        ```python
        from smlx.models.SmolLM2_135M import load, generate

        # Load language model
        model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

        env = TextWorldEnv(tasks=tasks)
        obs, info = env.reset()

        # Generate response using model
        prompt = obs['prompt']
        response = generate(model, tokenizer, prompt, max_tokens=50)

        # Take action in environment
        next_obs, reward, terminated, truncated, info = env.step(response)
        ```
    """

    def __init__(
        self,
        tasks: list[Task],
        max_episode_steps: Optional[int] = None,
        max_episode_length: Optional[int] = None,
        reward_fn: Optional[Callable[[str, str], float]] = None,
        reward_on_success: float = 1.0,
        reward_on_failure: float = 0.0,
        partial_reward: bool = False,
        similarity_threshold: float = 0.8,
        normalize_text: bool = True,
        shuffle_tasks: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize text-based environment.

        Args:
            tasks: List of Task objects defining the environment tasks
            max_episode_steps: Maximum steps per episode (deprecated, use max_episode_length)
            max_episode_length: Maximum steps per episode
            reward_fn: Custom reward function (response, target) -> reward.
                      If None, uses exact match or partial reward based on partial_reward
            reward_on_success: Reward for successful task completion
            reward_on_failure: Reward for failed task completion
            partial_reward: Whether to give partial credit for similar answers
            similarity_threshold: Minimum similarity for partial credit (when partial_reward=True)
            normalize_text: Whether to normalize text (lowercase, strip)
            shuffle_tasks: Whether to shuffle tasks between episodes
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__(render_mode=render_mode)

        if not tasks:
            raise ValueError("Must provide at least one task")

        self.tasks = tasks
        # Keep a copy of the original task order for reproducible shuffling
        self._original_tasks = list(tasks)

        # Support both max_episode_steps and max_episode_length for compatibility
        if max_episode_length is not None:
            self.max_episode_length = max_episode_length
        elif max_episode_steps is not None:
            self.max_episode_length = max_episode_steps
        else:
            self.max_episode_length = 100

        # For backwards compatibility
        self.max_episode_steps = self.max_episode_length

        self.reward_on_success = reward_on_success
        self.reward_on_failure = reward_on_failure
        self.partial_reward = partial_reward
        self.similarity_threshold = similarity_threshold
        self.normalize_text = normalize_text
        self.shuffle_tasks = shuffle_tasks

        # Set reward function
        if reward_fn is None:
            if partial_reward:
                self.reward_fn = self._partial_reward_fn
            else:
                self.reward_fn = self._default_reward
        else:
            self.reward_fn = reward_fn

        # Define observation space as Dict
        # Note: Gymnasium doesn't have native Text space, so we use Dict
        self.observation_space = spaces.Dict(
            {
                "prompt": spaces.Text(max_length=1000),
                "target": spaces.Text(max_length=1000),
                "context": spaces.Text(max_length=2000),
                "history": spaces.Text(max_length=5000),
            }
        )

        # Action space: Text space for agent responses
        self.action_space = spaces.Text(max_length=1000)

        # Episode state
        self.current_task: Optional[Task] = None
        self.current_step = 0
        self.conversation_history: list[str] = []
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
            observation: Dict containing prompt, context, and history
            info: Additional information dictionary
        """
        # Call gym.Env.reset() to handle seeding, skipping MLXEnv.reset()
        # which raises NotImplementedError
        gym.Env.reset(self, seed=seed)

        # Seed MLX RNG for Metal-accelerated randomness
        if seed is not None:
            self._mlx_rng_key = mx.random.key(seed)

        # Shuffle tasks if enabled (restore original order first for reproducibility)
        if self.shuffle_tasks:
            import random
            # Restore original task order before shuffling for reproducibility
            self.tasks = list(self._original_tasks)
            if seed is not None:
                # Create a new Random instance to avoid global state
                rng = random.Random(seed)
                rng.shuffle(self.tasks)
            else:
                random.shuffle(self.tasks)

        # Select task
        if options and "task_index" in options:
            self.task_index = options["task_index"] % len(self.tasks)
        else:
            self.task_index = 0

        self.current_task = self.tasks[self.task_index]
        self.current_step = 0
        self.conversation_history = []
        self.episode_correct = 0
        self.episode_total = 0

        # Build observation
        observation = self._get_observation()

        info = {
            "task_index": self.task_index,
            "task_prompt": self.current_task.prompt,
            "task_target": self.current_task.target,
            "task_metadata": self.current_task.metadata,
        }

        return observation, info

    def step(
        self, action: str
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Text response from the agent (string)

        Returns:
            observation: Next observation dict
            reward: Reward for this step
            terminated: Whether episode ended naturally (task completed)
            truncated: Whether episode was truncated (max steps reached)
            info: Additional information
        """
        if self.current_task is None:
            raise RuntimeError("Must call reset() before step()")

        # Normalize text if enabled
        if self.normalize_text:
            action = self._normalize_text(action)
            target = self._normalize_text(self.current_task.target)
        else:
            target = self.current_task.target

        # Calculate raw reward from reward function (0.0 to 1.0)
        raw_reward = self.reward_fn(action, target)

        # Scale reward based on success/failure
        is_success = raw_reward >= 1.0
        if is_success:
            reward = self.reward_on_success
        elif raw_reward > 0.0:
            # Partial credit case - interpolate between failure and success
            reward = self.reward_on_failure + (self.reward_on_success - self.reward_on_failure) * raw_reward
        else:
            reward = self.reward_on_failure

        # Update statistics
        self.episode_total += 1
        if is_success:
            self.episode_correct += 1

        # Update conversation history
        self.conversation_history.append(f"Q: {self.current_task.prompt}")
        self.conversation_history.append(f"A: {action}")

        # Update step counter
        self.current_step += 1

        # Check termination conditions
        # For single-step episodes (max_episode_length=1), terminate rather than truncate
        if self.max_episode_length == 1:
            terminated = True
            truncated = False
        else:
            terminated = is_success
            truncated = self.current_step >= self.max_episode_steps

        # Move to next task if not done
        if not (terminated or truncated):
            self.task_index = (self.task_index + 1) % len(self.tasks)
            self.current_task = self.tasks[self.task_index]

        # Build next observation
        observation = self._get_observation()

        # Build info dict
        info = {
            "response": action,
            "target": target,
            "reward": reward,
            "success": is_success,
            "episode_correct": self.episode_correct,
            "episode_total": self.episode_total,
            "accuracy": (
                self.episode_correct / self.episode_total if self.episode_total > 0 else 0.0
            ),
        }

        # Add similarity if using partial reward
        if self.partial_reward and raw_reward < 1.0:
            info["similarity"] = raw_reward

        if terminated or truncated:
            info["final_accuracy"] = (
                self.episode_correct / self.episode_total if self.episode_total > 0 else 0.0
            )

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the current environment state.

        In 'human' mode, prints the current task and conversation history.
        """
        if self.render_mode == "human":
            print("\n" + "=" * 60)
            print(f"Step: {self.current_step}/{self.max_episode_steps}")
            print(f"Task {self.task_index + 1}/{len(self.tasks)}")
            print("=" * 60)

            if self.current_task:
                print(f"\nPrompt: {self.current_task.prompt}")
                if self.current_task.context:
                    print(f"Context: {self.current_task.context}")
                print(f"Target: {self.current_task.target}")

            if self.conversation_history:
                print("\nConversation History:")
                for line in self.conversation_history[-6:]:  # Show last 3 exchanges
                    print(f"  {line}")

            print(f"\nAccuracy: {self.episode_correct}/{self.episode_total}")
            print("=" * 60 + "\n")

    def _get_observation(self) -> dict[str, Any]:
        """
        Build observation dictionary.

        Returns:
            Dict containing prompt, target, context, and history
        """
        if self.current_task is None:
            return {"prompt": "", "target": "", "context": "", "history": ""}

        # Build history string from last N exchanges
        history_str = "\n".join(self.conversation_history[-10:])

        return {
            "prompt": self.current_task.prompt,
            "target": self.current_task.target,
            "context": self.current_task.context,
            "history": history_str,
        }

    def _default_reward(self, response: str, target: str) -> float:
        """
        Default reward function: exact match.

        Args:
            response: Agent's response
            target: Target response

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if self.normalize_text:
            response = self._normalize_text(response)
            target = self._normalize_text(target)

        return 1.0 if response == target else 0.0

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text (lowercase and strip whitespace).

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        return text.lower().strip()

    def _partial_reward_fn(self, response: str, target: str) -> float:
        """
        Reward function with partial credit based on string similarity.

        Args:
            response: Agent's response
            target: Target response

        Returns:
            Reward in [0, 1] based on similarity
        """
        if self.normalize_text:
            response = self._normalize_text(response)
            target = self._normalize_text(target)

        # Exact match
        if response == target:
            return 1.0

        # Substring match
        if target in response or response in target:
            longer = max(len(target), len(response))
            shorter = min(len(target), len(response))
            similarity = shorter / longer if longer > 0 else 0.0
            if similarity >= self.similarity_threshold:
                return 0.9
            return 0.7

        # Word overlap (Jaccard similarity)
        response_words = set(response.split())
        target_words = set(target.split())
        if response_words and target_words:
            overlap = len(response_words & target_words)
            total = len(response_words | target_words)
            if total > 0:
                jaccard = overlap / total
                if jaccard >= self.similarity_threshold:
                    return jaccard
                elif jaccard > 0.3:
                    return jaccard * 0.6  # Reduced partial credit

        return 0.0

    def close(self):
        """Clean up environment resources."""
        pass


class QAEnv(TextWorldEnv):
    """
    Question-answering environment.

    Specialized version of TextWorldEnv for QA tasks. Provides
    utilities for loading QA datasets and scoring answers.

    Example:
        ```python
        # Create QA environment
        qa_tasks = [
            Task(prompt="What is the capital of France?", target="Paris"),
            Task(prompt="What is 10 * 5?", target="50"),
        ]

        env = QAEnv(tasks=qa_tasks, partial_credit=True)

        # Use with agent
        obs, info = env.reset()
        response = "paris"  # Case doesn't matter with normalize_text=True
        next_obs, reward, terminated, truncated, info = env.step(response)
        print(reward)  # 1.0 (exact match after normalization)
        ```
    """

    def __init__(
        self,
        tasks: Optional[list[Task]] = None,
        qa_pairs: Optional[list[dict[str, str]]] = None,
        max_episode_steps: int = 100,
        partial_credit: bool = False,
        **kwargs,
    ):
        """
        Initialize QA environment.

        Args:
            tasks: List of QA tasks (either tasks or qa_pairs must be provided)
            qa_pairs: List of dicts with 'question' and 'answer' keys (alternative to tasks)
            max_episode_steps: Maximum steps per episode
            partial_credit: Whether to give partial credit for similar answers
            **kwargs: Additional arguments passed to TextWorldEnv
        """
        # Convert qa_pairs to tasks if provided
        if qa_pairs is not None:
            tasks = [
                Task(prompt=pair["question"], target=pair["answer"])
                for pair in qa_pairs
            ]
        elif tasks is None:
            raise ValueError("Either tasks or qa_pairs must be provided")

        # Set custom reward function if partial credit enabled
        if partial_credit:
            reward_fn = self._partial_credit_reward
        else:
            reward_fn = None

        super().__init__(
            tasks=tasks, max_episode_steps=max_episode_steps, reward_fn=reward_fn, **kwargs
        )

    def _partial_credit_reward(self, response: str, target: str) -> float:
        """
        Reward function with partial credit based on string similarity.

        Args:
            response: Agent's response
            target: Target response

        Returns:
            Reward in [0, 1] based on similarity
        """
        if self.normalize_text:
            response = self._normalize_text(response)
            target = self._normalize_text(target)

        # Exact match
        if response == target:
            return 1.0

        # Substring match
        if target in response or response in target:
            return 0.7

        # Word overlap
        response_words = set(response.split())
        target_words = set(target.split())
        if response_words and target_words:
            overlap = len(response_words & target_words)
            total = len(response_words | target_words)
            if total > 0:
                return 0.5 * (overlap / total)

        return 0.0


class InstructionFollowingEnv(TextWorldEnv):
    """
    Instruction following environment.

    Specialized environment for instruction following tasks where the agent
    must execute a sequence of instructions and generate appropriate responses.

    Example:
        ```python
        # Create instruction following environment
        tasks = [
            Task(
                prompt="Sort these numbers: 3, 1, 4, 1, 5",
                target="1, 1, 3, 4, 5",
                context="Sort in ascending order"
            ),
            Task(
                prompt="Reverse this string: hello",
                target="olleh"
            ),
        ]

        env = InstructionFollowingEnv(tasks=tasks)

        # Use with agent
        obs, info = env.reset()
        print(obs['prompt'])  # "Sort these numbers: 3, 1, 4, 1, 5"
        print(obs['context'])  # "Sort in ascending order"
        ```
    """

    def __init__(
        self,
        tasks: Optional[list[Task]] = None,
        instructions: Optional[list[dict[str, Any]]] = None,
        max_episode_steps: int = 50,
        strict_format: bool = False,
        **kwargs,
    ):
        """
        Initialize instruction following environment.

        Args:
            tasks: List of instruction tasks (either tasks or instructions must be provided)
            instructions: List of dicts with 'instruction', 'expected_output', and optional 'reward' keys
            max_episode_steps: Maximum steps per episode
            strict_format: Whether to enforce strict format matching
            **kwargs: Additional arguments passed to TextWorldEnv
        """
        # Convert instructions to tasks if provided
        if instructions is not None:
            tasks = [
                Task(
                    prompt=inst["instruction"],
                    target=inst["expected_output"],
                    reward=inst.get("reward", 1.0)
                )
                for inst in instructions
            ]
        elif tasks is None:
            raise ValueError("Either tasks or instructions must be provided")

        self.strict_format = strict_format

        super().__init__(tasks=tasks, max_episode_steps=max_episode_steps, **kwargs)

    def _default_reward(self, response: str, target: str) -> float:
        """
        Custom reward for instruction following.

        Args:
            response: Agent's response
            target: Target response

        Returns:
            Reward based on instruction completion
        """
        if self.normalize_text and not self.strict_format:
            response = self._normalize_text(response)
            target = self._normalize_text(target)

        # Exact match
        if response == target:
            return 1.0

        # Partial credit for containing the answer
        if not self.strict_format and target in response:
            return 0.8

        return 0.0


def create_text_env(
    task_type: str,
    tasks: list[Task],
    **kwargs,
) -> TextWorldEnv:
    """
    Factory function for creating text environments.

    Args:
        task_type: Type of environment ('text', 'qa', 'instruction', 'generic')
        tasks: List of tasks
        **kwargs: Additional arguments passed to environment constructor

    Returns:
        Text environment instance

    Raises:
        ValueError: If task_type is not recognized or tasks list is empty

    Example:
        ```python
        tasks = [Task(prompt="Hello", target="Hi")]

        # Create QA environment
        env = create_text_env('qa', tasks, partial_credit=True)

        # Create instruction following environment
        env = create_text_env('instruction', tasks, strict_format=False)
        ```
    """
    # Validate tasks list
    if not tasks:
        raise ValueError("tasks list cannot be empty")

    if task_type == "text" or task_type == "generic":
        return TextWorldEnv(tasks=tasks, **kwargs)
    elif task_type == "qa":
        return QAEnv(tasks=tasks, **kwargs)
    elif task_type == "instruction":
        return InstructionFollowingEnv(tasks=tasks, **kwargs)
    else:
        raise ValueError(
            f"Unknown environment type: {task_type}. "
            f"Supported types: 'text', 'generic', 'qa', 'instruction'"
        )
