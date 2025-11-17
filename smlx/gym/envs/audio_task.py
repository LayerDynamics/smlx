"""
Audio-based environment for speech and audio RL tasks.

This module provides environments for audio-based reinforcement learning tasks
such as speech recognition, audio classification, speaker identification, and
other audio understanding tasks. All observations are MLX arrays for Metal GPU
acceleration.

Reference: SMLX_Gym.md, Section 4.1 (Environment Implementations)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import mlx.core as mx
import numpy as np

from smlx.gym.base import MLXEnv
from smlx.models.Whisper_tiny.audio import (
    SAMPLE_RATE,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
)

# Suppress unused import warning - field is used in dataclass
_ = field


@dataclass
class AudioTask:
    """
    Represents an audio-based task.

    Attributes:
        audio: Audio source (path or audio array)
        target: Target transcription, label, or answer
        prompt: Optional text prompt for audio QA tasks
        metadata: Additional task metadata
    """

    audio: Union[str, Path, mx.array]
    target: Any
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AudioTaskEnv(MLXEnv):
    """
    Audio-based environment for speech and audio understanding tasks.

    This environment provides a flexible interface for audio-based RL tasks
    including speech recognition, audio classification, and audio question
    answering. All observations are returned as MLX arrays for Metal GPU
    acceleration throughout the training pipeline.

    The environment can be configured with:
    - Custom audio task datasets
    - Flexible reward functions
    - Audio preprocessing pipelines (mel spectrograms)
    - Episode length limits

    Observation Space:
        Dict space containing:
        - 'audio': MLX array of raw audio waveform [T] or mel spectrogram [T, F]
        - 'mel_spectrogram': MLX array of log-mel spectrogram [T, n_mels]
        - 'prompt': Text prompt (for audio QA tasks)
        - 'duration': Audio duration in seconds

    Action Space:
        Discrete space for classification tasks, or custom for other tasks

    Rewards:
        - Default: Exact match with target transcription/label
        - Custom: User-provided reward function

    Example:
        ```python
        from smlx.gym.envs.audio_task import AudioTask, AudioTaskEnv

        # Create audio tasks
        tasks = [
            AudioTask(audio="hello.wav", target="hello world"),
            AudioTask(audio="goodbye.wav", target="goodbye"),
        ]

        env = AudioTaskEnv(
            tasks=tasks,
            use_mel_spectrogram=True,
            max_episode_steps=10
        )

        # Use with audio model
        obs, info = env.reset()
        mel_spec = obs['mel_spectrogram']  # MLX array [T, n_mels]
        print(f"Mel spectrogram shape: {mel_spec.shape}")

        # Agent transcribes audio
        transcription = "hello world"
        next_obs, reward, terminated, truncated, info = env.step(transcription)
        ```

    Integration with Whisper:
        ```python
        from smlx.models.Whisper_tiny import load, transcribe

        # Load Whisper model
        model, tokenizer = load("mlx-community/whisper-tiny")

        env = AudioTaskEnv(tasks=audio_tasks)
        obs, info = env.reset()

        # Transcribe audio
        audio = obs['audio']
        transcription = transcribe(model, tokenizer, audio)

        # Take action
        next_obs, reward, terminated, truncated, info = env.step(transcription)
        ```
    """

    def __init__(
        self,
        tasks: list[AudioTask],
        use_mel_spectrogram: bool = True,
        n_mels: int = 80,
        sample_rate: int = SAMPLE_RATE,
        max_audio_length: int = 30,  # seconds
        normalize_text: bool = True,
        max_episode_steps: int = 100,
        reward_fn: Optional[Callable[[Any, Any], float]] = None,
        shuffle_tasks: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize audio-based environment.

        Args:
            tasks: List of AudioTask objects defining the environment tasks
            use_mel_spectrogram: Whether to compute mel spectrograms (default: True)
            n_mels: Number of mel frequency bins (default: 80)
            sample_rate: Audio sample rate in Hz (default: 16000)
            max_audio_length: Maximum audio length in seconds (default: 30)
            normalize_text: Whether to normalize text targets (default: True)
            max_episode_steps: Maximum steps per episode
            reward_fn: Custom reward function (prediction, target) -> reward
            shuffle_tasks: Whether to shuffle tasks between episodes
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__(render_mode=render_mode)

        if not tasks:
            raise ValueError("Must provide at least one audio task")

        self.tasks = tasks
        self.use_mel_spectrogram = use_mel_spectrogram
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.max_samples = int(max_audio_length * sample_rate)
        self.normalize_text = normalize_text
        self.max_episode_steps = max_episode_steps
        self.shuffle_tasks = shuffle_tasks

        # Set reward function
        if reward_fn is None:
            self.reward_fn = self._default_reward
        else:
            self.reward_fn = reward_fn

        # Define observation space
        # Audio observations are MLX arrays
        max_frames = 3000  # For 30 seconds at 16kHz with standard Whisper settings

        self.observation_space = gym.spaces.Dict(
            {
                "audio": gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.max_samples,),
                    dtype=np.float32,
                ),
                "mel_spectrogram": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(max_frames, n_mels),
                    dtype=np.float32,
                ),
                "prompt": gym.spaces.Text(max_length=500),
                "duration": gym.spaces.Box(
                    low=0.0, high=float(max_audio_length), shape=(1,), dtype=np.float32
                ),
            }
        )

        # Action space: Discrete for classification (can be overridden)
        self.action_space = gym.spaces.Discrete(1000)

        # Episode state
        self.current_task: Optional[AudioTask] = None
        self.current_audio: Optional[mx.array] = None
        self.current_mel: Optional[mx.array] = None
        self.original_audio_length: int = 0  # Track original length before padding
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
            observation: Dict containing audio arrays, mel spectrogram, and metadata
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

        # Load and preprocess audio using MLX
        self._load_current_audio()

        # Build observation
        observation = self._get_observation()

        info = {
            "task_index": self.task_index,
            "task_target": self.current_task.target,
            "task_prompt": self.current_task.prompt,
            "task_metadata": self.current_task.metadata,
            "audio_source": str(self.current_task.audio),
        }

        return observation, info

    def step(
        self, action: Union[str, int]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Predicted transcription (str) or class label (int) from agent

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
            self._load_current_audio()

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

        In 'human' mode, displays the current task information.
        """
        if self.render_mode == "human":
            print("\n" + "=" * 60)
            print(f"Step: {self.current_step}/{self.max_episode_steps}")
            print(f"Task {self.task_index + 1}/{len(self.tasks)}")
            print("=" * 60)

            if self.current_task:
                print(f"\nPrompt: {self.current_task.prompt}")
                print(f"Target: {self.current_task.target}")
                print(f"Audio: {self.current_task.audio}")

                if self.current_audio is not None:
                    duration = self.original_audio_length / self.sample_rate
                    print(f"Duration: {duration:.2f} seconds")

            print(f"\nAccuracy: {self.episode_correct}/{self.episode_total}")
            print("=" * 60 + "\n")

    def _load_current_audio(self):
        """
        Load and preprocess the current task's audio using MLX.

        Converts the audio to MLX arrays for Metal GPU acceleration.
        """
        if self.current_task is None:
            return

        # Load audio using Whisper utilities
        if isinstance(self.current_task.audio, mx.array):
            audio = self.current_task.audio
        else:
            audio = load_audio(self.current_task.audio, sr=self.sample_rate)

        # Ensure it's an MLX array for Metal GPU operations
        if not isinstance(audio, mx.array):
            audio = mx.array(audio)

        # Track original length before padding
        self.original_audio_length = len(audio)

        # Pad or trim to max length using MLX operations
        self.current_audio = pad_or_trim(audio, length=self.max_samples)

        # Compute mel spectrogram if enabled
        if self.use_mel_spectrogram:
            # Use Whisper's log_mel_spectrogram function
            self.current_mel = log_mel_spectrogram(audio, n_mels=self.n_mels)

            # Ensure it's an MLX array
            if not isinstance(self.current_mel, mx.array):
                self.current_mel = mx.array(self.current_mel)
        else:
            # Create empty mel spectrogram using MLX
            max_frames = 3000
            self.current_mel = mx.zeros((max_frames, self.n_mels))

    def _get_observation(self) -> dict[str, Any]:
        """
        Build observation dictionary with MLX arrays.

        Returns:
            Dict containing audio, mel spectrogram, prompt, and duration
        """
        if self.current_task is None or self.current_audio is None:
            # Return empty observation using MLX arrays
            return {
                "audio": mx.zeros((self.max_samples,)),
                "mel_spectrogram": mx.zeros((3000, self.n_mels)),
                "prompt": "",
                "duration": mx.array([0.0]),
            }

        # Calculate duration using original audio length (before padding)
        duration_seconds = float(self.original_audio_length) / self.sample_rate
        duration = mx.array([duration_seconds])

        return {
            "audio": self.current_audio,  # MLX array [T]
            "mel_spectrogram": self.current_mel,  # MLX array [T, n_mels]
            "prompt": self.current_task.prompt,
            "duration": duration,  # MLX array [1]
        }

    def _default_reward(self, prediction: Union[str, int], target: Union[str, int]) -> float:
        """
        Default reward function: exact match.

        Uses MLX operations for numerical comparisons when applicable.

        Args:
            prediction: Agent's prediction (text or label)
            target: Target transcription or label

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Handle text predictions
        if isinstance(prediction, str) and isinstance(target, str):
            if self.normalize_text:
                prediction = self._normalize_text(prediction)
                target = self._normalize_text(target)
            return 1.0 if prediction == target else 0.0

        # Handle numerical predictions using MLX
        if isinstance(prediction, (int, float)) and isinstance(target, (int, float)):
            pred_mx = mx.array(prediction)
            target_mx = mx.array(target)
            match = mx.array_equal(pred_mx, target_mx)
            return 1.0 if match else 0.0

        return 0.0

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text (lowercase and strip whitespace).

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        return text.lower().strip()

    def close(self):
        """Clean up environment resources."""
        self.current_audio = None
        self.current_mel = None


class SpeechRecognitionEnv(AudioTaskEnv):
    """
    Speech recognition environment.

    Specialized environment for automatic speech recognition tasks.
    Provides utilities for evaluating transcriptions with metrics like
    Word Error Rate (WER).

    Example:
        ```python
        from smlx.gym.envs.audio_task import AudioTask, SpeechRecognitionEnv

        # Create speech recognition tasks
        tasks = [
            AudioTask(audio="speech1.wav", target="hello world"),
            AudioTask(audio="speech2.wav", target="how are you"),
        ]

        env = SpeechRecognitionEnv(
            tasks=tasks,
            partial_credit=True,
            case_sensitive=False
        )

        # Use with Whisper
        obs, info = env.reset()
        audio = obs['audio']  # MLX array

        # Transcribe using Whisper model
        transcription = whisper_model.transcribe(audio)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(transcription)
        ```
    """

    def __init__(
        self,
        tasks: list[AudioTask],
        partial_credit: bool = False,
        case_sensitive: bool = False,
        **kwargs,
    ):
        """
        Initialize speech recognition environment.

        Args:
            tasks: List of speech tasks with audio and target transcription
            partial_credit: Whether to give partial credit for similar transcriptions
            case_sensitive: Whether transcription matching is case-sensitive
            **kwargs: Additional arguments passed to AudioTaskEnv
        """
        # Set custom reward function
        if partial_credit:
            reward_fn = self._partial_credit_reward
        else:
            reward_fn = None

        super().__init__(tasks=tasks, reward_fn=reward_fn, normalize_text=not case_sensitive, **kwargs)

        self.case_sensitive = case_sensitive
        self.partial_credit = partial_credit

    def _partial_credit_reward(self, transcription: str, target: str) -> float:
        """
        Partial credit reward based on word overlap.

        Args:
            transcription: Generated transcription
            target: Target transcription

        Returns:
            Reward in [0, 1] based on word overlap
        """
        if not self.case_sensitive:
            transcription = self._normalize_text(transcription)
            target = self._normalize_text(target)

        # Exact match
        if transcription == target:
            return 1.0

        # Word-level overlap
        trans_words = transcription.split()
        target_words = target.split()

        if not trans_words or not target_words:
            return 0.0

        # Compute word overlap
        trans_set = set(trans_words)
        target_set = set(target_words)
        overlap = len(trans_set & target_set)
        total = len(target_set)

        return overlap / total if total > 0 else 0.0


class AudioClassificationEnv(AudioTaskEnv):
    """
    Audio classification environment.

    Environment for audio classification tasks such as:
    - Speaker identification
    - Sound event classification
    - Music genre classification
    - Emotion recognition from speech

    Example:
        ```python
        from smlx.gym.envs.audio_task import AudioTask, AudioClassificationEnv

        # Create audio classification tasks
        tasks = [
            AudioTask(audio="dog_bark.wav", target=0),  # dog
            AudioTask(audio="cat_meow.wav", target=1),  # cat
            AudioTask(audio="bird_chirp.wav", target=2),  # bird
        ]

        env = AudioClassificationEnv(
            tasks=tasks,
            num_classes=3,
            class_names=["dog", "cat", "bird"]
        )

        # Use with audio classifier
        obs, info = env.reset()
        mel_spec = obs['mel_spectrogram']  # MLX array

        # Classify audio using MLX operations
        import mlx.core as mx
        logits = audio_classifier(mx.expand_dims(mel_spec, 0))
        predicted_class = int(mx.argmax(logits, axis=-1)[0])

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(predicted_class)
        ```
    """

    def __init__(
        self,
        tasks: list[AudioTask],
        num_classes: int,
        class_names: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Initialize audio classification environment.

        Args:
            tasks: List of audio tasks with audio and target class
            num_classes: Number of classes
            class_names: Optional list of class names for logging
            **kwargs: Additional arguments passed to AudioTaskEnv
        """
        super().__init__(tasks=tasks, **kwargs)

        self.num_classes = num_classes
        self.class_names = class_names

        # Update action space for classification
        self.action_space = gym.spaces.Discrete(num_classes)

        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(
                f"Number of class names ({len(class_names)}) must match "
                f"num_classes ({num_classes})"
            )

    def _default_reward(self, prediction: int, target: int) -> float:
        """
        Classification reward.

        Uses MLX operations for GPU-accelerated comparison.

        Args:
            prediction: Predicted class
            target: Target class

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        pred_mx = mx.array(prediction)
        target_mx = mx.array(target)
        return 1.0 if mx.array_equal(pred_mx, target_mx) else 0.0


def create_audio_env(
    task_type: str,
    tasks: list[AudioTask],
    **kwargs,
) -> AudioTaskEnv:
    """
    Factory function for creating audio environments.

    All returned environments use MLX arrays for Metal GPU acceleration.

    Args:
        task_type: Type of environment ('audio', 'speech', 'classification')
        tasks: List of audio tasks
        **kwargs: Additional arguments passed to environment constructor

    Returns:
        Audio environment instance with MLX support

    Raises:
        ValueError: If task_type is not recognized

    Example:
        ```python
        from smlx.gym.envs.audio_task import AudioTask, create_audio_env

        # Create speech recognition environment
        tasks = [AudioTask(audio="speech.wav", target="hello world")]
        env = create_audio_env('speech', tasks, partial_credit=True)

        # Create audio classification environment
        class_tasks = [AudioTask(audio="sound.wav", target=0)]
        env = create_audio_env('classification', class_tasks, num_classes=10)
        ```
    """
    if task_type == "audio":
        return AudioTaskEnv(tasks=tasks, **kwargs)
    elif task_type == "speech":
        return SpeechRecognitionEnv(tasks=tasks, **kwargs)
    elif task_type == "classification":
        return AudioClassificationEnv(tasks=tasks, **kwargs)
    else:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Supported types: 'audio', 'speech', 'classification'"
        )
