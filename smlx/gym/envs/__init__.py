"""
Custom environments for SMLX Gym.

This module provides custom environment implementations for:
- Text-based tasks (QA, instruction following)
- Vision-based tasks (image classification, VQA)
- Audio-based tasks (speech recognition, audio classification)

All environments use MLX arrays for Metal GPU acceleration.

Example:
    ```python
    from smlx.gym.envs import TextWorldEnv, Task

    # Create text environment
    tasks = [
        Task(prompt="What is 2+2?", target="4"),
        Task(prompt="What color is sky?", target="blue"),
    ]
    env = TextWorldEnv(tasks=tasks)

    # Train agent
    obs, info = env.reset()
    ```
"""

# Text environments
# Audio environments
from smlx.gym.envs.audio_task import (
    AudioClassificationEnv,
    AudioTask,
    AudioTaskEnv,
    SpeechRecognitionEnv,
    create_audio_env,
)
from smlx.gym.envs.text_world import (
    InstructionFollowingEnv,
    QAEnv,
    Task,
    TextWorldEnv,
    create_text_env,
)

# Vision environments
from smlx.gym.envs.vision_task import (
    ImageClassificationEnv,
    VisionTask,
    VisionTaskEnv,
    VisualQAEnv,
    create_vision_env,
)

__all__ = [
    # Text
    "TextWorldEnv",
    "Task",
    "QAEnv",
    "InstructionFollowingEnv",
    "create_text_env",
    # Vision
    "VisionTaskEnv",
    "VisionTask",
    "ImageClassificationEnv",
    "VisualQAEnv",
    "create_vision_env",
    # Audio
    "AudioTaskEnv",
    "AudioTask",
    "SpeechRecognitionEnv",
    "AudioClassificationEnv",
    "create_audio_env",
]
