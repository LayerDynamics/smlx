"""
Unit tests for audio-based RL environments.

Tests AudioTaskEnv, SpeechRecognitionEnv, and AudioClassificationEnv.
"""

import mlx.core as mx
import numpy as np
import pytest

from smlx.gym.envs.audio_task import (
    AudioClassificationEnv,
    AudioTask,
    AudioTaskEnv,
    SpeechRecognitionEnv,
    create_audio_env,
)


@pytest.mark.unit
class TestAudioTask:
    """Tests for AudioTask dataclass."""

    def test_basic_task(self):
        """Test creating basic audio task."""
        task = AudioTask(audio="test.wav", target="hello world")

        assert task.audio == "test.wav"
        assert task.target == "hello world"
        assert task.prompt == ""
        assert task.metadata == {}

    def test_task_with_prompt(self):
        """Test audio task with prompt."""
        task = AudioTask(
            audio="test.wav",
            target="hello",
            prompt="Transcribe the audio",
            metadata={"speaker": "john"},
        )

        assert task.prompt == "Transcribe the audio"
        assert task.metadata["speaker"] == "john"

    def test_task_with_mlx_audio(self):
        """Test audio task with MLX array audio."""
        audio_array = mx.zeros((16000,))  # 1 second at 16kHz
        task = AudioTask(audio=audio_array, target="test")

        assert isinstance(task.audio, mx.array)
        assert task.audio.shape == (16000,)


@pytest.mark.unit
class TestAudioTaskEnv:
    """Tests for AudioTaskEnv base class."""

    @pytest.fixture
    def tasks(self):
        """Create test audio tasks with MLX arrays."""
        # Use MLX arrays directly to avoid file I/O
        return [
            AudioTask(audio=mx.zeros((16000,)), target="hello"),
            AudioTask(audio=mx.ones((16000,)), target="world"),
            AudioTask(audio=mx.full((16000,), 0.5), target="test"),
        ]

    @pytest.fixture
    def env(self, tasks):
        """Create audio task environment."""
        return AudioTaskEnv(
            tasks=tasks, use_mel_spectrogram=True, max_episode_steps=10, shuffle_tasks=False
        )

    def test_initialization(self, env, tasks):
        """Test environment initialization."""
        assert env.tasks == tasks
        assert env.use_mel_spectrogram is True
        assert env.n_mels == 80
        assert env.sample_rate == 16000
        assert env.max_episode_steps == 10

    def test_empty_tasks_raises_error(self):
        """Test that empty task list raises error."""
        with pytest.raises(ValueError, match="at least one audio task"):
            AudioTaskEnv(tasks=[])

    def test_observation_space(self, env):
        """Test observation space definition."""
        obs_space = env.observation_space

        assert "audio" in obs_space.spaces
        assert "mel_spectrogram" in obs_space.spaces
        assert "prompt" in obs_space.spaces
        assert "duration" in obs_space.spaces

    def test_action_space(self, env):
        """Test action space definition."""
        assert env.action_space is not None

    def test_reset(self, env):
        """Test environment reset."""
        observation, info = env.reset()

        # Check observation structure
        assert isinstance(observation, dict)
        assert "audio" in observation
        assert "mel_spectrogram" in observation
        assert "prompt" in observation
        assert "duration" in observation

        # Check MLX arrays
        assert isinstance(observation["audio"], mx.array)
        assert isinstance(observation["mel_spectrogram"], mx.array)
        assert isinstance(observation["duration"], mx.array)

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

    def test_step_with_correct_answer(self, env):
        """Test step with correct answer."""
        env.reset()

        # Get target from current task
        target = env.current_task.target

        # Step with correct answer
        observation, reward, terminated, truncated, info = env.step(target)

        assert reward == 1.0
        assert terminated is True
        assert info["correct"] is True

    def test_step_with_incorrect_answer(self, env):
        """Test step with incorrect answer."""
        env.reset()

        # Step with wrong answer
        observation, reward, terminated, truncated, info = env.step("wrong answer")

        assert reward == 0.0
        assert terminated is False
        assert info["correct"] is False

    def test_step_max_episodes(self, env):
        """Test truncation at max episode steps."""
        env.reset()

        for step in range(env.max_episode_steps + 1):
            observation, reward, terminated, truncated, info = env.step("wrong")

            if truncated:
                assert step + 1 >= env.max_episode_steps
                break

    def test_mel_spectrogram_generation(self, env):
        """Test that mel spectrograms are generated."""
        observation, info = env.reset()

        mel_spec = observation["mel_spectrogram"]
        assert isinstance(mel_spec, mx.array)
        assert mel_spec.ndim == 2
        assert mel_spec.shape[1] == env.n_mels

    def test_audio_is_mlx_array(self, env):
        """Test that audio observations are MLX arrays."""
        observation, info = env.reset()

        audio = observation["audio"]
        assert isinstance(audio, mx.array)
        assert audio.shape[0] == env.max_samples

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
        tasks = [AudioTask(audio=mx.zeros((16000,)), target="test")]
        env = AudioTaskEnv(tasks=tasks, render_mode="human")

        env.reset()
        # Should not raise an error
        env.render()

    def test_custom_reward_function(self):
        """Test custom reward function."""

        def custom_reward(prediction, target):
            return 0.5 if prediction == target else 0.0

        tasks = [AudioTask(audio=mx.zeros((16000,)), target="test")]
        env = AudioTaskEnv(tasks=tasks, reward_fn=custom_reward)

        env.reset()
        observation, reward, terminated, truncated, info = env.step("test")

        assert reward == 0.5

    def test_normalize_text(self, env):
        """Test text normalization."""
        env.reset()

        # Normalized comparison (case insensitive)
        env.normalize_text = True
        target = env.current_task.target.lower()

        observation, reward, terminated, truncated, info = env.step(target.upper())

        # Should match despite case difference
        assert reward == 1.0


@pytest.mark.unit
class TestSpeechRecognitionEnv:
    """Tests for SpeechRecognitionEnv."""

    @pytest.fixture
    def tasks(self):
        """Create speech recognition tasks."""
        return [
            AudioTask(audio=mx.zeros((16000,)), target="hello world"),
            AudioTask(audio=mx.ones((16000,)), target="how are you"),
        ]

    @pytest.fixture
    def env(self, tasks):
        """Create speech recognition environment."""
        return SpeechRecognitionEnv(tasks=tasks, partial_credit=False, case_sensitive=False)

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

    def test_partial_credit(self):
        """Test partial credit reward."""
        tasks = [AudioTask(audio=mx.zeros((16000,)), target="hello world")]
        env = SpeechRecognitionEnv(tasks=tasks, partial_credit=True)

        env.reset()

        # Partial match: only "hello"
        observation, reward, terminated, truncated, info = env.step("hello")

        # Should get partial credit
        assert 0.0 < reward < 1.0

    def test_partial_credit_word_overlap(self):
        """Test partial credit based on word overlap."""
        tasks = [AudioTask(audio=mx.zeros((16000,)), target="the quick brown fox")]
        env = SpeechRecognitionEnv(tasks=tasks, partial_credit=True)

        env.reset()

        # 2 out of 4 words correct
        observation, reward, terminated, truncated, info = env.step("the quick")

        # Should get some credit
        assert reward > 0.0

    def test_case_sensitive_mode(self):
        """Test case sensitive matching."""
        tasks = [AudioTask(audio=mx.zeros((16000,)), target="Hello")]
        env = SpeechRecognitionEnv(tasks=tasks, case_sensitive=True)

        env.reset()

        # Wrong case should not match
        observation, reward, terminated, truncated, info = env.step("hello")

        assert reward == 0.0


@pytest.mark.unit
class TestAudioClassificationEnv:
    """Tests for AudioClassificationEnv."""

    @pytest.fixture
    def tasks(self):
        """Create audio classification tasks."""
        return [
            AudioTask(audio=mx.zeros((16000,)), target=0),  # Class 0
            AudioTask(audio=mx.ones((16000,)), target=1),  # Class 1
            AudioTask(audio=mx.full((16000,), 0.5), target=2),  # Class 2
        ]

    @pytest.fixture
    def env(self, tasks):
        """Create audio classification environment."""
        return AudioClassificationEnv(
            tasks=tasks, num_classes=3, class_names=["dog", "cat", "bird"]
        )

    def test_initialization(self, env):
        """Test environment initialization."""
        assert env.num_classes == 3
        assert env.class_names == ["dog", "cat", "bird"]

    def test_action_space(self, env):
        """Test action space for classification."""
        assert env.action_space.n == 3

    def test_class_names_validation(self):
        """Test that class names must match num_classes."""
        tasks = [AudioTask(audio=mx.zeros((16000,)), target=0)]

        with pytest.raises(ValueError, match="Number of class names"):
            AudioClassificationEnv(tasks=tasks, num_classes=3, class_names=["a", "b"])

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

    def test_classification_with_mlx_operations(self, env):
        """Test classification using MLX operations."""
        observation, info = env.reset()

        # Simulate model prediction with MLX
        mel_spec = observation["mel_spectrogram"]
        assert isinstance(mel_spec, mx.array)

        # Simulate logits
        logits = mx.array([0.1, 0.8, 0.1])
        predicted_class = int(mx.argmax(logits))

        observation, reward, terminated, truncated, info = env.step(predicted_class)

        # Reward depends on whether prediction matches target
        assert isinstance(reward, float)


@pytest.mark.unit
class TestCreateAudioEnv:
    """Tests for create_audio_env factory function."""

    def test_create_basic_audio_env(self):
        """Test creating basic audio environment."""
        tasks = [AudioTask(audio=mx.zeros((16000,)), target="test")]
        env = create_audio_env("audio", tasks)

        assert isinstance(env, AudioTaskEnv)
        env.close()

    def test_create_speech_env(self):
        """Test creating speech recognition environment."""
        tasks = [AudioTask(audio=mx.zeros((16000,)), target="hello")]
        env = create_audio_env("speech", tasks, partial_credit=True)

        assert isinstance(env, SpeechRecognitionEnv)
        assert env.partial_credit is True
        env.close()

    def test_create_classification_env(self):
        """Test creating classification environment."""
        tasks = [AudioTask(audio=mx.zeros((16000,)), target=0)]
        env = create_audio_env("classification", tasks, num_classes=10)

        assert isinstance(env, AudioClassificationEnv)
        assert env.num_classes == 10
        env.close()

    def test_invalid_task_type(self):
        """Test that invalid task type raises error."""
        tasks = [AudioTask(audio=mx.zeros((16000,)), target="test")]

        with pytest.raises(ValueError, match="Unknown task_type"):
            create_audio_env("invalid_type", tasks)


@pytest.mark.integration
class TestAudioTaskIntegration:
    """Integration tests for audio task environments."""

    def test_complete_speech_episode(self):
        """Test complete speech recognition episode."""
        tasks = [
            AudioTask(audio=mx.zeros((16000,)), target="hello"),
            AudioTask(audio=mx.ones((16000,)), target="world"),
        ]

        env = SpeechRecognitionEnv(tasks=tasks, max_episode_steps=5)

        observation, info = env.reset()

        for step in range(5):
            # Random transcription
            transcription = "hello" if step % 2 == 0 else "world"
            observation, reward, terminated, truncated, info = env.step(transcription)

            if terminated or truncated:
                break

        assert "final_accuracy" in info or "accuracy" in info

    def test_complete_classification_episode(self):
        """Test complete classification episode."""
        tasks = [
            AudioTask(audio=mx.zeros((16000,)), target=i % 3) for i in range(10)
        ]

        env = AudioClassificationEnv(tasks=tasks, num_classes=3, max_episode_steps=15)

        observation, info = env.reset()

        for _ in range(15):
            # Random classification
            action = np.random.randint(0, 3)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Should have tracked accuracy
        assert "accuracy" in info

    def test_mel_spectrogram_with_real_audio(self):
        """Test mel spectrogram generation with realistic audio."""
        # Create synthetic audio signal
        t = mx.linspace(0, 1, 16000)
        frequency = 440.0  # A4 note
        audio = mx.sin(2 * mx.pi * frequency * t)

        tasks = [AudioTask(audio=audio, target="A4")]
        env = AudioTaskEnv(tasks=tasks, use_mel_spectrogram=True)

        observation, info = env.reset()

        # Mel spectrogram should be generated
        mel_spec = observation["mel_spectrogram"]
        assert mel_spec.shape[1] == 80  # n_mels
        assert mel_spec.shape[0] > 0  # time frames

    def test_multi_task_switching(self):
        """Test switching between multiple tasks."""
        tasks = [
            AudioTask(audio=mx.zeros((16000,)), target=f"task_{i}") for i in range(5)
        ]

        env = AudioTaskEnv(tasks=tasks, max_episode_steps=10, shuffle_tasks=False)

        # Track which tasks we see
        seen_targets = set()

        for _ in range(10):
            observation, info = env.reset()
            seen_targets.add(info["task_target"])

            # Take wrong action to move to next task
            observation, reward, terminated, truncated, info = env.step("wrong")

        # Should have seen multiple tasks
        assert len(seen_targets) > 1

    def test_audio_duration_tracking(self):
        """Test audio duration tracking in observations."""
        # Different length audios
        tasks = [
            AudioTask(audio=mx.zeros((8000,)), target="short"),  # 0.5 seconds
            AudioTask(audio=mx.zeros((24000,)), target="long"),  # 1.5 seconds
        ]

        env = AudioTaskEnv(tasks=tasks, sample_rate=16000)

        # Check first task
        observation, info = env.reset(options={"task_index": 0})
        duration = float(observation["duration"][0])
        assert 0.4 < duration < 0.6  # Around 0.5 seconds (padded to max_samples)

    def test_episode_statistics_accumulation(self):
        """Test that episode statistics accumulate correctly."""
        tasks = [AudioTask(audio=mx.zeros((16000,)), target="test") for _ in range(5)]

        env = AudioTaskEnv(tasks=tasks, max_episode_steps=10)

        observation, info = env.reset()

        # Make several predictions
        correct_count = 0
        for i in range(5):
            if i < 2:
                # Correct answers
                observation, reward, terminated, truncated, info = env.step("test")
                correct_count += 1
            else:
                # Wrong answers
                observation, reward, terminated, truncated, info = env.step("wrong")

            if terminated or truncated:
                break

        # Check final accuracy
        assert info["episode_correct"] == correct_count
        assert info["episode_total"] >= correct_count
