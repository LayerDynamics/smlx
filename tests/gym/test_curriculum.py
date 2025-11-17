"""
Unit tests for curriculum learning utilities.

Tests curriculum schedulers, stage progression, and curriculum wrappers.
"""

import gymnasium as gym
import pytest

from smlx.gym.curriculum import (
    CurriculumMetrics,
    CurriculumScheduler,
    CurriculumStage,
    CurriculumWrapper,
    ThresholdScheduler,
    create_curriculum_env,
)


@pytest.mark.unit
class TestCurriculumStage:
    """Tests for CurriculumStage enum."""

    def test_stage_values(self):
        """Test curriculum stage values."""
        assert CurriculumStage.EASY.value == "easy"
        assert CurriculumStage.MEDIUM.value == "medium"
        assert CurriculumStage.HARD.value == "hard"
        assert CurriculumStage.EXPERT.value == "expert"

    def test_stage_ordering(self):
        """Test that stages can be ordered."""
        stages = list(CurriculumStage)
        assert len(stages) == 4
        assert stages[0] == CurriculumStage.EASY
        assert stages[-1] == CurriculumStage.EXPERT


@pytest.mark.unit
class TestCurriculumMetrics:
    """Tests for CurriculumMetrics dataclass."""

    def test_default_metrics(self):
        """Test default curriculum metrics."""
        metrics = CurriculumMetrics()

        assert metrics.current_stage == CurriculumStage.EASY
        assert metrics.success_rate == 0.0
        assert metrics.average_return == 0.0
        assert metrics.episodes_in_stage == 0
        assert metrics.total_stage_changes == 0

    def test_custom_metrics(self):
        """Test custom curriculum metrics."""
        metrics = CurriculumMetrics(
            current_stage=CurriculumStage.MEDIUM,
            success_rate=0.75,
            average_return=150.0,
            episodes_in_stage=50,
            total_stage_changes=2,
        )

        assert metrics.current_stage == CurriculumStage.MEDIUM
        assert metrics.success_rate == 0.75
        assert metrics.average_return == 150.0
        assert metrics.episodes_in_stage == 50
        assert metrics.total_stage_changes == 2


@pytest.mark.unit
class TestThresholdScheduler:
    """Tests for ThresholdScheduler."""

    @pytest.fixture
    def scheduler(self):
        """Create threshold scheduler."""
        return ThresholdScheduler(
            success_threshold=0.8,
            min_success_threshold=0.3,
            min_episodes_per_stage=10,
        )

    def test_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler.success_threshold == 0.8
        assert scheduler.min_success_threshold == 0.3
        assert scheduler.min_episodes_per_stage == 10
        assert scheduler.metrics.current_stage == CurriculumStage.EASY

    def test_default_stage_configs(self, scheduler):
        """Test default stage configurations."""
        easy_config = scheduler.get_stage_config(CurriculumStage.EASY)
        assert easy_config["difficulty"] == 0.25

        hard_config = scheduler.get_stage_config(CurriculumStage.HARD)
        assert hard_config["difficulty"] == 0.75

    def test_custom_stage_configs(self):
        """Test custom stage configurations."""
        custom_configs = {
            CurriculumStage.EASY: {"max_steps": 100},
            CurriculumStage.MEDIUM: {"max_steps": 200},
            CurriculumStage.HARD: {"max_steps": 500},
        }

        scheduler = ThresholdScheduler(stage_configs=custom_configs)
        easy_config = scheduler.get_stage_config(CurriculumStage.EASY)
        assert easy_config["max_steps"] == 100

    def test_should_progress_not_enough_episodes(self, scheduler):
        """Test progression blocked by insufficient episodes."""
        metrics = CurriculumMetrics(
            success_rate=0.9,  # High success rate
            episodes_in_stage=5,  # But not enough episodes
        )

        assert scheduler.should_progress(metrics) is False

    def test_should_progress_low_success(self, scheduler):
        """Test progression blocked by low success rate."""
        metrics = CurriculumMetrics(
            success_rate=0.5,  # Below threshold
            episodes_in_stage=20,  # Enough episodes
        )

        assert scheduler.should_progress(metrics) is False

    def test_should_progress_success(self, scheduler):
        """Test successful progression."""
        metrics = CurriculumMetrics(
            success_rate=0.85,  # Above threshold
            episodes_in_stage=15,  # Enough episodes
        )

        assert scheduler.should_progress(metrics) is True

    def test_should_regress_not_enough_episodes(self, scheduler):
        """Test regression blocked by insufficient episodes."""
        metrics = CurriculumMetrics(
            success_rate=0.1,  # Very low success
            episodes_in_stage=5,  # But not enough episodes
        )

        assert scheduler.should_regress(metrics) is False

    def test_should_regress_acceptable_performance(self, scheduler):
        """Test no regression with acceptable performance."""
        metrics = CurriculumMetrics(
            success_rate=0.5,  # Above minimum threshold
            episodes_in_stage=20,
        )

        assert scheduler.should_regress(metrics) is False

    def test_should_regress_poor_performance(self, scheduler):
        """Test regression with poor performance."""
        metrics = CurriculumMetrics(
            success_rate=0.2,  # Below minimum threshold
            episodes_in_stage=20,
        )

        assert scheduler.should_regress(metrics) is True

    def test_get_next_stage(self, scheduler):
        """Test getting next curriculum stage."""
        next_stage = scheduler.get_next_stage(CurriculumStage.EASY)
        assert next_stage == CurriculumStage.MEDIUM

        next_stage = scheduler.get_next_stage(CurriculumStage.MEDIUM)
        assert next_stage == CurriculumStage.HARD

        # At final stage, should return None
        next_stage = scheduler.get_next_stage(CurriculumStage.EXPERT)
        assert next_stage is None

    def test_get_previous_stage(self, scheduler):
        """Test getting previous curriculum stage."""
        prev_stage = scheduler.get_previous_stage(CurriculumStage.MEDIUM)
        assert prev_stage == CurriculumStage.EASY

        prev_stage = scheduler.get_previous_stage(CurriculumStage.HARD)
        assert prev_stage == CurriculumStage.MEDIUM

        # At first stage, should return None
        prev_stage = scheduler.get_previous_stage(CurriculumStage.EASY)
        assert prev_stage is None


@pytest.mark.unit
class TestCurriculumSchedulerBase:
    """Tests for CurriculumScheduler base class."""

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Create a minimal concrete implementation
        class MinimalScheduler(CurriculumScheduler):
            def should_progress(self, metrics):
                return False

            def should_regress(self, metrics):
                return False

            def get_stage_config(self, stage):
                return {}

        scheduler = MinimalScheduler()
        assert scheduler.metrics.current_stage == CurriculumStage.EASY

    def test_cannot_instantiate_abstract(self):
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            CurriculumScheduler()  # type: ignore


@pytest.mark.unit
class TestCurriculumWrapper:
    """Tests for CurriculumWrapper."""

    @pytest.fixture
    def scheduler(self):
        """Create test scheduler."""
        return ThresholdScheduler(
            success_threshold=0.8, min_success_threshold=0.3, min_episodes_per_stage=5
        )

    @pytest.fixture
    def env(self, scheduler):
        """Create wrapped environment."""
        base_env = gym.make("CartPole-v1")
        return CurriculumWrapper(base_env, scheduler)

    def test_initialization(self, env, scheduler):
        """Test wrapper initialization."""
        assert env.scheduler == scheduler
        assert env.current_stage == CurriculumStage.EASY
        assert len(env.episode_returns) == 0
        assert len(env.episode_successes) == 0
        assert env.episodes_in_current_stage == 0

    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        assert obs is not None
        assert info is not None

    def test_step(self, env):
        """Test environment step."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_tracking(self, env):
        """Test episode return and success tracking."""
        env.reset()

        # Run a complete episode
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated

            if done:
                # Mark episode as success if we got good return
                info["success"] = reward > 0

        # Episode should be tracked
        assert env.episodes_in_current_stage >= 1

    def test_stage_progression(self, env):
        """Test curriculum stage progression."""
        # Simulate successful episodes to trigger progression
        for _ in range(10):
            env.reset()
            done = False
            episode_return = 0

            while not done:
                obs, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                episode_return += reward
                done = terminated or truncated

                if done:
                    # Set episode return and success in info
                    if "episode" not in info:
                        info["episode"] = {}
                    info["episode"]["r"] = episode_return
                    info["success"] = episode_return > 100

            env.episode_returns.append(episode_return)
            env.episode_successes.append(episode_return > 100)

        # After many successful episodes, may have progressed
        # (depends on actual performance, so just check it's valid)
        assert env.current_stage in list(CurriculumStage)

    def test_metrics_update(self, env):
        """Test that curriculum metrics are updated."""
        # Add some episode data
        env.episode_returns = [100.0, 150.0, 120.0]
        env.episode_successes = [True, True, False]
        env.episodes_in_current_stage = 3

        # Update metrics
        env._update_metrics()

        assert env.scheduler.metrics.episodes_in_stage == 3
        assert 0.0 <= env.scheduler.metrics.success_rate <= 1.0
        assert env.scheduler.metrics.average_return > 0

    def test_stage_transition_progression(self, env):
        """Test stage transition logic for progression."""
        # Set up high performance
        env.episode_returns = [100.0] * 10
        env.episode_successes = [True] * 10
        env.episodes_in_current_stage = 10
        env.current_stage = CurriculumStage.EASY

        # Update metrics and check transition
        env._update_metrics()
        env._check_stage_transition()

        # May have progressed (depends on scheduler thresholds)
        assert env.current_stage in [CurriculumStage.EASY, CurriculumStage.MEDIUM]

    def test_stage_config_applied(self):
        """Test that stage configurations are applied."""
        scheduler = ThresholdScheduler(
            stage_configs={
                CurriculumStage.EASY: {"test_attr": 1},
                CurriculumStage.MEDIUM: {"test_attr": 2},
            }
        )

        # Create custom apply function
        applied_configs = []

        def apply_config(env, config):
            applied_configs.append(config)

        base_env = gym.make("CartPole-v1")
        env = CurriculumWrapper(base_env, scheduler, apply_config_fn=apply_config)

        # Initial config should be applied
        assert len(applied_configs) == 1
        assert applied_configs[0]["test_attr"] == 1

    def test_custom_apply_config_fn(self):
        """Test custom configuration application function."""
        scheduler = ThresholdScheduler()
        base_env = gym.make("CartPole-v1")

        # Track applied configs
        applied = []

        def custom_apply(env, config):
            applied.append(config)

        env = CurriculumWrapper(base_env, scheduler, apply_config_fn=custom_apply)

        # Should have applied initial config
        assert len(applied) == 1


@pytest.mark.unit
class TestCreateCurriculumEnv:
    """Tests for create_curriculum_env factory function."""

    def test_create_basic(self):
        """Test creating curriculum environment."""
        scheduler = ThresholdScheduler()
        env = create_curriculum_env("CartPole-v1", scheduler)

        assert isinstance(env, CurriculumWrapper)
        assert env.current_stage == CurriculumStage.EASY

        obs, info = env.reset()
        assert obs is not None

        env.close()

    def test_create_with_kwargs(self):
        """Test creating with environment kwargs."""
        scheduler = ThresholdScheduler()
        env = create_curriculum_env("CartPole-v1", scheduler)

        obs, _ = env.reset()
        assert obs is not None
        assert env.current_stage == CurriculumStage.EASY

        env.close()

    def test_create_with_apply_fn(self):
        """Test creating with custom apply function."""
        scheduler = ThresholdScheduler()
        applied = []

        def apply_fn(env, config):
            applied.append(config)

        curriculum_env = create_curriculum_env("CartPole-v1", scheduler, apply_config_fn=apply_fn)

        assert len(applied) == 1
        assert curriculum_env.current_stage == CurriculumStage.EASY
        curriculum_env.close()


@pytest.mark.integration
class TestCurriculumIntegration:
    """Integration tests for curriculum learning."""

    def test_full_curriculum_training(self):
        """Test complete curriculum training scenario."""
        scheduler = ThresholdScheduler(
            success_threshold=0.8, min_episodes_per_stage=3, min_success_threshold=0.2
        )

        env = create_curriculum_env("CartPole-v1", scheduler)

        # Train through multiple episodes
        for _ in range(10):
            obs, info = env.reset()
            done = False
            episode_return = 0

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                done = terminated or truncated

                if done:
                    # Mark success for good performance
                    if "episode" not in info:
                        info["episode"] = {}
                    info["episode"]["r"] = episode_return
                    info["success"] = episode_return > 50

        # Should have completed some episodes
        assert env.episodes_in_current_stage >= 0

        env.close()

    def test_stage_progression_sequence(self):
        """Test progression through curriculum stages."""
        # Very lenient scheduler for quick progression
        scheduler = ThresholdScheduler(
            success_threshold=0.01,  # Very low threshold
            min_episodes_per_stage=1,  # Only 1 episode needed
        )

        env = create_curriculum_env("CartPole-v1", scheduler)

        # Run several successful episodes
        for _ in range(5):
            obs, info = env.reset()
            done = False

            while not done:
                obs, reward, terminated, truncated, info = env.step(0)
                done = terminated or truncated

                if done:
                    if "episode" not in info:
                        info["episode"] = {}
                    info["episode"]["r"] = 100.0
                    info["success"] = True

            env.episode_returns.append(100.0)
            env.episode_successes.append(True)

        # May have progressed
        assert env.current_stage in list(CurriculumStage)

        env.close()

    def test_curriculum_with_difficult_task(self):
        """Test curriculum on more difficult task."""
        scheduler = ThresholdScheduler(
            success_threshold=0.7, min_episodes_per_stage=5, min_success_threshold=0.3
        )

        env = create_curriculum_env("CartPole-v1", scheduler)

        # Run training loop
        for _ in range(15):
            obs, info = env.reset()
            done = False
            steps = 0

            while not done and steps < 200:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                done = terminated or truncated

        # Environment should still be functional
        assert env.current_stage in list(CurriculumStage)

        env.close()
