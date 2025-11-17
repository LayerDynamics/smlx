"""
Tests for RL agent evaluation framework.

Tests all evaluation utilities including:
- EvalConfig and EvalResults dataclasses
- evaluate_agent() function
- evaluate_multiple_agents() function
- compare_to_baseline() function
- generate_eval_report() function
- save/load evaluation results
"""

import gymnasium as gym
import numpy as np
import pytest

from smlx.agents.rl_agent import EpsilonGreedyAgent, GreedyAgent, RandomAgent
from smlx.evals.rl_eval import (
    EvalConfig,
    EvalResults,
    compare_to_baseline,
    evaluate_agent,
    evaluate_multiple_agents,
    generate_eval_report,
    load_eval_results,
    save_eval_results,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def env():
    """Create simple CartPole environment for testing."""
    env = gym.make("CartPole-v1")
    yield env
    env.close()


@pytest.fixture
def random_agent(env):
    """Create RandomAgent for testing."""
    return RandomAgent(env)


@pytest.fixture
def dummy_q_function():
    """Create dummy Q-function for testing."""

    def q_func(state, action):
        # Simple Q-function that always returns 1.0
        return 1.0

    return q_func


@pytest.fixture
def greedy_agent(env, dummy_q_function):
    """Create GreedyAgent for testing."""
    return GreedyAgent(env, q_function=dummy_q_function)


@pytest.fixture
def epsilon_greedy_agent(env, dummy_q_function):
    """Create EpsilonGreedyAgent for testing."""
    return EpsilonGreedyAgent(env, q_function=dummy_q_function, epsilon=0.1)


@pytest.fixture
def eval_config():
    """Create default EvalConfig for testing."""
    return EvalConfig(
        num_episodes=5,  # Use small number for fast tests
        max_steps_per_episode=100,
        seed=42,
        deterministic=True,
        render=False,
        save_trajectories=False,
    )


@pytest.fixture
def sample_eval_results():
    """Create sample EvalResults for testing."""
    return EvalResults(
        mean_return=100.0,
        std_return=10.0,
        min_return=80.0,
        max_return=120.0,
        median_return=98.0,
        mean_length=50.0,
        std_length=5.0,
        success_rate=0.8,
        total_time=10.0,
        steps_per_second=250.0,
        peak_memory_mb=50.0,
        episode_returns=[90.0, 100.0, 110.0, 95.0, 105.0],
        episode_lengths=[48, 50, 52, 49, 51],
        episode_successes=[True, True, True, False, True],
        metadata={"env_name": "CartPole-v1", "agent_type": "RandomAgent"},
    )


# ============================================================================
# Test EvalConfig
# ============================================================================


@pytest.mark.unit
def test_eval_config_creation():
    """Test EvalConfig creation with defaults."""
    config = EvalConfig()

    assert config.num_episodes == 100
    assert config.max_steps_per_episode == 1000
    assert config.seed == 42
    assert config.deterministic is True
    assert config.render is False
    assert config.save_trajectories is False
    assert config.video_freq is None
    assert config.checkpoint_path is None


@pytest.mark.unit
def test_eval_config_custom_values():
    """Test EvalConfig creation with custom values."""
    config = EvalConfig(
        num_episodes=50,
        max_steps_per_episode=500,
        seed=123,
        deterministic=False,
        render=True,
        save_trajectories=True,
        video_freq=10,
        checkpoint_path="/path/to/checkpoint.pkl",
    )

    assert config.num_episodes == 50
    assert config.max_steps_per_episode == 500
    assert config.seed == 123
    assert config.deterministic is False
    assert config.render is True
    assert config.save_trajectories is True
    assert config.video_freq == 10
    assert config.checkpoint_path == "/path/to/checkpoint.pkl"


# ============================================================================
# Test EvalResults
# ============================================================================


@pytest.mark.unit
def test_eval_results_creation(sample_eval_results):
    """Test EvalResults creation."""
    results = sample_eval_results

    assert results.mean_return == 100.0
    assert results.std_return == 10.0
    assert results.min_return == 80.0
    assert results.max_return == 120.0
    assert results.median_return == 98.0
    assert results.mean_length == 50.0
    assert results.std_length == 5.0
    assert results.success_rate == 0.8
    assert results.total_time == 10.0
    assert results.steps_per_second == 250.0
    assert results.peak_memory_mb == 50.0
    assert len(results.episode_returns) == 5
    assert len(results.episode_lengths) == 5
    assert len(results.episode_successes) == 5


@pytest.mark.unit
def test_eval_results_to_dict(sample_eval_results):
    """Test EvalResults.to_dict() method."""
    results = sample_eval_results
    result_dict = results.to_dict()

    assert isinstance(result_dict, dict)
    assert result_dict["mean_return"] == 100.0
    assert result_dict["std_return"] == 10.0
    assert result_dict["success_rate"] == 0.8
    assert "episode_returns" in result_dict
    assert "episode_lengths" in result_dict
    assert "metadata" in result_dict


@pytest.mark.unit
def test_eval_results_str(sample_eval_results):
    """Test EvalResults.__str__() method."""
    results = sample_eval_results
    result_str = str(results)

    assert isinstance(result_str, str)
    assert "RL Evaluation Results" in result_str
    assert "Mean Return:" in result_str
    assert "Success Rate:" in result_str
    assert "100.00" in result_str  # mean_return
    assert "80.0%" in result_str  # success_rate


# ============================================================================
# Test evaluate_agent
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_evaluate_agent_random(random_agent, eval_config):
    """Test evaluate_agent with RandomAgent."""
    results = evaluate_agent(random_agent, config=eval_config)

    # Check basic result structure
    assert isinstance(results, EvalResults)
    assert len(results.episode_returns) == eval_config.num_episodes
    assert len(results.episode_lengths) == eval_config.num_episodes
    assert len(results.episode_successes) == eval_config.num_episodes

    # Check statistics are computed
    assert results.mean_return == pytest.approx(np.mean(results.episode_returns), abs=1e-5)
    assert results.std_return == pytest.approx(np.std(results.episode_returns), abs=1e-5)
    assert results.min_return == min(results.episode_returns)
    assert results.max_return == max(results.episode_returns)
    assert results.median_return == pytest.approx(
        np.median(results.episode_returns), abs=1e-5
    )

    # Check performance metrics
    assert results.total_time > 0
    assert results.steps_per_second > 0

    # Check metadata
    assert "env_name" in results.metadata
    assert "agent_type" in results.metadata
    assert results.metadata["agent_type"] == "RandomAgent"


@pytest.mark.integration
@pytest.mark.slow
def test_evaluate_agent_with_trajectories(random_agent):
    """Test evaluate_agent with trajectory saving."""
    config = EvalConfig(
        num_episodes=3,
        max_steps_per_episode=50,
        save_trajectories=True,
    )

    results = evaluate_agent(random_agent, config=config)

    # Check trajectories are saved
    assert len(results.trajectories) == config.num_episodes
    for trajectory in results.trajectories:
        assert "observations" in trajectory
        assert "actions" in trajectory
        assert "rewards" in trajectory
        assert "infos" in trajectory
        assert len(trajectory["observations"]) > 0
        assert len(trajectory["actions"]) == len(trajectory["observations"])
        assert len(trajectory["rewards"]) == len(trajectory["observations"])


@pytest.mark.integration
@pytest.mark.slow
def test_evaluate_agent_deterministic(epsilon_greedy_agent):
    """Test that deterministic evaluation disables exploration."""
    config = EvalConfig(
        num_episodes=5,
        deterministic=True,
    )

    # Agent should have epsilon > 0 initially
    assert epsilon_greedy_agent.epsilon == 0.1

    results = evaluate_agent(epsilon_greedy_agent, config=config)

    # After evaluation, epsilon should be restored
    assert epsilon_greedy_agent.epsilon == 0.1

    # Evaluation should complete successfully
    assert isinstance(results, EvalResults)
    assert len(results.episode_returns) == 5


@pytest.mark.integration
@pytest.mark.slow
def test_evaluate_agent_with_env_override(random_agent):
    """Test evaluate_agent with different environment."""
    # Create separate environment
    eval_env = gym.make("CartPole-v1")

    config = EvalConfig(num_episodes=3)
    results = evaluate_agent(random_agent, env=eval_env, config=config)

    assert isinstance(results, EvalResults)
    assert len(results.episode_returns) == 3

    eval_env.close()


@pytest.mark.integration
def test_evaluate_agent_max_steps_limit(random_agent):
    """Test that max_steps_per_episode is enforced."""
    config = EvalConfig(
        num_episodes=3,
        max_steps_per_episode=10,  # Very short episodes
    )

    results = evaluate_agent(random_agent, config=config)

    # Episode lengths should not exceed max_steps
    for length in results.episode_lengths:
        assert length <= config.max_steps_per_episode


# ============================================================================
# Test evaluate_multiple_agents
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_evaluate_multiple_agents(env, random_agent, greedy_agent):
    """Test evaluate_multiple_agents with multiple agent types."""
    agents = {
        "random": random_agent,
        "greedy": greedy_agent,
    }

    config = EvalConfig(num_episodes=5)
    results = evaluate_multiple_agents(agents, env, config)

    # Check results for each agent
    assert isinstance(results, dict)
    assert "random" in results
    assert "greedy" in results

    for agent_name, result in results.items():
        assert isinstance(result, EvalResults)
        assert len(result.episode_returns) == 5
        assert result.metadata["agent_type"] in ["RandomAgent", "GreedyAgent"]
        assert agent_name in ["random", "greedy"]


@pytest.mark.integration
def test_evaluate_multiple_agents_empty():
    """Test evaluate_multiple_agents with empty agent dict."""
    env = gym.make("CartPole-v1")
    agents = {}

    results = evaluate_multiple_agents(agents, env)

    assert isinstance(results, dict)
    assert len(results) == 0

    env.close()


# ============================================================================
# Test compare_to_baseline
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_compare_to_baseline_random(random_agent, env):
    """Test compare_to_baseline with random baseline."""
    config = EvalConfig(num_episodes=5)

    comparison = compare_to_baseline(random_agent, env, baseline="random", config=config)

    # Check comparison structure
    assert isinstance(comparison, dict)
    assert "agent_results" in comparison
    assert "baseline_results" in comparison
    assert "improvement" in comparison
    assert "improvement_percent" in comparison
    assert "agent_better" in comparison

    # Check results are EvalResults
    assert isinstance(comparison["agent_results"], EvalResults)
    assert isinstance(comparison["baseline_results"], EvalResults)

    # Check improvement calculation
    agent_mean = comparison["agent_results"].mean_return
    baseline_mean = comparison["baseline_results"].mean_return
    expected_improvement = agent_mean - baseline_mean

    assert comparison["improvement"] == pytest.approx(expected_improvement, abs=1e-5)
    assert comparison["agent_better"] == (agent_mean > baseline_mean)


@pytest.mark.integration
def test_compare_to_baseline_unknown():
    """Test compare_to_baseline with unknown baseline raises error."""
    env = gym.make("CartPole-v1")
    agent = RandomAgent(env)

    with pytest.raises(ValueError, match="Unknown baseline"):
        compare_to_baseline(agent, env, baseline="unknown")

    env.close()


# ============================================================================
# Test generate_eval_report
# ============================================================================


@pytest.mark.unit
def test_generate_eval_report(sample_eval_results):
    """Test generate_eval_report with single agent."""
    results = {"agent1": sample_eval_results}

    report = generate_eval_report(results)

    assert isinstance(report, str)
    assert "RL Agent Evaluation Report" in report
    assert "agent1" in report
    assert "100.00" in report  # mean_return
    assert "80.0%" in report  # success_rate


@pytest.mark.unit
def test_generate_eval_report_multiple_agents():
    """Test generate_eval_report with multiple agents."""
    results = {
        "agent1": EvalResults(
            mean_return=100.0,
            std_return=10.0,
            min_return=80.0,
            max_return=120.0,
            median_return=100.0,
            mean_length=50.0,
            std_length=5.0,
            success_rate=0.8,
            total_time=10.0,
            steps_per_second=250.0,
            peak_memory_mb=50.0,
        ),
        "agent2": EvalResults(
            mean_return=120.0,
            std_return=8.0,
            min_return=100.0,
            max_return=140.0,
            median_return=120.0,
            mean_length=55.0,
            std_length=4.0,
            success_rate=0.9,
            total_time=12.0,
            steps_per_second=230.0,
            peak_memory_mb=55.0,
        ),
    }

    report = generate_eval_report(results)

    assert isinstance(report, str)
    assert "agent1" in report
    assert "agent2" in report

    # Agents should be sorted by mean return (agent2 first)
    # In the summary table, agent2 should appear before agent1
    # (higher mean return appears first)
    assert report.find("agent2") < report.find("agent1")


@pytest.mark.unit
def test_generate_eval_report_save(sample_eval_results, tmp_path):
    """Test generate_eval_report with file saving."""
    results = {"agent1": sample_eval_results}
    save_path = tmp_path / "report.txt"

    generate_eval_report(results, save_path=str(save_path))

    # Check report was saved
    assert save_path.exists()
    saved_content = save_path.read_text()
    assert "RL Agent Evaluation Report" in saved_content
    assert "agent1" in saved_content


# ============================================================================
# Test save_eval_results and load_eval_results
# ============================================================================


@pytest.mark.unit
def test_save_load_eval_results_json(sample_eval_results, tmp_path):
    """Test saving and loading EvalResults in JSON format."""
    save_path = tmp_path / "results.json"

    # Save results
    save_eval_results(sample_eval_results, str(save_path), format="json")

    # Check file exists
    assert save_path.exists()

    # Load results
    loaded_results = load_eval_results(str(save_path), format="json")

    # Check loaded results match original
    assert isinstance(loaded_results, EvalResults)
    assert loaded_results.mean_return == sample_eval_results.mean_return
    assert loaded_results.std_return == sample_eval_results.std_return
    assert loaded_results.success_rate == sample_eval_results.success_rate
    assert loaded_results.episode_returns == sample_eval_results.episode_returns
    assert loaded_results.metadata == sample_eval_results.metadata


@pytest.mark.unit
def test_save_load_eval_results_pickle(sample_eval_results, tmp_path):
    """Test saving and loading EvalResults in pickle format."""
    save_path = tmp_path / "results.pkl"

    # Save results
    save_eval_results(sample_eval_results, str(save_path), format="pickle")

    # Check file exists
    assert save_path.exists()

    # Load results
    loaded_results = load_eval_results(str(save_path), format="pickle")

    # Check loaded results match original
    assert isinstance(loaded_results, EvalResults)
    assert loaded_results.mean_return == sample_eval_results.mean_return
    assert loaded_results.std_return == sample_eval_results.std_return
    assert loaded_results.success_rate == sample_eval_results.success_rate
    assert loaded_results.episode_returns == sample_eval_results.episode_returns
    assert loaded_results.trajectories == sample_eval_results.trajectories


@pytest.mark.unit
def test_save_eval_results_unknown_format(sample_eval_results, tmp_path):
    """Test save_eval_results with unknown format raises error."""
    save_path = tmp_path / "results.unknown"

    with pytest.raises(ValueError, match="Unknown format"):
        save_eval_results(sample_eval_results, str(save_path), format="unknown")


@pytest.mark.unit
def test_load_eval_results_unknown_format(tmp_path):
    """Test load_eval_results with unknown format raises error."""
    load_path = tmp_path / "results.unknown"
    load_path.write_text("{}")  # Create dummy file

    with pytest.raises(ValueError, match="Unknown format"):
        load_eval_results(str(load_path), format="unknown")


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_full_evaluation_workflow(env, tmp_path):
    """Test complete evaluation workflow: evaluate, compare, report, save/load."""
    # Create agents
    random_agent = RandomAgent(env)
    greedy_agent = GreedyAgent(env, q_function=lambda s, a: 1.0)

    # Configure evaluation
    config = EvalConfig(num_episodes=5, deterministic=True)

    # Evaluate multiple agents
    agents = {"random": random_agent, "greedy": greedy_agent}
    results = evaluate_multiple_agents(agents, env, config)

    # Generate report
    report_path = tmp_path / "report.txt"
    generate_eval_report(results, save_path=str(report_path))

    assert report_path.exists()
    report_content = report_path.read_text()
    assert "random" in report_content
    assert "greedy" in report_content

    # Save individual results
    for name, result in results.items():
        save_path = tmp_path / f"{name}_results.json"
        save_eval_results(result, str(save_path), format="json")

        # Load and verify
        loaded = load_eval_results(str(save_path), format="json")
        assert loaded.mean_return == result.mean_return

    # Compare to baseline
    comparison = compare_to_baseline(random_agent, env, baseline="random", config=config)

    assert "agent_results" in comparison
    assert "baseline_results" in comparison


@pytest.mark.integration
@pytest.mark.slow
def test_evaluation_seeding(env):
    """Test that evaluation respects seed configuration."""
    # Note: RandomAgent uses env.action_space.sample() which uses Python's random module
    # This makes perfect reproducibility difficult, but we can verify seeding works
    config = EvalConfig(num_episodes=10, seed=42, deterministic=True)

    agent = RandomAgent(env)

    # Run evaluation - should complete successfully with seeding
    results = evaluate_agent(agent, config=config)

    # Verify evaluation completed correctly
    assert len(results.episode_returns) == 10
    assert len(results.episode_lengths) == 10
    assert results.metadata["seed"] == 42
    assert results.metadata["deterministic"] is True

    # Verify statistics are reasonable for CartPole
    assert 0 < results.mean_return < 500  # CartPole episodes end quickly for random policy
    assert results.min_return >= 0
    assert results.max_return >= results.mean_return


@pytest.mark.integration
def test_evaluation_different_seeds(random_agent):
    """Test that evaluation with different seeds produces different results."""
    config1 = EvalConfig(num_episodes=10, seed=42)
    config2 = EvalConfig(num_episodes=10, seed=123)

    results1 = evaluate_agent(random_agent, config=config1)
    results2 = evaluate_agent(random_agent, config=config2)

    # Results should be different (different seeds)
    # (Note: There's a tiny chance they could be the same by coincidence)
    assert results1.episode_returns != results2.episode_returns or results1.episode_lengths != results2.episode_lengths


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.unit
def test_eval_results_empty_episodes():
    """Test EvalResults with empty episode lists."""
    results = EvalResults(
        mean_return=0.0,
        std_return=0.0,
        min_return=0.0,
        max_return=0.0,
        median_return=0.0,
        mean_length=0.0,
        std_length=0.0,
        success_rate=0.0,
        total_time=0.0,
        steps_per_second=0.0,
        peak_memory_mb=0.0,
        episode_returns=[],
        episode_lengths=[],
        episode_successes=[],
    )

    # Should not crash
    result_dict = results.to_dict()
    result_str = str(results)

    assert isinstance(result_dict, dict)
    assert isinstance(result_str, str)


@pytest.mark.integration
def test_evaluate_agent_single_episode(random_agent):
    """Test evaluation with single episode."""
    config = EvalConfig(num_episodes=1)

    results = evaluate_agent(random_agent, config=config)

    assert len(results.episode_returns) == 1
    assert len(results.episode_lengths) == 1
    assert results.mean_return == results.episode_returns[0]
    assert results.std_return == 0.0  # Single episode has no variance
