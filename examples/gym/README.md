# SMLX Gym Examples

This directory contains example scripts demonstrating the SMLX Gym system for reinforcement learning.

## Examples

### 1. Basic CartPole (`basic_cartpole.py`)

**Description**: Simple example showing how to create a gym environment and run a random agent.

**Demonstrates**:
- Creating a gym environment
- Using `RandomAgent`
- Running episodes
- Recording statistics with wrappers

**Usage**:
```bash
python examples/gym/basic_cartpole.py
```

**Expected Output**: Episode statistics for random agent on CartPole

---

### 2. DQN Training (`dqn_cartpole.py`)

**Description**: Complete DQN training example on CartPole environment.

**Demonstrates**:
- Training a DQN agent from scratch
- Using experience replay
- Epsilon-greedy exploration with decay
- Target network updates
- Evaluating trained agent

**Usage**:
```bash
python examples/gym/dqn_cartpole.py
```

**Expected Output**: Training progress and evaluation results showing improved performance

**Training Details**:
- Episodes: 200
- Epsilon decay: 1.0 → 0.01
- Buffer size: 10,000 transitions
- Batch size: 64
- Target update frequency: Every 10 episodes

---

### 3. Text Environment (`text_environment.py`)

**Description**: Examples of text-based environments for language tasks.

**Demonstrates**:
- Question answering environment
- Instruction following environment
- Custom text environment creation
- Using the factory function

**Usage**:
```bash
python examples/gym/text_environment.py
```

**Expected Output**: Examples of QA, instruction following, and code generation tasks

**Use Cases**:
- Language model evaluation
- Instruction following benchmarks
- Code generation tasks
- Creative text generation

---

### 4. Benchmark Example (`benchmark_example.py`)

**Description**: Comprehensive benchmarking of RL agents.

**Demonstrates**:
- Basic agent benchmarking
- Comparing multiple agents
- Using benchmark configuration
- Detailed metrics analysis

**Usage**:
```bash
python examples/gym/benchmark_example.py
```

**Expected Output**: Performance metrics including returns, success rate, throughput, and memory usage

**Metrics Tracked**:
- Episode return (mean, std)
- Episode length
- Success rate
- Training/inference time
- Steps per second
- Peak memory usage

---

### 5. Wrappers Example (`wrappers_example.py`)

**Description**: Demonstrates environment wrappers for customizing behavior.

**Demonstrates**:
- Basic wrapper usage
- Observation and reward normalization
- Frame stacking for temporal context
- Reward clipping
- Comprehensive wrapper pipelines
- Episode logging

**Usage**:
```bash
python examples/gym/wrappers_example.py
```

**Expected Output**: Examples of different wrapper effects on observations and rewards

**Available Wrappers**:
- `MLXObservationWrapper` - Convert observations to MLX arrays
- `NormalizeObservation` - Normalize observations using running statistics
- `NormalizeReward` - Normalize rewards using return-based statistics
- `ClipReward` - Clip rewards to specified range
- `FrameStack` - Stack last N observations for temporal context
- `RecordEpisodeStatistics` - Track episode returns and lengths
- `TimeLimit` - Enforce maximum episode length
- `EpisodeLogger` - Log episode information to console

---

## Running All Examples

To run all examples in sequence:

```bash
for script in basic_cartpole.py dqn_cartpole.py text_environment.py benchmark_example.py wrappers_example.py; do
    echo "Running $script..."
    python examples/gym/$script
    echo ""
done
```

## Requirements

All examples require:
- `gymnasium` (OpenAI Gym successor)
- `mlx` (Apple MLX framework)
- `numpy`
- SMLX package installed (`pip install -e .`)

Some examples may require additional dependencies:
- DQN training: `mlx.optimizers`, `mlx.nn`
- Text environments: No additional dependencies
- Vision/audio environments: See respective environment documentation

## Environment Support

The examples use these environments:

**Classic Control**:
- CartPole-v1 (basic RL benchmark)

**Custom Environments**:
- TextWorldEnv (question answering, instruction following)
- VisionTaskEnv (image classification, visual QA)
- AudioTaskEnv (speech recognition, audio classification)

## Extending the Examples

### Adding New Agents

To add a new agent type:

1. Create agent class inheriting from `RLAgent`
2. Implement `select_action()` method
3. (Optional) Implement `train_step()` for learning agents

```python
from smlx.agents.rl_agent import RLAgent

class MyAgent(RLAgent):
    def select_action(self, observation):
        # Your action selection logic
        return action
```

### Creating Custom Environments

To create a custom environment:

1. Inherit from `MLXEnv` or use gym.Env
2. Implement `reset()` and `step()` methods
3. Define observation_space and action_space

```python
from smlx.gym.base import MLXEnv
import gymnasium as gym

class MyCustomEnv(MLXEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(...)
        self.action_space = gym.spaces.Discrete(...)

    def reset(self, **kwargs):
        # Return initial observation
        return observation, info

    def step(self, action):
        # Execute action and return results
        return observation, reward, terminated, truncated, info
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'smlx'`
**Solution**: Install SMLX package: `pip install -e .` from repository root

**Issue**: `ModuleNotFoundError: No module named 'gymnasium'`
**Solution**: Install gymnasium: `pip install gymnasium`

**Issue**: MLX not available
**Solution**: Ensure you're running on Apple Silicon (M1/M2/M3/M4) with macOS 13.5+

**Issue**: DQN training not converging
**Solution**: Try adjusting hyperparameters (learning rate, epsilon decay, buffer size)

## Additional Resources

- **SMLX_Gym.md**: Comprehensive design document for the gym system
- **smlx/gym/**: Source code for environments and wrappers
- **smlx/agents/**: Source code for RL agents
- **smlx/bench/suites/rl.py**: Benchmarking utilities
- **smlx/evals/rl_eval.py**: Evaluation utilities

## Contributing

To add new examples:

1. Create a new Python script in this directory
2. Follow the existing structure and documentation style
3. Include docstrings explaining what the example demonstrates
4. Add a section to this README
5. Test the example to ensure it runs correctly

## License

Copyright © 2025 SMLX Project
