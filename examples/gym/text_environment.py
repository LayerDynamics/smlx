"""
Text-based environment example.

Demonstrates:
- Creating custom text environments
- Question answering environment
- Instruction following environment
- Using the factory function
"""

from smlx.gym.envs.text_world import Task, TextWorldEnv, create_text_env


def qa_example():
    """Question answering environment example."""
    print("=" * 60)
    print("Question Answering Environment")
    print("=" * 60)

    # Define QA tasks
    qa_tasks = [
        Task(
            prompt="What is the capital of France?",
            target="Paris",
            reward=1.0,
            metadata={"type": "geography", "difficulty": "easy"},
        ),
        Task(
            prompt="What is 2 + 2?",
            target="4",
            reward=1.0,
            metadata={"type": "math", "difficulty": "easy"},
        ),
        Task(
            prompt="Who wrote Romeo and Juliet?",
            target="William Shakespeare",
            reward=1.0,
            metadata={"type": "literature", "difficulty": "medium"},
        ),
    ]

    # Create QA environment
    env = create_text_env("qa", tasks=qa_tasks)

    print(f"Environment: {type(env).__name__}")
    print(f"Number of tasks: {len(qa_tasks)}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Run a few episodes
    for episode in range(3):
        observation, info = env.reset()
        print(f"\nEpisode {episode + 1}:")
        print(f"  Question: {observation['prompt']}")
        print(f"  Target: {observation['target']}")

        # Simulate agent response (in practice, this would come from an LLM)
        if episode == 0:
            action = "Paris"  # Correct answer
        elif episode == 1:
            action = "4"  # Correct answer
        else:
            action = "Shakespeare"  # Partially correct

        observation, reward, terminated, truncated, info = env.step(action)

        print(f"  Agent answer: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Success: {info.get('success', False)}")

    env.close()
    print()


def instruction_following_example():
    """Instruction following environment example."""
    print("=" * 60)
    print("Instruction Following Environment")
    print("=" * 60)

    # Define instruction tasks
    instruction_tasks = [
        Task(
            prompt="Write a short greeting in French.",
            target="bonjour",
            reward=1.0,
            metadata={"language": "french", "type": "greeting"},
        ),
        Task(
            prompt="List three primary colors separated by commas.",
            target="red, blue, yellow",
            reward=1.0,
            metadata={"type": "list", "category": "colors"},
        ),
        Task(
            prompt="Describe the weather using exactly five words.",
            target="",  # Target is empty, will use partial matching
            reward=1.0,
            metadata={"type": "creative", "constraint": "word_count"},
        ),
    ]

    # Create instruction following environment
    env = create_text_env("instruction", tasks=instruction_tasks)

    print(f"Environment: {type(env).__name__}")
    print(f"Number of tasks: {len(instruction_tasks)}")
    print()

    # Run episodes
    for episode in range(3):
        observation, info = env.reset()
        print(f"\nEpisode {episode + 1}:")
        print(f"  Instruction: {observation['prompt']}")
        print(f"  Expected target: '{observation['target']}'")

        # Simulate agent responses
        if episode == 0:
            action = "Bonjour, comment allez-vous?"  # Contains target
        elif episode == 1:
            action = "red, blue, yellow"  # Exact match
        else:
            action = "The sunny day looks beautiful"  # 5 words, creative

        observation, reward, terminated, truncated, info = env.step(action)

        print(f"  Agent response: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Success: {info.get('success', False)}")

    env.close()
    print()


def custom_text_env_example():
    """Custom text environment example."""
    print("=" * 60)
    print("Custom Text Environment")
    print("=" * 60)

    # Create custom tasks for code generation
    code_tasks = [
        Task(
            prompt="Write a Python function that adds two numbers.",
            target="def add(a, b):\n    return a + b",
            reward=1.0,
            metadata={"language": "python", "difficulty": "easy"},
        ),
        Task(
            prompt="Write a function to check if a number is even.",
            target="def is_even(n):\n    return n % 2 == 0",
            reward=1.0,
            metadata={"language": "python", "difficulty": "easy"},
        ),
    ]

    # Create custom environment
    env = TextWorldEnv(
        tasks=code_tasks,
        max_episode_length=1,
        reward_on_success=2.0,  # Higher reward for correct code
        reward_on_failure=-0.5,
        partial_reward=True,
        similarity_threshold=0.7,
    )

    print(f"Environment: {type(env).__name__}")
    print(f"Tasks: Code generation")
    print(f"Reward on success: {env.reward_on_success}")
    print(f"Partial rewards enabled: {env.partial_reward}")
    print()

    # Run episode
    observation, info = env.reset()
    print("Task:")
    print(f"  Prompt: {observation['prompt']}")
    print(f"  Target code:\n{observation['target']}")

    # Simulate agent code generation
    action = """def add(a, b):
    return a + b"""

    observation, reward, terminated, truncated, info = env.step(action)

    print(f"\nAgent generated code:\n{action}")
    print(f"\nReward: {reward:.2f}")
    print(f"Success: {info.get('success', False)}")
    print(f"Similarity: {info.get('similarity', 0.0):.3f}")

    env.close()
    print()


def main():
    """Run all text environment examples."""
    print("\n" + "=" * 60)
    print("Text Environment Examples")
    print("=" * 60)
    print()

    qa_example()
    instruction_following_example()
    custom_text_env_example()

    print("=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
