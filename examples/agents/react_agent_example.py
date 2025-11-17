#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
ReAct Agent Examples.

Demonstrates the ReAct (Reasoning + Acting) agent:
1. Basic ReAct with built-in tools
2. Mathematical reasoning with calculator
3. Information retrieval with Wikipedia
4. Custom tools
5. Multi-step problem solving
"""

import sys
from pathlib import Path

# Add smlx to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smlx.agents import (
    ReActAgent,
    ToolParameter,
    ToolRegistry,
    create_default_registry,
)
from smlx.models.SmolLM2_135M import load


def example_1_basic_react():
    """Example 1: Basic ReAct agent with default tools."""
    print("\n" + "=" * 80)
    print("Example 1: Basic ReAct Agent")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create ReAct agent with default tools
    tools = create_default_registry()
    agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        name="ReActAgent",
        verbose=True,
        max_iterations=5,
    )

    # Simple task
    task = "What is 15 multiplied by 23?"

    print(f"\nTask: {task}")
    print("-" * 80)

    response = agent.run(task, temperature=0.3)

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"\nFinal Answer: {response.content}")
    print(f"\nSuccess: {response.success}")
    print(f"Tool calls made: {len(response.tool_calls)}")


def example_2_mathematical_reasoning():
    """Example 2: Multi-step mathematical problem."""
    print("\n" + "=" * 80)
    print("Example 2: Mathematical Reasoning")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create agent
    tools = create_default_registry()
    agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        verbose=True,
        max_iterations=10,
    )

    # Complex math problem
    tasks = [
        "If I have 47 apples and buy 23 more, then give away 15, how many do I have left?",
        "Calculate the area of a circle with radius 7. Use 3.14159 for pi.",
        "What is the sum of all numbers from 1 to 100?",
    ]

    for task in tasks:
        print(f"\n{'─' * 80}")
        print(f"Task: {task}")
        print(f"{'─' * 80}")

        response = agent.run(task, temperature=0.2)

        print(f"\nFinal Answer: {response.content}")
        print(f"Steps taken: {len(response.tool_calls)}")


def example_3_information_retrieval():
    """Example 3: Wikipedia information retrieval."""
    print("\n" + "=" * 80)
    print("Example 3: Information Retrieval")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create agent
    tools = create_default_registry()
    agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        verbose=True,
    )

    # Information tasks
    tasks = [
        "Who invented the telephone? Use Wikipedia to find out.",
        "What is the capital of Japan?",
    ]

    for task in tasks:
        print(f"\n{'─' * 80}")
        print(f"Task: {task}")
        print(f"{'─' * 80}")

        response = agent.run(task, temperature=0.3)
        print(f"\nAnswer: {response.content}")


def example_4_custom_tools():
    """Example 4: Agent with custom tools."""
    print("\n" + "=" * 80)
    print("Example 4: Custom Tools")
    print("=" * 80)

    # Define custom tools
    def string_length(text: str) -> int:
        """Calculate the length of a string."""
        return len(text)

    def reverse_string(text: str) -> str:
        """Reverse a string."""
        return text[::-1]

    def word_count(text: str) -> int:
        """Count words in text."""
        return len(text.split())

    # Create tool registry with custom tools
    tools = ToolRegistry()

    tools.register_function(
        name="string_length",
        description="Calculate the length of a string",
        func=string_length,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to measure",
                required=True,
            )
        ],
    )

    tools.register_function(
        name="reverse_string",
        description="Reverse a string",
        func=reverse_string,
    )

    tools.register_function(
        name="word_count",
        description="Count the number of words in text",
        func=word_count,
    )

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create agent
    agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        verbose=True,
    )

    # Tasks using custom tools
    tasks = [
        "How many characters are in the word 'supercalifragilisticexpialidocious'?",
        "Reverse the string 'hello world'",
        "How many words are in this sentence: 'The quick brown fox jumps over the lazy dog'?",
    ]

    for task in tasks:
        print(f"\n{'─' * 80}")
        print(f"Task: {task}")
        print(f"{'─' * 80}")

        response = agent.run(task, temperature=0.2)
        print(f"\nAnswer: {response.content}")


def example_5_multi_step_problem():
    """Example 5: Complex multi-step problem solving."""
    print("\n" + "=" * 80)
    print("Example 5: Multi-Step Problem Solving")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create agent
    tools = create_default_registry()
    agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        verbose=True,
        max_iterations=15,
    )

    # Complex problem requiring multiple steps
    task = """A store is having a sale. They have 150 items that normally cost $45 each.
They're offering a 20% discount. If someone buys 8 items, how much will they pay in total?"""

    print(f"\nTask: {task}")
    print("-" * 80)

    response = agent.run(task, temperature=0.2)

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"\nFinal Answer: {response.content}")
    print("\nReasoning Process:")
    print(response.reasoning)
    print(f"\nTool Calls: {len(response.tool_calls)}")
    for i, call in enumerate(response.tool_calls, 1):
        print(f"  {i}. {call['action']} → {call['result']}")


def example_6_error_handling():
    """Example 6: How ReAct handles errors."""
    print("\n" + "=" * 80)
    print("Example 6: Error Handling")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create agent
    tools = create_default_registry()
    agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        verbose=True,
    )

    # Task that might cause issues
    tasks = [
        "Calculate 10 divided by 0 using the calculator",
        "Search Wikipedia for 'xyzabc123impossible'",
    ]

    for task in tasks:
        print(f"\n{'─' * 80}")
        print(f"Task: {task}")
        print(f"{'─' * 80}")

        response = agent.run(task, temperature=0.2)

        print(f"\nSuccess: {response.success}")
        print(f"Answer: {response.content}")
        if response.error:
            print(f"Error encountered: {response.error}")


def main():
    """Run all examples."""
    print("=" * 80)
    print("SMLX Agent System - ReAct Examples")
    print("=" * 80)

    examples = [
        ("Basic ReAct", example_1_basic_react),
        ("Mathematical Reasoning", example_2_mathematical_reasoning),
        ("Information Retrieval", example_3_information_retrieval),
        ("Custom Tools", example_4_custom_tools),
        ("Multi-Step Problem", example_5_multi_step_problem),
        ("Error Handling", example_6_error_handling),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"\n\nError in {name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nKey Concepts Demonstrated:")
    print("  ✓ ReAct reasoning loop (Thought → Action → Observation)")
    print("  ✓ Built-in tools (calculator, time, Wikipedia)")
    print("  ✓ Custom tool creation")
    print("  ✓ Multi-step problem solving")
    print("  ✓ Error handling in agent execution")
    print("\nReAct Framework:")
    print("  - Synergizes reasoning and acting")
    print("  - Alternates between thinking and tool use")
    print("  - Enables complex problem solving")


if __name__ == "__main__":
    main()
