#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Chain-of-Thought (CoT) Agent Examples.

Demonstrates Chain-of-Thought reasoning:
1. Zero-shot CoT (just "Let's think step by step")
2. Few-shot CoT (with examples)
3. Self-Consistency CoT (multiple samples + voting)
4. Mathematical reasoning
5. Logical puzzles
"""

import sys
from pathlib import Path

# Add smlx to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smlx.agents import CoTAgent, SelfConsistencyCoTAgent
from smlx.models.SmolLM2_135M import load


def example_1_zero_shot_cot():
    """Example 1: Zero-shot Chain-of-Thought."""
    print("\n" + "=" * 80)
    print("Example 1: Zero-Shot CoT")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create zero-shot CoT agent
    agent = CoTAgent(
        model=model,
        tokenizer=tokenizer,
        name="ZeroShotCoT",
        zero_shot=True,
        verbose=True,
    )

    # Mathematical problem
    task = "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?"

    print(f"\nTask: {task}")
    print("-" * 80)

    response = agent.run(task, temperature=0.3)

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"\nReasoning:\n{response.reasoning}")
    print(f"\nFinal Answer: {response.content}")


def example_2_few_shot_cot():
    """Example 2: Few-shot CoT with examples."""
    print("\n" + "=" * 80)
    print("Example 2: Few-Shot CoT")
    print("=" * 80)

    # Define examples
    examples = [
        {
            "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many tennis balls does he have now?",
            "reasoning": "Roger started with 5 balls. 2 cans of 3 balls each is 2 × 3 = 6 balls. 5 + 6 = 11.",
            "answer": "11 tennis balls",
        },
        {
            "question": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?",
            "reasoning": "They started with 23 apples. They used 20, so 23 - 20 = 3 apples left. Then they bought 6 more, so 3 + 6 = 9.",
            "answer": "9 apples",
        },
    ]

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create few-shot CoT agent
    agent = CoTAgent(
        model=model,
        tokenizer=tokenizer,
        name="FewShotCoT",
        examples=examples,
        zero_shot=False,
        verbose=True,
    )

    # New problem
    task = "A garden has 15 roses. The gardener plants 8 more roses and then removes 5 wilted ones. How many roses are in the garden now?"

    print(f"\nTask: {task}")
    print("-" * 80)

    response = agent.run(task, temperature=0.3)

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"\nReasoning:\n{response.reasoning}")
    print(f"\nFinal Answer: {response.content}")


def example_3_self_consistency():
    """Example 3: Self-Consistency CoT."""
    print("\n" + "=" * 80)
    print("Example 3: Self-Consistency CoT")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create self-consistency CoT agent
    agent = SelfConsistencyCoTAgent(
        model=model,
        tokenizer=tokenizer,
        name="SelfConsistencyCoT",
        num_samples=5,
        zero_shot=True,
        verbose=True,
    )

    # Problem where self-consistency helps
    task = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"

    print(f"\nTask: {task}")
    print("-" * 80)

    response = agent.run(task, temperature=0.7)

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"\nAll answers generated: {response.metadata.get('all_answers', [])}")
    print(f"\nVote counts: {response.metadata.get('vote_counts', {})}")
    print(f"\nMajority Answer: {response.content}")


def example_4_mathematical_reasoning():
    """Example 4: Complex mathematical reasoning."""
    print("\n" + "=" * 80)
    print("Example 4: Mathematical Reasoning")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create CoT agent
    agent = CoTAgent(
        model=model,
        tokenizer=tokenizer,
        zero_shot=True,
        verbose=False,
    )

    # Various math problems
    problems = [
        "What is 15% of 80?",
        "If a rectangle has length 12 and width 5, what is its perimeter?",
        "A number is doubled and then increased by 7 to get 23. What was the original number?",
        "Three friends split a bill of $45. How much does each person pay?",
    ]

    for problem in problems:
        print(f"\n{'─' * 80}")
        print(f"Problem: {problem}")
        print(f"{'─' * 80}")

        response = agent.run(problem, temperature=0.2)

        print(f"\nReasoning: {response.reasoning[:200]}...")
        print(f"Answer: {response.content}")


def example_5_logical_puzzles():
    """Example 5: Logical reasoning puzzles."""
    print("\n" + "=" * 80)
    print("Example 5: Logical Puzzles")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create CoT agent
    agent = CoTAgent(
        model=model,
        tokenizer=tokenizer,
        zero_shot=True,
        verbose=True,
        max_tokens=1000,
    )

    # Logic puzzles
    puzzles = [
        """All roses are flowers. Some flowers fade quickly.
Therefore, do all roses fade quickly? Explain your reasoning.""",
        """If it's raining, the ground is wet. The ground is wet.
Is it necessarily raining? Why or why not?""",
    ]

    for puzzle in puzzles:
        print(f"\n{'─' * 80}")
        print(f"Puzzle: {puzzle}")
        print(f"{'─' * 80}")

        response = agent.run(puzzle, temperature=0.3)

        print(f"\nAnswer: {response.content}")


def example_6_comparison():
    """Example 6: Compare zero-shot vs few-shot CoT."""
    print("\n" + "=" * 80)
    print("Example 6: Zero-Shot vs Few-Shot Comparison")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Problem to solve
    task = "A store sells apples for $2 each and oranges for $3 each. If someone spends $19, buying more apples than oranges, how many of each fruit did they buy?"

    # Zero-shot CoT
    print("\n" + "-" * 80)
    print("Zero-Shot CoT:")
    print("-" * 80)

    zero_shot_agent = CoTAgent(
        model=model, tokenizer=tokenizer, zero_shot=True, verbose=False
    )

    zero_shot_response = zero_shot_agent.run(task, temperature=0.3)

    print(f"Reasoning: {zero_shot_response.reasoning[:300]}...")
    print(f"Answer: {zero_shot_response.content}")

    # Few-shot CoT
    print("\n" + "-" * 80)
    print("Few-Shot CoT:")
    print("-" * 80)

    examples = [
        {
            "question": "Pencils cost $1 and erasers cost $2. If someone spends $10 buying 3 more pencils than erasers, how many of each do they buy?",
            "reasoning": "Let's say they buy x erasers. Then they buy x+3 pencils. Cost = 2x + 1(x+3) = 10. So 3x + 3 = 10, meaning 3x = 7, x = 2.33... Since we need whole numbers, they buy 2 erasers and 5 pencils (2×2 + 5×1 = 9) or 1 eraser and 4 pencils (1×2 + 4×1 = 6). Trying 2 erasers and 6 pencils: 2×2 + 6×1 = 10. ✓",
            "answer": "2 erasers and 6 pencils",
        }
    ]

    few_shot_agent = CoTAgent(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        zero_shot=False,
        verbose=False,
    )

    few_shot_response = few_shot_agent.run(task, temperature=0.3)

    print(f"Reasoning: {few_shot_response.reasoning[:300]}...")
    print(f"Answer: {few_shot_response.content}")


def main():
    """Run all examples."""
    print("=" * 80)
    print("SMLX Agent System - Chain-of-Thought Examples")
    print("=" * 80)

    examples = [
        ("Zero-Shot CoT", example_1_zero_shot_cot),
        ("Few-Shot CoT", example_2_few_shot_cot),
        ("Self-Consistency", example_3_self_consistency),
        ("Mathematical Reasoning", example_4_mathematical_reasoning),
        ("Logical Puzzles", example_5_logical_puzzles),
        ("Comparison", example_6_comparison),
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
    print("  ✓ Zero-shot CoT (Let's think step by step)")
    print("  ✓ Few-shot CoT (learning from examples)")
    print("  ✓ Self-Consistency (multiple samples + voting)")
    print("  ✓ Mathematical reasoning")
    print("  ✓ Logical puzzle solving")
    print("\nChain-of-Thought Benefits:")
    print("  - Improves accuracy on complex reasoning tasks")
    print("  - Makes reasoning transparent and interpretable")
    print("  - Helps catch errors through step-by-step thinking")
    print("\nReferences:")
    print("  - Wei et al. (2022) 'Chain-of-Thought Prompting Elicits Reasoning'")
    print("  - Wang et al. (2022) 'Self-Consistency Improves Chain of Thought'")


if __name__ == "__main__":
    main()
