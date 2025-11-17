#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Multi-Agent System Examples.

Demonstrates how to:
1. Create multiple specialized agents
2. Combine different agent types (ReAct + CoT)
3. Build agent pipelines
4. Implement agent collaboration
5. Create a multi-agent workflow
"""

import sys
from pathlib import Path

# Add smlx to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smlx.agents import (
    CoTAgent,
    LLMAgent,
    ReActAgent,
    create_default_registry,
)
from smlx.models.SmolLM2_135M import load


def example_1_specialized_agents():
    """Example 1: Multiple specialized agents for different tasks."""
    print("\n" + "=" * 80)
    print("Example 1: Specialized Agents")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create specialized agents
    agents = {
        "summarizer": LLMAgent(
            model=model,
            tokenizer=tokenizer,
            name="Summarizer",
            system_prompt="You are an expert at creating concise summaries. Extract key points and present them clearly.",
        ),
        "translator": LLMAgent(
            model=model,
            tokenizer=tokenizer,
            name="Translator",
            system_prompt="You are a language expert who translates text accurately while preserving meaning.",
        ),
        "critic": LLMAgent(
            model=model,
            tokenizer=tokenizer,
            name="Critic",
            system_prompt="You are a constructive critic who provides thoughtful feedback and suggestions for improvement.",
        ),
    }

    # Original text
    text = """Artificial intelligence is transforming many industries. Machine learning
algorithms can now recognize patterns in data, make predictions, and even create
content. However, these systems also raise important questions about bias, privacy,
and the future of work."""

    # Process with different agents
    print("\nOriginal Text:")
    print(text)

    # Summarize
    print("\n" + "-" * 80)
    print("Summarizer Agent:")
    summary_response = agents["summarizer"].run(
        f"Summarize this text in one sentence: {text}", max_tokens=100
    )
    print(f"Summary: {summary_response.content}")

    # Translate (to simple language)
    print("\n" + "-" * 80)
    print("Translator Agent:")
    translate_response = agents["translator"].run(
        f"Explain this in simple terms for a 10-year-old: {text}", max_tokens=150
    )
    print(f"Simple Version: {translate_response.content}")

    # Critique
    print("\n" + "-" * 80)
    print("Critic Agent:")
    critic_response = agents["critic"].run(
        f"Provide constructive feedback on this text: {text}", max_tokens=150
    )
    print(f"Feedback: {critic_response.content}")


def example_2_agent_pipeline():
    """Example 2: Chain agents in a pipeline."""
    print("\n" + "=" * 80)
    print("Example 2: Agent Pipeline")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create pipeline: Research → Analyze → Summarize
    research_agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        tools=create_default_registry(),
        name="Researcher",
        verbose=False,
    )

    analyze_agent = CoTAgent(
        model=model,
        tokenizer=tokenizer,
        name="Analyzer",
        zero_shot=True,
        verbose=False,
    )

    summary_agent = LLMAgent(
        model=model,
        tokenizer=tokenizer,
        name="Summarizer",
        system_prompt="Create clear, concise summaries.",
    )

    # Initial query
    query = "What is the current time?"

    print(f"\nInitial Query: {query}")

    # Stage 1: Research
    print("\n" + "-" * 80)
    print("Stage 1: Research (ReAct Agent)")
    print("-" * 80)

    research_result = research_agent.run(query, temperature=0.3)
    print(f"Research Result: {research_result.content}")

    # Stage 2: Analyze
    print("\n" + "-" * 80)
    print("Stage 2: Analysis (CoT Agent)")
    print("-" * 80)

    analysis_prompt = f"Analyze this information and explain its significance: {research_result.content}"
    analysis_result = analyze_agent.run(analysis_prompt, temperature=0.3)
    print(f"Analysis: {analysis_result.content[:200]}...")

    # Stage 3: Summarize
    print("\n" + "-" * 80)
    print("Stage 3: Summary (LLM Agent)")
    print("-" * 80)

    summary_prompt = f"Create a one-sentence summary: {analysis_result.content}"
    summary_result = summary_agent.run(summary_prompt, max_tokens=100)
    print(f"Final Summary: {summary_result.content}")


def example_3_collaborative_agents():
    """Example 3: Agents working together."""
    print("\n" + "=" * 80)
    print("Example 3: Collaborative Agents")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create complementary agents
    creative_agent = LLMAgent(
        model=model,
        tokenizer=tokenizer,
        name="Creative",
        system_prompt="You are creative and come up with innovative ideas. Think outside the box.",
    )

    practical_agent = LLMAgent(
        model=model,
        tokenizer=tokenizer,
        name="Practical",
        system_prompt="You are practical and focus on feasibility. Consider real-world constraints.",
    )

    # Problem to solve
    problem = "How can we make studying more engaging for students?"

    print(f"\nProblem: {problem}")

    # Creative ideas
    print("\n" + "-" * 80)
    print("Creative Agent:")
    creative_response = creative_agent.run(
        f"Suggest 3 creative solutions to: {problem}", max_tokens=200
    )
    print(f"Ideas:\n{creative_response.content}")

    # Practical evaluation
    print("\n" + "-" * 80)
    print("Practical Agent:")
    practical_response = practical_agent.run(
        f"Evaluate these ideas for feasibility: {creative_response.content}",
        max_tokens=200,
    )
    print(f"Evaluation:\n{practical_response.content}")


def example_4_reasoning_debate():
    """Example 4: Multiple CoT agents debating."""
    print("\n" + "=" * 80)
    print("Example 4: Multi-Agent Reasoning Debate")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create two CoT agents with different approaches
    agent_a = CoTAgent(
        model=model,
        tokenizer=tokenizer,
        name="Agent-A",
        zero_shot=True,
        verbose=False,
        temperature=0.7,
    )

    agent_b = CoTAgent(
        model=model,
        tokenizer=tokenizer,
        name="Agent-B",
        zero_shot=True,
        verbose=False,
        temperature=0.3,
    )

    # Problem to solve
    problem = "Is it better to work on one big project or multiple small projects?"

    print(f"\nDebate Topic: {problem}")

    # Agent A's perspective
    print("\n" + "-" * 80)
    print("Agent A's Reasoning:")
    response_a = agent_a.run(problem, max_tokens=300)
    print(f"{response_a.content[:300]}...")

    # Agent B's perspective
    print("\n" + "-" * 80)
    print("Agent B's Reasoning:")
    response_b = agent_b.run(problem, max_tokens=300)
    print(f"{response_b.content[:300]}...")

    # Synthesis
    print("\n" + "-" * 80)
    print("Synthesis:")
    print("Both agents provided valid reasoning from different angles.")


def example_5_task_decomposition():
    """Example 5: Using agents to decompose complex tasks."""
    print("\n" + "=" * 80)
    print("Example 5: Task Decomposition")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create planner and executor agents
    planner_agent = CoTAgent(
        model=model,
        tokenizer=tokenizer,
        name="Planner",
        system_prompt="You are a strategic planner who breaks down complex tasks into clear steps.",
        zero_shot=True,
    )

    executor_agent = ReActAgent(
        model=model,
        tokenizer=tokenizer,
        tools=create_default_registry(),
        name="Executor",
        verbose=False,
    )

    # Complex task
    task = "Calculate the total cost if someone buys 3 items at $12.50 each, with 8% tax"

    print(f"\nComplex Task: {task}")

    # Step 1: Plan
    print("\n" + "-" * 80)
    print("Planning Phase (CoT Agent):")
    plan = planner_agent.run(
        f"Break down this task into steps: {task}", max_tokens=200
    )
    print(f"Plan:\n{plan.content}")

    # Step 2: Execute
    print("\n" + "-" * 80)
    print("Execution Phase (ReAct Agent):")
    result = executor_agent.run(task, temperature=0.2)
    print(f"Result: {result.content}")


def example_6_multi_agent_workflow():
    """Example 6: Complete multi-agent workflow."""
    print("\n" + "=" * 80)
    print("Example 6: Complete Multi-Agent Workflow")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create a team of agents
    team = {
        "intake": LLMAgent(
            model=model,
            tokenizer=tokenizer,
            name="Intake",
            system_prompt="You clarify user requests and extract key requirements.",
        ),
        "solver": ReActAgent(
            model=model,
            tokenizer=tokenizer,
            tools=create_default_registry(),
            name="Solver",
            verbose=False,
        ),
        "verifier": CoTAgent(
            model=model,
            tokenizer=tokenizer,
            name="Verifier",
            system_prompt="You verify results step-by-step and check for errors.",
            zero_shot=True,
        ),
        "presenter": LLMAgent(
            model=model,
            tokenizer=tokenizer,
            name="Presenter",
            system_prompt="You present results clearly and professionally.",
        ),
    }

    # User request
    user_request = "I need to calculate 144 divided by 12, then multiply that by 5"

    print(f"\nUser Request: {user_request}")

    # Workflow
    print("\n" + "-" * 80)
    print("1. Intake Agent - Clarifying Request:")
    intake_result = team["intake"].run(
        f"Clarify this request: {user_request}", max_tokens=100
    )
    print(f"Clarification: {intake_result.content}")

    print("\n" + "-" * 80)
    print("2. Solver Agent - Computing Result:")
    solver_result = team["solver"].run(user_request, temperature=0.2)
    print(f"Computation: {solver_result.content}")

    print("\n" + "-" * 80)
    print("3. Verifier Agent - Checking Result:")
    verifier_result = team["verifier"].run(
        f"Verify this calculation is correct: {solver_result.content}", max_tokens=200
    )
    print(f"Verification: {verifier_result.content[:150]}...")

    print("\n" + "-" * 80)
    print("4. Presenter Agent - Final Presentation:")
    presenter_result = team["presenter"].run(
        f"Present this result professionally: {solver_result.content}", max_tokens=100
    )
    print(f"Final Output: {presenter_result.content}")


def main():
    """Run all examples."""
    print("=" * 80)
    print("SMLX Agent System - Multi-Agent Examples")
    print("=" * 80)

    examples = [
        ("Specialized Agents", example_1_specialized_agents),
        ("Agent Pipeline", example_2_agent_pipeline),
        ("Collaborative Agents", example_3_collaborative_agents),
        ("Reasoning Debate", example_4_reasoning_debate),
        ("Task Decomposition", example_5_task_decomposition),
        ("Multi-Agent Workflow", example_6_multi_agent_workflow),
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
    print("  ✓ Specialized agents for different tasks")
    print("  ✓ Agent pipelines and workflows")
    print("  ✓ Collaborative problem solving")
    print("  ✓ Multi-agent reasoning and debate")
    print("  ✓ Task decomposition with planner/executor pattern")
    print("  ✓ Complete multi-agent workflow")
    print("\nMulti-Agent Benefits:")
    print("  - Specialization improves task performance")
    print("  - Collaboration enables complex problem solving")
    print("  - Verification reduces errors")
    print("  - Modular design allows flexibility")


if __name__ == "__main__":
    main()
