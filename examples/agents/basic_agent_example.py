#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Basic Agent Examples.

Demonstrates the fundamental agent capabilities:
1. Basic LLM agent with conversation
2. Multi-turn conversations with history
3. Custom system prompts
4. Memory management
"""

import sys
from pathlib import Path

# Add smlx to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smlx.models.SmolLM2_135M import load
from smlx.agents import LLMAgent, SimpleMemory, ConversationMemory


def example_1_basic_agent():
    """Example 1: Basic agent with simple Q&A."""
    print("\n" + "=" * 80)
    print("Example 1: Basic LLM Agent")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create basic agent
    agent = LLMAgent(
        model=model,
        tokenizer=tokenizer,
        name="Assistant",
        system_prompt="You are a helpful AI assistant.",
        verbose=True,
    )

    # Run agent on simple tasks
    tasks = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence.",
        "Write a haiku about coding.",
    ]

    for task in tasks:
        print(f"\n{'─' * 80}")
        print(f"Task: {task}")
        print(f"{'─' * 80}")

        response = agent.run(task, max_tokens=150)

        if response.success:
            print(f"\nResponse: {response.content}")
        else:
            print(f"\nError: {response.error}")


def example_2_conversation_history():
    """Example 2: Multi-turn conversation with history."""
    print("\n" + "=" * 80)
    print("Example 2: Multi-turn Conversation")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create agent
    agent = LLMAgent(
        model=model,
        tokenizer=tokenizer,
        name="ChatBot",
        system_prompt="You are a friendly chatbot. Remember the conversation context.",
        verbose=False,
    )

    # Multi-turn conversation
    conversation = [
        "My name is Alice.",
        "What's my name?",
        "I like pizza and ice cream.",
        "What are my favorite foods?",
    ]

    for turn in conversation:
        print(f"\nUser: {turn}")
        response = agent.run(turn, max_tokens=100)
        print(f"Assistant: {response.content}")


def example_3_custom_prompts():
    """Example 3: Agents with custom system prompts."""
    print("\n" + "=" * 80)
    print("Example 3: Custom System Prompts")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Different agent personalities
    agents = [
        {
            "name": "Poet",
            "prompt": "You are a poetic AI that responds in verse and metaphor.",
        },
        {
            "name": "Scientist",
            "prompt": "You are a scientific AI that explains things with precision and clarity.",
        },
        {
            "name": "Philosopher",
            "prompt": "You are a philosophical AI that ponders deep questions about existence.",
        },
    ]

    task = "What is consciousness?"

    for agent_config in agents:
        print(f"\n{'─' * 80}")
        print(f"Agent: {agent_config['name']}")
        print(f"{'─' * 80}")

        agent = LLMAgent(
            model=model,
            tokenizer=tokenizer,
            name=agent_config["name"],
            system_prompt=agent_config["prompt"],
        )

        response = agent.run(task, max_tokens=200)
        print(f"\nResponse: {response.content}")


def example_4_memory_management():
    """Example 4: Using memory systems."""
    print("\n" + "=" * 80)
    print("Example 4: Memory Management")
    print("=" * 80)

    # Create simple memory
    memory = SimpleMemory(max_memories=5)

    # Add some memories
    memories_to_add = [
        "User likes science fiction",
        "User's favorite color is blue",
        "User is learning Python",
        "User has a cat named Whiskers",
        "User enjoys hiking",
        "User works in tech",
    ]

    print("\nAdding memories...")
    for mem in memories_to_add:
        memory.add(mem, importance=0.8)
        print(f"  Added: {mem}")

    # Query recent memories
    print(f"\nTotal memories stored: {len(memory)}")
    print(f"(Max memories setting: {memory.max_memories})")

    recent = memory.get_recent(3)
    print("\nMost recent memories:")
    for i, mem in enumerate(recent, 1):
        print(f"  {i}. {mem.content}")

    # Search memories
    query = "cat"
    results = memory.search(query)
    print(f"\nSearching for '{query}':")
    for result in results:
        print(f"  - {result.content}")

    # Conversation memory
    print("\n" + "-" * 80)
    print("Conversation Memory:")
    print("-" * 80)

    conv_memory = ConversationMemory(max_turns=3)

    conv_memory.add_turn(
        "What's the weather like?", "I don't have access to weather data."
    )
    conv_memory.add_turn("Tell me a joke.", "Why did the chicken cross the road?")
    conv_memory.add_turn("That's not funny!", "I'll try to do better next time!")

    print("\nConversation context:")
    print(conv_memory.get_context())

    print(f"\nTotal turns: {len(conv_memory)}")


def example_5_agent_introspection():
    """Example 5: Inspecting agent state."""
    print("\n" + "=" * 80)
    print("Example 5: Agent Introspection")
    print("=" * 80)

    # Load model
    print("\nLoading SmolLM2-135M model...")
    model, tokenizer = load()

    # Create agent
    agent = LLMAgent(
        model=model,
        tokenizer=tokenizer,
        name="IntrospectiveAgent",
        verbose=False,
    )

    # Have conversation
    agent.run("Hello!")
    agent.run("What can you help me with?")
    agent.run("Thanks!")

    # Inspect agent state
    print("\nAgent State:")
    print(f"  Name: {agent.name}")
    print(f"  Total messages: {len(agent.messages)}")

    print("\nMessage History:")
    for msg in agent.messages:
        print(f"  [{msg.role}] {msg.content[:50]}...")

    # Get specific messages
    user_messages = agent.get_messages(role="user")
    print(f"\nUser messages: {len(user_messages)}")

    assistant_messages = agent.get_messages(role="assistant")
    print(f"Assistant messages: {len(assistant_messages)}")

    # Clear history
    print("\nClearing history (keeping system message)...")
    agent.clear_history(keep_system=True)
    print(f"Messages after clear: {len(agent.messages)}")


def main():
    """Run all examples."""
    print("=" * 80)
    print("SMLX Agent System - Basic Examples")
    print("=" * 80)

    examples = [
        ("Basic Agent", example_1_basic_agent),
        ("Conversation History", example_2_conversation_history),
        ("Custom Prompts", example_3_custom_prompts),
        ("Memory Management", example_4_memory_management),
        ("Agent Introspection", example_5_agent_introspection),
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
    print("  ✓ Basic LLM agent creation")
    print("  ✓ Multi-turn conversations")
    print("  ✓ Custom system prompts")
    print("  ✓ Memory management (simple & conversation)")
    print("  ✓ Agent state introspection")


if __name__ == "__main__":
    main()
