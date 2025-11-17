#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
OpenAI-Compatible Client Example

Demonstrates that SMLX Server is compatible with OpenAI's Python client.
You can drop SMLX Server into existing OpenAI code with minimal changes!

Prerequisites:
    pip install openai
    python -m smlx.server.app
"""

try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI client not installed")
    print("Install with: pip install openai")
    exit(1)


def main():
    print("=" * 70)
    print("OpenAI-Compatible Client Example")
    print("=" * 70)

    # Create OpenAI client pointing to SMLX Server
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # SMLX Server doesn't require auth (configure as needed)
    )

    # Example 1: Text Completion
    print("\n1. Text Completion")
    print("-" * 70)

    completion = client.completions.create(
        model="mlx-community/SmolLM2-135M-Instruct",
        prompt="Write a haiku about coding:",
        max_tokens=50,
        temperature=0.7,
    )

    print(f"Prompt: Write a haiku about coding:")
    print(f"Response:\n{completion.choices[0].text}")

    # Example 2: Chat Completion
    print("\n2. Chat Completion")
    print("-" * 70)

    chat_completion = client.chat.completions.create(
        model="mlx-community/SmolLM2-135M-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain recursion in simple terms."},
        ],
        max_tokens=100,
        temperature=0.7,
    )

    print("User: Explain recursion in simple terms.")
    print(f"Assistant: {chat_completion.choices[0].message.content}")

    # Example 3: Streaming
    print("\n3. Streaming Chat Completion")
    print("-" * 70)

    print("User: Tell me a joke.")
    print("Assistant: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="mlx-community/SmolLM2-135M-Instruct",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        max_tokens=80,
        temperature=0.8,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")

    # Example 4: Multi-turn Conversation
    print("\n4. Multi-turn Conversation")
    print("-" * 70)

    messages = [
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": "What are decorators?"},
    ]

    # First turn
    response1 = client.chat.completions.create(
        model="mlx-community/SmolLM2-135M-Instruct",
        messages=messages,
        max_tokens=80,
    )

    print("User: What are decorators?")
    print(f"Assistant: {response1.choices[0].message.content}")

    # Add to conversation history
    messages.append({
        "role": "assistant",
        "content": response1.choices[0].message.content,
    })
    messages.append({"role": "user", "content": "Show me an example"})

    # Second turn
    response2 = client.chat.completions.create(
        model="mlx-community/SmolLM2-135M-Instruct",
        messages=messages,
        max_tokens=100,
    )

    print("\nUser: Show me an example")
    print(f"Assistant: {response2.choices[0].message.content}")

    # Example 5: List Models
    print("\n5. List Available Models")
    print("-" * 70)

    models = client.models.list()
    print(f"Found {len(models.data)} models:")
    for model in models.data:
        print(f"  - {model.id}")

    print("\n" + "=" * 70)
    print("✅ OpenAI Compatibility Verified!")
    print("=" * 70)

    print("\n💡 Key Takeaway:")
    print("SMLX Server implements the OpenAI API specification,")
    print("so you can use the official OpenAI Python client!")

    print("\n📝 Migration from OpenAI to SMLX:")
    print("  1. Change base_url to your SMLX server")
    print("  2. Update model names to SMLX models")
    print("  3. Everything else stays the same!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. SMLX server is running (python -m smlx.server.app)")
        print("  2. OpenAI client is installed (pip install openai)")
