#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
SMLX Server Client Example

Demonstrates how to interact with the SMLX Server using HTTP requests.
Shows text completion, chat, and streaming examples.

Prerequisites:
    Start the server first:
    $ python -m smlx.server.app

    Or with uvicorn:
    $ uvicorn smlx.server.app:app --host 0.0.0.0 --port 8000
"""

import json
import requests
from typing import Iterator


# Server configuration
SERVER_URL = "http://localhost:8000"


def test_health():
    """Test server health check."""
    print("=" * 70)
    print("1. Health Check")
    print("=" * 70)

    response = requests.get(f"{SERVER_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_list_models():
    """List available models."""
    print("\n" + "=" * 70)
    print("2. List Available Models")
    print("=" * 70)

    response = requests.get(f"{SERVER_URL}/v1/models")
    data = response.json()

    print(f"Found {len(data['data'])} models:")
    for model in data["data"]:
        print(f"  - {model['id']}")


def test_text_completion():
    """Test text completion endpoint."""
    print("\n" + "=" * 70)
    print("3. Text Completion")
    print("=" * 70)

    payload = {
        "model": "mlx-community/SmolLM2-135M-Instruct",
        "prompt": "Write a Python function to calculate factorial:",
        "max_tokens": 150,
        "temperature": 0.7,
    }

    print(f"Prompt: {payload['prompt']}")
    print("Generating...")

    response = requests.post(f"{SERVER_URL}/v1/completions", json=payload)
    data = response.json()

    print(f"\nResponse:")
    print(data["choices"][0]["text"])
    print(f"\nUsage: {data['usage']}")


def test_chat_completion():
    """Test chat completion endpoint."""
    print("\n" + "=" * 70)
    print("4. Chat Completion")
    print("=" * 70)

    payload = {
        "model": "mlx-community/SmolLM2-135M-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"},
        ],
        "max_tokens": 150,
        "temperature": 0.7,
    }

    print("Messages:")
    for msg in payload["messages"]:
        print(f"  {msg['role']}: {msg['content']}")

    print("\nGenerating...")

    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload)
    data = response.json()

    print(f"\nAssistant: {data['choices'][0]['message']['content']}")
    print(f"\nUsage: {data['usage']}")


def test_streaming_completion():
    """Test streaming text completion."""
    print("\n" + "=" * 70)
    print("5. Streaming Text Completion")
    print("=" * 70)

    payload = {
        "model": "mlx-community/SmolLM2-135M-Instruct",
        "prompt": "Explain quantum computing:",
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True,
    }

    print(f"Prompt: {payload['prompt']}")
    print("Streaming output:\n")

    response = requests.post(
        f"{SERVER_URL}/v1/completions",
        json=payload,
        stream=True,
    )

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if data["choices"][0]["text"]:
                        print(data["choices"][0]["text"], end="", flush=True)
                except json.JSONDecodeError:
                    pass

    print("\n")


def test_streaming_chat():
    """Test streaming chat completion."""
    print("\n" + "=" * 70)
    print("6. Streaming Chat Completion")
    print("=" * 70)

    payload = {
        "model": "mlx-community/SmolLM2-135M-Instruct",
        "messages": [
            {"role": "user", "content": "Write a short poem about AI:"},
        ],
        "max_tokens": 100,
        "temperature": 0.8,
        "stream": True,
    }

    print("User: Write a short poem about AI:")
    print("Assistant: ", end="", flush=True)

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        stream=True,
    )

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta:
                        print(delta["content"], end="", flush=True)
                except json.JSONDecodeError:
                    pass

    print("\n")


def test_multi_turn_chat():
    """Test multi-turn conversation."""
    print("\n" + "=" * 70)
    print("7. Multi-turn Chat Conversation")
    print("=" * 70)

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
    ]

    # Turn 1
    messages.append({"role": "user", "content": "What is Python?"})

    payload = {
        "model": "mlx-community/SmolLM2-135M-Instruct",
        "messages": messages,
        "max_tokens": 80,
        "temperature": 0.7,
    }

    print("User: What is Python?")
    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload)
    assistant_msg = response.json()["choices"][0]["message"]["content"]
    print(f"Assistant: {assistant_msg}")

    messages.append({"role": "assistant", "content": assistant_msg})

    # Turn 2
    messages.append({"role": "user", "content": "Give me a code example"})

    payload["messages"] = messages

    print("\nUser: Give me a code example")
    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload)
    assistant_msg = response.json()["choices"][0]["message"]["content"]
    print(f"Assistant: {assistant_msg}")


def test_different_models():
    """Test different model configurations."""
    print("\n" + "=" * 70)
    print("8. Testing Different Models")
    print("=" * 70)

    models = [
        "mlx-community/SmolLM2-135M-Instruct",
        "mlx-community/SmolLM2-360M-Instruct",
    ]

    prompt = "Hello, world!"

    for model_id in models:
        print(f"\nModel: {model_id}")
        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": 30,
            "temperature": 0.0,  # Greedy for consistency
        }

        try:
            response = requests.post(f"{SERVER_URL}/v1/completions", json=payload)
            data = response.json()
            print(f"Output: {data['choices'][0]['text']}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    print("=" * 70)
    print("SMLX Server Client Examples")
    print("=" * 70)
    print("\nMake sure the server is running:")
    print("  $ python -m smlx.server.app")
    print()

    try:
        # Run all examples
        test_health()
        test_list_models()
        test_text_completion()
        test_chat_completion()
        test_streaming_completion()
        test_streaming_chat()
        test_multi_turn_chat()
        test_different_models()

        print("\n" + "=" * 70)
        print("✅ All Examples Complete!")
        print("=" * 70)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to server")
        print("Make sure the server is running on http://localhost:8000")
        print("\nStart the server with:")
        print("  $ python -m smlx.server.app")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
