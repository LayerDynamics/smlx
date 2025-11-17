#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Server Quantization Example

Demonstrates how to use the SMLX FastAPI server with automatic quantization.

The server supports quantization via environment variables:
- SMLX_AUTO_QUANTIZE: Quantization method (auto, 4bit, 8bit, gptq, awq, dwq)
- SMLX_QUANTIZE_BITS: Bits per weight (2, 3, 4, 6, 8)
- SMLX_QUANTIZE_GROUP_SIZE: Quantization group size (32, 64, 128)

Usage:
    1. Start server with quantization:
       $ SMLX_AUTO_QUANTIZE=4bit python -m smlx.server.app

    2. Run this example to test quantized inference:
       $ python examples/server/quantization_example.py
"""

import requests
import time


def test_chat_completion(base_url: str = "http://localhost:8000"):
    """Test chat completion endpoint."""
    print("=" * 70)
    print("Testing Chat Completion with Quantized Model")
    print("=" * 70)

    # Chat completion request
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "SmolLM2-135M",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in one sentence."},
            ],
            "max_tokens": 50,
            "temperature": 0.7,
        },
    )

    if response.status_code == 200:
        result = response.json()
        print(f"\nModel: {result['model']}")
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Tokens: {result['usage']['total_tokens']}")
        print(f"Time: {result.get('timing', {}).get('total_time', 'N/A')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_text_completion(base_url: str = "http://localhost:8000"):
    """Test text completion endpoint."""
    print("\n" + "=" * 70)
    print("Testing Text Completion with Quantized Model")
    print("=" * 70)

    # Text completion request
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": "SmolLM2-135M",
            "prompt": "The future of AI is",
            "max_tokens": 30,
            "temperature": 0.7,
        },
    )

    if response.status_code == 200:
        result = response.json()
        print(f"\nModel: {result['model']}")
        print(f"Prompt: The future of AI is")
        print(f"Completion: {result['choices'][0]['text']}")
        print(f"Tokens: {result['usage']['total_tokens']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def check_memory_status(base_url: str = "http://localhost:8000"):
    """Check server memory status."""
    print("\n" + "=" * 70)
    print("Server Memory Status")
    print("=" * 70)

    response = requests.get(f"{base_url}/memory")

    if response.status_code == 200:
        memory = response.json()
        print(f"\nActive memory:  {memory['active_gb']:.2f} GB")
        print(f"Cache memory:   {memory['cache_gb']:.2f} GB")
        print(f"Total memory:   {memory['total_gb']:.2f} GB")
        print(f"Max memory:     {memory['max_gb']:.2f} GB")
        print(f"Utilization:    {memory['utilization']*100:.1f}%")
        print(f"Watchdog:       {'Enabled' if memory['watchdog_enabled'] else 'Disabled'}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def main():
    """Main example entry point."""
    base_url = "http://localhost:8000"

    print("\n" + "=" * 70)
    print("SMLX Server Quantization Example")
    print("=" * 70)
    print("\nThis example demonstrates inference with quantized models.")
    print("Make sure the server is running with quantization enabled:")
    print("  $ SMLX_AUTO_QUANTIZE=4bit python -m smlx.server.app")
    print("")

    # Wait for server to be ready
    print("Checking server health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        if response.status_code == 200:
            print("✓ Server is running\n")
        else:
            print("✗ Server not healthy")
            return
    except requests.exceptions.RequestException:
        print("✗ Server not reachable. Please start the server first:")
        print("  $ SMLX_AUTO_QUANTIZE=4bit python -m smlx.server.app")
        return

    # Run tests
    test_chat_completion(base_url)
    time.sleep(0.5)

    test_text_completion(base_url)
    time.sleep(0.5)

    check_memory_status(base_url)

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print("\nQuantization Methods You Can Try:")
    print("  SMLX_AUTO_QUANTIZE=4bit     - Fast 4-bit quantization")
    print("  SMLX_AUTO_QUANTIZE=8bit     - High-quality 8-bit quantization")
    print("  SMLX_AUTO_QUANTIZE=gptq     - GPTQ 4-bit (best quality)")
    print("  SMLX_AUTO_QUANTIZE=awq      - AWQ activation-aware")
    print("  SMLX_AUTO_QUANTIZE=auto     - Automatic selection")
    print("\nWith custom config:")
    print("  SMLX_AUTO_QUANTIZE=gptq \\")
    print("  SMLX_QUANTIZE_BITS=4 \\")
    print("  SMLX_QUANTIZE_GROUP_SIZE=64 \\")
    print("  python -m smlx.server.app")


if __name__ == "__main__":
    main()
