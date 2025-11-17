#!/usr/bin/env python3
# Copyright � 2025 SMLX Project

"""
Agent System for SMLX.

Provides autonomous agents that can:
- Use language models for reasoning
- Call tools to perform actions
- Maintain conversation history
- Execute multi-step reasoning (ReAct, CoT)

Available Agents:
    - BaseAgent: Abstract base class for all agents
    - LLMAgent: Basic agent with language model
    - ReActAgent: Reasoning + Acting agent with tool use
    - CoTAgent: Chain-of-Thought reasoning agent
    - SelfConsistencyCoTAgent: CoT with multiple samples and voting

Tools:
    - Tool: Base class for tools
    - FunctionTool: Wrap Python functions as tools
    - ToolRegistry: Manage and execute tools
    - Built-in tools: calculator, get_current_time, search_wikipedia

Memory:
    - SimpleMemory: In-memory storage
    - PersistentMemory: Disk-persisted storage
    - SummarizedMemory: Automatic summarization
    - ConversationMemory: Conversation-specific memory

Example:
    >>> from smlx.agents import ReActAgent, create_default_registry
    >>> from smlx.models.SmolLM2_135M import load
    >>>
    >>> # Load model
    >>> model, tokenizer = load()
    >>>
    >>> # Create agent with tools
    >>> tools = create_default_registry()
    >>> agent = ReActAgent(model, tokenizer, tools=tools)
    >>>
    >>> # Run agent on task
    >>> response = agent.run("What is 25 * 37?")
    >>> print(response.content)

Example (CoT):
    >>> from smlx.agents import CoTAgent
    >>> agent = CoTAgent(model, tokenizer)
    >>> response = agent.run("If John has 5 apples and buys 3 more, how many does he have?")
    >>> print(response.reasoning)
    >>> print(response.content)
"""

from .base import AgentResponse, BaseAgent, LLMAgent, Message
from .cot import CoTAgent, SelfConsistencyCoTAgent
from .memory import (
    ConversationMemory,
    Memory,
    MemoryManager,
    PersistentMemory,
    SimpleMemory,
    SummarizedMemory,
)
from .react import ReActAgent
from .tools import (
    FunctionTool,
    Tool,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    calculator,
    create_default_registry,
    get_current_time,
    search_wikipedia,
)

__version__ = "0.1.0"

__all__ = [
    # Core agents
    "BaseAgent",
    "LLMAgent",
    "ReActAgent",
    "CoTAgent",
    "SelfConsistencyCoTAgent",
    # Agent types
    "Message",
    "AgentResponse",
    # Tools
    "Tool",
    "FunctionTool",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "calculator",
    "get_current_time",
    "search_wikipedia",
    "create_default_registry",
    # Memory
    "Memory",
    "MemoryManager",
    "SimpleMemory",
    "PersistentMemory",
    "SummarizedMemory",
    "ConversationMemory",
]
