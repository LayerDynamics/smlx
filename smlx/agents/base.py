#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Base Agent implementation.

Provides the foundation for building autonomous agents that can:
- Interact with language models
- Use tools
- Maintain conversation history
- Execute multi-step reasoning
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Message:
    """A message in the agent's conversation history."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


@dataclass
class AgentResponse:
    """Response from an agent execution."""

    content: str
    reasoning: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all agents.

    Agents are autonomous systems that can:
    1. Maintain conversation state
    2. Interact with language models
    3. Use tools to accomplish tasks
    4. Execute multi-step reasoning

    Subclasses should implement the `run()` method to define
    agent-specific behavior (e.g., ReAct, CoT, etc.).
    """

    def __init__(
        self,
        name: str = "Agent",
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """Initialize base agent.

        Args:
            name: Agent name
            system_prompt: System prompt for the agent
            max_iterations: Maximum reasoning iterations
            verbose: Whether to print debug information
        """
        self.name = name
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Conversation history
        self.messages: list[Message] = []

        # Add system message if provided
        if self.system_prompt:
            self.add_message("system", self.system_prompt)

    def _default_system_prompt(self) -> str:
        """Default system prompt for the agent."""
        return f"You are {self.name}, a helpful AI assistant."

    def add_message(
        self, role: str, content: str, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Add a message to conversation history.

        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            metadata: Optional metadata for the message
        """
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)

        if self.verbose:
            print(f"[{self.name}] {message}")

    def get_messages(self, role: Optional[str] = None) -> list[Message]:
        """Get messages from history, optionally filtered by role.

        Args:
            role: Optional role to filter by

        Returns:
            list of messages
        """
        if role:
            return [m for m in self.messages if m.role == role]
        return self.messages

    def clear_history(self, keep_system: bool = True) -> None:
        """Clear conversation history.

        Args:
            keep_system: Whether to keep system messages
        """
        if keep_system:
            self.messages = [m for m in self.messages if m.role == "system"]
        else:
            self.messages = []

    def format_messages_for_model(self) -> list[dict[str, str]]:
        """Format messages for model input.

        Returns:
            list of message dictionaries with 'role' and 'content'
        """
        return [{"role": m.role, "content": m.content} for m in self.messages]

    @abstractmethod
    def run(self, task: str, **kwargs) -> AgentResponse:
        """Execute the agent on a task.

        Args:
            task: Task description or user query
            **kwargs: Additional arguments

        Returns:
            AgentResponse with results
        """
        pass

    def __call__(self, task: str, **kwargs) -> AgentResponse:
        """Make agent callable."""
        return self.run(task, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class LLMAgent(BaseAgent):
    """Agent that uses a language model for generation.

    This is a concrete base class that can be used directly or
    subclassed for more specialized agent behaviors.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        name: str = "LLMAgent",
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        max_tokens: int = 500,
        temperature: float = 0.7,
        verbose: bool = False,
    ):
        """Initialize LLM agent.

        Args:
            model: Language model
            tokenizer: Tokenizer
            name: Agent name
            system_prompt: System prompt
            max_iterations: Maximum reasoning iterations
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            verbose: Verbose output
        """
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the language model.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        # Import generation utilities
        try:
            from smlx.models.SmolLM2_135M import generate
        except ImportError as e:
            raise ImportError(
                "Could not import generation utilities. Make sure SmolLM2 models are available."
            ) from e

        # Merge kwargs with defaults
        gen_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        gen_kwargs.update(kwargs)

        # Generate response
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            **gen_kwargs,
        )

        return response

    def run(self, task: str, **kwargs) -> AgentResponse:
        """Execute the agent on a task.

        Basic implementation that generates a single response.
        Subclasses can override for more complex behaviors.

        Args:
            task: Task or query
            **kwargs: Additional arguments

        Returns:
            AgentResponse with results
        """
        # Add user message
        self.add_message("user", task)

        # Format messages for model
        messages = self.format_messages_for_model()

        # Build prompt from messages
        prompt = self._build_prompt(messages)

        # Generate response
        try:
            response = self.generate(prompt, **kwargs)

            # Add assistant message
            self.add_message("assistant", response)

            return AgentResponse(
                content=response,
                messages=self.get_messages(),
                success=True,
            )

        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            if self.verbose:
                print(f"[{self.name}] {error_msg}")

            return AgentResponse(
                content="",
                success=False,
                error=error_msg,
                messages=self.get_messages(),
            )

    def _build_prompt(self, messages: list[dict[str, str]]) -> str:
        """Build prompt string from messages.

        Args:
            messages: list of message dictionaries

        Returns:
            Formatted prompt string
        """
        # Try to use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return prompt
            except Exception:
                pass

        # Fallback to simple formatting
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}")

        prompt_parts.append("assistant:")
        return "\n".join(prompt_parts)


__all__ = ["Message", "AgentResponse", "BaseAgent", "LLMAgent"]
