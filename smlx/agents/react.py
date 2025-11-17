#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
ReAct Agent Implementation.

ReAct (Reasoning + Acting) is a paradigm where agents:
1. Think/Reason about the current state and what to do next
2. Act by calling tools or providing final answers
3. Observe the results of actions
4. Repeat until task is complete

This creates a loop: Thought → Action → Observation → Thought → ...

Reference:
    Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models"
    https://arxiv.org/abs/2210.03629
"""

import re
from typing import Any, Optional

from .base import AgentResponse, LLMAgent
from .tools import ToolRegistry, ToolResult, create_default_registry


class ReActAgent(LLMAgent):
    """ReAct (Reasoning + Acting) agent.

    This agent follows the ReAct paradigm:
    1. Thought: Reason about the current state
    2. Action: Call a tool or provide final answer
    3. Observation: Observe the result
    4. Repeat until done

    The agent uses a specific prompt format to guide the model
    through the reasoning-acting loop.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        tools: Optional[ToolRegistry] = None,
        name: str = "ReActAgent",
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        max_tokens: int = 500,
        temperature: float = 0.7,
        verbose: bool = False,
    ):
        """Initialize ReAct agent.

        Args:
            model: Language model
            tokenizer: Tokenizer
            tools: Tool registry (uses default if None)
            name: Agent name
            system_prompt: Custom system prompt
            max_iterations: Maximum reasoning iterations
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            verbose: Verbose output
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            name=name,
            system_prompt=system_prompt or self._create_react_prompt(tools),
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
        )

        self.tools = tools or create_default_registry()

    def _default_system_prompt(self) -> str:
        """Override to provide ReAct-specific prompt."""
        return self._create_react_prompt(self.tools)

    def _create_react_prompt(self, tools: Optional[ToolRegistry]) -> str:
        """Create ReAct system prompt.

        Args:
            tools: Tool registry

        Returns:
            Formatted system prompt
        """
        tool_descriptions = ""
        if tools and len(tools) > 0:
            tool_descriptions = "\n\nAvailable tools:\n" + tools.get_tool_descriptions()

        prompt = f"""You are {self.name}, an agent that uses the ReAct (Reasoning + Acting) framework to solve problems.

You should follow this format:

Question: the input question or task
Thought: your reasoning about what to do next
Action: the action to take (either call a tool or provide final answer)
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: final reasoning
Answer: the final answer to the question

When calling tools, use this format:
Action: tool_name(param1="value1", param2="value2")

When you're ready to provide the final answer, use:
Action: Answer(result="your final answer here")
{tool_descriptions}

Important guidelines:
- Always start with a Thought before an Action
- Use tools when you need external information or computation
- Break complex tasks into smaller steps
- When you have enough information, provide the final Answer
- Be concise in your thoughts and observations"""

        return prompt

    def run(self, task: str, **kwargs) -> AgentResponse:
        """Execute ReAct loop on a task.

        Args:
            task: Task or question to solve
            **kwargs: Additional generation arguments

        Returns:
            AgentResponse with results
        """
        # Add user task
        self.add_message("user", f"Question: {task}")

        # Build initial prompt
        prompt = self._build_react_prompt(task)

        reasoning_steps = []
        tool_calls = []

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n[{self.name}] Iteration {iteration + 1}/{self.max_iterations}")

            # Generate next step
            try:
                response = self.generate(prompt, **kwargs)
            except Exception as e:
                return AgentResponse(
                    content="",
                    reasoning="\n".join(reasoning_steps),
                    tool_calls=tool_calls,
                    success=False,
                    error=f"Generation error: {str(e)}",
                    messages=self.get_messages(),
                )

            # Parse response for Thought, Action, etc.
            parsed = self._parse_react_response(response)

            if parsed["thought"]:
                reasoning_steps.append(f"Thought: {parsed['thought']}")
                if self.verbose:
                    print(f"Thought: {parsed['thought']}")

            if parsed["action"]:
                if self.verbose:
                    print(f"Action: {parsed['action']}")

                # Check if this is the final answer
                if self._is_final_answer(parsed["action"]):
                    final_answer = self._extract_final_answer(parsed["action"])
                    self.add_message("assistant", final_answer)

                    return AgentResponse(
                        content=final_answer,
                        reasoning="\n".join(reasoning_steps),
                        tool_calls=tool_calls,
                        success=True,
                        messages=self.get_messages(),
                    )

                # Execute tool
                tool_result = self._execute_action(parsed["action"])
                tool_calls.append(
                    {
                        "action": parsed["action"],
                        "result": str(tool_result.result),
                        "success": tool_result.success,
                    }
                )

                observation = f"Observation: {tool_result.result}"
                reasoning_steps.append(f"Action: {parsed['action']}")
                reasoning_steps.append(observation)

                if self.verbose:
                    print(observation)

                # Add observation to prompt for next iteration
                prompt += f"\n{response}\n{observation}\n"

            else:
                # No action found, might be final answer in different format
                if "answer:" in response.lower():
                    answer = response.split("answer:", 1)[1].strip()
                    self.add_message("assistant", answer)

                    return AgentResponse(
                        content=answer,
                        reasoning="\n".join(reasoning_steps),
                        tool_calls=tool_calls,
                        success=True,
                        messages=self.get_messages(),
                    )

                # Continue with response
                prompt += f"\n{response}\n"

        # Max iterations reached
        return AgentResponse(
            content="Maximum iterations reached without finding answer",
            reasoning="\n".join(reasoning_steps),
            tool_calls=tool_calls,
            success=False,
            error="Maximum iterations exceeded",
            messages=self.get_messages(),
        )

    def _build_react_prompt(self, task: str) -> str:
        """Build initial ReAct prompt.

        Args:
            task: Task description

        Returns:
            Formatted prompt
        """
        messages = self.format_messages_for_model()
        base_prompt = self._build_prompt(messages)
        return base_prompt

    def _parse_react_response(self, response: str) -> dict:
        """Parse ReAct formatted response.

        Args:
            response: Model response

        Returns:
            Dictionary with parsed components
        """
        parsed = {"thought": None, "action": None, "observation": None}

        # Extract Thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\n(?:Action|Observation|Answer):|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            parsed["thought"] = thought_match.group(1).strip()

        # Extract Action
        action_match = re.search(
            r"Action:\s*(.+?)(?=\n(?:Thought|Observation|Answer):|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            parsed["action"] = action_match.group(1).strip()

        # Extract Observation
        obs_match = re.search(
            r"Observation:\s*(.+?)(?=\n(?:Thought|Action|Answer):|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if obs_match:
            parsed["observation"] = obs_match.group(1).strip()

        return parsed

    def _is_final_answer(self, action: str) -> bool:
        """Check if action is the final answer.

        Args:
            action: Action string

        Returns:
            True if this is the final answer
        """
        return action.lower().startswith("answer(")

    def _extract_final_answer(self, action: str) -> str:
        """Extract final answer from action.

        Args:
            action: Action string

        Returns:
            Final answer
        """
        # Extract from Answer(result="...")
        match = re.search(r'Answer\(result="(.+?)"\)', action, re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback: return everything after Answer(
        match = re.search(r"Answer\((.+?)\)", action, re.IGNORECASE)
        if match:
            return match.group(1)

        return action

    def _execute_action(self, action: str) -> ToolResult:
        """Execute a tool action.

        Args:
            action: Action string in format "tool_name(param1=value1, ...)"

        Returns:
            ToolResult with execution results
        """
        # Parse tool call
        match = re.match(r"(\w+)\((.*?)\)", action)
        if not match:
            return ToolResult(
                tool_name="unknown",
                success=False,
                error=f"Could not parse action: {action}",
            )

        tool_name = match.group(1)
        params_str = match.group(2)

        # Parse parameters
        params = {}
        if params_str:
            # Simple parameter parsing (key="value" or key=value)
            for param in re.findall(r'(\w+)=(".*?"|[^,]+)', params_str):
                key = param[0]
                value = param[1].strip('"').strip("'")
                params[key] = value

        # Execute tool
        return self.tools.execute(tool_name, **params)


__all__ = ["ReActAgent"]
