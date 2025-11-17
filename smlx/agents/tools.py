#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Tool system for agents.

Provides a framework for defining and using tools that agents
can call to interact with external systems, perform calculations,
retrieve information, etc.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union


@dataclass
class ToolParameter:
    """Tool parameter definition."""

    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class ToolResult:
    """Result from tool execution."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            return f"Tool '{self.tool_name}' succeeded: {self.result}"
        else:
            return f"Tool '{self.tool_name}' failed: {self.error}"


class Tool(ABC):
    """Base class for agent tools.

    Tools are functions that agents can call to perform actions
    or retrieve information. Each tool has:
    - A name
    - A description
    - Input parameters
    - An execution method
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[List[ToolParameter]] = None,
    ):
        """Initialize tool.

        Args:
            name: Tool name
            description: Tool description
            parameters: List of tool parameters
        """
        self.name = name
        self.description = description
        self.parameters = parameters or []

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution results
        """
        pass

    def __call__(self, **kwargs) -> ToolResult:
        """Make tool callable."""
        return self.execute(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert tool to dictionary representation.

        Returns:
            dictionary with tool metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum,
                }
                for p in self.parameters
            ],
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


class FunctionTool(Tool):
    """Tool that wraps a Python function."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[List[ToolParameter]] = None,
    ):
        """Initialize function tool.

        Args:
            name: Tool name
            description: Tool description
            func: Python function to wrap
            parameters: Optional parameter definitions
        """
        super().__init__(name, description, parameters)
        self.func = func

        # Auto-generate parameters from function signature if not provided
        if not parameters:
            self.parameters = self._extract_parameters()

    def _extract_parameters(self) -> List[ToolParameter]:
        """Extract parameters from function signature."""
        sig = inspect.signature(self.func)
        parameters = []

        for param_name, param in sig.parameters.items():
            # Skip self/cls
            if param_name in ("self", "cls"):
                continue

            # Determine type
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in (int, float):
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation in (list, List):
                    param_type = "array"
                elif param.annotation in (dict, dict):
                    param_type = "object"

            # Check if required
            required = param.default == inspect.Parameter.empty

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter {param_name}",
                    required=required,
                    default=None if required else param.default,
                )
            )

        return parameters

    def execute(self, **kwargs) -> ToolResult:
        """Execute the wrapped function.

        Args:
            **kwargs: Function arguments

        Returns:
            ToolResult with execution results
        """
        try:
            result = self.func(**kwargs)
            return ToolResult(tool_name=self.name, success=True, result=result)
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
            )


class ToolRegistry:
    """Registry for managing tools.

    Provides a centralized place to register and retrieve tools
    that agents can use.
    """

    def __init__(self):
        """Initialize tool registry."""
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Union[Tool, Callable]) -> None:
        """Register a tool.

        Accepts either a Tool object or a callable function. If a function is provided,
        it will be automatically wrapped in a FunctionTool.

        Args:
            tool: Tool to register (Tool object or callable function)
        """
        if isinstance(tool, Tool):
            self.tools[tool.name] = tool
        elif callable(tool):
            # Auto-wrap function as FunctionTool
            # Use function name and docstring
            name = tool.__name__
            description = tool.__doc__ or f"Tool: {name}"
            # Clean up docstring - take first line
            description = description.strip().split("\n")[0]
            function_tool = FunctionTool(name, description, tool)
            self.tools[function_tool.name] = function_tool
        else:
            raise TypeError(f"Expected Tool or callable, got {type(tool)}")

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[List[ToolParameter]] = None,
    ) -> FunctionTool:
        """Register a function as a tool.

        Args:
            name: Tool name
            description: Tool description
            func: Function to wrap
            parameters: Optional parameter definitions

        Returns:
            Created FunctionTool
        """
        tool = FunctionTool(name, description, func, parameters)
        self.register(tool)
        return tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool if found, None otherwise
        """
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools.

        Returns:
            Formatted string with tool descriptions
        """
        descriptions = []
        for tool in self.tools.values():
            desc = f"- {tool.name}: {tool.description}"
            if tool.parameters:
                params = ", ".join(f"{p.name} ({p.type})" for p in tool.parameters)
                desc += f"\n  Parameters: {params}"
            descriptions.append(desc)

        return "\n".join(descriptions)

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution results
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found",
            )

        return tool.execute(**kwargs)

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self.tools)})"


# Built-in tools
def calculator(expression: str) -> Union[float, str]:
    """Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of evaluation
    """
    try:
        # Safe evaluation (only allow basic math)
        result = eval(
            expression,
            {"__builtins__": {}},
            {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
            },
        )
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"


def get_current_time() -> str:
    """Get the current time.

    Returns:
        Current time as formatted string
    """
    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Alias for backward compatibility
get_time = get_current_time


def search_wikipedia(query: str, max_length: int = 500) -> str:
    """Search Wikipedia for information.

    Args:
        query: Search query
        max_length: Maximum length of summary

    Returns:
        Wikipedia summary
    """
    try:
        import wikipedia

        # Search and get summary
        results = wikipedia.search(query, results=1)
        if not results:
            return f"No Wikipedia results found for '{query}'"

        page = wikipedia.page(results[0], auto_suggest=False)
        summary = page.summary[:max_length]

        if len(page.summary) > max_length:
            summary += "..."

        return summary

    except ImportError:
        return "Error: wikipedia package not installed (pip install wikipedia)"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


def create_default_registry() -> ToolRegistry:
    """Create a registry with default built-in tools.

    Returns:
        ToolRegistry with built-in tools
    """
    registry = ToolRegistry()

    # Register calculator
    registry.register_function(
        name="calculator",
        description="Evaluate mathematical expressions",
        func=calculator,
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate",
                required=True,
            )
        ],
    )

    # Register time tool
    registry.register_function(
        name="get_current_time",
        description="Get the current date and time",
        func=get_current_time,
        parameters=[],
    )

    # Register Wikipedia search
    registry.register_function(
        name="search_wikipedia",
        description="Search Wikipedia for information",
        func=search_wikipedia,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True,
            ),
            ToolParameter(
                name="max_length",
                type="number",
                description="Maximum length of summary",
                required=False,
                default=500,
            ),
        ],
    )

    return registry


__all__ = [
    "Tool",
    "FunctionTool",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "calculator",
    "get_current_time",
    "get_time",  # Alias for get_current_time
    "search_wikipedia",
    "create_default_registry",
]
