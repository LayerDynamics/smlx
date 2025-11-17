# SMLX Agent System

Autonomous agents for SMLX that can reason, use tools, and solve complex tasks.

## Overview

The SMLX agent system provides:

- **Base Agents**: Foundation for building autonomous agents
- **ReAct**: Reasoning + Acting paradigm with tool use
- **Chain-of-Thought**: Step-by-step reasoning for complex problems
- **Tools**: Extensible tool system for agent actions
- **Memory**: Various memory strategies for context management

## Quick Start

```python
from smlx.models.SmolLM2_135M import load
from smlx.agents import ReActAgent, create_default_registry

# Load model
model, tokenizer = load()

# Create agent with tools
tools = create_default_registry()
agent = ReActAgent(model, tokenizer, tools=tools)

# Run agent on task
response = agent.run("What is 25 * 37?")
print(response.content)
```

## Agent Types

### Base Agent

Abstract base class for all agents:

```python
from smlx.agents import BaseAgent

class MyAgent(BaseAgent):
    def run(self, task: str, **kwargs) -> AgentResponse:
        # Implement your agent logic
        pass
```

### LLM Agent

Basic agent with language model integration:

```python
from smlx.agents import LLMAgent

agent = LLMAgent(
    model=model,
    tokenizer=tokenizer,
    name="Assistant",
    system_prompt="You are a helpful AI assistant.",
    max_tokens=500,
    temperature=0.7
)

response = agent.run("Explain quantum computing")
```

### ReAct Agent

Reasoning + Acting agent with tool use:

```python
from smlx.agents import ReActAgent, create_default_registry

tools = create_default_registry()  # calculator, time, Wikipedia

agent = ReActAgent(
    model=model,
    tokenizer=tokenizer,
    tools=tools,
    max_iterations=10
)

response = agent.run("Calculate 15 * 23 and tell me the current time")
```

### Chain-of-Thought Agent

Step-by-step reasoning:

```python
from smlx.agents import CoTAgent

# Zero-shot CoT
agent = CoTAgent(
    model=model,
    tokenizer=tokenizer,
    zero_shot=True
)

response = agent.run("If a train travels 60 mph for 2.5 hours, how far does it go?")
print(response.reasoning)  # See step-by-step reasoning
print(response.content)     # Final answer

# Few-shot CoT with examples
examples = [
    {
        "question": "Roger has 5 balls. He buys 2 cans of 3 balls each. How many total?",
        "reasoning": "Roger started with 5. 2 cans × 3 balls = 6 balls. 5 + 6 = 11.",
        "answer": "11 balls"
    }
]

agent = CoTAgent(
    model=model,
    tokenizer=tokenizer,
    examples=examples,
    zero_shot=False
)
```

### Self-Consistency CoT

Generate multiple reasoning paths and vote:

```python
from smlx.agents import SelfConsistencyCoTAgent

agent = SelfConsistencyCoTAgent(
    model=model,
    tokenizer=tokenizer,
    num_samples=5  # Generate 5 reasoning paths
)

response = agent.run("A bat and ball cost $1.10. The bat costs $1 more. What does the ball cost?")
print(response.metadata['vote_counts'])  # See voting results
```

## Tools

### Built-in Tools

The agent system includes several built-in tools:

- **calculator**: Evaluate mathematical expressions
- **get_current_time**: Get current date and time
- **search_wikipedia**: Search Wikipedia for information

```python
from smlx.agents import create_default_registry

tools = create_default_registry()
print(tools.list_tools())  # ['calculator', 'get_current_time', 'search_wikipedia']
```

### Custom Tools

Create your own tools:

```python
from smlx.agents import ToolRegistry, ToolParameter

def string_length(text: str) -> int:
    """Calculate the length of a string."""
    return len(text)

tools = ToolRegistry()
tools.register_function(
    name="string_length",
    description="Calculate the length of a string",
    func=string_length,
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="The text to measure",
            required=True
        )
    ]
)
```

### Tool Classes

For more complex tools, create a Tool class:

```python
from smlx.agents import Tool, ToolResult

class DatabaseTool(Tool):
    def __init__(self, db_connection):
        super().__init__(
            name="query_database",
            description="Query the database",
            parameters=[...]
        )
        self.db = db_connection

    def execute(self, query: str) -> ToolResult:
        try:
            result = self.db.execute(query)
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=result
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e)
            )
```

## Memory

### Simple Memory

In-memory storage:

```python
from smlx.agents import SimpleMemory

memory = SimpleMemory(max_memories=100)
memory.add("User likes science fiction", importance=0.8)
memory.add("User's favorite color is blue", importance=0.6)

# Get recent memories
recent = memory.get_recent(n=5)

# Search memories
results = memory.search("color")
```

### Persistent Memory

Save to disk:

```python
from smlx.agents import PersistentMemory

memory = PersistentMemory(
    storage_path="~/.smlx/agent_memory.json",
    max_memories=1000
)

memory.add("Important information")
# Automatically saved to disk

# Load in next session
memory = PersistentMemory("~/.smlx/agent_memory.json")
# Previous memories loaded automatically
```

### Conversation Memory

Track conversation turns:

```python
from smlx.agents import ConversationMemory

memory = ConversationMemory(max_turns=10)

memory.add_turn(
    user_msg="What's the weather?",
    assistant_msg="I don't have weather data."
)

# Get formatted context
context = memory.get_context(include_last_n=3)
```

## Multi-Agent Systems

Combine multiple agents:

```python
from smlx.agents import LLMAgent, ReActAgent, CoTAgent

# Create specialized agents
planner = CoTAgent(
    model=model,
    tokenizer=tokenizer,
    name="Planner",
    system_prompt="Break tasks into steps"
)

executor = ReActAgent(
    model=model,
    tokenizer=tokenizer,
    tools=tools,
    name="Executor"
)

verifier = LLMAgent(
    model=model,
    tokenizer=tokenizer,
    name="Verifier",
    system_prompt="Verify results are correct"
)

# Build workflow
task = "Calculate the area of a circle with radius 7"

# Plan
plan = planner.run(f"Break down: {task}")

# Execute
result = executor.run(task)

# Verify
verification = verifier.run(f"Verify: {result.content}")
```

## Examples

Comprehensive examples are available in `examples/agents/`:

- **basic_agent_example.py**: Basic agent usage, conversation history, memory
- **react_agent_example.py**: ReAct with tools, custom tools, multi-step problem solving
- **cot_agent_example.py**: Chain-of-Thought reasoning, zero-shot vs few-shot
- **multi_agent_example.py**: Multi-agent systems, collaboration, workflows

Run an example:

```bash
python examples/agents/basic_agent_example.py
python examples/agents/react_agent_example.py
python examples/agents/cot_agent_example.py
python examples/agents/multi_agent_example.py
```

## Architecture

```text
smlx/agents/
├── __init__.py         # Public API
├── base.py            # BaseAgent, LLMAgent
├── react.py           # ReActAgent
├── cot.py             # CoTAgent, SelfConsistencyCoTAgent
├── tools.py           # Tool system, built-in tools
├── memory.py          # Memory management
└── README.md          # This file
```

## Best Practices

### 1. Choose the Right Agent

- **LLMAgent**: Simple Q&A, conversation
- **ReActAgent**: Tasks requiring tools or external actions
- **CoTAgent**: Complex reasoning, math, logic puzzles
- **Self-Consistency**: High-stakes decisions, ambiguous problems

### 2. Tool Design

- Keep tools focused and single-purpose
- Provide clear descriptions for the agent
- Handle errors gracefully
- Return structured results

### 3. Memory Management

- Use SimpleMemory for single sessions
- Use PersistentMemory for multi-session agents
- Set appropriate max_memories limits
- Mark important memories with higher importance scores

### 4. Prompting

- Clear, specific system prompts work best
- Include task-relevant constraints
- For few-shot CoT, provide diverse examples
- Test prompts with different temperatures

### 5. Multi-Agent Systems

- Specialize agents for specific roles
- Pass information explicitly between agents
- Include verification/validation steps
- Keep workflows simple and debuggable

## References

### Papers

- **ReAct**: Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models" [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)

- **Chain-of-Thought**: Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)

- **Self-Consistency**: Wang et al. (2022) "Self-Consistency Improves Chain of Thought Reasoning in Language Models" [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)

### Frameworks

- LangChain: <https://github.com/langchain-ai/langchain>
- AutoGPT: <https://github.com/Significant-Gravitas/AutoGPT>
- BabyAGI: <https://github.com/yoheinakajima/babyagi>

## API Reference

### BaseAgent

```python
class BaseAgent(ABC):
    def __init__(
        self,
        name: str = "Agent",
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False
    )

    @abstractmethod
    def run(self, task: str, **kwargs) -> AgentResponse

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None)
    def get_messages(self, role: Optional[str] = None) -> List[Message]
    def clear_history(self, keep_system: bool = True)
```

### AgentResponse

```python
@dataclass
class AgentResponse:
    content: str                          # Final response
    reasoning: Optional[str] = None       # Reasoning process
    tool_calls: List[Dict] = []          # Tools called
    metadata: Dict[str, Any] = {}         # Additional data
    messages: List[Message] = []          # Conversation history
    success: bool = True                  # Success flag
    error: Optional[str] = None           # Error message if failed
```

### ToolRegistry

```python
class ToolRegistry:
    def register(self, tool: Tool)
    def register_function(self, name: str, description: str, func: Callable, ...)
    def get(self, name: str) -> Optional[Tool]
    def list_tools(self) -> List[str]
    def execute(self, tool_name: str, **kwargs) -> ToolResult
```

## License

Copyright © 2025 SMLX Project
