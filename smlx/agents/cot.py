#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Chain-of-Thought (CoT) Agent Implementation.

Chain-of-Thought prompting encourages models to break down complex
reasoning into intermediate steps, improving accuracy on tasks
requiring multi-step reasoning.

Reference:
    Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    https://arxiv.org/abs/2201.11903
"""

from typing import Any, Optional

from .base import AgentResponse, LLMAgent


class CoTAgent(LLMAgent):
    """Chain-of-Thought (CoT) agent.

    This agent uses Chain-of-Thought prompting to solve complex
    reasoning tasks by breaking them down into steps.

    Supports two CoT modes:
    1. Zero-shot CoT: Simply adds "Let's think step by step"
    2. Few-shot CoT: Provides examples with reasoning steps
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        name: str = "CoTAgent",
        system_prompt: Optional[str] = None,
        examples: Optional[list[dict]] = None,
        zero_shot: bool = True,
        max_tokens: int = 800,
        temperature: float = 0.7,
        verbose: bool = False,
    ):
        """Initialize CoT agent.

        Args:
            model: Language model
            tokenizer: Tokenizer
            name: Agent name
            system_prompt: Custom system prompt
            examples: Few-shot examples (list of {"question": str, "reasoning": str, "answer": str})
            zero_shot: Use zero-shot CoT if True, few-shot if False
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            verbose: Verbose output
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            name=name,
            system_prompt=system_prompt or self._create_cot_prompt(examples, zero_shot),
            max_iterations=1,  # CoT typically does single-pass reasoning
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
        )

        self.examples = examples or []
        self.zero_shot = zero_shot

    def _default_system_prompt(self) -> str:
        """Override to provide CoT-specific prompt."""
        return self._create_cot_prompt(self.examples, self.zero_shot)

    def _create_cot_prompt(
        self, examples: Optional[list[dict]], zero_shot: bool
    ) -> str:
        """Create Chain-of-Thought system prompt.

        Args:
            examples: Few-shot examples
            zero_shot: Whether to use zero-shot CoT

        Returns:
            Formatted system prompt
        """
        if zero_shot:
            prompt = f"""You are {self.name}, an AI assistant that uses step-by-step reasoning to solve problems.

When given a question or task:
1. Break down the problem into clear steps
2. Reason through each step carefully
3. Show your work and intermediate calculations
4. Arrive at a final answer

Always start your response with "Let's think step by step:" and then work through the problem systematically."""

        else:
            # Few-shot CoT with examples
            prompt = f"""You are {self.name}, an AI assistant that uses step-by-step reasoning to solve problems.

Here are some examples of how to reason through problems:

"""
            if examples:
                for i, example in enumerate(examples, 1):
                    prompt += f"""Example {i}:
Question: {example['question']}
Reasoning: {example['reasoning']}
Answer: {example['answer']}

"""

            prompt += """Now, when given a new question, follow the same step-by-step reasoning approach."""

        return prompt

    def run(self, task: str, extract_answer: bool = True, **kwargs) -> AgentResponse:
        """Execute Chain-of-Thought reasoning on a task.

        Args:
            task: Task or question to solve
            extract_answer: Whether to extract final answer from reasoning
            **kwargs: Additional generation arguments

        Returns:
            AgentResponse with reasoning and answer
        """
        # Build prompt
        if self.zero_shot:
            # Zero-shot CoT: Add "Let's think step by step"
            prompt = f"{task}\n\nLet's think step by step:"
        else:
            # Few-shot CoT: Just add the task
            prompt = f"Question: {task}\nReasoning:"

        # Add to messages
        self.add_message("user", task)

        # Build full prompt with context
        messages = self.format_messages_for_model()
        full_prompt = self._build_prompt(messages[:-1])  # Exclude last user message
        full_prompt += f"\n{prompt}"

        if self.verbose:
            print(f"[{self.name}] Generating reasoning...")

        # Generate reasoning
        try:
            response = self.generate(full_prompt, **kwargs)
        except Exception as e:
            return AgentResponse(
                content="",
                success=False,
                error=f"Generation error: {str(e)}",
                messages=self.get_messages(),
            )

        # Extract answer if requested
        final_answer = response
        reasoning = response

        if extract_answer:
            # Try to extract answer from reasoning
            extracted = self._extract_answer(response)
            if extracted:
                final_answer = extracted
                reasoning = response

        # Add assistant message
        self.add_message("assistant", final_answer)

        if self.verbose:
            print(f"\n[{self.name}] Reasoning:\n{reasoning}\n")
            if extract_answer:
                print(f"[{self.name}] Final Answer: {final_answer}")

        return AgentResponse(
            content=final_answer,
            reasoning=reasoning,
            success=True,
            messages=self.get_messages(),
            metadata={"zero_shot": self.zero_shot},
        )

    def _extract_answer(self, reasoning: str) -> Optional[str]:
        """Extract final answer from reasoning.

        Args:
            reasoning: Full reasoning text

        Returns:
            Extracted answer or None
        """
        # Look for common answer patterns
        patterns = [
            r"(?:final answer|answer|therefore|thus|so)[:,\s]+(.+?)(?:\.|$)",
            r"the answer is\s+(.+?)(?:\.|$)",
            r"(?:^|\n)answer:\s*(.+?)(?:\.|$)",
        ]

        import re

        for pattern in patterns:
            match = re.search(pattern, reasoning, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        # If no pattern found, return last sentence
        sentences = [s.strip() for s in reasoning.split(".") if s.strip()]
        if sentences:
            return sentences[-1]

        return None


class SelfConsistencyCoTAgent(CoTAgent):
    """Self-Consistency Chain-of-Thought agent.

    Generates multiple reasoning paths and takes majority vote
    for the final answer, improving reliability.

    Reference:
        Wang et al. (2022) "Self-Consistency Improves Chain of Thought Reasoning"
        https://arxiv.org/abs/2203.11171
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        name: str = "SelfConsistencyCoTAgent",
        num_samples: int = 5,
        **kwargs,
    ):
        """Initialize Self-Consistency CoT agent.

        Args:
            model: Language model
            tokenizer: Tokenizer
            name: Agent name
            num_samples: Number of reasoning paths to generate
            **kwargs: Additional arguments for CoTAgent
        """
        super().__init__(
            model=model, tokenizer=tokenizer, name=name, **kwargs
        )
        self.num_samples = num_samples

    def run(self, task: str, **kwargs) -> AgentResponse:
        """Execute Self-Consistency CoT reasoning.

        Args:
            task: Task or question to solve
            **kwargs: Additional generation arguments

        Returns:
            AgentResponse with majority-vote answer
        """
        if self.verbose:
            print(f"[{self.name}] Generating {self.num_samples} reasoning paths...")

        # Generate multiple reasoning paths
        responses = []
        all_reasoning = []

        for i in range(self.num_samples):
            if self.verbose:
                print(f"\n[{self.name}] Sample {i + 1}/{self.num_samples}")

            # Generate single reasoning path
            response = super().run(task, extract_answer=True, **kwargs)

            if response.success:
                responses.append(response.content)
                all_reasoning.append(response.reasoning)

        if not responses:
            return AgentResponse(
                content="",
                success=False,
                error="Failed to generate any reasoning paths",
                messages=self.get_messages(),
            )

        # Take majority vote
        from collections import Counter

        # Normalize answers for comparison
        normalized = [self._normalize_answer(r) for r in responses]
        vote_counts = Counter(normalized)
        majority_answer_norm = vote_counts.most_common(1)[0][0]

        # Find original answer corresponding to majority
        for orig, norm in zip(responses, normalized):
            if norm == majority_answer_norm:
                majority_answer = orig
                break

        # Compile reasoning from all paths
        compiled_reasoning = "Self-Consistency CoT: Generated multiple reasoning paths\n\n"
        for i, reasoning in enumerate(all_reasoning, 1):
            compiled_reasoning += f"Path {i}:\n{reasoning}\n\n"

        compiled_reasoning += f"\nMajority Answer (appeared {vote_counts[majority_answer_norm]}/{self.num_samples} times):\n{majority_answer}"

        if self.verbose:
            print(f"\n[{self.name}] Answer counts: {dict(vote_counts)}")
            print(f"[{self.name}] Majority Answer: {majority_answer}")

        return AgentResponse(
            content=majority_answer,
            reasoning=compiled_reasoning,
            success=True,
            messages=self.get_messages(),
            metadata={
                "num_samples": self.num_samples,
                "vote_counts": dict(vote_counts),
                "all_answers": responses,
            },
        )

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison.

        Args:
            answer: Answer text

        Returns:
            Normalized answer
        """
        # Convert to lowercase, remove punctuation, strip whitespace
        import re

        normalized = answer.lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = normalized.strip()

        return normalized


__all__ = ["CoTAgent", "SelfConsistencyCoTAgent"]
