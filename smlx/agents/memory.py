#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Memory and history management for agents.

Provides different memory strategies for agents:
- Short-term memory (recent conversation)
- Long-term memory (persistent storage)
- Summarization (compress long histories)
- Retrieval (semantic search over memories)
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Memory:
    """A single memory entry."""

    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0  # 0-1 scale for memory importance

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 1.0),
        )


class MemoryManager(ABC):
    """Base class for memory management."""

    @abstractmethod
    def add(self, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Add a memory."""
        pass

    @abstractmethod
    def get_recent(self, n: int = 10) -> list[Memory]:
        """Get n most recent memories."""
        pass

    @abstractmethod
    def search(self, query: str, n: int = 5) -> list[Memory]:
        """Search memories by query."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories."""
        pass


class SimpleMemory(MemoryManager):
    """Simple in-memory storage.

    Stores memories in a list with no persistence.
    Good for single-session agents.
    """

    def __init__(self, max_memories: Optional[int] = None):
        """Initialize simple memory.

        Args:
            max_memories: Maximum memories to store (None = unlimited)
        """
        self.memories: list[Memory] = []
        self.max_memories = max_memories

    def add(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> None:
        """Add a memory.

        Args:
            content: Memory content
            metadata: Optional metadata
            importance: Memory importance (0-1)
        """
        memory = Memory(
            content=content,
            metadata=metadata or {},
            importance=importance,
        )
        self.memories.append(memory)

        # Trim if needed
        if self.max_memories and len(self.memories) > self.max_memories:
            # Remove least important old memories
            self.memories.sort(key=lambda m: (m.importance, m.timestamp))
            self.memories = self.memories[-self.max_memories :]

    def get_recent(self, n: int = 10) -> list[Memory]:
        """Get n most recent memories.

        Args:
            n: Number of memories to retrieve

        Returns:
            list of recent memories
        """
        return self.memories[-n:]

    def search(self, query: str, n: int = 5) -> list[Memory]:
        """Search memories by simple text matching.

        Args:
            query: Search query
            n: Number of results

        Returns:
            list of matching memories
        """
        query_lower = query.lower()
        matches = []

        for memory in self.memories:
            if query_lower in memory.content.lower():
                matches.append(memory)

        # Sort by importance and recency
        matches.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)

        return matches[:n]

    def get_all(self) -> list[Memory]:
        """Get all memories.

        Returns:
            list of all memories
        """
        return self.memories

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = []

    def __len__(self) -> int:
        return len(self.memories)


class PersistentMemory(SimpleMemory):
    """Persistent memory with file storage.

    Saves memories to disk for persistence across sessions.
    """

    def __init__(self, storage_path: str, max_memories: Optional[int] = None):
        """Initialize persistent memory.

        Args:
            storage_path: Path to storage file
            max_memories: Maximum memories to store
        """
        super().__init__(max_memories)
        self.storage_path = Path(storage_path)

        # Load existing memories
        self.load()

    def add(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> None:
        """Add a memory and save to disk.

        Args:
            content: Memory content
            metadata: Optional metadata
            importance: Memory importance
        """
        super().add(content, metadata, importance)
        self.save()

    def save(self) -> None:
        """Save memories to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = [m.to_dict() for m in self.memories]

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load memories from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            self.memories = [Memory.from_dict(m) for m in data]
        except Exception as e:
            print(f"Warning: Could not load memories from {self.storage_path}: {e}")

    def clear(self) -> None:
        """Clear memories and delete file."""
        super().clear()
        if self.storage_path.exists():
            self.storage_path.unlink()


class SummarizedMemory(MemoryManager):
    """Memory with automatic summarization.

    Compresses old memories into summaries to save space
    while maintaining important context.
    """

    def __init__(
        self,
        summarizer: Optional[Any] = None,
        window_size: int = 20,
        summary_size: int = 5,
    ):
        """Initialize summarized memory.

        Args:
            summarizer: Optional summarization function/model
            window_size: Number of recent memories to keep unsummarized
            summary_size: Target size for summaries
        """
        self.summarizer = summarizer
        self.window_size = window_size
        self.summary_size = summary_size

        self.recent_memories: list[Memory] = []
        self.summaries: list[Memory] = []

    def add(self, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Add a memory, triggering summarization if needed.

        Args:
            content: Memory content
            metadata: Optional metadata
        """
        memory = Memory(content=content, metadata=metadata or {})
        self.recent_memories.append(memory)

        # Check if we need to summarize
        if len(self.recent_memories) > self.window_size:
            self._summarize_old_memories()

    def _summarize_old_memories(self) -> None:
        """Summarize old memories."""
        # Take oldest memories to summarize
        to_summarize = self.recent_memories[: self.summary_size]
        self.recent_memories = self.recent_memories[self.summary_size :]

        # Create summary
        if self.summarizer:
            # Use provided summarizer
            combined = "\n".join([m.content for m in to_summarize])
            summary_text = self.summarizer(combined)
        else:
            # Simple concatenation with ellipsis
            summary_text = (
                "Summary of earlier conversation: "
                + " ... ".join([m.content[:100] for m in to_summarize])
            )

        summary = Memory(
            content=summary_text,
            metadata={"type": "summary", "num_memories": len(to_summarize)},
        )
        self.summaries.append(summary)

    def get_recent(self, n: int = 10) -> list[Memory]:
        """Get recent memories.

        Args:
            n: Number of memories

        Returns:
            list of recent memories (may include summaries)
        """
        # Return summaries + recent memories
        all_memories = self.summaries + self.recent_memories
        return all_memories[-n:]

    def search(self, query: str, n: int = 5) -> list[Memory]:
        """Search all memories including summaries.

        Args:
            query: Search query
            n: Number of results

        Returns:
            list of matching memories
        """
        query_lower = query.lower()
        matches = []

        for memory in self.summaries + self.recent_memories:
            if query_lower in memory.content.lower():
                matches.append(memory)

        return matches[:n]

    def clear(self) -> None:
        """Clear all memories and summaries."""
        self.recent_memories = []
        self.summaries = []


class ConversationMemory:
    """Specialized memory for conversation history.

    Tracks alternating user/assistant messages with context
    window management.
    """

    def __init__(self, max_turns: int = 10):
        """Initialize conversation memory.

        Args:
            max_turns: Maximum conversation turns to remember
        """
        self.max_turns = max_turns
        self.turns: list[dict[str, str]] = []

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Add a conversation turn.

        Args:
            user_msg: User message
            assistant_msg: Assistant response
        """
        self.turns.append({"user": user_msg, "assistant": assistant_msg})

        # Trim if needed
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def get_context(self, include_last_n: Optional[int] = None) -> str:
        """Get conversation context as formatted string.

        Args:
            include_last_n: Number of recent turns to include

        Returns:
            Formatted conversation history
        """
        turns = self.turns[-include_last_n:] if include_last_n else self.turns

        context_parts = []
        for turn in turns:
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Assistant: {turn['assistant']}")

        return "\n".join(context_parts)

    def clear(self) -> None:
        """Clear conversation history."""
        self.turns = []

    def __len__(self) -> int:
        return len(self.turns)


__all__ = [
    "Memory",
    "MemoryManager",
    "SimpleMemory",
    "PersistentMemory",
    "SummarizedMemory",
    "ConversationMemory",
]
