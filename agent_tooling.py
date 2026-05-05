"""
agent_tooling.py — Base class for server-side tool providers.

Subclass AgentTooling and implement all three methods. Pass an instance to
UIServer(tooling=...) or server.register_tooling(...) to enable server-side
intent classification + tool dispatch.
"""

from abc import ABC, abstractmethod
from typing import Any


class AgentTooling(ABC):
    """Protocol for server-side tool providers.

    The server calls these three methods on whatever instance is registered:

      - classifier_prompt() → str
            System prompt for the intent-classification pass.
            Should instruct the model to return a JSON object with an
            "intent" field and any tool-specific arguments.

      - normalize_result(obj: dict) → dict
            Validate and clean the JSON parsed from the classifier output.
            Drop unknown intents, filter unexpected fields, coerce types.

      - execute(name: str, arguments: dict) → Any
            Run the named tool with the given arguments and return its result.
            Typical implementation: getattr(self, name)(**arguments).
    """

    @abstractmethod
    def classifier_prompt(self) -> str:
        """Return the system prompt for the intent classifier."""

    @abstractmethod
    def normalize_result(self, obj: dict[str, Any]) -> dict[str, Any]:
        """Validate / clean the JSON returned by the classifier."""

    @abstractmethod
    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Run the named tool with the given arguments."""
