from __future__ import annotations

from typing import Any, Dict, Optional

from core.bases import BaseAction


class EchoAnswerAction(BaseAction):
    """A trivial action: return the provided 'answer' parameter.

    Demonstrates how an action can be parameterized and called.
    """

    async def __call__(self, **kwargs) -> str:
        # Accept 'answer' field or fallback to an empty string
        return str(kwargs.get("answer", ""))


def make_echo_answer_action() -> EchoAnswerAction:
    return EchoAnswerAction(
        name="echo_answer",
        description="Return the provided 'answer' parameter.",
        parameters={
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Text to echo back"}
            },
            "required": ["answer"],
        },
    )

