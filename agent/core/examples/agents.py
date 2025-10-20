from __future__ import annotations

from typing import Any, Dict, Optional

from core.bases import BaseAgent
from core.action_space import ActionSpace


class GAIAEchoAgent(BaseAgent):
    """GAIA demo agent that uses an ActionSpace action to produce an answer.

    For demonstration, it calls the 'echo_answer' action with the expected
    'answer' from the problem, simulating a correct prediction. In real cases
    the agent would reason or call tools to compute answers.
    """

    action_space: ActionSpace
    action_id: str

    async def step(self) -> str:
        return "noop"

    async def run(self, request: Optional[Dict[str, Any]] = None) -> str:
        problem = request or {}
        # Provide the ground truth as the parameter to the action (demo only)
        params = {"answer": problem.get("answer", "")}
        action = self.action_space.get(self.action_id)
        if action is None:
            raise KeyError(f"Action {self.action_id!r} not registered")
        return await action(**params)

