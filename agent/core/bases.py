"""Base classes for Actions, Workflows, and Agents.

Preferred path is to import from Example_code AutoEnv base classes.
When unavailable, provide minimal Pydantic-based fallbacks to preserve
the API surface used by this repository.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def _try_import():
    import sys, os
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, "Example_code", "AutoEnv"),
        os.path.join(cwd, "Example_code"),
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)
    try:
        from autoenv.agent.base.base_action import BaseAction as _BA  # type: ignore
        from autoenv.agent.base.base_agent import BaseAgent as _BAG  # type: ignore
        from autoenv.agent.base.base_workflow import BaseWorkflow as _BW  # type: ignore
        return _BA, _BAG, _BW
    except Exception:
        return None


_imported = _try_import()

if _imported is not None:
    BaseAction, BaseAgent, BaseWorkflow = _imported  # type: ignore
else:
    try:
        from pydantic import BaseModel, Field
    except Exception:  # very minimal fallback
        class _BM:  # type: ignore
            pass

        BaseModel = _BM  # type: ignore
        def Field(*args, **kwargs):  # type: ignore
            return None

    class BaseAction(BaseModel):  # type: ignore
        name: str
        description: Optional[str] = None
        parameters: Dict[str, Any] | None = None
        # governance and execution metadata (optional)
        version: Optional[str] = None
        provenance: Dict[str, Any] | None = None
        security: Dict[str, Any] | None = None
        effects_schema: Dict[str, Any] | None = None
        exec_policy: Dict[str, Any] | None = None  # e.g., timeout/retries/budget

        async def __call__(self, **kwargs) -> str:  # abstract in spirit
            raise NotImplementedError

        def to_param(self) -> Dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }

    class BaseAgent(BaseAction):  # type: ignore
        system_prompt: Optional[str] = None
        next_step_prompt: Optional[str] = None
        max_steps: int = 10
        current_step: int = 0

        async def step(self) -> str:
            raise NotImplementedError

        async def run(self, request: Optional[str] = None) -> str:
            raise NotImplementedError

        async def __call__(self, **kwargs) -> Any:
            return await self.run(**kwargs)

        def to_param(self) -> Dict[str, Any]:
            return {
                "type": "agent-as-function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }

    class BaseWorkflow(BaseAction):  # type: ignore
        llm_config: Any | None = None
        dataset: Any | None = None
        llm: Any | None = None

        async def __call__(self, **kwargs):
            raise NotImplementedError
