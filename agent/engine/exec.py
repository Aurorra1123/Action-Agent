from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Optional
import asyncio

from runtime.sandbox import sandbox


@dataclass
class ExecResult:
    ok: bool
    output: Any
    cost: float = 0.0
    logs: list[str] | None = None


class CallableAction(Protocol):
    async def __call__(self, **kwargs) -> Any: ...


class ToolExecutor:
    """Minimal tool executor with optional sandbox flag.

    This is a placeholder to unify execution semantics and accounting.
    """

    def __init__(self, timeout_s: Optional[float] = 30.0) -> None:
        self.timeout_s = timeout_s

    async def run(self, action: CallableAction, params: Dict[str, Any] | None = None, sandbox: bool = True) -> ExecResult:
        params = params or {}
        try:
            async def _invoke():
                return await action(**params)

            with sandbox(enabled=sandbox):
                if self.timeout_s is not None:
                    out = await asyncio.wait_for(_invoke(), timeout=self.timeout_s)
                else:
                    out = await _invoke()
            return ExecResult(ok=True, output=out, cost=0.0, logs=[])
        except asyncio.TimeoutError:
            return ExecResult(ok=False, output="timeout", cost=0.0, logs=["timeout"])
        except Exception as e:
            return ExecResult(ok=False, output=str(e), cost=0.0, logs=[f"error: {e}"])
