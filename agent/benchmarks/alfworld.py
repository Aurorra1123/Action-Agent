from __future__ import annotations

import json
from typing import Any, Callable, List, Tuple, Dict

from .base import BaseBenchmark


class ALFWorldBenchmark(BaseBenchmark):
    """ALFWorld-style evaluation wrapper.

    Expected problem format: {"id": str, "init": dict, ...}
    Agent returns trajectory info with success flag.
    """

    async def single_attempt(self, problem: dict, agent: Callable[..., Any]) -> Dict[str, Any]:
        pred = await agent(problem)
        passed = bool(pred.get("success")) if isinstance(pred, dict) else bool(pred)
        return {"ok": passed, "final": pred, "cost": {}, "meta": {}}

    async def evaluate_problem(self, problem: dict, agent: Callable[..., Any]) -> Tuple[Any, ...]:
        pid = problem.get("id")
        k = getattr(self, "pass_k", 1)
        rs = await self.run_sample(problem, agent, k)
        tries = rs.get("tries", [])
        succ = [bool(t.get("ok")) for t in tries]
        p1 = self.compute_pass_at_k(succ, 1)
        pk = self.compute_pass_at_k(succ, k)
        unit_cost = self.compute_unit_success_cost(tries)
        total_cost = 0.0
        for t in tries:
            try:
                total_cost += float((t.get("cost") or {}).get("usd", 0.0))
            except Exception:
                pass
        tries_json = json.dumps(tries, ensure_ascii=False)
        return (pid, p1, total_cost, pk, unit_cost, tries_json)

    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        # Usually decided by environment success, not explicit expected_output
        passed = bool(prediction.get("success")) if isinstance(prediction, dict) else bool(prediction)
        return (1.0 if passed else 0.0, prediction)

    def get_result_columns(self) -> List[str]:
        return ["id", "score", "cost", "pass@k", "unit_success_cost", "tries_json"]
