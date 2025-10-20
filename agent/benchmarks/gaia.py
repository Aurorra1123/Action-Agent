from __future__ import annotations

import json
from typing import Any, Callable, List, Tuple, Dict

from .base import BaseBenchmark


class GAIABenchmark(BaseBenchmark):
    """GAIA-style evaluation wrapper.

    Expected problem format: {"id": str, "question": str, "answer": Any}
    Agent signature: async agent(problem: dict) -> dict | str
    """

    async def single_attempt(self, problem: dict, agent: Callable[..., Any]) -> Dict[str, Any]:
        expected = problem.get("answer")
        prediction = await agent(problem)
        score, extracted = self.calculate_score(expected, prediction)
        ok = (score == 1.0)
        if not ok:
            self.log_mismatch(problem.get("question", ""), expected, str(prediction), extracted, extract_answer_code="None")
        return {"ok": ok, "final": prediction, "cost": {}, "meta": {"extracted": extracted}}

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
        # keep backward-compatible columns: 'score' and 'cost'
        return (pid, p1, total_cost, pk, unit_cost, tries_json)

    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        # naive exact-match baseline
        ok = str(expected_output).strip() == str(prediction).strip()
        return (1.0 if ok else 0.0, prediction)

    def get_result_columns(self) -> List[str]:
        # score=pass@1, cost=total_cost, plus pass@k/unit_success_cost/tries_json
        return ["id", "score", "cost", "pass@k", "unit_success_cost", "tries_json"]
