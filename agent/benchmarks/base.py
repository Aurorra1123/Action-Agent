import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Dict


class _Logger:
    @staticmethod
    def info(msg: str) -> None:
        print(f"[INFO] {msg}")

    @staticmethod
    def warning(msg: str) -> None:
        print(f"[WARN] {msg}")

    @staticmethod
    def error(msg: str) -> None:
        print(f"[ERROR] {msg}")


logger = _Logger()


class BaseBenchmark(ABC):
    """
    Async evaluation skeleton aligned with Example_code/benchmark.py
    plus Pass@k support and standardized result summarization.
    """

    PASS = "PASS"
    FAIL = "FAIL"

    def __init__(self, name: str, file_path: str, log_path: str):
        self.name = name
        self.file_path = file_path
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)
        # default pass@k setting used by evaluate_problem if applicable
        self.pass_k: int = 1

    # ----------------------------
    # Data loading
    # ----------------------------
    async def load_data(self, specific_indices: List[int] | None = None) -> List[dict]:
        data: List[dict] = []
        # streaming read lines of JSONL
        with open(self.file_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line in dataset.")
        if specific_indices is not None:
            return [data[i] for i in specific_indices if 0 <= i < len(data)]
        return data

    # ----------------------------
    # Result logging helpers
    # ----------------------------
    def _save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]) -> Tuple[float, float, float, str]:
        try:
            import pandas as pd  # optional
        except Exception:
            # Minimal fallback: write JSON for portability
            avg_score = sum(r[columns.index("score")] for r in results) / max(1, len(results))
            t_cost = max(r[columns.index("cost")] for r in results) if results else 0.0
            a_cost = t_cost / max(1, len(results))
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{avg_score:.5f}_{current_time}.json"
            output_file = os.path.join(self.log_path, filename)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"columns": columns, "rows": results}, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_file}")
            return avg_score, a_cost, t_cost, output_file

        import pandas as pd  # type: ignore
        df = pd.DataFrame(results, columns=columns)
        avg_score = float(df["score"].mean()) if not df.empty else 0.0
        t_cost = float(df["cost"].max()) if not df.empty else 0.0
        a_cost = t_cost / len(df) if len(df) > 0 else 0.0
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score, a_cost, t_cost, output_file

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ) -> None:
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        try:
            if log_file.exists():
                with log_file.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
            else:
                existing = []
        except Exception:
            existing = []
        existing.append(log_data)
        with log_file.open("w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    # ----------------------------
    # Abstract methods
    # ----------------------------
    @abstractmethod
    async def evaluate_problem(self, problem: dict, agent: Callable[..., Any]) -> Tuple[Any, ...]:
        """Return a row tuple matching self.get_result_columns()."""

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        """Return (score, extracted_output)."""

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        """Return table columns, must include at least 'score' and 'cost'."""

    # ----------------------------
    # Unified run_sample protocol
    # ----------------------------
    @abstractmethod
    async def single_attempt(self, problem: dict, agent: Callable[..., Any]) -> Dict[str, Any]:
        """Run one attempt for this problem with the provided agent.

        Returns a dict with keys:
          - ok: bool (success for this attempt under benchmark-specific judge)
          - final: Any (final output for this attempt)
          - cost: dict (optional cost fields: tokens/latency_s/tool_calls/usd)
          - meta: dict (optional auxiliary info, e.g., trace)
        """

    async def run_sample(self, problem: dict, agent: Callable[..., Any], k: int) -> Dict[str, Any]:
        """Run up to k attempts and return standardized tries list.

        Output: {"tries": [{ok, final, cost, meta}, ...], "context": {...}}
        """
        tries: List[Dict[str, Any]] = []
        for _ in range(max(1, int(k))):
            try:
                t = await self.single_attempt(problem, agent)
            except Exception as e:
                t = {"ok": False, "final": f"error: {e}", "cost": {}, "meta": {"error": str(e)}}
            # normalize fields
            t.setdefault("ok", False)
            t.setdefault("final", None)
            t.setdefault("cost", {})
            t.setdefault("meta", {})
            tries.append(t)
        return {"tries": tries, "context": {}}

    # ----------------------------
    # Orchestration helpers
    # ----------------------------
    async def _gather_with_limit(self, coros: Sequence, max_concurrent_tasks: int) -> List[Any]:
        sem = asyncio.Semaphore(max_concurrent_tasks)

        async def run(coro_fn):
            async with sem:
                return await coro_fn

        tasks = [run(c) for c in coros]
        return await asyncio.gather(*tasks)

    async def evaluate_all_problems(self, data: List[dict], agent: Callable[..., Any], max_concurrent_tasks: int = 50):
        coros = [self.evaluate_problem(problem, agent) for problem in data]
        return await self._gather_with_limit(coros, max_concurrent_tasks)

    async def run_evaluation(self, agent: Callable[..., Any], va_list: List[int], max_concurrent_tasks: int = 50, k: int = 1):
        self.pass_k = max(1, int(k))
        data = await self.load_data(va_list)
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost, out_file = self._save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        return average_score, average_cost, total_cost, out_file

    async def run_baseline(self, agent: Callable[..., Any], max_concurrent_tasks: int = 50, k: int = 1):
        self.pass_k = max(1, int(k))
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost, out_file = self._save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        return average_score, average_cost, total_cost, out_file

    # ----------------------------
    # Pass@k + metrics
    # ----------------------------
    @staticmethod
    def compute_pass_at_k(successes: Sequence[bool], k: int) -> float:
        """Simple pass@k estimator for a single item given multiple tries.

        For a single problem with multiple attempts, pass@k is 1 if any of the
        first k attempts succeed, else 0. For dataset-level, average across problems.
        """
        if k <= 0:
            return 0.0
        s = any(bool(x) for x in successes[:k])
        return 1.0 if s else 0.0

    @staticmethod
    def compute_unit_success_cost(tries: Sequence[Dict[str, Any]]) -> Any:
        """Accumulate USD cost to first success; return None if never succeeds.

        Expects each try has cost dict with optional 'usd' numeric field.
        """
        acc = 0.0
        for t in tries:
            c = t.get("cost") or {}
            try:
                acc += float(c.get("usd", 0.0))
            except Exception:
                acc += 0.0
            if bool(t.get("ok")):
                return acc
        return None

    @staticmethod
    def summarize_pass_at_k(per_problem_successes: Sequence[Sequence[bool]], k_list: Iterable[int] = (1, 5)) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        n = len(per_problem_successes)
        for k in k_list:
            total = 0.0
            for succ in per_problem_successes:
                total += BaseBenchmark.compute_pass_at_k(succ, k)
            metrics[f"pass@{k}"] = (total / n) if n > 0 else 0.0
        return metrics

    @staticmethod
    def summarize_metrics(rows: List[Tuple[Any, ...]], columns: List[str], k_list: Iterable[int] = (1,)) -> Dict[str, Any]:
        col_idx = {c: i for i, c in enumerate(columns)}
        out: Dict[str, Any] = {}
        if not rows:
            return {"avg_score": 0.0, "avg_cost": 0.0, **{f"pass@{k}": 0.0 for k in k_list}}
        # Aggregate basics
        scores = [float(r[col_idx.get("score", -1)]) for r in rows if col_idx.get("score", -1) >= 0]
        costs = [float(r[col_idx.get("cost", -1)]) for r in rows if col_idx.get("cost", -1) >= 0]
        out["avg_score"] = sum(scores) / len(scores) if scores else 0.0
        out["avg_cost"] = (sum(costs) / len(costs)) if costs else 0.0
        # pass@k expects per-problem multiple attempts; if the dataset logs per attempt with a boolean 'pass',
        # we group by problem id.
        if "id" in col_idx and "pass" in col_idx:
            from collections import defaultdict

            g: Dict[Any, List[bool]] = defaultdict(list)
            for r in rows:
                g[r[col_idx["id"]]].append(bool(r[col_idx["pass"]]))
            per_problem = list(g.values())
            out.update(BaseBenchmark.summarize_pass_at_k(per_problem, k_list))
        return out
