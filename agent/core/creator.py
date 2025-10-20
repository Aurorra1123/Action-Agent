from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from engine.async_llm import AsyncLLM, create_llm_instance
from engine.formatter import BaseFormatter, JSONListOfActionSpecsFormatter
from .action_space import ActionSpace, ActionSpec
from .bases import BaseAction, BaseAgent


@dataclass
class CreatorConfig:
    llm_config: Any | None = None
    max_candidates: int = 3
    max_steps: int = 10
    # synthesis triggers
    synth_min_candidates: int = 1  # trigger if retrieved candidates < M
    synth_min_avg_pass: float = 0.0  # trigger if avg per_env_pass of Top-N < theta


class AgentCreator:
    """Main loop that retrieves or synthesizes actions/agents/workflows.

    This is a lightweight orchestrator: given a goal/context, it searches
    the ActionSpace for candidates; if insufficient, it uses LLM to propose
    new actions or agent workflows (skeleton), validates their format and
    registers the successful ones.
    """

    def __init__(self, action_space: ActionSpace, config: CreatorConfig):
        self.action_space = action_space
        self.config = config
        self.llm: Optional[AsyncLLM] = None
        if config.llm_config is not None:
            self.llm = create_llm_instance(config.llm_config)

    async def retrieve_candidates(self, query: str, tags: Sequence[str] | None = None, limit: int = 5) -> List[str]:
        # Basic retrieval; could be swapped to search_with_scoring for env-specific weighting
        ids = self.action_space.search(query=query, tags=tags)
        return ids[:limit]

    async def synthesize_action_specs(self, goal: str, formatter: Optional[BaseFormatter] = None, k: int = 1) -> List[ActionSpec]:
        if self.llm is None:
            return []
        prompt = (
            f"Create {k} actions for the goal: {goal}. "
            f"Each action must specify: name, description, parameters."
        )
        fmt = formatter or JSONListOfActionSpecsFormatter()
        try:
            data = await self.llm.call_with_format(prompt, fmt)  # returns list[dict]
        except Exception:
            return []
        out: List[ActionSpec] = []
        for i, spec in enumerate(data or []):
            try:
                name = str(spec.get("name"))
                desc = str(spec.get("description", ""))
                params = spec.get("parameters", {})
                aid = f"synth:{name}"
                out.append(
                    ActionSpec(
                        id=aid,
                        name=name,
                        description=desc,
                        inputs_schema=params if isinstance(params, dict) else {"type": "array"},
                        provenance={"from": "llm", "idx": i, "goal": goal},
                    )
                )
            except Exception:
                continue
        return out

    async def choose_and_run(self, action_ids: Sequence[str], params: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for aid in action_ids:
            try:
                out = await self.action_space.use(aid, params)
                results[aid] = {"ok": True, "output": out}
            except Exception as e:
                results[aid] = {"ok": False, "error": str(e)}
        return results

    async def main(self, goal: str, context: Optional[Dict[str, Any]] = None, query_tags: Sequence[str] | None = None) -> Dict[str, Any]:
        context = context or {}
        # 1) retrieve candidates
        candidates = await self.retrieve_candidates(goal, query_tags, limit=self.config.max_candidates)

        # 2) decide whether to trigger synthesis
        trigger = False
        if len(candidates) < self.config.synth_min_candidates:
            trigger = True
        else:
            # compute avg per_env_pass over current Top-N
            env = None
            if isinstance(query_tags, (list, tuple)) and query_tags:
                env = query_tags[0]
            vals: List[float] = []
            for aid in candidates:
                spec = self.action_space.spec(aid)
                if not spec:
                    continue
                pep = 0.0
                try:
                    pep = float((spec.validation or {}).get("per_env_pass", {}).get(env, 0.0)) if env else 0.0
                except Exception:
                    pep = 0.0
                vals.append(pep)
            avg_pass = (sum(vals) / len(vals)) if vals else 0.0
            if avg_pass < self.config.synth_min_avg_pass:
                trigger = True

        if trigger and self.llm is not None:
            # Synthesize new specs (registration of concrete actions is up to integrator)
            await self.synthesize_action_specs(goal, formatter=None, k=self.config.max_candidates)
            candidates = await self.retrieve_candidates(goal, query_tags, limit=self.config.max_candidates)
        # 3) run top candidates with given params from context
        params = context.get("params", {})
        results = await self.choose_and_run(candidates, params)
        return {"candidates": candidates, "results": results}
