from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # optional; parsing guarded

from .bases import BaseAction


@dataclass
class ActionSpec:
    id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    inputs_schema: Dict[str, Any] = field(default_factory=dict)
    outputs_schema: Dict[str, Any] = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)
    environment_tags: List[str] = field(default_factory=list)
    security: Dict[str, Any] = field(default_factory=dict)  # minimal-privilege declaration
    effects_schema: Dict[str, Any] = field(default_factory=dict)  # structured side-effects
    provenance: Dict[str, Any] = field(default_factory=dict)  # origin/validation/source info
    validation: Dict[str, Any] = field(default_factory=dict)  # metrics: per_env_pass, last_verified_at


class ActionSpace:
    """Create/Register/Retrieve/Use actions across environments.

    This is a metadata registry; execution uses callables that conform
    to BaseAction (or Agent-as-Action) protocol.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, BaseAction] = {}
        self._specs: Dict[str, ActionSpec] = {}

    # ----------------------------
    # Register/Unregister
    # ----------------------------
    def register(self, action_id: str, action: BaseAction, spec: Optional[ActionSpec] = None) -> None:
        self._registry[action_id] = action
        if spec is None:
            spec = ActionSpec(id=action_id, name=action.name, description=action.description or "")
        self._specs[action_id] = spec

    def unregister(self, action_id: str) -> None:
        self._registry.pop(action_id, None)
        self._specs.pop(action_id, None)

    def get(self, action_id: str) -> Optional[BaseAction]:
        return self._registry.get(action_id)

    def spec(self, action_id: str) -> Optional[ActionSpec]:
        return self._specs.get(action_id)

    # ----------------------------
    # Retrieval
    # ----------------------------
    def list_actions(self) -> List[str]:
        return list(self._registry.keys())

    def search(self, query: str | None = None, tags: Sequence[str] | None = None) -> List[str]:
        query_l = (query or "").lower()
        tags = list(tags or [])
        out: List[str] = []
        for aid, spec in self._specs.items():
            if query and (query_l not in spec.name.lower() and query_l not in spec.description.lower()):
                continue
            if tags and not set(tags).issubset(set(spec.environment_tags)):
                continue
            out.append(aid)
        return out

    def search_with_scoring(
        self,
        query: str | None = None,
        tags: Sequence[str] | None = None,
        env: str | None = None,
        weights: Dict[str, float] | None = None,
        limit: int | None = None,
    ) -> List[str]:
        """Search and rank by a simplified scoring formula.

        score = a*semantic + b*per_env_pass + c*(1/avg_cost_norm)
        Where semantic is a trivial lexical match proxy here; per_env_pass and
        avg_cost_norm are taken from spec.validation if present.
        """
        weights = weights or {"semantic": 1.0, "per_env_pass": 1.0, "inv_cost": 0.0}
        candidates = self.search(query=query, tags=tags)
        q = (query or "").lower()
        scored: List[tuple[str, float]] = []
        for aid in candidates:
            spec = self._specs.get(aid)
            if not spec:
                continue
            # semantic proxy: 1 if any query token in name/desc, else 0
            sem = 0.0
            if q:
                tokens = [t for t in q.split() if t]
                text = f"{spec.name} {spec.description}".lower()
                sem = 1.0 if any(t in text for t in tokens) else 0.0
            val = spec.validation or {}
            pep = 0.0
            if env and isinstance(val.get("per_env_pass"), dict):
                try:
                    pep = float(val.get("per_env_pass", {}).get(env, 0.0) or 0.0)
                except Exception:
                    pep = 0.0
            avg_cost_norm = float(val.get("avg_cost_norm", 1.0) or 1.0)
            inv_cost = (1.0 / avg_cost_norm) if avg_cost_norm > 0 else 0.0
            score = (
                weights.get("semantic", 0.0) * sem
                + weights.get("per_env_pass", 0.0) * pep
                + weights.get("inv_cost", 0.0) * inv_cost
            )
            scored.append((aid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        ids = [aid for aid, _ in scored]
        return ids[:limit] if limit is not None else ids

    # ----------------------------
    # Creation from environment definitions
    # ----------------------------
    def import_from_action_space_txt(self, file_path: str, env_tag: str) -> List[ActionSpec]:
        """Parse simple action_space.txt (one action per line).

        This method creates metadata specs only; concrete implementations
        should be bound by the integrator as BaseAction subclasses.
        """
        specs: List[ActionSpec] = []
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            specs.append(
                ActionSpec(
                    id=f"{env_tag}:{name}",
                    name=name,
                    environment_tags=[env_tag],
                    description=f"Imported from {file_path}",
                )
            )
        return specs

    def import_from_env_yaml(self, yaml_path: str, env_tag: str) -> List[ActionSpec]:
        """Parse AutoEnv-like YAML with transition.actions structure.

        Requires PyYAML. If not available, returns empty list.
        """
        if yaml is None:
            return []
        data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
        actions = (((data or {}).get("transition") or {}).get("actions")) or []
        specs: List[ActionSpec] = []
        for a in actions:
            name = a.get("name")
            params = a.get("params", [])
            if not name:
                continue
            inputs_schema = {
                "type": "object",
                "properties": {p: {"type": "string"} for p in params},
                "required": list(params),
            }
            specs.append(
                ActionSpec(
                    id=f"{env_tag}:{name}",
                    name=name,
                    environment_tags=[env_tag],
                    description=f"Imported from {yaml_path}",
                    inputs_schema=inputs_schema,
                )
            )
        return specs

    # ----------------------------
    # Usage
    # ----------------------------
    async def use(self, action_id: str, params: Dict[str, Any]) -> Any:
        action = self.get(action_id)
        if action is None:
            raise KeyError(f"Action {action_id!r} not found")
        return await action(**params)

    # ----------------------------
    # Persistence (optional)
    # ----------------------------
    def dump_specs(self, out_path: str) -> None:
        rows = [spec.__dict__ for spec in self._specs.values()]
        Path(out_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
