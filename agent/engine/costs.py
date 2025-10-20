from __future__ import annotations

from typing import Any, Dict, Optional


def merge_usage_into_cost(
    usage: Dict[str, Any] | None,
    *,
    latency_s: float = 0.0,
    tool_calls: int | Dict[str, int] = 0,
    usd_rate: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Merge token usage + runtime signals into a unified cost structure.

    Returns:
      {
        "tokens": {"prompt": int, "completion": int, "total": int},
        "latency_s": float,
        "tool_calls": int,
        "usd": float
      }
    """
    usage = usage or {}
    toks = usage.get("tokens") or usage
    prompt = int(toks.get("prompt", 0) or 0)
    completion = int(toks.get("completion", 0) or 0)
    total = int(toks.get("total", prompt + completion))

    # Flatten tool_calls if dict
    if isinstance(tool_calls, dict):
        tc = int(sum(int(v or 0) for v in tool_calls.values()))
    else:
        tc = int(tool_calls or 0)

    usd = 0.0
    # Optional heuristic conversion
    if usd_rate:
        # supports model-dependent rates if usage contains model key
        # Example: usd_rate = {"prompt_per_1k": 0.5, "completion_per_1k": 1.5}
        p_rate = float(usd_rate.get("prompt_per_1k", 0.0))
        c_rate = float(usd_rate.get("completion_per_1k", 0.0))
        usd = (prompt / 1000.0) * p_rate + (completion / 1000.0) * c_rate

    return {
        "tokens": {"prompt": prompt, "completion": completion, "total": total},
        "latency_s": float(latency_s or 0.0),
        "tool_calls": tc,
        "usd": float(usd),
    }

