from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")


@dataclass
class Metrics:
    counters: Dict[str, float] = field(default_factory=dict)

    def incr(self, key: str, val: float = 1.0) -> None:
        self.counters[key] = self.counters.get(key, 0.0) + val

