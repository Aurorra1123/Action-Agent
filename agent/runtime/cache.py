from __future__ import annotations

from typing import Any, Dict, Tuple


class SimpleCache:
    def __init__(self) -> None:
        self._m: Dict[Tuple[str, str], Any] = {}

    def get(self, ns: str, key: str) -> Any:
        return self._m.get((ns, key))

    def put(self, ns: str, key: str, val: Any) -> None:
        self._m[(ns, key)] = val

