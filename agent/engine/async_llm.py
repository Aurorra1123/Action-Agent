"""Adapter for Example_code/AutoEnv AsyncLLM.

Tries to import and re-export AsyncLLM-related classes from
Example_code/AutoEnv/autoenv/engine/async_llm.py. Falls back to
minimal stubs if not available, preserving the interface expected by
the rest of this repository.
"""
from __future__ import annotations

from typing import Any, Optional


def _try_import() -> tuple[Any, ...] | None:
    import sys
    import os

    # Try to add Example_code/AutoEnv to sys.path for import
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, "Example_code", "AutoEnv"),
        os.path.join(cwd, "Example_code"),
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)

    try:
        from autoenv.engine.async_llm import (  # type: ignore
            AsyncLLM as _AsyncLLM,
            LLMConfig as _LLMConfig,
            LLMsConfig as _LLMsConfig,
            create_llm_instance as _create_llm_instance,
        )
        return _AsyncLLM, _LLMConfig, _LLMsConfig, _create_llm_instance
    except Exception:
        return None


_imported = _try_import()


if _imported is not None:
    AsyncLLM, LLMConfig, LLMsConfig, create_llm_instance = _imported  # type: ignore
else:
    class LLMConfig:  # minimal stub
        def __init__(self, config: dict):
            self.model = config.get("model", "gpt-4o-mini")
            self.temperature = config.get("temperature", 1)
            self.key = config.get("key")
            self.base_url = config.get("base_url")
            self.top_p = config.get("top_p", 1)

    class LLMsConfig:
        @classmethod
        def default(cls):
            raise RuntimeError("LLMsConfig.default is unavailable in stub mode.")

    class AsyncLLM:
        def __init__(self, config: LLMConfig | dict | str, system_msg: Optional[str] = None, max_completion_tokens: Optional[int] = None):
            self.config = config
            self.system_msg = system_msg
            self.max_completion_tokens = max_completion_tokens

        async def __call__(self, prompt: str, max_tokens: Optional[int] = None) -> str:
            raise RuntimeError("AsyncLLM is running in stub mode (Example_code missing).")

        async def call_with_format(self, prompt: str, formatter: Any):
            raise RuntimeError("AsyncLLM.call_with_format is unavailable in stub mode.")

        def get_usage_summary(self) -> dict:
            return {}

    def create_llm_instance(llm_config) -> AsyncLLM:  # type: ignore
        return AsyncLLM(llm_config)

