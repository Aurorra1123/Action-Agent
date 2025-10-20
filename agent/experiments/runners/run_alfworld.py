from __future__ import annotations

import asyncio
from typing import Any

from benchmarks.alfworld import ALFWorldBenchmark


async def dummy_agent(problem: dict) -> Any:
    # Placeholder: random result
    return {"success": False}


async def main():
    bench = ALFWorldBenchmark(name="ALFWorld", file_path="data/alfworld.jsonl", log_path="logs/alfworld")
    await bench.run_baseline(agent=dummy_agent)


if __name__ == "__main__":
    asyncio.run(main())

