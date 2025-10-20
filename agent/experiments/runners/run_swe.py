from __future__ import annotations

import asyncio
from typing import Any

from benchmarks.swe_bench import SWEBenchBenchmark


async def dummy_agent(problem: dict) -> Any:
    # Placeholder: always fail
    return {"pass": False}


async def main():
    bench = SWEBenchBenchmark(name="SWE-bench", file_path="data/swe.jsonl", log_path="logs/swe")
    await bench.run_baseline(agent=dummy_agent)


if __name__ == "__main__":
    asyncio.run(main())

