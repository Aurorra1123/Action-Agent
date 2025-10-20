from __future__ import annotations

import asyncio
from typing import Any

from benchmarks.gaia import GAIABenchmark
from core.action_space import ActionSpace, ActionSpec
from core.examples.actions import make_echo_answer_action
from core.examples.agents import GAIAEchoAgent


async def main():
    # Prepare action space and register a demo action
    asp = ActionSpace()
    echo = make_echo_answer_action()
    asp.register(
        action_id="demo:echo_answer",
        action=echo,
        spec=ActionSpec(id="demo:echo_answer", name="echo_answer", description=echo.description, inputs_schema=echo.parameters, environment_tags=["gaia"]),
    )

    # Create a simple agent that uses the demo action
    agent = GAIAEchoAgent(
        name="gaia_echo_agent",
        description="GAIA demo agent using echo action",
        parameters={},
        action_space=asp,
        action_id="demo:echo_answer",
    )

    async def agent_wrapper(problem: dict) -> Any:
        return await agent.run(problem)

    bench = GAIABenchmark(name="GAIA", file_path="data/gaia.jsonl", log_path="logs/gaia")
    await bench.run_baseline(agent=agent_wrapper)


if __name__ == "__main__":
    asyncio.run(main())
