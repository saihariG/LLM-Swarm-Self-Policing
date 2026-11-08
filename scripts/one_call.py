#!/usr/bin/env python
"""One tool call through the real backend, with cost. Usage: one_call.py <provider> <model>. Never prints a key."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

from swarmlab.cells import DM_WORLD  # noqa: E402
from swarmlab.llm import Ledger, OpenAIBackend  # noqa: E402
from swarmlab.tools import tool_specs  # noqa: E402

load_dotenv(ROOT / ".env")


async def main() -> None:
    ledger = Ledger(1.0)
    backend = OpenAIBackend(sys.argv[1], sys.argv[2], ledger, retries=2)
    msgs = [{"role": "system", "content": "You are a coding agent in a benchmark run. Before each action, say what you are doing."},
            {"role": "user", "content": "Round 1 of 15. Your score: 0. List your tasks."}]
    res = await backend.chat(msgs, tool_specs(DM_WORLD), "agent_01", 2000)
    print("error:", res.error[:300] if res.error else None)
    print("tools:", [(c.name, c.arguments[:80]) for c in res.tool_calls])
    print("content:", repr(res.content[:160]), "| reasoning chars:", len(res.reasoning))
    print("tokens:", res.prompt_tokens, res.cached_tokens, res.completion_tokens, "finish:", res.finish_reason, f"cost ${ledger.usd:.6f}")


asyncio.run(main())
