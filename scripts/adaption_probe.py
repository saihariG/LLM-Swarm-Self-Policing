#!/usr/bin/env python
"""Probe Adaption's hosted GLM 5.3 Flash with a real first call from block A. Never prints a key.

Reads ADAPTION_API_KEY from .env. Sends 1 models request and 2 responses requests.
The second repeats the first, to see whether Adaption caches the prompt.
Expected cost: well under 0.01 credits. Usage: .venv/bin/python scripts/adaption_probe.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from swarmlab.cells import Cell  # noqa: E402
from swarmlab.tools import tool_specs  # noqa: E402

BASE = "https://api.prod.adaptionlabs.ai/api/v1"  # from the Responses Create example in Adaption's docs
MODEL = "glm-53-flash"  # links to huggingface.co/zai-org/GLM-5.3-Flash on Adaption's models page
PRICE = (0.15, 0.03, 0.50)  # credits per 1M tokens: input, cached input, output (models page, 2026-11-08)
SWARM = ROOT / "runs/main_A/0000_R1P1C1H1E0_00"
AGENT = "agent_04"  # a GLM agent in that swarm


def responses_tools(chat_tools: list[dict]) -> list[dict]:
    """Our tools are in Chat Completions shape. The Responses API wants them flat."""
    return [{"type": "function", "name": t["function"]["name"], "description": t["function"]["description"],
             "parameters": t["function"]["parameters"]} for t in chat_tools]


async def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("ADAPTION_API_KEY"):
        sys.exit("ADAPTION_API_KEY is not in .env yet. Nothing was sent.")
    client = AsyncOpenAI(base_url=BASE, api_key=os.environ["ADAPTION_API_KEY"], timeout=300, max_retries=0)

    try:
        models = await client.models.list(extra_query={"type": "hosted"})
        print("hosted models for this key:", sorted(m.id for m in models.data))
    except Exception as exc:
        print("models list failed:", type(exc).__name__, str(exc)[:300])

    summary = json.loads((SWARM / "summary.json").read_text())
    msgs = json.loads((SWARM / "transcripts.json").read_text())[AGENT]
    system, user = msgs[0]["content"], msgs[1]["content"]
    tools = responses_tools(tool_specs(Cell.parse(summary["cell"])))
    total = 0.0
    for i in (1, 2):
        try:
            r = await client.responses.create(model=MODEL, instructions=system,
                                              input=[{"role": "user", "content": user}],
                                              tools=tools, max_output_tokens=8000)
        except Exception as exc:
            print(f"call {i} failed:", type(exc).__name__, str(exc)[:400])
            return
        print(f"call {i}: status={r.status} model={r.model} items={[o.type for o in r.output]}")
        print("  function calls:", [(o.name, o.arguments[:100]) for o in r.output if o.type == "function_call"])
        print("  text:", repr((r.output_text or "")[:200]))
        u = r.usage
        if u is None:
            print("  no usage returned")
            continue
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        reasoning = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", None)
        cost = (max(u.input_tokens - cached, 0) * PRICE[0] + cached * PRICE[1] + u.output_tokens * PRICE[2]) / 1e6
        total += cost
        print(f"  tokens: input {u.input_tokens}, cached {cached}, output {u.output_tokens}, reasoning {reasoning}")
        print(f"  credits at listed prices: {cost:.6f}  incomplete: {r.incomplete_details}  error: {r.error}")
    print(f"total credits at listed prices: {total:.6f}")


asyncio.run(main())
