"""Round loop for one swarm."""
from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from .cells import Cell, RunConfig
from .llm import ChatResult, parse_args
from .prompts import DEBRIEF, comms_prompt, system_prompt
from .scripted import honest_turn, reporter_turn, seed_turn
from .tools import TASK_TOOLS, tool_specs
from .world import World


class LiveAgent:
    def __init__(self, name: str, world: World):
        others = [a for a in world.names if a != name]
        self.name = name
        self.history: list[dict] = [{"role": "system",
                                     "content": system_prompt(name, others, world.cell, world.cfg)}]
        self.actions_left = 0
        self.done = False
        self.comms_only = False

    def start_round(self, header: str, actions: int) -> None:
        self.history.append({"role": "user", "content": header})
        self.actions_left = actions
        self.done = False
        self.comms_only = False

    def start_comms(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self.actions_left = 1
        self.done = False
        self.comms_only = True


async def _step(agent: LiveAgent, backend, tools: list[dict], cfg: RunConfig) -> ChatResult:
    return await backend.chat(agent.history, tools, agent.name, cfg.max_completion_tokens)


async def _apply(world: World, agent: LiveAgent, res: ChatResult) -> None:
    world.log("assistant", agent=agent.name, content=res.content, reasoning=res.reasoning,
              tool_calls=[{"name": c.name, "arguments": c.arguments} for c in res.tool_calls],
              prompt_tokens=res.prompt_tokens, cached_tokens=res.cached_tokens,
              completion_tokens=res.completion_tokens, finish=res.finish_reason, error=res.error)
    if res.error:
        agent.done = True
        return
    agent.history.append(res.history_message())
    if not res.tool_calls:
        agent.done = True
        return
    for call in res.tool_calls:
        if agent.actions_left <= 0:
            out = "Not run. You have no actions left this round."
        elif agent.comms_only and call.name in TASK_TOOLS:
            agent.actions_left -= 1
            out = "Not run. This action is for anything other than tasks."
        else:
            agent.actions_left -= 1
            args = parse_args(call.arguments)
            if args is None:
                out = "Could not parse the tool arguments as JSON."
            else:
                out = await asyncio.to_thread(world.dispatch, agent.name, call.name, args)
            if call.name == "end_turn":
                agent.done = True
        agent.history.append({"role": "tool", "tool_call_id": call.id, "content": out})
    if agent.actions_left <= 0:
        agent.done = True


async def run_swarm(cell: Cell, cfg: RunConfig, seed: int, run_dir: Path, backend, *, model: str,
                    provider: str, tag: str = "") -> dict:
    world = World(cell, cfg, seed, run_dir)
    agent_models = {}
    if hasattr(backend, "bind"):
        # A separate random stream, so the mix does not change the swarm's own draws.
        backend = backend.bind(world.live, random.Random(seed + 1))
        agent_models = {a: backend.model_of(a) for a in world.live}
    world.log("backend", model=model, provider=provider, tag=tag, agent_models=agent_models)
    agents = {name: LiveAgent(name, world) for name in world.live}
    tools = tool_specs(cell, cfg.peer_vote)

    for r in range(1, cfg.rounds + 1):
        world.begin_round(r)
        seed_turn(world)
        honest_turn(world)
        active = [a for n, a in agents.items() if n not in world.suspended]
        for agent in active:
            agent.start_round(world.header(agent.name), cfg.actions_per_round)
        for micro in range(1, cfg.actions_per_round + 1):
            world.step = micro
            ready = [a for a in active if not a.done and a.name not in world.suspended]
            if not ready:
                break
            results = await asyncio.gather(*(_step(a, backend, tools, cfg) for a in ready))
            pairs = list(zip(ready, results))
            world.rng.shuffle(pairs)  # seeded order decides ties for a locked task
            for agent, res in pairs:
                await _apply(world, agent, res)
        if cfg.comms_phase:
            world.step = cfg.actions_per_round + 1
            ready = [a for a in active if a.name not in world.suspended]
            for agent in ready:
                agent.start_comms(comms_prompt(r))
            results = await asyncio.gather(*(_step(a, backend, tools, cfg) for a in ready))
            pairs = list(zip(ready, results))
            world.rng.shuffle(pairs)
            for agent, res in pairs:
                await _apply(world, agent, res)
        world.step = 0
        reporter_turn(world)
        world.end_round()

    if cfg.debrief:
        world.round = cfg.rounds + 1

        async def ask(agent: LiveAgent):
            agent.history.append({"role": "user", "content": DEBRIEF})
            return agent, await backend.chat(agent.history, [], agent.name, cfg.max_completion_tokens)

        for agent, res in await asyncio.gather(*(ask(a) for a in agents.values())):
            world.log("debrief", agent=agent.name, content=res.content, reasoning=res.reasoning,
                      prompt_tokens=res.prompt_tokens, cached_tokens=res.cached_tokens,
                      completion_tokens=res.completion_tokens, error=res.error)
            agent.history.append(res.history_message())

    scores = await asyncio.to_thread(world.finalize)
    summary = {"cell": cell.id, "seed": seed, "model": model, "provider": provider, "tag": tag,
               "roles": world.role, "agent_models": agent_models, "scores": scores, "suspended": world.suspended,
               "config": cfg.as_dict()}
    world.log("run_end", scores=scores)
    world.close()
    (run_dir / "transcripts.json").write_text(json.dumps({a.name: a.history for a in agents.values()}))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
