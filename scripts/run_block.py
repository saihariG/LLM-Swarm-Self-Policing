#!/usr/bin/env python
"""Run a block of swarms from a fixed random schedule.

Examples
  python scripts/run_block.py --block mock_smoke --provider mock --cells hf dm --n 2
  python scripts/run_block.py --block pilot_gptoss --provider fireworks \
      --model accounts/fireworks/models/gpt-oss-120b --cells hf dm --n 2 --max-usd 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from swarmlab.cells import NAMED_CELLS, Cell, RunConfig, key_cells_plan, main_plan, make_schedule  # noqa: E402
from swarmlab.engine import run_swarm  # noqa: E402
from swarmlab.llm import Ledger, MixedBackend, OpenAIBackend  # noqa: E402
from swarmlab.mock import MockBackend  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--block", required=True)
    p.add_argument("--provider", required=True, choices=["fireworks", "meta", "modal", "mock"])
    p.add_argument("--model", default="mock")
    p.add_argument("--model2", help="second model for mixed swarms")
    p.add_argument("--provider2", help="provider of the second model, default same as --provider")
    p.add_argument("--n-model2", type=int, default=5, help="live agents per swarm on the second model")
    p.add_argument("--plan", choices=["main", "key"])
    p.add_argument("--cells", nargs="*", default=[])
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--master-seed", type=int, default=20261106)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--max-hours", type=float, default=None, help="start no new swarm after this long, to cap GPU spend")
    p.add_argument("--limit", type=int, default=None, help="run only the first N scheduled swarms")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--rounds", type=int)
    p.add_argument("--instruct-report", action="store_true")
    p.add_argument("--no-rule", action="store_true")
    p.add_argument("--no-debrief", action="store_true")
    p.add_argument("--no-seeds", action="store_true", help="control: scripted cheaters stay idle")
    p.add_argument("--peer-vote", action="store_true", help="exploratory arm: agents can flag and expel peers")
    p.add_argument("--random-audits", type=int, default=0,
                   help="tier 1 arm: organizers audit N random agents at the end of each round")
    p.add_argument("--scripted-reporter", action="store_true",
                   help="tier 1 arm: one scripted agent reports trick posts, taking one live slot")
    p.add_argument("--tag", default="")
    return p.parse_args()


def build_plan(a: argparse.Namespace) -> dict[str, int]:
    if a.plan == "main":
        return main_plan()
    if a.plan == "key":
        return key_cells_plan(a.n)
    if not a.cells:
        raise SystemExit("give --plan or --cells")
    cells = [NAMED_CELLS[c].id if c in NAMED_CELLS else Cell.parse(c).id for c in a.cells]
    return {c: a.n for c in cells}


async def main() -> None:
    a = parse()
    load_dotenv(ROOT / ".env")
    root = ROOT / ("runs_mock" if a.provider == "mock" else "runs") / a.block
    root.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(instruct_report=a.instruct_report, no_rule=a.no_rule, debrief=not a.no_debrief,
                    seeds_active=not a.no_seeds, peer_vote=a.peer_vote, random_audits=a.random_audits,
                    scripted_reporter=a.scripted_reporter)
    if a.peer_vote:
        cfg.prompt_version = "v1-vote"
    if a.random_audits:
        cfg.prompt_version = "v1-random-audit"
    if a.scripted_reporter:
        cfg.prompt_version = "v1-reporter"
        cfg.n_live -= 1  # keeps 13 agents
    if a.rounds:
        cfg.rounds = a.rounds
    schedule = make_schedule(build_plan(a), a.master_seed)
    if a.limit:
        schedule = schedule[: a.limit]
    sched_file = root / "schedule.json"
    if sched_file.exists() and json.loads(sched_file.read_text())["schedule"] != schedule:
        raise SystemExit(f"{sched_file} exists with a different schedule. Use a new block name.")
    sched_file.write_text(json.dumps({"args": vars(a), "config": cfg.as_dict(), "schedule": schedule}, indent=1))

    ledger = Ledger(a.max_usd)
    backend = None if a.provider == "mock" else OpenAIBackend(a.provider, a.model, ledger, temperature=a.temperature)
    if a.model2:
        provider2 = a.provider2 or a.provider
        second = OpenAIBackend(provider2, a.model2, ledger, temperature=a.temperature,
                               base_url=os.environ.get("VLLM_BASE_URL2") if provider2 == "modal" else None)
        backend = MixedBackend(backend, second, a.n_model2)
    gate = asyncio.Semaphore(a.concurrency)
    t0 = time.time()
    done = 0

    async def one(row: dict) -> None:
        nonlocal done
        run_dir = root / f"{row['order']:04d}_{row['cell']}_{row['replicate']:02d}"
        if (run_dir / "summary.json").exists():
            return
        async with gate:
            if ledger.over:
                print(f"budget reached (${ledger.usd:.2f}), skipping {run_dir.name}", flush=True)
                return
            if a.max_hours and time.time() - t0 > a.max_hours * 3600:
                print(f"time limit reached, skipping {run_dir.name}", flush=True)
                return
            if run_dir.exists():
                failed = root / "_failed"
                failed.mkdir(exist_ok=True)
                shutil.move(str(run_dir), str(failed / f"{run_dir.name}_{int(time.time())}"))
            cell = Cell.parse(row["cell"])
            use = MockBackend(cell, row["seed"]) if a.provider == "mock" else backend
            try:
                name = f"{a.model}+{a.model2}" if a.model2 else a.model
                await run_swarm(cell, cfg, row["seed"], run_dir, use, model=name, provider=a.provider, tag=a.tag)
            except Exception as exc:
                print(f"FAILED {run_dir.name}: {type(exc).__name__}: {exc}", flush=True)
                return
            done += 1
            print(f"[{time.time() - t0:7.0f}s] {done} done, {run_dir.name}, ${ledger.usd:.3f}, "
                  f"{ledger.calls} calls, {ledger.prompt:,} in ({ledger.cached:,} cached), "
                  f"{ledger.completion:,} out", flush=True)

    await asyncio.gather(*(one(r) for r in schedule))
    print(f"finished: {done} swarms, ${ledger.usd:.3f}, {ledger.calls} calls, "
          f"{ledger.prompt:,} in ({ledger.cached:,} cached), {ledger.completion:,} out, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
