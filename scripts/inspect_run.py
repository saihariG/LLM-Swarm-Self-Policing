#!/usr/bin/env python
"""Readable timeline of one swarm. Usage: python scripts/inspect_run.py runs/<block>/<run> [--agent agent_03] [--full]"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--agent")
    p.add_argument("--full", action="store_true", help="also show every assistant message")
    p.add_argument("--chars", type=int, default=260)
    a = p.parse_args()
    events = [json.loads(x) for x in (a.run / "events.jsonl").read_text().splitlines()]
    start = events[0]
    roles = start["roles"]
    verdicts = {e["id"]: e for e in events if e["type"] == "verdict"}
    print(start["cell"], "| seeds", [k for k, v in roles.items() if v.startswith("seed")],
          "| honest", [k for k, v in roles.items() if v == "honest"])
    tools = Counter()
    finish = Counter()
    for e in events:
        if a.agent and e.get("agent") not in (a.agent, None):
            continue
        kind = e["type"]
        who = f"{e.get('agent', '')}({roles.get(e.get('agent'), '?')[:6]})"
        if kind == "round_start":
            print(f"--- round {e['round']}")
        elif kind == "submission":
            verdict = verdicts.get(e["id"], {}).get("label", "") if e["accepted"] else ""
            sig = "EQ" if "__eq__" in e["code"] else ""
            print(f"  SUBMIT {who} {e['task']} {'imp' if e['impossible'] else 'ok '} "
                  f"{'ACCEPT' if e['accepted'] else 'reject'} {e['passed']}/{e['total']} {verdict} {sig}")
        elif kind in ("post", "dm", "feedback"):
            to = f" to {e['to']}" if kind == "dm" else ""
            print(f"  {kind.upper()} {who}{to}: {e['text'][:a.chars]!r}")
        elif kind in ("audit", "removal"):
            print(f"  {kind.upper()} {e}")
        elif kind == "debrief":
            print(f"  DEBRIEF {who}: {(e.get('content') or '')[:a.chars * 2]!r}")
        elif kind == "assistant":
            finish[e.get("finish")] += 1
            for call in e.get("tool_calls", []):
                tools[call["name"]] += 1
            if e.get("error"):
                print(f"  ERROR {who}: {e['error'][:200]}")
            if a.full:
                names = [c["name"] for c in e.get("tool_calls", [])]
                think = (e.get("reasoning") or "")[:a.chars]
                print(f"  SAY {who} {names}: {(e.get('content') or '')[:a.chars]!r} | think: {think!r}")
    print("tools", dict(tools))
    print("finish", dict(finish))
    last = next((e for e in reversed(events) if e["type"] == "run_end"), None)
    if last:
        print("scores", {f"{k}({roles[k][:6]})": v for k, v in sorted(last["scores"].items(), key=lambda x: -x[1])})


if __name__ == "__main__":
    main()
